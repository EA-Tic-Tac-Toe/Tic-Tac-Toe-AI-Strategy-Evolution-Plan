"""Genetic algorithm based agent using DEAP."""

from __future__ import annotations

import logging
import pickle
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

from deap import algorithms, base, creator, tools  # type: ignore[import-untyped]

from tictactoe.agents.base import Agent
from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.board import Board

logger = logging.getLogger(__name__)

BoardFactory = Callable[[], Board]

_RANDOM_MAX = 1_000_000_000


def _instantiate_agent(
    agent_cls: type[Agent],
    player: int,
    seed_rng: random.Random,
) -> Agent:
    """Create opponent instance with deterministic seeding when available."""
    if issubclass(agent_cls, RandomAgent):
        seed = seed_rng.randrange(_RANDOM_MAX)
        return agent_cls(player, seed=seed)
    return agent_cls(player)


@dataclass(frozen=True)
class EvolutionHistoryEntry:
    """Fitness statistics collected for a single generation."""

    generation: int
    min_fitness: float
    avg_fitness: float
    max_fitness: float


@dataclass(frozen=True)
class EvolutionResult:
    """Full record for a GA run."""

    best_weights: tuple[float, ...]
    history: tuple[EvolutionHistoryEntry, ...]
    best_fitness: float
    selection: str
    seed: int | None


class GeneticAgent(Agent):
    """
    Agent whose policy is a 9-dimensional weight vector; chooses the legal move whose
    weight is maximal. The weights are evolved with DEAP.
    """

    def __init__(
        self,
        player: int,
        weights: Sequence[float],
        name: str = "Genetic",
    ) -> None:
        """
        Args:
            player: 1 (X) or -1 (O)
            weights: list of 9 floats, one weight per board cell (0..8)
        """
        super().__init__(player, name)
        if len(weights) != 9:
            raise ValueError("weights must be length 9")
        self.weights = list(weights)
        self.opponent = -player

    @override
    def select_move(self, board: Board) -> int:
        legal = board.get_legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        # Choose legal move with max weight; break ties randomly
        best_val = None
        best_moves = []
        for m in legal:
            val = self.weights[m]
            if best_val is None or val > best_val:
                best_val = val
                best_moves = [m]
            elif val == best_val:
                best_moves.append(m)

        choice = random.choice(best_moves)
        logger.debug("%s selects %d (weight=%.3f)", self.name, choice, best_val)
        return choice


# --------------------------
# Evolution / Trainer
# --------------------------

def _evaluate_individual(  # pragma: no cover - exercised indirectly
    individual: list[float],
    player: int,
    n_games: int,
    opponents: Sequence[tuple[type[Agent], int]],
    board_factory: BoardFactory,
    seed_base: int | None,
) -> tuple[float, ...]:
    """
    Evaluate an individual by playing matches.

    Args:
        individual: genome (list of 9 floats)
        player: which side the GeneticAgent plays in evaluation (1 or -1)
        n_games: number of games per opponent
        opponents: list of tuples (AgentClass, play_as) describing
            which side the opponent plays.
        board_factory: callable that returns a fresh Board instance
    Returns:
        average score across games (higher is better)
    """
    ga = GeneticAgent(player, individual, name="EvoCandidate")
    total_score = 0.0
    total_games = 0
    seed_rng = random.Random(seed_base)

    for opp_class, opp_player in opponents:
        for _ in range(n_games):
            board = board_factory()
            # decide who moves first depending on player values
            # We'll assign agents according to their player field
            # GA plays as `player`, Opponent plays as `opp_player`
            # We assume Board.make_move(move, player) and check_win(player) exist
            opponent_agent = _instantiate_agent(opp_class, opp_player, seed_rng)
            agents = {player: ga, opp_player: opponent_agent}

            # play until terminal
            current_player = 1  # X starts (1)
            while True:
                current_agent = agents.get(current_player)
                if current_agent is None:
                    # if no agent assigned to current_player, default to random
                    current_agent = RandomAgent(current_player)

                move = current_agent.select_move(board)
                board.make_move(move, current_player)

                if board.check_win(current_player):
                    if current_player == player:
                        total_score += 1.0  # win
                    else:
                        total_score += 0.0  # loss
                    break
                if not board.get_legal_moves():
                    total_score += 0.5  # draw
                    break

                current_player = -current_player

            total_games += 1

    return (total_score / total_games,)


def _ensure_creator_types() -> None:
    """Create DEAP types once to allow repeated experiments."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _register_selection(toolbox: base.Toolbox, selection: str) -> str:
    """Register selection operator on toolbox and return normalized name."""
    selection_name = selection.lower()
    match selection_name:
        case "tournament":
            toolbox.register("select", tools.selTournament, tournsize=3)
        case "roulette":
            toolbox.register("select", tools.selRoulette)
        case "best":
            toolbox.register("select", tools.selBest)
        case _:
            msg = f"Unsupported selection strategy: {selection}"
            raise ValueError(msg)
    return selection_name


def run_evolution(
    board_factory: BoardFactory,
    *,
    pop_size: int = 100,
    generations: int = 40,
    cx_pb: float = 0.5,
    mut_pb: float = 0.2,
    n_games: int = 4,
    seed: int | None = None,
    selection: str = "tournament",
) -> EvolutionResult:
    """Execute the GA and return history + best weights."""
    if seed is not None:
        random.seed(seed)

    _ensure_creator_types()
    toolbox = base.Toolbox()

    # Attribute generator: random float in [-1, 1]
    toolbox.register(
        "attr_float",
        random.uniform,
        -1.0,
        1.0,
    )
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=9,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    opponents: list[tuple[type[Agent], int]] = [
        (HeuristicAgent, -1),
        (RandomAgent, -1),
    ]

    # Register evaluation function (wrap to pass board_factory, opponents etc.)
    def eval_wrapper(individual: list[float]) -> tuple[float, ...]:
        return _evaluate_individual(
            individual,
            player=1,
            n_games=n_games,
            opponents=opponents,
            board_factory=board_factory,
            seed_base=seed,
        )

    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
    selection_used = _register_selection(toolbox, selection)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(sum(f[0] for f in fits)) / len(fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))
    stats.register("min", lambda fits: min(f[0] for f in fits))

    _, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cx_pb,
        mutpb=mut_pb,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    best = tuple(hof[0])
    history = tuple(
        EvolutionHistoryEntry(
            generation=int(entry["gen"]),
            min_fitness=float(entry["min"]),
            avg_fitness=float(entry["avg"]),
            max_fitness=float(entry["max"]),
        )
        for entry in log
    )

    logger.info(
        "Evolution finished. Best fitness: %.4f (selection=%s)",
        hof[0].fitness.values[0],
        selection_used,
    )
    return EvolutionResult(
        best_weights=best,
        history=history,
        best_fitness=float(hof[0].fitness.values[0]),
        selection=selection_used,
        seed=seed,
    )


def evolve_population(
    board_factory: BoardFactory,
    pop_size: int = 100,
    generations: int = 40,
    cx_pb: float = 0.5,
    mut_pb: float = 0.2,
    n_games: int = 4,
    seed: int | None = None,
    selection: str = "tournament",
) -> list[float]:
    """
    Run DEAP GA to evolve a 9-float weight vector.

    Args:
        board_factory: callable that returns a fresh Board instance (for playing games).
        pop_size: population size
        generations: number of generations
        cx_pb: crossover probability
        mut_pb: mutation probability (per individual)
        n_games: number of games per opponent per individual evaluation
        seed: optional RNG seed

    Returns:
        best individual's weight list (length 9)
    """
    result = run_evolution(
        board_factory=board_factory,
        pop_size=pop_size,
        generations=generations,
        cx_pb=cx_pb,
        mut_pb=mut_pb,
        n_games=n_games,
        seed=seed,
        selection=selection,
    )
    return list(result.best_weights)


def save_weights(weights: Sequence[float], path: Path | None = None) -> None:
    """Persist genome weights to disk."""
    target = path or Path("src") / "tictactoe" / "weights" / "best"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fp:
        pickle.dump(list(weights), fp)


def load_weights(path: Path | None = None) -> list[float]:
    """Load previously saved genome weights."""
    target = path or Path("src") / "tictactoe" / "weights" / "best"
    if not target.exists():
        msg = "Weights file not found. \nPlease run: 'uv run tictactoe evolve' "
        raise FileNotFoundError(msg)
    with target.open("rb") as fp:
        weights: list[float] = pickle.load(fp)
    return weights
