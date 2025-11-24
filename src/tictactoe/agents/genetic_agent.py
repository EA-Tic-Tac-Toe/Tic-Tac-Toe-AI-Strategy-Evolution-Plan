"""Genetic algorithm based agent using DEAP (Distributed Evolutionary Algorithms in Python)."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, List, Optional, Tuple, override
import os
import pickle

from tictactoe.agents.base import Agent

# Import opponents used during evolution
from tictactoe.agents.heuristic_agent import HeuristicAgent  # your provided template
from tictactoe.agents.random_agent import RandomAgent  # assume you have a simple random agent

# External dependency: DEAP
from deap import base, creator, tools, algorithms

if TYPE_CHECKING:
    from tictactoe.board import Board

logger = logging.getLogger(__name__)


class GeneticAgent(Agent):
    """
    Agent whose policy is a 9-dimensional weight vector; chooses the legal move whose
    weight is maximal. The weights are evolved with DEAP.
    """

    def __init__(self, player: int, weights: List[float], name: str = "Genetic") -> None:
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
    def select_move(self, board: "Board") -> int:
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

def _evaluate_individual(individual: List[float],
                         player: int,
                         n_games: int,
                         opponents: List[Tuple[type, int]],
                         board_factory) -> float:
    """
    Evaluate an individual by playing matches.

    Args:
        individual: genome (list of 9 floats)
        player: which side the GeneticAgent plays in evaluation (1 or -1)
        n_games: number of games per opponent
        opponents: list of tuples (AgentClass, play_as) where play_as is which side the opponent plays.
        board_factory: callable that returns a fresh Board instance
    Returns:
        average score across games (higher is better)
    """
    ga = GeneticAgent(player, individual, name="EvoCandidate")
    total_score = 0.0
    total_games = 0

    for OppClass, opp_player in opponents:
        for g in range(n_games):
            board = board_factory()
            # decide who moves first depending on player values
            # We'll assign agents according to their player field
            # GA plays as `player`, Opponent plays as `opp_player`
            # We assume Board.make_move(move, player) and check_win(player) exist
            agents = {player: ga, opp_player: OppClass(opp_player)}

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


def evolve_population(board_factory,
                      pop_size: int = 100,
                      generations: int = 40,
                      cx_pb: float = 0.5,
                      mut_pb: float = 0.2,
                      n_games: int = 4,
                      seed: Optional[int] = None) -> List[float]:
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
    if seed is not None:
        random.seed(seed)

    # Setup DEAP types (single objective maximization)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator: random float in [-1, 1]
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=9)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Opponents used during evolution: heuristic (your HeuristicAgent) and random agent
    opponents = [(HeuristicAgent, -1), (RandomAgent, -1)]  # opponents play as -1 by default

    # We evaluate the individual for the side '1' (X); you may alternate in a more robust scheme.

    # Register evaluation function (wrap to pass board_factory, opponents etc.)
    def eval_wrapper(individual):
        return _evaluate_individual(individual, player=1, n_games=n_games, opponents=opponents,
                                    board_factory=board_factory)

    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)

    # Hall of Fame and stats
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: float(sum(f[0] for f in fits)) / len(fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))

    # Run evolution,
    # gen: number of generation
    # nevals: the number of evaluations in the last generation.
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cx_pb, mutpb=mut_pb, ngen=generations,
                                   stats=stats, halloffame=hof, verbose=True)

    best = list(hof[0])
    logger.info("Evolution finished. Best fitness: %.4f", hof[0].fitness.values[0])
    # Clean up creator types to allow repeated runs in same interpreter
    try:
        del creator.FitnessMax
        del creator.Individual
    except Exception:
        pass

    return best


def save_weights(weights: List[float]) -> None:
    os.makedirs("./src/tictactoe/weights", exist_ok=True)
    with open("./src/tictactoe/weights/best", "wb") as fp:  # Pickling
        pickle.dump(weights, fp)


def load_weights() -> List[float]:
    try:
        with open("./src/tictactoe/weights/best", "rb") as fp:
            weights = pickle.load(fp)
    except FileNotFoundError:
        raise FileNotFoundError("Weights file not found. \nPlease run: 'uv run tictactoe evolve' ")

    return weights
