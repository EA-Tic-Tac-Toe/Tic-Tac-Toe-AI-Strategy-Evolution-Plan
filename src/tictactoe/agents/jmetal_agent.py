"""Multi-objective genetic algorithm based agent using jMetalPy."""

from __future__ import annotations

import logging
import pickle
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

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
class MultiObjectiveHistoryEntry:
    """Statistics collected for a single generation in multi-objective evolution."""

    generation: int
    hypervolume: float
    n_solutions: int
    avg_fitness: float
    avg_complexity: float


@dataclass(frozen=True)
class MultiObjectiveResult:
    """Full record for a multi-objective GA run."""

    pareto_front: tuple[tuple[float, ...], ...]  # List of weight vectors
    pareto_objectives: tuple[tuple[float, ...], ...]  # List of (fitness, complexity)
    history: tuple[MultiObjectiveHistoryEntry, ...]
    algorithm: str
    seed: int | None


class TicTacToeProblem(FloatProblem):
    """
    Multi-objective optimization problem for Tic-Tac-Toe.

    Objectives:
    1. Maximize fitness (win rate against opponents)
    2. Minimize complexity (L1-norm of weights)
    """

    def __init__(
        self,
        player: int,
        n_games: int,
        opponents: Sequence[tuple[type[Agent], int]],
        board_factory: BoardFactory,
        seed_base: int | None = None,
    ):
        """
        Args:
            player: which side the agent plays (1 or -1)
            n_games: number of games per opponent for evaluation
            opponents: list of (AgentClass, player_value) tuples
            board_factory: callable that returns a fresh Board instance
            seed_base: seed for reproducible opponent creation
        """
        super().__init__()
        super().__init__()
        self.player = player
        self.n_games = n_games
        self.opponents = opponents
        self.board_factory = board_factory
        self.seed_base = seed_base

        # 9 weights, one per board cell
        self.number_of_variables = 9
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Negative Fitness", "Complexity"]

        # Weight bounds
        self.lower_bound = [-10.0] * self.number_of_variables
        self.upper_bound = [10.0] * self.number_of_variables

    def number_of_objectives(self) -> int:
        """Return number of objectives."""
        return 2

    def number_of_constraints(self) -> int:
        """Return number of constraints."""
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        """
        Evaluate solution on two objectives:
        1. Fitness (to maximize, so we negate it)
        2. Complexity (to minimize)
        """
        weights = solution.variables

        # Objective 1: Fitness (game performance)
        fitness = self._evaluate_fitness(weights)

        # Objective 2: Complexity (L1-norm of weights)
        complexity = sum(abs(w) for w in weights)

        # jMetalPy minimizes by default, so negate fitness to maximize it
        solution.objectives[
            0
        ] = -fitness  # Maximize fitness -> minimize negative fitness
        solution.objectives[1] = complexity  # Minimize complexity

        return solution

    def _evaluate_fitness(self, weights: list[float]) -> float:
        """Evaluate fitness by playing games against opponents."""
        from tictactoe.agents.jmetal_agent import JMetalAgent

        agent = JMetalAgent(self.player, weights, name="JMetalCandidate")
        total_score = 0.0
        total_games = 0
        seed_rng = random.Random(self.seed_base)

        for opp_class, opp_player in self.opponents:
            for _ in range(self.n_games):
                board = self.board_factory()
                opponent_agent = _instantiate_agent(opp_class, opp_player, seed_rng)
                agents = {self.player: agent, opp_player: opponent_agent}

                current_player = 1  # X starts
                while True:
                    current_agent = agents.get(current_player)
                    if current_agent is None:
                        move = random.choice(board.get_legal_moves())
                    else:
                        move = current_agent.select_move(board)

                    board.make_move(move, current_player)

                    if board.check_win(current_player):
                        if current_player == self.player:
                            total_score += 1.0
                        else:
                            total_score -= 1.0
                        break

                    if board.is_terminal():
                        break

                    current_player = -current_player

                total_games += 1

        return total_score / total_games if total_games > 0 else 0.0

    def create_solution(self) -> FloatSolution:
        """Create a random solution."""
        solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints(),
        )
        solution.variables = [
            random.uniform(self.lower_bound[i], self.upper_bound[i])
            for i in range(self.number_of_variables)
        ]
        return solution

    def name(self) -> str:
        return "TicTacToeProblem"


class JMetalAgent(Agent):
    """
    Agent whose policy is a 9-dimensional weight vector; chooses the legal move whose
    weight is maximal. The weights are evolved with jMetalPy using multi-objective
    optimization.
    """

    def __init__(
        self,
        player: int,
        weights: Sequence[float],
        name: str = "JMetal",
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


def run_multiobjective_evolution(
    pop_size: int = 100,
    max_evaluations: int = 25000,
    algorithm: str = "NSGA-II",
    player: int = 1,
    opponents: Sequence[tuple[type[Agent], int]] | None = None,
    board_factory: BoardFactory = Board,
    n_games: int = 10,
    seed: int | None = None,
) -> MultiObjectiveResult:
    """
    Run multi-objective evolution using jMetalPy.

    Args:
        pop_size: population size
        max_evaluations: maximum number of fitness evaluations
        algorithm: "NSGA-II" or "NSGA-III"
        player: which side to train (1=X, -1=O)
        opponents: list of (AgentClass, player_value) for evaluation
        board_factory: callable that returns Board instances
        n_games: games per opponent for fitness evaluation
        seed: random seed for reproducibility

    Returns:
        MultiObjectiveResult containing Pareto front and history
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if opponents is None:
        opponent_player = -player
        opponents = [
            (HeuristicAgent, opponent_player),
            (RandomAgent, opponent_player),
        ]

    problem = TicTacToeProblem(
        player=player,
        n_games=n_games,
        opponents=opponents,
        board_factory=board_factory,
        seed_base=seed,
    )

    # Configure operators
    crossover = SBXCrossover(probability=0.9, distribution_index=20)
    mutation = PolynomialMutation(
        probability=1.0 / problem.number_of_variables,
        distribution_index=20,
    )

    # Select algorithm (using NSGA-II)
    if algorithm.upper() not in ["NSGA-II", "NSGA-III"]:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if algorithm.upper() == "NSGA-III":
        # NSGA-III requires reference directions, use NSGA-II instead
        logger.warning("NSGA-III requires reference directions, using NSGA-II instead")

    algo = NSGAII(
        problem=problem,
        population_size=pop_size,
        offspring_population_size=pop_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    logger.info(
        "Starting %s with pop=%d, max_eval=%d, n_games=%d",
        algorithm,
        pop_size,
        max_evaluations,
        n_games,
    )

    # Run evolution
    algo.run()
    solutions = algo.result()

    logger.info("Evolution complete. Pareto front size: %d", len(solutions))

    # Extract Pareto front
    pareto_front = tuple(tuple(sol.variables) for sol in solutions)
    pareto_objectives = tuple(
        (-sol.objectives[0], sol.objectives[1]) for sol in solutions
    )

    # Create history (simplified for now -
    # jMetalPy doesn't have built-in history tracking)
    # We'll compute statistics from the final Pareto front
    fitness_values = [obj[0] for obj in pareto_objectives]
    complexity_values = [obj[1] for obj in pareto_objectives]

    history_entry = MultiObjectiveHistoryEntry(
        generation=max_evaluations // pop_size,
        hypervolume=0.0,  # TODO: compute actual hypervolume
        n_solutions=len(solutions),
        avg_fitness=sum(fitness_values) / len(fitness_values)
        if fitness_values
        else 0.0,
        avg_complexity=sum(complexity_values) / len(complexity_values)
        if complexity_values
        else 0.0,
    )

    return MultiObjectiveResult(
        pareto_front=pareto_front,
        pareto_objectives=pareto_objectives,
        history=(history_entry,),
        algorithm=algorithm,
        seed=seed,
    )


def save_pareto_front(result: MultiObjectiveResult, path: Path) -> None:
    """Save Pareto front to a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(result, f)
    logger.info("Saved Pareto front to %s", path)


def load_pareto_front(path: Path) -> MultiObjectiveResult:
    """Load Pareto front from a pickle file."""
    with path.open("rb") as f:
        result = pickle.load(f)
    logger.info("Loaded Pareto front from %s", path)
    return result


def select_solution_from_pareto(
    result: MultiObjectiveResult,
    strategy: str = "balanced",
) -> tuple[float, ...]:
    """
    Select a single solution from the Pareto front.

    Args:
        result: MultiObjectiveResult containing Pareto front
        strategy: selection strategy
            - "balanced": minimize distance to ideal point
            - "fitness": maximize fitness (ignore complexity)
            - "simple": minimize complexity (prefer simple solutions)

    Returns:
        Selected weight vector
    """
    if not result.pareto_front:
        raise ValueError("Empty Pareto front")

    if strategy == "fitness":
        # Choose solution with best fitness
        best_idx = max(
            range(len(result.pareto_objectives)),
            key=lambda i: result.pareto_objectives[i][0],
        )
        return result.pareto_front[best_idx]

    elif strategy == "simple":
        # Choose solution with lowest complexity
        best_idx = min(
            range(len(result.pareto_objectives)),
            key=lambda i: result.pareto_objectives[i][1],
        )
        return result.pareto_front[best_idx]

    else:  # balanced
        # Normalize objectives and find solution closest to ideal point (1, 0)
        fitness_values = [obj[0] for obj in result.pareto_objectives]
        complexity_values = [obj[1] for obj in result.pareto_objectives]

        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        max_complexity = max(complexity_values)
        min_complexity = min(complexity_values)

        fitness_range = max_fitness - min_fitness if max_fitness != min_fitness else 1.0
        complexity_range = (
            max_complexity - min_complexity if max_complexity != min_complexity else 1.0
        )

        best_idx = None
        best_distance = float("inf")

        for i, (fitness, complexity) in enumerate(result.pareto_objectives):
            # Normalize to [0, 1]
            norm_fitness = (fitness - min_fitness) / fitness_range
            norm_complexity = (complexity - min_complexity) / complexity_range

            # Distance to ideal point (1, 0) - high fitness, low complexity
            distance = ((1.0 - norm_fitness) ** 2 + (norm_complexity) ** 2) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_idx = i

        return result.pareto_front[best_idx]
