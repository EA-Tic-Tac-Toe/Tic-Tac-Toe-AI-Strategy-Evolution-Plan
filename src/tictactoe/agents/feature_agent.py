"""Feature-based genetic agent with strategic features and self-play evolution."""

from __future__ import annotations

import pickle
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from deap import base, creator, tools

from tictactoe.agents.base import Agent
from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.board import Board

# Feature indices
FEATURE_CENTER = 0
FEATURE_CORNER = 1
FEATURE_EDGE = 2
FEATURE_WIN_THREAT = 3
FEATURE_BLOCK_THREAT = 4
NUM_FEATURES = 5


def extract_features(board: Board, move: int, player: int) -> list[float]:
    """
    Extract strategic features for a given move.

    Features:
    0. Center control (1.0 if move is center, 0.0 otherwise)
    1. Corner control (1.0 if move is corner, 0.0 otherwise)
    2. Edge control (1.0 if move is edge, 0.0 otherwise)
    3. Win threat (1.0 if move creates immediate win, 0.0 otherwise)
    4. Block threat (1.0 if move blocks opponent win, 0.0 otherwise)

    Args:
        board: Current board state
        move: Position to evaluate (0-8)
        player: Player making the move (1 or -1)

    Returns:
        List of 5 feature values
    """
    features = [0.0] * NUM_FEATURES

    # Position-based features
    if move == 4:  # Center
        features[FEATURE_CENTER] = 1.0
    elif move in {0, 2, 6, 8}:  # Corners
        features[FEATURE_CORNER] = 1.0
    elif move in {1, 3, 5, 7}:  # Edges
        features[FEATURE_EDGE] = 1.0

    # Tactical features - need to check what happens after the move
    temp_board = Board()
    temp_state = board.get_state_flat().copy()

    # Check if this move creates a win
    temp_board._state = temp_state.copy()
    if temp_board.make_move(move, player) and temp_board.get_winner() == player:
        features[FEATURE_WIN_THREAT] = 1.0

    # Check if this move blocks opponent win
    opponent = -player

    # Try all opponent moves to see if any would win
    for opp_move in board.get_legal_moves():
        test_board = Board()
        test_board._state = temp_state.copy()
        # This opponent move would win, check if our move blocks it
        if (
            test_board.make_move(opp_move, opponent)
            and test_board.get_winner() == opponent
            and opp_move == move
        ):
            features[FEATURE_BLOCK_THREAT] = 1.0
            break

    return features


class FeatureGeneticAgent(Agent):
    """
    Genetic agent using feature-based weights.

    Selects moves based on weighted strategic features rather than
    individual cell weights. This reduces genome size from 9 to 5.
    """

    def __init__(self, player: int, weights: Sequence[float]) -> None:
        """
        Initialize feature-based genetic agent.

        Args:
            player: Player identifier (1 or -1)
            weights: Feature weights [center, corner, edge, win_threat, block_threat]
        """
        super().__init__(player)
        if len(weights) != NUM_FEATURES:
            msg = f"Expected {NUM_FEATURES} weights, got {len(weights)}"
            raise ValueError(msg)
        self.weights = np.array(weights, dtype=np.float64)

    @override
    def select_move(self, board: Board) -> int:
        """
        Select move by maximizing weighted feature score.

        Args:
            board: Current game state

        Returns:
            Selected move position (0-8)
        """
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            msg = "No legal moves available"
            raise ValueError(msg)

        best_move = legal_moves[0]
        best_score = float("-inf")

        for move in legal_moves:
            features = extract_features(board, move, self.player)
            score = np.dot(self.weights, features)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    @override
    def __repr__(self) -> str:
        return f"FeatureGA({self.player})"


@dataclass
class FeatureEvolutionResult:
    """Result of feature-based evolution run."""

    best_weights: tuple[float, ...]
    best_fitness: float
    history: list[dict[str, float]]
    algorithm: str = "DEAP"


def evaluate_feature_fitness(
    individual: Sequence[float],
    board_factory: Callable[[], Board],
    n_games: int,
    population: list[Sequence[float]] | None = None,
    self_play_fraction: float = 0.3,
) -> tuple[float]:
    """
    Evaluate fitness with self-play component.

    Args:
        individual: Feature weights to evaluate
        board_factory: Factory function for creating boards
        n_games: Number of games against each opponent type
        population: Current population for self-play (optional)
        self_play_fraction: Fraction of games to play against population members

    Returns:
        Tuple containing fitness score
    """
    agent = FeatureGeneticAgent(1, individual)
    total_score = 0.0
    total_games = 0

    # Games against fixed opponents
    for opponent_class in [RandomAgent, HeuristicAgent]:
        opponent = opponent_class(-1)
        for _ in range(n_games):
            board = board_factory()
            current_player = 1

            while not board.is_terminal():
                if current_player == 1:
                    move = agent.select_move(board)
                else:
                    move = opponent.select_move(board)
                board.make_move(move, current_player)
                current_player = -current_player

            winner = board.get_winner()
            if winner == 1:
                total_score += 1.0
            elif winner == 0:
                total_score += 0.5
            total_games += 1

    # Self-play games against population members
    if population and len(population) > 1:
        n_self_play = int(n_games * 2 * self_play_fraction)

        for _ in range(n_self_play):
            # Select random opponent from population (excluding self)
            opponent_weights = random.choice([w for w in population if w != individual])
            opponent = FeatureGeneticAgent(-1, opponent_weights)

            board = board_factory()
            current_player = 1

            while not board.is_terminal():
                if current_player == 1:
                    move = agent.select_move(board)
                else:
                    move = opponent.select_move(board)
                board.make_move(move, current_player)
                current_player = -current_player

            winner = board.get_winner()
            if winner == 1:
                total_score += 1.0
            elif winner == 0:
                total_score += 0.5
            total_games += 1

    return (total_score / total_games if total_games > 0 else 0.0,)


def run_feature_evolution(
    board_factory: Callable[[], Board],
    pop_size: int = 100,
    generations: int = 50,
    cx_pb: float = 0.5,
    mut_pb: float = 0.2,
    n_games: int = 10,
    seed: int | None = None,
    self_play_fraction: float = 0.3,
) -> FeatureEvolutionResult:
    """
    Run feature-based evolution with self-play.

    Args:
        board_factory: Factory for creating board instances
        pop_size: Population size
        generations: Number of generations
        cx_pb: Crossover probability
        mut_pb: Mutation probability
        n_games: Games per opponent type for fitness
        seed: Random seed for reproducibility
        self_play_fraction: Fraction of games against population (0.0-1.0)

    Returns:
        FeatureEvolutionResult with best weights and history
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create DEAP types if they don't exist
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Feature weights range: -2.0 to 2.0
    toolbox.register("attr_float", random.uniform, -2.0, 2.0)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=NUM_FEATURES,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evolution operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=pop_size)

    # Statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    # Track history
    history = []

    # Custom evaluation function with self-play
    def eval_with_population(individual: Sequence[float]) -> tuple[float]:
        return evaluate_feature_fitness(
            individual,
            board_factory,
            n_games,
            population=population,
            self_play_fraction=self_play_fraction,
        )

    toolbox.register("evaluate", eval_with_population)

    # Run evolution
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Record statistics
        record = stats.compile(population)
        history.append(
            {
                "generation": gen,
                "avg_fitness": record["avg"],
                "max_fitness": record["max"],
                "min_fitness": record["min"],
            }
        )

        print(f"Gen {gen:3d}: avg={record['avg']:.3f}, max={record['max']:.3f}")

        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace population
        population[:] = offspring

    # Final evaluation
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Get best individual
    best_individual = tools.selBest(population, 1)[0]

    return FeatureEvolutionResult(
        best_weights=tuple(best_individual),
        best_fitness=best_individual.fitness.values[0],
        history=history,
    )


def save_feature_weights(weights: Sequence[float], path: Path | None = None) -> None:
    """Save feature weights to disk."""
    target = (
        path or Path("src") / "tictactoe" / "weights" / "best" / "feature_weights.pkl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fp:
        pickle.dump(list(weights), fp)


def load_feature_weights(path: Path | None = None) -> list[float]:
    """Load feature weights from disk."""
    target = (
        path or Path("src") / "tictactoe" / "weights" / "best" / "feature_weights.pkl"
    )
    if not target.exists():
        msg = "Feature weights file not found. Run: 'uv run tictactoe evolve-features'"
        raise FileNotFoundError(msg)
    with target.open("rb") as fp:
        return pickle.load(fp)
