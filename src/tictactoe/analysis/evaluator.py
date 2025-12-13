"""Evaluation helpers for repeated genetic algorithm experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tictactoe.agents.base import Agent
from tictactoe.agents.genetic_agent import (
    BoardFactory,
    EvolutionHistoryEntry,
    EvolutionResult,
    GeneticAgent,
    run_evolution,
)
from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.board import Board
from tictactoe.game_runner import evaluate_matchup

AgentFactory = Callable[[int, int | None], Agent]


def _default_opponents() -> tuple[OpponentSpec, ...]:
    return (
        OpponentSpec("Heuristic", lambda player, _: HeuristicAgent(player)),
        OpponentSpec("Random", lambda player, seed: RandomAgent(player, seed=seed)),
    )


@dataclass(frozen=True)
class OpponentSpec:
    """Configuration for an opponent used in experiment evaluation."""

    name: str
    factory: AgentFactory


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for running repeated GA experiments."""

    runs: int = 3
    board_factory: BoardFactory = Board
    pop_size: int = 60
    generations: int = 20
    cx_pb: float = 0.5
    mut_pb: float = 0.2
    n_games: int = 4
    games_per_match: int = 20
    base_seed: int = 42
    selection: str = "tournament"
    opponents: tuple[OpponentSpec, ...] = field(default_factory=_default_opponents)


@dataclass(frozen=True)
class MatchOutcome:
    """Win/draw/loss summary for a single opponent matchup."""

    opponent: str
    wins: int
    losses: int
    draws: int
    total_games: int
    avg_game_length: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total_games if self.total_games else 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "opponent": self.opponent,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_games": self.total_games,
            "avg_game_length": self.avg_game_length,
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
        }


@dataclass(frozen=True)
class EvaluationRun:
    """Record for a single GA run."""

    run_id: int
    seed: int
    evolution: EvolutionResult
    outcomes: tuple[MatchOutcome, ...]

    @property
    def best_fitness(self) -> float:
        return self.evolution.best_fitness

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "best_fitness": self.best_fitness,
            "selection": self.evolution.selection,
            "best_weights": list(self.evolution.best_weights),
            "history": [
                {
                    "generation": entry.generation,
                    "min_fitness": entry.min_fitness,
                    "avg_fitness": entry.avg_fitness,
                    "max_fitness": entry.max_fitness,
                }
                for entry in self.evolution.history
            ],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True)
class EvaluationSeriesResult:
    """Full set of GA runs along with metadata."""

    config: EvaluationConfig
    runs: tuple[EvaluationRun, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "runs": self.config.runs,
                "pop_size": self.config.pop_size,
                "generations": self.config.generations,
                "cx_pb": self.config.cx_pb,
                "mut_pb": self.config.mut_pb,
                "n_games": self.config.n_games,
                "games_per_match": self.config.games_per_match,
                "base_seed": self.config.base_seed,
                "selection": self.config.selection,
                "opponents": [spec.name for spec in self.config.opponents],
            },
            "runs": [run.to_dict() for run in self.runs],
        }

    def aggregate(self) -> MatchOutcome:
        """Aggregate outcomes across all runs/opponents."""
        wins = losses = draws = total_games = 0
        weighted_length = 0.0
        for run in self.runs:
            for outcome in run.outcomes:
                wins += outcome.wins
                losses += outcome.losses
                draws += outcome.draws
                total_games += outcome.total_games
                weighted_length += outcome.avg_game_length * outcome.total_games
        avg_length = weighted_length / total_games if total_games else 0.0
        return MatchOutcome(
            opponent="aggregate",
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=total_games,
            avg_game_length=avg_length,
        )


def aggregate_outcomes_by_opponent(
    series: EvaluationSeriesResult,
) -> tuple[MatchOutcome, ...]:
    """Aggregate outcomes grouped by opponent name."""
    buckets: dict[str, dict[str, float]] = {}
    for run in series.runs:
        for outcome in run.outcomes:
            bucket = buckets.setdefault(
                outcome.opponent,
                {"wins": 0, "losses": 0, "draws": 0, "games": 0, "length": 0.0},
            )
            bucket["wins"] += outcome.wins
            bucket["losses"] += outcome.losses
            bucket["draws"] += outcome.draws
            bucket["games"] += outcome.total_games
            bucket["length"] += outcome.avg_game_length * outcome.total_games

    aggregated: list[MatchOutcome] = []
    for opponent, stats in sorted(buckets.items()):
        games = int(stats["games"])
        avg_length = stats["length"] / games if games else 0.0
        aggregated.append(
            MatchOutcome(
                opponent=opponent,
                wins=int(stats["wins"]),
                losses=int(stats["losses"]),
                draws=int(stats["draws"]),
                total_games=games,
                avg_game_length=avg_length,
            ),
        )
    return tuple(aggregated)


def run_evaluation_series(config: EvaluationConfig) -> EvaluationSeriesResult:
    """Execute multiple GA runs and evaluate each champion."""
    runs: list[EvaluationRun] = []
    print(
        f"Starting evaluation series: runs={config.runs}, pop={config.pop_size}, gens={config.generations}"
    )
    for run_idx in range(config.runs):
        print(f"  Running evaluation {run_idx + 1}/{config.runs}...")
        run_seed = config.base_seed + run_idx
        evo_result = run_evolution(
            board_factory=config.board_factory,
            pop_size=config.pop_size,
            generations=config.generations,
            cx_pb=config.cx_pb,
            mut_pb=config.mut_pb,
            n_games=config.n_games,
            seed=run_seed,
            selection=config.selection,
        )
        outcomes = tuple(
            _evaluate_against_opponent(
                weights=list(evo_result.best_weights),
                opponent=opponent,
                games=config.games_per_match,
                run_seed=run_seed + idx * 101,
            )
            for idx, opponent in enumerate(config.opponents)
        )
        runs.append(
            EvaluationRun(
                run_id=run_idx,
                seed=run_seed,
                evolution=evo_result,
                outcomes=outcomes,
            ),
        )
    return EvaluationSeriesResult(config=config, runs=tuple(runs))


def _evaluate_against_opponent(
    *,
    weights: Sequence[float],
    opponent: OpponentSpec,
    games: int,
    run_seed: int,
) -> MatchOutcome:
    """Evaluate evolved weights against a single opponent from both sides."""
    # GA as X
    ga_first = GeneticAgent(1, weights, name="GeneticEval")
    opponent_o = opponent.factory(-1, run_seed)
    stats_first = evaluate_matchup(
        ga_first,
        opponent_o,
        num_games=games,
        alternate_first=False,
    )

    # GA as O
    opponent_x = opponent.factory(1, run_seed + 1)
    ga_second = GeneticAgent(-1, weights, name="GeneticEval")
    stats_second = evaluate_matchup(
        opponent_x,
        ga_second,
        num_games=games,
        alternate_first=False,
    )

    wins = stats_first.agent1_wins + stats_second.agent2_wins
    losses = stats_first.agent2_wins + stats_second.agent1_wins
    draws = stats_first.draws + stats_second.draws
    total_games = stats_first.total_games + stats_second.total_games
    weighted_length = (
        stats_first.avg_game_length * stats_first.total_games
        + stats_second.avg_game_length * stats_second.total_games
    )
    avg_length = weighted_length / total_games if total_games else 0.0

    return MatchOutcome(
        opponent=opponent.name,
        wins=wins,
        losses=losses,
        draws=draws,
        total_games=total_games,
        avg_game_length=avg_length,
    )


def export_evaluation(
    series: EvaluationSeriesResult,
    output_base: Path,
) -> dict[str, Path]:
    """Persist evaluation data to JSON + CSV."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")

    json_path.write_text(json.dumps(series.to_dict(), indent=2))

    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "run_id",
                "generation",
                "min_fitness",
                "avg_fitness",
                "max_fitness",
            ],
        )
        writer.writeheader()
        for run in series.runs:
            for entry in run.evolution.history:
                writer.writerow(
                    {
                        "run_id": run.run_id,
                        "generation": entry.generation,
                        "min_fitness": entry.min_fitness,
                        "avg_fitness": entry.avg_fitness,
                        "max_fitness": entry.max_fitness,
                    },
                )
    return {"json": json_path, "csv": csv_path}


def load_evaluation(json_path: Path) -> EvaluationSeriesResult:
    """Load evaluation result from JSON."""
    data = json.loads(json_path.read_text())
    config_data = data["config"]
    config = EvaluationConfig(
        runs=config_data["runs"],
        pop_size=config_data["pop_size"],
        generations=config_data["generations"],
        cx_pb=config_data["cx_pb"],
        mut_pb=config_data["mut_pb"],
        n_games=config_data["n_games"],
        games_per_match=config_data["games_per_match"],
        base_seed=config_data["base_seed"],
        selection=config_data["selection"],
    )
    runs: list[EvaluationRun] = []
    for run_data in data["runs"]:
        history = tuple(
            EvolutionHistoryEntry(
                generation=entry["generation"],
                min_fitness=entry["min_fitness"],
                avg_fitness=entry["avg_fitness"],
                max_fitness=entry["max_fitness"],
            )
            for entry in run_data["history"]
        )
        evolution = EvolutionResult(
            best_weights=tuple(run_data["best_weights"]),
            history=history,
            best_fitness=run_data["best_fitness"],
            selection=run_data.get("selection", config.selection),
            seed=run_data["seed"],
        )
        outcomes = tuple(
            MatchOutcome(
                opponent=out["opponent"],
                wins=out["wins"],
                losses=out["losses"],
                draws=out["draws"],
                total_games=out["total_games"],
                avg_game_length=out["avg_game_length"],
            )
            for out in run_data["outcomes"]
        )
        runs.append(
            EvaluationRun(
                run_id=run_data["run_id"],
                seed=run_data["seed"],
                evolution=evolution,
                outcomes=outcomes,
            ),
        )
    return EvaluationSeriesResult(config=config, runs=tuple(runs))


# --------------------------
# jMetalPy Comparison Tools
# --------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Comparison between DEAP and jMetalPy evolution results."""

    deap_fitness: float
    deap_complexity: float
    deap_weights: tuple[float, ...]
    jmetal_pareto_size: int
    jmetal_best_fitness: float
    jmetal_best_complexity: float
    jmetal_selected_weights: tuple[float, ...]
    jmetal_objectives: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "deap": {
                "fitness": self.deap_fitness,
                "complexity": self.deap_complexity,
                "weights": list(self.deap_weights),
            },
            "jmetal": {
                "pareto_size": self.jmetal_pareto_size,
                "best_fitness": self.jmetal_best_fitness,
                "best_complexity": self.jmetal_best_complexity,
                "selected_weights": list(self.jmetal_selected_weights),
                "objectives": [
                    {"fitness": obj[0], "complexity": obj[1]}
                    for obj in self.jmetal_objectives
                ],
            },
        }


def compare_deap_jmetal(
    deap_weights: Sequence[float],
    jmetal_pareto_front: Sequence[tuple[float, ...]],
    jmetal_objectives: Sequence[tuple[float, float]],
    selection_strategy: str = "balanced",
) -> ComparisonResult:
    """
    Compare DEAP single-objective result with jMetalPy multi-objective results.

    Args:
        deap_weights: Best weights from DEAP evolution
        jmetal_pareto_front: Pareto front solutions from jMetalPy
        jmetal_objectives: Corresponding (fitness, complexity) values
        selection_strategy: How to select from Pareto front ("balanced", "fitness", "simple")

    Returns:
        ComparisonResult with metrics for both approaches
    """
    # Calculate DEAP complexity
    deap_complexity = sum(abs(w) for w in deap_weights)

    # We need to evaluate DEAP fitness, but for now we'll use a placeholder
    # In practice, you'd evaluate it the same way as during evolution
    deap_fitness = 0.0  # Placeholder - should be evaluated

    # Find best fitness in jMetal pareto front
    best_fitness_idx = max(
        range(len(jmetal_objectives)), key=lambda i: jmetal_objectives[i][0]
    )
    jmetal_best_fitness = jmetal_objectives[best_fitness_idx][0]

    # Find best (lowest) complexity
    best_complexity_idx = min(
        range(len(jmetal_objectives)), key=lambda i: jmetal_objectives[i][1]
    )
    jmetal_best_complexity = jmetal_objectives[best_complexity_idx][1]

    # Select solution based on strategy
    if selection_strategy == "fitness":
        selected_idx = best_fitness_idx
    elif selection_strategy == "simple":
        selected_idx = best_complexity_idx
    else:  # balanced
        # Find solution closest to ideal point (max fitness, min complexity)
        fitness_values = [obj[0] for obj in jmetal_objectives]
        complexity_values = [obj[1] for obj in jmetal_objectives]

        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        max_complexity = max(complexity_values)
        min_complexity = min(complexity_values)

        fitness_range = max_fitness - min_fitness if max_fitness != min_fitness else 1.0
        complexity_range = (
            max_complexity - min_complexity if max_complexity != min_complexity else 1.0
        )

        best_distance = float("inf")
        selected_idx = 0

        for i, (fitness, complexity) in enumerate(jmetal_objectives):
            norm_fitness = (fitness - min_fitness) / fitness_range
            norm_complexity = (complexity - min_complexity) / complexity_range

            # Distance to ideal (1, 0)
            distance = ((1.0 - norm_fitness) ** 2 + norm_complexity**2) ** 0.5

            if distance < best_distance:
                best_distance = distance
                selected_idx = i

    jmetal_selected_weights = jmetal_pareto_front[selected_idx]

    return ComparisonResult(
        deap_fitness=deap_fitness,
        deap_complexity=deap_complexity,
        deap_weights=tuple(deap_weights),
        jmetal_pareto_size=len(jmetal_pareto_front),
        jmetal_best_fitness=jmetal_best_fitness,
        jmetal_best_complexity=jmetal_best_complexity,
        jmetal_selected_weights=jmetal_selected_weights,
        jmetal_objectives=tuple(jmetal_objectives),
    )
