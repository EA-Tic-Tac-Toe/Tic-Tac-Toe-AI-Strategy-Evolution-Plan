"""Tests for analysis utilities."""

from pathlib import Path

from tictactoe.agents.genetic_agent import EvolutionHistoryEntry, EvolutionResult
from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.analysis import evaluator, plots, tuning
from tictactoe.board import Board
from tictactoe.game_runner import evaluate_matchup


def test_evaluation_series_export_and_load(tmp_path) -> None:
    """Evaluation pipeline should be reproducible and exportable."""
    config = evaluator.EvaluationConfig(
        runs=1,
        board_factory=lambda: Board(),
        pop_size=6,
        generations=1,
        cx_pb=0.5,
        mut_pb=0.1,
        n_games=1,
        games_per_match=2,
        base_seed=321,
    )
    series = evaluator.run_evaluation_series(config)
    assert len(series.runs) == 1
    aggregate = evaluator.aggregate_outcomes_by_opponent(series)
    assert aggregate

    output_base = tmp_path / "results" / "experiment"
    paths = evaluator.export_evaluation(series, output_base)
    assert paths["json"].exists()
    assert paths["csv"].exists()

    loaded = evaluator.load_evaluation(paths["json"])
    assert loaded.runs[0].best_fitness == series.runs[0].best_fitness


def test_plot_generation(tmp_path) -> None:
    """Plot helpers should produce PNG and SVG outputs."""
    history = [
        EvolutionHistoryEntry(
            generation=i,
            min_fitness=0.1 * i,
            avg_fitness=0.2 * i,
            max_fitness=0.3 * i,
        )
        for i in range(3)
    ]
    outcomes = (
        evaluator.MatchOutcome(
            opponent="Random",
            wins=3,
            losses=1,
            draws=0,
            total_games=4,
            avg_game_length=5.0,
        ),
    )

    paths = plots.plot_fitness_curve(history, tmp_path / "fitness")
    assert Path(paths["png"]).exists()
    outcome_paths = plots.plot_outcome_bars(outcomes, tmp_path / "outcomes")
    assert Path(outcome_paths["svg"]).exists()
    heatmap_paths = plots.plot_weight_heatmap(
        [0.1 * i for i in range(9)],
        tmp_path / "heatmap",
    )
    assert Path(heatmap_paths["png"]).exists()

    sample_record = tuning.TuningRecord(
        index=0,
        params=tuning.HyperParams(
            pop_size=4,
            mut_pb=0.2,
            cx_pb=0.6,
            selection="tournament",
        ),
        runs=(
            EvolutionResult(
                best_weights=tuple(0.0 for _ in range(9)),
                history=tuple(history),
                best_fitness=0.5,
                selection="tournament",
                seed=1,
            ),
        ),
    )
    comparison = plots.plot_tuning_comparison([sample_record], tmp_path / "comparison")
    assert Path(comparison["svg"]).exists()


def test_tuning_grid_deterministic(tmp_path) -> None:
    """Grid search should be deterministic with fixed seeds."""
    config = tuning.TuningConfig(
        populations=(4,),
        mutation_probs=(0.2,),
        crossover_probs=(0.5,),
        selections=("tournament",),
        generations=1,
        n_games=1,
        runs_per_setting=1,
        board_factory=lambda: Board(),
        base_seed=77,
    )
    first = tuning.grid_search(config)
    second = tuning.grid_search(config)
    assert first[0].best_run.best_fitness == second[0].best_run.best_fitness

    artifacts = tuning.export_tuning_results(first, tmp_path / "tuning" / "grid")
    assert artifacts["json"].exists()
    assert artifacts["csv"].exists()


def test_gameplay_regression_check() -> None:
    """Existing gameplay logic should remain stable."""
    stats = evaluate_matchup(
        HeuristicAgent(1),
        HeuristicAgent(-1),
        num_games=4,
        alternate_first=False,
    )
    assert stats.draws == 4
