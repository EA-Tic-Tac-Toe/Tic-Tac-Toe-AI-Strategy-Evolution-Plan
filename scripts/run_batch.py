"""Run multiple GA evolutions and export aggregated reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tictactoe.analysis import evaluator
from tictactoe.analysis.evaluator import aggregate_outcomes_by_opponent
from tictactoe.agents.genetic_agent import save_weights
from tictactoe.board import Board
from tictactoe.cli import heatmap_cli, tune_ga_cli, visualize_experiment_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate GA evolution runs and export plots/reports.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of GA runs")
    parser.add_argument("--pop-size", type=int, default=60, help="Population size")
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations per run",
    )
    parser.add_argument(
        "--cx-pb",
        type=float,
        default=0.5,
        help="Crossover probability",
    )
    parser.add_argument(
        "--mut-pb",
        type=float,
        default=0.2,
        help="Mutation probability",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=4,
        help="Games per opponent during fitness evaluation",
    )
    parser.add_argument(
        "--games-per-match",
        type=int,
        default=20,
        help="Games against each opponent in final report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility",
    )
    parser.add_argument(
        "--selection",
        default="tournament",
        help="Selection strategy (tournament/roulette/best)",
    )
    parser.add_argument(
        "--output",
        default="results/experiments",
        help="Directory for experiment exports",
    )
    parser.add_argument(
        "--prefix",
        default="batch",
        help="Filename prefix for generated artifacts",
    )
    parser.add_argument(
        "--weights-out",
        default="src/tictactoe/weights/best",
        help="Path where the best run's weights should be pickled",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> evaluator.EvaluationConfig:
    return evaluator.EvaluationConfig(
        runs=args.runs,
        board_factory=lambda: Board(),
        pop_size=args.pop_size,
        generations=args.generations,
        cx_pb=args.cx_pb,
        mut_pb=args.mut_pb,
        n_games=args.n_games,
        games_per_match=args.games_per_match,
        base_seed=args.seed,
        selection=args.selection,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    print(
        "Running GA batch: runs={runs}, pop={pop}, gens={gens}, seed={seed}".format(
            runs=config.runs,
            pop=config.pop_size,
            gens=config.generations,
            seed=config.base_seed,
        )
    )
    series = evaluator.run_evaluation_series(config)
    for run in series.runs:
        print(
            f"Run {run.run_id}: best_fitness={run.best_fitness:.3f} seed={run.seed}"
        )
    print("Evolution batch finished. Aggregating results...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)
    experiment_base = base_dir / f"{args.prefix}_{timestamp}"

    export_paths = evaluator.export_evaluation(series, experiment_base)
    best_run = max(series.runs, key=lambda run: run.best_fitness)
    aggregate = aggregate_outcomes_by_opponent(series)

    summary_path = experiment_base.with_name(experiment_base.name + "_summary.json")
    summary = {
        "export": {k: str(v) for k, v in export_paths.items()},
        "best_run": {
            "run_id": best_run.run_id,
            "seed": best_run.seed,
            "best_fitness": best_run.best_fitness,
        },
        "aggregate_outcomes": [entry.to_dict() for entry in aggregate],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    weights_target = Path(args.weights_out)
    save_weights(best_run.evolution.best_weights, weights_target)

    plots_dir = Path("results/plots") / experiment_base.name
    visualize_experiment_cli(str(export_paths["json"]), str(plots_dir))

    tune_output = Path("results/tuning/latest")
    tune_ga_cli("configs/tuning.yaml", str(tune_output))

    heatmap_output = Path("results/plots") / f"{experiment_base.name}_weights"
    heatmap_cli(str(weights_target), str(heatmap_output))

    print("Experiment exports:")
    print(f"  JSON: {export_paths['json']}")
    print(f"  CSV:  {export_paths['csv']}")
    print(f"Summary: {summary_path}")
    print(f"Best weights saved to: {weights_target}")
    print(f"Plots stored under: {plots_dir}")
    print(f"Tuning artifacts under: {tune_output}")
    print(f"Heatmap stored at: {heatmap_output}")


if __name__ == "__main__":
    main()
