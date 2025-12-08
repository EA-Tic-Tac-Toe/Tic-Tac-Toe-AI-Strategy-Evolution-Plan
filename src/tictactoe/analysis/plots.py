"""Matplotlib plotting utilities for GA experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tictactoe.agents.genetic_agent import EvolutionHistoryEntry
from tictactoe.analysis.evaluator import MatchOutcome
from tictactoe.analysis.tuning import TuningRecord

plt.style.use("dark_background")
plt.rcParams.update(
    {
        "figure.facecolor": "#111111",
        "axes.facecolor": "#111111",
        "savefig.facecolor": "#111111",
        "axes.labelcolor": "#EEEEEE",
        "text.color": "#EEEEEE",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.color": "#DDDDDD",
        "ytick.color": "#DDDDDD",
    },
)

_SMALL = (7, 5)
_LARGE = (11, 7)


def _figure_size(size: str) -> tuple[float, float]:
    return _LARGE if size == "large" else _SMALL


def _save_figure(fig: Figure, output_base: Path) -> dict[str, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "svg": svg_path}


def plot_fitness_curve(
    history: Sequence[EvolutionHistoryEntry],
    output_base: Path,
    *,
    label: str | None = None,
    size: str = "large",
) -> dict[str, Path]:
    """Plot min/avg/max fitness per generation."""
    generations = [entry.generation for entry in history]
    avg = [entry.avg_fitness for entry in history]
    min_vals = [entry.min_fitness for entry in history]
    max_vals = [entry.max_fitness for entry in history]

    fig, ax = plt.subplots(figsize=_figure_size(size))
    ax.plot(generations, avg, color="#4FD1C5", linewidth=2.5, label=label or "Mean")
    ax.fill_between(
        generations,
        min_vals,
        max_vals,
        color="#2A9D8F",
        alpha=0.3,
        label="Min-Max range",
    )

    ax.set_title("Fitness Curve")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(True, color="#333333", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, loc="lower right")
    return _save_figure(fig, output_base)


def plot_outcome_bars(
    outcomes: Sequence[MatchOutcome],
    output_base: Path,
    *,
    size: str = "small",
) -> dict[str, Path]:
    """Plot win/draw/loss ratios for each opponent."""
    labels = [outcome.opponent for outcome in outcomes]
    wins = [outcome.win_rate * 100 for outcome in outcomes]
    draws = [outcome.draw_rate * 100 for outcome in outcomes]
    losses = [outcome.loss_rate * 100 for outcome in outcomes]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=_figure_size(size))
    ax.bar(x - width, wins, width, label="Wins", color="#2ECC40")
    ax.bar(x, draws, width, label="Draws", color="#FFDC00")
    ax.bar(x + width, losses, width, label="Losses", color="#FF4136")

    ax.set_title("Match Outcomes")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    for idx, value in enumerate(wins):
        ax.text(x[idx] - width, value + 2, f"{value:.1f}%", ha="center", va="bottom")
    return _save_figure(fig, output_base)


def plot_weight_heatmap(
    weights: Sequence[float],
    output_base: Path,
    *,
    size: str = "small",
) -> dict[str, Path]:
    """Visualize genome weights in 3x3 board layout."""
    grid = np.array(weights, dtype=float).reshape(3, 3)
    fig, ax = plt.subplots(figsize=_figure_size(size))
    heatmap = ax.imshow(grid, cmap="viridis")
    ax.set_title("Strategy Heatmap")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_yticklabels(["0", "1", "2"])

    for (i, j), value in np.ndenumerate(grid):
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            color="#FFFFFF",
        )

    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label="Weight")
    return _save_figure(fig, output_base)


def plot_tuning_comparison(
    records: Sequence[TuningRecord],
    output_base: Path,
    *,
    metric: str = "avg_best_fitness",
    size: str = "large",
) -> dict[str, Path]:
    """Compare GA configurations using bar chart."""
    if not records:
        raise ValueError("No tuning records provided")

    metric_values = [getattr(record, metric) for record in records]
    labels = [
        f"pop={record.params.pop_size}, cx={record.params.cx_pb:.2f}, "
        f"mut={record.params.mut_pb:.2f}, sel={record.params.selection}"
        for record in records
    ]

    sorted_indices = np.argsort(metric_values)[::-1]
    sorted_values = [metric_values[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    fig_height = max(_figure_size(size)[1], max(4.0, len(records) * 0.6))
    fig, ax = plt.subplots(figsize=(_figure_size(size)[0], fig_height))
    y = np.arange(len(sorted_labels))
    ax.barh(y, sorted_values, color="#3498DB")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels)
    ax.invert_yaxis()
    ax.set_title("GA Configuration Comparison")
    ax.set_xlabel(metric.replace("_", " ").title())

    for idx, value in enumerate(sorted_values):
        ax.text(value + 0.01, y[idx], f"{value:.3f}", va="center")

    return _save_figure(fig, output_base)
