"""Hyperparameter tuning utilities for the genetic algorithm."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from typing import TypeVar, cast

from tictactoe.agents.genetic_agent import BoardFactory, EvolutionResult, run_evolution
from tictactoe.board import Board

_Src = TypeVar("_Src")
_Dst = TypeVar("_Dst")


@dataclass(frozen=True)
class HyperParams:
    """Single GA hyperparameter configuration."""

    pop_size: int
    mut_pb: float
    cx_pb: float
    selection: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "pop_size": self.pop_size,
            "mut_pb": self.mut_pb,
            "cx_pb": self.cx_pb,
            "selection": self.selection,
        }


@dataclass(frozen=True)
class TuningConfig:
    """Grid search configuration."""

    populations: tuple[int, ...]
    mutation_probs: tuple[float, ...]
    crossover_probs: tuple[float, ...]
    selections: tuple[str, ...]
    generations: int = 20
    n_games: int = 4
    runs_per_setting: int = 2
    board_factory: BoardFactory = Board
    base_seed: int = 900


@dataclass(frozen=True)
class TuningRecord:
    """Aggregated results for a single hyperparameter set."""

    index: int
    params: HyperParams
    runs: tuple[EvolutionResult, ...]

    @property
    def avg_best_fitness(self) -> float:
        return mean(run.best_fitness for run in self.runs)

    @property
    def max_best_fitness(self) -> float:
        return max(run.best_fitness for run in self.runs)

    @property
    def min_best_fitness(self) -> float:
        return min(run.best_fitness for run in self.runs)

    @property
    def best_run(self) -> EvolutionResult:
        return max(self.runs, key=lambda run: run.best_fitness)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "params": self.params.to_dict(),
            "avg_best_fitness": self.avg_best_fitness,
            "max_best_fitness": self.max_best_fitness,
            "min_best_fitness": self.min_best_fitness,
            "best_run": {
                "seed": self.best_run.seed,
                "best_fitness": self.best_run.best_fitness,
                "selection": self.best_run.selection,
                "weights": list(self.best_run.best_weights),
            },
        }


def grid_search(config: TuningConfig) -> tuple[TuningRecord, ...]:
    """Execute grid search over provided hyperparameters."""
    if (
        not config.populations
        or not config.mutation_probs
        or not config.crossover_probs
        or not config.selections
    ):
        msg = "Tuning grid must contain at least one value for each parameter"
        raise ValueError(msg)
    if config.runs_per_setting < 1:
        msg = "runs_per_setting must be >= 1"
        raise ValueError(msg)

    records: list[TuningRecord] = []
    seed_cursor = config.base_seed

    combos = list(
        product(
            config.populations,
            config.mutation_probs,
            config.crossover_probs,
            config.selections,
        ),
    )

    for idx, (pop_size, mut_pb, cx_pb, selection) in enumerate(combos):
        runs: list[EvolutionResult] = []
        for _ in range(config.runs_per_setting):
            evo_result = run_evolution(
                board_factory=config.board_factory,
                pop_size=pop_size,
                generations=config.generations,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                n_games=config.n_games,
                seed=seed_cursor,
                selection=selection,
            )
            runs.append(evo_result)
            seed_cursor += 1
        records.append(
            TuningRecord(
                index=idx,
                params=HyperParams(
                    pop_size=pop_size,
                    mut_pb=mut_pb,
                    cx_pb=cx_pb,
                    selection=selection,
                ),
                runs=tuple(runs),
            ),
        )
    return tuple(records)


def export_tuning_results(
    records: Sequence[TuningRecord],
    output_base: Path,
) -> dict[str, Path]:
    """Export tuning results to JSON and CSV files."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")

    json_path.write_text(
        json.dumps(
            {"records": [record.to_dict() for record in records]},
            indent=2,
        ),
    )

    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "index",
                "pop_size",
                "mut_pb",
                "cx_pb",
                "selection",
                "avg_best_fitness",
                "max_best_fitness",
                "min_best_fitness",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "index": record.index,
                    "pop_size": record.params.pop_size,
                    "mut_pb": record.params.mut_pb,
                    "cx_pb": record.params.cx_pb,
                    "selection": record.params.selection,
                    "avg_best_fitness": record.avg_best_fitness,
                    "max_best_fitness": record.max_best_fitness,
                    "min_best_fitness": record.min_best_fitness,
                },
            )
    return {"json": json_path, "csv": csv_path}


def _require_sequence(value: object, key: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        msg = f"{key} must be provided as a sequence"
        raise TypeError(msg)
    return value


def _convert_sequence(  # noqa: UP047
    data: Mapping[str, object],
    key: str,
    converter: Callable[[_Src], _Dst],
) -> tuple[_Dst, ...]:
    if key not in data:
        msg = f"Missing tuning key: {key}"
        raise ValueError(msg)
    sequence = _require_sequence(data[key], key)
    return tuple(converter(cast(_Src, item)) for item in sequence)


def _item_to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    msg = "Sequence entries must be int-compatible"
    raise TypeError(msg)


def _item_to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    msg = "Sequence entries must be float-compatible"
    raise TypeError(msg)


def _item_to_str(value: object) -> str:
    return str(value)


def _as_int(value: object, key: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    msg = f"{key} must be convertible to int"
    raise TypeError(msg)


def config_from_mapping(data: Mapping[str, object]) -> TuningConfig:
    """Create configuration from parsed YAML/JSON mapping."""
    populations = _convert_sequence(data, "populations", _item_to_int)
    mutation_probs = _convert_sequence(data, "mutation_probs", _item_to_float)
    crossover_probs = _convert_sequence(data, "crossover_probs", _item_to_float)
    selections = _convert_sequence(data, "selections", _item_to_str)
    return TuningConfig(
        populations=populations,
        mutation_probs=mutation_probs,
        crossover_probs=crossover_probs,
        selections=selections,
        generations=_as_int(data.get("generations", 20), "generations", 20),
        n_games=_as_int(data.get("n_games", 4), "n_games", 4),
        runs_per_setting=_as_int(
            data.get("runs_per_setting", 2),
            "runs_per_setting",
            2,
        ),
        base_seed=_as_int(data.get("base_seed", 900), "base_seed", 900),
    )
