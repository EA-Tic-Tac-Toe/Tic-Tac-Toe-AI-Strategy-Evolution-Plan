# Tic-Tac-Toe AI Strategy Evolution Plan

**Authors:** Adrian Jaśkowiec, Jakub Kubicki, Tomasz Makowski  
**Python Version:** 3.14.0  
**Libraries:** NumPy, DEAP, pytest, ruff, mypy

## Overview

| Aspect     | Description                                                                                          |
|-------------|------------------------------------------------------------------------------------------------------|
| Objective   | Create an AI agent that learns to play Tic-Tac-Toe optimally using evolutionary strategies.          |
| Method      | Encode a strategy (or small neural net) as a genome, evolve via GA to maximize performance.          |
| Outcome     | Demonstrate improvement of agent fitness over generations and compare GA-based vs. fixed-rule play.  |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/EA-Tic-Tac-Toe/Tic-Tac-Toe-AI-Strategy-Evolution-Plan.git
cd Tic-Tac-Toe-AI-Strategy-Evolution-Plan

# Install dependencies (using uv)
uv sync

# Or run directly with uv
uv run tictactoe --help
```

### Usage

**Play against AI heuristic agent:**
```bash
uv run tictactoe play --opponent heuristic --player X
```

**Generate weights for Genetic agent:
```bash
uv run tictactoe evolve --pop_size 200 --generations 40 --n_games 40
```

**Play against Genetic agent:**
```bash
uv run tictactoe play --opponent genetic --player X
```

**Watch AI vs AI:**
```bash
uv run tictactoe demo --agent1 heuristic --agent2 genetic
```

**Evaluate agents:**
```bash
uv run tictactoe evaluate --agent1 heuristic --agent2 random --games 1000
```

**Visualize GA experiment analytics:**
```bash
uv run tictactoe visualize --experiment results/experiments/sample_experiment.json --output results/plots
```

**Run GA hyperparameter tuning:**
```bash
uv run tictactoe tune --config configs/tuning.yaml --output results/tuning/latest
```

**Render a weights heatmap:**
```bash
uv run tictactoe heatmap --weights src/tictactoe/weights/best --output results/plots/best_heatmap
```

### Evolution & Reporting Workflow

1. **Evolve new weights** (saves to `src/tictactoe/weights/best`):
   ```bash
   uv run tictactoe evolve --pop_size 80 --generations 30 --cx_pb 0.5 --mut_pb 0.2 --n_games 4 --seed 123
   ```
2. **Run batched experiments + reports** (JSON/CSV + plots + summary + optional pickle export):
   ```bash
   uv run python scripts/run_batch.py \
       --runs 5 \
       --pop-size 80 \
       --generations 30 \
       --output results/experiments \
       --prefix studyA \
       --weights-out src/tictactoe/weights/best/best_weights.pkl
   ```
   The script exports:
   - `*.json` + `*.csv` with full run histories,
   - `*_summary.json` containing aggregate win/draw/loss rates,
   - dark-mode plots (fitness curve, outcome bars, heatmap) under the same prefix,
   - `weights/studyA_best.pkl` (or the path you provide) holding the best genome.
3. **Generate additional visuals** from a stored experiment:
   ```bash
   uv run tictactoe visualize --experiment results/experiments/studyA_YYYYMMDD-HHMMSS.json --output results/plots/studyA
   ```
4. **Plot the final genome**:
   ```bash
   uv run tictactoe heatmap --weights src/tictactoe/weights/best --output results/plots/studyA_weights
   ```

### Development

**Run tests:**
```bash
uv run pytest -v
```

**Run tests with coverage:**
```bash
uv run pytest --cov=src --cov-report=term-missing
```

**Run linters:**
```bash
uv run ruff check src/ tests/
uv run mypy src/
```

---

## Project Structure

```
Tic-Tac-Toe-AI-Strategy-Evolution-Plan/
├── README.md
├── pyproject.toml          # Project configuration
├── .python-version         # Python 3.14.0
├── configs/
│   └── tuning.yaml         # Example GA tuning grid
├── scripts/
│   └── run_batch.py        # Batch execution/export helper
├── src/
│   └── tictactoe/
│       ├── __init__.py
│       ├── board.py
│       ├── game_runner.py
│       ├── cli.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   ├── tuning.py
│       │   └── plots.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── genetic_agent.py
│       │   ├── random_agent.py
│       │   └── heuristic_agent.py
│       └── weights/
│           └── best (pickle with latest evolved genome)
├── tests/
│   ├── conftest.py
│   ├── test_board.py
│   ├── test_agents.py
│   ├── test_game_runner.py
│   └── test_analysis.py
├── results/
│   ├── experiments/        # JSON/CSV exports from batch runs
│   ├── plots/              # Generated figures (PNG/SVG)
│   └── tuning/             # Grid-search outputs
└── notebooks/
    └── evolution_analysis.ipynb
```

---

## Features Implemented (Phase 1)

✅ **Modern Python 3.14 Features:**
- Structural pattern matching (`match`/`case`)
- Union type operator (`X | None`)
- `@override` decorator
- Walrus operator (`:=`)
- Frozen dataclasses

✅ **Game Environment:**
- Efficient NumPy-based board representation
- Vectorized win detection
- Move history tracking
- Legal move validation

✅ **Baseline Agents:**
- **RandomAgent:** Uniform random move selection (with seeding for reproducibility)
- **HeuristicAgent:** Strategic agent with:
  - Win detection
  - Opponent blocking
  - Positional priorities (center > corners > edges)

✅ **Evaluation Framework:**
- Single game execution
- Multi-game matchup evaluation
- Performance statistics
- Alternating first-player fairness

✅ **CLI Interface:**
- Interactive human vs AI play
- AI vs AI demonstrations
- Statistical evaluation of agent matchups

✅ **Testing & Quality:**
- 43 comprehensive tests (all passing ✅)
- 98% coverage on core game logic
- Ruff linter (passing ✅)
- Mypy strict type checking (passing ✅)

---

## Features Implemented (Phase 2)
✅ **Genetic Algorithm System:**
- DEAP‑based evolutionary optimization,
- Configurable population size, generations, crossover, mutation.

✅ **GeneticAgent Implementation:**
- Weight‑based policy engine (cell‑weights or feature‑weights),
- Immediate tactical fallback:
  - Win if possible,
  - Block opponent win,
  - Deterministic behavior ensures tactical correctness even with imperfect evolved weights.

✅ **Persistence Layer:**
- Save weights,
- Load weights for reuse,
- Compatible with CLI and integration tests.

✅ **Evolution Framework:**
- Multi‑game fitness evaluation,
- Opponent diversity: RandomAgent, HeuristicAgent, GeneticAgent,

## Milestone 3 Highlights

- **Analysis Toolkit:** `tictactoe.analysis.{evaluator,tuning,plots}` delivers repeated GA evaluations, grid-search tuning, and dark-mode Matplotlib charts (fitness curve, W/D/L bars, heatmaps, configuration comparisons).
- **Batch Automation:** `scripts/run_batch.py` orchestrates multiple GA runs, logs per-run fitness, exports JSON/CSV summaries, generates presentation-ready plots, and stores the best genome (`--weights-out`).
- **CLI Enhancements:** `tictactoe visualize/tune/heatmap` commands hook directly into the analysis modules for quick reporting.
- **Notebook Workflow:** `notebooks/evolution_analysis.ipynb` demonstrates loading experiment artifacts, comparing runs, and interpreting results.

## Example Results

**Heuristic vs Random (100 games):**
```
Heuristic wins:  82 (82.0%)
Random wins:      0 ( 0.0%)
Draws:           18 (18.0%)
Avg game length: 6.4 moves
```

**Heuristic vs Heuristic (10 games):**
```
Agent1 wins:     0 ( 0.0%)
Agent2 wins:     0 ( 0.0%)
Draws:          10 (100.0%)
```

**Genetic vs Heuristic (1000 games):**
```
Heuristic wins:   500 (50.0%)
Genetic wins:   500 (50.0%)
Draws:            0 ( 0.0%)
Avg game length: 7.0 moves
```

**Genetic vs Random (1000 games):**
```
Random wins:   262 (26.2%)
Genetic wins:   705 (70.5%)
Draws:           33 ( 3.3%)
Avg game length: 6.3 moves
```


---

## Estimated Timeline

| Week | Task                                      |
|------|-------------------------------------------|
| 2    | Game environment + baseline agent         |
| 4    | GA setup with DEAP                        |
| 6    | Fitness evaluation design + initial experiments |
| 8    | Optimization, tuning, visualization       |
| 9    | Optional jMetalPy experiments             |
| 10   | Report + presentation preparation         |

