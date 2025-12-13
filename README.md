# Tic-Tac-Toe AI Strategy Evolution Plan

**Authors:** Adrian Jaśkowiec, Jakub Kubicki, Tomasz Makowski  
**Python Version:** 3.14.0  
**Libraries:** NumPy, DEAP, jMetalPy, pytest, ruff, mypy

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

### Multi-Objective Evolution with jMetalPy

**Run multi-objective evolution (NSGA-II):**
```bash
uv run tictactoe evolve-jmetal --pop_size 100 --max_evaluations 25000 --algorithm NSGA-II --n_games 10
```

**Play against jMetalPy-evolved agent:**
```bash
uv run tictactoe play --opponent jmetal --player X
```

**Visualize Pareto front:**
```bash
uv run tictactoe pareto --pareto src/tictactoe/weights/best/pareto_front.pkl --output results/plots/pareto --strategy balanced
```

**Compare DEAP vs jMetalPy:**
```bash
uv run tictactoe compare \
    --deap src/tictactoe/weights/best/best_weights.pkl \
    --jmetal src/tictactoe/weights/best/pareto_front.pkl \
    --output results/experiments/comparison
```

### Feature-Based Evolution with Self-Play

**Run feature-based evolution:**
```bash
uv run tictactoe evolve-features --pop_size 100 --generations 50 --n_games 10 --self_play 0.3
```

**Play against feature-based agent:**
```bash
uv run tictactoe play --opponent feature --player X
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

## Milestone 4: Multi-Objective Optimization with jMetalPy

✅ **Multi-Objective Evolution:**
- **jMetalPy Integration:** NSGA-II and NSGA-III algorithms for multi-objective optimization
- **Dual Objectives:** 
  - Objective 1: Maximize fitness (win rate against opponents)
  - Objective 2: Minimize complexity (L1-norm of weights)
- **Pareto Front:** Discover multiple optimal solutions with different trade-offs

✅ **JMetalAgent Implementation:**
- Weight-based policy like GeneticAgent
- Evolved using multi-objective algorithms
- Multiple solution strategies: fitness-focused, simplicity-focused, or balanced

✅ **Comparison Framework:**
- Direct comparison between DEAP (single-objective) and jMetalPy (multi-objective)
- Pareto front visualization
- Trade-off analysis between performance and model complexity
- Solution selection strategies

✅ **CLI Commands:**
- `evolve-jmetal`: Run multi-objective evolution with NSGA-II or NSGA-III
- `compare`: Compare DEAP and jMetalPy results
- `pareto`: Visualize and select solutions from Pareto front
- Play against jMetalPy-evolved agents

✅ **Analysis Tools:**
- Pareto front plotting
- DEAP vs jMetalPy comparison plots
- Weight heatmaps for different selection strategies
- Interactive Jupyter notebook: `notebooks/jmetal_comparison.ipynb`

### Why Multi-Objective Optimization?

**DEAP (Single-Objective):**
- Maximizes only fitness (win rate)
- Returns a single "best" solution
- May produce complex, overfitted solutions

**jMetalPy (Multi-Objective):**
- Optimizes both fitness AND simplicity
- Returns a Pareto front of optimal trade-offs
- Allows selection based on priorities:
  - **Fitness strategy:** Best performance, may be complex
  - **Simple strategy:** Minimal complexity, interpretable
  - **Balanced strategy:** Compromise between both objectives

**Use Cases:**
- When model interpretability matters
- When you want to avoid overfitting
- When multiple stakeholders have different priorities
- When exploring design space is valuable

---

## Milestone 5: Feature-Based Evolution with Self-Play

### Rationale

**Traditional Approach (Cell Weights):**
- 9 individual weights (one per board position)
- No strategic knowledge encoded
- High dimensionality
- Position-specific, not generalizable

**Feature-Based Approach:**
- 5 strategic features:
  - **Center control:** Value of occupying center (position 4)
  - **Corner control:** Value of corner positions (0, 2, 6, 8)
  - **Edge control:** Value of edge positions (1, 3, 5, 7)
  - **Win threat:** Value of moves creating immediate wins
  - **Block threat:** Value of moves blocking opponent wins
- Reduced genome size (9 → 5 parameters)
- Strategic knowledge encoded
- More interpretable weights

**Self-Play Co-Evolution:**
- Traditional fitness: Only plays against fixed opponents (Random, Heuristic)
- Self-play fitness: Also plays against members of evolving population
- Benefits:
  - Population adapts to counter each other's strategies
  - Prevents overfitting to specific opponents
  - Encourages diverse, robust strategies
  - Simulates competitive environment

### Usage

```bash
# Basic feature evolution
uv run tictactoe evolve-features --pop_size 100 --generations 50

# With custom self-play fraction (30% of games against population)
uv run tictactoe evolve-features --pop_size 100 --generations 50 --self_play 0.3

# High self-play for competitive co-evolution
uv run tictactoe evolve-features --pop_size 150 --generations 100 --self_play 0.5 --n_games 15

# Reproducible run
uv run tictactoe evolve-features --seed 42
```

### Features

- ✅ 5-feature genome (reduced from 9 weights)
- ✅ Strategic feature extraction (position + tactical features)
- ✅ Self-play fitness evaluation (configurable 0.0-1.0)
- ✅ Competitive co-evolution
- ✅ CLI integration
- ✅ Comprehensive test coverage

### Comparison: Cell-Based vs Feature-Based

| Aspect | Cell Weights | Feature Weights |
|--------|--------------|-----------------|
| Genome size | 9 parameters | 5 parameters |
| Interpretability | Low (position-specific) | High (strategic meaning) |
| Search space | Larger | Smaller |
| Training opponents | Fixed (Random, Heuristic) | Fixed + Self-play |
| Strategic knowledge | None (learned from scratch) | Encoded in features |
| Generalization | Position-specific | Strategy-based |

---

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

## Timeline Status

| Week | Task                                      | Status |
|------|-------------------------------------------|--------|
| 2    | Game environment + baseline agent         | ✅ Complete |
| 4    | GA setup with DEAP                        | ✅ Complete |
| 6    | Fitness evaluation design + initial experiments | ✅ Complete |
| 8    | Optimization, tuning, visualization       | ✅ Complete |
| 9    | jMetalPy multi-objective experiments      | ✅ Complete |
| 10   | Feature-based evolution with self-play    | ✅ Complete |
| 11   | Report + presentation preparation         | 🔄 In Progress |

