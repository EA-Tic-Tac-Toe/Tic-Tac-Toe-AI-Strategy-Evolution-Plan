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
├── src/
│   └── tictactoe/
│       ├── __init__.py
│       ├── board.py        # Game board implementation
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract agent class
│       │   ├── genetic_agent.py    # Genetic agent using DEAP (Distributed Evolutionary Algorithms in Python)
│       │   ├── random_agent.py     # Random baseline
│       │   └── heuristic_agent.py  # Strategic agent
│       ├── weights/
│       │   └──  best                # pickle file with weights for genetic agent
│       ├── game_runner.py  # Game orchestration & evaluation
│       └── cli.py          # Command-line interface
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── test_board.py
│   ├── test_agents.py
│   └── test_game_runner.py
└── notebooks/              # Jupyter notebooks for experiments
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

*Perfect play leads to draws!*


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

