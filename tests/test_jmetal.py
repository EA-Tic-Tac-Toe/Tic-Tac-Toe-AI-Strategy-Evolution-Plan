"""Tests for jMetalPy multi-objective evolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.jmetal_agent import (
    JMetalAgent,
    MultiObjectiveResult,
    TicTacToeProblem,
    load_pareto_front,
    run_multiobjective_evolution,
    save_pareto_front,
    select_solution_from_pareto,
)
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.board import Board


class TestJMetalAgent:
    """Tests for JMetalAgent class."""

    def test_init_valid_weights(self) -> None:
        """Test agent initialization with valid weights."""
        weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        agent = JMetalAgent(player=1, weights=weights)
        assert agent.player == 1
        assert agent.weights == weights
        assert agent.name == "JMetal"

    def test_init_custom_name(self) -> None:
        """Test agent initialization with custom name."""
        weights = [1.0] * 9
        agent = JMetalAgent(player=-1, weights=weights, name="CustomJMetal")
        assert agent.name == "CustomJMetal"

    def test_init_invalid_weights_length(self) -> None:
        """Test agent initialization fails with wrong number of weights."""
        with pytest.raises(ValueError, match="weights must be length 9"):
            JMetalAgent(player=1, weights=[1.0, 2.0, 3.0])

    def test_select_move_chooses_max_weight(self) -> None:
        """Test that agent selects move with highest weight."""
        # Weights favor position 4 (center)
        weights = [1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0]
        agent = JMetalAgent(player=1, weights=weights)
        board = Board()

        move = agent.select_move(board)
        assert move == 4

    def test_select_move_respects_legal_moves(self) -> None:
        """Test that agent only chooses from legal moves."""
        weights = [10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        agent = JMetalAgent(player=1, weights=weights)
        board = Board()

        # Block position 0
        board.make_move(0, 1)

        # Agent should choose next highest available
        move = agent.select_move(board)
        assert move != 0
        assert move in board.get_legal_moves()

    def test_select_move_no_legal_moves(self) -> None:
        """Test that agent raises error when no legal moves."""
        weights = [1.0] * 9
        agent = JMetalAgent(player=1, weights=weights)
        board = Board()

        # Fill board
        for i in range(9):
            board.make_move(i, 1 if i % 2 == 0 else -1)

        with pytest.raises(ValueError, match="No legal moves available"):
            agent.select_move(board)


class TestTicTacToeProblem:
    """Tests for TicTacToeProblem class."""

    def test_init(self) -> None:
        """Test problem initialization."""
        opponents = [(RandomAgent, -1), (HeuristicAgent, -1)]
        problem = TicTacToeProblem(
            player=1,
            n_games=5,
            opponents=opponents,
            board_factory=Board,
            seed_base=42,
        )

        assert problem.number_of_variables == 9
        assert problem.number_of_objectives() == 2
        assert problem.number_of_constraints() == 0
        assert problem.player == 1
        assert problem.n_games == 5

    def test_create_solution(self) -> None:
        """Test solution creation."""
        problem = TicTacToeProblem(
            player=1,
            n_games=5,
            opponents=[(RandomAgent, -1)],
            board_factory=Board,
        )

        solution = problem.create_solution()
        assert len(solution.variables) == 9
        assert all(-10.0 <= v <= 10.0 for v in solution.variables)

    def test_evaluate_solution(self) -> None:
        """Test solution evaluation."""
        problem = TicTacToeProblem(
            player=1,
            n_games=5,
            opponents=[(RandomAgent, -1)],
            board_factory=Board,
            seed_base=42,
        )

        solution = problem.create_solution()
        solution.variables = [5.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0]
        evaluated = problem.evaluate(solution)

        # Should have two objectives
        assert len(evaluated.objectives) == 2

        # Objective 0: negative fitness (to minimize)
        # Objective 1: complexity (L1-norm)
        fitness = -evaluated.objectives[0]
        complexity = evaluated.objectives[1]

        assert -1.0 <= fitness <= 1.0  # Fitness in [-1, 1] range
        assert complexity > 0  # Complexity should be positive

    def test_evaluate_complexity_calculation(self) -> None:
        """Test that complexity is correctly calculated as L1-norm."""
        problem = TicTacToeProblem(
            player=1,
            n_games=1,
            opponents=[(RandomAgent, -1)],
            board_factory=Board,
        )

        solution = problem.create_solution()
        # Set known weights
        weights = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0]
        solution.variables = weights

        evaluated = problem.evaluate(solution)
        complexity = evaluated.objectives[1]

        expected_complexity = sum(abs(w) for w in weights)
        assert complexity == expected_complexity


class TestMultiObjectiveEvolution:
    """Tests for multi-objective evolution functions."""

    def test_run_evolution_basic(self) -> None:
        """Test basic evolution run."""
        result = run_multiobjective_evolution(
            pop_size=10,
            max_evaluations=100,
            algorithm="NSGA-II",
            n_games=2,
            seed=42,
        )

        assert isinstance(result, MultiObjectiveResult)
        assert len(result.pareto_front) > 0
        assert len(result.pareto_objectives) == len(result.pareto_front)
        assert result.algorithm == "NSGA-II"
        assert result.seed == 42

    def test_run_evolution_nsga3(self) -> None:
        """Test evolution with NSGA-III algorithm."""
        result = run_multiobjective_evolution(
            pop_size=10,
            max_evaluations=100,
            algorithm="NSGA-III",
            n_games=2,
            seed=42,
        )

        assert result.algorithm == "NSGA-III"
        assert len(result.pareto_front) > 0

    def test_run_evolution_invalid_algorithm(self) -> None:
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            run_multiobjective_evolution(
                pop_size=10,
                max_evaluations=100,
                algorithm="INVALID",
                n_games=2,
            )

    def test_objectives_structure(self) -> None:
        """Test that objectives have correct structure."""
        result = run_multiobjective_evolution(
            pop_size=10,
            max_evaluations=100,
            algorithm="NSGA-II",
            n_games=2,
            seed=42,
        )

        # Each objective should be (fitness, complexity)
        for fitness, complexity in result.pareto_objectives:
            assert isinstance(fitness, float)
            assert isinstance(complexity, float)
            assert -1.0 <= fitness <= 1.0  # Fitness in valid range
            assert complexity >= 0  # Complexity non-negative


class TestSolutionSelection:
    """Tests for Pareto front solution selection."""

    @pytest.fixture
    def mock_result(self) -> MultiObjectiveResult:
        """Create mock result for testing."""
        pareto_front = (
            (
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ),  # High fitness, high complexity
            (
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ),  # Low fitness, low complexity
            (0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5),  # Balanced
        )
        pareto_objectives = (
            (0.8, 45.0),  # High fitness, high complexity
            (0.3, 4.5),  # Low fitness, low complexity
            (0.6, 22.5),  # Balanced
        )
        return MultiObjectiveResult(
            pareto_front=pareto_front,
            pareto_objectives=pareto_objectives,
            history=tuple(),
            algorithm="NSGA-II",
            seed=None,
        )

    def test_select_fitness_strategy(self, mock_result: MultiObjectiveResult) -> None:
        """Test fitness-focused selection."""
        weights = select_solution_from_pareto(mock_result, strategy="fitness")
        # Should select solution with best fitness (index 0)
        assert weights == mock_result.pareto_front[0]

    def test_select_simple_strategy(self, mock_result: MultiObjectiveResult) -> None:
        """Test simplicity-focused selection."""
        weights = select_solution_from_pareto(mock_result, strategy="simple")
        # Should select solution with lowest complexity (index 1)
        assert weights == mock_result.pareto_front[1]

    def test_select_balanced_strategy(self, mock_result: MultiObjectiveResult) -> None:
        """Test balanced selection."""
        weights = select_solution_from_pareto(mock_result, strategy="balanced")
        # Should be one of the solutions
        assert weights in mock_result.pareto_front

    def test_select_empty_pareto_front(self) -> None:
        """Test that empty Pareto front raises error."""
        empty_result = MultiObjectiveResult(
            pareto_front=tuple(),
            pareto_objectives=tuple(),
            history=tuple(),
            algorithm="NSGA-II",
            seed=None,
        )

        with pytest.raises(ValueError, match="Empty Pareto front"):
            select_solution_from_pareto(empty_result, strategy="balanced")


class TestSaveLoadParetoFront:
    """Tests for saving and loading Pareto fronts."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading Pareto front."""
        # Create mock result
        result = MultiObjectiveResult(
            pareto_front=(
                (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
                (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            ),
            pareto_objectives=((0.8, 45.0), (0.3, 4.5)),
            history=tuple(),
            algorithm="NSGA-II",
            seed=42,
        )

        # Save
        save_path = tmp_path / "pareto_front.pkl"
        save_pareto_front(result, save_path)
        assert save_path.exists()

        # Load
        loaded_result = load_pareto_front(save_path)
        assert loaded_result.pareto_front == result.pareto_front
        assert loaded_result.pareto_objectives == result.pareto_objectives
        assert loaded_result.algorithm == result.algorithm
        assert loaded_result.seed == result.seed

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        result = MultiObjectiveResult(
            pareto_front=(tuple([1.0] * 9),),
            pareto_objectives=((0.5, 9.0),),
            history=tuple(),
            algorithm="NSGA-II",
            seed=None,
        )

        save_path = tmp_path / "subdir" / "pareto_front.pkl"
        save_pareto_front(result, save_path)
        assert save_path.exists()


class TestIntegration:
    """Integration tests for jMetalPy workflow."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: evolve -> save -> load -> select -> play."""
        # 1. Run evolution
        result = run_multiobjective_evolution(
            pop_size=10,
            max_evaluations=100,
            algorithm="NSGA-II",
            n_games=2,
            seed=42,
        )

        assert len(result.pareto_front) > 0

        # 2. Save result
        save_path = tmp_path / "pareto_front.pkl"
        save_pareto_front(result, save_path)

        # 3. Load result
        loaded_result = load_pareto_front(save_path)
        assert len(loaded_result.pareto_front) == len(result.pareto_front)

        # 4. Select solution
        weights = select_solution_from_pareto(loaded_result, strategy="balanced")
        assert len(weights) == 9

        # 5. Create agent and play
        agent = JMetalAgent(player=1, weights=weights)
        board = Board()
        move = agent.select_move(board)
        assert move in range(9)
        assert move in board.get_legal_moves()
