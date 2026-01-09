"""Tests for agent implementations."""

from tictactoe.agents.genetic_agent import (
    GeneticAgent,
    evolve_population,
    load_weights,
    save_weights,
)
from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.board import Board


class TestRandomAgent:
    """Tests for RandomAgent."""

    def test_random_agent_creation(self, random_agent: RandomAgent) -> None:
        """Test random agent initialization."""
        assert random_agent.player == 1
        assert random_agent.name == "Random"

    def test_random_agent_selects_valid_move(
        self, random_agent: RandomAgent, empty_board: Board
    ) -> None:
        """Test that random agent selects legal moves."""
        move = random_agent.select_move(empty_board)
        assert 0 <= move <= 8
        assert move in empty_board.get_legal_moves()

    def test_random_agent_with_seed_reproducible(self, empty_board: Board) -> None:
        """Test that seeded random agent is reproducible."""
        agent1 = RandomAgent(1, seed=123)
        agent2 = RandomAgent(1, seed=123)

        move1 = agent1.select_move(empty_board)
        move2 = agent2.select_move(empty_board)

        assert move1 == move2

    def test_random_agent_raises_on_no_moves(self, random_agent: RandomAgent) -> None:
        """Test that random agent raises error when no moves available."""
        board = Board()
        # Fill the board
        for i in range(9):
            board.make_move(i, 1 if i % 2 == 0 else -1)

        try:
            random_agent.select_move(board)
            msg = "Should have raised ValueError"
            raise AssertionError(msg)
        except ValueError as e:
            assert "No legal moves" in str(e)


class TestHeuristicAgent:
    """Tests for HeuristicAgent."""

    def test_heuristic_agent_creation(self, heuristic_agent: HeuristicAgent) -> None:
        """Test heuristic agent initialization."""
        assert heuristic_agent.player == 1
        assert heuristic_agent.name == "Heuristic"
        assert heuristic_agent.opponent == -1

    def test_heuristic_takes_winning_move(self) -> None:
        """Test that heuristic agent takes winning move when available."""
        agent = HeuristicAgent(1)
        board = Board()

        # Set up: X X _
        board.make_move(0, 1)
        board.make_move(1, 1)

        move = agent.select_move(board)
        assert move == 2  # Should complete the row

    def test_heuristic_blocks_opponent_win(self) -> None:
        """Test that heuristic agent blocks opponent's winning move."""
        agent = HeuristicAgent(1)
        board = Board()

        # Set up: O O _
        board.make_move(0, -1)
        board.make_move(1, -1)

        move = agent.select_move(board)
        assert move == 2  # Should block opponent

    def test_heuristic_prefers_center(self) -> None:
        """Test that heuristic agent prefers center when no immediate threats."""
        agent = HeuristicAgent(1)
        board = Board()

        move = agent.select_move(board)
        assert move == 4  # Center position

    def test_heuristic_prefers_corners_over_edges(self) -> None:
        """Test that heuristic agent prefers corners when center taken."""
        agent = HeuristicAgent(1)
        board = Board()

        # Take center
        board.make_move(4, -1)

        move = agent.select_move(board)
        assert move in [0, 2, 6, 8]  # Should be a corner

    def test_heuristic_win_priority_over_block(self) -> None:
        """Test that winning takes priority over blocking."""
        agent = HeuristicAgent(1)
        board = Board()

        # Set up scenario where agent can win or block
        # X X _
        # O O _
        board.make_move(0, 1)
        board.make_move(1, 1)
        board.make_move(3, -1)
        board.make_move(4, -1)

        move = agent.select_move(board)
        assert move == 2  # Should win, not block at position 5

    def test_heuristic_vertical_win_detection(self) -> None:
        """Test that heuristic detects vertical winning moves."""
        agent = HeuristicAgent(1)
        board = Board()

        # Set up vertical: X _ _ / X _ _ / _ _ _
        board.make_move(0, 1)
        board.make_move(3, 1)

        move = agent.select_move(board)
        assert move == 6  # Should complete column

    def test_heuristic_diagonal_win_detection(self) -> None:
        """Test that heuristic detects diagonal winning moves."""
        agent = HeuristicAgent(1)
        board = Board()

        # Set up diagonal: X _ _ / _ X _ / _ _ _
        board.make_move(0, 1)
        board.make_move(4, 1)

        move = agent.select_move(board)
        assert move == 8  # Should complete diagonal


class TestAgentInteraction:
    """Tests for agent interactions in games."""

    def test_agents_play_complete_game(self) -> None:
        """Test that agents can play a complete game."""
        board = Board()
        agent1 = RandomAgent(1, seed=42)
        agent2 = HeuristicAgent(-1)

        current_player = 1
        move_count = 0
        max_moves = 9

        while not board.is_terminal() and move_count < max_moves:
            agent = agent1 if current_player == 1 else agent2
            move = agent.select_move(board)
            board.make_move(move, current_player)
            current_player = -current_player
            move_count += 1

        # Game should end
        assert board.is_terminal() or move_count == max_moves


"""Tests for GeneticAgent implementation."""


class TestGeneticAgent:
    """Tests for GeneticAgent."""

    def test_genetic_agent_creation(self) -> None:
        """Test genetic agent initialization."""
        weights = [0.1] * 9
        agent = GeneticAgent(player=1, weights=weights)

        assert agent.player == 1
        assert agent.name == "Genetic"
        assert len(agent.weights) == 9

    def test_genetic_agent_requires_9_weights(self) -> None:
        """Test that genetic agent validates weight length."""
        try:
            GeneticAgent(1, [0.0] * 8)  # wrong shape
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "length 9" in str(e)

    def test_genetic_agent_selects_valid_move(self, empty_board: Board) -> None:
        """Test that genetic agent selects legal moves."""
        weights = [0.5] * 9
        agent = GeneticAgent(1, weights)

        move = agent.select_move(empty_board)

        assert move in empty_board.get_legal_moves()
        assert 0 <= move <= 8

    def test_genetic_agent_raises_on_no_moves(self) -> None:
        """Test that genetic agent raises error when no legal moves exist."""
        weights = [0.3] * 9
        agent = GeneticAgent(1, weights)
        board = Board()

        # fill the board
        for i in range(9):
            board.make_move(i, 1 if i % 2 == 0 else -1)

        try:
            agent.select_move(board)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "No legal moves" in str(e)

    def test_genetic_agent_takes_max_weight(self, empty_board: Board) -> None:
        """Test that genetic agent selects the move with highest weight."""
        # give cell 7 the largest weight
        weights = [0.1] * 9
        weights[7] = 2.0

        agent = GeneticAgent(1, weights)
        move = agent.select_move(empty_board)

        assert move == 7


class TestGeneticPersistence:
    """Tests for saving and loading genetic weights."""

    def test_save_and_load_weights(self, tmp_path) -> None:
        """Test that weights can be saved and loaded."""
        weights = [0.1 * i for i in range(9)]

        # run inside temp directory
        (tmp_path / "src/tictactoe/weights").mkdir(parents=True)
        save_path = tmp_path / "src/tictactoe/weights/best"

        # redirect save location
        # monkeypatching not needed — change cwd
        import os

        cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            save_weights(weights)
            assert save_path.exists()

            loaded = load_weights()

            assert isinstance(loaded, list)
            assert len(loaded) == 9
            assert loaded == weights
        finally:
            os.chdir(cwd)


class TestGeneticEvolution:
    """Tests for population evolution."""

    def test_evolve_population_smoke_test(self) -> None:
        """Small evolution smoke test to ensure GA pipeline runs."""
        best = evolve_population(
            board_factory=lambda: Board(),
            pop_size=6,
            generations=2,
            cx_pb=0.5,
            mut_pb=0.2,
            n_games=1,
            seed=123,
        )

        assert isinstance(best, list)
        assert len(best) == 9
        assert all(isinstance(w, (float, int)) for w in best)
