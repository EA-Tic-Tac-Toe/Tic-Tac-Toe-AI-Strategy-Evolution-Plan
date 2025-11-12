"""Tests for agent implementations."""

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
