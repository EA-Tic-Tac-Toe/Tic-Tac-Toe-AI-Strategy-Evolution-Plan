"""Tests for Board class."""

from tictactoe.board import Board


class TestBoardInitialization:
    """Tests for board initialization and reset."""

    def test_empty_board_creation(self, empty_board: Board) -> None:
        """Test that new board is empty."""
        assert len(empty_board.get_legal_moves()) == 9
        assert not empty_board.is_terminal()
        assert empty_board.get_winner() is None

    def test_board_reset(self, empty_board: Board) -> None:
        """Test board reset functionality."""
        empty_board.make_move(0, 1)
        empty_board.make_move(4, -1)
        empty_board.reset()

        assert len(empty_board.get_legal_moves()) == 9
        assert not empty_board.is_terminal()


class TestMoves:
    """Tests for move execution and validation."""

    def test_legal_move(self, empty_board: Board) -> None:
        """Test making a legal move."""
        result = empty_board.make_move(0, 1)
        assert result is True
        assert 0 not in empty_board.get_legal_moves()
        assert len(empty_board.get_legal_moves()) == 8

    def test_illegal_move_occupied(self, empty_board: Board) -> None:
        """Test making move on occupied position."""
        empty_board.make_move(0, 1)
        result = empty_board.make_move(0, -1)
        assert result is False

    def test_illegal_move_out_of_bounds(self, empty_board: Board) -> None:
        """Test making move outside board."""
        assert empty_board.make_move(-1, 1) is False
        assert empty_board.make_move(9, 1) is False

    def test_move_history(self, empty_board: Board) -> None:
        """Test that move history is tracked."""
        empty_board.make_move(0, 1)
        empty_board.make_move(4, -1)
        history = empty_board.get_move_history()

        assert len(history) == 2
        assert history[0] == (0, 1)
        assert history[1] == (4, -1)


class TestWinDetection:
    """Tests for win condition detection."""

    def test_horizontal_win_top(self, empty_board: Board) -> None:
        """Test top row win."""
        empty_board.make_move(0, 1)
        empty_board.make_move(1, 1)
        empty_board.make_move(2, 1)
        assert empty_board.check_win(1)
        assert empty_board.get_winner() == 1

    def test_horizontal_win_middle(self, empty_board: Board) -> None:
        """Test middle row win."""
        empty_board.make_move(3, -1)
        empty_board.make_move(4, -1)
        empty_board.make_move(5, -1)
        assert empty_board.check_win(-1)
        assert empty_board.get_winner() == -1

    def test_horizontal_win_bottom(self, empty_board: Board) -> None:
        """Test bottom row win."""
        empty_board.make_move(6, 1)
        empty_board.make_move(7, 1)
        empty_board.make_move(8, 1)
        assert empty_board.check_win(1)

    def test_vertical_win_left(self, empty_board: Board) -> None:
        """Test left column win."""
        empty_board.make_move(0, 1)
        empty_board.make_move(3, 1)
        empty_board.make_move(6, 1)
        assert empty_board.check_win(1)

    def test_vertical_win_middle(self, empty_board: Board) -> None:
        """Test middle column win."""
        empty_board.make_move(1, -1)
        empty_board.make_move(4, -1)
        empty_board.make_move(7, -1)
        assert empty_board.check_win(-1)

    def test_vertical_win_right(self, empty_board: Board) -> None:
        """Test right column win."""
        empty_board.make_move(2, 1)
        empty_board.make_move(5, 1)
        empty_board.make_move(8, 1)
        assert empty_board.check_win(1)

    def test_diagonal_win_main(self, empty_board: Board) -> None:
        """Test main diagonal win."""
        empty_board.make_move(0, 1)
        empty_board.make_move(4, 1)
        empty_board.make_move(8, 1)
        assert empty_board.check_win(1)

    def test_diagonal_win_anti(self, empty_board: Board) -> None:
        """Test anti-diagonal win."""
        empty_board.make_move(2, -1)
        empty_board.make_move(4, -1)
        empty_board.make_move(6, -1)
        assert empty_board.check_win(-1)

    def test_no_win(self, empty_board: Board) -> None:
        """Test that incomplete lines don't win."""
        empty_board.make_move(0, 1)
        empty_board.make_move(1, 1)
        assert not empty_board.check_win(1)
        assert empty_board.get_winner() is None


class TestDraw:
    """Tests for draw detection."""

    def test_draw_game(self, empty_board: Board) -> None:
        """Test that full board with no winner is draw."""
        # X X O
        # O O X
        # X O X
        moves = [
            (0, 1),
            (2, -1),
            (1, 1),
            (3, -1),
            (5, 1),
            (4, -1),
            (6, 1),
            (8, 1),
            (7, -1),
        ]
        for pos, player in moves:
            empty_board.make_move(pos, player)

        assert empty_board.is_terminal()
        assert empty_board.get_winner() == 0


class TestBoardCopy:
    """Tests for board copying."""

    def test_copy_independence(self, empty_board: Board) -> None:
        """Test that copied board is independent."""
        empty_board.make_move(0, 1)
        board_copy = empty_board.copy()
        board_copy.make_move(4, -1)

        assert 4 in empty_board.get_legal_moves()
        assert 4 not in board_copy.get_legal_moves()
        assert len(empty_board.get_move_history()) == 1
        assert len(board_copy.get_move_history()) == 2


class TestBoardDisplay:
    """Tests for board string representation."""

    def test_empty_board_display(self, empty_board: Board) -> None:
        """Test display of empty board."""
        display = str(empty_board)
        assert "." in display
        assert "|" in display

    def test_partial_board_display(self, empty_board: Board) -> None:
        """Test display of board with moves."""
        empty_board.make_move(0, 1)
        empty_board.make_move(4, -1)
        display = str(empty_board)
        assert "X" in display
        assert "O" in display
