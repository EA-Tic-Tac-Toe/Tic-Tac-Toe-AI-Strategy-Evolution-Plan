"""Tic-Tac-Toe board implementation with modern Python features."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Self

import numpy as np
import numpy.typing as npt


class Player(IntEnum):
    """Player representation using IntEnum for type safety."""

    EMPTY = 0
    X = 1
    O = -1  # noqa: E741


@dataclass
class Board:
    """
    Tic-Tac-Toe game board with state management.

    Uses numpy array for efficient operations.
    Board positions indexed 0-8:
        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
    """

    _state: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.zeros(9, dtype=np.int8)
    )
    _move_history: list[tuple[int, int]] = field(default_factory=list)
    _cached_legal_moves: list[int] | None = field(default=None, repr=False)

    def reset(self) -> Self:
        """Reset board to initial empty state."""
        self._state.fill(0)
        self._move_history.clear()
        self._cached_legal_moves = None
        return self  # Method chaining with Self type

    def get_state(self) -> npt.NDArray[np.int8]:
        """Return copy of current board state as 3x3 grid."""
        return self._state.reshape(3, 3).copy()

    def get_state_flat(self) -> npt.NDArray[np.int8]:
        """Return copy of current board state as flat 1D array."""
        return self._state.copy()

    def get_legal_moves(self) -> list[int]:
        """Return list of legal move positions (0-8)."""
        if self._cached_legal_moves is None:
            self._cached_legal_moves = np.where(self._state == 0)[0].tolist()
        assert self._cached_legal_moves is not None  # For type checker
        return self._cached_legal_moves

    def make_move(self, position: int, player: int) -> bool:
        """
        Execute move for player at position.

        Args:
            position: Board position (0-8)
            player: Player identifier (1 for X, -1 for O)

        Returns:
            True if move successful, False if illegal.
        """
        if not (0 <= position <= 8) or self._state[position] != 0:
            return False

        self._state[position] = player
        self._move_history.append((position, player))
        self._cached_legal_moves = None  # Invalidate cache
        return True

    def check_win(self, player: int) -> bool:
        """
        Check if specific player has won using vectorized operations.

        Uses pattern matching for elegant win detection logic.
        """
        grid = self._state.reshape(3, 3)

        # Check rows, columns, and diagonals (convert numpy bool_ to Python bool)
        rows_check = bool(np.any(grid.sum(axis=1) == player * 3))
        cols_check = bool(np.any(grid.sum(axis=0) == player * 3))
        main_diag_check = bool(np.trace(grid) == player * 3)
        anti_diag_check = bool(np.trace(np.fliplr(grid)) == player * 3)

        # Use pattern matching for result
        match (rows_check, cols_check, main_diag_check, anti_diag_check):
            case (True, _, _, _) | (_, True, _, _) | (_, _, True, _) | (_, _, _, True):
                return True
            case _:
                return False

    def get_winner(self) -> int | None:
        """
        Return game result.

        Returns:
            1: X wins
            -1: O wins
            0: draw
            None: game ongoing
        """
        if self.check_win(1):
            return 1
        if self.check_win(-1):
            return -1
        if len(self.get_legal_moves()) == 0:
            return 0  # Draw
        return None  # Ongoing

    def is_terminal(self) -> bool:
        """Check if game has ended."""
        return self.get_winner() is not None

    def copy(self) -> Self:
        """Return deep copy of board."""
        new_board = self.__class__()
        new_board._state = self._state.copy()
        new_board._move_history = self._move_history.copy()
        return new_board

    def get_move_history(self) -> list[tuple[int, int]]:
        """Return copy of move history."""
        return self._move_history.copy()

    def __str__(self) -> str:
        """Human-readable board display."""
        symbols = {-1: "O", 0: ".", 1: "X"}
        grid = self._state.reshape(3, 3)
        lines = []
        for i, row in enumerate(grid):
            line = " | ".join(symbols[val] for val in row)
            lines.append(line)
            if i < 2:
                lines.append("---------")
        return "\n".join(lines)
