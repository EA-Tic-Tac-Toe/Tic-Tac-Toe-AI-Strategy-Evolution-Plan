"""Abstract base class for Tic-Tac-Toe agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tictactoe.board import Board


class Agent(ABC):
    """
    Abstract base class for all Tic-Tac-Toe agents.

    Uses ABC for interface definition.
    """

    def __init__(self, player: int, name: str | None = None) -> None:
        """
        Initialize agent.

        Args:
            player: 1 (X) or -1 (O)
            name: Optional agent identifier

        Raises:
            ValueError: If player is not 1 or -1
        """
        if player not in (-1, 1):
            msg = f"Player must be 1 or -1, got {player}"
            raise ValueError(msg)

        self.player = player
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select_move(self, board: Board) -> int:
        """
        Choose move given current board state.

        Args:
            board: Current game board

        Returns:
            Position (0-8) to play

        Raises:
            ValueError: If no legal moves available
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset agent state (for stateful agents)."""
        pass

    def __repr__(self) -> str:
        """String representation of agent."""
        player_symbol = "X" if self.player == 1 else "O"
        return f"{self.name}({player_symbol})"
