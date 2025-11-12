"""Random baseline agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from tictactoe.agents.base import Agent

if TYPE_CHECKING:
    from tictactoe.board import Board


class RandomAgent(Agent):
    """Agent that selects moves uniformly at random."""

    def __init__(
        self,
        player: int,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        """
        Initialize random agent.

        Args:
            player: 1 (X) or -1 (O)
            seed: Random seed for reproducibility
            name: Optional custom name
        """
        super().__init__(player, name or "Random")
        self.rng = np.random.default_rng(seed)

    @override
    def select_move(self, board: Board) -> int:
        """Select random move from legal moves."""
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            msg = "No legal moves available"
            raise ValueError(msg)

        return int(self.rng.choice(legal_moves))
