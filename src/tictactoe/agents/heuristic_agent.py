"""Strategic heuristic agent with win/block detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

from tictactoe.agents.base import Agent

if TYPE_CHECKING:
    from tictactoe.board import Board

logger = logging.getLogger(__name__)


class HeuristicAgent(Agent):
    """
    Rule-based agent with strategic priorities.

    Strategy (in order):
    1. Win if possible
    2. Block opponent's win
    3. Take center
    4. Take corner
    5. Take edge
    """

    def __init__(self, player: int) -> None:
        """
        Initialize heuristic agent.

        Args:
            player: 1 (X) or -1 (O)
        """
        super().__init__(player, "Heuristic")
        self.opponent = -player

    @override
    def select_move(self, board: Board) -> int:
        """Select move using strategic heuristics."""
        legal_moves = board.get_legal_moves()

        if not legal_moves:
            msg = "No legal moves available"
            raise ValueError(msg)

        # Priority 1: Win
        if win_move := self._find_winning_move(board, self.player):
            logger.debug("%s taking winning move: %d", self.name, win_move)
            return win_move

        # Priority 2: Block
        if block_move := self._find_winning_move(board, self.opponent):
            logger.debug("%s blocking at: %d", self.name, block_move)
            return block_move

        # Priority 3: Center
        if 4 in legal_moves:
            logger.debug("%s taking center", self.name)
            return 4

        # Priority 4: Corners
        corners = [pos for pos in [0, 2, 6, 8] if pos in legal_moves]
        if corners:
            logger.debug("%s taking corner: %d", self.name, corners[0])
            return corners[0]

        # Priority 5: Edges
        logger.debug("%s taking edge: %d", self.name, legal_moves[0])
        return legal_moves[0]

    def _find_winning_move(self, board: Board, player: int) -> int | None:
        """
        Find move that wins game for player.

        Simulates each legal move and checks for win.

        Args:
            board: Current game board
            player: Player to find winning move for

        Returns:
            Winning move position or None if no winning move exists
        """
        for move in board.get_legal_moves():
            test_board = board.copy()
            test_board.make_move(move, player)

            if test_board.check_win(player):
                return move

        return None
