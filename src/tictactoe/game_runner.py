"""Game orchestration and evaluation framework."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tictactoe.agents.base import Agent

from tictactoe.board import Board

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameResult:
    """Immutable game result data."""

    winner: int | None  # 1, -1, 0 (draw), None (error)
    num_moves: int
    game_history: tuple[tuple[int, int], ...]  # Immutable history

    @property
    def is_draw(self) -> bool:
        """Check if game was a draw."""
        return self.winner == 0

    @property
    def winner_name(self) -> str:
        """Get human-readable winner name."""
        match self.winner:
            case 1:
                return "X"
            case -1:
                return "O"
            case 0:
                return "Draw"
            case _:
                return "Error"


def play_game(
    agent1: Agent,
    agent2: Agent,
    verbose: bool = False,
    store_history: bool = True,
) -> GameResult:
    """
    Play single game between two agents.

    Args:
        agent1: Player 1 (X, goes first)
        agent2: Player 2 (O, goes second)
        verbose: Print move-by-move updates
        store_history: Store full game history

    Returns:
        GameResult with winner and statistics
    """
    board = Board()
    agents = {1: agent1, -1: agent2}
    current_player = 1

    if verbose:
        print(f"\n🎮 Game: {agent1} vs {agent2}\n")
        print(board)

    while not board.is_terminal():
        agent = agents[current_player]

        try:
            move = agent.select_move(board)
            board.make_move(move, current_player)

            if verbose:
                print(f"\n{agent} plays position {move}")
                print(board)

            current_player = -current_player

        except Exception as e:
            logger.exception("Error in game: %s", e)
            return GameResult(
                winner=None,
                num_moves=len(board._move_history),
                game_history=(tuple(board._move_history) if store_history else ()),
            )

    winner = board.get_winner()

    if verbose:
        match winner:
            case 1:
                result_msg = "🎉 X (Agent1) wins!"
            case -1:
                result_msg = "🎉 O (Agent2) wins!"
            case 0:
                result_msg = "🤝 Draw!"
            case _:
                result_msg = "❌ Error!"
        print(f"\n{result_msg}\n")

    return GameResult(
        winner=winner,
        num_moves=len(board._move_history),
        game_history=tuple(board._move_history) if store_history else (),
    )


@dataclass
class MatchupStats:
    """Statistics from multiple games."""

    agent1_wins: int
    agent2_wins: int
    draws: int
    total_games: int
    avg_game_length: float

    @property
    def agent1_win_rate(self) -> float:
        """Calculate agent1's win rate."""
        return self.agent1_wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def agent2_win_rate(self) -> float:
        """Calculate agent2's win rate."""
        return self.agent2_wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Calculate draw rate."""
        return self.draws / self.total_games if self.total_games > 0 else 0.0


def evaluate_matchup(
    agent1: Agent,
    agent2: Agent,
    num_games: int = 100,
    alternate_first: bool = True,
) -> MatchupStats:
    """
    Evaluate agents over multiple games.

    Args:
        agent1: First agent to evaluate
        agent2: Second agent to evaluate
        num_games: Number of games to play
        alternate_first: Switch starting player each game

    Returns:
        MatchupStats with aggregated results
    """
    agent1_wins = agent2_wins = draws = 0
    total_moves = 0

    for game_idx in range(num_games):
        # Alternate who goes first
        if alternate_first and game_idx % 2 == 1:
            result = play_game(agent2, agent1, verbose=False, store_history=False)
            # Flip winner perspective
            winner = -result.winner if result.winner else result.winner
        else:
            result = play_game(agent1, agent2, verbose=False, store_history=False)
            winner = result.winner

        match winner:
            case 1:
                agent1_wins += 1
            case -1:
                agent2_wins += 1
            case 0:
                draws += 1

        total_moves += result.num_moves

    return MatchupStats(
        agent1_wins=agent1_wins,
        agent2_wins=agent2_wins,
        draws=draws,
        total_games=num_games,
        avg_game_length=total_moves / num_games,
    )
