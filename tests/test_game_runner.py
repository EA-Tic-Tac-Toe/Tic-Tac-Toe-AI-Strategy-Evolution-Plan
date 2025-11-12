"""Tests for game runner functionality."""

from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.game_runner import GameResult, MatchupStats, evaluate_matchup, play_game


class TestPlayGame:
    """Tests for play_game function."""

    def test_play_game_completes(self) -> None:
        """Test that game plays to completion."""
        agent1 = RandomAgent(1, seed=42)
        agent2 = RandomAgent(-1, seed=43)

        result = play_game(agent1, agent2)

        assert isinstance(result, GameResult)
        assert result.winner in [1, -1, 0]
        assert 0 < result.num_moves <= 9

    def test_play_game_stores_history(self) -> None:
        """Test that game history is stored when requested."""
        agent1 = RandomAgent(1, seed=42)
        agent2 = RandomAgent(-1, seed=43)

        result = play_game(agent1, agent2, store_history=True)

        assert len(result.game_history) == result.num_moves
        assert all(isinstance(move, tuple) for move in result.game_history)

    def test_play_game_no_history(self) -> None:
        """Test that history can be disabled."""
        agent1 = RandomAgent(1, seed=42)
        agent2 = RandomAgent(-1, seed=43)

        result = play_game(agent1, agent2, store_history=False)

        assert len(result.game_history) == 0

    def test_game_result_properties(self) -> None:
        """Test GameResult properties."""
        result = GameResult(winner=0, num_moves=9, game_history=())

        assert result.is_draw
        assert result.winner_name == "Draw"

        result_x = GameResult(winner=1, num_moves=5, game_history=())
        assert result_x.winner_name == "X"

        result_o = GameResult(winner=-1, num_moves=6, game_history=())
        assert result_o.winner_name == "O"

    def test_heuristic_beats_random_consistently(self) -> None:
        """Test that heuristic agent dominates random agent."""
        agent1 = HeuristicAgent(1)
        agent2 = RandomAgent(-1, seed=42)

        wins = 0
        draws = 0
        losses = 0

        for _ in range(10):
            result = play_game(agent1, agent2, store_history=False)
            if result.winner == 1:
                wins += 1
            elif result.winner == -1:
                losses += 1
            else:
                draws += 1

        # Heuristic should never lose to random
        assert losses == 0
        # Heuristic should win most games
        assert wins >= draws


class TestEvaluateMatchup:
    """Tests for evaluate_matchup function."""

    def test_evaluate_matchup_basic(self) -> None:
        """Test basic matchup evaluation."""
        agent1 = RandomAgent(1, seed=42)
        agent2 = RandomAgent(-1, seed=43)

        stats = evaluate_matchup(agent1, agent2, num_games=10)

        assert isinstance(stats, MatchupStats)
        assert stats.total_games == 10
        assert stats.agent1_wins + stats.agent2_wins + stats.draws == 10
        assert 0 <= stats.avg_game_length <= 9

    def test_matchup_stats_properties(self) -> None:
        """Test MatchupStats property calculations."""
        stats = MatchupStats(
            agent1_wins=7,
            agent2_wins=2,
            draws=1,
            total_games=10,
            avg_game_length=6.5,
        )

        assert stats.agent1_win_rate == 0.7
        assert stats.agent2_win_rate == 0.2
        assert stats.draw_rate == 0.1

    def test_evaluate_matchup_with_alternating(self) -> None:
        """Test that alternating first player works."""
        agent1 = HeuristicAgent(1)
        agent2 = RandomAgent(-1, seed=42)

        stats = evaluate_matchup(agent1, agent2, num_games=20, alternate_first=True)

        # Heuristic should dominate
        assert stats.agent1_win_rate > 0.7
        assert stats.agent2_wins == 0  # Random shouldn't beat heuristic

    def test_evaluate_matchup_without_alternating(self) -> None:
        """Test evaluation without alternating first player."""
        agent1 = RandomAgent(1, seed=42)
        agent2 = RandomAgent(-1, seed=43)

        stats = evaluate_matchup(agent1, agent2, num_games=10, alternate_first=False)

        assert stats.total_games == 10

    def test_heuristic_vs_heuristic_all_draws(self) -> None:
        """Test that two heuristic agents always draw."""
        agent1 = HeuristicAgent(1)
        agent2 = HeuristicAgent(-1)

        stats = evaluate_matchup(agent1, agent2, num_games=10)

        # Perfect play should lead to draws
        assert stats.draws == 10
        assert stats.agent1_wins == 0
        assert stats.agent2_wins == 0


class TestEdgeCases:
    """Tests for edge cases in game runner."""

    def test_empty_game_stats(self) -> None:
        """Test stats with zero games."""
        stats = MatchupStats(
            agent1_wins=0,
            agent2_wins=0,
            draws=0,
            total_games=0,
            avg_game_length=0.0,
        )

        assert stats.agent1_win_rate == 0.0
        assert stats.agent2_win_rate == 0.0
        assert stats.draw_rate == 0.0
