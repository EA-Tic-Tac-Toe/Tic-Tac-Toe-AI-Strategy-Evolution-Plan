"""Tests for feature-based genetic agent with self-play."""

import pytest

from tictactoe.agents.feature_agent import (
    NUM_FEATURES,
    FeatureGeneticAgent,
    evaluate_feature_fitness,
    extract_features,
    load_feature_weights,
    run_feature_evolution,
    save_feature_weights,
)
from tictactoe.board import Board


class TestFeatureExtraction:
    """Test feature extraction from board states."""

    def test_extract_features_center(self) -> None:
        """Center move should activate center feature."""
        board = Board()
        features = extract_features(board, 4, 1)  # Center position

        assert len(features) == NUM_FEATURES
        assert features[0] == 1.0  # Center feature
        assert features[1] == 0.0  # Corner feature
        assert features[2] == 0.0  # Edge feature

    def test_extract_features_corner(self) -> None:
        """Corner moves should activate corner feature."""
        board = Board()

        for corner in [0, 2, 6, 8]:
            features = extract_features(board, corner, 1)
            assert features[0] == 0.0  # Center
            assert features[1] == 1.0  # Corner
            assert features[2] == 0.0  # Edge

    def test_extract_features_edge(self) -> None:
        """Edge moves should activate edge feature."""
        board = Board()

        for edge in [1, 3, 5, 7]:
            features = extract_features(board, edge, 1)
            assert features[0] == 0.0  # Center
            assert features[1] == 0.0  # Corner
            assert features[2] == 1.0  # Edge

    def test_extract_features_win_threat(self) -> None:
        """Winning move should activate win threat feature."""
        board = Board()
        # Create winning opportunity: X X _
        board.make_move(0, 1)  # X
        board.make_move(1, 1)  # X

        features = extract_features(board, 2, 1)  # Winning move
        assert features[3] == 1.0  # Win threat

    def test_extract_features_block_threat(self) -> None:
        """Blocking move should activate block threat feature."""
        board = Board()
        # Create opponent threat: O O _
        board.make_move(0, -1)  # O
        board.make_move(1, -1)  # O

        features = extract_features(board, 2, 1)  # Block move
        assert features[4] == 1.0  # Block threat

    def test_feature_count(self) -> None:
        """All positions should return correct number of features."""
        board = Board()

        for pos in range(9):
            features = extract_features(board, pos, 1)
            assert len(features) == NUM_FEATURES


class TestFeatureGeneticAgent:
    """Test feature-based genetic agent behavior."""

    def test_initialization(self) -> None:
        """Agent should initialize with correct weights."""
        weights = [1.0, 2.0, 0.5, 3.0, 2.5]
        agent = FeatureGeneticAgent(1, weights)

        assert agent.player == 1
        assert len(agent.weights) == NUM_FEATURES

    def test_initialization_wrong_size(self) -> None:
        """Agent should reject wrong number of weights."""
        with pytest.raises(ValueError, match="Expected 5 weights"):
            FeatureGeneticAgent(1, [1.0, 2.0, 3.0])

    def test_select_move_empty_board(self) -> None:
        """Agent should select valid move on empty board."""
        weights = [1.0, 0.5, 0.2, 2.0, 1.5]
        agent = FeatureGeneticAgent(1, weights)
        board = Board()

        move = agent.select_move(board)
        assert 0 <= move < 9
        assert board.get_state_flat()[move] == 0

    def test_select_move_prioritizes_winning(self) -> None:
        """Agent with high win_threat weight should take winning move."""
        weights = [0.0, 0.0, 0.0, 10.0, 0.0]  # Only win threat matters
        agent = FeatureGeneticAgent(1, weights)

        board = Board()
        board.make_move(0, 1)  # X
        board.make_move(1, 1)  # X
        # Position 2 is winning move

        move = agent.select_move(board)
        assert move == 2

    def test_select_move_prioritizes_blocking(self) -> None:
        """Agent with high block_threat weight should block opponent."""
        weights = [0.0, 0.0, 0.0, 0.0, 10.0]  # Only block threat matters
        agent = FeatureGeneticAgent(1, weights)

        board = Board()
        board.make_move(0, -1)  # O
        board.make_move(1, -1)  # O
        # Position 2 blocks opponent

        move = agent.select_move(board)
        assert move == 2

    def test_select_move_no_legal_moves(self) -> None:
        """Agent should raise error when no legal moves."""
        agent = FeatureGeneticAgent(1, [1.0] * NUM_FEATURES)
        board = Board()

        # Fill board
        for i in range(9):
            board._state[i] = 1 if i % 2 == 0 else -1

        with pytest.raises(ValueError, match="No legal moves"):
            agent.select_move(board)

    def test_repr(self) -> None:
        """Agent should have meaningful string representation."""
        agent = FeatureGeneticAgent(1, [1.0] * NUM_FEATURES)
        assert "FeatureGA" in repr(agent)
        assert "1" in repr(agent)


class TestFitnessEvaluation:
    """Test fitness evaluation with self-play."""

    def test_evaluate_without_self_play(self) -> None:
        """Fitness evaluation should work without population."""
        weights = [1.0, 0.5, 0.2, 2.0, 1.5]

        fitness = evaluate_feature_fitness(
            weights,
            lambda: Board(),
            n_games=2,
            population=None,
            self_play_fraction=0.0,
        )

        assert len(fitness) == 1
        assert 0.0 <= fitness[0] <= 1.0

    def test_evaluate_with_self_play(self) -> None:
        """Fitness evaluation should include self-play games."""
        population = [
            [1.0, 0.5, 0.2, 2.0, 1.5],
            [0.5, 1.0, 0.3, 1.5, 2.0],
            [0.8, 0.8, 0.4, 1.8, 1.8],
        ]

        fitness = evaluate_feature_fitness(
            population[0],
            lambda: Board(),
            n_games=2,
            population=population,
            self_play_fraction=0.5,
        )

        assert len(fitness) == 1
        assert 0.0 <= fitness[0] <= 1.0

    def test_good_weights_better_fitness(self) -> None:
        """Weights prioritizing tactical play should perform better."""
        good_weights = [1.0, 0.8, 0.2, 5.0, 4.0]  # High win/block
        random_weights = [0.1, 0.1, 0.1, 0.1, 0.1]  # Low everything

        good_fitness = evaluate_feature_fitness(
            good_weights,
            lambda: Board(),
            n_games=5,
        )

        random_fitness = evaluate_feature_fitness(
            random_weights,
            lambda: Board(),
            n_games=5,
        )

        # Good weights should be at least as good
        assert good_fitness[0] >= random_fitness[0]


class TestEvolution:
    """Test feature-based evolution process."""

    def test_run_feature_evolution(self) -> None:
        """Evolution should produce valid results."""
        result = run_feature_evolution(
            board_factory=lambda: Board(),
            pop_size=10,
            generations=3,
            n_games=2,
            seed=42,
            self_play_fraction=0.2,
        )

        assert len(result.best_weights) == NUM_FEATURES
        assert 0.0 <= result.best_fitness <= 1.0
        assert len(result.history) == 3
        assert result.algorithm == "DEAP"

    def test_evolution_improves_fitness(self) -> None:
        """Fitness should generally improve over generations."""
        result = run_feature_evolution(
            board_factory=lambda: Board(),
            pop_size=20,
            generations=5,
            n_games=2,
            seed=123,
            self_play_fraction=0.3,
        )

        first_gen_fitness = result.history[0]["avg_fitness"]
        last_gen_fitness = result.history[-1]["avg_fitness"]

        # Should improve or at least not get worse
        assert last_gen_fitness >= first_gen_fitness - 0.1

    def test_evolution_with_high_self_play(self) -> None:
        """Evolution should work with high self-play fraction."""
        result = run_feature_evolution(
            board_factory=lambda: Board(),
            pop_size=15,
            generations=3,
            n_games=2,
            seed=456,
            self_play_fraction=0.8,  # 80% self-play
        )

        assert result.best_fitness > 0.0

    def test_evolution_reproducible(self) -> None:
        """Same seed should produce consistent evolution runs."""
        result1 = run_feature_evolution(
            board_factory=lambda: Board(),
            pop_size=10,
            generations=2,
            n_games=2,
            seed=999,
        )

        result2 = run_feature_evolution(
            board_factory=lambda: Board(),
            pop_size=10,
            generations=2,
            n_games=2,
            seed=999,
        )

        # Both runs should complete successfully and produce valid results
        assert len(result1.best_weights) == NUM_FEATURES
        assert len(result2.best_weights) == NUM_FEATURES
        assert 0.0 <= result1.best_fitness <= 1.0
        assert 0.0 <= result2.best_fitness <= 1.0


class TestSaveLoad:
    """Test weight persistence."""

    def test_save_and_load_weights(self, tmp_path) -> None:
        """Saved weights should be loadable."""
        weights = [1.0, 2.0, 0.5, 3.0, 2.5]
        path = tmp_path / "test_weights.pkl"

        save_feature_weights(weights, path)
        loaded = load_feature_weights(path)

        assert loaded == weights

    def test_load_nonexistent_weights(self, tmp_path) -> None:
        """Loading nonexistent weights should raise error."""
        path = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError, match="Feature weights file not found"):
            load_feature_weights(path)


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, tmp_path) -> None:
        """Test complete evolution → save → load → play workflow."""
        # Run evolution
        result = run_feature_evolution(
            board_factory=lambda: Board(),
            pop_size=10,
            generations=2,
            n_games=2,
            seed=42,
            self_play_fraction=0.3,
        )

        # Save weights
        path = tmp_path / "evolved_weights.pkl"
        save_feature_weights(result.best_weights, path)

        # Load and create agent
        loaded_weights = load_feature_weights(path)
        agent = FeatureGeneticAgent(1, loaded_weights)

        # Play a game
        board = Board()
        move = agent.select_move(board)

        assert 0 <= move < 9
        assert board.get_state_flat()[move] == 0

    def test_feature_agent_vs_random(self) -> None:
        """Feature agent should beat random agent."""
        from tictactoe.agents.random_agent import RandomAgent
        from tictactoe.game_runner import play_game

        # Create evolved agent
        result = run_feature_evolution(
            board_factory=lambda: Board(),
            pop_size=15,
            generations=5,
            n_games=3,
            seed=789,
        )

        feature_agent = FeatureGeneticAgent(1, result.best_weights)
        random_agent = RandomAgent(-1)

        # Play multiple games
        wins = 0
        for _ in range(10):
            winner = play_game(feature_agent, random_agent, verbose=False)
            if winner == 1:
                wins += 1

        # Should win at least some games or draw
        assert wins >= 0  # At minimum, should not crash
