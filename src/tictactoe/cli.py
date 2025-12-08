"""Command-line interface for Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

import yaml

from tictactoe.agents.base import Agent
from tictactoe.agents.genetic_agent import (
    GeneticAgent,
    evolve_population,
    load_weights,
    save_weights,
)
from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.analysis.evaluator import (
    aggregate_outcomes_by_opponent,
    load_evaluation,
)
from tictactoe.analysis.plots import (
    plot_fitness_curve,
    plot_outcome_bars,
    plot_tuning_comparison,
    plot_weight_heatmap,
)
from tictactoe.analysis.tuning import (
    config_from_mapping,
    export_tuning_results,
    grid_search,
)
from tictactoe.board import Board
from tictactoe.game_runner import evaluate_matchup, play_game


def create_agent(agent_type: str, player: int) -> Agent:
    """
    Factory function to create agents.

    Args:
        agent_type: Type of agent ('random' or 'heuristic')
        player: Player identifier (1 or -1)

    Returns:
        Agent instance

    Raises:
        ValueError: If agent_type is unknown
    """
    match agent_type.lower():
        case "random":
            return RandomAgent(player)
        case "heuristic":
            return HeuristicAgent(player)
        case "genetic":
            weights = load_weights()
            return GeneticAgent(player, weights)
        case _:
            msg = f"Unknown agent type: {agent_type}"
            raise ValueError(msg)


def play_interactive(opponent_type: str, human_player: int) -> None:
    """
    Human vs AI interactive game.

    Args:
        opponent_type: Type of AI opponent
        human_player: Player identifier for human (1 or -1)
    """
    ai_player = -human_player
    ai_agent = create_agent(opponent_type, ai_player)

    board = Board()
    current_player = 1  # X always starts

    print("\n" + "=" * 40)
    print("🎮  TIC-TAC-TOE  🎮")
    print("=" * 40)
    print(f"\nYou are {'X' if human_player == 1 else 'O'}")
    print(f"AI is {ai_agent}\n")
    print("Positions:")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 \n")

    while not board.is_terminal():
        print(board)
        print()

        if current_player == human_player:
            # Human turn
            while True:
                try:
                    move = int(input("Your move (0-8): "))
                    if board.make_move(move, current_player):
                        break
                    print("❌ Illegal move! Try again.")
                except ValueError:
                    print("❌ Please enter a number between 0-8.")
                except KeyboardInterrupt:
                    print("\n👋 Game cancelled.")
                    return
        else:
            # AI turn
            move = ai_agent.select_move(board)
            board.make_move(move, current_player)
            print(f"🤖 AI plays: {move}")

        current_player = -current_player

    print(board)
    print()

    match board.get_winner():
        case w if w == human_player:
            print("🎉 You win! 🎉")
        case w if w == ai_player:
            print("😔 AI wins!")
        case 0:
            print("🤝 It's a draw!")


def demo_game(agent1_type: str, agent2_type: str) -> None:
    """
    Watch two AIs play.

    Args:
        agent1_type: Type of first agent
        agent2_type: Type of second agent
    """
    agent1 = create_agent(agent1_type, 1)
    agent2 = create_agent(agent2_type, -1)

    play_game(agent1, agent2, verbose=True)


def evaluate_agents_cli(
    agent1_type: str,
    agent2_type: str,
    num_games: int,
) -> None:
    """
    Evaluate two agents over multiple games.

    Args:
        agent1_type: Type of first agent
        agent2_type: Type of second agent
        num_games: Number of games to play
    """
    agent1 = create_agent(agent1_type, 1)
    agent2 = create_agent(agent2_type, -1)

    print(f"\n⚔️  Evaluating {agent1} vs {agent2} over {num_games} games...\n")

    stats = evaluate_matchup(agent1, agent2, num_games)

    print("=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(
        f"{agent1.name} wins:  {stats.agent1_wins:>4} ({stats.agent1_win_rate:>5.1%})"
    )
    print(
        f"{agent2.name} wins:  {stats.agent2_wins:>4} ({stats.agent2_win_rate:>5.1%})"
    )
    print(f"Draws:         {stats.draws:>4} ({stats.draw_rate:>5.1%})")
    print(f"Avg game length: {stats.avg_game_length:.1f} moves")
    print("=" * 50 + "\n")


def visualize_experiment_cli(experiment: str, output_dir: str) -> None:
    """Generate plots for a saved experiment."""
    experiment_path = Path(experiment)
    if not experiment_path.exists():
        msg = f"Experiment file not found: {experiment}"
        raise FileNotFoundError(msg)

    series = load_evaluation(experiment_path)
    best_run = max(series.runs, key=lambda run: run.best_fitness)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aggregate_outcomes = aggregate_outcomes_by_opponent(series)
    fitness_paths = plot_fitness_curve(
        best_run.evolution.history,
        output_path / f"run{best_run.run_id}_fitness",
    )
    outcome_paths = plot_outcome_bars(
        aggregate_outcomes,
        output_path / "outcomes",
    )
    heatmap_paths = plot_weight_heatmap(
        best_run.evolution.best_weights,
        output_path / f"run{best_run.run_id}_heatmap",
    )

    print("Generated plots:")
    for label, paths in [
        ("Fitness", fitness_paths),
        ("Outcomes", outcome_paths),
        ("Heatmap", heatmap_paths),
    ]:
        print(f"  {label}: {paths['png']} / {paths['svg']}")


def tune_ga_cli(config_path: str, output_prefix: str | None) -> None:
    """Run GA tuning based on YAML configuration."""
    yaml_path = Path(config_path)
    if not yaml_path.exists():
        msg = f"Tuning config not found: {config_path}"
        raise FileNotFoundError(msg)

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = "Tuning configuration must be a mapping"
        raise ValueError(msg)

    tuning_config = config_from_mapping(data)
    records = grid_search(tuning_config)
    best_record = max(records, key=lambda record: record.avg_best_fitness)

    base = (
        Path(output_prefix)
        if output_prefix
        else Path("results") / "tuning" / yaml_path.stem
    )

    artifacts = export_tuning_results(records, base)
    comparison_paths = plot_tuning_comparison(
        records,
        base.with_name(base.name + "_comparison"),
    )
    print("Best configuration:")
    print(f"  params = {best_record.params.to_dict()}")
    print(f"  avg fitness = {best_record.avg_best_fitness:.3f}")
    print(f"Tuning artifacts saved to: {artifacts['json']} / {artifacts['csv']}")
    print(f"Comparison plot: {comparison_paths['png']} / {comparison_paths['svg']}")


def heatmap_cli(weights_path: str, output_base: str) -> None:
    """Render a heatmap for a saved weights file."""
    weights_file = Path(weights_path)
    weights = load_weights(weights_file)
    target = Path(output_base)
    paths = plot_weight_heatmap(weights, target)
    print(f"Heatmap saved to: {paths['png']} / {paths['svg']}")


def main() -> NoReturn:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="🎮 Tic-Tac-Toe Game & AI Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Play interactive
    play_parser = subparsers.add_parser("play", help="Play against AI")
    play_parser.add_argument(
        "--opponent",
        choices=["random", "heuristic", "genetic"],
        default="heuristic",
        help="AI opponent type",
    )
    play_parser.add_argument(
        "--player",
        choices=["X", "O"],
        default="X",
        help="Play as X or O",
    )

    # Evaluate agents
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate two agents")
    eval_parser.add_argument("--agent1", required=True, help="First agent type")
    eval_parser.add_argument("--agent2", required=True, help="Second agent type")
    eval_parser.add_argument("--games", type=int, default=100, help="Number of games")

    # Demo game
    demo_parser = subparsers.add_parser("demo", help="Watch AI vs AI")
    demo_parser.add_argument("--agent1", default="heuristic", help="First agent")
    demo_parser.add_argument("--agent2", default="random", help="Second agent")

    genetic_parser = subparsers.add_parser(
        "evolve",
        help="Generate weights for genetic agent",
    )
    genetic_parser.add_argument(
        "--pop_size",
        default="100",
        type=int,
        help="Population size",
    )
    genetic_parser.add_argument(
        "--generations",
        default="40",
        type=int,
        help="Number of generations",
    )
    genetic_parser.add_argument(
        "--cx_pb",
        default="0.5",
        type=float,
        help="Crossover probability",
    )
    genetic_parser.add_argument(
        "--mut_pb",
        default="0.2",
        type=float,
        help="Mutation probability",
    )
    genetic_parser.add_argument(
        "--n_games",
        default="4",
        type=int,
        help="Number of games",
    )
    genetic_parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional random seed",
    )

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate plots from a saved experiment JSON",
    )
    visualize_parser.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment JSON (exported via analysis.evaluator)",
    )
    visualize_parser.add_argument(
        "--output",
        default="results/plots",
        help="Directory for generated figures",
    )

    tune_parser = subparsers.add_parser(
        "tune",
        help="Run GA hyperparameter grid search",
        description="Execute GA tuning using a YAML configuration file.",
    )
    tune_parser.add_argument("--config", required=True, help="Path to tuning YAML file")
    tune_parser.add_argument(
        "--output",
        help="Optional output prefix (defaults to results/tuning/<config name>)",
    )

    heatmap_parser = subparsers.add_parser(
        "heatmap",
        help="Render heatmap for saved genetic weights",
    )
    heatmap_parser.add_argument(
        "--weights",
        required=True,
        help="Path to weights file (pickle)",
    )
    heatmap_parser.add_argument(
        "--output",
        default="results/plots/weights_heatmap",
        help="Output path prefix for heatmap images",
    )

    args = parser.parse_args()

    match args.command:
        case "play":
            human_player = 1 if args.player == "X" else -1
            play_interactive(args.opponent, human_player)
        case "evaluate":
            evaluate_agents_cli(args.agent1, args.agent2, args.games)
        case "demo":
            demo_game(args.agent1, args.agent2)
        case "evolve":
            weights = evolve_population(
                board_factory=lambda: Board(),
                pop_size=args.pop_size,
                generations=args.generations,
                cx_pb=args.cx_pb,
                mut_pb=args.mut_pb,
                n_games=args.n_games,
                seed=args.seed,
            )
            save_weights(weights)
        case "visualize":
            visualize_experiment_cli(args.experiment, args.output)
        case "tune":
            tune_ga_cli(args.config, args.output)
        case "heatmap":
            heatmap_cli(args.weights, args.output)

    sys.exit(0)


if __name__ == "__main__":
    main()
