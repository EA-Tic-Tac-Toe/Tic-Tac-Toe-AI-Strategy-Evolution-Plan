"""Command-line interface for Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from tictactoe.agents.base import Agent
from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
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
        choices=["random", "heuristic"],
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

    args = parser.parse_args()

    match args.command:
        case "play":
            human_player = 1 if args.player == "X" else -1
            play_interactive(args.opponent, human_player)
        case "evaluate":
            evaluate_agents_cli(args.agent1, args.agent2, args.games)
        case "demo":
            demo_game(args.agent1, args.agent2)

    sys.exit(0)


if __name__ == "__main__":
    main()
