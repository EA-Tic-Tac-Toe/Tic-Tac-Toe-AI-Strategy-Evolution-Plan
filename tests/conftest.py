"""Pytest fixtures for Tic-Tac-Toe tests."""

import pytest

from tictactoe.agents.heuristic_agent import HeuristicAgent
from tictactoe.agents.random_agent import RandomAgent
from tictactoe.board import Board


@pytest.fixture
def empty_board() -> Board:
    """Provide fresh board for each test."""
    return Board()


@pytest.fixture
def random_agent() -> RandomAgent:
    """Random agent with fixed seed."""
    return RandomAgent(player=1, seed=42)


@pytest.fixture
def heuristic_agent() -> HeuristicAgent:
    """Heuristic agent."""
    return HeuristicAgent(player=1)
