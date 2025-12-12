"""Pytest configuration and fixtures for testing."""

import numpy as np
import pytest

from src.main import CorticalSimulation
from src.model.config import get_default_seed, seed_random


@pytest.fixture
def grid_size():
    """Default grid size for faster tests."""
    return 10


@pytest.fixture
def random_seed():
    """Random seed for reproducible tests."""
    return get_default_seed()


@pytest.fixture
def simulation(grid_size, random_seed):
    """CorticalSimulation instance for testing."""
    seed_random(random_seed)
    sim = CorticalSimulation(grid_size=grid_size)
    return sim
