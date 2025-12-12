"""Tests for CorticalSimulation."""

import numpy as np
import pytest

from src.main import CorticalSimulation
from src.model.config import CELL_TYPES, GRID_SIZE, LAYERS, RANDOM_SEED


def test_simulation_initialization(grid_size):
    """Test CorticalSimulation initializes correctly."""
    sim = CorticalSimulation(grid_size=grid_size)

    assert sim.grid_size == grid_size
    assert sim.circuit is not None
    assert sim.thalamus is not None
    assert sim.connectivity is not None


def test_simulation_reset(grid_size, random_seed):
    """Test CorticalSimulation reset produces same initial state."""
    np.random.seed(random_seed)
    sim1 = CorticalSimulation(grid_size=grid_size)
    initial_state1 = sim1.update()

    # Reset and get initial state again
    sim1.reset()
    initial_state2 = sim1.update()

    # Initial states should be the same after reset
    # (with same random seed)
    for layer_name in LAYERS:
        for cell_type in CELL_TYPES:
            np.testing.assert_array_almost_equal(
                initial_state1[layer_name][cell_type],
                initial_state2[layer_name][cell_type],
                decimal=5,
            )


def test_simulation_update_structure(grid_size):
    """Test CorticalSimulation.update() returns expected structure."""
    sim = CorticalSimulation(grid_size=grid_size)

    activities = sim.update()

    # Check structure
    assert isinstance(activities, dict)

    # Check all layers present
    for layer_name in LAYERS:
        assert layer_name in activities

        # Check all cell types present
        for cell_type in CELL_TYPES:
            assert cell_type in activities[layer_name]

            # Check shape
            assert activities[layer_name][cell_type].shape == (grid_size, grid_size)

    # Check thalamus is present
    assert "thalamus" in activities
    assert activities["thalamus"].shape == (grid_size, grid_size)


def test_simulation_time_constant_setter_getter(grid_size):
    """Test time constant setter/getter methods."""
    sim = CorticalSimulation(grid_size=grid_size)

    # Set time constants
    sim.set_time_constant("E", 10.0)
    sim.set_time_constant("SST", 20.0)
    sim.set_time_constant("PV", 30.0)

    # Get time constants
    taus = sim.get_time_constants()

    assert taus["E"] == 10.0
    assert taus["SST"] == 20.0
    assert taus["PV"] == 30.0


def test_simulation_gain_setter_getter(grid_size):
    """Test gain setter/getter methods."""
    sim = CorticalSimulation(grid_size=grid_size)

    # Set gains
    sim.set_gain("E", 1.5)
    sim.set_gain("SST", 2.0)
    sim.set_gain("PV", 1.2)

    # Get gains
    gains = sim.get_gains()

    assert gains["E"] == 1.5
    assert gains["SST"] == 2.0
    assert gains["PV"] == 1.2


def test_simulation_connection_sigma_setter_getter(grid_size):
    """Test connection sigma setter/getter methods."""
    sim = CorticalSimulation(grid_size=grid_size)

    # Set connection sigma
    new_sigma = 3.5
    sim.set_connection_sigma("L4", "E", "L4", "E", new_sigma)

    # Get connection sigma
    sigma = sim.get_connection_sigma("L4", "E", "L4", "E")

    assert sigma == new_sigma


def test_simulation_strength_scaling(grid_size):
    """Test strength scaling setter/getter."""
    sim = CorticalSimulation(grid_size=grid_size)

    scaling = 2.0
    sim.set_strength_scaling("E", scaling)

    retrieved = sim.get_strength_scaling("E")

    assert retrieved == scaling


def test_simulation_background_input(grid_size):
    """Test background input setter."""
    sim = CorticalSimulation(grid_size=grid_size)

    # Set background input
    sim.set_background_input("E", 0.1)
    sim.set_background_input("SST", -0.05)
    sim.set_background_input("PV", 0.0)

    # Verify by checking if simulation still works
    activities = sim.update()
    assert activities is not None


def test_simulation_connectivity_property(grid_size):
    """Test connectivity property access."""
    sim = CorticalSimulation(grid_size=grid_size)

    connectivity = sim.connectivity

    assert connectivity is not None
    assert connectivity.grid_size == grid_size
