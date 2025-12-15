"""Tests for neural dynamics components."""

import numpy as np

from src.model.config import CELL_TYPES, DT, LAYERS, seed_random
from src.model.neurons import CorticalCircuit, NeuralLayer


def test_neural_layer_initialization(grid_size):
    """Test NeuralLayer initializes correctly."""
    layer = NeuralLayer(grid_size, dt=DT)

    assert layer.grid_size == grid_size
    assert layer.dt == DT

    # Check all cell types have voltage and rate states
    for cell_type in CELL_TYPES:
        assert cell_type in layer.V
        assert cell_type in layer.r
        assert layer.V[cell_type].shape == (grid_size, grid_size)
        assert layer.r[cell_type].shape == (grid_size, grid_size)


def test_neural_layer_reset(grid_size):
    """Test NeuralLayer reset sets states to zero."""
    layer = NeuralLayer(grid_size, dt=DT)

    # Set some non-zero values
    for cell_type in CELL_TYPES:
        layer.V[cell_type].fill(1.0)
        layer.r[cell_type].fill(0.5)

    # Reset and verify
    layer.reset()

    for cell_type in CELL_TYPES:
        assert np.allclose(layer.V[cell_type], 0.0)
        assert np.allclose(layer.r[cell_type], 0.0)


def test_cortical_circuit_initialization(grid_size):
    """Test CorticalCircuit initializes correctly."""
    circuit = CorticalCircuit(grid_size)

    assert circuit.grid_size == grid_size

    # Check all layers exist
    for layer_name in LAYERS:
        assert layer_name in circuit.layers
        assert circuit.layers[layer_name].grid_size == grid_size


def test_cortical_circuit_update(grid_size, random_seed):
    """Test CorticalCircuit update returns expected structure."""
    seed_random(random_seed)
    circuit = CorticalCircuit(grid_size)

    # Set some thalamic input
    circuit.thalamus = np.zeros((grid_size, grid_size))

    # Update
    activities = circuit.update(n_steps=1)

    # Check structure
    assert isinstance(activities, dict)
    for layer_name in LAYERS:
        assert layer_name in activities
        for cell_type in CELL_TYPES:
            assert cell_type in activities[layer_name]
            assert activities[layer_name][cell_type].shape == (grid_size, grid_size)


def test_neural_layer_time_constant_setter(grid_size):
    """Test NeuralLayer time constant setter/getter."""
    layer = NeuralLayer(grid_size, dt=DT)

    # Set time constants
    layer.set_time_constant("E", 10.0)
    layer.set_time_constant("SST", 20.0)
    layer.set_time_constant("PV", 30.0)

    # Get time constants
    taus = layer.get_time_constants()
    assert taus["E"] == 10.0
    assert taus["SST"] == 20.0
    assert taus["PV"] == 30.0
