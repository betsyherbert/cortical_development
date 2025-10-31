"""Tests for neural dynamics components."""

import numpy as np
import pytest
from src.model.neurons import NoiseGenerator, NeuralLayer, CorticalCircuit
from src.model.config import CELL_TYPES, LAYERS, DT, NOISE_TAU, GRID_SIZE


def test_noise_generator_initialization(grid_size):
    """Test NoiseGenerator initializes correctly."""
    noise_gen = NoiseGenerator(grid_size, dt=DT, tau=NOISE_TAU)
    
    assert noise_gen.grid_size == grid_size
    assert noise_gen.dt == DT
    assert noise_gen.tau == NOISE_TAU
    
    # Check all cell types have noise states
    for cell_type in CELL_TYPES:
        assert cell_type in noise_gen.private_noise
        assert noise_gen.private_noise[cell_type].shape == (grid_size, grid_size)


def test_noise_generator_update(grid_size):
    """Test NoiseGenerator produces noise on update."""
    noise_gen = NoiseGenerator(grid_size, dt=DT, tau=NOISE_TAU)
    
    # Reset to zero
    noise_gen.reset()
    
    # Update and check noise is generated
    noise = noise_gen.update()
    
    assert isinstance(noise, dict)
    for cell_type in CELL_TYPES:
        assert cell_type in noise
        assert noise[cell_type].shape == (grid_size, grid_size)
        # After reset and one update, noise should be non-zero (stochastic)
        # (very unlikely all values remain exactly zero)


def test_noise_generator_parameters(grid_size):
    """Test NoiseGenerator parameter getter/setter."""
    noise_gen = NoiseGenerator(grid_size, dt=DT, tau=NOISE_TAU)
    
    # Test setting parameters
    noise_gen.set_parameters('E', mean=0.1, std=0.2, correlation=0.5)
    
    # Test getting parameters
    mean, std, corr = noise_gen.get_parameters('E')
    assert mean == 0.1
    assert std == 0.2
    assert corr == 0.5


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
    np.random.seed(random_seed)
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


def test_random_seed_reproducibility(grid_size):
    """Test that random seed produces reproducible results."""
    # First run
    np.random.seed(42)
    noise_gen1 = NoiseGenerator(grid_size, dt=DT, tau=NOISE_TAU)
    noise1 = noise_gen1.update()
    
    # Second run with same seed
    np.random.seed(42)
    noise_gen2 = NoiseGenerator(grid_size, dt=DT, tau=NOISE_TAU)
    noise2 = noise_gen2.update()
    
    # Results should be identical
    for cell_type in CELL_TYPES:
        assert np.allclose(noise1[cell_type], noise2[cell_type])


def test_neural_layer_time_constant_setter(grid_size):
    """Test NeuralLayer time constant setter/getter."""
    layer = NeuralLayer(grid_size, dt=DT)
    
    # Set time constants
    layer.set_time_constant('E', 10.0)
    layer.set_time_constant('SST', 20.0)
    layer.set_time_constant('PV', 30.0)
    
    # Get time constants
    taus = layer.get_time_constants()
    assert taus['E'] == 10.0
    assert taus['SST'] == 20.0
    assert taus['PV'] == 30.0

