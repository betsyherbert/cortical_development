"""Tests for connectivity components."""

import numpy as np
import pytest
from src.model.connectivity import ConnectivityProfile, LayerConnectivity
from src.model.config import GRID_SIZE, LAYERS, CELL_TYPES


def test_connectivity_profile_initialization(grid_size):
    """Test ConnectivityProfile initializes correctly."""
    profile = ConnectivityProfile(grid_size)
    
    assert profile.grid_size == grid_size
    assert profile.center == (grid_size // 2, grid_size // 2)


def test_connectivity_profile_gaussian_profile(grid_size):
    """Test ConnectivityProfile generates Gaussian profiles."""
    profile = ConnectivityProfile(grid_size)
    
    sigma = 2.0
    gaussian = profile.gaussian_profile(sigma)
    
    assert gaussian.shape == (grid_size, grid_size)
    assert gaussian.sum() > 0  # Should be normalized
    # Center should have highest value
    center_idx = grid_size // 2
    center_value = gaussian[center_idx, center_idx]
    assert center_value > 0


def test_connectivity_profile_weight_matrix(grid_size):
    """Test ConnectivityProfile computes weight matrices."""
    profile = ConnectivityProfile(grid_size)
    
    amplitude = 1.0
    sigma = 2.0
    source_size = (grid_size, grid_size)
    target_size = (grid_size, grid_size)
    
    weight_matrix = profile.compute_weight_matrix(
        amplitude, sigma, source_size, target_size
    )
    
    n_neurons = grid_size * grid_size
    assert weight_matrix.shape == (n_neurons, n_neurons)


def test_layer_connectivity_initialization(grid_size):
    """Test LayerConnectivity initializes correctly."""
    connectivity = LayerConnectivity(grid_size)
    
    assert connectivity.grid_size == grid_size
    assert connectivity.profile is not None


def test_layer_connectivity_get_connection_sigma(grid_size):
    """Test LayerConnectivity connection sigma getter."""
    connectivity = LayerConnectivity(grid_size)
    
    # Get default sigma
    sigma = connectivity.get_connection_sigma('L4', 'E', 'L4', 'E')
    
    assert isinstance(sigma, float)
    assert sigma > 0


def test_layer_connectivity_set_connection_sigma(grid_size):
    """Test LayerConnectivity connection sigma setter/getter."""
    connectivity = LayerConnectivity(grid_size)
    
    # Set a new sigma value
    new_sigma = 3.5
    connectivity.set_connection_sigma('L4', 'E', 'L4', 'E', new_sigma)
    
    # Get it back
    sigma = connectivity.get_connection_sigma('L4', 'E', 'L4', 'E')
    
    assert sigma == new_sigma


def test_layer_connectivity_strength_scaling(grid_size):
    """Test LayerConnectivity strength scaling setter/getter."""
    connectivity = LayerConnectivity(grid_size)
    
    # Set scaling
    scaling = 2.0
    connectivity.set_strength_scaling('E', scaling)
    
    # Get scaling
    retrieved_scaling = connectivity.get_strength_scaling('E')
    
    assert retrieved_scaling == scaling


def test_connectivity_profile_caching(grid_size):
    """Test ConnectivityProfile caches computed profiles."""
    profile = ConnectivityProfile(grid_size)
    
    sigma = 2.0
    gaussian1 = profile.gaussian_profile(sigma)
    
    # Compute again - should use cache
    gaussian2 = profile.gaussian_profile(sigma)
    
    # Should be the same object (cached)
    assert np.array_equal(gaussian1, gaussian2)

