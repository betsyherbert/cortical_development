"""Tests for connectivity components.

Note: All sigma (spatial width) parameters are in μm (anatomical units).
Default anatomical grid size is 1000 μm × 1000 μm.
"""

import numpy as np
import pytest
from src.model.connectivity import ConnectivityProfile, LayerConnectivity
from src.model.config import GRID_SIZE, LAYERS, CELL_TYPES, ANATOMICAL_GRID_SIZE


def test_connectivity_profile_initialization(grid_size):
    """Test ConnectivityProfile initializes correctly."""
    profile = ConnectivityProfile(grid_size)
    
    assert profile.grid_size == grid_size
    assert profile.center == (grid_size // 2, grid_size // 2)
    assert profile.anatomical_grid_size == ANATOMICAL_GRID_SIZE
    assert profile.grid_scale == ANATOMICAL_GRID_SIZE / grid_size


def test_connectivity_profile_gaussian_profile(grid_size):
    """Test ConnectivityProfile generates Gaussian profiles."""
    profile = ConnectivityProfile(grid_size)
    
    # sigma is now in μm (100 μm = 2 grid units at default 50 μm/grid)
    sigma_um = 100.0
    gaussian = profile.gaussian_profile(sigma_um)
    
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
    sigma_um = 100.0  # sigma in μm
    source_size = (grid_size, grid_size)
    target_size = (grid_size, grid_size)
    
    weight_matrix = profile.compute_weight_matrix(
        amplitude, sigma_um, source_size, target_size
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
    
    # Set a new sigma value in μm
    new_sigma_um = 175.0  # 175 μm
    connectivity.set_connection_sigma('L4', 'E', 'L4', 'E', new_sigma_um)
    
    # Get it back
    sigma = connectivity.get_connection_sigma('L4', 'E', 'L4', 'E')
    
    assert sigma == new_sigma_um


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
    
    # sigma in μm
    sigma_um = 100.0
    gaussian1 = profile.gaussian_profile(sigma_um)
    
    # Compute again - should use cache
    gaussian2 = profile.gaussian_profile(sigma_um)
    
    # Should be the same object (cached)
    assert np.array_equal(gaussian1, gaussian2)


def test_spatial_scale_conversion():
    """Test that spatial scale conversion works correctly."""
    from src.model.config import get_grid_scale, um_to_grid, grid_to_um
    
    # Default values: 1000 μm / 20 grid = 50 μm/grid
    scale = get_grid_scale()
    assert scale == 50.0
    
    # Convert 100 μm to grid units
    grid_units = um_to_grid(100.0)
    assert grid_units == 2.0
    
    # Convert 2 grid units to μm
    um = grid_to_um(2.0)
    assert um == 100.0
    
    # Round-trip conversion
    original_um = 150.0
    grid = um_to_grid(original_um)
    back_to_um = grid_to_um(grid)
    assert abs(back_to_um - original_um) < 1e-10

