"""Configuration parameters for the cortical circuit simulation.

This module contains all the key parameters for:
1. Simulation settings (grid size, time step)
2. Thalamic input generation
3. Neural network connectivity
4. Visualization settings
"""

import numpy as np
from .presets import P0_PRESET

#------------------------------------------------------------------------------
# Core Simulation Parameters
#------------------------------------------------------------------------------
GRID_SIZE = 20  # Number of grid points in each dimension (N x N neurons per layer)
ANATOMICAL_GRID_SIZE = 1000.0  # Anatomical size of the grid in μm (default: 1000 μm × 1000 μm)
DT = 1.5  # Time step in milliseconds - INCREASE to speed up simulation
INTEGRATION_STEPS = 3  # Number of steps per update cycle - INCREASE for faster visual changes
VISUALIZATION_STEPS = 5  # Number of simulation steps per visualization update - DECREASE for more frequent visual updates
UPDATE_INTERVAL = 60  # Wall clock time between dashboard UI refresh (ms) - DECREASE for more frequent visual updates
RANDOM_SEED = 9  # Global random seed for reproducible simulations


#------------------------------------------------------------------------------
# Spatial Scale Conversion Functions
#------------------------------------------------------------------------------

def get_grid_scale(grid_size: int = None, anatomical_size: float = None) -> float:
    """Get the spatial scale in μm per grid unit.
    
    Args:
        grid_size: Number of grid points (default: GRID_SIZE)
        anatomical_size: Anatomical size in μm (default: ANATOMICAL_GRID_SIZE)
        
    Returns:
        Scale factor in μm per grid unit
    """
    if grid_size is None:
        grid_size = GRID_SIZE
    if anatomical_size is None:
        anatomical_size = ANATOMICAL_GRID_SIZE
    return anatomical_size / grid_size


def um_to_grid(value_um: float, grid_size: int = None, anatomical_size: float = None) -> float:
    """Convert a distance from μm to grid units.
    
    Args:
        value_um: Distance in μm
        grid_size: Number of grid points (default: GRID_SIZE)
        anatomical_size: Anatomical size in μm (default: ANATOMICAL_GRID_SIZE)
        
    Returns:
        Distance in grid units
    """
    scale = get_grid_scale(grid_size, anatomical_size)
    return value_um / scale


def grid_to_um(value_grid: float, grid_size: int = None, anatomical_size: float = None) -> float:
    """Convert a distance from grid units to μm.
    
    Args:
        value_grid: Distance in grid units
        grid_size: Number of grid points (default: GRID_SIZE)
        anatomical_size: Anatomical size in μm (default: ANATOMICAL_GRID_SIZE)
        
    Returns:
        Distance in μm
    """
    scale = get_grid_scale(grid_size, anatomical_size)
    return value_grid * scale


#------------------------------------------------------------------------------
# Network Structure
#------------------------------------------------------------------------------
CELL_TYPES = ['E', 'SST', 'PV']  # Available cell types
LAYERS = ['L23', 'L4', 'L5']  # Cortical layers
LAYER_NAMES = {'L23': 'L2/3', 'L4': 'L4', 'L5': 'L5'}  # Display names for layers

# Basic connection list (source -> target) for convenience
CONNECTIONS = [
    # Excitatory connections
    ('E', 'E'), ('E', 'SST'), ('E', 'PV'),
    # Inhibitory connections
    ('SST', 'E'), ('SST', 'PV'),
    ('PV', 'E'), ('PV', 'SST'), ('PV', 'PV')
]

#------------------------------------------------------------------------------
# Thalamic Input Parameters
#------------------------------------------------------------------------------

# Intrinsic burst parameters
THALAMIC_INTRINSIC_SIGMA = 100.0  # Mean spatial spread of intrinsic bursts (μm)
THALAMIC_INTRINSIC_DURATION = 30.0  # Mean duration of intrinsic bursts (ms)
THALAMIC_INTRINSIC_INTERVAL = 20.0  # Mean interval between intrinsic bursts (ms)
THALAMIC_INTRINSIC_AMP = 3.0  # Mean amplitude of intrinsic bursts

# Sensory burst parameters
THALAMIC_SENSORY_SIGMA = 25.0  # Spatial spread of sensory inputs (μm)
THALAMIC_SENSORY_DURATION = 10.0  # Duration of sensory bursts (ms)
THALAMIC_SENSORY_INTERVAL = 10.0  # Mean interval between sensory bursts (ms)
THALAMIC_SENSORY_AMP = 1.0  # Mean amplitude of sensory bursts

#------------------------------------------------------------------------------
# Initial Network Parameters
#------------------------------------------------------------------------------
INITIAL_THALAMIC_WIDTHS = P0_PRESET['thalamic_widths']
INITIAL_OUTGOING_WIDTHS = P0_PRESET['outgoing_widths']
INITIAL_STRENGTH_SCALING = P0_PRESET['strength_scaling']
INITIAL_TIME_CONSTANTS = P0_PRESET['time_constants']
INITIAL_BACKGROUND_INPUT = P0_PRESET['background_input']  
THALAMIC_ALPHA = P0_PRESET['thalamic_alpha']  # Balance between intrinsic (0) and sensory (1) thalamic activity

#------------------------------------------------------------------------------
# Visualization Settings
#------------------------------------------------------------------------------
# Color configuration for cell types
CELL_COLORS = {
    'E': '#4292c2',    # Blue  #17BFD8
    'SST': '#FF630C',  # Orange
    'PV': '#D91B12'    # Red
}

# Colormaps for heatmaps and connectivity visualization
COLORMAPS = {
    'E': [[0, 'white'], [1, CELL_COLORS['E']]],
    'SST': [[0, 'white'], [1, CELL_COLORS['SST']]],
    'PV': [[0, 'white'], [1, CELL_COLORS['PV']]],
    'thalamus': [[0, 'white'], [1, 'black']],  # Black-on-white for thalamic activity
}

# Layer colors for visualizations
LAYER_COLORS = {
    'L5': '#999999',     # Light grey
    'L4': '#555555',     # Medium grey  
    'L23': '#222222'     # Dark grey
}

# Cell activity colors for connectivity matrix
CELL_ACTIVITY_COLORS = {
    'E': {
        'bg': lambda i: f"rgba{(*(int(CELL_COLORS['E'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)), i)}",
        'hover': lambda i: f"rgba{(*(int(CELL_COLORS['E'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)), min(i + 0.2, 1.0))}"
    },
    'SST': {
        'bg': lambda i: f"rgba{(*(int(CELL_COLORS['SST'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)), i)}",
        'hover': lambda i: f"rgba{(*(int(CELL_COLORS['SST'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)), min(i + 0.2, 1.0))}"
    },
    'PV': {
        'bg': lambda i: f"rgba{(*(int(CELL_COLORS['PV'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)), i)}",
        'hover': lambda i: f"rgba{(*(int(CELL_COLORS['PV'].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)), min(i + 0.2, 1.0))}"
    },
    'inactive': {
        'bg': "rgba(200, 200, 200, 0.2)",
        'hover': "rgba(180, 180, 180, 0.3)"
    }
}

#------------------------------------------------------------------------------
# Layer Connectivity Parameters
#------------------------------------------------------------------------------
# Format: '{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
# Note: All sigma values are in μm (anatomical units)
LAYER_CONNECTIVITY_PARAMS = {
    # -----------------------------------------------------
    # L2/3 -> L2/3 connections
    # -----------------------------------------------------
    'L23_E_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L23_E'], 'sigma': 100.0},
    'L23_E_to_L23_SST': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L23_SST'], 'sigma': 100.0},
    'L23_E_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L23_PV'], 'sigma': 100.0},
    'L23_SST_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L23_SST_to_L23_E'], 'sigma': 150.0},
    'L23_SST_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_SST_to_L23_PV'], 'sigma': 150.0},
    'L23_PV_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L23_E'], 'sigma': 75.0},
    'L23_PV_to_L23_SST': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L23_SST'], 'sigma': 75.0},
    'L23_PV_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L23_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L2/3 -> L4 connections
    # -----------------------------------------------------
    'L23_E_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L4_E'], 'sigma': 125.0},
    'L23_E_to_L4_SST': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L4_SST'], 'sigma': 125.0},
    'L23_E_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L4_PV'], 'sigma': 125.0},
    'L23_SST_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L23_SST_to_L4_E'], 'sigma': 150.0},
    'L23_SST_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_SST_to_L4_PV'], 'sigma': 150.0},
    'L23_PV_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L4_E'], 'sigma': 75.0},
    'L23_PV_to_L4_SST': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L4_SST'], 'sigma': 75.0},
    'L23_PV_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L4_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L2/3 -> L5 connections
    # -----------------------------------------------------
    'L23_E_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L5_E'], 'sigma': 150.0},
    'L23_E_to_L5_SST': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L5_SST'], 'sigma': 150.0},
    'L23_E_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_E_to_L5_PV'], 'sigma': 150.0},
    'L23_SST_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L23_SST_to_L5_E'], 'sigma': 150.0},
    'L23_SST_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_SST_to_L5_PV'], 'sigma': 150.0},
    'L23_PV_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L5_E'], 'sigma': 75.0},
    'L23_PV_to_L5_SST': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L5_SST'], 'sigma': 75.0},
    'L23_PV_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L23_PV_to_L5_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L4 -> L2/3 connections
    # -----------------------------------------------------
    'L4_E_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L23_E'], 'sigma': 125.0},
    'L4_E_to_L23_SST': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L23_SST'], 'sigma': 125.0},
    'L4_E_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L23_PV'], 'sigma': 125.0},
    'L4_SST_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L4_SST_to_L23_E'], 'sigma': 150.0},
    'L4_SST_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_SST_to_L23_PV'], 'sigma': 150.0},
    'L4_PV_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L23_E'], 'sigma': 75.0},
    'L4_PV_to_L23_SST': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L23_SST'], 'sigma': 75.0},
    'L4_PV_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L23_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L4 -> L4 connections
    # -----------------------------------------------------
    'L4_E_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L4_E'], 'sigma': 100.0},
    'L4_E_to_L4_SST': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L4_SST'], 'sigma': 100.0},
    'L4_E_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L4_PV'], 'sigma': 100.0},
    'L4_SST_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L4_SST_to_L4_E'], 'sigma': 150.0},
    'L4_SST_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_SST_to_L4_PV'], 'sigma': 150.0},
    'L4_PV_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L4_E'], 'sigma': 75.0},
    'L4_PV_to_L4_SST': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L4_SST'], 'sigma': 75.0},
    'L4_PV_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L4_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L4 -> L5 connections
    # -----------------------------------------------------
    'L4_E_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L5_E'], 'sigma': 125.0},
    'L4_E_to_L5_SST': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L5_SST'], 'sigma': 125.0},
    'L4_E_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_E_to_L5_PV'], 'sigma': 125.0},
    'L4_SST_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L4_SST_to_L5_E'], 'sigma': 150.0},
    'L4_SST_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_SST_to_L5_PV'], 'sigma': 150.0},
    'L4_PV_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L5_E'], 'sigma': 75.0},
    'L4_PV_to_L5_SST': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L5_SST'], 'sigma': 75.0},
    'L4_PV_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L4_PV_to_L5_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L5 -> L2/3 connections
    # -----------------------------------------------------
    'L5_E_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L23_E'], 'sigma': 150.0},
    'L5_E_to_L23_SST': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L23_SST'], 'sigma': 150.0},
    'L5_E_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L23_PV'], 'sigma': 150.0},
    'L5_SST_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L5_SST_to_L23_E'], 'sigma': 150.0},
    'L5_SST_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_SST_to_L23_PV'], 'sigma': 150.0},
    'L5_PV_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L23_E'], 'sigma': 75.0},
    'L5_PV_to_L23_SST': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L23_SST'], 'sigma': 75.0},
    'L5_PV_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L23_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L5 -> L4 connections
    # -----------------------------------------------------
    'L5_E_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L4_E'], 'sigma': 150.0},
    'L5_E_to_L4_SST': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L4_SST'], 'sigma': 150.0},
    'L5_E_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L4_PV'], 'sigma': 150.0},
    'L5_SST_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L5_SST_to_L4_E'], 'sigma': 150.0},
    'L5_SST_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_SST_to_L4_PV'], 'sigma': 150.0},
    'L5_PV_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L4_E'], 'sigma': 75.0},
    'L5_PV_to_L4_SST': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L4_SST'], 'sigma': 75.0},
    'L5_PV_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L4_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # L5 -> L5 connections
    # -----------------------------------------------------
    'L5_E_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L5_E'], 'sigma': 100.0},
    'L5_E_to_L5_SST': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L5_SST'], 'sigma': 100.0},
    'L5_E_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_E_to_L5_PV'], 'sigma': 100.0},
    'L5_SST_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L5_SST_to_L5_E'], 'sigma': 150.0},
    'L5_SST_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_SST_to_L5_PV'], 'sigma': 150.0},
    'L5_PV_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L5_E'], 'sigma': 75.0},
    'L5_PV_to_L5_SST': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L5_SST'], 'sigma': 75.0},
    'L5_PV_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['L5_PV_to_L5_PV'], 'sigma': 75.0},
    
    # -----------------------------------------------------
    # Thalamic connections to all layers
    # -----------------------------------------------------
    'thalamus_to_L23_E': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L23_E'], 'sigma': 100.0},
    'thalamus_to_L23_SST': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L23_SST'], 'sigma': 100.0},
    'thalamus_to_L23_PV': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L23_PV'], 'sigma': 100.0},
    
    'thalamus_to_L4_E': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L4_E'], 'sigma': 100.0},
    'thalamus_to_L4_SST': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L4_SST'], 'sigma': 100.0},
    'thalamus_to_L4_PV': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L4_PV'], 'sigma': 100.0},
    
    'thalamus_to_L5_E': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L5_E'], 'sigma': 100.0},
    'thalamus_to_L5_SST': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L5_SST'], 'sigma': 100.0},
    'thalamus_to_L5_PV': {'amplitude': P0_PRESET['connection_strengths']['thalamus_to_L5_PV'], 'sigma': 100.0}
}

#------------------------------------------------------------------------------
# Random Seed Management
#------------------------------------------------------------------------------

def seed_random(seed=None):
    """Set random seed for reproducible results.
    
    This function should be used instead of calling np.random.seed() directly.
    It ensures consistent seeding behavior across the entire codebase.
    
    Args:
        seed: Random seed to use. If None, uses RANDOM_SEED from config.
        
    Returns:
        The seed that was set (useful for logging/debugging)
        
    Example:
        >>> from src.model.config import seed_random
        >>> seed_random()  # Uses default seed from config
        >>> seed_random(42)  # Uses custom seed
    """
    if seed is None:
        seed = RANDOM_SEED
    
    np.random.seed(seed)
    return seed


def get_default_seed():
    """Get the default random seed from configuration.
    
    Returns:
        The default RANDOM_SEED value
    """
    return RANDOM_SEED 