"""Configuration parameters for the cortical circuit simulation.

This module contains all the key parameters for:
1. Simulation settings (grid size, time step)
2. Thalamic input generation
3. Neural network connectivity
4. Visualization settings
"""

from .presets import P4_PRESET

#------------------------------------------------------------------------------
# Core Simulation Parameters
#------------------------------------------------------------------------------
GRID_SIZE = 20  # Size of the 2D grid
DT = 1.5  # Time step in milliseconds - INCREASE to speed up simulation
INTEGRATION_STEPS = 5  # Number of steps per update cycle - INCREASE for faster visual changes
VISUALIZATION_STEPS = 3  # Number of simulation steps per visualization update - DECREASE for more frequent visual updates
UPDATE_INTERVAL = 60  # Time between visualization updates (ms) - DECREASE for more frequent visual updates

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
THALAMIC_SCALING = 1.0  # Overall scaling factor for thalamic input
THALAMIC_ALPHA = P4_PRESET['thalamic_alpha']  # Balance between intrinsic (0) and sensory (1) thalamic activity

# Intrinsic burst parameters
THALAMIC_INTRINSIC_SIGMA = 4.0  # Mean spatial spread of intrinsic bursts (grid units)
THALAMIC_INTRINSIC_DURATION = 30.0  # Mean duration of intrinsic bursts (ms)
THALAMIC_INTRINSIC_INTERVAL = 20.0  # Mean interval between intrinsic bursts (ms)
THALAMIC_INTRINSIC_AMP = 2.0  # Mean amplitude of intrinsic bursts

# Sensory burst parameters
THALAMIC_SENSORY_SIGMA = 1.5  # Spatial spread of sensory inputs (grid units)
THALAMIC_SENSORY_DURATION = 10.0  # Duration of sensory bursts (ms)
THALAMIC_SENSORY_INTERVAL = 10.0  # Mean interval between sensory bursts (ms)
THALAMIC_SENSORY_AMP = 1.0  # Mean amplitude of sensory bursts

#------------------------------------------------------------------------------
# Initial Network Parameters
#------------------------------------------------------------------------------
INITIAL_THALAMIC_WIDTHS = P4_PRESET['thalamic_widths']
INITIAL_OUTGOING_WIDTHS = P4_PRESET['outgoing_widths']
INITIAL_STRENGTH_SCALING = P4_PRESET['strength_scaling']
INITIAL_SPARSITY = P4_PRESET['sparsity']
INITIAL_TIME_CONSTANTS = P4_PRESET['time_constants']
INITIAL_GAINS = P4_PRESET['gains']

#------------------------------------------------------------------------------
# Visualization Settings
#------------------------------------------------------------------------------
# Color configuration for cell types
CELL_COLORS = {
    'E': '#17BFD8',    # Blue
    'SST': '#FF630C',  # Orange
    'PV': '#D91B12'    # Red
}

# Colormaps for heatmaps and connectivity visualization
COLORMAPS = {
    'E': [[0, 'black'], [1, CELL_COLORS['E']]],
    'SST': [[0, 'black'], [1, CELL_COLORS['SST']]],
    'PV': [[0, 'black'], [1, CELL_COLORS['PV']]],
}

# Cell activity colors for connectivity matrix
CELL_ACTIVITY_COLORS = {
    'E': {
        'bg': lambda i: f"rgba{(*hex_to_rgb(CELL_COLORS['E']), i)}",
        'hover': lambda i: f"rgba{(*hex_to_rgb(CELL_COLORS['E']), min(i + 0.2, 1.0))}"
    },
    'SST': {
        'bg': lambda i: f"rgba{(*hex_to_rgb(CELL_COLORS['SST']), i)}",
        'hover': lambda i: f"rgba{(*hex_to_rgb(CELL_COLORS['SST']), min(i + 0.2, 1.0))}"
    },
    'PV': {
        'bg': lambda i: f"rgba{(*hex_to_rgb(CELL_COLORS['PV']), i)}",
        'hover': lambda i: f"rgba{(*hex_to_rgb(CELL_COLORS['PV']), min(i + 0.2, 1.0))}"
    },
    'inactive': {
        'bg': "rgba(80, 80, 80, 0.1)",
        'hover': "rgba(100, 100, 100, 0.3)"
    }
}

#------------------------------------------------------------------------------
# Layer Connectivity Parameters
#------------------------------------------------------------------------------
# Format: '{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
LAYER_CONNECTIVITY_PARAMS = {
    # -----------------------------------------------------
    # L2/3 -> L2/3 connections
    # -----------------------------------------------------
    'L23_E_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L23_E'], 'sigma': 2.0},
    'L23_E_to_L23_SST': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L23_SST'], 'sigma': 2.0},
    'L23_E_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L23_PV'], 'sigma': 2.0},
    'L23_SST_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L23_SST_to_L23_E'], 'sigma': 3.0},
    'L23_SST_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_SST_to_L23_PV'], 'sigma': 3.0},
    'L23_PV_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L23_E'], 'sigma': 1.5},
    'L23_PV_to_L23_SST': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L23_SST'], 'sigma': 1.5},
    'L23_PV_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L23_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L2/3 -> L4 connections
    # -----------------------------------------------------
    'L23_E_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L4_E'], 'sigma': 2.5},
    'L23_E_to_L4_SST': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L4_SST'], 'sigma': 2.5},
    'L23_E_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L4_PV'], 'sigma': 2.5},
    'L23_SST_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L23_SST_to_L4_E'], 'sigma': 3.0},
    'L23_SST_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_SST_to_L4_PV'], 'sigma': 3.0},
    'L23_PV_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L4_E'], 'sigma': 1.5},
    'L23_PV_to_L4_SST': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L4_SST'], 'sigma': 1.5},
    'L23_PV_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L4_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L2/3 -> L5 connections
    # -----------------------------------------------------
    'L23_E_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L5_E'], 'sigma': 3.0},
    'L23_E_to_L5_SST': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L5_SST'], 'sigma': 3.0},
    'L23_E_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_E_to_L5_PV'], 'sigma': 3.0},
    'L23_SST_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L23_SST_to_L5_E'], 'sigma': 3.0},
    'L23_SST_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_SST_to_L5_PV'], 'sigma': 3.0},
    'L23_PV_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L5_E'], 'sigma': 1.5},
    'L23_PV_to_L5_SST': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L5_SST'], 'sigma': 1.5},
    'L23_PV_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L23_PV_to_L5_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L2/3 connections
    # -----------------------------------------------------
    'L4_E_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L23_E'], 'sigma': 2.5},
    'L4_E_to_L23_SST': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L23_SST'], 'sigma': 2.5},
    'L4_E_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L23_PV'], 'sigma': 2.5},
    'L4_SST_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L4_SST_to_L23_E'], 'sigma': 3.0},
    'L4_SST_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_SST_to_L23_PV'], 'sigma': 3.0},
    'L4_PV_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L23_E'], 'sigma': 1.5},
    'L4_PV_to_L23_SST': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L23_SST'], 'sigma': 1.5},
    'L4_PV_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L23_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L4 connections
    # -----------------------------------------------------
    'L4_E_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L4_E'], 'sigma': 2.0},
    'L4_E_to_L4_SST': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L4_SST'], 'sigma': 2.0},
    'L4_E_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L4_PV'], 'sigma': 2.0},
    'L4_SST_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L4_SST_to_L4_E'], 'sigma': 3.0},
    'L4_SST_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_SST_to_L4_PV'], 'sigma': 3.0},
    'L4_PV_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L4_E'], 'sigma': 1.5},
    'L4_PV_to_L4_SST': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L4_SST'], 'sigma': 1.5},
    'L4_PV_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L4_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L5 connections
    # -----------------------------------------------------
    'L4_E_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L5_E'], 'sigma': 2.5},
    'L4_E_to_L5_SST': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L5_SST'], 'sigma': 2.5},
    'L4_E_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_E_to_L5_PV'], 'sigma': 2.5},
    'L4_SST_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L4_SST_to_L5_E'], 'sigma': 3.0},
    'L4_SST_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_SST_to_L5_PV'], 'sigma': 3.0},
    'L4_PV_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L5_E'], 'sigma': 1.5},
    'L4_PV_to_L5_SST': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L5_SST'], 'sigma': 1.5},
    'L4_PV_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L4_PV_to_L5_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L2/3 connections
    # -----------------------------------------------------
    'L5_E_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L23_E'], 'sigma': 3.0},
    'L5_E_to_L23_SST': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L23_SST'], 'sigma': 3.0},
    'L5_E_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L23_PV'], 'sigma': 3.0},
    'L5_SST_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L5_SST_to_L23_E'], 'sigma': 3.0},
    'L5_SST_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_SST_to_L23_PV'], 'sigma': 3.0},
    'L5_PV_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L23_E'], 'sigma': 1.5},
    'L5_PV_to_L23_SST': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L23_SST'], 'sigma': 1.5},
    'L5_PV_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L23_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L4 connections
    # -----------------------------------------------------
    'L5_E_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L4_E'], 'sigma': 3.0},
    'L5_E_to_L4_SST': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L4_SST'], 'sigma': 3.0},
    'L5_E_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L4_PV'], 'sigma': 3.0},
    'L5_SST_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L5_SST_to_L4_E'], 'sigma': 3.0},
    'L5_SST_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_SST_to_L4_PV'], 'sigma': 3.0},
    'L5_PV_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L4_E'], 'sigma': 1.5},
    'L5_PV_to_L4_SST': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L4_SST'], 'sigma': 1.5},
    'L5_PV_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L4_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L5 connections
    # -----------------------------------------------------
    'L5_E_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L5_E'], 'sigma': 2.0},
    'L5_E_to_L5_SST': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L5_SST'], 'sigma': 2.0},
    'L5_E_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_E_to_L5_PV'], 'sigma': 2.0},
    'L5_SST_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L5_SST_to_L5_E'], 'sigma': 3.0},
    'L5_SST_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_SST_to_L5_PV'], 'sigma': 3.0},
    'L5_PV_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L5_E'], 'sigma': 1.5},
    'L5_PV_to_L5_SST': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L5_SST'], 'sigma': 1.5},
    'L5_PV_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['L5_PV_to_L5_PV'], 'sigma': 1.5},
    
    # -----------------------------------------------------
    # Thalamic connections to all layers
    # -----------------------------------------------------
    'thalamus_to_L23_E': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L23_E'], 'sigma': 2.0},
    'thalamus_to_L23_SST': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L23_SST'], 'sigma': 2.0},
    'thalamus_to_L23_PV': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L23_PV'], 'sigma': 2.0},
    
    'thalamus_to_L4_E': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L4_E'], 'sigma': 2.0},
    'thalamus_to_L4_SST': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L4_SST'], 'sigma': 2.0},
    'thalamus_to_L4_PV': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L4_PV'], 'sigma': 2.0},
    
    'thalamus_to_L5_E': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L5_E'], 'sigma': 2.0},
    'thalamus_to_L5_SST': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L5_SST'], 'sigma': 2.0},
    'thalamus_to_L5_PV': {'amplitude': P4_PRESET['connection_strengths']['thalamus_to_L5_PV'], 'sigma': 2.0}
}

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------
def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) 