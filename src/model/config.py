"""Configuration parameters for the cortical circuit simulation.

This module contains all the key parameters for:
1. Simulation settings (grid size, time step)
2. Thalamic input generation
3. Neural network connectivity
4. Visualization settings
"""

# Simulation parameters
GRID_SIZE = 20  # Size of the 2D grid
DT = 1.5  # Time step in milliseconds - INCREASE to speed up simulation
INTEGRATION_STEPS = 5  # Number of steps per update cycle - INCREASE for faster visual changes
VISUALIZATION_STEPS = 3  # Number of simulation steps per visualization update - DECREASE for more frequent visual updates

# Thalamic input parameters
THALAMIC_INTRINSIC_SIGMA = 4.0  # Mean spatial spread of intrinsic bursts (grid units)
THALAMIC_INTRINSIC_DURATION = 30.0  # Mean duration of intrinsic bursts (ms)
THALAMIC_INTRINSIC_INTERVAL = 20.0  # Mean interval between intrinsic bursts (ms)
THALAMIC_INTRINSIC_AMP = 2.0  # Mean amplitude of intrinsic bursts

THALAMIC_SENSORY_SIGMA = 1.5  # Spatial spread of sensory inputs (grid units)
THALAMIC_SENSORY_DURATION = 10.0  # Duration of sensory bursts (ms)
THALAMIC_SENSORY_INTERVAL = 10.0  # Mean interval between sensory bursts (ms)
THALAMIC_SENSORY_AMP = 1.0  # Mean amplitude of sensory bursts

THALAMIC_SCALING = 1.0  # Overall scaling factor for thalamic input
THALAMIC_ALPHA = 0.2  # Balance between intrinsic (0) and sensory (1) thalamic activity

# Neural network parameters
NOISE_AMPLITUDE = 0.00  # Standard deviation of neural noise - INCREASE for more rapid random activity changes

# Network structure
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

# Layer-specific connection parameters
# Format: '{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
LAYER_CONNECTIVITY_PARAMS = {
    # -----------------------------------------------------
    # L2/3 -> L2/3 connections
    # -----------------------------------------------------
    'L23_E_to_L23_E': {'amplitude': 0.2, 'sigma': 2.0},
    'L23_E_to_L23_SST': {'amplitude': 0.1, 'sigma': 2.0},
    'L23_E_to_L23_PV': {'amplitude': 0.2, 'sigma': 2.0},
    'L23_SST_to_L23_E': {'amplitude': 0.1, 'sigma': 3.0},
    'L23_SST_to_L23_PV': {'amplitude': 0.0, 'sigma': 3.0},
    'L23_PV_to_L23_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L23_PV_to_L23_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L23_PV_to_L23_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L2/3 -> L4 connections
    # -----------------------------------------------------
    'L23_E_to_L4_E': {'amplitude': 0.1, 'sigma': 2.5},
    'L23_E_to_L4_SST': {'amplitude': 0.1, 'sigma': 2.5},
    'L23_E_to_L4_PV': {'amplitude': 0.5, 'sigma': 2.5},
    'L23_SST_to_L4_E': {'amplitude': 0.0, 'sigma': 3.0},
    'L23_SST_to_L4_PV': {'amplitude': 0.0, 'sigma': 3.0},
    'L23_PV_to_L4_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L23_PV_to_L4_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L23_PV_to_L4_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L2/3 -> L5 connections
    # -----------------------------------------------------
    'L23_E_to_L5_E': {'amplitude': 0.1, 'sigma': 3.0},
    'L23_E_to_L5_SST': {'amplitude': 0.1, 'sigma': 3.0},
    'L23_E_to_L5_PV': {'amplitude': 0.1, 'sigma': 3.0},
    'L23_SST_to_L5_E': {'amplitude': 0.0, 'sigma': 3.0},
    'L23_SST_to_L5_PV': {'amplitude': 0.0, 'sigma': 3.0},
    'L23_PV_to_L5_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L23_PV_to_L5_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L23_PV_to_L5_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L2/3 connections
    # -----------------------------------------------------
    'L4_E_to_L23_E': {'amplitude': 0.1, 'sigma': 2.5},
    'L4_E_to_L23_SST': {'amplitude': 0.1, 'sigma': 2.5},
    'L4_E_to_L23_PV': {'amplitude': 0.1, 'sigma': 2.5},
    'L4_SST_to_L23_E': {'amplitude': 0.1, 'sigma': 3.0},
    'L4_SST_to_L23_PV': {'amplitude': 0.0, 'sigma': 3.0},
    'L4_PV_to_L23_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L4_PV_to_L23_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L4_PV_to_L23_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L4 connections
    # -----------------------------------------------------
    'L4_E_to_L4_E': {'amplitude': 0.1, 'sigma': 2.0},
    'L4_E_to_L4_SST': {'amplitude': 0.1, 'sigma': 2.0},
    'L4_E_to_L4_PV': {'amplitude': 0.1, 'sigma': 2.0},
    'L4_SST_to_L4_E': {'amplitude': 0.0, 'sigma': 3.0},
    'L4_SST_to_L4_PV': {'amplitude': 0.0, 'sigma': 3.0},
    'L4_PV_to_L4_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L4_PV_to_L4_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L4_PV_to_L4_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L5 connections
    # -----------------------------------------------------
    'L4_E_to_L5_E': {'amplitude': 0.1, 'sigma': 2.5},
    'L4_E_to_L5_SST': {'amplitude': 0.5, 'sigma': 2.5},
    'L4_E_to_L5_PV': {'amplitude': 0.1, 'sigma': 2.5},
    'L4_SST_to_L5_E': {'amplitude': 0.0, 'sigma': 3.0},
    'L4_SST_to_L5_PV': {'amplitude': 0.0, 'sigma': 3.0},
    'L4_PV_to_L5_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L4_PV_to_L5_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L4_PV_to_L5_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L2/3 connections
    # -----------------------------------------------------
    'L5_E_to_L23_E': {'amplitude': 0.5, 'sigma': 3.0},
    'L5_E_to_L23_SST': {'amplitude': 0.1, 'sigma': 3.0},
    'L5_E_to_L23_PV': {'amplitude': 0.1, 'sigma': 3.0},
    'L5_SST_to_L23_E': {'amplitude': 0.2, 'sigma': 3.0},
    'L5_SST_to_L23_PV': {'amplitude': 0.0, 'sigma': 3.0},
    'L5_PV_to_L23_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L5_PV_to_L23_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L5_PV_to_L23_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L4 connections
    # -----------------------------------------------------
    'L5_E_to_L4_E': {'amplitude': 0.1, 'sigma': 3.0},
    'L5_E_to_L4_SST': {'amplitude': 0.1, 'sigma': 3.0},
    'L5_E_to_L4_PV': {'amplitude': 0.1, 'sigma': 3.0},
    'L5_SST_to_L4_E': {'amplitude': 0.5, 'sigma': 3.0},
    'L5_SST_to_L4_PV': {'amplitude': 0.5, 'sigma': 3.0},
    'L5_PV_to_L4_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L5_PV_to_L4_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L5_PV_to_L4_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L5 connections
    # -----------------------------------------------------
    'L5_E_to_L5_E': {'amplitude': 0.1, 'sigma': 2.0},
    'L5_E_to_L5_SST': {'amplitude': 0.1, 'sigma': 2.0},
    'L5_E_to_L5_PV': {'amplitude': 0.1, 'sigma': 2.0},
    'L5_SST_to_L5_E': {'amplitude': 0.5, 'sigma': 3.0},
    'L5_SST_to_L5_PV': {'amplitude': 0.5, 'sigma': 3.0},
    'L5_PV_to_L5_E': {'amplitude': 0.0, 'sigma': 1.5},
    'L5_PV_to_L5_SST': {'amplitude': 0.0, 'sigma': 1.5},
    'L5_PV_to_L5_PV': {'amplitude': 0.0, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # Thalamic connections to all layers
    # -----------------------------------------------------
    'thalamus_to_L23_E': {'amplitude': 0.1, 'sigma': 2.0},
    'thalamus_to_L23_SST': {'amplitude': 0.0, 'sigma': 2.0},
    'thalamus_to_L23_PV': {'amplitude': 0.0, 'sigma': 2.0},
    
    'thalamus_to_L4_E': {'amplitude': 0.5, 'sigma': 2.0},
    'thalamus_to_L4_SST': {'amplitude': 0.3, 'sigma': 2.0},
    'thalamus_to_L4_PV': {'amplitude': 0.1, 'sigma': 2.0},
    
    'thalamus_to_L5_E': {'amplitude': 0.1, 'sigma': 2.0},
    'thalamus_to_L5_SST': {'amplitude': 0.5, 'sigma': 2.0},
    'thalamus_to_L5_PV': {'amplitude': 0.1, 'sigma': 2.0}
}

# Initial values for connection widths
INITIAL_THALAMIC_WIDTHS = {
    'E': 6.0,    # Initial thalamic input width for E cells
    'SST': 6.0,  # Initial thalamic input width for SST cells
    'PV': 6.0    # Initial thalamic input width for PV cells
}

INITIAL_OUTGOING_WIDTHS = {
    'E': 4.0,    # Initial outgoing width for E cells
    'SST': 4.0,  # Initial outgoing width for SST cells
    'PV': 4.0    # Initial outgoing width for PV cells
}

# Initial values for time constants (ms)
INITIAL_TIME_CONSTANTS = {
    'E': 80.0,   # Initial time constant for E cells
    'SST': 60.0, # Initial time constant for SST cells
    'PV': 40.0   # Initial time constant for PV cells
}

# Initial values for gains
INITIAL_GAINS = {
    'E': 1.0,    # Initial gain for E cells
    'SST': 1.0,  # Initial gain for SST cells
    'PV': 1.0    # Initial gain for PV cells
}

# Layer specific connection list (source_layer, source_cell, target_layer, target_cell)
LAYER_CONNECTIONS = [
    # Generate all valid layer-specific connections
    (source_layer, source_cell, target_layer, target_cell)
    for source_layer in LAYERS
    for source_cell in CELL_TYPES 
    for target_layer in LAYERS
    for target_cell in CELL_TYPES
    # Skip connections that don't exist in the base model (e.g. SST to SST)
    if (source_cell, target_cell) in CONNECTIONS
] + [
    # Add thalamic connections
    ('thalamus', None, layer, cell_type)
    for layer in LAYERS
    for cell_type in CELL_TYPES
]

# Color configuration for cell types
CELL_COLORS = {
    'E': '#17BFD8',    # Blue
    'SST': '#FF630C',  # Orange
    'PV': '#D91B12'    # Red
}

# Helper function to convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Colormaps for heatmaps and connectivity visualization
COLORMAPS = {
    'E': [[0, 'black'], [1, CELL_COLORS['E']]],
    'SST': [[0, 'black'], [1, CELL_COLORS['SST']]],
    'PV': [[0, 'black'], [1, CELL_COLORS['PV']]],
    'thalamus': [[0, 'black'], [1, 'white']]
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

# Visualization settings
UPDATE_INTERVAL = 60  # Time between visualization updates (ms) - DECREASE for more frequent visual updates 