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
THALAMIC_INTRINSIC_FREQ = 10.0  # Base frequency for intrinsic oscillations (Hz) - INCREASE for faster oscillations
THALAMIC_INTRINSIC_SIGMA = 4.0  # Spatial spread of intrinsic activity (grid units) - DECREASE for more rapid spatial changes
THALAMIC_SENSORY_SIGMA = 1.5  # Spatial spread of sensory inputs (grid units)
THALAMIC_SENSORY_DURATION = 20.0  # Duration of sensory bursts (ms) - DECREASE for faster transitions
THALAMIC_SENSORY_INTERVAL = 50.0  # Mean interval between sensory bursts (ms) - DECREASE for more frequent activity changes
THALAMIC_SCALING = 1.0  # Overall scaling factor for thalamic input - INCREASE for more dramatic visual changes
THALAMIC_ALPHA = 0.2  # Balance between intrinsic (0) and sensory (1) thalamic activity

# Neural network parameters
NEURAL_TAU = 10.0  # Membrane time constant (ms) - DECREASE for faster neural dynamics and visual changes
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
    'L23_E_to_L23_E': {'amplitude': 0.35, 'sigma': 2.0},
    'L23_E_to_L23_SST': {'amplitude': 0.15, 'sigma': 2.0},
    'L23_E_to_L23_PV': {'amplitude': 0.20, 'sigma': 2.0},
    'L23_SST_to_L23_E': {'amplitude': -0.15, 'sigma': 3.0},
    'L23_SST_to_L23_PV': {'amplitude': -0.08, 'sigma': 3.0},
    'L23_PV_to_L23_E': {'amplitude': -0.20, 'sigma': 1.5},
    'L23_PV_to_L23_SST': {'amplitude': -0.15, 'sigma': 1.5},
    'L23_PV_to_L23_PV': {'amplitude': -0.12, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L2/3 -> L4 connections
    # -----------------------------------------------------
    'L23_E_to_L4_E': {'amplitude': 0.20, 'sigma': 2.5},
    'L23_E_to_L4_SST': {'amplitude': 0.10, 'sigma': 2.5},
    'L23_E_to_L4_PV': {'amplitude': 0.12, 'sigma': 2.5},
    'L23_SST_to_L4_E': {'amplitude': -0.10, 'sigma': 3.0},
    'L23_SST_to_L4_PV': {'amplitude': -0.05, 'sigma': 3.0},
    'L23_PV_to_L4_E': {'amplitude': -0.15, 'sigma': 1.5},
    'L23_PV_to_L4_SST': {'amplitude': -0.10, 'sigma': 1.5},
    'L23_PV_to_L4_PV': {'amplitude': -0.08, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L2/3 -> L5 connections
    # -----------------------------------------------------
    'L23_E_to_L5_E': {'amplitude': 0.25, 'sigma': 3.0},
    'L23_E_to_L5_SST': {'amplitude': 0.12, 'sigma': 3.0},
    'L23_E_to_L5_PV': {'amplitude': 0.15, 'sigma': 3.0},
    'L23_SST_to_L5_E': {'amplitude': -0.12, 'sigma': 3.0},
    'L23_SST_to_L5_PV': {'amplitude': -0.06, 'sigma': 3.0},
    'L23_PV_to_L5_E': {'amplitude': -0.15, 'sigma': 1.5},
    'L23_PV_to_L5_SST': {'amplitude': -0.10, 'sigma': 1.5},
    'L23_PV_to_L5_PV': {'amplitude': -0.08, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L2/3 connections
    # -----------------------------------------------------
    'L4_E_to_L23_E': {'amplitude': 0.20, 'sigma': 2.5},
    'L4_E_to_L23_SST': {'amplitude': 0.10, 'sigma': 2.5},
    'L4_E_to_L23_PV': {'amplitude': 0.12, 'sigma': 2.5},
    'L4_SST_to_L23_E': {'amplitude': -0.12, 'sigma': 3.0},
    'L4_SST_to_L23_PV': {'amplitude': -0.06, 'sigma': 3.0},
    'L4_PV_to_L23_E': {'amplitude': -0.18, 'sigma': 1.5},
    'L4_PV_to_L23_SST': {'amplitude': -0.12, 'sigma': 1.5},
    'L4_PV_to_L23_PV': {'amplitude': -0.10, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L4 connections
    # -----------------------------------------------------
    'L4_E_to_L4_E': {'amplitude': 0.35, 'sigma': 2.0},
    'L4_E_to_L4_SST': {'amplitude': 0.15, 'sigma': 2.0},
    'L4_E_to_L4_PV': {'amplitude': 0.20, 'sigma': 2.0},
    'L4_SST_to_L4_E': {'amplitude': -0.15, 'sigma': 3.0},
    'L4_SST_to_L4_PV': {'amplitude': -0.08, 'sigma': 3.0},
    'L4_PV_to_L4_E': {'amplitude': -0.20, 'sigma': 1.5},
    'L4_PV_to_L4_SST': {'amplitude': -0.15, 'sigma': 1.5},
    'L4_PV_to_L4_PV': {'amplitude': -0.12, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L4 -> L5 connections
    # -----------------------------------------------------
    'L4_E_to_L5_E': {'amplitude': 0.18, 'sigma': 2.5},
    'L4_E_to_L5_SST': {'amplitude': 0.09, 'sigma': 2.5},
    'L4_E_to_L5_PV': {'amplitude': 0.12, 'sigma': 2.5},
    'L4_SST_to_L5_E': {'amplitude': -0.10, 'sigma': 3.0},
    'L4_SST_to_L5_PV': {'amplitude': -0.05, 'sigma': 3.0},
    'L4_PV_to_L5_E': {'amplitude': -0.15, 'sigma': 1.5},
    'L4_PV_to_L5_SST': {'amplitude': -0.10, 'sigma': 1.5},
    'L4_PV_to_L5_PV': {'amplitude': -0.08, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L2/3 connections
    # -----------------------------------------------------
    'L5_E_to_L23_E': {'amplitude': 0.20, 'sigma': 3.0},
    'L5_E_to_L23_SST': {'amplitude': 0.10, 'sigma': 3.0},
    'L5_E_to_L23_PV': {'amplitude': 0.12, 'sigma': 3.0},
    'L5_SST_to_L23_E': {'amplitude': -0.10, 'sigma': 3.0},
    'L5_SST_to_L23_PV': {'amplitude': -0.05, 'sigma': 3.0},
    'L5_PV_to_L23_E': {'amplitude': -0.15, 'sigma': 1.5},
    'L5_PV_to_L23_SST': {'amplitude': -0.10, 'sigma': 1.5},
    'L5_PV_to_L23_PV': {'amplitude': -0.08, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L4 connections
    # -----------------------------------------------------
    'L5_E_to_L4_E': {'amplitude': 0.15, 'sigma': 3.0},
    'L5_E_to_L4_SST': {'amplitude': 0.08, 'sigma': 3.0},
    'L5_E_to_L4_PV': {'amplitude': 0.10, 'sigma': 3.0},
    'L5_SST_to_L4_E': {'amplitude': -0.10, 'sigma': 3.0},
    'L5_SST_to_L4_PV': {'amplitude': -0.05, 'sigma': 3.0},
    'L5_PV_to_L4_E': {'amplitude': -0.15, 'sigma': 1.5},
    'L5_PV_to_L4_SST': {'amplitude': -0.10, 'sigma': 1.5},
    'L5_PV_to_L4_PV': {'amplitude': -0.08, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # L5 -> L5 connections
    # -----------------------------------------------------
    'L5_E_to_L5_E': {'amplitude': 0.35, 'sigma': 2.0},
    'L5_E_to_L5_SST': {'amplitude': 0.15, 'sigma': 2.0},
    'L5_E_to_L5_PV': {'amplitude': 0.20, 'sigma': 2.0},
    'L5_SST_to_L5_E': {'amplitude': -0.15, 'sigma': 3.0},
    'L5_SST_to_L5_PV': {'amplitude': -0.08, 'sigma': 3.0},
    'L5_PV_to_L5_E': {'amplitude': -0.20, 'sigma': 1.5},
    'L5_PV_to_L5_SST': {'amplitude': -0.15, 'sigma': 1.5},
    'L5_PV_to_L5_PV': {'amplitude': -0.12, 'sigma': 1.5},
    
    # -----------------------------------------------------
    # Thalamic connections to all layers
    # -----------------------------------------------------
    'thalamus_to_L23_E': {'amplitude': 0.25, 'sigma': 2.0},
    'thalamus_to_L23_SST': {'amplitude': -0.10, 'sigma': 2.0},
    'thalamus_to_L23_PV': {'amplitude': -0.12, 'sigma': 2.0},
    
    'thalamus_to_L4_E': {'amplitude': 0.35, 'sigma': 2.0},
    'thalamus_to_L4_SST': {'amplitude': -0.15, 'sigma': 2.0},
    'thalamus_to_L4_PV': {'amplitude': -0.18, 'sigma': 2.0},
    
    'thalamus_to_L5_E': {'amplitude': 0.30, 'sigma': 2.0},
    'thalamus_to_L5_SST': {'amplitude': -0.12, 'sigma': 2.0},
    'thalamus_to_L5_PV': {'amplitude': -0.15, 'sigma': 2.0}
}

# Keep the old format for backward compatibility
CONNECTIVITY_PARAMS = {
    # Excitatory connections
    'E_to_E': {'amplitude': 0.2, 'sigma': 2.0},
    'E_to_SST': {'amplitude': 0.1, 'sigma': 2.0},
    'E_to_PV': {'amplitude': 0.15, 'sigma': 2.0},
    
    # Inhibitory connections
    'SST_to_E': {'amplitude': -0.1, 'sigma': 3.0},
    'SST_to_PV': {'amplitude': -0.05, 'sigma': 3.0},
    'PV_to_E': {'amplitude': -0.15, 'sigma': 1.5},
    'PV_to_SST': {'amplitude': -0.1, 'sigma': 1.5},
    'PV_to_PV': {'amplitude': -0.1, 'sigma': 1.5},
    
    # Thalamic inputs
    'thalamus_to_E': {'amplitude': 0.15, 'sigma': 2.0},
    'thalamus_to_SST': {'amplitude': 0.1, 'sigma': 2.0},
    'thalamus_to_PV': {'amplitude': 0.12, 'sigma': 2.0}
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

# Visualization settings
UPDATE_INTERVAL = 60  # Time between visualization updates (ms) - DECREASE for more frequent visual updates
COLORMAPS = {
    'E': [[0, 'black'], [1, 'blue']],
    'SST': [[0, 'black'], [1, 'orange']],
    'PV': [[0, 'black'], [1, 'red']],
    'thalamus': [[0, 'black'], [1, 'white']]
} 