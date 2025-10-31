"""Configuration parameters for perturbation analysis."""

from src.model.config import CELL_TYPES, LAYERS, RANDOM_SEED
from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS, ERROR_BAR_ALPHA, LINE_WIDTH, MARKER_SIZE, SEM_FACTOR, FIGSIZE_TRENDS, DPI

# Core analysis parameters
ANALYSIS_PARAMS = {
    # Simulation parameters
    'duration': 5.0,                    # Duration for snapshot collection (seconds)
    'percentiles': [10, 90],            # [Idle, Driven] thresholds for thalamic activity
    'n_snapshots': 1,                   # Number of snapshots per regime
    
    # Spatial parameters 
    'layer_patch_size': 2,              # Small patch size for strongest effects (was 6)
    'column_patch_size': 2,             # Small patch size for strongest effects (was 6)
    'boundary_exclude': 1,              # Pixels to exclude from edges
    'target_region_size': 4,            # Size of region to find max/min thalamic activity
    
    # Perturbation parameters 
    'perturbation_amplitude': 1.5,      # Low amplitude optimal for strongest paradoxical effects (was 1.0)
    'perturbation_duration': 20.0,      # Optimal duration for clear effects (was 20.0)
    'post_perturbation_delay': 20.0,    # Delay for network response to develop (was 30.0)
    'measurement_window': 100.0,        # Optimal measurement window for stable signals
}

# Analysis setup
PERTURBATION_TYPES = ['SST', 'PV', 'both']
REGIMES = ['driven', 'idle']

# Output
OUTPUT_DIR = 'outputs/perturbation'

# Visualization  
COLORBAR_PARAMS = {'response_min': -0.01, 'response_max': 0.01}  # Expanded range based on diagnostic test 

# Developmental trend plot styling (matching descriptive analysis)
FONT_SIZES_TRENDS = {
    'title': 14,
    'ylabel': 11,
    'xlabel': 11,
    'tick_labels': 9
}

# Perturbation cell types for analysis (SST, PV only - exclude both)
PERTURBATION_CELL_TYPES = ['SST', 'PV'] 