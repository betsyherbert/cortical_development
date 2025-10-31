"""Configuration parameters for stability analysis."""

from src.model.config import CELL_TYPES, LAYERS, DT
from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS, DPI

# Analysis parameters
ANALYSIS_PARAMS = {
    'duration': 10.0,              # Simulation duration for snapshot selection (seconds)
    'percentiles': [10, 90],       # [Idle, Driven] thresholds
    'n_snapshots': 30,             # Number of snapshots per regime
    'layer_patch_size': 5,         # Patch size for layer-wise analysis
    'column_patch_size': 5,        # Patch size for column-wise analysis
    'boundary_exclude': 2,         # Pixels to exclude from edges
}

# Analysis conditions and regimes
CONDITIONS = ['full', 'e_only', 'e_pv_only', 'e_sst_only']  # Updated to reflect actual conditions
REGIMES = ['idle', 'driven']

# Output - standardized relative path from project root
OUTPUT_DIR = 'outputs/stability'

# Visualization
COLORMAP = 'RdBu_r'  # Red-blue colormap for lambda_max (red=unstable, blue=stable)
FIGSIZE_LAYER = (24, 9)  # Size for layer-wise plots (4x12, including thalamic input row)
FIGSIZE_COLUMN = (10, 12)  # Size for column-wise plots (4x4, including thalamic input row)

# Colorbar ranges
COLORBAR_PARAMS = {
    'lambda_max_min': -0.2,    # Minimum value for lambda_max colorbar (blue = most stable)
    'lambda_max_max': 0.2,     # Maximum value for lambda_max colorbar (red = most unstable)  
    'difference_max': 0.4,     # Maximum absolute value for difference plots (symmetric around 0)
} 