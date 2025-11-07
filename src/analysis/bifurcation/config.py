"""Configuration parameters for bifurcation analysis."""

from src.model.config import CELL_TYPES, LAYERS, DT, CELL_COLORS
from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS, DPI

# Analysis parameters
ANALYSIS_PARAMS = {
    'n_modes': 10,                  # Number of Fourier modes to scan in each direction (increased to check boundary-locking)
    'tolerance': 1e-6,              # Convergence tolerance for steady state finder
    'max_iters': 1000,              # Maximum iterations for steady state finder
    'grid_size': 20,                # Standard grid size for spatial scaling
    'domain_length': 20.0,          # Domain length (must equal grid_size): σ values in presets are in grid cells, normalize by grid_size to get physical units
}

# Parameter sanity check: domain_length must equal grid_size for consistent spatial units
assert ANALYSIS_PARAMS['domain_length'] == ANALYSIS_PARAMS['grid_size'], \
    f"domain_length ({ANALYSIS_PARAMS['domain_length']}) must equal grid_size ({ANALYSIS_PARAMS['grid_size']}) for consistent spatial units"

# Analysis mode options
ANALYSIS_MODES = ['silent', 'driven', 'both']
DEFAULT_ANALYSIS_MODE = 'silent'
THALAMIC_INPUT_MAGNITUDE = 3.0  # Default magnitude for driven analysis (increased to ensure activation)

# Output - standardized relative path from project root
OUTPUT_DIR = 'outputs/bifurcation'

# Visualization settings
COLORMAP = 'RdBu_r'
FIGSIZE = (10, 8)  # Standard figure size

# Developmental stage colors
STAGE_COLORS = {
    'P4': '#1e3a5f',
    'P8': '#2e5a8f',
    'P12': '#4e8abf',
    'P16': '#8ebfdf'
}

# Default layer configuration
DEFAULT_LAYERS = ['L4']
ALL_LAYERS = ['L23', 'L4', 'L5']
