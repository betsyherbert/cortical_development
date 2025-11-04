"""Configuration parameters for bifurcation analysis."""

from src.model.config import CELL_TYPES, LAYERS, DT, CELL_COLORS
from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS, DPI

# Analysis parameters
ANALYSIS_PARAMS = {
    'n_modes': 10,                  # Number of Fourier modes to scan in each direction
    'tolerance': 1e-6,              # Convergence tolerance for steady state finder
    'max_iters': 1000,              # Maximum iterations for steady state finder
    'grid_size': 20,                # Standard grid size for spatial scaling
}

# Output - standardized relative path from project root
OUTPUT_DIR = 'outputs/bifurcation'

# Visualization settings
COLORMAP = 'RdBu_r'  # Red-blue colormap for connection matrices
FIGSIZE_DETAIL = (14, 8)  # Size for detail plots
FIGSIZE_COMPARISON = (14, 10)  # Size for developmental comparison
FIGSIZE_EIGENVALUE = (10, 8)  # Size for eigenvalue spectrum plots
FIGSIZE_COUPLING = (10, 6)  # Size for layer coupling comparison

# Developmental stage colors (dark blue to light blue continuum)
STAGE_COLORS = {
    'P4': '#1e3a5f',   # Dark blue
    'P8': '#2e5a8f',   # Medium-dark blue
    'P12': '#4e8abf',  # Medium-light blue
    'P16': '#8ebfdf'   # Light blue
}

# Colorbar ranges
COLORBAR_PARAMS = {
    'connection_min': -0.2,
    'connection_max': 0.2,
}

# Population configuration
N_POPULATIONS_OPTIONS = [2, 3]  # 2 for E-I, 3 for E-SST-PV
DEFAULT_N_POPULATIONS = 3

# Default layer configuration
DEFAULT_LAYERS = ['L4']  # Default to single layer
ALL_LAYERS = ['L23', 'L4', 'L5']
