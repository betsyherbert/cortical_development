"""Configuration parameters for bifurcation analysis."""

from src.model.config import CELL_TYPES, LAYERS, DT, CELL_COLORS, RANDOM_SEED
from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS, DPI

# Analysis parameters
ANALYSIS_PARAMS = {
    'n_modes': 10,                  # Number of Fourier modes to scan in each direction
    'tolerance': 1e-6,              # Convergence tolerance for steady state finder
    'max_iters': 2000,              # Maximum iterations for steady state finder
    'grid_size': 20,                # Standard grid size for spatial scaling
    'domain_length': 20.0,          # Domain length (must equal grid_size): σ values in presets are in grid cells, normalize by grid_size to get physical units
}

# Parameter sanity check: domain_length must equal grid_size for consistent spatial units
assert ANALYSIS_PARAMS['domain_length'] == ANALYSIS_PARAMS['grid_size'], \
    f"domain_length ({ANALYSIS_PARAMS['domain_length']}) must equal grid_size ({ANALYSIS_PARAMS['grid_size']}) for consistent spatial units"

# Output - standardized relative path from project root
OUTPUT_DIR = 'outputs/bifurcation'

# Default layer configuration
ALL_LAYERS = ['L23', 'L4', 'L5']

# ============================================================================
# Developmental Bifurcation Maps Configuration
# ============================================================================

# Parameter space scan ranges (matching dashboard limits)
TAU_MIN = 1.0    # Minimum time constant (ms) on dashboard
TAU_MAX = 28.0   # Maximum time constant (ms) on dashboard
SIGMA_MIN = 0.1  # Minimum spatial width on dashboard
SIGMA_MAX = 6.5  # Maximum spatial width on dashboard (must be >= max preset σ value)

# Fixed ratio mode specific limits (applied to τ_inh/τ_E and σ_inh/σ_E ratios)
FIXED_RATIO_TAU_MIN = 0.1    # Minimum tau ratio to scan
FIXED_RATIO_TAU_MAX = 2.0    # Maximum tau ratio to scan
FIXED_RATIO_SIGMA_MIN = 0.1  # Minimum sigma ratio to scan
FIXED_RATIO_SIGMA_MAX = 4.0  # Maximum sigma ratio to scan

# Ratio ranges for bifurcation diagrams:
#   - fixed_absolute mode: computed dynamically per stage from TAU_MIN/MAX and SIGMA_MIN/MAX
#     tau_ratio ∈ [TAU_MIN/τ_E, TAU_MAX/τ_E], sigma_ratio ∈ [SIGMA_MIN/σ_E, SIGMA_MAX/σ_E]
#   - fixed_ratio mode: uses FIXED_RATIO_* constants directly
GRID_RESOLUTION = 20             # Number of points per axis (40×40 = 1600 points per map)
MEAN_STATE_SEED = RANDOM_SEED    # Random seed for SteadyStateFinder reproducibility

# Visualization parameters
BIFURCATION_COLORMAP = 'viridis'  # Colormap for wavenumber

# Opacity levels for stability visualization
OPACITY_STABLE_FAR = 0.3     # Alpha for max_real < -0.05 (stable, far from boundary)
OPACITY_STABLE_NEAR = 0.6    # Alpha for -0.05 ≤ max_real < 0 (stable, near boundary)
OPACITY_UNSTABLE = 1.0       # Alpha for max_real ≥ 0 (unstable)
STABILITY_THRESHOLD = -0.05  # Threshold for "near boundary" vs "far from boundary"
