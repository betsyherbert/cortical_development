"""Configuration parameters for bifurcation analysis.

Note: All spatial parameters (sigma, wavelength) are in μm (anatomical units).
Wavenumber k is in cycles/μm.
"""

from src.analysis.common import get_output_dir
from src.model.config import ANATOMICAL_GRID_SIZE, RANDOM_SEED

# Analysis parameters
ANALYSIS_PARAMS = {
    "n_modes": 10,  # Number of Fourier modes to scan in each direction
    "tolerance": 1e-6,  # Convergence tolerance for steady state finder
    "max_iters": 1000,  # Maximum iterations for steady state finder
    "grid_size": 20,  # Standard grid size (number of grid points) for Fourier normalization
    "anatomical_grid_size": ANATOMICAL_GRID_SIZE,  # Anatomical size in μm (default: 1000 μm)
}

# Output directory (use shared helper for consistency)
OUTPUT_DIR = str(get_output_dir("bifurcation", create=False))

# Default layer configuration
ALL_LAYERS = ["L23", "L4", "L5"]

# ============================================================================
# Parameter Specification System
# ============================================================================


class ParameterSpec:
    """Specification for a scannable parameter.

    This class defines how parameters can be scanned and visualized in
    bifurcation analyses, supporting both absolute and ratio modes.
    """

    def __init__(
        self,
        path: list[str],
        display_name: str,
        units: str = "",
        use_ratio: bool = False,
        reference_param: str | None = None,
        default_range: tuple[float, float] = (0.1, 10.0),
    ):
        """Initialize parameter specification.

        Args:
            path: Path to parameter in preset dict (e.g., ['time_constants', 'E'])
            display_name: Display name for plots (e.g., 'τ_E')
            units: Physical units (e.g., 'ms', 'grid units')
            use_ratio: If True, display as ratio relative to reference
            reference_param: Key of reference parameter for ratio mode
            default_range: Default (min, max) range for scanning
        """
        self.path = path
        self.display_name = display_name
        self.units = units
        self.use_ratio = use_ratio
        self.reference_param = reference_param
        self.default_range = default_range

    def get_axis_label(self, absolute: bool = True) -> str:
        """Get axis label for plots.

        Args:
            absolute: If True, use absolute units; if False, use ratio

        Returns:
            Formatted axis label string
        """
        if absolute or not self.use_ratio:
            if self.units:
                return f"{self.display_name} ({self.units})"
            return self.display_name

        if self.reference_param:
            ref_spec = SCANNABLE_PARAMETERS.get(self.reference_param)
            if ref_spec:
                return f"{self.display_name} / {ref_spec.display_name}"
        return f"{self.display_name} (ratio)"


SCANNABLE_PARAMETERS = {
    # Time constants
    "tau_E": ParameterSpec(
        path=["time_constants", "E"],
        display_name="τ_E",
        units="ms",
        use_ratio=False,
        default_range=(1.0, 28.0),
    ),
    "tau_SST": ParameterSpec(
        path=["time_constants", "SST"],
        display_name="τ_SST",
        units="ms",
        use_ratio=True,
        reference_param="tau_E",
        default_range=(1.0, 28.0),
    ),
    "tau_PV": ParameterSpec(
        path=["time_constants", "PV"],
        display_name="τ_PV",
        units="ms",
        use_ratio=True,
        reference_param="tau_E",
        default_range=(1.0, 28.0),
    ),
    # Spatial widths (outgoing connections) - in μm
    "sigma_E": ParameterSpec(
        path=["outgoing_widths", "E"],
        display_name="σ_E",
        units="μm",
        use_ratio=False,
        default_range=(5.0, 325.0),
    ),
    "sigma_SST": ParameterSpec(
        path=["outgoing_widths", "SST"],
        display_name="σ_SST",
        units="μm",
        use_ratio=True,
        reference_param="sigma_E",
        default_range=(5.0, 325.0),
    ),
    "sigma_PV": ParameterSpec(
        path=["outgoing_widths", "PV"],
        display_name="σ_PV",
        units="μm",
        use_ratio=True,
        reference_param="sigma_E",
        default_range=(5.0, 325.0),
    ),
    # Thalamic widths (for spectrum analysis) - in μm
    "thalamic_width_E": ParameterSpec(
        path=["thalamic_widths", "E"],
        display_name="σ_thal→E",
        units="μm",
        use_ratio=False,
        default_range=(25.0, 500.0),
    ),
    "thalamic_width_SST": ParameterSpec(
        path=["thalamic_widths", "SST"],
        display_name="σ_thal→SST",
        units="μm",
        use_ratio=False,
        default_range=(25.0, 500.0),
    ),
    "thalamic_width_PV": ParameterSpec(
        path=["thalamic_widths", "PV"],
        display_name="σ_thal→PV",
        units="μm",
        use_ratio=False,
        default_range=(25.0, 500.0),
    ),
}


# Default parameter pair configurations for 2D maps
DEFAULT_STABILITY_PAIRS = [
    ("tau_SST", "sigma_SST"),
    ("tau_PV", "sigma_PV"),
]

DEFAULT_GAIN_PAIRS = [
    ("tau_SST", "sigma_SST"),
    ("tau_PV", "sigma_PV"),
]

# Default parameters for 1D spectrum sweeps
DEFAULT_SPECTRUM_SWEEPS = [
    "tau_E",
    "tau_SST",
    "tau_PV",
    "sigma_E",
    "sigma_SST",
    "sigma_PV",
]

# ============================================================================
# Developmental Bifurcation Maps Configuration
# ============================================================================

# Parameter space scan ranges (matching dashboard limits)
TAU_MIN = 1.0  # Minimum time constant (ms)
TAU_MAX = 28.0  # Maximum time constant (ms)
SIGMA_MIN = 5.0  # Minimum spatial width (μm)
SIGMA_MAX = 325.0  # Maximum spatial width (μm), must be >= max preset σ value

# Fixed ratio mode limits (applied to τ_inh/τ_E and σ_inh/σ_E ratios)
FIXED_RATIO_TAU_MIN = 0.1  # Minimum tau ratio
FIXED_RATIO_TAU_MAX = 2.0  # Maximum tau ratio
FIXED_RATIO_SIGMA_MIN = 0.1  # Minimum sigma ratio
FIXED_RATIO_SIGMA_MAX = 4.0  # Maximum sigma ratio

# Ratio ranges for bifurcation diagrams:
#   - fixed_absolute mode: computed dynamically per stage from TAU_MIN/MAX and SIGMA_MIN/MAX
#     tau_ratio ∈ [TAU_MIN/τ_E, TAU_MAX/τ_E], sigma_ratio ∈ [SIGMA_MIN/σ_E, SIGMA_MAX/σ_E]
#   - fixed_ratio mode: uses FIXED_RATIO_* constants directly
GRID_RESOLUTION = 20  # Number of points per axis
MEAN_STATE_SEED = RANDOM_SEED  # Random seed for SteadyStateFinder reproducibility

# Visualization parameters
BIFURCATION_COLORMAP = "viridis"  # Colormap for wavenumber

# Opacity levels for stability visualization
OPACITY_STABLE_FAR = 0.3  # Alpha for max_real < -0.05 (stable, far from boundary)
OPACITY_STABLE_NEAR = 0.6  # Alpha for -0.05 ≤ max_real < 0 (stable, near boundary)
OPACITY_UNSTABLE = 1.0  # Alpha for max_real ≥ 0 (unstable)
STABILITY_THRESHOLD = -0.05  # Threshold for "near boundary" vs "far from boundary"

# ============================================================================
# Developmental Gain Maps Configuration
# ============================================================================

# Gain map visualization parameters
GAIN_COLORMAP = "viridis"  # Colormap for k values
GAIN_CLIP_MAX = 50.0  # Maximum gain for opacity (clipping only affects display, not argmax)
GAIN_OPACITY_MIN = 0.3  # Minimum opacity (low gain)
GAIN_OPACITY_MAX = 1.0  # Maximum opacity (high gain)

# ============================================================================
# Gain Spectrum Analysis Configuration
# ============================================================================

# Parameter sweep configuration
SPECTRUM_PARAM_SWEEP_RANGE = (0.2, 3.0)  # Relative to preset value (0.2x to 3x)
SPECTRUM_PARAM_RESOLUTION = 20  # Number of parameter values
SPECTRUM_K_MAX = 10.0  # Maximum k value for spectrum
SPECTRUM_COLORMAP = "inferno"  # Colormap for gain values
SPECTRUM_LOG_SCALE = True  # Use log10(gain) for colormap

# Visualization parameters
SPECTRUM_Y_MARGIN = (0.5, 2.0)  # Y-axis range per stage (relative to preset: 0.5x to 2x)
