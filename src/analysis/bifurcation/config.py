"""Configuration parameters for bifurcation analysis.

Note: All spatial parameters (sigma, wavelength) are in μm (anatomical units).
Wavenumber k is in cycles/μm.
"""

from src.analysis.common import get_output_dir
from src.model.config import ANATOMICAL_GRID_SIZE, GRID_SIZE, RANDOM_SEED

# Analysis parameters
ANALYSIS_PARAMS = {
    "n_modes": 10,  # Number of Fourier modes to scan in each direction
    "tolerance": 1e-6,  # Convergence tolerance for steady state finder
    "max_iters": 1000,  # Maximum iterations for steady state finder
    "grid_size": GRID_SIZE,  # Inherit from simulation grid size
    "anatomical_grid_size": ANATOMICAL_GRID_SIZE,  # Anatomical size in μm
}

# Operating point configuration
THALAMIC_MAGNITUDE = 0.2  # Thalamic input magnitude for steady-state computation (used by both stability and gain maps)

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
        is_derived_ratio: bool = False,
        base_param: str | None = None,
    ):
        """Initialize parameter specification.

        Args:
            path: Path to parameter in preset dict (e.g., ['time_constants', 'E'])
            display_name: Display name for plots (e.g., 'τ_E')
            units: Physical units (e.g., 'ms', 'grid units')
            use_ratio: If True, display as ratio relative to reference
            reference_param: Key of reference parameter for ratio mode
            default_range: Default (min, max) range for scanning
            is_derived_ratio: If True, scanned value is a ratio multiplied by base_param
            base_param: Key of parameter to multiply ratio by (for derived ratio params)
        """
        self.path = path
        self.display_name = display_name
        self.units = units
        self.use_ratio = use_ratio
        self.reference_param = reference_param
        self.default_range = default_range
        self.is_derived_ratio = is_derived_ratio
        self.base_param = base_param

    def get_axis_label(self, absolute: bool = True) -> str:
        """Get axis label for plots.

        Args:
            absolute: If True, use absolute units; if False, use ratio

        Returns:
            Formatted axis label string
        """
        if self.is_derived_ratio:
            return self.display_name

        if absolute or not self.use_ratio:
            if self.units:
                return f"{self.display_name} ({self.units})"
            return self.display_name

        if self.reference_param:
            ref_spec = SCANNABLE_PARAMETERS.get(self.reference_param)
            if ref_spec:
                return f"{self.display_name} / {ref_spec.display_name}"
        return f"{self.display_name} (ratio)"


# SST-PV ratio ranges (τ_PV/τ_SST and σ_PV/σ_SST for compressed-style stability maps)
COMPRESSED_MAP_TAU_RATIO_RANGE = (0.1, 1.0)  # τ_PV/τ_SST ratio
COMPRESSED_MAP_SIGMA_RATIO_RANGE = (0.5, 1.8)  # σ_PV/σ_SST ratio

SCANNABLE_PARAMETERS = {
    # Time constants
    "tau_E": ParameterSpec(
        path=["time_constants", "E"],
        display_name=r"$\tau_{\mathrm{E}}$",
        units="ms",
        use_ratio=False,
        # Must include preset values (E: 25–50 ms across stages)
        default_range=(10.0, 70.0),
    ),
    "tau_SST": ParameterSpec(
        path=["time_constants", "SST"],
        display_name=r"$\tau_{\mathrm{SST}}$",
        units="ms",
        use_ratio=True,
        reference_param="tau_E",
        # Must include preset values (SST: 30–60 ms across stages)
        default_range=(10.0, 70.0),
    ),
    "tau_PV": ParameterSpec(
        path=["time_constants", "PV"],
        display_name=r"$\tau_{\mathrm{PV}}$",
        units="ms",
        use_ratio=True,
        reference_param="tau_E",
        # Must include preset values (PV: 12–40 ms across stages)
        default_range=(5.0, 60.0),
    ),
    # Spatial widths (outgoing connections) - in μm
    "sigma_E": ParameterSpec(
        path=["outgoing_widths", "E"],
        display_name=r"$\sigma_{\mathrm{E}}$",
        units="μm",
        use_ratio=False,
        # Must include preset values (E: 200–300 μm across stages)
        default_range=(50.0, 450.0),
    ),
    "sigma_SST": ParameterSpec(
        path=["outgoing_widths", "SST"],
        display_name=r"$\sigma_{\mathrm{SST}}$",
        units="μm",
        use_ratio=True,
        reference_param="sigma_E",
        # Must include preset values (SST: 200–400 μm across stages)
        default_range=(50.0, 450.0),
    ),
    "sigma_PV": ParameterSpec(
        path=["outgoing_widths", "PV"],
        display_name=r"$\sigma_{\mathrm{PV}}$",
        units="μm",
        use_ratio=True,
        reference_param="sigma_E",
        # Must include preset values (PV: 100–250 μm across stages)
        default_range=(50.0, 450.0),
    ),
    # Derived ratio parameters (SST-PV ratios for compressed-style maps)
    "tau_PV_over_SST": ParameterSpec(
        path=["time_constants", "PV"],
        display_name=r"$\tau_{\mathrm{PV}} / \tau_{\mathrm{SST}}$",
        units="",
        is_derived_ratio=True,
        base_param="tau_SST",
        default_range=COMPRESSED_MAP_TAU_RATIO_RANGE,
    ),
    "sigma_PV_over_SST": ParameterSpec(
        path=["outgoing_widths", "PV"],
        display_name=r"$\sigma_{\mathrm{PV}} / \sigma_{\mathrm{SST}}$",
        units="",
        is_derived_ratio=True,
        base_param="sigma_SST",
        default_range=COMPRESSED_MAP_SIGMA_RATIO_RANGE,
    ),
    # Thalamic widths (for spectrum analysis) - in μm
    "thalamic_width_E": ParameterSpec(
        path=["thalamic_widths", "E"],
        display_name=r"$\sigma_{\mathrm{thal}\to\mathrm{E}}$",
        units="μm",
        use_ratio=False,
        default_range=(25.0, 500.0),
    ),
    "thalamic_width_SST": ParameterSpec(
        path=["thalamic_widths", "SST"],
        display_name=r"$\sigma_{\mathrm{thal}\to\mathrm{SST}}$",
        units="μm",
        use_ratio=False,
        default_range=(25.0, 500.0),
    ),
    "thalamic_width_PV": ParameterSpec(
        path=["thalamic_widths", "PV"],
        display_name=r"$\sigma_{\mathrm{thal}\to\mathrm{PV}}$",
        units="μm",
        use_ratio=False,
        default_range=(25.0, 500.0),
    ),
}

# Default parameter pair configurations for 2D maps
DEFAULT_STABILITY_PAIRS = [
    ("tau_SST", "sigma_SST"),
    ("tau_PV", "sigma_PV"),
    ("sigma_E", "sigma_SST"),
    ("sigma_SST", "sigma_PV"),
    ("sigma_E", "sigma_PV"),
    ("tau_PV_over_SST", "sigma_PV_over_SST"),
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

# Parameter space scan ranges (must include all preset values across stages)
TAU_MIN = 5.0  # Minimum time constant (ms)
TAU_MAX = 70.0  # Maximum time constant (ms)
SIGMA_MIN = 50.0  # Minimum spatial width (μm)
SIGMA_MAX = 500.0  # Maximum spatial width (μm), must be >= max preset σ value

# Fixed ratio mode limits (applied to τ_inh/τ_E and σ_inh/σ_E ratios)
FIXED_RATIO_TAU_MIN = 0.1  # Minimum tau ratio
FIXED_RATIO_TAU_MAX = 1.5  # Maximum tau ratio
FIXED_RATIO_SIGMA_MIN = 0.1  # Minimum sigma ratio
FIXED_RATIO_SIGMA_MAX = 4.0  # Maximum sigma ratio

# Ratio ranges for bifurcation diagrams:
#   - fixed_absolute mode: computed dynamically per stage from TAU_MIN/MAX and SIGMA_MIN/MAX
#     tau_ratio ∈ [TAU_MIN/τ_E, TAU_MAX/τ_E], sigma_ratio ∈ [SIGMA_MIN/σ_E, SIGMA_MAX/σ_E]
#   - fixed_ratio mode: uses FIXED_RATIO_* constants directly
GRID_RESOLUTION = 50  # Number of points per axis
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
GAIN_CLIP_MIN = 5.0  # Minimum gain for opacity scale (gains below this get min opacity)
GAIN_CLIP_MAX = 500.0  # Maximum gain for opacity (clipping only affects display, not argmax)
GAIN_OPACITY_MIN = 0.05  # Minimum opacity (low gain) – low value emphasizes gain differences
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

# ============================================================================
# Maturity Index Stability Maps Configuration
# ============================================================================

# Maturity index reference values (immature=0, mature=1)
# These define the normalization bounds for each parameter component.
# Values are chosen to span the full range observed across all presets (P0-P15),
# accounting for non-monotonic trajectories (e.g., σ_SST/σ_E peaks at P5).
#
# Actual preset values:
#   P0:  τ_SST/τ_E=1.2, σ_SST/σ_E=1.5, s_SST=3.0 | τ_PV/τ_E=0.8, σ_PV/σ_E=0.83, s_PV=0.2
#   P5:  τ_SST/τ_E=1.25, σ_SST/σ_E=3.0, s_SST=4.5 | τ_PV/τ_E=0.875, σ_PV/σ_E=1.67, s_PV=0.7
#   P10: τ_SST/τ_E=1.33, σ_SST/σ_E=1.0, s_SST=4.0 | τ_PV/τ_E=0.67, σ_PV/σ_E=1.0, s_PV=3.8
#   P15: τ_SST/τ_E=1.5, σ_SST/σ_E=1.0, s_SST=4.0 | τ_PV/τ_E=0.5, σ_PV/σ_E=1.0, s_PV=4.0
MATURITY_REFERENCE_VALUES = {
    "SST": {
        # τ_SST/τ_E increases slightly: 1.2 (P0) → 1.5 (P15)
        "tau_ratio": {"immature": 1.2, "mature": 1.5},
        # σ_SST/σ_E: non-monotonic! peaks at P5 (3.0), converges to 1.0 at P10/P15
        # Use P5 as immature (furthest from mature state) and P15 as mature
        "sigma_ratio": {"immature": 3.0, "mature": 1.0},
        # s_SST: peaks at P5 (4.5), settles at 4.0; use range 3.0→4.0
        "strength": {"immature": 3.0, "mature": 4.0},
    },
    "PV": {
        # τ_PV/τ_E decreases: 0.875 (P5, slowest) → 0.5 (P15, fastest)
        "tau_ratio": {"immature": 0.875, "mature": 0.5},
        # σ_PV/σ_E: peaks at P5 (1.67), converges to 1.0 at P10/P15
        "sigma_ratio": {"immature": 1.67, "mature": 1.0},
        # s_PV: dramatic increase 0.2 (P0) → 4.0 (P15)
        "strength": {"immature": 0.2, "mature": 4.0},
    },
}

# Maturity scan configuration
MATURITY_SCAN_MARGIN = 0.5  # +/- around stage's natural maturity value
