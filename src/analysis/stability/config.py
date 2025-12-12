"""Configuration parameters for stability analysis.

All constants for both analysis and visualization live here.
Import from here, not from visualizer.py or other modules.
"""

from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS, get_output_dir
from src.model.config import CELL_TYPES, DT, LAYERS

# Re-export for convenience (single import point)
__all__ = [
    "ANALYSIS_PARAMS",
    "CELL_TYPES",
    "COLORBAR_PARAMS",
    "COLORMAP",
    "CONDITIONS",
    "DEVELOPMENTAL_STAGES",
    "DT",
    "FONT_CONFIG",
    "LAYERS",
    "OUTPUT_DIR",
    "PRESETS",
    "REGIMES",
    "REGIME_COLORS",
    "REGIME_LABELS",
    "STANDARD_COLORS",
]

# =============================================================================
# Analysis Parameters
# =============================================================================

ANALYSIS_PARAMS = {
    "duration": 10.0,  # Simulation duration for snapshot selection (seconds)
    "percentiles": [10, 90],  # [Idle, Driven] thresholds
    "n_snapshots": 30,  # Number of snapshots per regime
    "layer_patch_size": 5,  # Patch size for layer-wise analysis
    "column_patch_size": 5,  # Patch size for column-wise analysis
    "boundary_exclude": 2,  # Pixels to exclude from edges
}

CONDITIONS = ["full", "e_only", "e_pv_only", "e_sst_only"]
REGIMES = ["idle", "driven"]

# Output directory (use shared helper for consistency)
OUTPUT_DIR = str(get_output_dir("stability", create=False))

# =============================================================================
# Visualization Constants
# =============================================================================

COLORMAP = "RdBu_r"  # Red-blue colormap for lambda_max

COLORBAR_PARAMS = {
    "lambda_max_min": -0.2,
    "lambda_max_max": 0.2,
    "difference_max": 0.4,
}

# Regime classification colors and labels (order matters for colorbar)
REGIME_COLORS = {
    "inhibition \n destabilised": "#151515",
    "intrinsically \n unstable": "#AC1E12",
    "intrinsically \n stable": "#CCCCCC",
    "inhibition \n stabilised": "#214F7F",
}

REGIME_LABELS = [
    "inhibition \n destabilised",
    "intrinsically \n unstable",
    "intrinsically \n stable",
    "inhibition \n stabilised",
]

# Publication-quality font and figure settings
FONT_CONFIG = {
    "font_family": "Latin Modern Sans",
    "font_sizes": {
        "title": 18,
        "ylabel": 16,
        "colorbar": 14,
        "colorbar_ticks": 12,
        "tick_labels": 12,
        "condition_labels": 14,
    },
    "figure_sizes": {
        "layer_wise": (16, 4),
        "column_wise": (8, 10),
        "effectiveness": (5, 5),
        "effectiveness_2x2": (8, 8),
        "phase_diagram": (16, 4),
        "regime_percentages": (10, 10),
        "heatmap_single": (8, 6),
        "heatmap_dual": (10, 5),
    },
    "dpi": 300,
    "colorbar_width": 0.008,
}

# Standardized color schemes
STANDARD_COLORS = {
    "cell_types": {"SST": "#FF630C", "PV": "#D91B12"},
    "layers": {
        "L5": "#999999",
        "L4": "#555555",
        "L23": "#222222",
        "L2/3": "#222222",
    },
    "colormaps": {"heatmap": "Reds", "diverging": "RdBu_r", "sequential": "viridis"},
}
