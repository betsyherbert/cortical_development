"""Configuration parameters for stability analysis.

All constants for both analysis and visualization live here.
Import from here, not from visualizer.py or other modules.
"""

from src.analysis.common import (
    DOUBLE_COLUMN_WIDTH_MM,
    FIGURE_FONT_SIZES_PT,
    compute_figsize_inches,
    get_output_dir,
)
from src.model.config import CELL_TYPES, DT, LAYERS

# Re-export for convenience (single import point)
__all__ = [
    "ANALYSIS_PARAMS",
    "CELL_TYPES",
    "COLORBAR_PARAMS",
    "COLORMAP",
    "CONDITIONS",
    "DT",
    "FONT_CONFIG",
    "LAYERS",
    "OUTPUT_DIR",
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
# Figure sizes are defined in mm (Nature double-column standard) and converted to inches
FONT_CONFIG = {
    "font_family": "DejaVu Sans",  # Matches centralized style
    "font_sizes": {
        "title": FIGURE_FONT_SIZES_PT["figure_title"],
        "ylabel": FIGURE_FONT_SIZES_PT["axis_label"],
        "colorbar": FIGURE_FONT_SIZES_PT["colorbar_label"],
        "colorbar_ticks": FIGURE_FONT_SIZES_PT["colorbar_tick"],
        "tick_labels": FIGURE_FONT_SIZES_PT["tick_label"],
        "condition_labels": FIGURE_FONT_SIZES_PT["axes_title"],
    },
    "figure_sizes": {
        # All sizes in inches (converted from mm for Matplotlib)
        # Widths use double-column (183 mm) or appropriate fraction
        "layer_wise": compute_figsize_inches(DOUBLE_COLUMN_WIDTH_MM, 40.0),  # Wide, short
        "column_wise": compute_figsize_inches(90.0, 120.0),  # Narrow, tall
        "effectiveness": compute_figsize_inches(90.0, 90.0),  # Square
        "effectiveness_2x2": compute_figsize_inches(DOUBLE_COLUMN_WIDTH_MM, DOUBLE_COLUMN_WIDTH_MM),  # Square, full width
        "phase_diagram": compute_figsize_inches(DOUBLE_COLUMN_WIDTH_MM, 40.0),  # Wide, short
        "regime_percentages": compute_figsize_inches(120.0, 120.0),  # Square, medium
        "heatmap_single": compute_figsize_inches(90.0, 70.0),  # Medium rectangle
        "heatmap_dual": compute_figsize_inches(120.0, 60.0),  # Wide rectangle
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
