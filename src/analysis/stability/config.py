"""Configuration parameters for stability analysis.

All constants for both analysis and visualization live here.
Import from here, not from visualizer.py or other modules.

Note:
    Styling constants (DPI, font sizes, line widths) are centralized in
    `src.analysis.common`. This module keeps only stability-analysis-specific
    constants and re-exports shared ones for convenience.
"""

from src.analysis.common import (
    DPI,
    ERROR_BAR_ALPHA,
    FIGURE_FONT_SIZES_PT,
    LINE_WIDTH,
    MARKER_SIZE,
    SUBPLOT_WIDTH_MM,
    compute_subplot_figsize,
    get_output_dir,
)
from src.model.config import CELL_COLORS, CELL_TYPES, DT, LAYER_COLORS, LAYERS

# Re-export for convenience (single import point)
__all__ = [
    "ANALYSIS_PARAMS",
    "CELL_COLORS",
    "CELL_TYPES",
    "COLORBAR_PARAMS",
    "COLORBAR_WIDTH",
    "COLORMAP",
    "COLORMAPS",
    "compute_subplot_figsize",
    "CONDITIONS",
    "DPI",
    "DT",
    "ERROR_BAR_ALPHA",
    "FONT_SIZES",
    "LAYER_COLORS",
    "LAYERS",
    "LINE_WIDTH",
    "MARKER_SIZE",
    "OUTPUT_DIR",
    "REFERENCE_LINE_WIDTH",
    "REGIMES",
    "REGIME_COLORS",
    "REGIME_LABELS",
    "SUBPLOT_ASPECTS",
    "SUBPLOT_WIDTH_MM",
]

# =============================================================================
# Analysis Parameters
# =============================================================================

ANALYSIS_PARAMS = {
    "duration": 10.0,  # Simulation duration for snapshot selection (seconds)
    "percentiles": [10, 90],  # [Idle, Driven] thresholds
    "n_snapshots": 10,  # Number of snapshots per regime
    "layer_patch_size": 3,  # Patch size for layer-wise analysis
    "column_patch_size": 3,  # Patch size for column-wise analysis
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

# Font sizes - maps local names to centralized FIGURE_FONT_SIZES_PT
# (matches pattern in descriptive/config.py for consistency)
FONT_SIZES = {
    "title": FIGURE_FONT_SIZES_PT["figure_title"],
    "ylabel": FIGURE_FONT_SIZES_PT["axis_label"],
    "colorbar": FIGURE_FONT_SIZES_PT["colorbar_label"],
    "colorbar_ticks": FIGURE_FONT_SIZES_PT["colorbar_tick"],
    "tick_labels": FIGURE_FONT_SIZES_PT["tick_label"],
    "condition_labels": FIGURE_FONT_SIZES_PT["axes_title"],
}

# Colorbar width for manual positioning (fraction of figure width)
COLORBAR_WIDTH = 0.008

# Subplot aspect ratios for different plot types
# (used with compute_subplot_figsize to derive figure dimensions)
SUBPLOT_ASPECTS = {
    "square": 1.0,  # Trend plots, phase diagrams
    "wide": 2.5,  # Timeseries, layer-wise snapshots
    "tall": 0.75,  # Column-wise snapshots (5 rows, 4 cols)
    "heatmap": 0.5,  # Heatmaps (wider than tall)
}

# Line and marker styling (single place for all stability plots)
REFERENCE_LINE_WIDTH = 1.0  # For axhline/axvline reference lines

# Colormaps for specific plot types
COLORMAPS = {
    "heatmap": "Reds",
    "diverging": "RdBu_r",
    "sequential": "viridis",
}
