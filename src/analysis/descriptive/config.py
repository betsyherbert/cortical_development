"""Configuration parameters for descriptive analysis.

Note:
    Developmental stage selection and preset mappings are centralized in
    `src.analysis.common` (DEVELOPMENTAL_STAGES, PRESETS). This module keeps
    only descriptive-analysis-specific constants.
"""

from src.analysis.common import (
    DOUBLE_COLUMN_WIDTH_MM,
    DPI,
    ERROR_BAR_ALPHA,
    FIGURE_FONT_SIZES_PT,
    LINE_WIDTH,
    MARKER_SIZE,
    SEM_FACTOR,
    compute_figsize_inches,
    get_output_dir,
)
from src.model.config import CELL_COLORS, CELL_TYPES, LAYER_COLORS, LAYERS

# Re-export for convenience (single import point)
__all__ = [
    "ANALYSIS_PARAMS",
    "AVERAGE_FIRING_RATE_YLIM",
    "CELL_COLORS",
    "CELL_TYPES",
    "DPI",
    "ERROR_BAR_ALPHA",
    "FIGSIZE_TIMESERIES",
    "FIGSIZE_TRENDS",
    "FONT_SIZES",
    "HEATMAP_VMAX",
    "HEATMAP_VMIN",
    "LAYERS",
    "LAYER_COLORS",
    "LINE_WIDTH",
    "MARKER_SIZE",
    "OUTPUT_DIR",
    "POSTER_CELL_TYPES",
    "SEM_FACTOR",
    "SUBPLOT_PADDING",
    "Y_MARGIN_FACTOR",
]

# Output directory (use shared helper for consistency)
OUTPUT_DIR = str(get_output_dir("descriptive", create=False))

# Analysis parameters
ANALYSIS_PARAMS = {
    "warmup_duration": 2.0,  # Warmup time before data collection (seconds)
    "simulation_duration": 10.0,  # Simulation duration (seconds)
    "activity_threshold": 0.25,  # Firing rate threshold for "active" cells
    "synchronous_event_threshold": 0.02,  # Fraction of cells for "large synchronous events"
    "min_mean_rate": 10e-5,  # Min mean firing rate to include in analysis (correlation and dimensionality calculations)
    "sampling_interval": 20.0,  # Sampling interval (ms)
    "dimensionality_min_variance": 0.001,  # Min eigenvalue to include in dimensionality calculation
    "spatial_correlation_bins": 20,  # Number of distance bins for C(r) spatial correlation
}

# Visualization constants (mm-based, Nature double-column standard)
# Timeseries: wide figure for 3x4 subplot layout
FIGSIZE_TIMESERIES = compute_figsize_inches(DOUBLE_COLUMN_WIDTH_MM, 60.0)
# Trends: narrower figure for 1x3 subplot layout
FIGSIZE_TRENDS = compute_figsize_inches(DOUBLE_COLUMN_WIDTH_MM, 50.0)

# Font sizes for poster format - optimized for readability
FONT_SIZES = {
    "title": FIGURE_FONT_SIZES_PT["figure_title"],
    "subtitle": FIGURE_FONT_SIZES_PT["axes_title"],
    "ylabel": FIGURE_FONT_SIZES_PT["axis_label"],
    "xlabel": FIGURE_FONT_SIZES_PT["axis_label"],
    "tick_labels": FIGURE_FONT_SIZES_PT["tick_label"],
}

# Poster layout configuration
POSTER_CELL_TYPES = ["SST", "E", "PV"]  # Reordered for poster layout

# Heatmap visualization settings
HEATMAP_VMIN = 0
HEATMAP_VMAX = 0.5

# Plot layout constants
SUBPLOT_PADDING = {
    "hspace": 0.15,
    "wspace": 0.15,
    "right_margin": 0.85,
    "colorbar_width": 0.02,
    "colorbar_padding": 0.02,
}

# Statistical visualization constants
Y_MARGIN_FACTOR = 0.05  # Margin factor for y-axis limits

# Plot-specific y-axis limits (set to None for auto-scaling)
AVERAGE_FIRING_RATE_YLIM = [0, 1.0]  # [min, max] for average firing rate plots
