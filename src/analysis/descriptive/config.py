"""Configuration parameters for descriptive analysis.

Note:
    Developmental stage selection and preset mappings are centralized in
    `src.analysis.common` (DEVELOPMENTAL_STAGES, PRESETS). This module keeps
    only descriptive-analysis-specific constants.
"""

from src.analysis.common import (
    DPI,
    ERROR_BAR_ALPHA,
    FIGSIZE_TRENDS,
    LINE_WIDTH,
    MARKER_SIZE,
    SEM_FACTOR,
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
    "activity_threshold": 0.1,  # Firing rate threshold for "active" cells
    "synchronous_event_threshold": 0.1,  # Fraction of cells for "large synchronous events"
    "correlation_activity_threshold": 0.00,  # Min network activity for correlation calculation
    "sampling_interval": 20.0,  # Sampling interval (ms)
}

# Visualization constants
FIGSIZE_TIMESERIES = (10, 4)  # Optimized for 3x4 subplot layout (wider for better readability)

# Font sizes for poster format - optimized for readability
FONT_SIZES = {"title": 14, "subtitle": 12, "ylabel": 11, "xlabel": 11, "tick_labels": 9}

# Poster layout configuration
POSTER_CELL_TYPES = ["SST", "E", "PV"]  # Reordered for poster layout

# Heatmap visualization settings
HEATMAP_VMIN = 0
HEATMAP_VMAX = 1.0

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
