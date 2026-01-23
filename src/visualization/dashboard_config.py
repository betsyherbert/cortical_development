"""Centralized constants for the cortical simulation dashboard.

This module contains all dashboard-specific configuration values.
Import these constants instead of duplicating them across modules.
"""

from src.analysis.descriptive.config import ANALYSIS_PARAMS
from src.model.config import UPDATE_INTERVAL

# =============================================================================
# Spectrum/Analysis Update Intervals
# =============================================================================

# How often to update stability/gain spectra (milliseconds)
SPECTRUM_INTERVAL_MS = 1000

# =============================================================================
# Correlation Tracking
# =============================================================================

# Rolling window size for correlation computation (milliseconds)
CORRELATION_WINDOW_MS = 10000

# Display window for correlation time series (seconds)
CORRELATION_DISPLAY_SECONDS = 10

# Update correlation every N frames to reduce computational load
CORRELATION_UPDATE_INTERVAL = 20

# Sample every Nth cell to reduce correlation matrix size
CORRELATION_CELL_SAMPLE_RATE = 4

# History buffer length for correlation (computed from window and update interval)
CORRELATION_HISTORY_LENGTH = int(CORRELATION_WINDOW_MS / UPDATE_INTERVAL)

# =============================================================================
# Synchronous Event Detection
# =============================================================================

# Import thresholds from descriptive analysis for consistency
# Changes to these values in descriptive/config.py will apply to dashboard
ACTIVITY_THRESHOLD = ANALYSIS_PARAMS["activity_threshold"]
SYNCHRONOUS_EVENT_THRESHOLD = ANALYSIS_PARAMS["synchronous_event_threshold"]
min_mean_rate = ANALYSIS_PARAMS["min_mean_rate"]

