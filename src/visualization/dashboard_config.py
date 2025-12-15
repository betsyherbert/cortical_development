"""Centralized constants for the cortical simulation dashboard.

This module contains all dashboard-specific configuration values.
Import these constants instead of duplicating them across modules.
"""

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

# Minimum activity level to count a cell as active
ACTIVITY_THRESHOLD = 0.1

# Fraction of cells that must be active for a synchronous event
SYNCHRONOUS_EVENT_THRESHOLD = 0.1

