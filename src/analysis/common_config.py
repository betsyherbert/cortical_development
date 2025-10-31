"""Shared configuration constants for all analysis modules."""

from src.model.presets import P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET

# Developmental stages - used by all analysis modules
DEVELOPMENTAL_STAGES = ['P4', 'P8', 'P12', 'P16']
PRESETS = {
    'P4': P4_PRESET,
    'P8': P8_PRESET, 
    'P12': P12_PRESET,
    'P16': P16_PRESET
}

# Statistical visualization constants
ERROR_BAR_ALPHA = 0.2
LINE_WIDTH = 2
MARKER_SIZE = 6
SEM_FACTOR = 0.1

# Figure settings
FIGSIZE_TRENDS = (7, 2.5)
DPI = 300

