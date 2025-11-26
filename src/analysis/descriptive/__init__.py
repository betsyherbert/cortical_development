"""Descriptive analysis module for cortical network activity patterns.

This module provides comprehensive analysis and visualization of cortical circuit
activity across developmental stages (P0, P5, P10, P15).
"""

from .activity_analysis import DescriptiveAnalysis
from .visualizer import ActivityVisualizer
from .config import (
    ANALYSIS_PARAMS, 
    DEVELOPMENTAL_STAGES, 
    PRESETS,
    POSTER_CELL_TYPES,
    LAYER_COLORS,
    FONT_SIZES
)
from .run_analysis import run_descriptive_analysis

__all__ = [
    'DescriptiveAnalysis',
    'ActivityVisualizer', 
    'run_descriptive_analysis',
    'ANALYSIS_PARAMS',
    'DEVELOPMENTAL_STAGES',
    'PRESETS',
    'POSTER_CELL_TYPES',
    'LAYER_COLORS',
    'FONT_SIZES'
] 