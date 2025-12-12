"""Descriptive analysis module for cortical network activity patterns.

This module provides comprehensive analysis and visualization of cortical circuit
activity across developmental stages (P0, P5, P10, P15).
"""

from .activity_analysis import DescriptiveAnalysis
from .config import (
    ANALYSIS_PARAMS,
    DEVELOPMENTAL_STAGES,
    FONT_SIZES,
    LAYER_COLORS,
    POSTER_CELL_TYPES,
    PRESETS,
)
from .run_analysis import run_descriptive_analysis
from .visualizer import ActivityVisualizer

__all__ = [
    "ANALYSIS_PARAMS",
    "DEVELOPMENTAL_STAGES",
    "FONT_SIZES",
    "LAYER_COLORS",
    "POSTER_CELL_TYPES",
    "PRESETS",
    "ActivityVisualizer",
    "DescriptiveAnalysis",
    "run_descriptive_analysis",
]
