"""Descriptive analysis module for cortical network activity patterns.

This module provides comprehensive analysis and visualization of cortical circuit
activity across developmental stages (P0, P5, P10, P15).

Main exports:
- DescriptiveAnalysis: core analysis class
- DescriptivePipeline: orchestration class for running analysis
- ActivityVisualizer: plotting class
- run_analysis.main(): CLI entrypoint
"""

from .activity_analysis import DescriptiveAnalysis
from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS
from .config import (
    ANALYSIS_PARAMS,
    FONT_SIZES,
    LAYER_COLORS,
    POSTER_CELL_TYPES,
)
from .run_analysis import DescriptivePipeline
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
    "DescriptivePipeline",
]
