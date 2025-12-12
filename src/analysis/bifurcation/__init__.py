"""Bifurcation analysis module for cortical circuit stability.

This module provides tools for analyzing network stability and gain through:
- Linear stability analysis (eigenvalue analysis)
- Gain spectrum computation (amplification of thalamic input)
- Parameter space scanning (2D maps and 1D sweeps)
- Multi-stage developmental comparison

Main components:
- core: Network models and analysis infrastructure
- stability_maps: Stability analysis with bifurcation detection
- gain_maps: Gain analysis (2D maps and 1D spectra)
- run_analysis: Pipeline orchestration
- visualizer: Figure generation
"""

from .core import NetworkModel, StabilityAnalyzer, SteadyStateFinder
from .gain_maps import compute_gain_maps_all_stages, compute_gain_spectra_all_stages
from .run_analysis import BifurcationAnalysis
from .stability_maps import compute_stability_maps_all_stages
from .visualizer import BifurcationVisualizer

__all__ = [
    "BifurcationAnalysis",
    "BifurcationVisualizer",
    "NetworkModel",
    "StabilityAnalyzer",
    "SteadyStateFinder",
    "compute_gain_maps_all_stages",
    "compute_gain_spectra_all_stages",
    "compute_stability_maps_all_stages",
]
