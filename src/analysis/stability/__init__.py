"""Stability analysis module for cortical circuit dynamics.

Main exports:
- StabilityAnalysis: core analysis class
- StabilityVisualizer: plotting class
- run_analysis.main(): CLI entrypoint
"""

from .stability_analysis import StabilityAnalysis
from .visualizer import StabilityVisualizer

__all__ = ["StabilityAnalysis", "StabilityVisualizer"]
