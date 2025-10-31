"""Bifurcation analysis module for cortical circuit stability.

This module implements Figure 5A-style bifurcation analysis using Fourier-space
linearization around spatially uniform fixed points. The analysis sweeps parameters
like inhibitory time constants and connection widths to map stability boundaries
and dominant spatial frequencies of unstable modes.
"""

from .config import (
    BIFURCATION_PARAMS,
    PARAMETER_RANGES,
    FOURIER_GRID_PARAMS,
    NUMERICAL_TOLERANCES,
    VALIDATION_SETTINGS
)

from .fourier_analysis import (
    GaussianKernelValidator,
    GaussianFourierTransform,
    ConnectivityNormalizer,
    ConnectionMatrixBuilder,
    validate_fourier_analysis_setup
)

from .fixed_point_solver import (
    FixedPointResult,
    ExternalInputCalculator,
    ReLUProcessor,
    HybridFixedPointSolver,
    validate_fixed_point_solver
)

from .jacobian_builder import (
    JacobianData,
    PerModeJacobianBuilder,
    ModeGridProcessor,
    validate_jacobian_builder
)

from .eigenvalue_analysis import (
    StabilityRegime,
    EigenvalueAnalysisResult,
    EigenvalueAnalyzer,
    BifurcationPointClassifier,
    validate_eigenvalue_analysis
)

from .parameter_sweeper import (
    ParameterPoint,
    BifurcationAnalysisResult,
    ParameterApplicator,
    BifurcationAnalyzer,
    validate_parameter_sweeper
)

from .validation import (
    ValidationReport,
    DCGainValidator,
    MathematicalConsistencyValidator,
    ParameterContinuityValidator,
    CrossValidator,
    BifurcationResultsValidator,
    validate_bifurcation_results,
    load_and_validate_results
)

from .visualizer import (
    BifurcationPlotter,
    BifurcationVisualizationSuite,
    create_publication_plots,
    plot_multiple_analyses,
    load_and_visualize_results
)

from .run_analysis import (
    IntegrationTester,
    run_production_analysis,
    run_comparison_analysis
)

__all__ = [
    'BIFURCATION_PARAMS',
    'PARAMETER_RANGES',
    'FOURIER_GRID_PARAMS', 
    'NUMERICAL_TOLERANCES',
    'VALIDATION_SETTINGS',
    'GaussianKernelValidator',
    'GaussianFourierTransform',
    'ConnectivityNormalizer',
    'ConnectionMatrixBuilder',
    'validate_fourier_analysis_setup',
    'FixedPointResult',
    'ExternalInputCalculator',
    'ReLUProcessor',
    'HybridFixedPointSolver',
    'validate_fixed_point_solver',
    'JacobianData',
    'PerModeJacobianBuilder',
    'ModeGridProcessor',
    'validate_jacobian_builder',
    'StabilityRegime',
    'EigenvalueAnalysisResult',
    'EigenvalueAnalyzer',
    'BifurcationPointClassifier',
    'validate_eigenvalue_analysis',
    'ParameterPoint',
    'BifurcationAnalysisResult',
    'ParameterApplicator',
    'BifurcationAnalyzer',
    'validate_parameter_sweeper',
    'ValidationReport',
    'DCGainValidator',
    'MathematicalConsistencyValidator',
    'ParameterContinuityValidator',
    'CrossValidator',
    'BifurcationResultsValidator',
    'validate_bifurcation_results',
    'load_and_validate_results',
    'BifurcationPlotter',
    'BifurcationVisualizationSuite',
    'create_publication_plots',
    'plot_multiple_analyses',
    'load_and_visualize_results',
    'IntegrationTester',
    'run_production_analysis',
    'run_comparison_analysis'
]
