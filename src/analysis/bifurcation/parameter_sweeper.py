"""Parameter space sweeper for bifurcation analysis.

This module implements the main orchestration for bifurcation analysis, sweeping
parameter spaces systematically to create Figure 5A-style bifurcation diagrams.
It coordinates fixed point solving, eigenvalue analysis, and result collection.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Callable
import time
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from src.main import CorticalSimulation
from src.analysis.common import save_with_version
from .config import (
    PARAMETER_RANGES,
    NUMERICAL_TOLERANCES
)
from .fixed_point_solver import HybridFixedPointSolver, FixedPointResult
from .jacobian_builder import PerModeJacobianBuilder
from .eigenvalue_analysis import EigenvalueAnalyzer, EigenvalueAnalysisResult, BifurcationPointClassifier


@dataclass
class ParameterPoint:
    """Represents a single point in parameter space."""
    param1_value: float  # First parameter value
    param2_value: float  # Second parameter value
    param1_name: str  # First parameter name
    param2_name: str  # Second parameter name
    grid_indices: Tuple[int, int]  # (i, j) indices in parameter grid
    
    # Analysis results (filled during sweep)
    fixed_point_result: Optional[FixedPointResult] = None
    eigenvalue_result: Optional[EigenvalueAnalysisResult] = None
    classification: Optional[Dict[str, Any]] = None
    
    # Timing and status
    analysis_time: float = 0.0
    analysis_success: bool = False
    error_message: Optional[str] = None


@dataclass
class BifurcationAnalysisResult:
    """Complete results from bifurcation analysis."""
    # Analysis configuration
    analysis_type: str  # 'pv_analysis', 'sst_analysis', etc.
    param1_name: str  # Parameter 1 name
    param2_name: str  # Parameter 2 name
    param1_range: np.ndarray  # Parameter 1 values
    param2_range: np.ndarray  # Parameter 2 values
    grid_shape: Tuple[int, int]  # (n_param1, n_param2)
    
    # Results grid
    parameter_points: List[List[ParameterPoint]]  # 2D grid of parameter points
    
    # Processed results for visualization
    stability_map: np.ndarray  # Boolean array: True = stable
    color_map: np.ndarray  # Color values for unstable regions (NaN for stable)
    eigenvalue_map: np.ndarray  # Maximum eigenvalues
    regime_map: np.ndarray  # Stability regime strings
    
    # Analysis statistics
    total_points: int
    successful_points: int
    stable_points: int
    unstable_points: int
    analysis_time: float
    
    # Metadata
    timestamp: str
    git_commit: Optional[str] = None
    analysis_parameters: Dict[str, Any] = field(default_factory=dict)


class ParameterApplicator:
    """Applies parameter changes to the cortical circuit."""
    
    def __init__(self, simulation: CorticalSimulation):
        """Initialize parameter applicator.
        
        Args:
            simulation: CorticalSimulation instance to modify
        """
        self.simulation = simulation
        self.original_params = self._save_original_parameters()
        
        # Apply P12 preset parameters for more realistic bifurcation analysis
        self._apply_p12_base_parameters()
        
    def _save_original_parameters(self) -> Dict[str, Any]:
        """Save original parameters for restoration."""
        return {
            'time_constants': self.simulation.get_time_constants().copy(),
            'gains': self.simulation.get_gains().copy(),
            'connection_params': {
                key: params.copy() 
                for key, params in self.simulation.circuit.connectivity.layer_params.items()
            },
            'strength_scaling': self.simulation.circuit.connectivity.strength_scaling.copy()
        }
    
    def apply_parameters(self, param_point: ParameterPoint, analysis_config: Dict[str, Any]):
        """Apply parameter values to the circuit.
        
        Args:
            param_point: Parameter point to apply
            analysis_config: Analysis configuration
        """
        analysis_type = analysis_config['analysis_type']
        
        if analysis_type == 'pv_analysis':
            self._apply_pv_parameters(param_point, analysis_config)
        elif analysis_type == 'sst_analysis':
            self._apply_sst_parameters(param_point, analysis_config)
        elif analysis_type == 'strength_width_analysis':
            self._apply_strength_width_parameters(param_point, analysis_config)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def _apply_p12_base_parameters(self):
        """Apply P12 preset base parameters for more realistic bifurcation analysis."""
        try:
            # Import P12 preset
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from src.model.presets import P12_PRESET
            
            # Apply P12 time constants
            for cell_type, tau in P12_PRESET['time_constants'].items():
                self.simulation.set_time_constant(cell_type, tau)
            
            # Apply P12 strength scaling (this makes connections much stronger)
            for cell_type, scaling in P12_PRESET['strength_scaling'].items():
                self.simulation.circuit.connectivity.set_strength_scaling(cell_type, scaling)
            
            # Apply P12 thalamic alpha by temporarily modifying the global config
            # This ensures the fixed point solver uses the correct thalamic alpha
            import model.config as model_config
            self.original_thalamic_alpha = model_config.THALAMIC_ALPHA
            model_config.THALAMIC_ALPHA = P12_PRESET['thalamic_alpha']
            
            print(f"Applied P12 preset parameters for bifurcation analysis:")
            print(f"  Time constants: E={P12_PRESET['time_constants']['E']}, SST={P12_PRESET['time_constants']['SST']}, PV={P12_PRESET['time_constants']['PV']}")
            print(f"  Strength scaling: E={P12_PRESET['strength_scaling']['E']}, SST={P12_PRESET['strength_scaling']['SST']}, PV={P12_PRESET['strength_scaling']['PV']}")
            print(f"  Thalamic alpha: {P12_PRESET['thalamic_alpha']}")
            
        except ImportError:
            warnings.warn("Could not import P12 preset, using current parameters")
    
    def _apply_pv_parameters(self, param_point: ParameterPoint, analysis_config: Dict[str, Any]):
        """Apply PV-focused parameter changes."""
        # Get reference parameters
        tau_e_ref = analysis_config['reference_params']['tau_e']
        sigma_e_ref = analysis_config['reference_params']['sigma_e']
        
        # Compute actual parameter values
        tau_pv = param_point.param1_value * tau_e_ref  # τ_PV = ratio * τ_E
        sigma_pv = param_point.param2_value * sigma_e_ref  # σ_PV = ratio * σ_E
        
        # Apply time constant
        self.simulation.set_time_constant('PV', tau_pv)
        
        # Apply connection widths for all PV outgoing connections
        for conn_key, _ in self.simulation.circuit.connectivity.layer_params.items():
            if '_PV_to_' in conn_key and not conn_key.startswith('thalamus_'):
                # Parse connection key to get components
                parts = conn_key.split('_to_')
                source_parts = parts[0].split('_')
                target_parts = parts[1].split('_')
                
                if len(source_parts) >= 2 and len(target_parts) >= 2:
                    source_layer, source_cell = source_parts[0], source_parts[1]
                    target_layer, target_cell = target_parts[0], target_parts[1]
                    
                    if source_cell == 'PV':
                        self.simulation.circuit.connectivity.set_connection_sigma(
                            source_layer, source_cell, target_layer, target_cell, sigma_pv
                        )
    
    def _apply_sst_parameters(self, param_point: ParameterPoint, analysis_config: Dict[str, Any]):
        """Apply SST-focused parameter changes."""
        # Get reference parameters
        tau_e_ref = analysis_config['reference_params']['tau_e']
        sigma_e_ref = analysis_config['reference_params']['sigma_e']
        
        # Compute actual parameter values
        tau_sst = param_point.param1_value * tau_e_ref  # τ_SST = ratio * τ_E
        sigma_sst = param_point.param2_value * sigma_e_ref  # σ_SST = ratio * σ_E
        
        # Apply time constant
        self.simulation.set_time_constant('SST', tau_sst)
        
        # Apply connection widths for all SST outgoing connections
        for conn_key, _ in self.simulation.circuit.connectivity.layer_params.items():
            if '_SST_to_' in conn_key and not conn_key.startswith('thalamus_'):
                # Parse connection key to get components
                parts = conn_key.split('_to_')
                source_parts = parts[0].split('_')
                target_parts = parts[1].split('_')
                
                if len(source_parts) >= 2 and len(target_parts) >= 2:
                    source_layer, source_cell = source_parts[0], source_parts[1]
                    target_layer, target_cell = target_parts[0], target_parts[1]
                    
                    if source_cell == 'SST':
                        self.simulation.circuit.connectivity.set_connection_sigma(
                            source_layer, source_cell, target_layer, target_cell, sigma_sst
                        )
    
    def _apply_strength_width_parameters(self, param_point: ParameterPoint, analysis_config: Dict[str, Any]):
        """Apply connection strength vs width parameter changes."""
        # Get reference parameters
        sigma_e_ref = analysis_config['reference_params']['sigma_e']
        
        # Parameter 1: inhibitory connection strength (absolute)
        inh_strength = param_point.param1_value
        
        # Parameter 2: inhibitory width ratio
        sigma_inh = param_point.param2_value * sigma_e_ref
        
        # Apply to both SST and PV connections
        for cell_type in ['SST', 'PV']:
            # Set connection strengths (negative for inhibitory)
            for conn_key, _ in self.simulation.circuit.connectivity.layer_params.items():
                if f'_{cell_type}_to_' in conn_key and not conn_key.startswith('thalamus_'):
                    parts = conn_key.split('_to_')
                    source_parts = parts[0].split('_')
                    target_parts = parts[1].split('_')
                    
                    if len(source_parts) >= 2 and len(target_parts) >= 2:
                        source_layer, source_cell = source_parts[0], source_parts[1]
                        target_layer, target_cell = target_parts[0], target_parts[1]
                        
                        if source_cell == cell_type:
                            # Set strength (negative for inhibitory)
                            self.simulation.circuit.connectivity.set_connection_strength(
                                source_layer, source_cell, target_layer, target_cell, -inh_strength
                            )
                            # Set width
                            self.simulation.circuit.connectivity.set_connection_sigma(
                                source_layer, source_cell, target_layer, target_cell, sigma_inh
                            )
    
    def restore_original_parameters(self):
        """Restore original parameters."""
        # Restore time constants
        for cell_type, tau in self.original_params['time_constants'].items():
            self.simulation.set_time_constant(cell_type, tau)
        
        # Restore gains
        for cell_type, gain in self.original_params['gains'].items():
            self.simulation.set_gain(cell_type, gain)
        
        # Restore connection parameters
        for conn_key, params in self.original_params['connection_params'].items():
            self.simulation.circuit.connectivity.layer_params[conn_key] = params.copy()
        
        # Restore strength scaling
        for cell_type, scaling in self.original_params['strength_scaling'].items():
            self.simulation.circuit.connectivity.strength_scaling[cell_type] = scaling
        
        # Restore thalamic alpha if it was modified
        if hasattr(self, 'original_thalamic_alpha'):
            import model.config as model_config
            model_config.THALAMIC_ALPHA = self.original_thalamic_alpha
        
        # Update weight matrices
        self.simulation.circuit.connectivity.update_weights()


class BifurcationAnalyzer:
    """Main class for orchestrating bifurcation analysis."""
    
    def __init__(self, simulation: Optional[CorticalSimulation] = None):
        """Initialize bifurcation analyzer.
        
        Args:
            simulation: Optional CorticalSimulation instance (creates new if None)
        """
        self.simulation = simulation or CorticalSimulation()
        self.parameter_applicator = ParameterApplicator(self.simulation)
        
        # Analysis components (initialized per analysis)
        self.fixed_point_solver = None
        self.jacobian_builder = None
        self.eigenvalue_analyzer = None
        self.bifurcation_classifier = None
        
        # Progress tracking
        self.progress_callback = None
        self.current_analysis = None
        
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress reporting.
        
        Args:
            callback: Function called with (current, total, status_message)
        """
        self.progress_callback = callback
    
    def _report_progress(self, current: int, total: int, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total, message)
        else:
            print(f"[{current}/{total}] {message}")
    
    def run_analysis(self, analysis_type: str, 
                    grid_resolution: Optional[Tuple[int, int]] = None,
                    fourier_grid_size: int = 20,
                    save_results: bool = True,
                    output_dir: Optional[str] = None) -> BifurcationAnalysisResult:
        """Run complete bifurcation analysis.
        
        Args:
            analysis_type: Type of analysis ('pv_analysis', 'sst_analysis', etc.)
            grid_resolution: Parameter grid resolution (uses config default if None)
            fourier_grid_size: Size of Fourier mode grid for eigenvalue analysis
            save_results: Whether to save results to disk
            output_dir: Output directory (uses default if None)
            
        Returns:
            BifurcationAnalysisResult with complete analysis
        """
        print(f"Starting bifurcation analysis: {analysis_type}")
        start_time = time.time()
        
        # Get analysis configuration
        analysis_config = self._get_analysis_config(analysis_type, grid_resolution)
        
        # Initialize analysis components
        self._initialize_analysis_components(fourier_grid_size)
        
        # Create parameter grid
        parameter_grid = self._create_parameter_grid(analysis_config)
        
        # Run parameter sweep
        print(f"Sweeping {parameter_grid.shape[0]}×{parameter_grid.shape[1]} parameter grid...")
        parameter_points = self._sweep_parameter_space(parameter_grid, analysis_config)
        
        # Process results
        print("Processing results...")
        processed_results = self._process_results(parameter_points, analysis_config)
        
        # Create final result object
        analysis_time = time.time() - start_time
        result = self._create_analysis_result(
            analysis_config, parameter_points, processed_results, analysis_time
        )
        
        # Save results if requested
        if save_results:
            self._save_results(result, output_dir)
        
        print(f"Analysis completed in {analysis_time:.1f} seconds")
        print(f"Success rate: {result.successful_points}/{result.total_points} ({result.successful_points/result.total_points:.1%})")
        print(f"Stable points: {result.stable_points}/{result.successful_points} ({result.stable_points/result.successful_points:.1%})")
        
        return result
    
    def _get_analysis_config(self, analysis_type: str, 
                           grid_resolution: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        """Get configuration for the specified analysis type."""
        if analysis_type not in PARAMETER_RANGES:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        config = PARAMETER_RANGES[analysis_type].copy()
        config['analysis_type'] = analysis_type
        
        # Set grid resolution
        if grid_resolution is None:
            # Use default resolution from parameter ranges
            param1_size = len(config['tau_ratio_range']) if 'tau_ratio_range' in config else len(config['strength_range'])
            param2_size = len(config['sigma_ratio_range'])
            grid_resolution = (param1_size, param2_size)
        
        config['grid_resolution'] = grid_resolution
        
        # Add reference parameters
        from .config import REFERENCE_PARAMS
        config['reference_params'] = REFERENCE_PARAMS
        
        return config
    
    def _initialize_analysis_components(self, fourier_grid_size: int):
        """Initialize analysis components."""
        # Get current circuit parameters
        time_constants = self.simulation.get_time_constants()
        gains = self.simulation.get_gains()
        
        # Initialize components
        self.fixed_point_solver = HybridFixedPointSolver(
            self.simulation.circuit.connectivity, time_constants, gains
        )
        self.jacobian_builder = PerModeJacobianBuilder(
            self.simulation.circuit.connectivity, time_constants, gains
        )
        self.eigenvalue_analyzer = EigenvalueAnalyzer(self.jacobian_builder)
        self.bifurcation_classifier = BifurcationPointClassifier()
        
        # Set Fourier grid size
        self.jacobian_builder.fourier_transform.grid_size = fourier_grid_size
    
    def _create_parameter_grid(self, analysis_config: Dict[str, Any]) -> np.ndarray:
        """Create parameter grid for the analysis."""
        grid_resolution = analysis_config['grid_resolution']
        
        if analysis_config['analysis_type'] == 'strength_width_analysis':
            param1_range = analysis_config['strength_range']
            param2_range = analysis_config['sigma_ratio_range']
        else:
            param1_range = analysis_config['tau_ratio_range']
            param2_range = analysis_config['sigma_ratio_range']
        
        # Ensure we have the right number of points
        if len(param1_range) != grid_resolution[0]:
            param1_range = np.linspace(param1_range[0], param1_range[-1], grid_resolution[0])
        if len(param2_range) != grid_resolution[1]:
            param2_range = np.linspace(param2_range[0], param2_range[-1], grid_resolution[1])
        
        # Create 2D grid
        param1_grid, param2_grid = np.meshgrid(param1_range, param2_range, indexing='ij')
        
        return np.stack([param1_grid, param2_grid], axis=-1)
    
    def _sweep_parameter_space(self, parameter_grid: np.ndarray, 
                             analysis_config: Dict[str, Any]) -> List[List[ParameterPoint]]:
        """Sweep through parameter space and analyze each point."""
        grid_shape = parameter_grid.shape[:2]
        total_points = grid_shape[0] * grid_shape[1]
        
        parameter_points = []
        current_point = 0
        
        for i in range(grid_shape[0]):
            row_points = []
            for j in range(grid_shape[1]):
                current_point += 1
                
                # Create parameter point
                param_point = ParameterPoint(
                    param1_value=parameter_grid[i, j, 0],
                    param2_value=parameter_grid[i, j, 1],
                    param1_name=analysis_config['param1_name'],
                    param2_name=analysis_config['param2_name'],
                    grid_indices=(i, j)
                )
                
                # Analyze this parameter point
                self._analyze_parameter_point(param_point, analysis_config, current_point, total_points)
                
                row_points.append(param_point)
            
            parameter_points.append(row_points)
        
        return parameter_points
    
    def _analyze_parameter_point(self, param_point: ParameterPoint, 
                                analysis_config: Dict[str, Any],
                                current_point: int, total_points: int):
        """Analyze a single parameter point."""
        start_time = time.time()
        
        try:
            # Report progress
            self._report_progress(
                current_point, total_points,
                f"Analyzing ({param_point.param1_value:.3f}, {param_point.param2_value:.3f})"
            )
            
            # Apply parameters
            self.parameter_applicator.apply_parameters(param_point, analysis_config)
            
            # Update solver with new parameters (parameters may have changed)
            time_constants = self.simulation.get_time_constants()
            gains = self.simulation.get_gains()
            
            # Create fresh solver instances for this parameter point
            solver = HybridFixedPointSolver(
                self.simulation.circuit.connectivity, time_constants, gains
            )
            jacobian_builder = PerModeJacobianBuilder(
                self.simulation.circuit.connectivity, time_constants, gains
            )
            eigenvalue_analyzer = EigenvalueAnalyzer(jacobian_builder)
            
            # Solve fixed point
            fixed_point_result = solver.solve()
            param_point.fixed_point_result = fixed_point_result
            
            if not fixed_point_result.converged:
                param_point.analysis_success = False
                param_point.error_message = f"Fixed point failed to converge (error: {fixed_point_result.final_error:.2e})"
                return
            
            # Analyze eigenvalues
            eigenvalue_result = eigenvalue_analyzer.analyze_stability(fixed_point_result)
            param_point.eigenvalue_result = eigenvalue_result
            
            # Classify for bifurcation diagram
            classification = self.bifurcation_classifier.classify_point(eigenvalue_result)
            param_point.classification = classification
            
            param_point.analysis_success = True
            
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            param_point.analysis_success = False
            param_point.error_message = str(e)
            warnings.warn(f"Analysis failed at point ({param_point.param1_value:.3f}, {param_point.param2_value:.3f}): {e}")
        
        finally:
            param_point.analysis_time = time.time() - start_time
            
            # Restore original parameters for next point
            self.parameter_applicator.restore_original_parameters()
    
    def _process_results(self, parameter_points: List[List[ParameterPoint]], 
                        analysis_config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process parameter points into visualization-ready arrays."""
        grid_shape = (len(parameter_points), len(parameter_points[0]))
        
        # Initialize arrays
        stability_map = np.zeros(grid_shape, dtype=bool)
        color_map = np.full(grid_shape, np.nan)
        eigenvalue_map = np.zeros(grid_shape)
        regime_map = np.full(grid_shape, '', dtype='<U20')
        
        # Process each point
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                param_point = parameter_points[i][j]
                
                if param_point.analysis_success and param_point.classification:
                    classification = param_point.classification
                    
                    stability_map[i, j] = classification['is_stable']
                    eigenvalue_map[i, j] = classification['max_eigenvalue']
                    regime_map[i, j] = classification['stability_regime']
                    
                    if classification['color_value'] is not None:
                        color_map[i, j] = classification['color_value']
                else:
                    # Failed analysis - mark as NaN
                    eigenvalue_map[i, j] = np.nan
                    regime_map[i, j] = 'failed'
        
        return {
            'stability_map': stability_map,
            'color_map': color_map,
            'eigenvalue_map': eigenvalue_map,
            'regime_map': regime_map
        }
    
    def _create_analysis_result(self, analysis_config: Dict[str, Any],
                              parameter_points: List[List[ParameterPoint]],
                              processed_results: Dict[str, np.ndarray],
                              analysis_time: float) -> BifurcationAnalysisResult:
        """Create final analysis result object."""
        grid_shape = (len(parameter_points), len(parameter_points[0]))
        
        # Create parameter ranges
        param1_values = [parameter_points[i][0].param1_value for i in range(grid_shape[0])]
        param2_values = [parameter_points[0][j].param2_value for j in range(grid_shape[1])]
        
        # Count statistics
        total_points = grid_shape[0] * grid_shape[1]
        successful_points = sum(
            1 for i in range(grid_shape[0]) for j in range(grid_shape[1])
            if parameter_points[i][j].analysis_success
        )
        stable_points = int(np.sum(processed_results['stability_map']))
        unstable_points = successful_points - stable_points
        
        return BifurcationAnalysisResult(
            analysis_type=analysis_config['analysis_type'],
            param1_name=analysis_config['param1_name'],
            param2_name=analysis_config['param2_name'],
            param1_range=np.array(param1_values),
            param2_range=np.array(param2_values),
            grid_shape=grid_shape,
            parameter_points=parameter_points,
            stability_map=processed_results['stability_map'],
            color_map=processed_results['color_map'],
            eigenvalue_map=processed_results['eigenvalue_map'],
            regime_map=processed_results['regime_map'],
            total_points=total_points,
            successful_points=successful_points,
            stable_points=stable_points,
            unstable_points=unstable_points,
            analysis_time=analysis_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            analysis_parameters={
                'fourier_grid_size': self.jacobian_builder.fourier_transform.grid_size if self.jacobian_builder else None,
                'stability_threshold': NUMERICAL_TOLERANCES['stability_threshold'],
                'grid_resolution': analysis_config['grid_resolution']
            }
        )
    
    def _save_results(self, result: BifurcationAnalysisResult, output_dir: Optional[str]):
        """Save analysis results to disk."""
        if output_dir is None:
            output_dir = Path('outputs') / 'bifurcation'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'bifurcation_{result.analysis_type}_{timestamp}.pkl'
        filepath = output_dir / filename
        
        # Save with version metadata
        print(f"Saving results to {filepath}")
        save_with_version(result, str(filepath))
        
        # Also save a summary text file
        summary_filename = f'bifurcation_{result.analysis_type}_{timestamp}_summary.txt'
        summary_filepath = output_dir / summary_filename
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Bifurcation Analysis Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Analysis type: {result.analysis_type}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Grid shape: {result.grid_shape}\n")
            f.write(f"Parameter 1: {result.param1_name} [{result.param1_range[0]:.3f}, {result.param1_range[-1]:.3f}]\n")
            f.write(f"Parameter 2: {result.param2_name} [{result.param2_range[0]:.3f}, {result.param2_range[-1]:.3f}]\n")
            f.write(f"\nResults:\n")
            f.write(f"  Total points: {result.total_points}\n")
            f.write(f"  Successful: {result.successful_points} ({result.successful_points/result.total_points:.1%})\n")
            f.write(f"  Stable: {result.stable_points} ({result.stable_points/result.successful_points:.1%})\n")
            f.write(f"  Unstable: {result.unstable_points} ({result.unstable_points/result.successful_points:.1%})\n")
            f.write(f"  Analysis time: {result.analysis_time:.1f} seconds\n")
            
            if result.analysis_parameters:
                f.write(f"\nAnalysis parameters:\n")
                for key, value in result.analysis_parameters.items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"Summary saved to {summary_filepath}")


def validate_parameter_sweeper(analysis_type: str = 'pv_analysis',
                             grid_resolution: Tuple[int, int] = (3, 3)) -> Dict[str, Any]:
    """Validate parameter sweeper with a small test analysis.
    
    Args:
        analysis_type: Type of analysis to test
        grid_resolution: Small grid for testing
        
    Returns:
        Validation results dictionary
    """
    print("Validating Parameter Sweeper...")
    
    # Initialize analyzer
    analyzer = BifurcationAnalyzer()
    
    # Run small test analysis
    print(f"  Running {grid_resolution[0]}×{grid_resolution[1]} test analysis...")
    
    try:
        result = analyzer.run_analysis(
            analysis_type=analysis_type,
            grid_resolution=grid_resolution,
            fourier_grid_size=5,  # Small Fourier grid for speed
            save_results=False  # Don't save test results
        )
        
        analysis_success = True
        error_message = None
        
    except Exception as e:
        analysis_success = False
        error_message = str(e)
        result = None
    
    # Validate results if successful
    if analysis_success and result:
        expected_points = grid_resolution[0] * grid_resolution[1]
        
        results_valid = (
            result.total_points == expected_points and
            result.successful_points > 0 and
            result.stability_map.shape == grid_resolution and
            result.eigenvalue_map.shape == grid_resolution
        )
        
        has_classifications = any(
            param_point.classification is not None
            for row in result.parameter_points
            for param_point in row
            if param_point.analysis_success
        )
        
    else:
        results_valid = False
        has_classifications = False
    
    # Summary
    validation_results = {
        'analysis_success': analysis_success,
        'results_valid': results_valid,
        'has_classifications': has_classifications,
        'error_message': error_message,
        'test_result': result,
        'overall_success': analysis_success and results_valid and has_classifications
    }
    
    if validation_results['overall_success']:
        print(f"    ✓ Analysis completed successfully")
        print(f"    ✓ Results structure valid")
        print(f"    ✓ Classifications generated")
        print(f"    Success rate: {result.successful_points}/{result.total_points}")
        print(f"    Stable/unstable: {result.stable_points}/{result.unstable_points}")
    else:
        print(f"    ✗ Validation failed: {error_message}")
    
    print(f"\nParameter Sweeper Validation {'PASSED' if validation_results['overall_success'] else 'FAILED'}")
    
    return validation_results
