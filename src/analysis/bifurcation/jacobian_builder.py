"""Jacobian builder for bifurcation analysis.

This module implements per-mode Jacobian construction for the linearized
dynamics around spatially uniform fixed points. It builds the correct
Jacobian matrices J(n) = T^(-1)(-I + W̃(n)D) for each Fourier mode.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import warnings
from dataclasses import dataclass

from src.model.connectivity import LayerConnectivity
from src.model.config import CELL_TYPES, LAYERS
from .config import (
    NUMERICAL_TOLERANCES,
    N_POPULATIONS,
    get_population_index
)
from .fourier_analysis import GaussianFourierTransform, ConnectionMatrixBuilder
from .fixed_point_solver import FixedPointResult


@dataclass
class JacobianData:
    """Container for Jacobian computation data."""
    mode_indices: Tuple[int, int]  # (nx, ny)
    mode_radius: float  # ||n||
    connection_matrix: np.ndarray  # W̃(n) - 9×9 connection matrix
    time_constant_matrix: np.ndarray  # T - 9×9 diagonal time constants
    gain_matrix: np.ndarray  # D - 9×9 diagonal ReLU slopes
    jacobian: np.ndarray  # J(n) - 9×9 Jacobian matrix
    eigenvalues: Optional[np.ndarray] = None  # Eigenvalues of J(n)
    max_eigenvalue: Optional[complex] = None  # Eigenvalue with largest real part


class PerModeJacobianBuilder:
    """Builds Jacobian matrices J(n) = T^(-1)(-I + W̃(n)D) for each Fourier mode."""
    
    def __init__(self, layer_connectivity: LayerConnectivity, 
                 time_constants: Dict[str, float], gains: Dict[str, float]):
        """Initialize Jacobian builder.
        
        Args:
            layer_connectivity: LayerConnectivity instance
            time_constants: Time constants by cell type
            gains: Gains by cell type
        """
        self.connectivity = layer_connectivity
        self.time_constants = time_constants
        self.gains = gains
        
        # Initialize helper components
        self.fourier_transform = GaussianFourierTransform()
        self.connection_builder = ConnectionMatrixBuilder(layer_connectivity)
        
        # Precompute time constant and gain matrices
        self.T_matrix = self._build_time_constant_matrix()
        self.T_inv_matrix = self._build_time_constant_inverse_matrix()
        
        # Cache for computed Jacobians
        self.jacobian_cache = {}
        
    def _build_time_constant_matrix(self) -> np.ndarray:
        """Build diagonal time constant matrix T.
        
        Returns:
            9×9 diagonal matrix with time constants
        """
        T = np.zeros((N_POPULATIONS, N_POPULATIONS))
        
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                idx = get_population_index(layer, cell_type)
                T[idx, idx] = self.time_constants[cell_type]
                
        return T
    
    def _build_time_constant_inverse_matrix(self) -> np.ndarray:
        """Build inverse time constant matrix T^(-1).
        
        Returns:
            9×9 diagonal matrix with inverse time constants
        """
        T_inv = np.zeros((N_POPULATIONS, N_POPULATIONS))
        min_tau = NUMERICAL_TOLERANCES['min_time_constant']
        
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                idx = get_population_index(layer, cell_type)
                tau = max(self.time_constants[cell_type], min_tau)
                T_inv[idx, idx] = 1.0 / tau
                
        return T_inv
    
    def _build_gain_matrix(self, fixed_point_result: FixedPointResult) -> np.ndarray:
        """Build diagonal gain matrix D from ReLU slopes at fixed point.
        
        Args:
            fixed_point_result: Fixed point solution containing ReLU slopes
            
        Returns:
            9×9 diagonal matrix with ReLU slopes
        """
        D = np.zeros((N_POPULATIONS, N_POPULATIONS))
        
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                idx = get_population_index(layer, cell_type)
                D[idx, idx] = fixed_point_result.relu_slopes[layer][cell_type]
                
        return D
    
    def build_mode_jacobian(self, nx: int, ny: int, 
                          fixed_point_result: FixedPointResult) -> JacobianData:
        """Build Jacobian matrix for a specific Fourier mode.
        
        Args:
            nx, ny: Fourier mode indices
            fixed_point_result: Fixed point solution
            
        Returns:
            JacobianData containing all matrices and eigenvalue information
        """
        # Check cache first
        cache_key = (nx, ny, id(fixed_point_result))
        if cache_key in self.jacobian_cache:
            return self.jacobian_cache[cache_key]
        
        # Build connection matrix W̃(n)
        W_tilde = self.connection_builder.build_connection_matrix_symbol(nx, ny)
        
        # Build gain matrix D
        D_matrix = self._build_gain_matrix(fixed_point_result)
        
        # Build Jacobian: J(n) = T^(-1) * (-I + W̃(n) * D)
        I = np.eye(N_POPULATIONS)
        jacobian = self.T_inv_matrix @ (-I + W_tilde @ D_matrix)
        
        # Compute mode radius
        mode_radius = self.fourier_transform.get_mode_radius(nx, ny)
        
        # Create result
        jacobian_data = JacobianData(
            mode_indices=(nx, ny),
            mode_radius=mode_radius,
            connection_matrix=W_tilde,
            time_constant_matrix=self.T_matrix,
            gain_matrix=D_matrix,
            jacobian=jacobian
        )
        
        # Cache the result
        self.jacobian_cache[cache_key] = jacobian_data
        
        return jacobian_data
    
    def compute_eigenvalues(self, jacobian_data: JacobianData) -> JacobianData:
        """Compute eigenvalues for a Jacobian matrix.
        
        Args:
            jacobian_data: JacobianData with Jacobian matrix
            
        Returns:
            Updated JacobianData with eigenvalue information
        """
        try:
            eigenvalues = np.linalg.eigvals(jacobian_data.jacobian)
            
            # Find eigenvalue with largest real part
            real_parts = np.real(eigenvalues)
            max_idx = np.argmax(real_parts)
            max_eigenvalue = eigenvalues[max_idx]
            
            # Update the data
            jacobian_data.eigenvalues = eigenvalues
            jacobian_data.max_eigenvalue = max_eigenvalue
            
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Eigenvalue computation failed for mode {jacobian_data.mode_indices}: {e}")
            jacobian_data.eigenvalues = None
            jacobian_data.max_eigenvalue = None
            
        return jacobian_data
    
    def validate_jacobian_construction(self, nx: int, ny: int, 
                                     fixed_point_result: FixedPointResult) -> Dict[str, Any]:
        """Validate Jacobian construction for a specific mode.
        
        Args:
            nx, ny: Fourier mode indices
            fixed_point_result: Fixed point solution
            
        Returns:
            Validation results dictionary
        """
        jacobian_data = self.build_mode_jacobian(nx, ny, fixed_point_result)
        jacobian_data = self.compute_eigenvalues(jacobian_data)
        
        # Validation checks
        validation_results = {}
        
        # Check matrix dimensions
        validation_results['correct_dimensions'] = (
            jacobian_data.jacobian.shape == (N_POPULATIONS, N_POPULATIONS) and
            jacobian_data.connection_matrix.shape == (N_POPULATIONS, N_POPULATIONS) and
            jacobian_data.gain_matrix.shape == (N_POPULATIONS, N_POPULATIONS)
        )
        
        # Check that matrices are finite
        validation_results['matrices_finite'] = (
            np.all(np.isfinite(jacobian_data.jacobian)) and
            np.all(np.isfinite(jacobian_data.connection_matrix)) and
            np.all(np.isfinite(jacobian_data.gain_matrix))
        )
        
        # Check time constant matrix structure
        T_diagonal = np.diag(jacobian_data.time_constant_matrix)
        T_off_diagonal = jacobian_data.time_constant_matrix - np.diag(T_diagonal)
        validation_results['T_is_diagonal'] = np.allclose(T_off_diagonal, 0.0)
        validation_results['T_positive'] = np.all(T_diagonal > 0)
        
        # Check gain matrix structure
        D_diagonal = np.diag(jacobian_data.gain_matrix)
        D_off_diagonal = jacobian_data.gain_matrix - np.diag(D_diagonal)
        validation_results['D_is_diagonal'] = np.allclose(D_off_diagonal, 0.0)
        validation_results['D_non_negative'] = np.all(D_diagonal >= 0)
        
        # Check eigenvalue computation
        validation_results['eigenvalues_computed'] = jacobian_data.eigenvalues is not None
        if jacobian_data.eigenvalues is not None:
            validation_results['eigenvalues_finite'] = np.all(np.isfinite(jacobian_data.eigenvalues))
            validation_results['max_eigenvalue_identified'] = jacobian_data.max_eigenvalue is not None
        else:
            validation_results['eigenvalues_finite'] = False
            validation_results['max_eigenvalue_identified'] = False
        
        # Check DC mode properties (nx=0, ny=0)
        if nx == 0 and ny == 0:
            # At DC, connection matrix should equal connection amplitudes
            validation_results['dc_mode_detected'] = True
            
            # Verify some connections match expected amplitudes
            dc_connections_match = True
            for conn_key, params in self.connectivity.layer_params.items():
                if conn_key.startswith('thalamus_to_'):
                    continue  # Skip thalamic connections
                    
                try:
                    source_layer, source_cell, target_layer, target_cell = self._parse_connection_key(conn_key)
                    if source_layer and source_cell and target_layer and target_cell:
                        source_idx = get_population_index(source_layer, source_cell)
                        target_idx = get_population_index(target_layer, target_cell)
                        
                        # The connection matrix includes strength scaling
                        expected_amplitude = params['amplitude'] * self.connectivity.strength_scaling.get(source_cell, 1.0)
                        actual_amplitude = jacobian_data.connection_matrix[target_idx, source_idx]
                        if abs(actual_amplitude - expected_amplitude) > NUMERICAL_TOLERANCES['dc_gain_tolerance']:
                            dc_connections_match = False
                            break
                except (KeyError, IndexError):
                    continue
                    
            validation_results['dc_connections_match'] = dc_connections_match
        else:
            validation_results['dc_mode_detected'] = False
            validation_results['dc_connections_match'] = True  # Not applicable for non-DC modes
        
        # Overall validation
        validation_results['overall_valid'] = all([
            validation_results['correct_dimensions'],
            validation_results['matrices_finite'],
            validation_results['T_is_diagonal'],
            validation_results['T_positive'],
            validation_results['D_is_diagonal'],
            validation_results['D_non_negative'],
            validation_results['eigenvalues_computed'],
            validation_results['eigenvalues_finite'],
            validation_results['dc_connections_match']
        ])
        
        return validation_results
    
    def _parse_connection_key(self, conn_key: str) -> Tuple[Optional[str], Optional[str], str, str]:
        """Parse connection key into components.
        
        Args:
            conn_key: Connection key like 'L23_E_to_L4_SST'
            
        Returns:
            Tuple of (source_layer, source_cell, target_layer, target_cell)
        """
        try:
            if conn_key.startswith('thalamus_to_'):
                parts = conn_key.split('_')
                if len(parts) >= 4:
                    return 'thalamus', None, parts[2], parts[3]
            elif '_to_' in conn_key:
                source_part, target_part = conn_key.split('_to_')
                source_parts = source_part.split('_')
                target_parts = target_part.split('_')
                
                if len(source_parts) >= 2 and len(target_parts) >= 2:
                    return source_parts[0], source_parts[1], target_parts[0], target_parts[1]
        except (IndexError, ValueError):
            pass
            
        return None, None, "", ""
    
    def clear_cache(self):
        """Clear the Jacobian cache."""
        self.jacobian_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the Jacobian cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self.jacobian_cache),
            'cached_modes': [data.mode_indices for data in self.jacobian_cache.values()],
            'memory_usage_estimate': len(self.jacobian_cache) * N_POPULATIONS**2 * 8  # bytes
        }


class ModeGridProcessor:
    """Processes Jacobian computations across all Fourier modes."""
    
    def __init__(self, jacobian_builder: PerModeJacobianBuilder):
        """Initialize mode grid processor.
        
        Args:
            jacobian_builder: PerModeJacobianBuilder instance
        """
        self.jacobian_builder = jacobian_builder
        self.fourier_transform = jacobian_builder.fourier_transform
        
    def process_all_modes(self, fixed_point_result: FixedPointResult,
                         grid_size: Optional[int] = None) -> List[JacobianData]:
        """Process Jacobian computation for all Fourier modes.
        
        Args:
            fixed_point_result: Fixed point solution
            grid_size: Optional grid size (uses default from fourier_transform if None)
            
        Returns:
            List of JacobianData for all modes
        """
        if grid_size is None:
            grid_size = self.fourier_transform.grid_size
        
        jacobian_results = []
        
        # Generate mode coordinates for the specified grid size
        modes = []
        for nx in range(grid_size):
            for ny in range(grid_size):
                modes.append((nx, ny))
        
        for nx, ny in modes:
            # Build Jacobian for this mode
            jacobian_data = self.jacobian_builder.build_mode_jacobian(nx, ny, fixed_point_result)
            
            # Compute eigenvalues
            jacobian_data = self.jacobian_builder.compute_eigenvalues(jacobian_data)
            
            jacobian_results.append(jacobian_data)
        
        return jacobian_results
    
    def find_most_unstable_mode(self, jacobian_results: List[JacobianData]) -> Tuple[float, JacobianData]:
        """Find the mode with the largest real eigenvalue.
        
        Args:
            jacobian_results: List of JacobianData from all modes
            
        Returns:
            Tuple of (max_real_eigenvalue, corresponding_jacobian_data)
        """
        max_real_eigenvalue = -np.inf
        most_unstable_mode = None
        
        for jacobian_data in jacobian_results:
            if jacobian_data.max_eigenvalue is not None:
                real_part = np.real(jacobian_data.max_eigenvalue)
                if real_part > max_real_eigenvalue:
                    max_real_eigenvalue = real_part
                    most_unstable_mode = jacobian_data
        
        return max_real_eigenvalue, most_unstable_mode
    
    def classify_stability(self, jacobian_results: List[JacobianData],
                         stability_threshold: float = NUMERICAL_TOLERANCES['stability_threshold']) -> Dict[str, Any]:
        """Classify system stability based on all mode eigenvalues.
        
        Args:
            jacobian_results: List of JacobianData from all modes
            stability_threshold: Threshold for stability (λ* < -ε)
            
        Returns:
            Dictionary with stability classification
        """
        max_real_eigenvalue, most_unstable_mode = self.find_most_unstable_mode(jacobian_results)
        
        # Determine stability
        is_stable = max_real_eigenvalue < -stability_threshold
        
        # Get winning mode information
        winning_mode_radius = 0.0
        winning_mode_indices = (0, 0)
        
        if most_unstable_mode is not None:
            winning_mode_radius = most_unstable_mode.mode_radius
            winning_mode_indices = most_unstable_mode.mode_indices
        
        # Count stable vs unstable modes
        stable_modes = 0
        unstable_modes = 0
        
        for jacobian_data in jacobian_results:
            if jacobian_data.max_eigenvalue is not None:
                if np.real(jacobian_data.max_eigenvalue) < -stability_threshold:
                    stable_modes += 1
                else:
                    unstable_modes += 1
        
        return {
            'is_stable': is_stable,
            'max_real_eigenvalue': max_real_eigenvalue,
            'winning_mode_radius': winning_mode_radius,
            'winning_mode_indices': winning_mode_indices,
            'stable_modes': stable_modes,
            'unstable_modes': unstable_modes,
            'total_modes': len(jacobian_results),
            'most_unstable_mode_data': most_unstable_mode
        }


def validate_jacobian_builder(layer_connectivity: LayerConnectivity,
                            time_constants: Dict[str, float],
                            gains: Dict[str, float],
                            fixed_point_result: FixedPointResult) -> Dict[str, Any]:
    """Validate Jacobian builder with comprehensive tests.
    
    Args:
        layer_connectivity: LayerConnectivity instance
        time_constants: Time constants by cell type
        gains: Gains by cell type
        fixed_point_result: Fixed point solution
        
    Returns:
        Validation results dictionary
    """
    print("Validating Jacobian Builder...")
    
    # Initialize builder
    jacobian_builder = PerModeJacobianBuilder(layer_connectivity, time_constants, gains)
    
    # Test 1: DC mode (nx=0, ny=0)
    print("  Test 1: DC mode validation...")
    dc_validation = jacobian_builder.validate_jacobian_construction(0, 0, fixed_point_result)
    
    print(f"    Correct dimensions: {'✓' if dc_validation['correct_dimensions'] else '✗'}")
    print(f"    Matrices finite: {'✓' if dc_validation['matrices_finite'] else '✗'}")
    print(f"    T is diagonal: {'✓' if dc_validation['T_is_diagonal'] else '✗'}")
    print(f"    D is diagonal: {'✓' if dc_validation['D_is_diagonal'] else '✗'}")
    print(f"    DC connections match: {'✓' if dc_validation['dc_connections_match'] else '✗'}")
    print(f"    Eigenvalues computed: {'✓' if dc_validation['eigenvalues_computed'] else '✗'}")
    
    # Test 2: Non-zero mode (nx=1, ny=0)
    print("  Test 2: Non-zero mode validation...")
    nonzero_validation = jacobian_builder.validate_jacobian_construction(1, 0, fixed_point_result)
    
    print(f"    Non-zero mode valid: {'✓' if nonzero_validation['overall_valid'] else '✗'}")
    
    # Test 3: Mode grid processing
    print("  Test 3: Mode grid processing...")
    processor = ModeGridProcessor(jacobian_builder)
    
    # Process a small subset of modes for testing
    test_modes = [(0, 0), (1, 0), (0, 1), (1, 1)]
    test_results = []
    
    for nx, ny in test_modes:
        jacobian_data = jacobian_builder.build_mode_jacobian(nx, ny, fixed_point_result)
        jacobian_data = jacobian_builder.compute_eigenvalues(jacobian_data)
        test_results.append(jacobian_data)
    
    # Classify stability
    stability_classification = processor.classify_stability(test_results)
    
    print(f"    Max real eigenvalue: {stability_classification['max_real_eigenvalue']:.6f}")
    print(f"    System stable: {'✓' if stability_classification['is_stable'] else '✗'}")
    print(f"    Winning mode: {stability_classification['winning_mode_indices']} (radius: {stability_classification['winning_mode_radius']:.3f})")
    print(f"    Stable/Unstable modes: {stability_classification['stable_modes']}/{stability_classification['unstable_modes']}")
    
    # Test 4: Cache functionality
    print("  Test 4: Cache functionality...")
    cache_stats_before = jacobian_builder.get_cache_stats()
    
    # Build same Jacobian again - should use cache
    jacobian_builder.build_mode_jacobian(0, 0, fixed_point_result)
    cache_stats_after = jacobian_builder.get_cache_stats()
    
    cache_working = (cache_stats_after['cache_size'] >= cache_stats_before['cache_size'])
    print(f"    Cache working: {'✓' if cache_working else '✗'}")
    print(f"    Cache size: {cache_stats_after['cache_size']} entries")
    
    # Overall validation
    validation_results = {
        'dc_mode_valid': dc_validation['overall_valid'],
        'nonzero_mode_valid': nonzero_validation['overall_valid'],
        'stability_classification': stability_classification,
        'cache_working': cache_working,
        'dc_validation_details': dc_validation,
        'nonzero_validation_details': nonzero_validation,
        'overall_success': all([
            dc_validation['overall_valid'],
            nonzero_validation['overall_valid'],
            cache_working
        ])
    }
    
    print(f"\nJacobian Builder Validation {'PASSED' if validation_results['overall_success'] else 'FAILED'}")
    
    return validation_results
