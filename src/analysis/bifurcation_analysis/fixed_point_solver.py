"""Fixed point solver for bifurcation analysis.

This module implements a robust hybrid fixed point solver that handles ReLU
nonlinearities and includes all external inputs (thalamic drive and noise means).
It finds spatially uniform steady states for the 9-population system.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import warnings
from dataclasses import dataclass

from src.model.connectivity import LayerConnectivity
from src.model.config import CELL_TYPES, LAYERS
from .config import (
    NUMERICAL_TOLERANCES,
    VALIDATION_SETTINGS,
    N_POPULATIONS,
    get_population_index
)


@dataclass
class FixedPointResult:
    """Result container for fixed point computation."""
    voltages: Dict[str, Dict[str, float]]  # V*[layer][cell_type]
    firing_rates: Dict[str, Dict[str, float]]  # r*[layer][cell_type] 
    relu_slopes: Dict[str, Dict[str, float]]  # α[layer][cell_type]
    converged: bool
    iterations_phase1: int
    iterations_phase2: int
    final_error: float
    external_inputs: Dict[str, Dict[str, float]]  # External contributions
    validation_info: Dict[str, Any]


class ExternalInputCalculator:
    """Calculates mean external inputs from thalamic drive and noise."""
    
    def __init__(self, circuit_connectivity: LayerConnectivity):
        """Initialize with circuit connectivity.
        
        Args:
            circuit_connectivity: LayerConnectivity instance
        """
        self.connectivity = circuit_connectivity
        
    def compute_mean_thalamic_input(self) -> Dict[str, Dict[str, float]]:
        """Compute mean thalamic input for each population.
        
        Returns:
            Dictionary of mean thalamic inputs by layer and cell type
        """
        thalamic_inputs = {}
        
        # Get thalamic parameters from existing config
        # These should be imported from the main model config
        try:
            from src.model.config import (
                THALAMIC_INTRINSIC_AMP, 
                THALAMIC_SENSORY_AMP,
                THALAMIC_ALPHA
            )
            
            # Compute weighted average of intrinsic and sensory inputs
            mean_thalamic_rate = (
                (1.0 - THALAMIC_ALPHA) * THALAMIC_INTRINSIC_AMP +
                THALAMIC_ALPHA * THALAMIC_SENSORY_AMP
            )
            
        except ImportError:
            # Fallback values if thalamic parameters not available
            mean_thalamic_rate = 1.0
            warnings.warn("Could not import thalamic parameters, using default value")
        
        # Apply thalamic input through connection strengths
        for layer in LAYERS:
            thalamic_inputs[layer] = {}
            for cell_type in CELL_TYPES:
                # Get thalamic connection strength
                conn_key = f'thalamus_to_{layer}_{cell_type}'
                if conn_key in self.connectivity.layer_params:
                    thalamic_strength = self.connectivity.layer_params[conn_key]['amplitude']
                    # Apply strength scaling
                    scaled_strength = thalamic_strength * self.connectivity.strength_scaling.get('thalamus', 1.0)
                    thalamic_inputs[layer][cell_type] = scaled_strength * mean_thalamic_rate
                else:
                    thalamic_inputs[layer][cell_type] = 0.0
                    
        return thalamic_inputs
    
    def compute_mean_noise_input(self) -> Dict[str, Dict[str, float]]:
        """Compute mean noise input for each population.
        
        Returns:
            Dictionary of mean noise inputs by layer and cell type
        """
        noise_inputs = {}
        
        # Get noise parameters from existing config
        try:
            from src.model.config import INITIAL_NOISE_PARAMS
            
            for layer in LAYERS:
                noise_inputs[layer] = {}
                for cell_type in CELL_TYPES:
                    if cell_type in INITIAL_NOISE_PARAMS:
                        noise_inputs[layer][cell_type] = INITIAL_NOISE_PARAMS[cell_type]['mean']
                    else:
                        noise_inputs[layer][cell_type] = 0.0
                        
        except ImportError:
            # Fallback: zero noise means
            for layer in LAYERS:
                noise_inputs[layer] = {}
                for cell_type in CELL_TYPES:
                    noise_inputs[layer][cell_type] = 0.0
            warnings.warn("Could not import noise parameters, using zero means")
            
        return noise_inputs
    
    def compute_total_external_input(self) -> Dict[str, Dict[str, float]]:
        """Compute total external input (thalamic + noise) for each population.
        
        Returns:
            Dictionary of total external inputs by layer and cell type
        """
        thalamic_inputs = self.compute_mean_thalamic_input()
        noise_inputs = self.compute_mean_noise_input()
        
        total_inputs = {}
        for layer in LAYERS:
            total_inputs[layer] = {}
            for cell_type in CELL_TYPES:
                total_inputs[layer][cell_type] = (
                    thalamic_inputs[layer][cell_type] + 
                    noise_inputs[layer][cell_type]
                )
                
        return total_inputs


class ReLUProcessor:
    """Handles ReLU nonlinearity and slope computation."""
    
    def __init__(self, voltage_threshold: float = NUMERICAL_TOLERANCES['voltage_zero_threshold']):
        """Initialize ReLU processor.
        
        Args:
            voltage_threshold: Threshold for considering voltage as zero
        """
        self.voltage_threshold = voltage_threshold
        
    def compute_firing_rates(self, voltages: Dict[str, Dict[str, float]], 
                           gains: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compute firing rates using ReLU: r = max(0, g * V).
        
        Args:
            voltages: Membrane voltages by layer and cell type
            gains: Gains by cell type
            
        Returns:
            Firing rates by layer and cell type
        """
        firing_rates = {}
        
        for layer in LAYERS:
            firing_rates[layer] = {}
            for cell_type in CELL_TYPES:
                voltage = voltages[layer][cell_type]
                gain = gains[cell_type]
                
                # Apply ReLU nonlinearity
                firing_rates[layer][cell_type] = max(0.0, gain * voltage)
                
        return firing_rates
    
    def compute_relu_slopes(self, voltages: Dict[str, Dict[str, float]], 
                          gains: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compute ReLU slopes at the fixed point: α = g if g*V > 0, else 0.
        
        Args:
            voltages: Membrane voltages by layer and cell type
            gains: Gains by cell type
            
        Returns:
            ReLU slopes by layer and cell type
        """
        relu_slopes = {}
        
        for layer in LAYERS:
            relu_slopes[layer] = {}
            for cell_type in CELL_TYPES:
                voltage = voltages[layer][cell_type]
                gain = gains[cell_type]
                
                # Compute slope of ReLU at this point
                if abs(gain * voltage) < self.voltage_threshold:
                    # At or very close to zero - use zero slope for numerical stability
                    relu_slopes[layer][cell_type] = 0.0
                elif gain * voltage > 0:
                    # Above threshold - slope is the gain
                    relu_slopes[layer][cell_type] = gain
                else:
                    # Below zero (shouldn't happen with ReLU, but handle gracefully)
                    relu_slopes[layer][cell_type] = 0.0
                    
        return relu_slopes


class HybridFixedPointSolver:
    """Hybrid fixed point solver using fixed-point iteration followed by Newton refinement."""
    
    def __init__(self, circuit_connectivity: LayerConnectivity, 
                 time_constants: Dict[str, float], gains: Dict[str, float]):
        """Initialize solver with circuit parameters.
        
        Args:
            circuit_connectivity: LayerConnectivity instance
            time_constants: Time constants by cell type
            gains: Gains by cell type
        """
        self.connectivity = circuit_connectivity
        self.time_constants = time_constants
        self.gains = gains
        
        # Initialize helper classes
        self.external_calculator = ExternalInputCalculator(circuit_connectivity)
        self.relu_processor = ReLUProcessor()
        
        # Get tolerances
        self.convergence_tolerance = NUMERICAL_TOLERANCES['fixed_point_convergence']
        self.max_iter_phase1 = NUMERICAL_TOLERANCES['fixed_point_max_iter_phase1']
        self.max_iter_phase2 = NUMERICAL_TOLERANCES['fixed_point_max_iter_phase2']
        
        # Compute external inputs once
        self.external_inputs = self.external_calculator.compute_total_external_input()
        
    def _initialize_voltages(self, initial_guess: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
        """Initialize voltage guess for iteration.
        
        Args:
            initial_guess: Optional initial voltage guess
            
        Returns:
            Initial voltage dictionary
        """
        if initial_guess is not None:
            return {layer: cell_dict.copy() for layer, cell_dict in initial_guess.items()}
        
        # Default initialization: small positive values
        voltages = {}
        for layer in LAYERS:
            voltages[layer] = {}
            for cell_type in CELL_TYPES:
                voltages[layer][cell_type] = 1.0  # Start with reasonable positive value
                
        return voltages
    
    def _compute_recurrent_input(self, firing_rates: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute recurrent input from all populations.
        
        Args:
            firing_rates: Current firing rates by layer and cell type
            
        Returns:
            Recurrent inputs by layer and cell type
        """
        recurrent_inputs = {}
        
        # Initialize to zero
        for layer in LAYERS:
            recurrent_inputs[layer] = {}
            for cell_type in CELL_TYPES:
                recurrent_inputs[layer][cell_type] = 0.0
        
        # Add contributions from all connections
        for conn_key, params in self.connectivity.layer_params.items():
            # Parse connection
            source_layer, source_cell, target_layer, target_cell = self._parse_connection_key(conn_key)
            
            if source_layer is None or source_layer == 'thalamus':
                continue  # Skip invalid or thalamic connections
                
            # Get connection strength (amplitude at k=0)
            amplitude = params['amplitude']
            
            # Apply strength scaling
            scaled_amplitude = amplitude * self.connectivity.strength_scaling.get(source_cell, 1.0)
            
            # Add contribution
            if (source_layer in firing_rates and source_cell in firing_rates[source_layer] and
                target_layer in recurrent_inputs and target_cell in recurrent_inputs[target_layer]):
                
                source_rate = firing_rates[source_layer][source_cell]
                recurrent_inputs[target_layer][target_cell] += scaled_amplitude * source_rate
        
        return recurrent_inputs
    
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
    
    def _fixed_point_iteration_step(self, voltages: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Perform one step of fixed-point iteration.
        
        Args:
            voltages: Current voltage estimate
            
        Returns:
            Updated voltage estimate
        """
        # Compute firing rates
        firing_rates = self.relu_processor.compute_firing_rates(voltages, self.gains)
        
        # Compute recurrent input
        recurrent_inputs = self._compute_recurrent_input(firing_rates)
        
        # Update voltages: V_new = recurrent_input + external_input
        new_voltages = {}
        for layer in LAYERS:
            new_voltages[layer] = {}
            for cell_type in CELL_TYPES:
                new_voltages[layer][cell_type] = (
                    recurrent_inputs[layer][cell_type] + 
                    self.external_inputs[layer][cell_type]
                )
                
        return new_voltages
    
    def _compute_voltage_error(self, v1: Dict[str, Dict[str, float]], 
                             v2: Dict[str, Dict[str, float]]) -> float:
        """Compute maximum voltage difference between two estimates.
        
        Args:
            v1, v2: Voltage dictionaries to compare
            
        Returns:
            Maximum absolute difference
        """
        max_error = 0.0
        
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                error = abs(v1[layer][cell_type] - v2[layer][cell_type])
                max_error = max(max_error, error)
                
        return max_error
    
    def _phase1_fixed_point_iteration(self, initial_voltages: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], int, float]:
        """Phase 1: Fixed-point iteration to handle ReLU kinks.
        
        Args:
            initial_voltages: Starting voltage guess
            
        Returns:
            Tuple of (final_voltages, iterations, final_error)
        """
        voltages = initial_voltages
        
        for iteration in range(self.max_iter_phase1):
            new_voltages = self._fixed_point_iteration_step(voltages)
            error = self._compute_voltage_error(voltages, new_voltages)
            
            if error < self.convergence_tolerance:
                return new_voltages, iteration + 1, error
                
            voltages = new_voltages
            
        return voltages, self.max_iter_phase1, error
    
    def _build_jacobian_matrix(self, voltages: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Build Jacobian matrix for Newton method.
        
        Args:
            voltages: Current voltage estimate
            
        Returns:
            9×9 Jacobian matrix
        """
        jacobian = np.zeros((N_POPULATIONS, N_POPULATIONS))
        
        # Compute ReLU slopes
        relu_slopes = self.relu_processor.compute_relu_slopes(voltages, self.gains)
        
        # Build Jacobian: J[i,j] = δF_i/δV_j
        # F_i(V) = -V_i + Σ_j W_ij * ReLU(g_j * V_j) + I_ext_i
        
        for target_layer in LAYERS:
            for target_cell in CELL_TYPES:
                target_idx = get_population_index(target_layer, target_cell)
                
                # Diagonal term: -1
                jacobian[target_idx, target_idx] = -1.0
                
                # Off-diagonal terms: connection weights times ReLU slopes
                for source_layer in LAYERS:
                    for source_cell in CELL_TYPES:
                        source_idx = get_population_index(source_layer, source_cell)
                        
                        # Get connection strength
                        conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
                        if conn_key in self.connectivity.layer_params:
                            amplitude = self.connectivity.layer_params[conn_key]['amplitude']
                            scaled_amplitude = amplitude * self.connectivity.strength_scaling.get(source_cell, 1.0)
                            
                            # Multiply by ReLU slope
                            relu_slope = relu_slopes[source_layer][source_cell]
                            jacobian[target_idx, source_idx] += scaled_amplitude * self.gains[source_cell] * relu_slope
        
        return jacobian
    
    def _flatten_voltages(self, voltages: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Convert voltage dictionary to flat array.
        
        Args:
            voltages: Voltage dictionary
            
        Returns:
            Flat voltage array
        """
        voltage_array = np.zeros(N_POPULATIONS)
        
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                idx = get_population_index(layer, cell_type)
                voltage_array[idx] = voltages[layer][cell_type]
                
        return voltage_array
    
    def _unflatten_voltages(self, voltage_array: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Convert flat voltage array to dictionary.
        
        Args:
            voltage_array: Flat voltage array
            
        Returns:
            Voltage dictionary
        """
        voltages = {}
        
        for layer in LAYERS:
            voltages[layer] = {}
            for cell_type in CELL_TYPES:
                idx = get_population_index(layer, cell_type)
                voltages[layer][cell_type] = voltage_array[idx]
                
        return voltages
    
    def _compute_residual(self, voltages: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Compute residual F(V) for Newton method.
        
        Args:
            voltages: Current voltage estimate
            
        Returns:
            Residual vector
        """
        # Compute what the voltages should be
        target_voltages = self._fixed_point_iteration_step(voltages)
        
        # Residual is F(V) = V - (recurrent + external)
        residual = np.zeros(N_POPULATIONS)
        
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                idx = get_population_index(layer, cell_type)
                residual[idx] = voltages[layer][cell_type] - target_voltages[layer][cell_type]
                
        return residual
    
    def _phase2_newton_refinement(self, initial_voltages: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], int, float]:
        """Phase 2: Newton-Raphson refinement for high precision.
        
        Args:
            initial_voltages: Starting voltage from Phase 1
            
        Returns:
            Tuple of (final_voltages, iterations, final_error)
        """
        voltages = initial_voltages
        
        for iteration in range(self.max_iter_phase2):
            # Compute Jacobian and residual
            try:
                jacobian = self._build_jacobian_matrix(voltages)
                residual = self._compute_residual(voltages)
                
                # Solve linear system: J * delta_V = -residual
                delta_v = np.linalg.solve(jacobian, -residual)
                
                # Update voltages
                voltage_array = self._flatten_voltages(voltages)
                voltage_array += delta_v
                new_voltages = self._unflatten_voltages(voltage_array)
                
                # Check convergence
                error = np.linalg.norm(delta_v)
                
                if error < self.convergence_tolerance:
                    return new_voltages, iteration + 1, error
                    
                voltages = new_voltages
                
            except np.linalg.LinAlgError:
                # Singular matrix - return current estimate
                residual = self._compute_residual(voltages)
                error = np.linalg.norm(residual)
                return voltages, iteration + 1, error
        
        # Compute final error
        residual = self._compute_residual(voltages)
        error = np.linalg.norm(residual)
        
        return voltages, self.max_iter_phase2, error
    
    def solve(self, initial_guess: Optional[Dict[str, Dict[str, float]]] = None,
              use_newton_refinement: bool = True) -> FixedPointResult:
        """Solve for the spatially uniform fixed point.
        
        Args:
            initial_guess: Optional initial voltage guess
            use_newton_refinement: Whether to use Newton refinement (Phase 2)
            
        Returns:
            FixedPointResult with solution and diagnostics
        """
        # Initialize voltages
        initial_voltages = self._initialize_voltages(initial_guess)
        
        # Phase 1: Fixed-point iteration
        voltages_phase1, iter_phase1, error_phase1 = self._phase1_fixed_point_iteration(initial_voltages)
        
        # Phase 2: Newton refinement (optional)
        if use_newton_refinement and error_phase1 > self.convergence_tolerance:
            voltages_final, iter_phase2, error_final = self._phase2_newton_refinement(voltages_phase1)
        else:
            voltages_final, iter_phase2, error_final = voltages_phase1, 0, error_phase1
        
        # Compute final firing rates and ReLU slopes
        firing_rates = self.relu_processor.compute_firing_rates(voltages_final, self.gains)
        relu_slopes = self.relu_processor.compute_relu_slopes(voltages_final, self.gains)
        
        # Check convergence
        converged = error_final < self.convergence_tolerance
        
        # Validation information
        validation_info = {
            'phase1_error': error_phase1,
            'phase2_error': error_final if use_newton_refinement else None,
            'used_newton': use_newton_refinement,
            'voltage_range': self._get_voltage_statistics(voltages_final),
            'firing_rate_range': self._get_voltage_statistics(firing_rates),
            'active_populations': self._count_active_populations(relu_slopes)
        }
        
        return FixedPointResult(
            voltages=voltages_final,
            firing_rates=firing_rates,
            relu_slopes=relu_slopes,
            converged=converged,
            iterations_phase1=iter_phase1,
            iterations_phase2=iter_phase2,
            final_error=error_final,
            external_inputs=self.external_inputs,
            validation_info=validation_info
        )
    
    def _get_voltage_statistics(self, voltages: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Get statistics of voltage values.
        
        Args:
            voltages: Voltage dictionary
            
        Returns:
            Statistics dictionary
        """
        values = []
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                values.append(voltages[layer][cell_type])
        
        values = np.array(values)
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    def _count_active_populations(self, relu_slopes: Dict[str, Dict[str, float]]) -> int:
        """Count populations with non-zero ReLU slopes.
        
        Args:
            relu_slopes: ReLU slopes dictionary
            
        Returns:
            Number of active populations
        """
        count = 0
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                if relu_slopes[layer][cell_type] > 0:
                    count += 1
        return count


def validate_fixed_point_solver(circuit_connectivity: LayerConnectivity,
                               time_constants: Dict[str, float],
                               gains: Dict[str, float]) -> Dict[str, Any]:
    """Validate fixed point solver with comprehensive tests.
    
    Args:
        circuit_connectivity: LayerConnectivity instance
        time_constants: Time constants by cell type
        gains: Gains by cell type
        
    Returns:
        Validation results dictionary
    """
    print("Validating Fixed Point Solver...")
    
    # Initialize solver
    solver = HybridFixedPointSolver(circuit_connectivity, time_constants, gains)
    
    # Test 1: Basic convergence
    print("  Test 1: Basic convergence...")
    result = solver.solve()
    
    basic_converged = result.converged
    reasonable_voltages = (-10.0 < result.validation_info['voltage_range']['min'] and
                          result.validation_info['voltage_range']['max'] < 10.0)
    
    print(f"    Converged: {'✓' if basic_converged else '✗'}")
    print(f"    Reasonable voltages: {'✓' if reasonable_voltages else '✗'}")
    print(f"    Final error: {result.final_error:.2e}")
    print(f"    Active populations: {result.validation_info['active_populations']}/9")
    
    # Test 2: Multiple initial conditions
    if VALIDATION_SETTINGS['validate_fixed_point_convergence']:
        print("  Test 2: Multiple initial conditions...")
        
        consistent_results = True
        reference_voltages = result.voltages
        
        for _ in range(VALIDATION_SETTINGS['num_random_starts']):
            # Random initial guess
            random_guess = {}
            for layer in LAYERS:
                random_guess[layer] = {}
                for cell_type in CELL_TYPES:
                    random_guess[layer][cell_type] = np.random.uniform(-2.0, 2.0)
            
            test_result = solver.solve(initial_guess=random_guess)
            
            if test_result.converged:
                # Check if we get the same result
                max_diff = 0.0
                for layer in LAYERS:
                    for cell_type in CELL_TYPES:
                        diff = abs(test_result.voltages[layer][cell_type] - 
                                 reference_voltages[layer][cell_type])
                        max_diff = max(max_diff, diff)
                
                if max_diff > 10 * NUMERICAL_TOLERANCES['fixed_point_convergence']:
                    consistent_results = False
                    break
        
        print(f"    Consistent results: {'✓' if consistent_results else '✗'}")
    else:
        consistent_results = True
    
    # Test 3: External input validation
    print("  Test 3: External input validation...")
    
    external_calculator = ExternalInputCalculator(circuit_connectivity)
    thalamic_inputs = external_calculator.compute_mean_thalamic_input()
    external_calculator.compute_mean_noise_input()  # Validate noise computation
    
    has_thalamic_input = any(
        abs(thalamic_inputs[layer][cell_type]) > 1e-10
        for layer in LAYERS for cell_type in CELL_TYPES
    )
    
    has_external_contributions = any(
        abs(result.external_inputs[layer][cell_type]) > 1e-10
        for layer in LAYERS for cell_type in CELL_TYPES
    )
    
    print(f"    Has thalamic input: {'✓' if has_thalamic_input else '✗'}")
    print(f"    External inputs included: {'✓' if has_external_contributions else '✗'}")
    
    # Test 4: ReLU consistency
    print("  Test 4: ReLU consistency...")
    
    relu_consistent = True
    for layer in LAYERS:
        for cell_type in CELL_TYPES:
            voltage = result.voltages[layer][cell_type]
            firing_rate = result.firing_rates[layer][cell_type]
            relu_slope = result.relu_slopes[layer][cell_type]
            gain = gains[cell_type]
            
            # Check ReLU consistency
            expected_rate = max(0.0, gain * voltage)
            if abs(firing_rate - expected_rate) > 1e-10:
                relu_consistent = False
                break
                
            # Check slope consistency
            if voltage * gain > NUMERICAL_TOLERANCES['voltage_zero_threshold']:
                expected_slope = gain
            else:
                expected_slope = 0.0
                
            if abs(relu_slope - expected_slope) > 1e-10:
                relu_consistent = False
                break
    
    print(f"    ReLU consistency: {'✓' if relu_consistent else '✗'}")
    
    # Summary
    validation_results = {
        'basic_convergence': basic_converged,
        'reasonable_voltages': reasonable_voltages,
        'consistent_results': consistent_results,
        'external_inputs_included': has_external_contributions,
        'relu_consistent': relu_consistent,
        'final_error': result.final_error,
        'active_populations': result.validation_info['active_populations'],
        'solver_result': result,
        'overall_success': all([
            basic_converged, reasonable_voltages, consistent_results,
            has_external_contributions, relu_consistent
        ])
    }
    
    print(f"\nFixed Point Solver Validation {'PASSED' if validation_results['overall_success'] else 'FAILED'}")
    
    return validation_results
