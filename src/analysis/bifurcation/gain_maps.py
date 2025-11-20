"""Gain analysis module for bifurcation analysis.

This module implements gain map and spectrum computations, showing how circuits
amplify thalamic input at different spatial frequencies across parameter spaces.
"""

import numpy as np
import multiprocessing as mp
import os
import copy
from typing import Dict, Tuple, List

from .core import (
    NetworkModel, 
    SteadyStateFinder, 
    StabilityAnalyzer,
    compute_B_fourier,
    set_nested_value,
    get_nested_value
)
from .config import (
    ANALYSIS_PARAMS,
    ALL_LAYERS,
    SCANNABLE_PARAMETERS,
    ParameterSpec,
    TAU_MIN,
    TAU_MAX,
    SIGMA_MIN,
    SIGMA_MAX,
    FIXED_RATIO_TAU_MIN,
    FIXED_RATIO_TAU_MAX,
    FIXED_RATIO_SIGMA_MIN,
    FIXED_RATIO_SIGMA_MAX,
    GRID_RESOLUTION,
    MEAN_STATE_SEED,
    SPECTRUM_PARAM_SWEEP_RANGE,
    SPECTRUM_PARAM_RESOLUTION,
    SPECTRUM_K_MAX,
)
from src.analysis.common import PRESETS


# ============================================================================
# 2D Gain Maps
# ============================================================================

def compute_gain_for_point(preset: Dict, verbose: bool = False) -> Tuple[float, float, bool]:
    """Compute static gain spectrum for a single parameter point.
    
    Args:
        preset: Network preset dictionary with parameters
        verbose: If True, print diagnostic information
        
    Returns:
        Tuple of (k_critical, max_gain, is_flat)
        - k_critical: Wavenumber with maximum gain G(k)
        - max_gain: Maximum gain value across all k
        - is_flat: True if gain spectrum is flat (no dominant k)
    """
    network = NetworkModel(preset, layers=ALL_LAYERS)
    finder = SteadyStateFinder(network)
    
    # Use weak thalamic input to avoid divergence
    thalamic_magnitude = 0.2
    thalamic_input = network.compute_thalamic_input(thalamic_magnitude)
    
    # Find steady state
    r_star, status = finder.find_steady_state(thalamic_input=thalamic_input)
    
    # Handle failed convergence: use minimal activity state
    if status in ['diverged', 'not_converged'] or np.any(r_star > 100):
        r_star = np.ones(len(network.tau)) * 0.15
        weak_input = network.mu + thalamic_input * 0.1
        for _ in range(10):
            input_vec = network.A @ r_star + weak_input
            r_new = np.maximum(0.0, network.gain * input_vec)
            r_star = 0.9 * r_star + 0.1 * r_new
            r_star = np.clip(r_star, 0.05, 0.5)
    
    analyzer = StabilityAnalyzer(network, r_star)
    
    # Get analysis parameters
    n_modes = ANALYSIS_PARAMS['n_modes']
    grid_size = ANALYSIS_PARAMS['grid_size']
    domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)
    n_modes_effective = min(n_modes, int(0.6 * domain_length))
    
    total_pops = len(network.tau)
    
    # Pre-compute unique k² values
    k_squared_set = set()
    for n1 in range(0, n_modes_effective + 1):
        for n2 in range(0, n_modes_effective + 1):
            k_squared_set.add(n1**2 + n2**2)
    
    # Cache exponential factors for spatial filtering
    exp_cache = {}
    for k_squared in k_squared_set:
        exp_cache[k_squared] = np.zeros((total_pops, total_pops))
        for i in range(total_pops):
            for j in range(total_pops):
                sigma_ij = network.sigma[i, j] / domain_length
                exp_cache[k_squared][i, j] = np.exp(
                    -2 * np.pi**2 * k_squared * sigma_ij**2
                )
    
    # Dictionary to store results by k²
    results_by_k2 = {}
    
    # Scan positive quadrant only
    for n1 in range(0, n_modes_effective + 1):
        for n2 in range(0, n_modes_effective + 1):
            k_squared = n1**2 + n2**2
            k = np.sqrt(k_squared)
            
            if k > n_modes:
                continue
            
            # Build Jacobian using cached exponentials
            J = np.zeros((total_pops, total_pops))
            exp_factors = exp_cache[k_squared]
            
            for i in range(total_pops):
                for j in range(total_pops):
                    w_tilde = network.A[i, j] * exp_factors[i, j]
                    if i == j:
                        J[i, j] = (-1.0 / network.tau[i] + 
                                  (analyzer.g_eff[i] * w_tilde) / network.tau[i])
                    else:
                        J[i, j] = (analyzer.g_eff[i] * w_tilde) / network.tau[i]
            
            # Compute B(k) with thalamic spatial filtering
            B_k = compute_B_fourier(network, k_squared, domain_length)
            
            # Check if B(k) is non-zero
            if np.linalg.norm(B_k) < 1e-10:
                continue
            
            # Compute gain: G(k) = ||−J(k)^(-1) B(k)||
            try:
                # Solve -J(k) @ x = B(k) for x
                response = np.linalg.solve(-J, B_k)
                gain = np.linalg.norm(response)
                
                # Store or update max gain for this k
                if k_squared not in results_by_k2:
                    results_by_k2[k_squared] = {'k': k, 'gain': gain}
                else:
                    if gain > results_by_k2[k_squared]['gain']:
                        results_by_k2[k_squared]['gain'] = gain
            except np.linalg.LinAlgError:
                # Singular matrix - skip this mode
                continue
    
    # Find critical k and check for flat spectrum
    if results_by_k2:
        max_entry = max(results_by_k2.values(), key=lambda x: x['gain'])
        min_entry = min(results_by_k2.values(), key=lambda x: x['gain'])
        
        k_critical = max_entry['k']
        max_gain = max_entry['gain']
        
        # Check for flat spectrum (less than 20% variation)
        gain_range = (max_entry['gain'] - min_entry['gain']) / (max_entry['gain'] + 1e-10)
        is_flat = gain_range < 0.2
    else:
        # Failed computation: all modes skipped or singular
        k_critical = 0.0
        max_gain = np.nan
        is_flat = True  # Mark as flat (will be grey in visualization)
    
    return k_critical, max_gain, is_flat


def _gain_map_worker(args: Tuple) -> Tuple[float, float, Tuple[float, float, bool]]:
    """Worker function for parallel gain map computation.
    
    Args:
        args: Tuple of (preset, x_val, y_val, param_x_spec, param_y_spec)
        
    Returns:
        Tuple of (x_val, y_val, result) where result is from compute_gain_for_point
    """
    preset, x_val, y_val, param_x_spec, param_y_spec = args
    
    # Modify preset
    modified_preset = copy.deepcopy(preset)
    set_nested_value(modified_preset, param_x_spec.path, x_val)
    set_nested_value(modified_preset, param_y_spec.path, y_val)
    
    # Compute gain
    result = compute_gain_for_point(modified_preset)
    
    return (x_val, y_val, result)


def scan_parameter_space_parallel(
    preset: Dict,
    param_x_spec: ParameterSpec,
    param_y_spec: ParameterSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_processes: int = None
) -> Dict:
    """Scan 2D parameter space for gain using multiprocessing.
    
    Args:
        preset: Base developmental preset
        param_x_spec: Specification for x-axis parameter
        param_y_spec: Specification for y-axis parameter
        x_values: Array of x parameter values to scan
        y_values: Array of y parameter values to scan
        n_processes: Number of processes (default: cpu_count - 1)
        
    Returns:
        Dict with:
        - k_matrix: Critical wavenumbers (n_y × n_x)
        - gain_matrix: Max gain values (n_y × n_x)
        - flatness_matrix: Flat spectrum flags (n_y × n_x)
        - param_x_values: x parameter values
        - param_y_values: y parameter values
        - param_x_spec: x parameter specification
        - param_y_spec: y parameter specification
    """
    n_x = len(x_values)
    n_y = len(y_values)
    
    k_matrix = np.zeros((n_y, n_x))
    gain_matrix = np.zeros((n_y, n_x))
    flatness_matrix = np.zeros((n_y, n_x), dtype=bool)
    
    # Prepare tasks for parallel execution
    tasks = []
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            tasks.append((preset, x_val, y_val, param_x_spec, param_y_spec))
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 1)
    
    print(f"  Running {len(tasks)} gain computations using {n_processes} processes...")
    
    # Execute in parallel
    with mp.Pool(n_processes) as pool:
        results = pool.map(_gain_map_worker, tasks)
    
    # Unpack results into matrices
    for x_val, y_val, (k_crit, max_gain, is_flat) in results:
        # Find indices
        i = np.argmin(np.abs(y_values - y_val))
        j = np.argmin(np.abs(x_values - x_val))
        
        k_matrix[i, j] = k_crit
        gain_matrix[i, j] = max_gain
        flatness_matrix[i, j] = is_flat
    
    return {
        'k_matrix': k_matrix,
        'gain_matrix': gain_matrix,
        'flatness_matrix': flatness_matrix,
        'param_x_values': x_values,
        'param_y_values': y_values,
        'param_x_spec': param_x_spec,
        'param_y_spec': param_y_spec,
    }


def compute_gain_maps_single_stage(
    stage_name: str,
    parameter_pairs: List[Tuple[str, str]],
    mode: str = 'fixed_absolute',
    n_processes: int = None
) -> Dict:
    """Compute gain maps for a single developmental stage.
    
    Args:
        stage_name: Developmental stage ('P4', 'P8', 'P12', 'P16')
        parameter_pairs: List of (param_x_key, param_y_key) tuples
        mode: Range mode ('fixed_absolute' or 'fixed_ratio')
        n_processes: Number of processes for parallelization
        
    Returns:
        Dict mapping parameter pair tuples to their results
    """
    preset = PRESETS[stage_name.upper()]
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Computing gain maps for {stage_name}")
    print(f"{'='*70}")
    
    for param_x_key, param_y_key in parameter_pairs:
        param_x_spec = SCANNABLE_PARAMETERS[param_x_key]
        param_y_spec = SCANNABLE_PARAMETERS[param_y_key]
        
        print(f"\nScanning {param_x_key} vs {param_y_key}...")
        
        # Determine parameter ranges based on mode (same logic as stability_maps)
        if mode == 'fixed_absolute':
            x_min, x_max = param_x_spec.default_range
            y_min, y_max = param_y_spec.default_range
        elif mode == 'fixed_ratio':
            if param_x_spec.use_ratio and param_x_spec.reference_param:
                ref_spec = SCANNABLE_PARAMETERS[param_x_spec.reference_param]
                ref_value = get_nested_value(preset, ref_spec.path)
                if 'tau' in param_x_key:
                    x_min = FIXED_RATIO_TAU_MIN * ref_value
                    x_max = FIXED_RATIO_TAU_MAX * ref_value
                else:
                    x_min = FIXED_RATIO_SIGMA_MIN * ref_value
                    x_max = FIXED_RATIO_SIGMA_MAX * ref_value
            else:
                x_min, x_max = param_x_spec.default_range
            
            if param_y_spec.use_ratio and param_y_spec.reference_param:
                ref_spec = SCANNABLE_PARAMETERS[param_y_spec.reference_param]
                ref_value = get_nested_value(preset, ref_spec.path)
                if 'tau' in param_y_key:
                    y_min = FIXED_RATIO_TAU_MIN * ref_value
                    y_max = FIXED_RATIO_TAU_MAX * ref_value
                else:
                    y_min = FIXED_RATIO_SIGMA_MIN * ref_value
                    y_max = FIXED_RATIO_SIGMA_MAX * ref_value
            else:
                y_min, y_max = param_y_spec.default_range
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Generate parameter grids
        x_values = np.linspace(x_min, x_max, GRID_RESOLUTION)
        y_values = np.linspace(y_min, y_max, GRID_RESOLUTION)
        
        print(f"  {param_x_key} range: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  {param_y_key} range: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Grid: {GRID_RESOLUTION}×{GRID_RESOLUTION} = {GRID_RESOLUTION**2} points")
        
        # Scan parameter space with parallelization
        result = scan_parameter_space_parallel(
            preset, param_x_spec, param_y_spec,
            x_values, y_values, n_processes
        )
        
        # Add preset value information
        result['preset'] = preset
        result['preset_x_value'] = get_nested_value(preset, param_x_spec.path)
        result['preset_y_value'] = get_nested_value(preset, param_y_spec.path)
        
        # Compute gain at exact preset parameters
        k_preset, gain_preset, flat_preset = compute_gain_for_point(preset)
        result['preset_k'] = k_preset
        result['preset_gain'] = gain_preset
        result['preset_flat'] = flat_preset
        
        print(f"  Preset values: k={k_preset:.3f}, max_gain={gain_preset:.3f}, flat={flat_preset}")
        
        results[(param_x_key, param_y_key)] = result
    
    return results


def compute_gain_maps_all_stages(
    parameter_pairs: List[Tuple[str, str]],
    stages: List[str] = None,
    mode: str = 'fixed_absolute',
    n_processes: int = None
) -> Dict:
    """Compute gain maps for all stages and parameter pairs.
    
    Args:
        parameter_pairs: List of (param_x_key, param_y_key) tuples to scan
        stages: List of developmental stages (default: ['P4', 'P8', 'P12', 'P16'])
        mode: Range mode ('fixed_absolute' or 'fixed_ratio')
        n_processes: Number of processes for parallelization
        
    Returns:
        Nested dict: {param_pair: {stage: results}}
    """
    if stages is None:
        stages = ['P4', 'P8', 'P12', 'P16']
    
    print("\n" + "="*70)
    print(f"  GAIN MAPS - Computing All Stages ({mode})")
    print("="*70 + "\n")
    
    # Organize results by parameter pair, then by stage
    all_results = {pair: {} for pair in parameter_pairs}
    
    for stage_name in stages:
        stage_results = compute_gain_maps_single_stage(
            stage_name, parameter_pairs, mode, n_processes
        )
        
        # Distribute results into the organized structure
        for pair, result in stage_results.items():
            all_results[pair][stage_name] = result
    
    return all_results


# ============================================================================
# 1D Gain Spectra
# ============================================================================

def compute_gain_spectrum(preset: Dict, k_values: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Compute full gain spectrum G(k) for all k values at a single parameter point.
    
    Args:
        preset: Network preset dictionary with parameters
        k_values: Array of k values to compute gain for
        verbose: If True, print diagnostic information
        
    Returns:
        gain_spectrum: Array of gain values, one per k (NaN for failed computations)
    """
    network = NetworkModel(preset, layers=ALL_LAYERS)
    finder = SteadyStateFinder(network)
    
    # Use weak thalamic input to avoid divergence
    thalamic_magnitude = 0.2
    thalamic_input = network.compute_thalamic_input(thalamic_magnitude)
    
    # Find steady state
    r_star, status = finder.find_steady_state(thalamic_input=thalamic_input)
    
    # Handle failed convergence
    if status in ['diverged', 'not_converged'] or np.any(r_star > 100):
        r_star = np.ones(len(network.tau)) * 0.15
        weak_input = network.mu + thalamic_input * 0.1
        for _ in range(10):
            input_vec = network.A @ r_star + weak_input
            r_new = np.maximum(0.0, network.gain * input_vec)
            r_star = 0.9 * r_star + 0.1 * r_new
            r_star = np.clip(r_star, 0.05, 0.5)
    
    analyzer = StabilityAnalyzer(network, r_star)
    
    # Get analysis parameters
    grid_size = ANALYSIS_PARAMS['grid_size']
    domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)
    total_pops = len(network.tau)
    
    # Pre-compute exponential cache for all k values
    k_squared_values = k_values ** 2
    exp_cache = {}
    for k_squared in k_squared_values:
        exp_cache[k_squared] = np.zeros((total_pops, total_pops))
        for i in range(total_pops):
            for j in range(total_pops):
                sigma_ij = network.sigma[i, j] / domain_length
                exp_cache[k_squared][i, j] = np.exp(
                    -2 * np.pi**2 * k_squared * sigma_ij**2
                )
    
    # Compute gain for each k value
    gain_spectrum = np.full(len(k_values), np.nan)
    
    for idx, k in enumerate(k_values):
        k_squared = k ** 2
        
        # Build Jacobian using cached exponentials
        J = np.zeros((total_pops, total_pops))
        exp_factors = exp_cache[k_squared]
        
        for i in range(total_pops):
            for j in range(total_pops):
                w_tilde = network.A[i, j] * exp_factors[i, j]
                if i == j:
                    J[i, j] = (-1.0 / network.tau[i] + 
                              (analyzer.g_eff[i] * w_tilde) / network.tau[i])
                else:
                    J[i, j] = (analyzer.g_eff[i] * w_tilde) / network.tau[i]
        
        # Compute B(k) with thalamic spatial filtering
        B_k = compute_B_fourier(network, k_squared, domain_length)
        
        # Check if B(k) is non-zero
        if np.linalg.norm(B_k) < 1e-10:
            continue
        
        # Compute gain: G(k) = ||−J(k)^(-1) B(k)||
        try:
            response = np.linalg.solve(-J, B_k)
            gain = np.linalg.norm(response)
            gain_spectrum[idx] = gain
        except np.linalg.LinAlgError:
            # Singular matrix - leave as NaN
            pass
    
    return gain_spectrum


def get_k_values() -> np.ndarray:
    """Generate k values for spectrum computation.
    
    Returns:
        Array of unique k values sorted in ascending order
    """
    n_modes = ANALYSIS_PARAMS['n_modes']
    grid_size = ANALYSIS_PARAMS['grid_size']
    domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)
    n_modes_effective = min(n_modes, int(0.6 * domain_length))
    
    # Generate unique k values from (n1, n2) grid
    k_squared_set = set()
    for n1 in range(0, n_modes_effective + 1):
        for n2 in range(0, n_modes_effective + 1):
            k_squared = n1**2 + n2**2
            k = np.sqrt(k_squared)
            if k <= n_modes and k <= SPECTRUM_K_MAX:
                k_squared_set.add(k_squared)
    
    # Convert to sorted array of k values
    k_values = np.sqrt(np.array(sorted(k_squared_set)))
    return k_values


def _spectrum_worker(args: Tuple) -> Tuple[float, np.ndarray]:
    """Worker function for parallel spectrum computation.
    
    Args:
        args: Tuple of (preset, param_val, param_spec, k_values)
        
    Returns:
        Tuple of (param_val, gain_spectrum)
    """
    preset, param_val, param_spec, k_values = args
    
    # Modify preset
    modified_preset = copy.deepcopy(preset)
    set_nested_value(modified_preset, param_spec.path, param_val)
    
    # Compute gain spectrum
    gain_spectrum = compute_gain_spectrum(modified_preset, k_values)
    
    return (param_val, gain_spectrum)


def sweep_parameter_parallel(
    preset: Dict,
    param_spec: ParameterSpec,
    param_values: np.ndarray,
    k_values: np.ndarray,
    n_processes: int = None
) -> Dict:
    """Sweep single parameter and compute G(k) using multiprocessing.
    
    Args:
        preset: Base developmental preset
        param_spec: Specification for parameter to sweep
        param_values: Array of parameter values to sweep
        k_values: Array of k values for spectrum
        n_processes: Number of processes (default: cpu_count - 1)
        
    Returns:
        Dict with:
        - param_values: Parameter values
        - k_values: k values
        - gain_matrix: Gain values (n_params × n_k)
        - preset_value: Original preset value
        - param_spec: Parameter specification
    """
    n_params = len(param_values)
    n_k = len(k_values)
    
    gain_matrix = np.full((n_params, n_k), np.nan)
    
    # Get original preset value
    preset_value = get_nested_value(preset, param_spec.path)
    
    print(f"  Sweeping {param_spec.display_name} (preset: {preset_value:.3f})")
    print(f"  Range: [{param_values[0]:.3f}, {param_values[-1]:.3f}]")
    print(f"  Computing {n_params} parameter values × {n_k} k values...")
    
    # Prepare tasks for parallel execution
    tasks = [(preset, param_val, param_spec, k_values) for param_val in param_values]
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 1)
    
    print(f"  Using {n_processes} processes...")
    
    # Execute in parallel
    with mp.Pool(n_processes) as pool:
        results = pool.map(_spectrum_worker, tasks)
    
    # Unpack results into matrix
    for param_val, gain_spectrum in results:
        i = np.argmin(np.abs(param_values - param_val))
        gain_matrix[i, :] = gain_spectrum
    
    return {
        'param_values': param_values,
        'k_values': k_values,
        'gain_matrix': gain_matrix,
        'preset_value': preset_value,
        'param_spec': param_spec,
    }


def compute_gain_spectra_single_stage(
    stage_name: str,
    parameter_keys: List[str],
    n_processes: int = None
) -> Dict:
    """Compute gain spectra for a single developmental stage.
    
    Args:
        stage_name: Developmental stage ('P4', 'P8', 'P12', 'P16')
        parameter_keys: List of parameter keys to sweep
        n_processes: Number of processes for parallelization
        
    Returns:
        Dict mapping parameter keys to their sweep results
    """
    preset = PRESETS[stage_name.upper()]
    
    print(f"\n{'='*70}")
    print(f"Computing gain spectra for {stage_name}")
    print(f"{'='*70}")
    
    # Generate k values
    k_values = get_k_values()
    print(f"Generated {len(k_values)} unique k values (max k = {k_values[-1]:.2f})")
    
    results = {}
    
    for param_key in parameter_keys:
        param_spec = SCANNABLE_PARAMETERS[param_key]
        preset_value = get_nested_value(preset, param_spec.path)
        
        # Generate parameter values: relative range around preset value
        min_factor, max_factor = SPECTRUM_PARAM_SWEEP_RANGE
        param_min = preset_value * min_factor
        param_max = preset_value * max_factor
        param_values = np.linspace(param_min, param_max, SPECTRUM_PARAM_RESOLUTION)
        
        # Sweep this parameter
        sweep_result = sweep_parameter_parallel(
            preset, param_spec, param_values, k_values, n_processes
        )
        
        results[param_key] = sweep_result
    
    return results


def compute_gain_spectra_all_stages(
    parameter_keys: List[str],
    stages: List[str] = None,
    n_processes: int = None
) -> Dict:
    """Compute gain spectra for all stages and parameters.
    
    Args:
        parameter_keys: List of parameter keys to sweep
        stages: List of developmental stages (default: ['P4', 'P8', 'P12', 'P16'])
        n_processes: Number of processes for parallelization
        
    Returns:
        Nested dict: {param_key: {stage: results}}
    """
    if stages is None:
        stages = ['P4', 'P8', 'P12', 'P16']
    
    print("\n" + "="*70)
    print("  GAIN SPECTRA - Computing All Stages")
    print("="*70 + "\n")
    
    # Organize results by parameter, then by stage
    all_results = {param_key: {} for param_key in parameter_keys}
    
    for stage_name in stages:
        stage_results = compute_gain_spectra_single_stage(
            stage_name, parameter_keys, n_processes
        )
        
        # Distribute results into the organized structure
        for param_key, result in stage_results.items():
            all_results[param_key][stage_name] = result
    
    return all_results

