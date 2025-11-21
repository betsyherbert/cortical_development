"""Stability analysis module for bifurcation analysis.

This module implements stability map computations, scanning parameter spaces to
find critical spatial modes and bifurcation boundaries through eigenvalue analysis.
"""

import numpy as np
import multiprocessing as mp
import os
import copy
from typing import Dict, Tuple, List
from pathlib import Path

from .core import NetworkModel, SteadyStateFinder, StabilityAnalyzer, set_nested_value, get_nested_value
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
)
from src.analysis.common import PRESETS


def compute_stability_for_point(preset: Dict, verbose: bool = False) -> Tuple[float, float, bool]:
    """Compute stability spectrum for a single parameter point.
    
    Args:
        preset: Network preset dictionary with parameters
        verbose: If True, print diagnostic information
        
    Returns:
        Tuple of (k_critical, max_real_eigenvalue, is_flat)
        - k_critical: Wavenumber with maximum Re(λ)
        - max_real_eigenvalue: Maximum Re(λ) across all k
        - is_flat: True if spectrum is flat (all k have similar Re(λ))
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
    n_modes_effective = min(n_modes, int(0.6 * grid_size))
    
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
                sigma_ij = network.sigma[i, j] / grid_size
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
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(J)
            max_real = np.max(eigenvalues.real)
            
            # Store or update max real eigenvalue for this k
            if k_squared not in results_by_k2:
                results_by_k2[k_squared] = {'k': k, 'max_real': max_real}
            else:
                if max_real > results_by_k2[k_squared]['max_real']:
                    results_by_k2[k_squared]['max_real'] = max_real
    
    # Find critical k and check for flat spectrum
    if results_by_k2:
        max_entry = max(results_by_k2.values(), key=lambda x: x['max_real'])
        min_entry = min(results_by_k2.values(), key=lambda x: x['max_real'])
        
        k_critical = max_entry['k']
        max_real_eigenvalue = max_entry['max_real']
        
        spectrum_range = max_entry['max_real'] - min_entry['max_real']
        is_flat = spectrum_range < 0.001
    else:
        k_critical = 0.0
        max_real_eigenvalue = -np.inf
        is_flat = False
    
    return k_critical, max_real_eigenvalue, is_flat


def _stability_worker(args: Tuple) -> Tuple[float, float, Tuple[float, float, bool]]:
    """Worker function for parallel stability computation.
    
    Args:
        args: Tuple of (preset, x_val, y_val, param_x_spec, param_y_spec)
        
    Returns:
        Tuple of (x_val, y_val, result) where result is from compute_stability_for_point
    """
    preset, x_val, y_val, param_x_spec, param_y_spec = args
    
    # Modify preset
    modified_preset = copy.deepcopy(preset)
    set_nested_value(modified_preset, param_x_spec.path, x_val)
    set_nested_value(modified_preset, param_y_spec.path, y_val)
    
    # Compute stability
    result = compute_stability_for_point(modified_preset)
    
    return (x_val, y_val, result)


def scan_parameter_space_parallel(
    preset: Dict,
    param_x_spec: ParameterSpec,
    param_y_spec: ParameterSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_processes: int = None
) -> Dict:
    """Scan 2D parameter space for stability using multiprocessing.
    
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
        - stability_matrix: Max Re(λ) values (n_y × n_x)
        - flatness_matrix: Flat spectrum flags (n_y × n_x)
        - param_x_values: x parameter values
        - param_y_values: y parameter values
        - param_x_spec: x parameter specification
        - param_y_spec: y parameter specification
    """
    n_x = len(x_values)
    n_y = len(y_values)
    
    k_matrix = np.zeros((n_y, n_x))
    stability_matrix = np.zeros((n_y, n_x))
    flatness_matrix = np.zeros((n_y, n_x), dtype=bool)
    
    # Prepare tasks for parallel execution
    tasks = []
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            tasks.append((preset, x_val, y_val, param_x_spec, param_y_spec))
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 1)
    
    print(f"  Running {len(tasks)} stability computations using {n_processes} processes...")
    
    # Execute in parallel
    with mp.Pool(n_processes) as pool:
        results = pool.map(_stability_worker, tasks)
    
    # Unpack results into matrices
    for x_val, y_val, (k_crit, max_real, is_flat) in results:
        # Find indices
        i = np.argmin(np.abs(y_values - y_val))
        j = np.argmin(np.abs(x_values - x_val))
        
        k_matrix[i, j] = k_crit
        stability_matrix[i, j] = max_real
        flatness_matrix[i, j] = is_flat
    
    return {
        'k_matrix': k_matrix,
        'stability_matrix': stability_matrix,
        'flatness_matrix': flatness_matrix,
        'param_x_values': x_values,
        'param_y_values': y_values,
        'param_x_spec': param_x_spec,
        'param_y_spec': param_y_spec,
    }


def compute_stability_maps_single_stage(
    stage_name: str,
    parameter_pairs: List[Tuple[str, str]],
    mode: str = 'fixed_absolute',
    n_processes: int = None
) -> Dict:
    """Compute stability maps for a single developmental stage.
    
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
    print(f"Computing stability maps for {stage_name}")
    print(f"{'='*70}")
    
    for param_x_key, param_y_key in parameter_pairs:
        param_x_spec = SCANNABLE_PARAMETERS[param_x_key]
        param_y_spec = SCANNABLE_PARAMETERS[param_y_key]
        
        print(f"\nScanning {param_x_key} vs {param_y_key}...")
        
        # Determine parameter ranges based on mode
        if mode == 'fixed_absolute':
            # Use absolute ranges
            x_min, x_max = param_x_spec.default_range
            y_min, y_max = param_y_spec.default_range
        elif mode == 'fixed_ratio':
            # Use ratio ranges if applicable, otherwise absolute
            if param_x_spec.use_ratio and param_x_spec.reference_param:
                ref_spec = SCANNABLE_PARAMETERS[param_x_spec.reference_param]
                ref_value = get_nested_value(preset, ref_spec.path)
                # Apply fixed ratio ranges
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
        
        # Compute stability at exact preset parameters
        k_preset, stability_preset, flat_preset = compute_stability_for_point(preset)
        result['preset_k'] = k_preset
        result['preset_stability'] = stability_preset
        result['preset_flat'] = flat_preset
        
        print(f"  Preset values: k={k_preset:.3f}, Re(λ)_max={stability_preset:.6f}, flat={flat_preset}")
        
        results[(param_x_key, param_y_key)] = result
    
    return results


def compute_stability_maps_all_stages(
    parameter_pairs: List[Tuple[str, str]],
    stages: List[str] = None,
    mode: str = 'fixed_absolute',
    n_processes: int = None
) -> Dict:
    """Compute stability maps for all stages and parameter pairs.
    
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
    print(f"  STABILITY MAPS - Computing All Stages ({mode})")
    print("="*70 + "\n")
    
    # Organize results by parameter pair, then by stage
    all_results = {pair: {} for pair in parameter_pairs}
    
    for stage_name in stages:
        stage_results = compute_stability_maps_single_stage(
            stage_name, parameter_pairs, mode, n_processes
        )
        
        # Distribute results into the organized structure
        for pair, result in stage_results.items():
            all_results[pair][stage_name] = result
    
    return all_results

