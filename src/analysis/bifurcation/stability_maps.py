"""Stability analysis module for bifurcation analysis.

This module implements stability map computations, scanning parameter spaces to
find critical spatial modes and bifurcation boundaries through eigenvalue analysis.

Note: All spatial parameters (sigma, wavelength) are in μm (anatomical units).
Wavenumber k is in cycles/μm.
"""

import copy
import multiprocessing as mp
import os

import numpy as np

from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS

from .config import (
    ALL_LAYERS,
    ANALYSIS_PARAMS,
    FIXED_RATIO_SIGMA_MAX,
    FIXED_RATIO_SIGMA_MIN,
    FIXED_RATIO_TAU_MAX,
    FIXED_RATIO_TAU_MIN,
    GRID_RESOLUTION,
    MATURITY_REFERENCE_VALUES,
    MATURITY_SCAN_MARGIN,
    SCANNABLE_PARAMETERS,
    THALAMIC_MAGNITUDE,
    ParameterSpec,
)
from .core import (
    NetworkModel,
    StabilityAnalyzer,
    SteadyStateFinder,
    get_nested_value,
    set_nested_value,
)


def compute_stability_for_point(preset: dict, verbose: bool = False) -> tuple[float, float, bool]:
    """Compute stability spectrum for a single parameter point.

    Args:
        preset: Network preset dictionary with parameters
        verbose: If True, print diagnostic information (currently unused)

    Returns:
        Tuple of (k_critical, max_real_eigenvalue, is_flat)
        - k_critical: Wavenumber with maximum Re(λ) (NaN if steady state failed)
        - max_real_eigenvalue: Maximum Re(λ) across all k (NaN if steady state failed)
        - is_flat: True if spectrum is flat (all k have similar Re(λ))
    """
    _ = verbose  # Suppress unused argument warning (reserved for future use)
    network = NetworkModel(preset, layers=ALL_LAYERS)
    finder = SteadyStateFinder(network)

    # Use consistent thalamic input magnitude from config
    thalamic_input = network.compute_thalamic_input(THALAMIC_MAGNITUDE)

    # Find steady state
    r_star, status = finder.find_steady_state(thalamic_input=thalamic_input)

    # Handle failed convergence: return NaN to mark invalid parameter point
    # Accept both 'converged' and 'approximate' as valid operating points
    if status in ["diverged", "not_converged"] or np.any(r_star > 100):
        return np.nan, np.nan, False

    analyzer = StabilityAnalyzer(network, r_star)

    # Get analysis parameters
    n_modes = ANALYSIS_PARAMS["n_modes"]
    grid_size = ANALYSIS_PARAMS["grid_size"]
    anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
    n_modes_effective = min(n_modes, int(0.6 * grid_size))

    total_pops = len(network.tau)

    # Pre-compute unique k² values
    k_squared_set = set()
    for n1 in range(0, n_modes_effective + 1):
        for n2 in range(0, n_modes_effective + 1):
            k_squared_set.add(n1**2 + n2**2)

    # Cache exponential factors for spatial filtering
    # sigma values are in μm, normalize by anatomical_grid_size (also in μm)
    exp_cache = {}
    for k_squared in k_squared_set:
        exp_cache[k_squared] = np.zeros((total_pops, total_pops))
        for i in range(total_pops):
            for j in range(total_pops):
                sigma_ij = network.sigma[i, j] / anatomical_grid_size
                exp_cache[k_squared][i, j] = np.exp(-2 * np.pi**2 * k_squared * sigma_ij**2)

    # Dictionary to store results by k²
    results_by_k2 = {}

    # Scan positive quadrant only
    for n1 in range(0, n_modes_effective + 1):
        for n2 in range(0, n_modes_effective + 1):
            k_squared = n1**2 + n2**2
            k_mode = np.sqrt(k_squared)  # Mode number (dimensionless)

            if k_mode > n_modes:
                continue

            # Build Jacobian using cached exponentials
            J = np.zeros((total_pops, total_pops))
            exp_factors = exp_cache[k_squared]

            for i in range(total_pops):
                for j in range(total_pops):
                    w_tilde = network.A[i, j] * exp_factors[i, j]
                    if i == j:
                        J[i, j] = (
                            -1.0 / network.tau[i] + (analyzer.g_eff[i] * w_tilde) / network.tau[i]
                        )
                    else:
                        J[i, j] = (analyzer.g_eff[i] * w_tilde) / network.tau[i]

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(J)
            max_real = np.max(eigenvalues.real)

            # Store or update max real eigenvalue for this k (k_mode is mode number)
            if k_squared not in results_by_k2:
                results_by_k2[k_squared] = {"k_mode": k_mode, "max_real": max_real}
            else:
                if max_real > results_by_k2[k_squared]["max_real"]:
                    results_by_k2[k_squared]["max_real"] = max_real

    # Find critical k and check for flat spectrum
    if results_by_k2:
        max_entry = max(results_by_k2.values(), key=lambda x: x["max_real"])
        min_entry = min(results_by_k2.values(), key=lambda x: x["max_real"])

        # Convert k from mode number to cycles/μm
        k_critical = max_entry["k_mode"] / anatomical_grid_size  # cycles/μm
        max_real_eigenvalue = max_entry["max_real"]

        spectrum_range = max_entry["max_real"] - min_entry["max_real"]
        is_flat = spectrum_range < 0.01
    else:
        k_critical = 0.0
        max_real_eigenvalue = -np.inf
        is_flat = False

    return k_critical, max_real_eigenvalue, is_flat


def _stability_worker(args: tuple) -> tuple[float, float, tuple[float, float, bool]]:
    """Worker function for parallel stability computation.

    Args:
        args: Tuple of (preset, x_val, y_val, param_x_spec, param_y_spec)

    Returns:
        Tuple of (x_val, y_val, result) where result is from compute_stability_for_point
    """
    preset, x_val, y_val, param_x_spec, param_y_spec = args

    modified_preset = copy.deepcopy(preset)

    for val, spec in [(x_val, param_x_spec), (y_val, param_y_spec)]:
        if getattr(spec, "is_derived_ratio", False) and getattr(spec, "base_param", None):
            base_spec = SCANNABLE_PARAMETERS[spec.base_param]
            base_value = get_nested_value(preset, base_spec.path)
            actual_value = val * base_value
            set_nested_value(modified_preset, spec.path, actual_value)
        else:
            set_nested_value(modified_preset, spec.path, val)

    result = compute_stability_for_point(modified_preset)
    return (x_val, y_val, result)


def scan_parameter_space_parallel(
    preset: dict,
    param_x_spec: ParameterSpec,
    param_y_spec: ParameterSpec,
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_processes: int = None,
) -> dict:
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

    # Unpack results into matrices (NaN values will be preserved)
    for x_val, y_val, (k_crit, max_real, is_flat) in results:
        # Find indices
        i = np.argmin(np.abs(y_values - y_val))
        j = np.argmin(np.abs(x_values - x_val))

        k_matrix[i, j] = k_crit
        stability_matrix[i, j] = max_real
        flatness_matrix[i, j] = is_flat

    return {
        "k_matrix": k_matrix,
        "stability_matrix": stability_matrix,
        "flatness_matrix": flatness_matrix,
        "param_x_values": x_values,
        "param_y_values": y_values,
        "param_x_spec": param_x_spec,
        "param_y_spec": param_y_spec,
    }


def compute_stability_maps_single_stage(
    stage_name: str,
    parameter_pairs: list[tuple[str, str]],
    mode: str = "fixed_absolute",
    n_processes: int = None,
) -> dict:
    """Compute stability maps for a single developmental stage.

    Args:
        stage_name: Developmental stage ('P0', 'P5', 'P10', 'P15')
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

        # Determine parameter ranges
        # Derived ratio params and fixed_absolute mode: use default_range
        # Fixed_ratio mode: compute from reference param
        def _get_range(spec, param_key):
            if getattr(spec, "is_derived_ratio", False) or mode == "fixed_absolute":
                return spec.default_range
            if mode == "fixed_ratio" and spec.use_ratio and spec.reference_param:
                ref_spec = SCANNABLE_PARAMETERS[spec.reference_param]
                ref_value = get_nested_value(preset, ref_spec.path)
                if "tau" in param_key:
                    return (FIXED_RATIO_TAU_MIN * ref_value, FIXED_RATIO_TAU_MAX * ref_value)
                return (FIXED_RATIO_SIGMA_MIN * ref_value, FIXED_RATIO_SIGMA_MAX * ref_value)
            if mode == "fixed_ratio":
                return spec.default_range
            raise ValueError(f"Unknown mode: {mode}")

        x_min, x_max = _get_range(param_x_spec, param_x_key)
        y_min, y_max = _get_range(param_y_spec, param_y_key)

        # Generate parameter grids
        x_values = np.linspace(x_min, x_max, GRID_RESOLUTION)
        y_values = np.linspace(y_min, y_max, GRID_RESOLUTION)

        print(f"  {param_x_key} range: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  {param_y_key} range: [{y_min:.3f}, {y_max:.3f}]")
        print(f"  Grid: {GRID_RESOLUTION}×{GRID_RESOLUTION} = {GRID_RESOLUTION**2} points")

        # Scan parameter space with parallelization
        result = scan_parameter_space_parallel(
            preset, param_x_spec, param_y_spec, x_values, y_values, n_processes
        )

        # Add preset value information
        result["preset"] = preset
        if getattr(param_x_spec, "is_derived_ratio", False) and param_x_spec.base_param:
            base_spec = SCANNABLE_PARAMETERS[param_x_spec.base_param]
            result["preset_x_value"] = get_nested_value(preset, param_x_spec.path) / get_nested_value(
                preset, base_spec.path
            )
        else:
            result["preset_x_value"] = get_nested_value(preset, param_x_spec.path)
        if getattr(param_y_spec, "is_derived_ratio", False) and param_y_spec.base_param:
            base_spec = SCANNABLE_PARAMETERS[param_y_spec.base_param]
            result["preset_y_value"] = get_nested_value(preset, param_y_spec.path) / get_nested_value(
                preset, base_spec.path
            )
        else:
            result["preset_y_value"] = get_nested_value(preset, param_y_spec.path)

        # Compute stability at exact preset parameters
        k_preset, stability_preset, flat_preset = compute_stability_for_point(preset)
        result["preset_k"] = k_preset
        result["preset_stability"] = stability_preset
        result["preset_flat"] = flat_preset

        print(
            f"  Preset values: k={k_preset:.3f}, Re(λ)_max={stability_preset:.6f}, flat={flat_preset}"
        )

        results[(param_x_key, param_y_key)] = result

    return results


def compute_stability_maps_all_stages(
    parameter_pairs: list[tuple[str, str]],
    stages: list[str] = None,
    mode: str = "fixed_absolute",
    n_processes: int = None,
) -> dict:
    """Compute stability maps for all stages and parameter pairs.

    Args:
        parameter_pairs: List of (param_x_key, param_y_key) tuples to scan
        stages: List of developmental stages (default: ['P0', 'P5', 'P10', 'P15'])
        mode: Range mode ('fixed_absolute' or 'fixed_ratio')
        n_processes: Number of processes for parallelization

    Returns:
        Nested dict: {param_pair: {stage: results}}
    """
    if stages is None:
        stages = DEVELOPMENTAL_STAGES

    print("\n" + "=" * 70)
    print(f"  STABILITY MAPS - Computing All Stages ({mode})")
    print("=" * 70 + "\n")

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


# ============================================================================
# Maturity Index Stability Maps
# ============================================================================


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b with parameter t in [0, 1]."""
    return a + t * (b - a)


def compute_maturity_from_preset(preset: dict, cell_type: str) -> float:
    """Compute maturity index for SST or PV from a preset.

    Maturity is the average of three normalized components:
    - tau_ratio: normalized τ_X/τ_E
    - sigma_ratio: normalized σ_X/σ_E
    - strength: normalized strength_scaling

    Each component is 0 at P0 reference and 1 at P15 reference.

    Args:
        preset: Developmental preset dictionary
        cell_type: 'SST' or 'PV'

    Returns:
        Maturity index in [0, 1] (approximately, can exceed bounds for non-preset values)
    """
    ref = MATURITY_REFERENCE_VALUES[cell_type]

    # Get current values from preset
    tau_E = preset["time_constants"]["E"]
    tau_X = preset["time_constants"][cell_type]
    sigma_E = preset["outgoing_widths"]["E"]
    sigma_X = preset["outgoing_widths"][cell_type]
    strength_X = preset["strength_scaling"][cell_type]

    # Compute current ratios
    tau_ratio = tau_X / tau_E
    sigma_ratio = sigma_X / sigma_E

    # Normalize each component to [0, 1] based on immature→mature progression
    tau_imm, tau_mat = ref["tau_ratio"]["immature"], ref["tau_ratio"]["mature"]
    sigma_imm, sigma_mat = ref["sigma_ratio"]["immature"], ref["sigma_ratio"]["mature"]
    strength_imm, strength_mat = ref["strength"]["immature"], ref["strength"]["mature"]

    # Handle direction of change (some increase, some decrease with maturity)
    if tau_mat != tau_imm:
        norm_tau = (tau_ratio - tau_imm) / (tau_mat - tau_imm)
    else:
        norm_tau = 0.5

    if sigma_mat != sigma_imm:
        norm_sigma = (sigma_ratio - sigma_imm) / (sigma_mat - sigma_imm)
    else:
        norm_sigma = 0.5

    if strength_mat != strength_imm:
        norm_strength = (strength_X - strength_imm) / (strength_mat - strength_imm)
    else:
        norm_strength = 0.5

    # Average the three components with equal weights
    maturity = (norm_tau + norm_sigma + norm_strength) / 3.0

    return maturity


def compute_parameters_from_maturity(
    maturity: float, cell_type: str, tau_E: float, sigma_E: float
) -> tuple[float, float, float]:
    """Compute absolute parameters from a maturity index.

    This is the inverse of compute_maturity_from_preset: given a maturity value,
    compute the corresponding τ, σ, and strength values.

    All three parameters are interpolated in lockstep (same maturity → same
    normalized position for all three).

    Args:
        maturity: Maturity index (typically in [0, 1])
        cell_type: 'SST' or 'PV'
        tau_E: Stage-specific τ_E value (ms)
        sigma_E: Stage-specific σ_E value (μm)

    Returns:
        Tuple of (tau_X, sigma_X, strength_X)
    """
    ref = MATURITY_REFERENCE_VALUES[cell_type]

    # Interpolate ratios and strength based on maturity (immature=0, mature=1)
    tau_ratio = _lerp(ref["tau_ratio"]["immature"], ref["tau_ratio"]["mature"], maturity)
    sigma_ratio = _lerp(ref["sigma_ratio"]["immature"], ref["sigma_ratio"]["mature"], maturity)
    strength = _lerp(ref["strength"]["immature"], ref["strength"]["mature"], maturity)

    # Convert ratios to absolute values using stage-specific E parameters
    tau_X = tau_E * tau_ratio
    sigma_X = sigma_E * sigma_ratio

    return tau_X, sigma_X, strength


def _maturity_stability_worker(args: tuple) -> tuple[float, float, tuple[float, float, bool]]:
    """Worker function for maturity stability map computation.

    Args:
        args: Tuple of (preset, sst_maturity, pv_maturity, tau_E, sigma_E)

    Returns:
        Tuple of (sst_maturity, pv_maturity, result) where result is from compute_stability_for_point
    """
    preset, sst_maturity, pv_maturity, tau_E, sigma_E = args

    # Compute absolute parameters from maturity indices
    tau_SST, sigma_SST, strength_SST = compute_parameters_from_maturity(
        sst_maturity, "SST", tau_E, sigma_E
    )
    tau_PV, sigma_PV, strength_PV = compute_parameters_from_maturity(
        pv_maturity, "PV", tau_E, sigma_E
    )

    # Modify preset with computed parameters
    modified_preset = copy.deepcopy(preset)
    modified_preset["time_constants"]["SST"] = tau_SST
    modified_preset["time_constants"]["PV"] = tau_PV
    modified_preset["outgoing_widths"]["SST"] = sigma_SST
    modified_preset["outgoing_widths"]["PV"] = sigma_PV
    modified_preset["strength_scaling"]["SST"] = strength_SST
    modified_preset["strength_scaling"]["PV"] = strength_PV

    # Compute stability
    result = compute_stability_for_point(modified_preset)

    return (sst_maturity, pv_maturity, result)


def compute_maturity_stability_maps_all_stages(
    stages: list[str] = None,
    n_processes: int = None,
) -> dict:
    """Compute stability maps as a function of SST and PV maturity indices.

    For each developmental stage, uses that stage's E parameters as baseline
    and scans a grid of (SST_maturity, PV_maturity) values centered on the
    stage's natural maturity.

    Args:
        stages: List of developmental stages (default: ['P0', 'P5', 'P10', 'P15'])
        n_processes: Number of processes for parallelization

    Returns:
        Dict mapping stage names to results dicts with:
        - k_matrix: Critical wavenumbers (n_y × n_x)
        - stability_matrix: Max Re(λ) values (n_y × n_x)
        - flatness_matrix: Flat spectrum flags (n_y × n_x)
        - sst_maturity_values: SST maturity values (x-axis)
        - pv_maturity_values: PV maturity values (y-axis)
        - preset: Original preset dict
        - preset_sst_maturity: Preset SST maturity value
        - preset_pv_maturity: Preset PV maturity value
        - sst_maturity_range: (min, max) range scanned for SST
        - pv_maturity_range: (min, max) range scanned for PV
    """
    if stages is None:
        stages = DEVELOPMENTAL_STAGES

    print("\n" + "=" * 70)
    print("  MATURITY INDEX STABILITY MAPS")
    print("=" * 70 + "\n")

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 1)

    all_results = {}

    for stage_name in stages:
        preset = PRESETS[stage_name.upper()]

        print(f"\n{'='*70}")
        print(f"Computing maturity stability map for {stage_name}")
        print(f"{'='*70}")

        # Get stage-specific E parameters
        tau_E = preset["time_constants"]["E"]
        sigma_E = preset["outgoing_widths"]["E"]

        # Compute natural maturity for this stage
        preset_sst_maturity = compute_maturity_from_preset(preset, "SST")
        preset_pv_maturity = compute_maturity_from_preset(preset, "PV")

        print(f"  Stage E parameters: τ_E={tau_E} ms, σ_E={sigma_E} μm")
        print(f"  Natural SST maturity: {preset_sst_maturity:.3f}")
        print(f"  Natural PV maturity: {preset_pv_maturity:.3f}")

        # Determine scan range centered on natural maturity, clamped to [0, 1]
        sst_min = max(0.0, preset_sst_maturity - MATURITY_SCAN_MARGIN)
        sst_max = min(1.0, preset_sst_maturity + MATURITY_SCAN_MARGIN)
        pv_min = max(0.0, preset_pv_maturity - MATURITY_SCAN_MARGIN)
        pv_max = min(1.0, preset_pv_maturity + MATURITY_SCAN_MARGIN)

        print(f"  Scanning SST maturity ∈ [{sst_min:.2f}, {sst_max:.2f}]")
        print(f"  Scanning PV maturity ∈ [{pv_min:.2f}, {pv_max:.2f}]")
        print(f"  Grid: {GRID_RESOLUTION}×{GRID_RESOLUTION} = {GRID_RESOLUTION**2} points")

        # Generate maturity grids
        sst_maturity_values = np.linspace(sst_min, sst_max, GRID_RESOLUTION)
        pv_maturity_values = np.linspace(pv_min, pv_max, GRID_RESOLUTION)

        n_x = len(sst_maturity_values)
        n_y = len(pv_maturity_values)

        # Prepare tasks for parallel execution
        tasks = []
        for pv_mat in pv_maturity_values:
            for sst_mat in sst_maturity_values:
                tasks.append((preset, sst_mat, pv_mat, tau_E, sigma_E))

        print(f"  Running {len(tasks)} stability computations using {n_processes} processes...")

        # Execute in parallel
        with mp.Pool(n_processes) as pool:
            results = pool.map(_maturity_stability_worker, tasks)

        # Unpack results into matrices
        k_matrix = np.zeros((n_y, n_x))
        stability_matrix = np.zeros((n_y, n_x))
        flatness_matrix = np.zeros((n_y, n_x), dtype=bool)

        for sst_mat, pv_mat, (k_crit, max_real, is_flat) in results:
            # Find indices
            i = np.argmin(np.abs(pv_maturity_values - pv_mat))
            j = np.argmin(np.abs(sst_maturity_values - sst_mat))

            k_matrix[i, j] = k_crit
            stability_matrix[i, j] = max_real
            flatness_matrix[i, j] = is_flat

        # Compute stability at exact preset parameters
        k_preset, stability_preset, flat_preset = compute_stability_for_point(preset)

        print(
            f"  Preset stability: k={k_preset:.3f}, Re(λ)_max={stability_preset:.6f}, flat={flat_preset}"
        )

        # Store results
        all_results[stage_name] = {
            "k_matrix": k_matrix,
            "stability_matrix": stability_matrix,
            "flatness_matrix": flatness_matrix,
            "sst_maturity_values": sst_maturity_values,
            "pv_maturity_values": pv_maturity_values,
            "preset": preset,
            "preset_sst_maturity": preset_sst_maturity,
            "preset_pv_maturity": preset_pv_maturity,
            "sst_maturity_range": (sst_min, sst_max),
            "pv_maturity_range": (pv_min, pv_max),
            "preset_k": k_preset,
            "preset_stability": stability_preset,
            "preset_flat": flat_preset,
        }

    return all_results
