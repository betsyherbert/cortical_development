"""Developmental bifurcation maps showing parameter space stability landscapes.

This module creates bifurcation diagrams in (τ/τ_E, σ/σ_E) parameter space for
SST and PV interneurons across developmental stages.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import copy
import pickle

from .bifurcation_analysis import NetworkModel, StabilityAnalyzer
from .config import (
    ANALYSIS_PARAMS,
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
    BIFURCATION_COLORMAP,
    OPACITY_STABLE_FAR,
    OPACITY_STABLE_NEAR,
    OPACITY_UNSTABLE,
    STABILITY_THRESHOLD,
    OUTPUT_DIR,
    ALL_LAYERS,
)
from src.analysis.common import PRESETS


def compute_stability_for_point(preset: Dict) -> Tuple[float, float, bool]:
    """
    Compute stability spectrum for a single parameter point.
    
    Args:
        preset: Network preset dictionary with parameters
        
    Returns:
        Tuple of (k_critical, max_real_eigenvalue, is_flat)
        - k_critical: Wavenumber with maximum Re(λ)
        - max_real_eigenvalue: Maximum Re(λ) across all k
        - is_flat: True if spectrum is flat (all k have similar Re(λ))
    """
    from .bifurcation_analysis import SteadyStateFinder
    
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


def scan_parameter_space(preset: Dict, cell_type: str, 
                        tau_ratios: np.ndarray, sigma_ratios: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan parameter space by varying one interneuron type's parameters.
    
    Args:
        preset: Base developmental preset
        cell_type: Which interneuron to vary ('SST' or 'PV')
        tau_ratios: Array of τ_inh / τ_E ratios to scan
        sigma_ratios: Array of σ_inh / σ_E ratios to scan
        
    Returns:
        Tuple of (k_matrix, stability_matrix, flatness_matrix)
    """
    n_tau = len(tau_ratios)
    n_sigma = len(sigma_ratios)
    
    k_matrix = np.zeros((n_tau, n_sigma))
    stability_matrix = np.zeros((n_tau, n_sigma))
    flatness_matrix = np.zeros((n_tau, n_sigma), dtype=bool)
    
    # Get reference E parameters
    tau_E = preset['time_constants']['E']
    sigma_E = preset['outgoing_widths']['E']
    
    print(f"\nScanning {cell_type} parameter space ({n_tau}×{n_sigma} = {n_tau*n_sigma} points)...")
    
    for i, tau_ratio in enumerate(tau_ratios):
        if i % 5 == 0:
            print(f"  Progress: {i}/{n_tau} tau ratios completed")
        
        for j, sigma_ratio in enumerate(sigma_ratios):
            # Create modified preset
            modified_preset = copy.deepcopy(preset)
            modified_preset['time_constants'][cell_type] = tau_ratio * tau_E
            modified_preset['outgoing_widths'][cell_type] = sigma_ratio * sigma_E
            
            # Compute stability for this point (steady state computed internally)
            k_crit, max_real, is_flat = compute_stability_for_point(modified_preset)
            
            k_matrix[i, j] = k_crit
            stability_matrix[i, j] = max_real
            flatness_matrix[i, j] = is_flat
    
    print(f"  Completed {cell_type} scan!")
    
    return k_matrix, stability_matrix, flatness_matrix


def compute_bifurcation_maps(stage_name: str, tau_ratios: np.ndarray, 
                             sigma_ratios: np.ndarray, 
                             seed: int = None) -> Dict:
    """
    Compute bifurcation maps for both SST and PV at a developmental stage.
    
    Args:
        stage_name: Developmental stage ('P4', 'P8', 'P12', 'P16')
        tau_ratios: Array of τ_inh / τ_E ratios
        sigma_ratios: Array of σ_inh / σ_E ratios
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results for SST and PV
    """
    preset = PRESETS[stage_name.upper()]
    
    print(f"\n{'='*60}")
    print(f"Computing bifurcation maps for {stage_name}")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility in steady state finding
    if seed is not None:
        from src.model.config import seed_random
        seed_random(seed)
    
    # Compute stability at exact preset parameters (the "star" point)
    print("\nComputing stability at exact preset parameters...")
    k_preset, stability_preset, flat_preset = compute_stability_for_point(preset)
    print(f"  Preset: k_crit = {k_preset:.3f}, Re(λ)_max = {stability_preset:.6f}, flat = {flat_preset}")
    
    # Scan SST parameter space
    k_sst, stability_sst, flat_sst = scan_parameter_space(
        preset, 'SST', tau_ratios, sigma_ratios
    )
    
    # Scan PV parameter space
    k_pv, stability_pv, flat_pv = scan_parameter_space(
        preset, 'PV', tau_ratios, sigma_ratios
    )
    
    return {
        'preset': preset,
        'preset_k': k_preset,
        'preset_stability': stability_preset,
        'preset_flat': flat_preset,
        'sst_k': k_sst,
        'sst_stability': stability_sst,
        'sst_flat': flat_sst,
        'pv_k': k_pv,
        'pv_stability': stability_pv,
        'pv_flat': flat_pv,
        'tau_ratios': tau_ratios,
        'sigma_ratios': sigma_ratios,
    }


def create_combined_figure(
    all_results: Dict[str, Dict],
    output_path: Path,
    mode: str = 'fixed_absolute',
) -> None:
    """
    Create combined bifurcation diagram with all developmental stages.
    
    Args:
        all_results: Dictionary mapping stage names to their results
        output_path: Path to save figure
        mode: Which range mode produced these results (fixed_absolute or fixed_ratio)
    """
    stages = ['P4', 'P8', 'P12', 'P16']
    n_stages = len(stages)
    cell_types = ['SST', 'PV']
    n_cell_types = len(cell_types)
    
    # Determine which axes to emphasize (ratio axes or absolute axes)
    emphasize_ratio_axes = mode == 'fixed_ratio'
    emphasize_absolute_axes = mode == 'fixed_absolute'
    
    default_spine_width = 0.8
    bold_spine_width = 1.6
    primary_spine_width = bold_spine_width if emphasize_ratio_axes else default_spine_width
    secondary_spine_width = bold_spine_width if emphasize_absolute_axes else default_spine_width
    
    # Create figure with 2×4 grid (2 cell types × 4 stages)
    # Use GridSpec for precise control over subplot layout
    from matplotlib.gridspec import GridSpec
    
    # Layout: 2 rows (SST top, PV bottom) × 4 columns (P4, P8, P12, P16)
    # Figure dimensions calculated to make subplots perfectly square
    fig_width = 13.7
    fig_height = 8.5
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_cell_types, n_stages, figure=fig,
                  hspace=0.45, wspace=0.18,
                  left=0.07, right=0.83, top=0.74, bottom=0.08,
                  height_ratios=[1]*n_cell_types, width_ratios=[1]*n_stages)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_stages)] for i in range(n_cell_types)])
    
    # Determine global k range for consistent colormap
    all_k_values = []
    for stage_name in stages:
        if stage_name in all_results:
            all_k_values.append(all_results[stage_name]['sst_k'])
            all_k_values.append(all_results[stage_name]['pv_k'])
    
    if all_k_values:
        k_min = 0
        k_max = min(np.max([np.max(k) for k in all_k_values]), 5.0)
    else:
        k_min, k_max = 0, 5.0
    
    norm = colors.Normalize(vmin=k_min, vmax=k_max)
    cmap = plt.colormaps[BIFURCATION_COLORMAP]
    
    # Process all combinations of cell types and stages
    for cell_type_idx, cell_type in enumerate(cell_types):
        for stage_idx, stage_name in enumerate(stages):
            if stage_name not in all_results:
                print(f"Warning: {stage_name} results not found, skipping")
                continue
                
            results = all_results[stage_name]
            preset = results['preset']
            tau_ratios = results['tau_ratios']
            sigma_ratios = results['sigma_ratios']
            
            ax = axes[cell_type_idx, stage_idx]
            k_matrix = results[f'{cell_type.lower()}_k']
            stability_matrix = results[f'{cell_type.lower()}_stability']
            
            # Get flatness matrix (for backwards compatibility with old results)
            flat_key = f'{cell_type.lower()}_flat'
            flatness_matrix = results.get(flat_key, None)
            
            # Transpose matrices to swap axes: original is (n_tau, n_sigma), we want (n_sigma, n_tau)
            k_matrix_T = k_matrix.T
            stability_matrix_T = stability_matrix.T
            flatness_matrix_T = flatness_matrix.T if flatness_matrix is not None else None
            
            # Compute alpha values based on stability
            alpha_matrix = np.zeros_like(stability_matrix_T)
            alpha_matrix[stability_matrix_T < STABILITY_THRESHOLD] = OPACITY_STABLE_FAR
            alpha_matrix[(stability_matrix_T >= STABILITY_THRESHOLD) & (stability_matrix_T < 0)] = OPACITY_STABLE_NEAR
            alpha_matrix[stability_matrix_T >= 0] = OPACITY_UNSTABLE
            
            # Create RGBA image (grey for flat spectra, colormap otherwise)
            rgba_image = np.zeros((*k_matrix_T.shape, 4))
            grey_color = (0.5, 0.5, 0.5)
            for i in range(k_matrix_T.shape[0]):
                for j in range(k_matrix_T.shape[1]):
                    if flatness_matrix_T is not None and flatness_matrix_T[i, j]:
                        color_rgb = grey_color
                    else:
                        color_rgb = cmap(norm(k_matrix_T[i, j]))[:3]
                    rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])
            
            # Display image (swapped axes: x=tau, y=sigma)
            extent = [tau_ratios[0], tau_ratios[-1], sigma_ratios[0], sigma_ratios[-1]]
            ax.imshow(rgba_image, origin='lower', extent=extent, interpolation='nearest')
            
            # Add contour at Re(λ) = 0 (stability boundary)
            try:
                ax.contour(tau_ratios, sigma_ratios, stability_matrix_T,
                          levels=[0], colors='white', linewidths=1.5, linestyles='--', alpha=0.8)
            except (ValueError, RuntimeError):
                pass
            
            # Mark developmental preset point (swapped coordinates)
            actual_tau_ratio = preset['time_constants'][cell_type] / preset['time_constants']['E']
            actual_sigma_ratio = preset['outgoing_widths'][cell_type] / preset['outgoing_widths']['E']
            ax.scatter(actual_tau_ratio, actual_sigma_ratio,
                      marker='o', s=120, edgecolor='black', linewidth=1.5,
                      facecolor='white', zorder=10)
            
            ax.set_xlim(tau_ratios[0], tau_ratios[-1])
            ax.set_ylim(sigma_ratios[0], sigma_ratios[-1])
            ax.set_aspect('auto')
            ax.locator_params(axis='x', nbins=4)
            ax.locator_params(axis='y', nbins=5)
            
            # Primary axes spines (ratio axes)
            ax.spines['bottom'].set_linewidth(primary_spine_width)
            ax.spines['left'].set_linewidth(primary_spine_width)
            ax.spines['top'].set_linewidth(default_spine_width)
            ax.spines['right'].set_linewidth(default_spine_width)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Primary axes labels (only on left and bottom edges)
            if stage_idx == 0:  # Leftmost column
                ax.set_ylabel(f'$\\sigma_{{{cell_type}}} / \\sigma_E$', fontsize=11, labelpad=8)
            else:
                ax.set_ylabel('')
            
            ax.set_xlabel(f'$\\tau_{{{cell_type}}} / \\tau_E$', fontsize=11, labelpad=8)
            
            ax.tick_params(labelsize=9, length=3, width=0.5)
            
            # Secondary axes (absolute values)
            tau_E = preset['time_constants']['E']
            sigma_E = preset['outgoing_widths']['E']
            
            ax2 = ax.secondary_yaxis('right', functions=(
                lambda x, se=sigma_E: x * se, 
                lambda x, se=sigma_E: x / se
            ))
            if stage_idx == n_stages - 1:  # Rightmost column shows labels
                ax2.set_ylabel(f'$\\sigma_{{{cell_type}}}$ (grid units)', fontsize=10, labelpad=8)
                ax2.tick_params(labelsize=8, length=2, width=0.5)
            else:
                ax2.set_ylabel('')
                ax2.tick_params(labelright=False, length=0)
            ax2.spines['right'].set_linewidth(secondary_spine_width)
            for spine_name in ['left', 'top', 'bottom']:
                ax2.spines[spine_name].set_visible(False)
            
            # Top axis on both rows (SST and PV)
            ax3 = ax.secondary_xaxis('top', functions=(
                lambda x, te=tau_E: x * te, 
                lambda x, te=tau_E: x / te
            ))
            ax3.set_xlabel(f'$\\tau_{{{cell_type}}}$ (ms)', fontsize=10, labelpad=8)
            ax3.tick_params(labelsize=8, length=2, width=0.5)
            ax3.spines['top'].set_linewidth(secondary_spine_width)
            for spine_name in ['bottom', 'left', 'right']:
                ax3.spines[spine_name].set_visible(False)
            
            # Add stage labels as column titles
            if cell_type_idx == 0:
                ax.set_title(stage_name, fontsize=13, fontweight='bold', pad=21)
            
            # Add cell type labels on the left
            if stage_idx == 0:
                ax.text(-0.28, 0.5, cell_type, transform=ax.transAxes,
                       fontsize=13, fontweight='bold', va='center', rotation=90, ha='center')
    
    # Add colorbar for wavenumber on the right side (centered vertically)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.885, 0.22, 0.015, 0.50])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Spatial freq. w/ max Re($\\lambda$)', fontsize=11, labelpad=18)
    cbar.ax.tick_params(labelsize=9, length=3, width=0.6)
    
    # Add grey legend bar for stable region
    stable_ax = fig.add_axes([0.885, 0.14, 0.015, 0.05])
    stable_ax.imshow(np.full((10, 1), 0.5), cmap='Greys', vmin=0, vmax=1, origin='lower', aspect='auto')
    stable_ax.set_xticks([])
    stable_ax.set_yticks([])
    # Keep spines visible for black border (like colorbar)
    for spine in stable_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    # Add label below the grey bar (split across two lines to fit better)
    stable_ax.text(0.5, -0.25, 'No dominant\nspatial mode', transform=stable_ax.transAxes,
                   fontsize=11, rotation=0, va='top', ha='center')
    
    # Overall title (positioned to avoid overlap)
    main_title = 'Developmental Stability Landscape'
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.94)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"\nCombined figure saved to: {output_path}")
    plt.close()


def compute_all_stages(stages: List[str] = None, mode: str = 'fixed_absolute') -> Dict[str, Dict]:
    """
    Compute bifurcation maps for all developmental stages.
    
    Args:
        stages: List of stage names to compute (default: ['P4', 'P8', 'P12', 'P16'])
        mode: Range calculation mode:
              'fixed_absolute' - Same absolute τ and σ ranges across all stages (default)
              'fixed_ratio' - Same ratio ranges (τ_inh/τ_E, σ_inh/σ_E) across all stages
        
    Returns:
        Dictionary mapping stage names to their results
    """
    if stages is None:
        stages = ['P4', 'P8', 'P12', 'P16']
    
    all_results = {}
    
    print("\n" + "="*70)
    print(f"  Developmental Bifurcation Maps - Computing All Stages ({mode})")
    print("="*70 + "\n")
    
    if mode == 'fixed_absolute':
        # Fixed absolute ranges: same biological parameter space across all stages
        print("\n[Fixed Absolute Range Mode]")
        print(f"  τ: [{TAU_MIN:.1f}, {TAU_MAX:.1f}] ms")
        print(f"  σ: [{SIGMA_MIN:.1f}, {SIGMA_MAX:.1f}]")
        print(f"  Grid resolution: {GRID_RESOLUTION}×{GRID_RESOLUTION} = {GRID_RESOLUTION**2} points per map\n")
        
        for stage_name in stages:
            stage_name = stage_name.upper()
            print(f"\n{'='*70}")
            print(f"  Stage: {stage_name}")
            print(f"{'='*70}")
            
            # Get preset
            preset = PRESETS[stage_name]
            tau_E = preset['time_constants']['E']
            sigma_E = preset['outgoing_widths']['E']
            
            # Compute stage-specific ratio ranges from fixed absolute ranges
            tau_ratio_min = TAU_MIN / tau_E
            tau_ratio_max = TAU_MAX / tau_E
            sigma_ratio_min = SIGMA_MIN / sigma_E
            sigma_ratio_max = SIGMA_MAX / sigma_E
            
            print(f"\nStage parameters: τ_E = {tau_E:.1f} ms, σ_E = {sigma_E:.1f}")
            print("  Ratio ranges for this stage:")
            print(f"    τ_ratio: [{tau_ratio_min:.3f}, {tau_ratio_max:.3f}]")
            print(f"    σ_ratio: [{sigma_ratio_min:.3f}, {sigma_ratio_max:.3f}]")
            
            # Define stage-specific parameter ranges
            tau_ratios = np.linspace(tau_ratio_min, tau_ratio_max, GRID_RESOLUTION)
            sigma_ratios = np.linspace(sigma_ratio_min, sigma_ratio_max, GRID_RESOLUTION)
            
            # Compute bifurcation maps
            results = compute_bifurcation_maps(
                stage_name,
                tau_ratios,
                sigma_ratios,
                seed=MEAN_STATE_SEED
            )
            
            all_results[stage_name] = results
    
    elif mode == 'fixed_ratio':
        # Fixed ratio ranges: same relative timescales across all stages
        print("\n[Fixed Ratio Range Mode]")
        print("Using fixed ratio limits:")
        print(f"  τ_ratio: [{FIXED_RATIO_TAU_MIN:.3f}, {FIXED_RATIO_TAU_MAX:.3f}]")
        print(f"  σ_ratio: [{FIXED_RATIO_SIGMA_MIN:.3f}, {FIXED_RATIO_SIGMA_MAX:.3f}]")
        print(f"  Grid resolution: {GRID_RESOLUTION}×{GRID_RESOLUTION} = {GRID_RESOLUTION**2} points per map\n")
        
        # Define common parameter ranges using fixed ratio limits
        tau_ratios = np.linspace(FIXED_RATIO_TAU_MIN, FIXED_RATIO_TAU_MAX, GRID_RESOLUTION)
        sigma_ratios = np.linspace(FIXED_RATIO_SIGMA_MIN, FIXED_RATIO_SIGMA_MAX, GRID_RESOLUTION)
        
        for stage_name in stages:
            stage_name = stage_name.upper()
            print(f"\n{'='*70}")
            print(f"  Stage: {stage_name}")
            print(f"{'='*70}")
            
            # Get preset
            preset = PRESETS[stage_name]
            tau_E = preset['time_constants']['E']
            sigma_E = preset['outgoing_widths']['E']
            
            print(f"\nStage parameters: τ_E = {tau_E:.1f} ms, σ_E = {sigma_E:.1f}")
            
            # Compute bifurcation maps
            results = compute_bifurcation_maps(
                stage_name,
                tau_ratios,
                sigma_ratios,
                seed=MEAN_STATE_SEED
            )
            
            all_results[stage_name] = results
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'fixed_absolute' or 'fixed_ratio'")
    
    return all_results


def save_results(all_results: Dict[str, Dict], output_path: Path) -> None:
    """
    Save computed results to disk.
    
    Args:
        all_results: Dictionary mapping stage names to their results
        output_path: Path to save pickle file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {output_path}")


def load_results(input_path: Path) -> Dict[str, Dict]:
    """
    Load previously computed results from disk.
    
    Args:
        input_path: Path to pickle file
        
    Returns:
        Dictionary mapping stage names to their results
    """
    with open(input_path, 'rb') as f:
        all_results = pickle.load(f)
    print(f"\nResults loaded from: {input_path}")
    return all_results


def main():
    """Main entry point for developmental bifurcation maps analysis."""
    parser = argparse.ArgumentParser(
        description='Developmental bifurcation maps analysis and visualization'
    )
    parser.add_argument('--mode', type=str, default='both',
                       choices=['compute', 'plot', 'both'],
                       help='Mode: compute, plot, or both (default: both - generates both range modes)')
    parser.add_argument('--stages', type=str, nargs='+',
                       choices=['P4', 'P8', 'P12', 'P16'],
                       help='Developmental stages to analyze (default: all)')
    parser.add_argument('--range-mode', type=str, default='all',
                       choices=['all', 'fixed_absolute', 'fixed_ratio'],
                       help='Range mode: all (both modes), fixed_absolute only, or fixed_ratio only (default: all)')
    
    args = parser.parse_args()
    
    # Determine which range modes to process
    if args.range_mode == 'all':
        range_modes = ['fixed_absolute', 'fixed_ratio']
    else:
        range_modes = [args.range_mode]
    
    # Process each range mode
    for range_mode in range_modes:
        # Determine file names based on range mode
        results_file = f'bifurcation_results_{range_mode}.pkl'
        figure_file = f'developmental_maps_{range_mode}.svg'
        
        results_path = Path(OUTPUT_DIR) / results_file
        figure_path = Path(OUTPUT_DIR) / figure_file
        
        print(f"\n{'#'*70}")
        print(f"  Processing: {range_mode}")
        print(f"{'#'*70}")
        
        # Compute mode
        if args.mode in ['compute', 'both']:
            all_results = compute_all_stages(args.stages, mode=range_mode)
            save_results(all_results, results_path)
            
            print("\n" + "="*70)
            print(f"  Computation complete for {range_mode}!")
            print("="*70 + "\n")
        
        # Plot mode
        if args.mode in ['plot', 'both']:
            if args.mode == 'plot':
                # Load results from file
                all_results = load_results(results_path)
            
            # Create combined figure
            create_combined_figure(all_results, figure_path, mode=range_mode)
            
            print("\n" + "="*70)
            print(f"  Visualization complete for {range_mode}!")
            print("="*70 + "\n")


if __name__ == '__main__':
    main()

