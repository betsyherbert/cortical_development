"""Main entry point for bifurcation analysis."""

import argparse
import numpy as np
from typing import Dict, List

from .bifurcation_analysis import NetworkModel, SteadyStateFinder, StabilityAnalyzer
from .visualizer import BifurcationVisualizer
from .config import (
    DEVELOPMENTAL_STAGES, PRESETS,
    DEFAULT_LAYERS,
    ANALYSIS_MODES, DEFAULT_ANALYSIS_MODE, THALAMIC_INPUT_MAGNITUDE
)


def analyze_single_stage_data(stage_name: str, 
                              layers: List[str] = None,
                              analysis_mode: str = None,
                              thalamic_magnitude: float = None,
                              verbose: bool = False) -> Dict:
    """
    Analyze a single stage and return results as dictionary.
    
    Args:
        stage_name: Developmental stage name (P4, P8, P12, P16)
        layers: List of layers to analyze (defaults to config)
        analysis_mode: 'silent', 'driven', or 'both' (defaults to config)
        thalamic_magnitude: Magnitude of thalamic input for driven analysis (defaults to config)
        
    Returns:
        Dictionary with analysis results
    """
    if layers is None:
        layers = DEFAULT_LAYERS
    if analysis_mode is None:
        analysis_mode = DEFAULT_ANALYSIS_MODE
    if thalamic_magnitude is None:
        thalamic_magnitude = THALAMIC_INPUT_MAGNITUDE
    
    preset = PRESETS[stage_name.upper()]
    network = NetworkModel(preset, layers=layers)
    params = network.get_parameters()
    
    finder = SteadyStateFinder(network)
    
    # Perform analysis based on mode
    if analysis_mode in ['silent', 'both']:
        # Silent fixed point analysis
        r_star_silent, status_silent = finder.find_steady_state(thalamic_input=None)
        analyzer_silent = StabilityAnalyzer(network, r_star_silent)
        distance_silent, critical_mode_silent, critical_k_silent, wavelength_silent = analyzer_silent.compute_stability(verbose=verbose)
        is_silent = np.all(np.abs(r_star_silent) < 1e-10)
        
        # Compute forced response at silent operating point
        forced_response_silent = analyzer_silent.compute_forced_response(
            network.thalamic_strengths, verbose=verbose
        )

        silent_results = {
            'steady_state': r_star_silent,
            'distance': distance_silent,
            'critical_mode': critical_mode_silent,
            'critical_k': critical_k_silent,
            'wavelength': wavelength_silent,
            'status': status_silent,
            'converged': (status_silent == 'converged'),
            'is_silent': is_silent,
            'forced_response_max_gain': forced_response_silent['max_gain'],
            'forced_response_critical_mode': forced_response_silent['critical_mode'],
            'forced_response_critical_k': forced_response_silent['critical_k'],
            'forced_response_max_condition': forced_response_silent['max_condition'],
            'forced_response_k_values': forced_response_silent['k_values'],
            'forced_response_gain_profile': forced_response_silent['gain_profile'],
            'forced_response_max_real_profile': forced_response_silent['max_real_profile']
        }
    else:
        silent_results = None
    
    if analysis_mode in ['driven', 'both']:
        # Driven operating point analysis
        # The driven fixed point represents the spatially uniform mean firing rate
        # during thalamic input epochs (captures mean depolarization, not spatial burst structure)
        thalamic_input = network.compute_thalamic_input(thalamic_magnitude)
        
        # Diagnostic: Log thalamic input details (only if verbose)
        if verbose:
            print(f"  [Input] Thalamic strengths: {network.thalamic_strengths}")
            print(f"  [Input] Thalamic input (mag={thalamic_magnitude}): {thalamic_input}")
            print(f"  [Input] Noise mean (mu): {network.mu}")
            print(f"  [Input] Total external: {network.mu + thalamic_input}")
        
        r_star_driven, status_driven = finder.find_steady_state(thalamic_input=thalamic_input)
        analyzer_driven = StabilityAnalyzer(network, r_star_driven)
        distance_driven, critical_mode_driven, critical_k_driven, wavelength_driven = analyzer_driven.compute_stability(verbose=verbose)
        is_silent_driven = np.all(np.abs(r_star_driven) < 1e-10)
        
        # Compute forced response at driven operating point
        forced_response_driven = analyzer_driven.compute_forced_response(
            network.thalamic_strengths, verbose=verbose
        )

        driven_results = {
            'steady_state': r_star_driven,
            'distance': distance_driven,
            'critical_mode': critical_mode_driven,
            'critical_k': critical_k_driven,
            'wavelength': wavelength_driven,
            'status': status_driven,
            'converged': (status_driven == 'converged'),
            'is_silent': is_silent_driven,
            'thalamic_magnitude': thalamic_magnitude,
            'forced_response_max_gain': forced_response_driven['max_gain'],
            'forced_response_critical_mode': forced_response_driven['critical_mode'],
            'forced_response_critical_k': forced_response_driven['critical_k'],
            'forced_response_max_condition': forced_response_driven['max_condition'],
            'forced_response_k_values': forced_response_driven['k_values'],
            'forced_response_gain_profile': forced_response_driven['gain_profile'],
            'forced_response_max_real_profile': forced_response_driven['max_real_profile']
        }
        
        # Diagnostic: Compare driven vs silent (if both modes and verbose)
        if analysis_mode == 'both' and verbose:
            r_diff = np.linalg.norm(r_star_driven - r_star_silent, ord=np.inf)
            n_active_silent = np.sum(r_star_silent > 1e-10)
            n_active_driven = np.sum(r_star_driven > 1e-10)
            
            print(f"  [Debug] Stage {stage_name}:")
            print(f"    ||r_driven - r_silent||_∞ = {r_diff:.6f}")
            print(f"    Active units: silent={n_active_silent}, driven={n_active_driven}")
            print(f"    Max r_silent: {np.max(r_star_silent):.6f}, max r_driven: {np.max(r_star_driven):.6f}")
            print(f"    Max Re(λ): silent={-distance_silent:.6f}, driven={-distance_driven:.6f}")
    else:
        driven_results = None
    
    # Build result dictionary
    result = {
        'stage': stage_name.upper(),
        'network_params': params,
        'analysis_mode': analysis_mode
    }
    
    # Add mode-specific results
    if analysis_mode == 'silent':
        result.update(silent_results)
    elif analysis_mode == 'driven':
        result.update(driven_results)
    elif analysis_mode == 'both':
        result['silent'] = silent_results
        result['driven'] = driven_results
        # For backward compatibility, default to silent results at top level
        result.update(silent_results)
    
    return result


def analyze_all_stages(layers: List[str] = None,
                       compare_layers: bool = False, analysis_mode: str = None,
                       thalamic_magnitude: float = None, verbose: bool = False):
    """
    Compare stability across all developmental stages.
    
    Args:
        layers: List of layers to analyze (defaults to config)
        compare_layers: If True, also run single-layer analysis and compare
        analysis_mode: 'silent', 'driven', or 'both' (defaults to config)
        thalamic_magnitude: Magnitude of thalamic input for driven analysis (defaults to config)
        verbose: If True, print diagnostic information
    """
    if layers is None:
        layers = DEFAULT_LAYERS
    if analysis_mode is None:
        analysis_mode = DEFAULT_ANALYSIS_MODE
    if thalamic_magnitude is None:
        thalamic_magnitude = THALAMIC_INPUT_MAGNITUDE
    
    # Analyze all stages
    results_dict = {}
    
    for stage in DEVELOPMENTAL_STAGES:
        print(f"{stage}", end=' ')
        result = analyze_single_stage_data(stage, layers, analysis_mode, thalamic_magnitude, verbose)
        results_dict[stage] = result
        
        # Show status: ✓ for stable, ✗ for unstable, ⚠ for divergence/no convergence
        if analysis_mode == 'both':
            # Check both modes
            silent_status = result['silent'].get('status', 'unknown')
            driven_status = result['driven'].get('status', 'unknown')
            distance = result['distance']  # Silent distance (top-level)
            if distance > 0:
                status_symbol = "✓"
            elif silent_status == 'diverged' or driven_status == 'diverged':
                status_symbol = "⚠"
            else:
                status_symbol = "✗"
        else:
            status_str = result.get('status', 'unknown')
            distance = result['distance']
            if distance > 0:
                status_symbol = "✓"
            elif status_str == 'diverged':
                status_symbol = "⚠"
            else:
                status_symbol = "✗"
        print(f"{status_symbol}")
    
    # Generate visualizations
    visualizer = BifurcationVisualizer()
    visualizer.plot_developmental_comparison(results_dict)
    
    # For mode='both', generate separate spectrum plots for silent and driven
    if analysis_mode == 'both':
        # Plot silent spectrum
        visualizer.plot_eigenvalue_spectra(results_dict, mode_override='silent')
        # Plot driven spectrum
        visualizer.plot_eigenvalue_spectra(results_dict, mode_override='driven')
    else:
        visualizer.plot_eigenvalue_spectra(results_dict)
    
    # If requested, compare with individual layer analyses
    if compare_layers and len(layers) > 1:
        l23_results = {}
        l4_results = {}
        l5_results = {}
        for stage in DEVELOPMENTAL_STAGES:
            l23_result = analyze_single_stage_data(stage, ['L23'], analysis_mode, thalamic_magnitude, verbose)
            l4_result = analyze_single_stage_data(stage, ['L4'], analysis_mode, thalamic_magnitude, verbose)
            l5_result = analyze_single_stage_data(stage, ['L5'], analysis_mode, thalamic_magnitude, verbose)
            l23_results[stage] = l23_result
            l4_results[stage] = l4_result
            l5_results[stage] = l5_result
        
        visualizer.plot_layer_coupling_comparison(l5_results, l4_results, l23_results, results_dict)
        
        # Compute global y-limits across all forced response plots for consistency
        all_results_for_ylims = {
            'full': results_dict,
            'L23': l23_results,
            'L4': l4_results,
            'L5': l5_results
        }
        global_ylims = visualizer.compute_global_forced_response_ylims(all_results_for_ylims)
        
        # Generate forced response plots with consistent y-limits
        visualizer.plot_forced_response_development(results_dict, global_ylims=global_ylims)
        visualizer.plot_forced_response_development(l23_results, global_ylims=global_ylims)
        visualizer.plot_forced_response_development(l4_results, global_ylims=global_ylims)
        visualizer.plot_forced_response_development(l5_results, global_ylims=global_ylims)
    else:
        # Single scope analysis - no need for global y-limits
        visualizer.plot_forced_response_development(results_dict)
    
    # Determine scope and mode for filenames
    scope = layers[0] if len(layers) == 1 else 'full'
    mode_suffix = 'silent' if analysis_mode == 'both' else analysis_mode
    
    # Build figure list
    figure_list = []
    if analysis_mode == 'both':
        figure_list.append(f"{scope}_development_{mode_suffix}.svg")
        figure_list.append(f"{scope}_spectrum_silent.svg")
        figure_list.append(f"{scope}_spectrum_driven.svg")
        figure_list.append(f"{scope}_forced_response_silent.svg")
        figure_list.append(f"{scope}_forced_response_driven.svg")
    else:
        figure_list.append(f"{scope}_development_{mode_suffix}.svg")
        figure_list.append(f"{scope}_spectrum_{mode_suffix}.svg")
        figure_list.append(f"{scope}_forced_response_{mode_suffix}.svg")
    
    if compare_layers and len(layers) > 1:
        figure_list.append(f"layers_comparison_{mode_suffix}.svg")
    
    print(f"Figures: {', '.join(figure_list)}")


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description='Analyze network stability across developmental stages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # L4 layer, silent mode (default)
  %(prog)s --layers L23 L4 L5        # Full network
  %(prog)s --layers L23 L4 L5 --compare-layers  # Full network + layer comparison
  %(prog)s --mode driven              # Driven operating point
  %(prog)s --mode both --verbose      # Compare modes with diagnostics
        """
    )
    parser.add_argument('--layers', nargs='+', choices=['L23', 'L4', 'L5'], default=DEFAULT_LAYERS,
                       help='Layers to analyze (default: L4). Use --layers L23 L4 L5 for full network')
    parser.add_argument('--compare-layers', action='store_true',
                       help='Compare isolated layers vs full network')
    parser.add_argument('--mode', type=str, choices=ANALYSIS_MODES, default=DEFAULT_ANALYSIS_MODE,
                       help=f'Analysis mode: silent, driven, or both (default: {DEFAULT_ANALYSIS_MODE})')
    parser.add_argument('--thalamic-input', type=float, default=THALAMIC_INPUT_MAGNITUDE,
                       help=f'Thalamic input magnitude for driven mode (default: {THALAMIC_INPUT_MAGNITUDE})')
    parser.add_argument('--verbose', action='store_true',
                       help='Print diagnostic information')
    
    args = parser.parse_args()
    
    analyze_all_stages(args.layers, args.compare_layers, args.mode, args.thalamic_input, args.verbose)
