"""Main entry point for bifurcation analysis."""

import argparse
import numpy as np
from typing import Dict, List, Optional

from .bifurcation_analysis import NetworkModel, SteadyStateFinder, StabilityAnalyzer
from .visualizer import BifurcationVisualizer
from .config import (
    DEVELOPMENTAL_STAGES, PRESETS, DEFAULT_N_POPULATIONS,
    DEFAULT_LAYERS, ALL_LAYERS, ANALYSIS_PARAMS
)


def analyze_single_stage_data(stage_name: str, n_populations: int = None, 
                              layers: List[str] = None) -> Dict:
    """
    Analyze a single stage and return results as dictionary.
    
    Args:
        stage_name: Developmental stage name (P4, P8, P12, P16)
        n_populations: Number of populations (2 or 3, defaults to config)
        layers: List of layers to analyze (defaults to config)
        
    Returns:
        Dictionary with analysis results
    """
    if n_populations is None:
        n_populations = DEFAULT_N_POPULATIONS
    if layers is None:
        layers = DEFAULT_LAYERS
    
    preset = PRESETS[stage_name.upper()]
    network = NetworkModel(preset, layers=layers, n_populations=n_populations)
    params = network.get_parameters()
    
    finder = SteadyStateFinder(network)
    r_star, converged = finder.find_steady_state()
    
    analyzer = StabilityAnalyzer(network, r_star)
    distance, critical_mode, critical_k = analyzer.compute_stability()
    
    # Determine if network is silent (all rates near zero)
    is_silent = np.all(np.abs(r_star) < 1e-10)
    
    return {
        'stage': stage_name.upper(),
        'network_params': params,
        'steady_state': r_star,
        'distance': distance,
        'critical_mode': critical_mode,
        'critical_k': critical_k,
        'converged': converged,
        'is_silent': is_silent
    }


def analyze_all_stages(n_populations: int = None, layers: List[str] = None,
                       compare_layers: bool = False):
    """
    Compare stability across all developmental stages.
    
    Args:
        n_populations: Number of populations (2 or 3, defaults to config)
        layers: List of layers to analyze (defaults to config)
        compare_layers: If True, also run single-layer analysis and compare
    """
    if n_populations is None:
        n_populations = DEFAULT_N_POPULATIONS
    if layers is None:
        layers = DEFAULT_LAYERS
    
    layers_str = '-'.join(layers)
    pop_desc = f"{n_populations}-population ({', '.join(['E', 'SST', 'PV'][:n_populations])})" if n_populations == 3 else "2-population (E-I)"
    
    print(f"Analyzing developmental stages ({pop_desc}, layers: {layers_str})...")
    print()
    
    # Analyze all stages
    results_dict = {}
    
    for stage in DEVELOPMENTAL_STAGES:
        print(f"  {stage}", end=' ')
        result = analyze_single_stage_data(stage, n_populations, layers)
        results_dict[stage] = result
        status_symbol = "✓" if result['distance'] > 0 else "✗"
        print(f"{status_symbol}")
    
    print()
    
    # Generate visualizations
    visualizer = BifurcationVisualizer()
    print("Generating figures...")
    visualizer.plot_developmental_comparison(results_dict, n_populations)
    visualizer.plot_eigenvalue_spectra(results_dict, n_populations)
    
    # Generate detail plots for each stage
    for stage in DEVELOPMENTAL_STAGES:
        visualizer.plot_single_stage_detail(results_dict[stage], stage, n_populations)
    
    # If requested, compare with individual layer analyses
    if compare_layers and len(layers) > 1:
        print()
        print("Running layer isolation comparison...")
        l23_results = {}
        l4_results = {}
        l5_results = {}
        for stage in DEVELOPMENTAL_STAGES:
            print(f"  {stage}", end=' ')
            l23_result = analyze_single_stage_data(stage, n_populations, ['L23'])
            l4_result = analyze_single_stage_data(stage, n_populations, ['L4'])
            l5_result = analyze_single_stage_data(stage, n_populations, ['L5'])
            l23_results[stage] = l23_result
            l4_results[stage] = l4_result
            l5_results[stage] = l5_result
            print("✓")
        
        print()
        visualizer.plot_layer_coupling_comparison(l5_results, l4_results, l23_results, results_dict, n_populations)
    
    # Print concise summary
    print()
    print("Results Summary:")
    max_real_eigenvalues = [-results_dict[s]['distance'] for s in DEVELOPMENTAL_STAGES]
    min_stage = DEVELOPMENTAL_STAGES[np.argmax(max_real_eigenvalues)]  # Most positive (most unstable)
    max_real_str = ' → '.join([f"{s}={max_real_eigenvalues[i]:.4f}" for i, s in enumerate(DEVELOPMENTAL_STAGES)])
    print(f"  Max Re(λ): {max_real_str}")
    print(f"  Most unstable: {min_stage} (Max Re(λ) = {max(max_real_eigenvalues):.4f})")
    
    # Check for critical transitions
    transitions = []
    for i in range(len(DEVELOPMENTAL_STAGES) - 1):
        prev_k = results_dict[DEVELOPMENTAL_STAGES[i]]['critical_k']
        curr_k = results_dict[DEVELOPMENTAL_STAGES[i+1]]['critical_k']
        if prev_k >= 0.1 and curr_k < 0.1:
            transitions.append(f"  {DEVELOPMENTAL_STAGES[i]}→{DEVELOPMENTAL_STAGES[i+1]}: Transition to global patterns (k → 0)")
    
    if transitions:
        print("  Critical transitions:")
        for t in transitions:
            print(t)
    
    print()
    print("Figures saved (SVG format):")
    print(f"  • outputs/bifurcation/stability_across_development_{n_populations}pop.svg")
    print(f"  • outputs/bifurcation/eigenvalue_spectrum_{n_populations}pop.svg")
    print(f"  • outputs/bifurcation/<STAGE>_stability_detail_{n_populations}pop.svg (for each stage)")
    if compare_layers and len(layers) > 1:
        print(f"  • outputs/bifurcation/layer_isolation_comparison_{n_populations}pop.svg")


def analyze_single_stage(stage_name: str, n_populations: int = None, 
                         layers: List[str] = None):
    """
    Run detailed stability analysis for a single developmental stage.
    
    Args:
        stage_name: Developmental stage name
        n_populations: Number of populations (2 or 3, defaults to config)
        layers: List of layers to analyze (defaults to config)
    """
    if n_populations is None:
        n_populations = DEFAULT_N_POPULATIONS
    if layers is None:
        layers = DEFAULT_LAYERS
    
    preset = PRESETS.get(stage_name.upper())
    if preset is None:
        print(f"Error: Unknown stage '{stage_name}'. Available: {list(PRESETS.keys())}")
        return
    
    layers_str = '-'.join(layers)
    pop_desc = f"{n_populations}-population ({', '.join(['E', 'SST', 'PV'][:n_populations])})" if n_populations == 3 else "2-population (E-I)"
    
    print(f"Analyzing {stage_name.upper()} ({pop_desc}, layers: {layers_str})...")
    
    # Run analysis
    result = analyze_single_stage_data(stage_name, n_populations, layers)
    
    # Generate detail plot
    visualizer = BifurcationVisualizer()
    visualizer.plot_single_stage_detail(result, stage_name.upper(), n_populations)
    
    # Print summary
    max_real_eigenvalue = -result['distance']
    print()
    print("Analysis Complete:")
    print(f"  Stage: {result['stage']}")
    print(f"  Max Re(λ): {max_real_eigenvalue:.6f}")
    print(f"  Status: {'STABLE ✓' if max_real_eigenvalue < 0 else 'UNSTABLE ✗'}")
    print(f"  Critical mode: {result['critical_mode']} (k = {result['critical_k']:.4f})")
    print()
    print(f"  Figure saved: outputs/bifurcation/{stage_name.upper()}_stability_detail_{n_populations}pop.svg")


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description='Analyze network stability across developmental stages using bifurcation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # 3-population analysis, L4 (default)
  %(prog)s --cell-types 2            # 2-population E-I analysis
  %(prog)s --stage P4                # Single stage, 3-population
  %(prog)s --layers L23 L4 L5        # Multi-layer (full network)
  %(prog)s --layers L23 L4 L5 --compare-layers  # Full network + single-layer comparison
        """
    )
    parser.add_argument('--cell-types', type=int, choices=[2, 3], default=DEFAULT_N_POPULATIONS,
                       help=f'Number of cell types: 2 for E-I, 3 for E-SST-PV (default: {DEFAULT_N_POPULATIONS})')
    parser.add_argument('--stage', type=str, choices=DEVELOPMENTAL_STAGES,
                       help='Analyze single stage instead of all stages')
    parser.add_argument('--layers', nargs='+', choices=['L23', 'L4', 'L5'], default=DEFAULT_LAYERS,
                       help='Layers to analyze (default: L4). Can specify multiple: --layers L23 L4 L5')
    parser.add_argument('--compare-layers', action='store_true',
                       help='If using multi-layer, also run single-layer (L4) analysis and compare')
    
    args = parser.parse_args()
    
    if args.stage:
        analyze_single_stage(args.stage, args.cell_types, args.layers)
    else:
        analyze_all_stages(args.cell_types, args.layers, args.compare_layers)
