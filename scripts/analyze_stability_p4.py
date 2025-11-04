#!/usr/bin/env python3
"""Analyze network stability across developmental stages.

This script computes the steady state and stability of a simplified L4 network
at different developmental stages (P4, P8, P12, P16), providing insights into
how network stability evolves during development.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.presets import P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET
from src.analysis.bifurcation import NetworkModel, SteadyStateFinder, StabilityAnalyzer


# Map stage names to presets
PRESET_MAP = {
    'P4': P4_PRESET,
    'P8': P8_PRESET,
    'P12': P12_PRESET,
    'P16': P16_PRESET
}


def analyze_single_stage(stage_name: str):
    """Run stability analysis for a single developmental stage."""
    preset = PRESET_MAP.get(stage_name.upper())
    if preset is None:
        print(f"Error: Unknown stage '{stage_name}'. Available: {list(PRESET_MAP.keys())}")
        return
    
    print("=" * 60)
    print(f"{stage_name.upper()} L4 Network Stability Analysis (E-I simplified)")
    print("=" * 60)
    print()
    
    # Create simplified network
    print("Initializing simplified network...")
    network = NetworkModel(preset, layers=['L4'])
    params = network.get_parameters()
    
    print("\nNetwork Parameters:")
    print(f"  Time constants: τ_E = {params['tau_e']:.2f} ms, τ_I = {params['tau_i']:.2f} ms")
    print(f"  Gains: g_E = {params['g_e']:.2f}, g_I = {params['g_i']:.2f}")
    print(f"  Connection strengths:")
    print(f"    A_EE = {params['A_ee']:.4f}")
    print(f"    A_EI = {params['A_ei']:.4f}")
    print(f"    A_IE = {params['A_ie']:.4f}")
    print(f"    A_II = {params['A_ii']:.4f}")
    print(f"  Spatial scales (σ):")
    print(f"    σ_EE = {params['sigma_ee']:.2f}")
    print(f"    σ_EI = {params['sigma_ei']:.2f}")
    print(f"    σ_IE = {params['sigma_ie']:.2f}")
    print(f"    σ_II = {params['sigma_ii']:.2f}")
    print()
    
    # Find steady state
    print("Finding steady state...")
    finder = SteadyStateFinder(network)
    r_e_star, r_i_star, converged = finder.find_steady_state()
    
    if not converged:
        print("WARNING: Steady state finder did not converge!")
        print(f"  Final values: r_E = {r_e_star:.6f}, r_I = {r_i_star:.6f}")
        print()
    else:
        print(f"  Converged in {finder.max_iters} iterations or less")
    
    print("\nSteady State:")
    print(f"  E firing rate: {r_e_star:.10f} Hz")
    print(f"  I firing rate: {r_i_star:.10f} Hz")
    print()
    
    # Check if network is active at steady state
    eps = 1e-10
    is_silent = abs(r_e_star) < eps and abs(r_i_star) < eps
    if is_silent:
        print("  Status: Network is SILENT at steady state (zero firing rates)")
        print("  Interpretation: Without external input, network returns to zero activity")
    else:
        print(f"  Status: Network has NON-ZERO steady state activity")
        print(f"  Interpretation: Network maintains activity even without external input")
    print()
    
    # Compute stability
    print("Computing stability (scanning Fourier modes)...")
    analyzer = StabilityAnalyzer(network, (r_e_star, r_i_star), n_modes=10)
    distance, critical_mode, critical_k = analyzer.compute_stability()
    
    print(f"  Scanned {2 * analyzer.n_modes + 1} x {2 * analyzer.n_modes + 1} = {(2 * analyzer.n_modes + 1)**2} modes")
    print()
    
    print("Stability Analysis:")
    print(f"  Distance to instability: {distance:.6f}")
    print(f"  Critical mode: ({critical_mode[0]}, {critical_mode[1]}) with k = {critical_k:.4f}")
    
    if distance > 0:
        print(f"  Status: STABLE (positive distance)")
        print(f"  Interpretation:")
        print(f"    Network is {distance:.6f} units away from instability.")
        print(f"    Perturbations decay with time constant ~{1.0/distance:.2f} ms")
        if critical_k < 0.1:
            print(f"    Critical mode is GLOBAL (k ≈ 0) - would create population-wide fluctuations")
        else:
            print(f"    Critical mode has k = {critical_k:.4f} - would create patchy spatial patterns")
    else:
        print(f"  Status: UNSTABLE (negative distance)")
        print(f"  Interpretation:")
        print(f"    Network has crossed the instability boundary.")
        print(f"    Spontaneous activity will emerge without external input.")
    
    print()
    print("=" * 60)
    print("Analysis Complete")
    print("=" * 60)


def analyze_single_stage_data(stage_name: str) -> dict:
    """Analyze a single stage and return results as dictionary."""
    preset = PRESET_MAP[stage_name.upper()]
    network = NetworkModel(preset, layers=['L4'])
    params = network.get_parameters()
    
    finder = SteadyStateFinder(network)
    r_e_star, r_i_star, converged = finder.find_steady_state()
    
    analyzer = StabilityAnalyzer(network, (r_e_star, r_i_star), n_modes=10)
    distance, critical_mode, critical_k = analyzer.compute_stability()
    
    return {
        'stage': stage_name.upper(),
        'tau_e': params['tau_e'],
        'tau_i': params['tau_i'],
        'A_ee': params['A_ee'],
        'A_ei': params['A_ei'],
        'A_ie': params['A_ie'],
        'A_ii': params['A_ii'],
        'r_e_star': r_e_star,
        'r_i_star': r_i_star,
        'distance': distance,
        'critical_mode': critical_mode,
        'critical_k': critical_k,
        'converged': converged,
        'is_silent': abs(r_e_star) < 1e-10 and abs(r_i_star) < 1e-10
    }


def analyze_all_stages():
    """Compare stability across all developmental stages."""
    print("=" * 80)
    print("Developmental Stage Comparison: L4 Network Stability Analysis")
    print("=" * 80)
    print()
    
    # Analyze all stages
    stages = ['P4', 'P8', 'P12', 'P16']
    results = []
    
    for stage in stages:
        print(f"Analyzing {stage}...", end=' ')
        result = analyze_single_stage_data(stage)
        results.append(result)
        status_symbol = "✓" if result['distance'] > 0 else "✗"
        print(f"{status_symbol}")
    
    print()
    
    # Print comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()
    
    # Table header
    header = f"{'Stage':<6} | {'τ_E':<6} | {'τ_I':<6} | {'A_EE':<7} | {'A_IE':<7} | {'r_E*':<10} | {'r_I*':<10} | {'Distance':<10} | {'k_crit':<8} | {'Status':<8}"
    print(header)
    print("-" * len(header))
    
    # Table rows
    for r in results:
        status = "STABLE ✓" if r['distance'] > 0 else "UNSTABLE ✗"
        row = (f"{r['stage']:<6} | "
               f"{r['tau_e']:<6.2f} | "
               f"{r['tau_i']:<6.2f} | "
               f"{r['A_ee']:<7.4f} | "
               f"{r['A_ie']:<7.4f} | "
               f"{r['r_e_star']:<10.6f} | "
               f"{r['r_i_star']:<10.6f} | "
               f"{r['distance']:<10.6f} | "
               f"{r['critical_k']:<8.4f} | "
               f"{status:<8}")
        print(row)
    
    print()
    print("=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze trends
    print("1. Distance to Instability:")
    prev_distance = None
    for r in results:
        change_str = ""
        if prev_distance is not None:
            change = r['distance'] - prev_distance
            change_pct = (change / abs(prev_distance)) * 100 if prev_distance != 0 else 0
            arrow = "↓↓" if change < -0.01 else "↑" if change > 0.01 else "→"
            change_str = f" ({arrow} {change:+.6f}, {change_pct:+.1f}%)"
        print(f"   {r['stage']}: {r['distance']:.6f}{change_str}")
        prev_distance = r['distance']
    print()
    
    print("2. Critical Mode Spatial Frequency (k_crit):")
    prev_k = None
    for r in results:
        change_str = ""
        if prev_k is not None:
            change = r['critical_k'] - prev_k
            arrow = "↓↓" if change < -0.5 else "↑" if change > 0.5 else "→"
            change_str = f" ({arrow} {change:+.4f})"
        k_desc = "GLOBAL" if r['critical_k'] < 0.1 else "PATCHY"
        print(f"   {r['stage']}: k = {r['critical_k']:.4f} ({k_desc}){change_str}")
        prev_k = r['critical_k']
    print()
    
    print("3. Key Parameter Changes:")
    prev_result = None
    for r in results:
        if prev_result is not None:
            tau_e_change = ((r['tau_e'] - prev_result['tau_e']) / prev_result['tau_e']) * 100
            tau_i_change = ((r['tau_i'] - prev_result['tau_i']) / prev_result['tau_i']) * 100
            A_ee_change = ((r['A_ee'] - prev_result['A_ee']) / abs(prev_result['A_ee'])) * 100 if prev_result['A_ee'] != 0 else 0
            A_ie_change = ((r['A_ie'] - prev_result['A_ie']) / abs(prev_result['A_ie'])) * 100 if prev_result['A_ie'] != 0 else 0
            
            print(f"   {prev_result['stage']} → {r['stage']}:")
            if abs(tau_e_change) > 5:
                print(f"      τ_E: {prev_result['tau_e']:.2f} → {r['tau_e']:.2f} ({tau_e_change:+.1f}%)")
            if abs(tau_i_change) > 5:
                print(f"      τ_I: {prev_result['tau_i']:.2f} → {r['tau_i']:.2f} ({tau_i_change:+.1f}%)")
            if abs(A_ee_change) > 10:
                print(f"      A_EE: {prev_result['A_ee']:.4f} → {r['A_ee']:.4f} ({A_ee_change:+.1f}%)")
            if abs(A_ie_change) > 10:
                print(f"      A_IE: {prev_result['A_ie']:.4f} → {r['A_ie']:.4f} ({A_ie_change:+.1f}%)")
        prev_result = r
    print()
    
    print("4. Key Transitions:")
    transitions = []
    for i, r in enumerate(results):
        if i == 0:
            continue
        prev_r = results[i-1]
        
        # Check for stability boundary crossing
        if prev_r['distance'] > 0 and r['distance'] <= 0:
            transitions.append(f"   • {prev_r['stage']} → {r['stage']}: CROSSES INSTABILITY BOUNDARY (stable → unstable)")
        elif prev_r['distance'] <= 0 and r['distance'] > 0:
            transitions.append(f"   • {prev_r['stage']} → {r['stage']}: BECOMES STABLE (unstable → stable)")
        
        # Check for transition to global patterns (k ≈ 0)
        if prev_r['critical_k'] >= 0.1 and r['critical_k'] < 0.1:
            transitions.append(f"   • {prev_r['stage']} → {r['stage']}: TRANSITION TO GLOBAL PATTERNS (k → 0)")
    
    if transitions:
        for t in transitions:
            print(t)
    else:
        print("   • No major transitions detected")
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    # Overall interpretation
    p4_dist = results[0]['distance']
    p16_dist = results[-1]['distance']
    
    print("Developmental Trajectory:")
    print(f"  • P4 (Early): Distance = {p4_dist:.6f} - {'Very stable' if p4_dist > 0.1 else 'Moderately stable'}")
    print(f"  • P16 (Late): Distance = {p16_dist:.6f} - {'Stable' if p16_dist > 0 else 'UNSTABLE'}")
    print()
    
    if p16_dist <= 0:
        print("  ✓ Network becomes UNSTABLE by P16 → explains spontaneous activity!")
    elif p16_dist < 0.01:
        print("  ⚠ Network is VERY CLOSE to instability at P16 → sensitive to perturbations")
    else:
        print("  ℹ Network remains stable but approaches instability boundary during development")
    
    print()
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


def main():
    """Main entry point with CLI support."""
    # Check for single stage analysis
    if '--stage' in sys.argv:
        idx = sys.argv.index('--stage')
        if idx + 1 < len(sys.argv):
            stage_name = sys.argv[idx + 1]
            analyze_single_stage(stage_name)
        else:
            print("Error: --stage requires a stage name (P4, P8, P12, P16)")
    else:
        # Default: compare all stages
        analyze_all_stages()


if __name__ == '__main__':
    main()
