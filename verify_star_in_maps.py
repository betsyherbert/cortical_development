"""Verify star values in actual saved bifurcation maps."""

import pickle
import numpy as np
from pathlib import Path
from src.analysis.common import PRESETS
from src.analysis.bifurcation.config import ANALYSIS_PARAMS

def find_nearest_index(array, value):
    """Find index of nearest value in array."""
    return np.argmin(np.abs(array - value))

def verify_saved_maps():
    """Check saved bifurcation results for star consistency."""
    
    results_path = Path('outputs/bifurcation/bifurcation_results.pkl')
    
    if not results_path.exists():
        print(f"No saved results found at {results_path}")
        print("Run the bifurcation analysis first:")
        print("  python -m src.analysis.bifurcation.developmental_maps --mode both")
        return
    
    print("="*70)
    print("VERIFICATION: Star Values in Saved Bifurcation Maps")
    print("="*70)
    
    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)
    
    for stage_name in ['P4', 'P8', 'P12', 'P16']:
        print(f"\n{stage_name}:")
        print("-" * 50)
        
        preset = PRESETS[stage_name]
        results = all_results[stage_name]
        
        tau_ratios = results['tau_ratios']
        sigma_ratios = results['sigma_ratios']
        
        tau_E = preset['time_constants']['E']
        sigma_E = preset['outgoing_widths']['E']
        
        # Find star locations for each cell type
        for cell_type in ['SST', 'PV']:
            tau_ratio = preset['time_constants'][cell_type] / tau_E
            sigma_ratio = preset['outgoing_widths'][cell_type] / sigma_E
            
            # Find nearest grid points
            tau_idx = find_nearest_index(tau_ratios, tau_ratio)
            sigma_idx = find_nearest_index(sigma_ratios, sigma_ratio)
            
            # Get values at star from map
            k_matrix = results[f'{cell_type.lower()}_k']
            stab_matrix = results[f'{cell_type.lower()}_stability']
            
            k_at_star = k_matrix[tau_idx, sigma_idx]
            stab_at_star = stab_matrix[tau_idx, sigma_idx]
            
            print(f"  {cell_type} map:")
            print(f"    Star at (σ={sigma_ratio:.3f}, τ={tau_ratio:.3f})")
            print(f"    Grid point: ({sigma_idx}, {tau_idx})")
            print(f"    Grid values: (σ={sigma_ratios[sigma_idx]:.3f}, τ={tau_ratios[tau_idx]:.3f})")
            print(f"    k_crit = {k_at_star:.3f}, Re(λ)_max = {stab_at_star:.6f}")
        
        # Check exact preset value (if available)
        if 'preset_stability' in results:
            preset_stab = results['preset_stability']
            print(f"\n  EXACT preset stability: {preset_stab:.6f}")
            print(f"  ✓ This value will be shown at star on both maps")
            
            # Show difference from grid values
            sst_stab = results['sst_stability'][
                find_nearest_index(tau_ratios, preset['time_constants']['SST'] / tau_E),
                find_nearest_index(sigma_ratios, preset['outgoing_widths']['SST'] / sigma_E)
            ]
            pv_stab = results['pv_stability'][
                find_nearest_index(tau_ratios, preset['time_constants']['PV'] / tau_E),
                find_nearest_index(sigma_ratios, preset['outgoing_widths']['PV'] / sigma_E)
            ]
            print(f"  Grid interpolation errors:")
            print(f"    SST: {abs(sst_stab - preset_stab):.6f}")
            print(f"    PV:  {abs(pv_stab - preset_stab):.6f}")
        else:
            # Old results without exact preset value
            sst_stab = results['sst_stability'][
                find_nearest_index(tau_ratios, preset['time_constants']['SST'] / tau_E),
                find_nearest_index(sigma_ratios, preset['outgoing_widths']['SST'] / sigma_E)
            ]
            pv_stab = results['pv_stability'][
                find_nearest_index(tau_ratios, preset['time_constants']['PV'] / tau_E),
                find_nearest_index(sigma_ratios, preset['outgoing_widths']['PV'] / sigma_E)
            ]
            
            diff = abs(sst_stab - pv_stab)
            if diff < 1e-6:
                print(f"\n  ✓ SST and PV star values match (diff = {diff:.2e})")
            else:
                print(f"\n  ✗ WARNING: SST and PV differ by {diff:.6f}")
                print(f"  → Regenerate maps to fix this issue")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    verify_saved_maps()

