"""Configuration parameters for bifurcation analysis.

This module defines all parameters for the Figure 5A-style bifurcation analysis,
including Fourier grid settings, parameter sweep ranges, numerical tolerances,
and validation settings.
"""

import numpy as np
from typing import Tuple, List

from src.model.config import GRID_SIZE, CELL_TYPES, LAYERS

#------------------------------------------------------------------------------
# Fourier Grid Parameters  
#------------------------------------------------------------------------------

# Use integer mode indices n = (nx, ny) where nx, ny ∈ {0, ..., GRID_SIZE-1}
# This gives the correct Fourier convention: exp(-2π²σ²||n||²)
FOURIER_GRID_PARAMS = {
    'grid_size': GRID_SIZE,  # Use existing grid size from main model
    'mode_convention': 'integer',  # Use integer mode indices (not divided by grid size)
    'physical_units': 'raw',  # Use raw ||n|| for mode radius (not 2π||n||/L)
    'domain_length': 1.0,  # Physical domain size (if needed for unit conversion)
    'max_mode_radius': GRID_SIZE // 2,  # Maximum meaningful mode radius
}

#------------------------------------------------------------------------------
# Parameter Sweep Ranges
#------------------------------------------------------------------------------

# Parameter ranges for bifurcation analysis
# Each analysis sweeps two parameters while keeping others fixed
PARAMETER_RANGES = {
    # PV-focused analysis: vary (τ_PV/τ_E, σ_PV/σ_E) 
    # Using biologically relevant ranges based on developmental data
    'pv_analysis': {
        'tau_ratio_range': np.linspace(0.5, 3.0, 50),  # 0.5 to 3.0 (biologically relevant)
        'sigma_ratio_range': np.linspace(0.5, 2.0, 50),  # 0.5 to 2.0 (biologically relevant)
        'param1_name': 'tau_pv_ratio',
        'param2_name': 'sigma_pv_ratio',
        'param1_label': r'$\tau_{PV} / \tau_E$',
        'param2_label': r'$\sigma_{PV} / \sigma_E$',
        'fixed_cell_type': 'SST',  # Keep SST parameters fixed
        'varied_cell_type': 'PV'   # Vary PV parameters
    },
    
    # SST-focused analysis: vary (τ_SST/τ_E, σ_SST/σ_E)
    # Using biologically relevant ranges based on developmental data
    'sst_analysis': {
        'tau_ratio_range': np.linspace(0.5, 3.0, 50),  # 0.5 to 3.0 (biologically relevant)
        'sigma_ratio_range': np.linspace(0.5, 2.0, 50),  # 0.5 to 2.0 (biologically relevant)
        'param1_name': 'tau_sst_ratio',
        'param2_name': 'sigma_sst_ratio', 
        'param1_label': r'$\tau_{SST} / \tau_E$',
        'param2_label': r'$\sigma_{SST} / \sigma_E$',
        'fixed_cell_type': 'PV',   # Keep PV parameters fixed
        'varied_cell_type': 'SST'  # Vary SST parameters
    },
    
    # Alternative analysis: connection strength vs width
    'strength_width_analysis': {
        'strength_range': np.linspace(0.0, 0.5, 50),  # Connection strength range
        'sigma_ratio_range': np.logspace(-1, 1, 50),  # Width ratio range
        'param1_name': 'connection_strength',
        'param2_name': 'sigma_ratio',
        'param1_label': r'Inhibitory Strength',
        'param2_label': r'$\sigma_{inh} / \sigma_E$',
        'fixed_cell_type': None,   # Both PV and SST varied together
        'varied_cell_type': 'both' # Vary both inhibitory types
    }
}

# Reference parameter values (kept fixed during sweeps)
# Using P12 preset values for more realistic bifurcation analysis
REFERENCE_PARAMS = {
    'tau_e': 7.0,     # P12 excitatory time constant (ms) - much more realistic
    'sigma_e': 2.0,   # P12 excitatory connection width (grid units)
    'gain_e': 1.0,    # Reference excitatory gain
    'thalamic_alpha': 0.6,  # P12 thalamic alpha (balance between intrinsic and sensory)
}

#------------------------------------------------------------------------------
# Numerical Tolerances
#------------------------------------------------------------------------------

NUMERICAL_TOLERANCES = {
    # Fixed point solver tolerances
    'fixed_point_convergence': 1e-10,     # Convergence tolerance for voltage changes
    'fixed_point_max_iter_phase1': 1000,  # Max iterations for fixed-point iteration
    'fixed_point_max_iter_phase2': 100,   # Max iterations for Newton refinement
    'voltage_zero_threshold': 1e-12,      # Threshold for considering voltage as zero
    'min_time_constant': 1e-6,            # Minimum time constant to avoid singularities
    
    # Stability analysis tolerances  
    'stability_threshold': 1e-6,          # Threshold for stability (λ* < -ε)
    'eigenvalue_tolerance': 1e-12,        # Tolerance for eigenvalue computation
    'matrix_condition_threshold': 1e12,   # Maximum condition number for matrices
    
    # Kernel normalization tolerances
    'kernel_normalization_tolerance': 1e-10,  # Tolerance for kernel sum = 1.0
    'dc_gain_tolerance': 1e-10,              # Tolerance for DC gain validation
}

#------------------------------------------------------------------------------
# Validation Settings
#------------------------------------------------------------------------------

VALIDATION_SETTINGS = {
    # Kernel normalization validation
    'check_kernel_normalization': True,    # Validate Gaussian kernels sum to 1.0
    'force_kernel_normalization': True,    # Force normalization if not satisfied
    'report_normalization_status': True,   # Print normalization diagnostics
    
    # Fixed point validation
    'validate_fixed_point_convergence': True,  # Check convergence from multiple starts
    'num_random_starts': 5,                   # Number of random initial conditions to test
    'validate_external_inputs': True,         # Check thalamic and noise contributions
    
    # Stability validation  
    'validate_parameter_limits': True,     # Test extreme parameter values
    'validate_boundary_smoothness': True,  # Check smooth transitions
    'validate_mode_analysis': True,        # Verify spatial frequency analysis
    
    # Cross-validation with existing analysis
    'cross_validate_stability': False,     # Compare with existing StabilityAnalysis
    'stability_comparison_tolerance': 1e-3, # Tolerance for stability comparison
    
    # Performance validation
    'max_analysis_time_minutes': 30,       # Maximum allowed analysis time
    'memory_usage_limit_gb': 8,            # Maximum memory usage limit
}

#------------------------------------------------------------------------------
# Analysis Output Settings
#------------------------------------------------------------------------------

OUTPUT_SETTINGS = {
    # Result storage
    'save_raw_results': True,              # Save detailed numerical results
    'save_processed_results': True,        # Save processed/summarized results  
    'result_file_format': 'pickle',        # Format for saving results
    'compress_results': True,              # Compress result files
    
    # Diagnostic output
    'save_diagnostic_plots': True,         # Save validation/diagnostic plots
    'save_intermediate_data': False,       # Save intermediate computation steps
    'verbose_output': True,                # Print detailed progress information
    
    # Plot settings
    'plot_dpi': 300,                       # Resolution for saved plots
    'plot_format': 'png',                  # Format for plot files
    'colormap_stable': 'gray',             # Color for stable regions
    'colormap_unstable': 'viridis',        # Colormap for unstable regions (by wavenumber)
}

#------------------------------------------------------------------------------
# Population Indexing Convention
#------------------------------------------------------------------------------

# Define consistent indexing for 9-population system: (E, SST, PV) × (L23, L4, L5)
POPULATION_INDICES = {}
idx = 0
for layer_name in LAYERS:
    for cell_type_name in CELL_TYPES:
        POPULATION_INDICES[(layer_name, cell_type_name)] = idx
        idx += 1

# Reverse mapping for convenience
INDEX_TO_POPULATION = {v: k for k, v in POPULATION_INDICES.items()}

# Total number of populations
N_POPULATIONS = len(POPULATION_INDICES)

#------------------------------------------------------------------------------
# Physical Parameter Constraints
#------------------------------------------------------------------------------

# Biologically reasonable parameter ranges for validation
BIOLOGICAL_CONSTRAINTS = {
    'time_constants': {
        'E': {'min': 20.0, 'max': 200.0},    # Excitatory time constants (ms)
        'SST': {'min': 30.0, 'max': 150.0},  # SST time constants (ms)  
        'PV': {'min': 10.0, 'max': 80.0},    # PV time constants (ms)
    },
    
    'connection_widths': {
        'E': {'min': 0.5, 'max': 5.0},       # Excitatory connection widths (grid units)
        'SST': {'min': 1.0, 'max': 8.0},     # SST connection widths (grid units)
        'PV': {'min': 0.3, 'max': 3.0},      # PV connection widths (grid units)  
    },
    
    'connection_strengths': {
        'excitatory': {'min': 0.0, 'max': 1.0},    # Excitatory connection strengths
        'inhibitory': {'min': -1.0, 'max': 0.0},   # Inhibitory connection strengths
    }
}

#------------------------------------------------------------------------------
# Computational Optimization Settings  
#------------------------------------------------------------------------------

OPTIMIZATION_SETTINGS = {
    # Caching settings
    'cache_gaussian_symbols': True,        # Cache Fourier transforms of Gaussians
    'cache_connection_matrices': True,     # Cache connection matrices per parameter set
    'cache_mode_grids': True,             # Cache Fourier mode coordinate grids
    'max_cache_size_mb': 500,             # Maximum cache size in MB
    
    # Parallelization settings
    'enable_parallel_processing': False,   # Enable parallel parameter sweep (future)
    'max_parallel_workers': 4,            # Maximum parallel workers
    
    # Memory management
    'batch_parameter_points': True,       # Process parameters in batches
    'batch_size': 100,                    # Number of parameter points per batch
    'clear_intermediate_cache': True,     # Clear caches between batches
}

#------------------------------------------------------------------------------
# Consolidated Parameter Dictionary
#------------------------------------------------------------------------------

# Main configuration dictionary for easy access
BIFURCATION_PARAMS = {
    'fourier_grid': FOURIER_GRID_PARAMS,
    'parameter_ranges': PARAMETER_RANGES,
    'reference_params': REFERENCE_PARAMS,
    'numerical_tolerances': NUMERICAL_TOLERANCES,
    'validation_settings': VALIDATION_SETTINGS,
    'output_settings': OUTPUT_SETTINGS,
    'population_indices': POPULATION_INDICES,
    'biological_constraints': BIOLOGICAL_CONSTRAINTS,
    'optimization_settings': OPTIMIZATION_SETTINGS,
    'n_populations': N_POPULATIONS,
}

#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

def get_population_index(layer: str, cell_type: str) -> int:
    """Get the index for a specific population in the 9-population system.
    
    Args:
        layer: Layer name ('L23', 'L4', 'L5')
        cell_type: Cell type ('E', 'SST', 'PV')
        
    Returns:
        Index in the 9-population system (0-8)
    """
    return POPULATION_INDICES[(layer, cell_type)]

def get_population_from_index(index: int) -> Tuple[str, str]:
    """Get layer and cell type from population index.
    
    Args:
        index: Population index (0-8)
        
    Returns:
        Tuple of (layer, cell_type)
    """
    return INDEX_TO_POPULATION[index]

def get_mode_coordinates(grid_size: int) -> List[Tuple[int, int]]:
    """Generate all Fourier mode coordinates for the grid.
    
    Args:
        grid_size: Size of the spatial grid
        
    Returns:
        List of (nx, ny) tuples for all modes
    """
    modes = []
    for nx in range(grid_size):
        for ny in range(grid_size):
            modes.append((nx, ny))
    return modes

def compute_mode_radius(nx: int, ny: int) -> float:
    """Compute the radius ||n|| for a Fourier mode.
    
    Args:
        nx, ny: Mode indices
        
    Returns:
        Mode radius sqrt(nx² + ny²)
    """
    return np.sqrt(nx**2 + ny**2)

def print_config_summary():
    """Print a summary of the bifurcation analysis configuration."""
    print("Bifurcation Analysis Configuration Summary:")
    print(f"  Grid size: {FOURIER_GRID_PARAMS['grid_size']}")
    print(f"  Number of populations: {N_POPULATIONS}")
    print(f"  Parameter sweep resolution: {len(PARAMETER_RANGES['pv_analysis']['tau_ratio_range'])}×{len(PARAMETER_RANGES['pv_analysis']['sigma_ratio_range'])}")
    print(f"  Stability threshold: {NUMERICAL_TOLERANCES['stability_threshold']}")
    print(f"  Kernel normalization: {'Enabled' if VALIDATION_SETTINGS['check_kernel_normalization'] else 'Disabled'}")
    print(f"  Validation: {'Enabled' if VALIDATION_SETTINGS['validate_fixed_point_convergence'] else 'Disabled'}")
