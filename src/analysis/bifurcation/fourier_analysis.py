"""Fourier analysis utilities for bifurcation analysis.

This module implements Gaussian kernel normalization, Fourier transform computation,
and connection matrix building for the bifurcation analysis. It ensures proper
normalization of Gaussian kernels and correct DC gain validation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from functools import lru_cache
import warnings

from src.model.connectivity import ConnectivityProfile, LayerConnectivity
from .config import (
    FOURIER_GRID_PARAMS, 
    NUMERICAL_TOLERANCES, 
    VALIDATION_SETTINGS,
    N_POPULATIONS,
    get_population_index,
    compute_mode_radius
)


class GaussianKernelValidator:
    """Validates and normalizes Gaussian kernels for proper Fourier analysis."""
    
    def __init__(self, tolerance: float = NUMERICAL_TOLERANCES['kernel_normalization_tolerance']):
        """Initialize validator with normalization tolerance.
        
        Args:
            tolerance: Tolerance for kernel sum validation
        """
        self.tolerance = tolerance
        self.validation_results = {}
        
    def validate_kernel_normalization(self, kernel: np.ndarray, 
                                    kernel_name: str = "unknown") -> bool:
        """Check if a discrete Gaussian kernel is properly normalized.
        
        Args:
            kernel: 2D array representing the Gaussian kernel
            kernel_name: Name for diagnostic purposes
            
        Returns:
            True if kernel sums to 1.0 within tolerance, False otherwise
        """
        kernel_sum = np.sum(kernel)
        is_normalized = abs(kernel_sum - 1.0) < self.tolerance
        
        # Store validation result for diagnostics
        self.validation_results[kernel_name] = {
            'sum': kernel_sum,
            'is_normalized': is_normalized,
            'error': abs(kernel_sum - 1.0)
        }
        
        if not is_normalized and VALIDATION_SETTINGS['report_normalization_status']:
            warnings.warn(f"Kernel '{kernel_name}' not normalized: sum = {kernel_sum:.10f}")
            
        return is_normalized
    
    def normalize_gaussian_kernel(self, kernel: np.ndarray, 
                                kernel_name: str = "unknown") -> np.ndarray:
        """Force normalization of a Gaussian kernel to unit mass.
        
        Args:
            kernel: 2D array representing the Gaussian kernel
            kernel_name: Name for diagnostic purposes
            
        Returns:
            Normalized kernel that sums to 1.0
        """
        kernel_sum = np.sum(kernel)
        
        if kernel_sum == 0:
            raise ValueError(f"Cannot normalize kernel '{kernel_name}': sum is zero")
            
        normalized_kernel = kernel / kernel_sum
        
        # Validate the normalized kernel
        if VALIDATION_SETTINGS['check_kernel_normalization']:
            self.validate_kernel_normalization(normalized_kernel, f"{kernel_name}_normalized")
            
        return normalized_kernel
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get a summary of all kernel validation results.
        
        Returns:
            Dictionary containing validation statistics
        """
        if not self.validation_results:
            return {"message": "No kernels validated yet"}
            
        total_kernels = len(self.validation_results)
        normalized_kernels = sum(1 for result in self.validation_results.values() 
                               if result['is_normalized'])
        max_error = max(result['error'] for result in self.validation_results.values())
        
        return {
            'total_kernels': total_kernels,
            'normalized_kernels': normalized_kernels,
            'normalization_rate': normalized_kernels / total_kernels,
            'max_normalization_error': max_error,
            'tolerance': self.tolerance,
            'details': self.validation_results
        }


class GaussianFourierTransform:
    """Computes Fourier transforms of Gaussian connectivity kernels."""
    
    def __init__(self, grid_size: int = FOURIER_GRID_PARAMS['grid_size']):
        """Initialize Fourier transform utilities.
        
        Args:
            grid_size: Size of the spatial grid
        """
        self.grid_size = grid_size
        self.symbol_cache = {}
        self.mode_grid_cache = None
        
    @lru_cache(maxsize=1000)
    def compute_gaussian_symbol(self, sigma: float, nx: int, ny: int) -> float:
        """Compute Fourier symbol for Gaussian kernel at mode (nx, ny).
        
        Uses the correct convention: exp(-2π²σ²||n||²) where ||n||² = nx² + ny²
        
        Args:
            sigma: Width of the Gaussian kernel
            nx, ny: Integer mode indices
            
        Returns:
            Fourier symbol value at this mode
        """
        mode_radius_squared = nx * nx + ny * ny
        symbol = np.exp(-2.0 * np.pi**2 * sigma**2 * mode_radius_squared)
        return symbol
    
    def compute_connection_symbol(self, amplitude: float, sigma: float, 
                                nx: int, ny: int) -> float:
        """Compute connection Fourier symbol: A * exp(-2π²σ²||n||²).
        
        Args:
            amplitude: Connection amplitude (signed: positive for excitatory, negative for inhibitory)
            sigma: Connection width
            nx, ny: Integer mode indices
            
        Returns:
            Connection symbol value at this mode
        """
        gaussian_symbol = self.compute_gaussian_symbol(sigma, nx, ny)
        return amplitude * gaussian_symbol
    
    def validate_dc_gain(self, amplitude: float, sigma: float, 
                        tolerance: float = NUMERICAL_TOLERANCES['dc_gain_tolerance']) -> bool:
        """Validate that DC gain (n=0) equals the amplitude exactly.
        
        Args:
            amplitude: Expected amplitude
            sigma: Connection width (should not affect DC gain)
            tolerance: Tolerance for validation
            
        Returns:
            True if DC gain matches amplitude within tolerance
        """
        dc_symbol = self.compute_connection_symbol(amplitude, sigma, 0, 0)
        error = abs(dc_symbol - amplitude)
        
        is_valid = error < tolerance
        
        if not is_valid and VALIDATION_SETTINGS['report_normalization_status']:
            warnings.warn(f"DC gain validation failed: expected {amplitude}, got {dc_symbol}")
            
        return is_valid
    
    def generate_mode_grid(self) -> List[Tuple[int, int]]:
        """Generate all Fourier mode coordinates for the grid.
        
        Returns:
            List of (nx, ny) tuples for all modes
        """
        if self.mode_grid_cache is None:
            modes = []
            for nx in range(self.grid_size):
                for ny in range(self.grid_size):
                    modes.append((nx, ny))
            self.mode_grid_cache = modes
            
        return self.mode_grid_cache
    
    def get_mode_radius(self, nx: int, ny: int) -> float:
        """Compute mode radius ||n|| = sqrt(nx² + ny²).
        
        Args:
            nx, ny: Mode indices
            
        Returns:
            Mode radius
        """
        return compute_mode_radius(nx, ny)
    
    def get_physical_wavenumber(self, nx: int, ny: int, 
                              domain_length: float = FOURIER_GRID_PARAMS['domain_length']) -> float:
        """Convert mode indices to physical wavenumber.
        
        Args:
            nx, ny: Mode indices
            domain_length: Physical size of the domain
            
        Returns:
            Physical wavenumber 2π||n||/L
        """
        mode_radius = self.get_mode_radius(nx, ny)
        return 2.0 * np.pi * mode_radius / domain_length
    
    def clear_cache(self):
        """Clear all cached computations."""
        self.compute_gaussian_symbol.cache_clear()
        self.symbol_cache.clear()
        self.mode_grid_cache = None


class ConnectivityNormalizer:
    """Ensures existing connectivity profiles are properly normalized."""
    
    def __init__(self, connectivity_profile: ConnectivityProfile):
        """Initialize with existing connectivity profile.
        
        Args:
            connectivity_profile: Existing ConnectivityProfile instance
        """
        self.connectivity_profile = connectivity_profile
        self.validator = GaussianKernelValidator()
        self.normalization_applied = {}
        
    def check_and_normalize_profile(self, sigma: float, 
                                  center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Check and normalize a Gaussian profile if needed.
        
        Args:
            sigma: Width of the Gaussian
            center: Center coordinates (optional)
            
        Returns:
            Normalized Gaussian profile
        """
        # Get the profile from existing connectivity system
        profile = self.connectivity_profile.gaussian_profile(sigma, center)
        
        # Create a name for this profile
        center_str = f"_{center[0]}_{center[1]}" if center else "_center"
        profile_name = f"sigma_{sigma:.3f}{center_str}"
        
        # Check if it's normalized
        is_normalized = self.validator.validate_kernel_normalization(profile, profile_name)
        
        if not is_normalized and VALIDATION_SETTINGS['force_kernel_normalization']:
            # Normalize the profile
            normalized_profile = self.validator.normalize_gaussian_kernel(profile, profile_name)
            
            # Update the cache in the connectivity profile
            cache_key = (sigma, center if center else 'center')
            self.connectivity_profile._profile_cache[cache_key] = normalized_profile
            
            self.normalization_applied[profile_name] = True
            
            if VALIDATION_SETTINGS['report_normalization_status']:
                print(f"Applied normalization to profile: {profile_name}")
                
            return normalized_profile
        else:
            self.normalization_applied[profile_name] = False
            return profile
    
    def normalize_all_cached_profiles(self):
        """Check and normalize all cached profiles in the connectivity system."""
        if not hasattr(self.connectivity_profile, '_profile_cache'):
            return
            
        profiles_to_update = {}
        
        for cache_key, profile in self.connectivity_profile._profile_cache.items():
            sigma, _ = cache_key if isinstance(cache_key, tuple) else (cache_key, None)
            
            # Skip non-numeric sigma values
            if not isinstance(sigma, (int, float)):
                continue
                
            profile_name = f"cached_sigma_{sigma:.3f}"
            is_normalized = self.validator.validate_kernel_normalization(profile, profile_name)
            
            if not is_normalized and VALIDATION_SETTINGS['force_kernel_normalization']:
                normalized_profile = self.validator.normalize_gaussian_kernel(profile, profile_name)
                profiles_to_update[cache_key] = normalized_profile
                self.normalization_applied[profile_name] = True
        
        # Update the cache with normalized profiles
        for cache_key, normalized_profile in profiles_to_update.items():
            self.connectivity_profile._profile_cache[cache_key] = normalized_profile
            
        if profiles_to_update and VALIDATION_SETTINGS['report_normalization_status']:
            print(f"Normalized {len(profiles_to_update)} cached profiles")
    
    def get_normalization_report(self) -> Dict[str, Any]:
        """Get report on normalization activities.
        
        Returns:
            Dictionary with normalization statistics
        """
        validator_report = self.validator.get_validation_report()
        
        total_profiles = len(self.normalization_applied)
        normalized_profiles = sum(1 for applied in self.normalization_applied.values() if applied)
        
        return {
            'profiles_processed': total_profiles,
            'profiles_normalized': normalized_profiles,
            'validator_report': validator_report,
            'normalization_details': self.normalization_applied
        }


class ConnectionMatrixBuilder:
    """Builds 9×9 Fourier symbol matrices for each spatial mode."""
    
    def __init__(self, layer_connectivity: LayerConnectivity):
        """Initialize with existing layer connectivity.
        
        Args:
            layer_connectivity: Existing LayerConnectivity instance
        """
        self.layer_connectivity = layer_connectivity
        self.fourier_transform = GaussianFourierTransform()
        self.normalizer = ConnectivityNormalizer(layer_connectivity.profile)
        
        # Ensure all profiles are normalized
        if VALIDATION_SETTINGS['check_kernel_normalization']:
            self.normalizer.normalize_all_cached_profiles()
    
    def build_connection_matrix_symbol(self, nx: int, ny: int) -> np.ndarray:
        """Build 9×9 connection matrix W̃(n) for Fourier mode (nx, ny).
        
        Args:
            nx, ny: Fourier mode indices
            
        Returns:
            9×9 connection matrix with Fourier symbols
        """
        # Initialize 9×9 matrix (rows=targets, columns=sources)
        W_tilde = np.zeros((N_POPULATIONS, N_POPULATIONS))
        
        # Loop over all possible connections
        for conn_key, params in self.layer_connectivity.layer_params.items():
            # Parse connection key to extract source and target
            source_layer, source_cell, target_layer, target_cell = self._parse_connection_key(conn_key)
            
            if source_layer is None:  # Skip invalid connections
                continue
                
            # Get population indices
            try:
                source_idx = get_population_index(source_layer, source_cell) if source_cell else None
                target_idx = get_population_index(target_layer, target_cell)
            except KeyError:
                # Skip connections with invalid population names
                continue
            
            # Skip thalamic connections (they're not part of the 9×9 recurrent matrix)
            if source_layer == 'thalamus':
                continue
                
            # Get connection parameters
            amplitude = params['amplitude']  # Signed amplitude
            sigma = params['sigma']  # Connection width
            
            # Apply strength scaling
            scaled_amplitude = amplitude * self.layer_connectivity.strength_scaling.get(source_cell, 1.0)
            
            # Validate kernel normalization for this sigma
            if VALIDATION_SETTINGS['check_kernel_normalization']:
                self.normalizer.check_and_normalize_profile(sigma)
            
            # Compute Fourier symbol with scaled amplitude
            connection_symbol = self.fourier_transform.compute_connection_symbol(
                scaled_amplitude, sigma, nx, ny
            )
            
            # Store in matrix
            W_tilde[target_idx, source_idx] = connection_symbol
        
        return W_tilde
    
    def _parse_connection_key(self, conn_key: str) -> Tuple[Optional[str], Optional[str], str, str]:
        """Parse connection key into source and target components.
        
        Args:
            conn_key: Connection key like 'L23_E_to_L4_SST' or 'thalamus_to_L4_E'
            
        Returns:
            Tuple of (source_layer, source_cell, target_layer, target_cell)
        """
        try:
            if conn_key.startswith('thalamus_to_'):
                # Thalamic connection: 'thalamus_to_L4_E'
                parts = conn_key.split('_')
                if len(parts) >= 4:
                    target_layer = parts[2]
                    target_cell = parts[3]
                    return 'thalamus', None, target_layer, target_cell
            elif '_to_' in conn_key:
                # Regular connection: 'L23_E_to_L4_SST'
                source_part, target_part = conn_key.split('_to_')
                source_parts = source_part.split('_')
                target_parts = target_part.split('_')
                
                if len(source_parts) >= 2 and len(target_parts) >= 2:
                    source_layer = source_parts[0]
                    source_cell = source_parts[1]
                    target_layer = target_parts[0]
                    target_cell = target_parts[1]
                    return source_layer, source_cell, target_layer, target_cell
        except (IndexError, ValueError):
            pass
            
        return None, None, "", ""
    
    def validate_dc_gains(self) -> Dict[str, bool]:
        """Validate DC gains for all connections.
        
        Returns:
            Dictionary mapping connection keys to validation results
        """
        validation_results = {}
        
        for conn_key, params in self.layer_connectivity.layer_params.items():
            amplitude = params['amplitude']
            sigma = params['sigma']
            
            # Skip thalamic connections for now
            if conn_key.startswith('thalamus_to_'):
                continue
                
            is_valid = self.fourier_transform.validate_dc_gain(amplitude, sigma)
            validation_results[conn_key] = is_valid
            
        return validation_results
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get summary of connection matrix properties.
        
        Returns:
            Dictionary with connection statistics
        """
        # Count connections by type
        excitatory_connections = 0
        inhibitory_connections = 0
        zero_connections = 0
        
        for params in self.layer_connectivity.layer_params.items():
            amplitude = params[1]['amplitude']
            if amplitude > 0:
                excitatory_connections += 1
            elif amplitude < 0:
                inhibitory_connections += 1
            else:
                zero_connections += 1
        
        # Get normalization report
        normalization_report = self.normalizer.get_normalization_report()
        
        # Validate DC gains
        dc_validation = self.validate_dc_gains()
        valid_dc_gains = sum(1 for is_valid in dc_validation.values() if is_valid)
        
        return {
            'total_connections': len(self.layer_connectivity.layer_params),
            'excitatory_connections': excitatory_connections,
            'inhibitory_connections': inhibitory_connections,
            'zero_connections': zero_connections,
            'valid_dc_gains': valid_dc_gains,
            'dc_validation_rate': valid_dc_gains / len(dc_validation) if dc_validation else 0,
            'normalization_report': normalization_report
        }


def validate_fourier_analysis_setup(layer_connectivity: LayerConnectivity) -> Dict[str, Any]:
    """Run comprehensive validation of Fourier analysis setup.
    
    Args:
        layer_connectivity: LayerConnectivity instance to validate
        
    Returns:
        Dictionary with validation results
    """
    print("Validating Fourier Analysis Setup...")
    
    # Initialize components
    matrix_builder = ConnectionMatrixBuilder(layer_connectivity)
    fourier_transform = GaussianFourierTransform()
    
    # Test basic Fourier symbol computation
    print("  Testing Fourier symbol computation...")
    test_sigma = 2.0
    dc_symbol = fourier_transform.compute_gaussian_symbol(test_sigma, 0, 0)
    nonzero_symbol = fourier_transform.compute_gaussian_symbol(test_sigma, 1, 1)
    
    dc_correct = abs(dc_symbol - 1.0) < NUMERICAL_TOLERANCES['dc_gain_tolerance']
    nonzero_reasonable = 0 < nonzero_symbol < 1.0
    
    print(f"    DC symbol (should be 1.0): {dc_symbol:.10f} - {'✓' if dc_correct else '✗'}")
    print(f"    Non-zero mode symbol: {nonzero_symbol:.6f} - {'✓' if nonzero_reasonable else '✗'}")
    
    # Test connection matrix building
    print("  Testing connection matrix building...")
    try:
        W_dc = matrix_builder.build_connection_matrix_symbol(0, 0)
        matrix_builder.build_connection_matrix_symbol(1, 1)  # Test non-zero mode
        
        matrix_shape_correct = W_dc.shape == (N_POPULATIONS, N_POPULATIONS)
        has_nonzero_entries = np.any(W_dc != 0)
        
        print(f"    Matrix shape: {W_dc.shape} - {'✓' if matrix_shape_correct else '✗'}")
        print(f"    Has non-zero entries: {'✓' if has_nonzero_entries else '✗'}")
        
    except (ValueError, KeyError, IndexError) as e:
        print(f"    Matrix building failed: {e}")
        W_dc = None
        matrix_shape_correct = False
        has_nonzero_entries = False
    
    # Get connection summary
    connection_summary = matrix_builder.get_connection_summary()
    print("  Connection Summary:")
    print(f"    Total connections: {connection_summary['total_connections']}")
    print(f"    Excitatory: {connection_summary['excitatory_connections']}")
    print(f"    Inhibitory: {connection_summary['inhibitory_connections']}")
    print(f"    DC gain validation rate: {connection_summary['dc_validation_rate']:.2%}")
    
    # Test mode grid generation
    print("  Testing mode grid generation...")
    modes = fourier_transform.generate_mode_grid()
    expected_modes = FOURIER_GRID_PARAMS['grid_size'] ** 2
    mode_count_correct = len(modes) == expected_modes
    
    print(f"    Mode count: {len(modes)} (expected {expected_modes}) - {'✓' if mode_count_correct else '✗'}")
    
    validation_results = {
        'fourier_symbols': {
            'dc_correct': dc_correct,
            'nonzero_reasonable': nonzero_reasonable
        },
        'connection_matrices': {
            'shape_correct': matrix_shape_correct,
            'has_entries': has_nonzero_entries
        },
        'mode_grid': {
            'count_correct': mode_count_correct,
            'total_modes': len(modes)
        },
        'connection_summary': connection_summary,
        'overall_success': all([
            dc_correct, nonzero_reasonable, matrix_shape_correct, 
            has_nonzero_entries, mode_count_correct
        ])
    }
    
    print(f"\nValidation {'PASSED' if validation_results['overall_success'] else 'FAILED'}")
    
    return validation_results
