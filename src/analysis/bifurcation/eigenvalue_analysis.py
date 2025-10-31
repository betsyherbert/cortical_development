"""Eigenvalue analysis for bifurcation analysis.

This module implements comprehensive eigenvalue analysis for stability determination
and mode classification. It finds the most unstable modes, computes spatial frequencies,
and classifies system stability according to Figure 5A methodology.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from enum import Enum

from .config import (
    NUMERICAL_TOLERANCES,
    FOURIER_GRID_PARAMS
)
from .jacobian_builder import JacobianData, PerModeJacobianBuilder, ModeGridProcessor
from .fixed_point_solver import FixedPointResult


class StabilityRegime(Enum):
    """Classification of stability regimes."""
    STABLE = "stable"
    UNSTABLE_DC = "unstable_dc"
    UNSTABLE_FINITE = "unstable_finite"
    UNKNOWN = "unknown"


@dataclass
class EigenvalueAnalysisResult:
    """Result container for eigenvalue analysis."""
    max_real_eigenvalue: float  # λ* = max_k Re(λ_max(k))
    is_stable: bool  # True if λ* < -ε
    stability_regime: StabilityRegime  # Classification of stability
    winning_mode_indices: Tuple[int, int]  # (nx*, ny*) of most unstable mode
    winning_mode_radius: float  # ||n*|| of most unstable mode
    physical_wavenumber: float  # 2π||n*||/L for plotting
    
    # Statistics
    total_modes: int  # Total number of modes analyzed
    stable_modes: int  # Number of stable modes
    unstable_modes: int  # Number of unstable modes
    
    # Detailed results
    all_mode_data: List[JacobianData]  # All Jacobian data
    eigenvalue_spectrum: Dict[Tuple[int, int], complex]  # Mode -> max eigenvalue
    mode_radii: Dict[Tuple[int, int], float]  # Mode -> ||n||
    
    # Analysis metadata
    stability_threshold: float  # ε threshold used
    grid_size: int  # Size of Fourier grid
    domain_length: float  # Physical domain size


class EigenvalueAnalyzer:
    """Comprehensive eigenvalue analysis for stability determination."""
    
    def __init__(self, jacobian_builder: PerModeJacobianBuilder,
                 stability_threshold: float = NUMERICAL_TOLERANCES['stability_threshold'],
                 domain_length: float = FOURIER_GRID_PARAMS['domain_length']):
        """Initialize eigenvalue analyzer.
        
        Args:
            jacobian_builder: PerModeJacobianBuilder instance
            stability_threshold: Threshold ε for stability (λ* < -ε)
            domain_length: Physical domain size for wavenumber conversion
        """
        self.jacobian_builder = jacobian_builder
        self.mode_processor = ModeGridProcessor(jacobian_builder)
        self.stability_threshold = stability_threshold
        self.domain_length = domain_length
        
        # Analysis cache
        self.analysis_cache = {}
        
    def analyze_stability(self, fixed_point_result: FixedPointResult,
                         grid_size: Optional[int] = None) -> EigenvalueAnalysisResult:
        """Perform complete eigenvalue analysis for stability determination.
        
        Args:
            fixed_point_result: Fixed point solution
            grid_size: Optional grid size override
            
        Returns:
            EigenvalueAnalysisResult with complete analysis
        """
        if grid_size is None:
            grid_size = self.jacobian_builder.fourier_transform.grid_size
        
        # Check cache
        cache_key = (id(fixed_point_result), grid_size)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Process all modes
        print(f"Analyzing eigenvalues for {grid_size}×{grid_size} mode grid...")
        all_mode_data = self.mode_processor.process_all_modes(fixed_point_result, grid_size)
        
        # Find most unstable mode
        max_real_eigenvalue, most_unstable_mode = self.mode_processor.find_most_unstable_mode(all_mode_data)
        
        # Determine stability
        is_stable = max_real_eigenvalue < -self.stability_threshold
        
        # Get winning mode information
        if most_unstable_mode is not None:
            winning_mode_indices = most_unstable_mode.mode_indices
            winning_mode_radius = most_unstable_mode.mode_radius
        else:
            winning_mode_indices = (0, 0)
            winning_mode_radius = 0.0
        
        # Compute physical wavenumber
        physical_wavenumber = 2.0 * np.pi * winning_mode_radius / self.domain_length
        
        # Classify stability regime
        stability_regime = self._classify_stability_regime(
            is_stable, winning_mode_indices, max_real_eigenvalue
        )
        
        # Count stable vs unstable modes
        stable_modes = 0
        unstable_modes = 0
        eigenvalue_spectrum = {}
        mode_radii = {}
        
        for mode_data in all_mode_data:
            if mode_data.max_eigenvalue is not None:
                real_part = np.real(mode_data.max_eigenvalue)
                mode_indices = mode_data.mode_indices
                
                # Store in spectrum
                eigenvalue_spectrum[mode_indices] = mode_data.max_eigenvalue
                mode_radii[mode_indices] = mode_data.mode_radius
                
                # Count stability
                if real_part < -self.stability_threshold:
                    stable_modes += 1
                else:
                    unstable_modes += 1
        
        # Create result
        result = EigenvalueAnalysisResult(
            max_real_eigenvalue=max_real_eigenvalue,
            is_stable=is_stable,
            stability_regime=stability_regime,
            winning_mode_indices=winning_mode_indices,
            winning_mode_radius=winning_mode_radius,
            physical_wavenumber=physical_wavenumber,
            total_modes=len(all_mode_data),
            stable_modes=stable_modes,
            unstable_modes=unstable_modes,
            all_mode_data=all_mode_data,
            eigenvalue_spectrum=eigenvalue_spectrum,
            mode_radii=mode_radii,
            stability_threshold=self.stability_threshold,
            grid_size=grid_size,
            domain_length=self.domain_length
        )
        
        # Cache result
        self.analysis_cache[cache_key] = result
        
        return result
    
    def _classify_stability_regime(self, is_stable: bool, 
                                 winning_mode: Tuple[int, int],
                                 max_eigenvalue: float) -> StabilityRegime:
        """Classify the stability regime based on eigenvalue analysis.
        
        Args:
            is_stable: Whether system is stable overall
            winning_mode: Mode indices of most unstable mode
            max_eigenvalue: Maximum real eigenvalue (currently unused but kept for future extensions)
            
        Returns:
            StabilityRegime classification
        """
        # Note: max_eigenvalue could be used for finer classification in the future
        _ = max_eigenvalue  # Acknowledge unused parameter
        
        if is_stable:
            return StabilityRegime.STABLE
        elif winning_mode == (0, 0):
            # DC mode instability - global fluctuations
            return StabilityRegime.UNSTABLE_DC
        else:
            # Finite mode instability - spatially structured patterns
            return StabilityRegime.UNSTABLE_FINITE
    
    def compute_eigenvalue_spectrum(self, analysis_result: EigenvalueAnalysisResult) -> Dict[str, Any]:
        """Compute detailed eigenvalue spectrum analysis.
        
        Args:
            analysis_result: EigenvalueAnalysisResult from analyze_stability
            
        Returns:
            Dictionary with spectrum analysis
        """
        spectrum_data = {
            'mode_radii': [],
            'real_eigenvalues': [],
            'imaginary_eigenvalues': [],
            'mode_indices': [],
            'stability_flags': []
        }
        
        # Extract data for all modes
        for mode_indices, eigenvalue in analysis_result.eigenvalue_spectrum.items():
            mode_radius = analysis_result.mode_radii[mode_indices]
            real_part = np.real(eigenvalue)
            imag_part = np.imag(eigenvalue)
            is_stable = real_part < -self.stability_threshold
            
            spectrum_data['mode_radii'].append(mode_radius)
            spectrum_data['real_eigenvalues'].append(real_part)
            spectrum_data['imaginary_eigenvalues'].append(imag_part)
            spectrum_data['mode_indices'].append(mode_indices)
            spectrum_data['stability_flags'].append(is_stable)
        
        # Convert to arrays for analysis
        mode_radii = np.array(spectrum_data['mode_radii'])
        real_eigenvalues = np.array(spectrum_data['real_eigenvalues'])
        
        # Compute statistics
        spectrum_analysis = {
            'spectrum_data': spectrum_data,
            'statistics': {
                'min_real_eigenvalue': np.min(real_eigenvalues),
                'max_real_eigenvalue': np.max(real_eigenvalues),
                'mean_real_eigenvalue': np.mean(real_eigenvalues),
                'std_real_eigenvalue': np.std(real_eigenvalues),
                'min_mode_radius': np.min(mode_radii),
                'max_mode_radius': np.max(mode_radii),
                'dc_mode_eigenvalue': analysis_result.eigenvalue_spectrum.get((0, 0), None)
            }
        }
        
        # Find eigenvalue vs mode radius relationship
        if len(mode_radii) > 1:
            # Sort by mode radius for analysis
            sort_indices = np.argsort(mode_radii)
            sorted_radii = mode_radii[sort_indices]
            sorted_eigenvalues = real_eigenvalues[sort_indices]
            
            spectrum_analysis['sorted_by_radius'] = {
                'mode_radii': sorted_radii,
                'real_eigenvalues': sorted_eigenvalues
            }
        
        return spectrum_analysis
    
    def find_stability_boundary(self, analysis_result: EigenvalueAnalysisResult) -> Dict[str, Any]:
        """Find the boundary between stable and unstable modes in k-space.
        
        Args:
            analysis_result: EigenvalueAnalysisResult from analyze_stability
            
        Returns:
            Dictionary with stability boundary information
        """
        if analysis_result.is_stable:
            return {
                'has_boundary': False,
                'reason': 'All modes stable'
            }
        
        # Group modes by stability
        stable_radii = []
        unstable_radii = []
        
        for mode_indices, eigenvalue in analysis_result.eigenvalue_spectrum.items():
            mode_radius = analysis_result.mode_radii[mode_indices]
            real_part = np.real(eigenvalue)
            
            if real_part < -self.stability_threshold:
                stable_radii.append(mode_radius)
            else:
                unstable_radii.append(mode_radius)
        
        stable_radii = np.array(stable_radii)
        unstable_radii = np.array(unstable_radii)
        
        boundary_info = {
            'has_boundary': True,
            'stable_radius_range': (np.min(stable_radii), np.max(stable_radii)) if len(stable_radii) > 0 else None,
            'unstable_radius_range': (np.min(unstable_radii), np.max(unstable_radii)) if len(unstable_radii) > 0 else None,
            'n_stable_modes': len(stable_radii),
            'n_unstable_modes': len(unstable_radii)
        }
        
        # Estimate boundary location
        if len(stable_radii) > 0 and len(unstable_radii) > 0:
            # Simple estimate: midpoint between max stable and min unstable
            max_stable_radius = np.max(stable_radii)
            min_unstable_radius = np.min(unstable_radii)
            boundary_info['estimated_boundary_radius'] = (max_stable_radius + min_unstable_radius) / 2.0
        
        return boundary_info
    
    def compare_parameter_points(self, results_list: List[EigenvalueAnalysisResult]) -> Dict[str, Any]:
        """Compare eigenvalue analysis results across multiple parameter points.
        
        Args:
            results_list: List of EigenvalueAnalysisResult from different parameter points
            
        Returns:
            Dictionary with comparison analysis
        """
        if not results_list:
            return {'error': 'No results provided'}
        
        comparison = {
            'n_points': len(results_list),
            'stability_summary': {
                'stable_points': 0,
                'unstable_dc_points': 0,
                'unstable_finite_points': 0
            },
            'eigenvalue_trends': {
                'max_eigenvalues': [],
                'winning_mode_radii': [],
                'stability_flags': []
            },
            'regime_transitions': []
        }
        
        # Analyze each result
        for i, result in enumerate(results_list):
            # Count stability regimes
            if result.stability_regime == StabilityRegime.STABLE:
                comparison['stability_summary']['stable_points'] += 1
            elif result.stability_regime == StabilityRegime.UNSTABLE_DC:
                comparison['stability_summary']['unstable_dc_points'] += 1
            elif result.stability_regime == StabilityRegime.UNSTABLE_FINITE:
                comparison['stability_summary']['unstable_finite_points'] += 1
            
            # Track trends
            comparison['eigenvalue_trends']['max_eigenvalues'].append(result.max_real_eigenvalue)
            comparison['eigenvalue_trends']['winning_mode_radii'].append(result.winning_mode_radius)
            comparison['eigenvalue_trends']['stability_flags'].append(result.is_stable)
            
            # Detect regime transitions
            if i > 0:
                prev_regime = results_list[i-1].stability_regime
                curr_regime = result.stability_regime
                if prev_regime != curr_regime:
                    comparison['regime_transitions'].append({
                        'from_index': i-1,
                        'to_index': i,
                        'from_regime': prev_regime.value,
                        'to_regime': curr_regime.value
                    })
        
        # Convert to arrays for statistics
        max_eigenvalues = np.array(comparison['eigenvalue_trends']['max_eigenvalues'])
        winning_radii = np.array(comparison['eigenvalue_trends']['winning_mode_radii'])
        
        comparison['statistics'] = {
            'eigenvalue_range': (np.min(max_eigenvalues), np.max(max_eigenvalues)),
            'eigenvalue_mean': np.mean(max_eigenvalues),
            'radius_range': (np.min(winning_radii), np.max(winning_radii)),
            'radius_mean': np.mean(winning_radii),
            'stability_rate': np.mean(comparison['eigenvalue_trends']['stability_flags'])
        }
        
        return comparison
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        self.jacobian_builder.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the analysis cache.
        
        Returns:
            Dictionary with cache statistics
        """
        jacobian_stats = self.jacobian_builder.get_cache_stats()
        
        return {
            'analysis_cache_size': len(self.analysis_cache),
            'jacobian_cache_stats': jacobian_stats,
            'total_memory_estimate': (
                len(self.analysis_cache) * 1000 +  # Rough estimate for analysis results
                jacobian_stats['memory_usage_estimate']
            )
        }


class BifurcationPointClassifier:
    """Classifies parameter points for bifurcation diagram construction."""
    
    def __init__(self, stability_threshold: float = NUMERICAL_TOLERANCES['stability_threshold']):
        """Initialize bifurcation point classifier.
        
        Args:
            stability_threshold: Threshold for stability classification
        """
        self.stability_threshold = stability_threshold
        
    def classify_point(self, analysis_result: EigenvalueAnalysisResult) -> Dict[str, Any]:
        """Classify a single parameter point for bifurcation diagram.
        
        Args:
            analysis_result: EigenvalueAnalysisResult from eigenvalue analysis
            
        Returns:
            Dictionary with classification for plotting
        """
        classification = {
            'is_stable': analysis_result.is_stable,
            'stability_regime': analysis_result.stability_regime.value,
            'color_value': self._compute_color_value(analysis_result),
            'max_eigenvalue': analysis_result.max_real_eigenvalue,
            'winning_mode_radius': analysis_result.winning_mode_radius,
            'physical_wavenumber': analysis_result.physical_wavenumber
        }
        
        return classification
    
    def _compute_color_value(self, analysis_result: EigenvalueAnalysisResult) -> Optional[float]:
        """Compute color value for bifurcation diagram (Figure 5A style).
        
        Args:
            analysis_result: EigenvalueAnalysisResult from eigenvalue analysis
            
        Returns:
            Color value (mode radius) or None for stable regions
        """
        if analysis_result.is_stable:
            # Stable regions are colored gray (None indicates this)
            return None
        else:
            # Unstable regions colored by winning mode radius ||n*||
            return analysis_result.winning_mode_radius
    
    def create_bifurcation_map(self, parameter_grid: np.ndarray,
                             analysis_results: List[List[EigenvalueAnalysisResult]]) -> Dict[str, Any]:
        """Create bifurcation map data for visualization.
        
        Args:
            parameter_grid: 2D grid of parameter values
            analysis_results: 2D grid of EigenvalueAnalysisResult objects
            
        Returns:
            Dictionary with bifurcation map data
        """
        grid_shape = parameter_grid.shape[:2]  # (n_param1, n_param2)
        
        # Initialize output arrays
        stability_map = np.zeros(grid_shape, dtype=bool)
        color_map = np.full(grid_shape, np.nan)
        eigenvalue_map = np.zeros(grid_shape)
        regime_map = np.full(grid_shape, '', dtype='<U20')
        
        # Process each point
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if i < len(analysis_results) and j < len(analysis_results[i]):
                    result = analysis_results[i][j]
                    classification = self.classify_point(result)
                    
                    stability_map[i, j] = classification['is_stable']
                    eigenvalue_map[i, j] = classification['max_eigenvalue']
                    regime_map[i, j] = classification['stability_regime']
                    
                    if classification['color_value'] is not None:
                        color_map[i, j] = classification['color_value']
        
        # Compute statistics
        total_points = grid_shape[0] * grid_shape[1]
        stable_points = np.sum(stability_map)
        unstable_points = total_points - stable_points
        
        bifurcation_map = {
            'grid_shape': grid_shape,
            'stability_map': stability_map,
            'color_map': color_map,
            'eigenvalue_map': eigenvalue_map,
            'regime_map': regime_map,
            'statistics': {
                'total_points': total_points,
                'stable_points': int(stable_points),
                'unstable_points': int(unstable_points),
                'stability_fraction': stable_points / total_points,
                'eigenvalue_range': (np.min(eigenvalue_map), np.max(eigenvalue_map)),
                'color_range': (np.nanmin(color_map), np.nanmax(color_map))
            }
        }
        
        return bifurcation_map


def validate_eigenvalue_analysis(jacobian_builder: PerModeJacobianBuilder,
                               fixed_point_result: FixedPointResult) -> Dict[str, Any]:
    """Validate eigenvalue analysis with comprehensive tests.
    
    Args:
        jacobian_builder: PerModeJacobianBuilder instance
        fixed_point_result: Fixed point solution
        
    Returns:
        Validation results dictionary
    """
    print("Validating Eigenvalue Analysis...")
    
    # Initialize analyzer
    analyzer = EigenvalueAnalyzer(jacobian_builder)
    
    # Test 1: Basic stability analysis
    print("  Test 1: Basic stability analysis...")
    analysis_result = analyzer.analyze_stability(fixed_point_result, grid_size=10)  # Small grid for testing
    
    basic_analysis_valid = (
        analysis_result.total_modes == 100 and  # 10x10 grid
        analysis_result.max_real_eigenvalue is not None and
        analysis_result.winning_mode_indices is not None and
        analysis_result.stability_regime != StabilityRegime.UNKNOWN
    )
    
    print(f"    Basic analysis valid: {'✓' if basic_analysis_valid else '✗'}")
    print(f"    Total modes: {analysis_result.total_modes}")
    print(f"    Max eigenvalue: {analysis_result.max_real_eigenvalue:.6f}")
    print(f"    System stable: {'Yes' if analysis_result.is_stable else 'No'}")
    print(f"    Stability regime: {analysis_result.stability_regime.value}")
    print(f"    Winning mode: {analysis_result.winning_mode_indices} (radius: {analysis_result.winning_mode_radius:.3f})")
    
    # Test 2: Eigenvalue spectrum analysis
    print("  Test 2: Eigenvalue spectrum analysis...")
    spectrum_analysis = analyzer.compute_eigenvalue_spectrum(analysis_result)
    
    spectrum_valid = (
        'spectrum_data' in spectrum_analysis and
        'statistics' in spectrum_analysis and
        len(spectrum_analysis['spectrum_data']['mode_radii']) == analysis_result.total_modes
    )
    
    print(f"    Spectrum analysis valid: {'✓' if spectrum_valid else '✗'}")
    if spectrum_valid:
        stats = spectrum_analysis['statistics']
        print(f"    Eigenvalue range: [{stats['min_real_eigenvalue']:.6f}, {stats['max_real_eigenvalue']:.6f}]")
        print(f"    Mode radius range: [{stats['min_mode_radius']:.3f}, {stats['max_mode_radius']:.3f}]")
    
    # Test 3: Bifurcation point classification
    print("  Test 3: Bifurcation point classification...")
    classifier = BifurcationPointClassifier()
    classification = classifier.classify_point(analysis_result)
    
    classification_valid = (
        'is_stable' in classification and
        'stability_regime' in classification and
        'color_value' in classification and
        'max_eigenvalue' in classification
    )
    
    print(f"    Classification valid: {'✓' if classification_valid else '✗'}")
    if classification_valid:
        print(f"    Color value: {classification['color_value']}")
        print(f"    Physical wavenumber: {classification['physical_wavenumber']:.6f}")
    
    # Test 4: Cache functionality
    print("  Test 4: Cache functionality...")
    cache_stats_before = analyzer.get_cache_stats()
    
    # Run same analysis again - should use cache
    analyzer.analyze_stability(fixed_point_result, grid_size=10)
    cache_stats_after = analyzer.get_cache_stats()
    
    cache_working = (
        cache_stats_after['analysis_cache_size'] >= cache_stats_before['analysis_cache_size']
    )
    
    print(f"    Cache working: {'✓' if cache_working else '✗'}")
    print(f"    Analysis cache size: {cache_stats_after['analysis_cache_size']}")
    
    # Overall validation
    validation_results = {
        'basic_analysis_valid': basic_analysis_valid,
        'spectrum_valid': spectrum_valid,
        'classification_valid': classification_valid,
        'cache_working': cache_working,
        'analysis_result': analysis_result,
        'spectrum_analysis': spectrum_analysis,
        'classification': classification,
        'overall_success': all([
            basic_analysis_valid,
            spectrum_valid,
            classification_valid,
            cache_working
        ])
    }
    
    print(f"\nEigenvalue Analysis Validation {'PASSED' if validation_results['overall_success'] else 'FAILED'}")
    
    return validation_results
