"""Main analysis runner for bifurcation analysis.

This module provides the main entry point for running complete bifurcation
analyses with comprehensive testing, validation, and visualization.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import time
from pathlib import Path
import argparse

from .parameter_sweeper import BifurcationAnalyzer, BifurcationAnalysisResult
from .validation import validate_bifurcation_results, ValidationReport
from .visualizer import create_publication_plots
from .config import PARAMETER_RANGES, print_config_summary


class IntegrationTester:
    """Comprehensive integration testing for the bifurcation analysis pipeline."""
    
    def __init__(self):
        """Initialize integration tester."""
        self.test_results = {}
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test of the entire pipeline.
        
        Returns:
            Dictionary with test results
        """
        print("Running Comprehensive Integration Test")
        print("=" * 50)
        
        test_results = {
            'component_tests': {},
            'pipeline_tests': {},
            'performance_tests': {},
            'overall_success': False
        }
        
        # Test 1: Individual component validation
        print("Phase 1: Component Validation")
        print("-" * 30)
        component_results = self._test_individual_components()
        test_results['component_tests'] = component_results
        
        # Test 2: Pipeline integration
        print("\nPhase 2: Pipeline Integration")
        print("-" * 30)
        pipeline_results = self._test_pipeline_integration()
        test_results['pipeline_tests'] = pipeline_results
        
        # Test 3: Performance and scalability
        print("\nPhase 3: Performance Testing")
        print("-" * 30)
        performance_results = self._test_performance()
        test_results['performance_tests'] = performance_results
        
        # Overall assessment
        overall_success = self._assess_overall_success(test_results)
        test_results['overall_success'] = overall_success
        
        print("\n" + "=" * 50)
        print(f"Integration Test {'PASSED' if overall_success else 'FAILED'}")
        
        return test_results
    
    def _test_individual_components(self) -> Dict[str, Any]:
        """Test individual components of the analysis pipeline."""
        component_results = {}
        
        # Test imports
        print("  Testing module imports...")
        try:
            from . import (
                validate_fourier_analysis_setup,
                validate_fixed_point_solver,
                validate_jacobian_builder,
                validate_eigenvalue_analysis,
                validate_parameter_sweeper
            )
            from src.main import CorticalSimulation
            
            component_results['imports'] = {'success': True, 'error': None}
            print("    ✓ All imports successful")
            
        except Exception as e:
            component_results['imports'] = {'success': False, 'error': str(e)}
            print(f"    ✗ Import failed: {e}")
            return component_results
        
        # Test simulation initialization
        print("  Testing simulation initialization...")
        try:
            simulation = CorticalSimulation()
            time_constants = simulation.get_time_constants()
            gains = simulation.get_gains()
            
            component_results['simulation'] = {
                'success': True,
                'time_constants': time_constants,
                'gains': gains
            }
            print("    ✓ Simulation initialized successfully")
            
        except Exception as e:
            component_results['simulation'] = {'success': False, 'error': str(e)}
            print(f"    ✗ Simulation initialization failed: {e}")
            return component_results
        
        # Test Fourier analysis
        print("  Testing Fourier analysis...")
        try:
            fourier_validation = validate_fourier_analysis_setup(simulation.circuit.connectivity)
            component_results['fourier'] = fourier_validation
            print(f"    {'✓' if fourier_validation['overall_success'] else '✗'} Fourier analysis")
            
        except Exception as e:
            component_results['fourier'] = {'overall_success': False, 'error': str(e)}
            print(f"    ✗ Fourier analysis failed: {e}")
        
        # Test fixed point solver
        print("  Testing fixed point solver...")
        try:
            fp_validation = validate_fixed_point_solver(
                simulation.circuit.connectivity, time_constants, gains
            )
            component_results['fixed_point'] = fp_validation
            print(f"    {'✓' if fp_validation['overall_success'] else '✗'} Fixed point solver")
            
        except Exception as e:
            component_results['fixed_point'] = {'overall_success': False, 'error': str(e)}
            print(f"    ✗ Fixed point solver failed: {e}")
        
        # Test Jacobian builder
        print("  Testing Jacobian builder...")
        try:
            if component_results['fixed_point']['overall_success']:
                fp_result = component_results['fixed_point']['solver_result']
                jacobian_validation = validate_jacobian_builder(
                    simulation.circuit.connectivity, time_constants, gains, fp_result
                )
                component_results['jacobian'] = jacobian_validation
                print(f"    {'✓' if jacobian_validation['overall_success'] else '✗'} Jacobian builder")
            else:
                component_results['jacobian'] = {'overall_success': False, 'error': 'Fixed point failed'}
                print("    ✗ Jacobian builder (skipped due to fixed point failure)")
                
        except Exception as e:
            component_results['jacobian'] = {'overall_success': False, 'error': str(e)}
            print(f"    ✗ Jacobian builder failed: {e}")
        
        # Test eigenvalue analysis
        print("  Testing eigenvalue analysis...")
        try:
            if (component_results['fixed_point']['overall_success'] and 
                component_results['jacobian']['overall_success']):
                
                from .jacobian_builder import PerModeJacobianBuilder
                jacobian_builder = PerModeJacobianBuilder(
                    simulation.circuit.connectivity, time_constants, gains
                )
                fp_result = component_results['fixed_point']['solver_result']
                
                eigenvalue_validation = validate_eigenvalue_analysis(jacobian_builder, fp_result)
                component_results['eigenvalue'] = eigenvalue_validation
                print(f"    {'✓' if eigenvalue_validation['overall_success'] else '✗'} Eigenvalue analysis")
            else:
                component_results['eigenvalue'] = {'overall_success': False, 'error': 'Prerequisites failed'}
                print("    ✗ Eigenvalue analysis (skipped due to prerequisite failure)")
                
        except Exception as e:
            component_results['eigenvalue'] = {'overall_success': False, 'error': str(e)}
            print(f"    ✗ Eigenvalue analysis failed: {e}")
        
        # Test parameter sweeper
        print("  Testing parameter sweeper...")
        try:
            sweeper_validation = validate_parameter_sweeper('pv_analysis', (2, 2))
            component_results['parameter_sweeper'] = sweeper_validation
            print(f"    {'✓' if sweeper_validation['overall_success'] else '✗'} Parameter sweeper")
            
        except Exception as e:
            component_results['parameter_sweeper'] = {'overall_success': False, 'error': str(e)}
            print(f"    ✗ Parameter sweeper failed: {e}")
        
        return component_results
    
    def _test_pipeline_integration(self) -> Dict[str, Any]:
        """Test complete pipeline integration."""
        pipeline_results = {}
        
        # Test complete analysis pipeline
        print("  Testing complete analysis pipeline...")
        try:
            analyzer = BifurcationAnalyzer()
            
            # Run small test analysis
            result = analyzer.run_analysis(
                'pv_analysis',
                grid_resolution=(3, 3),
                fourier_grid_size=5,
                save_results=False  # Don't save during testing
            )
            
            pipeline_success = (
                result.successful_points == result.total_points and
                result.total_points == 9 and
                hasattr(result, 'stability_map') and
                hasattr(result, 'color_map')
            )
            
            pipeline_results['complete_analysis'] = {
                'success': pipeline_success,
                'total_points': result.total_points,
                'successful_points': result.successful_points,
                'stable_points': result.stable_points,
                'analysis_time': result.analysis_time
            }
            
            print(f"    {'✓' if pipeline_success else '✗'} Complete analysis pipeline")
            print(f"      Points: {result.successful_points}/{result.total_points}")
            print(f"      Time: {result.analysis_time:.1f} seconds")
            
        except Exception as e:
            pipeline_results['complete_analysis'] = {'success': False, 'error': str(e)}
            print(f"    ✗ Complete analysis pipeline failed: {e}")
            return pipeline_results
        
        # Test validation integration
        print("  Testing validation integration...")
        try:
            validation_report = validate_bifurcation_results(result, save_report=False)
            
            validation_success = (
                validation_report.overall_valid and
                validation_report.validation_score > 0.5
            )
            
            pipeline_results['validation_integration'] = {
                'success': validation_success,
                'validation_score': validation_report.validation_score,
                'convergence_rate': validation_report.convergence_rate
            }
            
            print(f"    {'✓' if validation_success else '✗'} Validation integration")
            print(f"      Score: {validation_report.validation_score:.3f}")
            
        except Exception as e:
            pipeline_results['validation_integration'] = {'success': False, 'error': str(e)}
            print(f"    ✗ Validation integration failed: {e}")
        
        # Test visualization integration
        print("  Testing visualization integration...")
        try:
            saved_plots = create_publication_plots(
                result,
                validation_report=validation_report,
                output_dir='outputs/bifurcation/integration_test',
                show_plots=False
            )
            
            visualization_success = len(saved_plots) > 0
            
            pipeline_results['visualization_integration'] = {
                'success': visualization_success,
                'plots_created': len(saved_plots),
                'plot_types': list(saved_plots.keys())
            }
            
            print(f"    {'✓' if visualization_success else '✗'} Visualization integration")
            print(f"      Plots: {len(saved_plots)}")
            
        except Exception as e:
            pipeline_results['visualization_integration'] = {'success': False, 'error': str(e)}
            print(f"    ✗ Visualization integration failed: {e}")
        
        return pipeline_results
    
    def _test_performance(self) -> Dict[str, Any]:
        """Test performance and scalability."""
        performance_results = {}
        
        # Test different grid sizes
        print("  Testing scalability...")
        grid_sizes = [(2, 2), (3, 3), (5, 5)]
        
        for grid_size in grid_sizes:
            try:
                start_time = time.time()
                
                analyzer = BifurcationAnalyzer()
                result = analyzer.run_analysis(
                    'pv_analysis',
                    grid_resolution=grid_size,
                    fourier_grid_size=5,
                    save_results=False
                )
                
                analysis_time = time.time() - start_time
                time_per_point = analysis_time / result.total_points
                
                performance_results[f'grid_{grid_size[0]}x{grid_size[1]}'] = {
                    'success': True,
                    'total_time': analysis_time,
                    'time_per_point': time_per_point,
                    'total_points': result.total_points,
                    'successful_points': result.successful_points
                }
                
                print(f"    ✓ {grid_size[0]}×{grid_size[1]} grid: {analysis_time:.1f}s ({time_per_point:.2f}s/point)")
                
            except Exception as e:
                performance_results[f'grid_{grid_size[0]}x{grid_size[1]}'] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"    ✗ {grid_size[0]}×{grid_size[1]} grid failed: {e}")
        
        # Test memory usage (approximate)
        print("  Testing memory efficiency...")
        try:
            # This is a basic test - in practice we'd use memory profiling tools
            analyzer = BifurcationAnalyzer()
            
            # Clear caches before testing
            if hasattr(analyzer, 'jacobian_builder') and analyzer.jacobian_builder:
                analyzer.jacobian_builder.clear_cache()
            
            result = analyzer.run_analysis(
                'pv_analysis',
                grid_resolution=(3, 3),
                fourier_grid_size=5,
                save_results=False
            )
            
            performance_results['memory_test'] = {
                'success': True,
                'note': 'Basic memory test completed - no detailed profiling'
            }
            
            print("    ✓ Memory efficiency test completed")
            
        except Exception as e:
            performance_results['memory_test'] = {'success': False, 'error': str(e)}
            print(f"    ✗ Memory test failed: {e}")
        
        return performance_results
    
    def _assess_overall_success(self, test_results: Dict[str, Any]) -> bool:
        """Assess overall success of integration tests."""
        # Check component tests
        component_success = all(
            result.get('overall_success', result.get('success', False))
            for result in test_results['component_tests'].values()
            if isinstance(result, dict)
        )
        
        # Check pipeline tests
        pipeline_success = all(
            result.get('success', False)
            for result in test_results['pipeline_tests'].values()
            if isinstance(result, dict)
        )
        
        # Check performance tests
        performance_success = all(
            result.get('success', False)
            for result in test_results['performance_tests'].values()
            if isinstance(result, dict)
        )
        
        return component_success and pipeline_success and performance_success


def run_production_analysis(analysis_type: str = 'pv_analysis',
                          grid_resolution: Tuple[int, int] = (25, 25),
                          fourier_grid_size: int = 20,
                          output_dir: Optional[str] = None,
                          validate_results: bool = True,
                          create_visualizations: bool = True) -> Dict[str, Any]:
    """Run production-quality bifurcation analysis.
    
    Args:
        analysis_type: Type of analysis to run
        grid_resolution: Parameter grid resolution
        fourier_grid_size: Fourier mode grid size
        output_dir: Output directory for results
        validate_results: Whether to validate results
        create_visualizations: Whether to create visualizations
        
    Returns:
        Dictionary with analysis results and status
    """
    print(f"Running Production Analysis: {analysis_type}")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BifurcationAnalyzer()
    
    # Run analysis
    print(f"Running {grid_resolution[0]}×{grid_resolution[1]} parameter sweep...")
    start_time = time.time()
    
    try:
        result = analyzer.run_analysis(
            analysis_type=analysis_type,
            grid_resolution=grid_resolution,
            fourier_grid_size=fourier_grid_size,
            save_results=True,
            output_dir=output_dir
        )
        
        analysis_time = time.time() - start_time
        analysis_success = True
        
        print(f"✓ Analysis completed in {analysis_time:.1f} seconds")
        print(f"  Success rate: {result.successful_points}/{result.total_points} ({result.successful_points/result.total_points:.1%})")
        print(f"  Stable points: {result.stable_points}/{result.successful_points} ({result.stable_points/result.successful_points:.1%})")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return {
            'analysis_success': False,
            'error': str(e)
        }
    
    # Validate results
    validation_report = None
    if validate_results:
        print("\nValidating results...")
        try:
            validation_report = validate_bifurcation_results(result, save_report=True, output_dir=output_dir)
            
            if validation_report.overall_valid:
                print("✓ Validation PASSED")
            else:
                print("✗ Validation FAILED")
                print(f"  Validation score: {validation_report.validation_score:.3f}")
                
        except Exception as e:
            print(f"✗ Validation failed: {e}")
    
    # Create visualizations
    saved_plots = {}
    if create_visualizations:
        print("\nCreating visualizations...")
        try:
            saved_plots = create_publication_plots(
                result,
                validation_report=validation_report,
                output_dir=output_dir,
                show_plots=False
            )
            
            print(f"✓ Created {len(saved_plots)} visualizations")
            
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
    
    return {
        'analysis_success': analysis_success,
        'analysis_result': result,
        'validation_report': validation_report,
        'saved_plots': saved_plots,
        'analysis_time': analysis_time,
        'summary': {
            'total_points': result.total_points,
            'successful_points': result.successful_points,
            'stable_points': result.stable_points,
            'unstable_points': result.unstable_points,
            'validation_score': validation_report.validation_score if validation_report else None,
            'plots_created': len(saved_plots)
        }
    }


def run_comparison_analysis(grid_resolution: Tuple[int, int] = (15, 15),
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run comparative analysis of PV vs SST bifurcation diagrams.
    
    Args:
        grid_resolution: Parameter grid resolution
        output_dir: Output directory for results
        
    Returns:
        Dictionary with comparison results
    """
    print("Running Comparative Analysis: PV vs SST")
    print("=" * 50)
    
    analyzer = BifurcationAnalyzer()
    results = {}
    
    # Run PV analysis
    print("Running PV analysis...")
    try:
        pv_result = analyzer.run_analysis(
            'pv_analysis',
            grid_resolution=grid_resolution,
            save_results=True,
            output_dir=output_dir
        )
        results['pv'] = pv_result
        print(f"✓ PV analysis: {pv_result.successful_points}/{pv_result.total_points} points")
        
    except Exception as e:
        print(f"✗ PV analysis failed: {e}")
        return {'success': False, 'error': f'PV analysis failed: {e}'}
    
    # Run SST analysis
    print("Running SST analysis...")
    try:
        sst_result = analyzer.run_analysis(
            'sst_analysis',
            grid_resolution=grid_resolution,
            save_results=True,
            output_dir=output_dir
        )
        results['sst'] = sst_result
        print(f"✓ SST analysis: {sst_result.successful_points}/{sst_result.total_points} points")
        
    except Exception as e:
        print(f"✗ SST analysis failed: {e}")
        return {'success': False, 'error': f'SST analysis failed: {e}'}
    
    # Create comparison visualizations
    print("Creating comparison visualizations...")
    try:
        from .visualizer import plot_multiple_analyses
        
        comparison_plots = plot_multiple_analyses(
            [pv_result, sst_result],
            output_dir=output_dir,
            show_plots=False
        )
        
        print(f"✓ Created {len(comparison_plots)} comparison plots")
        
    except Exception as e:
        print(f"✗ Comparison visualization failed: {e}")
        comparison_plots = {}
    
    return {
        'success': True,
        'pv_result': pv_result,
        'sst_result': sst_result,
        'comparison_plots': comparison_plots,
        'summary': {
            'pv_success_rate': pv_result.successful_points / pv_result.total_points,
            'sst_success_rate': sst_result.successful_points / sst_result.total_points,
            'pv_stability_rate': pv_result.stable_points / pv_result.successful_points,
            'sst_stability_rate': sst_result.stable_points / sst_result.successful_points
        }
    }


def main():
    """Main function for running analyses."""
    parser = argparse.ArgumentParser(description='Run bifurcation analysis')
    parser.add_argument('--test', action='store_true', help='Run integration tests')
    parser.add_argument('--production', action='store_true', help='Run production analysis')
    parser.add_argument('--comparison', action='store_true', help='Run PV vs SST comparison')
    parser.add_argument('--resolution', type=int, default=25, help='Grid resolution')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if args.test:
        # Run integration tests
        tester = IntegrationTester()
        test_results = tester.run_comprehensive_test()
        
        if test_results['overall_success']:
            print("\n🎉 All integration tests PASSED!")
            return 0
        else:
            print("\n❌ Integration tests FAILED!")
            return 1
    
    elif args.production:
        # Run production analysis
        result = run_production_analysis(
            grid_resolution=(args.resolution, args.resolution),
            output_dir=args.output_dir
        )
        
        if result['analysis_success']:
            print(f"\n🎉 Production analysis completed successfully!")
            summary = result['summary']
            print(f"  Success rate: {summary['successful_points']}/{summary['total_points']}")
            print(f"  Stability rate: {summary['stable_points']}/{summary['successful_points']}")
            return 0
        else:
            print(f"\n❌ Production analysis failed!")
            return 1
    
    elif args.comparison:
        # Run comparison analysis
        result = run_comparison_analysis(
            grid_resolution=(args.resolution, args.resolution),
            output_dir=args.output_dir
        )
        
        if result['success']:
            print(f"\n🎉 Comparison analysis completed successfully!")
            summary = result['summary']
            print(f"  PV success: {summary['pv_success_rate']:.1%}")
            print(f"  SST success: {summary['sst_success_rate']:.1%}")
            return 0
        else:
            print(f"\n❌ Comparison analysis failed!")
            return 1
    
    else:
        # Default: print configuration and run basic test
        print_config_summary()
        print("\nUse --test, --production, or --comparison to run analysis")
        print("Example: python -m analysis.bifurcation_analysis.run_analysis --test")
        return 0


if __name__ == '__main__':
    exit(main())
