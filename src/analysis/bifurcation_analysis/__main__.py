"""Command-line interface for bifurcation analysis.

This module provides a command-line interface for running the bifurcation analysis
with different parameter configurations and analysis types.

Usage:
    python -m analysis.bifurcation_analysis --analysis pv
    python -m analysis.bifurcation_analysis --analysis sst  
    python -m analysis.bifurcation_analysis --analysis both
"""

import argparse
from pathlib import Path

from .config import print_config_summary


def main():
    """Main entry point for bifurcation analysis."""
    parser = argparse.ArgumentParser(
        description='Run bifurcation analysis for cortical circuit model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m analysis.bifurcation_analysis --analysis pv
  python -m analysis.bifurcation_analysis --analysis sst --resolution 25
  python -m analysis.bifurcation_analysis --analysis both --validate
        """
    )
    
    parser.add_argument(
        '--analysis', 
        choices=['pv', 'sst', 'both', 'strength_width'],
        default='pv',
        help='Type of bifurcation analysis to run (default: pv)'
    )
    
    parser.add_argument(
        '--resolution',
        type=int,
        default=50,
        help='Grid resolution for parameter sweep (default: 50)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation checks during analysis'
    )
    
    parser.add_argument(
        '--validate-results',
        type=str,
        help='Validate existing results file (provide path to .pkl file)'
    )
    
    parser.add_argument(
        '--visualize-results',
        type=str,
        help='Create visualizations for existing results file (provide path to .pkl file)'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively (in addition to saving)'
    )
    
    parser.add_argument(
        '--config-summary',
        action='store_true',
        help='Print configuration summary and exit'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: outputs/bifurcation/)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Print configuration summary if requested
    if args.config_summary:
        print_config_summary()
        return
    
    # Print analysis information
    print("Bifurcation Analysis")
    print("===================")
    print(f"Analysis type: {args.analysis}")
    print(f"Grid resolution: {args.resolution}×{args.resolution}")
    print(f"Validation: {'Enabled' if args.validate else 'Disabled'}")
    print(f"Verbose output: {'Enabled' if args.verbose else 'Disabled'}")
    
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    
    print()
    
    # Print configuration summary
    print_config_summary()
    print()
    
    # Check for results validation request
    if args.validate_results:
        try:
            from .validation import load_and_validate_results
            print(f"Validating results file: {args.validate_results}")
            validation_report = load_and_validate_results(args.validate_results)
            
            if validation_report.overall_valid:
                print("✓ Results validation PASSED")
                return
            else:
                print("✗ Results validation FAILED")
                return
                
        except Exception as e:
            print(f"Results validation failed: {e}")
            return
    
    # Check for visualization request
    if args.visualize_results:
        try:
            from .visualizer import load_and_visualize_results
            print(f"Creating visualizations for: {args.visualize_results}")
            saved_plots = load_and_visualize_results(
                args.visualize_results, 
                output_dir=args.output_dir,
                show_plots=args.show_plots
            )
            
            print(f"✓ Created {len(saved_plots)} visualizations")
            for plot_type, path in saved_plots.items():
                print(f"  {plot_type}: {path}")
            return
            
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Import and run the actual analysis
    try:
        from .parameter_sweeper import BifurcationAnalyzer, validate_parameter_sweeper
        from .validation import validate_bifurcation_results
        
        if args.analysis in ['pv', 'sst', 'both']:
            # Run full bifurcation analysis
            analyzer = BifurcationAnalyzer()
            
            if args.analysis == 'both':
                # Run both PV and SST analyses
                print("Running PV analysis...")
                pv_result = analyzer.run_analysis(
                    'pv_analysis', 
                    grid_resolution=(args.resolution, args.resolution),
                    save_results=True,
                    output_dir=args.output_dir
                )
                
                print("\nRunning SST analysis...")
                sst_result = analyzer.run_analysis(
                    'sst_analysis',
                    grid_resolution=(args.resolution, args.resolution), 
                    save_results=True,
                    output_dir=args.output_dir
                )
                
                print(f"\nCompleted both analyses:")
                print(f"  PV: {pv_result.successful_points}/{pv_result.total_points} successful points")
                print(f"  SST: {sst_result.successful_points}/{sst_result.total_points} successful points")
                
            else:
                # Run single analysis
                analysis_type = 'pv_analysis' if args.analysis == 'pv' else 'sst_analysis'
                result = analyzer.run_analysis(
                    analysis_type,
                    grid_resolution=(args.resolution, args.resolution),
                    save_results=True,
                    output_dir=args.output_dir
                )
                
                print(f"\nCompleted {args.analysis} analysis:")
                print(f"  {result.successful_points}/{result.total_points} successful points")
                print(f"  {result.stable_points}/{result.successful_points} stable points")
                
                # Validate and visualize results if requested
                validation_report = None
                if args.validate:
                    print("\nValidating analysis results...")
                    validation_report = validate_bifurcation_results(result)
                    if validation_report.overall_valid:
                        print("✓ Results validation PASSED")
                    else:
                        print("✗ Results validation FAILED")
                
                # Create visualizations
                print("\nCreating visualizations...")
                from .visualizer import create_publication_plots
                saved_plots = create_publication_plots(
                    result, 
                    validation_report=validation_report,
                    output_dir=args.output_dir,
                    show_plots=args.show_plots
                )
                print(f"✓ Created {len(saved_plots)} visualizations")
                
        elif args.analysis == 'strength_width':
            # Run strength vs width analysis
            analyzer = BifurcationAnalyzer()
            result = analyzer.run_analysis(
                'strength_width_analysis',
                grid_resolution=(args.resolution, args.resolution),
                save_results=True,
                output_dir=args.output_dir
            )
            
            print(f"\nCompleted strength-width analysis:")
            print(f"  {result.successful_points}/{result.total_points} successful points")
            
        # Run validation if requested
        if args.validate:
            print("\nRunning validation...")
            validation_result = validate_parameter_sweeper('pv_analysis', (3, 3))
            if validation_result['overall_success']:
                print("✓ Validation passed")
            else:
                print("✗ Validation failed")
                
    except ImportError as e:
        print(f"Error importing analysis modules: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
