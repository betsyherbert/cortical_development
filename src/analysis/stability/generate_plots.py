"""Generate plots from existing stability analysis results."""

import os
import sys
import pickle
import argparse
from pathlib import Path

from src.analysis.stability.visualizer import StabilityVisualizer
from src.analysis.stability.config import OUTPUT_DIR
from src.analysis.common import load_with_version


def load_results(results_file=None):
    """Load pre-computed stability analysis results."""
    if results_file is None:
        results_file = os.path.join(OUTPUT_DIR, 'stability_analysis_results.pkl')
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run the stability analysis first:")
        print("  python -m src.analysis.stability")
        sys.exit(1)
    
    print(f"Loading results from: {results_file}")
    
    # Load with version checking
    versioned_data = load_with_version(results_file)
    results = versioned_data['data']
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Generate plots from existing stability analysis results')
    parser.add_argument('--results', type=str, 
                       help='Path to results file (default: auto-detect)')
    parser.add_argument('--type', choices=['all', 'layer_wise', 'column_wise', 'effectiveness', 
                                          'phase_diagram', 'regime_percentages'],
                       default='all', help='Type of plots to generate (default: all)')
    parser.add_argument('--regime', choices=['driven', 'idle'], 
                       help='Regime for snapshot-based figures (required for layer_wise/column_wise)')
    parser.add_argument('--snapshot', type=int, default=0,
                       help='Snapshot index (default: 0)')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    
    # Initialize visualizer
    visualizer = StabilityVisualizer()
    
    # Generate requested plots
    if args.type == 'all':
        print("Generating all figures...")
        visualizer.generate_all_figures(results)
    
    elif args.type in ['layer_wise', 'column_wise']:
        if not args.regime:
            parser.error(f"--regime required for {args.type} figures")
        
        if args.type == 'layer_wise':
            visualizer.create_layer_wise_figure(results, args.regime, args.snapshot)
        else:
            visualizer.create_column_wise_figure(results, args.regime, args.snapshot)
    
    elif args.type == 'effectiveness':
        visualizer.create_inhibition_effectiveness_plot(results)
        visualizer.create_layer_effectiveness_plot(results)
        visualizer.create_celltype_effectiveness_plot(results)
        visualizer.create_layer_specific_heatmaps(results)
    
    elif args.type == 'phase_diagram':
        visualizer.create_stability_phase_diagrams(results)
    
    elif args.type == 'regime_percentages':
        visualizer.create_regime_percentage_plots(results)
    
    print("Plot generation complete!")
    print(f"Summary figures saved to: {visualizer.summary_dir}")
    print(f"Snapshot figures saved to: {visualizer.snapshots_dir}")


if __name__ == "__main__":
    main()
