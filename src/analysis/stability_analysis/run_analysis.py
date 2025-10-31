"""Run complete stability analysis pipeline."""

import time
import os
import pickle
from pathlib import Path

from src.analysis.stability_analysis.stability_analysis import StabilityAnalysis
from src.analysis.utils import save_with_version
from src.analysis.stability_analysis.visualizer import StabilityVisualizer  
from src.analysis.stability_analysis.config import ANALYSIS_PARAMS, DEVELOPMENTAL_STAGES, OUTPUT_DIR, REGIMES


def main():
    """Run complete stability analysis pipeline."""
    print("=" * 60)
    print("CORTICAL CIRCUIT STABILITY ANALYSIS")
    print("=" * 60)
    print(f"Duration: {ANALYSIS_PARAMS['duration']}s | Snapshots: {ANALYSIS_PARAMS['n_snapshots']} | "
          f"Patches: {ANALYSIS_PARAMS['layer_patch_size']}x{ANALYSIS_PARAMS['layer_patch_size']}")
    print(f"Stages: {', '.join(DEVELOPMENTAL_STAGES)}")
    print()
    
    start_time = time.time()
    
    try:
        # Initialize analysis
        print("Initializing stability analysis...")
        analyzer = StabilityAnalysis()
        
        # Run analysis for all conditions
        print("Running stability analysis...")
        results = analyzer.run_analysis()
        
        analysis_time = time.time() - start_time
        print(f"\nAnalysis completed in {analysis_time:.1f} seconds")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualizer = StabilityVisualizer()
        visualizer.generate_all_figures(results)
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.1f} seconds")
        
        # Save results for later use with version metadata
        print("Saving results...")
        results_file = os.path.join(OUTPUT_DIR, 'stability_analysis_results.pkl')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_with_version(results, results_file)
        print(f"Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_snapshots = 0
        for stage in DEVELOPMENTAL_STAGES:
            if stage in results:
                stage_snapshots = 0
                for regime in REGIMES:
                    if regime in results[stage]:
                        regime_snapshots = len(results[stage][regime])
                        stage_snapshots += regime_snapshots
                        print(f"  {stage} {regime}: {regime_snapshots} snapshots")
                total_snapshots += stage_snapshots
        
        print(f"\nTotal snapshots analyzed: {total_snapshots}")
        print(f"Output directory: src/analysis/analysis_plots/stability")
        print("\nStability analysis complete!")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 