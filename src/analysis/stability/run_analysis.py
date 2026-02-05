"""Run complete stability analysis pipeline.

This module provides the main analysis orchestrator and CLI entry point for
stability analysis, coordinating snapshot collection, Jacobian computation,
and visualization generation.
"""

import argparse
import time
from pathlib import Path

from src.analysis.common import DEVELOPMENTAL_STAGES
from src.analysis.common import make_run_metadata, save_with_version
from src.model.config import seed_random

from .config import ANALYSIS_PARAMS, OUTPUT_DIR, REGIMES
from .stability_analysis import StabilityAnalysis
from .visualizer import StabilityVisualizer


class StabilityPipeline:
    """Main orchestrator for stability analysis pipeline."""

    def __init__(self, config: dict | None = None):
        """Initialize stability analysis pipeline.

        Args:
            config: Configuration dictionary with analysis settings.
                   If None, uses default configuration.
        """
        self.config = config or self._default_config()
        self._seed_used = None  # Track the actual seed used for metadata

    def _default_config(self) -> dict:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "stages": DEVELOPMENTAL_STAGES,
            "duration": ANALYSIS_PARAMS["duration"],
            "n_snapshots": ANALYSIS_PARAMS["n_snapshots"],
            "layer_patch_size": ANALYSIS_PARAMS["layer_patch_size"],
            "output_dir": OUTPUT_DIR,
        }

    def run(self) -> dict:
        """Run complete stability analysis pipeline.

        Returns:
            Dictionary with analysis results organized by stage and regime
        """
        # Seed RNG once at the start of the pipeline for reproducibility
        self._seed_used = seed_random()

        print("\n" + "=" * 70)
        print("STABILITY ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"Random seed: {self._seed_used}")
        print(
            f"Duration: {self.config['duration']}s | "
            f"Snapshots: {self.config['n_snapshots']} | "
            f"Patches: {self.config['layer_patch_size']}x{self.config['layer_patch_size']}"
        )
        print(f"Stages: {', '.join(self.config['stages'])}")
        print(f"Output: {self.config['output_dir']}")

        start_time = time.time()

        # Initialize core analysis
        print("\nInitializing stability analysis...")
        analyzer = StabilityAnalysis()

        # Run analysis for all conditions
        print("Running stability analysis...")
        results = analyzer.run_analysis()

        elapsed = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed:.1f} seconds")

        return results

    def run_global(self) -> dict:
        """Run whole-network (global) stability analysis only.

        Returns:
            Dictionary with results[stage][regime][snapshot_idx]["global"].
        """
        self._seed_used = seed_random()

        print("\n" + "=" * 70)
        print("GLOBAL STABILITY ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"Random seed: {self._seed_used}")
        print(
            f"Duration: {self.config['duration']}s | "
            f"Snapshots: {self.config['n_snapshots']} | "
            f"Whole network (full grid)"
        )
        print(f"Stages: {', '.join(self.config['stages'])}")
        print(f"Output: {self.config['output_dir']}")

        start_time = time.time()

        print("\nInitializing stability analysis...")
        analyzer = StabilityAnalysis()

        print("Running global stability analysis...")
        results = analyzer.run_global_analysis()

        elapsed = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed:.1f} seconds")

        return results

    def generate_global_visualizations(self, results: dict) -> None:
        """Generate visualization figures from global analysis results.

        Args:
            results: Analysis results with global key per snapshot.
        """
        print("\nGenerating global visualizations...")
        visualizer = StabilityVisualizer()
        visualizer.create_global_effectiveness_plot(results)
        visualizer.create_global_celltype_effectiveness_plot(results)
        print("Global visualization complete!")

    def save_results(self, results: dict) -> Path:
        """Save analysis results to disk.

        Args:
            results: Analysis results dictionary

        Returns:
            Path to saved results file
        """
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = (
            "stability_analysis_global_results.pkl"
            if self.config.get("global")
            else "stability_analysis_results.pkl"
        )
        results_file = output_dir / base_name
        params = {
            "duration": self.config["duration"],
            "n_snapshots": self.config["n_snapshots"],
        }
        if not self.config.get("global"):
            params["layer_patch_size"] = self.config["layer_patch_size"]
        else:
            params["mode"] = "global"
        metadata = make_run_metadata(
            seed=self._seed_used,
            stages=self.config["stages"],
            params=params,
        )
        save_with_version(results, str(results_file), metadata=metadata)
        print(f"Results saved to: {results_file}")

        return results_file

    def generate_visualizations(self, results: dict) -> None:
        """Generate visualization figures from results.

        Args:
            results: Analysis results dictionary
        """
        print("\nGenerating visualizations...")
        visualizer = StabilityVisualizer()
        visualizer.generate_all_figures(results)
        print("Visualization complete!")

    def print_summary(self, results: dict) -> None:
        """Print analysis summary.

        Args:
            results: Analysis results dictionary
        """
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)

        total_snapshots = 0
        for stage in self.config["stages"]:
            if stage in results:
                stage_snapshots = 0
                for regime in REGIMES:
                    if regime in results[stage]:
                        regime_snapshots = len(results[stage][regime])
                        stage_snapshots += regime_snapshots
                        print(f"  {stage} {regime}: {regime_snapshots} snapshots")
                total_snapshots += stage_snapshots

        print(f"\nTotal snapshots analyzed: {total_snapshots}")
        print(f"Output directory: {self.config['output_dir']}")


def main():
    """CLI entry point for stability analysis."""
    parser = argparse.ArgumentParser(
        description="Stability analysis for cortical circuit development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with default parameters
  python -m src.analysis.stability

  # Run for specific stages
  python -m src.analysis.stability --stages P0 P5

  # Skip visualization generation
  python -m src.analysis.stability --no-viz

  # Whole-network (global) stability analysis only
  python -m src.analysis.stability --global
        """,
    )

    parser.add_argument(
        "--global",
        dest="global_analysis",
        action="store_true",
        help="Run whole-network stability analysis (no patch-based analysis)",
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=DEVELOPMENTAL_STAGES,
        default=list(DEVELOPMENTAL_STAGES),
        help="Developmental stages to analyze (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})",
    )

    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    args = parser.parse_args()

    # Build configuration
    config = {
        "stages": args.stages,
        "duration": ANALYSIS_PARAMS["duration"],
        "n_snapshots": ANALYSIS_PARAMS["n_snapshots"],
        "layer_patch_size": ANALYSIS_PARAMS["layer_patch_size"],
        "output_dir": args.output_dir,
        "global": getattr(args, "global_analysis", False),
    }

    # Initialize pipeline
    pipeline = StabilityPipeline(config)

    # Run analysis
    try:
        start_time = time.time()

        if config.get("global"):
            results = pipeline.run_global()
            pipeline.save_results(results)
            if not args.no_viz:
                pipeline.generate_global_visualizations(results)
        else:
            results = pipeline.run()
            pipeline.save_results(results)
            if not args.no_viz:
                pipeline.generate_visualizations(results)

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.1f} seconds")

        pipeline.print_summary(results)

        print("\n" + "=" * 70)
        print("STABILITY ANALYSIS COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        print("\nERROR: Analysis failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
