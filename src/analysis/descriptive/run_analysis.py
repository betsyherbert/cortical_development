"""Run descriptive analysis pipeline.

This module provides the main analysis orchestrator and CLI entry point for
descriptive activity analysis, coordinating data collection, metric computation,
and visualization generation.
"""

import argparse
import time
from pathlib import Path
from typing import Any

from src.analysis.common import make_run_metadata, save_with_version

from .activity_analysis import DescriptiveAnalysis
from .config import ANALYSIS_PARAMS, DEVELOPMENTAL_STAGES, OUTPUT_DIR
from .visualizer import ActivityVisualizer


class DescriptivePipeline:
    """Main orchestrator for descriptive activity analysis pipeline."""

    def __init__(self, config: dict | None = None):
        """Initialize descriptive analysis pipeline.

        Args:
            config: Configuration dictionary with analysis settings.
                   If None, uses default configuration.
        """
        self.config = config or self._default_config()

    def _default_config(self) -> dict:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "stages": DEVELOPMENTAL_STAGES,
            "warmup_duration": ANALYSIS_PARAMS["warmup_duration"],
            "simulation_duration": ANALYSIS_PARAMS["simulation_duration"],
            "sampling_interval": ANALYSIS_PARAMS["sampling_interval"],
            "output_dir": OUTPUT_DIR,
        }

    def run(self) -> dict[str, Any]:
        """Run complete descriptive analysis pipeline.

        Returns:
            Dictionary with analysis results organized by stage
        """
        print("\n" + "=" * 70)
        print("DESCRIPTIVE ACTIVITY ANALYSIS PIPELINE")
        print("=" * 70)
        print(
            f"Warmup: {self.config['warmup_duration']}s | "
            f"Duration: {self.config['simulation_duration']}s | "
            f"Sampling: {self.config['sampling_interval']}ms"
        )
        print(f"Stages: {', '.join(self.config['stages'])}")
        print(f"Output: {self.config['output_dir']}")

        start_time = time.time()

        # Initialize core analysis
        analyzer = DescriptiveAnalysis()

        # Run analysis for all stages
        results = analyzer.run_analysis()

        elapsed = time.time() - start_time
        print(f"\nData analysis completed in {elapsed:.1f} seconds")

        return results

    def save_results(self, results: dict) -> Path:
        """Save analysis results to disk.

        Args:
            results: Analysis results dictionary

        Returns:
            Path to saved results file
        """
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "descriptive_analysis_results.pkl"
        metadata = make_run_metadata(
            stages=self.config["stages"],
            params={
                "warmup_duration": self.config["warmup_duration"],
                "simulation_duration": self.config["simulation_duration"],
                "sampling_interval": self.config["sampling_interval"],
            },
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
        visualizer = ActivityVisualizer()
        visualizer.generate_all_plots(results)
        print("Visualization complete!")

    def print_summary(self, results: dict) -> None:
        """Print analysis summary.

        Args:
            results: Analysis results dictionary
        """
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)

        for stage in self.config["stages"]:
            if stage in results:
                n_timepoints = len(results[stage].get("time", []))
                correlations = results[stage].get("correlations", {})
                total_corr = correlations.get("total", float("nan"))
                sync_events = results[stage].get("synchronous_events", {})
                total_sync = sync_events.get("total", 0)

                print(
                    f"  {stage}: {n_timepoints} timepoints, "
                    f"avg corr={total_corr:.3f}, sync events={total_sync}"
                )

        print(f"\nOutput directory: {self.config['output_dir']}")


def main():
    """CLI entry point for descriptive analysis."""
    parser = argparse.ArgumentParser(
        description="Descriptive activity analysis for cortical circuit development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with default parameters
  python -m src.analysis.descriptive

  # Run for specific stages
  python -m src.analysis.descriptive --stages P0 P5

  # Skip visualization generation
  python -m src.analysis.descriptive --no-viz
        """,
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["P0", "P5", "P10", "P15"],
        default=["P0", "P5", "P10", "P15"],
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
        "warmup_duration": ANALYSIS_PARAMS["warmup_duration"],
        "simulation_duration": ANALYSIS_PARAMS["simulation_duration"],
        "sampling_interval": ANALYSIS_PARAMS["sampling_interval"],
        "output_dir": args.output_dir,
    }

    # Initialize pipeline
    pipeline = DescriptivePipeline(config)

    # Run analysis
    try:
        start_time = time.time()

        results = pipeline.run()
        pipeline.save_results(results)

        if not args.no_viz:
            pipeline.generate_visualizations(results)

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.1f} seconds")

        pipeline.print_summary(results)

        print("\n" + "=" * 70)
        print("DESCRIPTIVE ANALYSIS COMPLETE")
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
