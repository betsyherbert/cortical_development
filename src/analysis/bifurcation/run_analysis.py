"""Run bifurcation analysis pipeline with orchestration.

This module provides the main analysis orchestrator and CLI entry point for
bifurcation analysis, coordinating stability and gain map computations.
"""

import argparse
import time
from pathlib import Path

from src.analysis.common import make_run_metadata, save_with_version

from .config import (
    DEFAULT_GAIN_PAIRS,
    DEFAULT_SPECTRUM_SWEEPS,
    DEFAULT_STABILITY_PAIRS,
    OUTPUT_DIR,
)
from .gain_maps import compute_gain_maps_all_stages, compute_gain_spectra_all_stages
from .stability_maps import compute_stability_maps_all_stages


class BifurcationPipeline:
    """Main orchestrator for bifurcation analysis pipeline."""

    def __init__(self, config: dict | None = None):
        """Initialize bifurcation analysis.

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
            "stages": ["P0", "P5", "P10", "P15"],
            "mode": "fixed_absolute",
            "n_processes": None,  # Will use cpu_count - 1
            "stability_pairs": DEFAULT_STABILITY_PAIRS,
            "gain_pairs": DEFAULT_GAIN_PAIRS,
            "spectrum_sweeps": DEFAULT_SPECTRUM_SWEEPS,
            "output_dir": OUTPUT_DIR,
        }

    def run_stability_analysis(self) -> dict:
        """Run stability map analysis.

        Returns:
            Dictionary with stability results organized by parameter pair and stage
        """
        print("\n" + "=" * 70)
        print("STABILITY ANALYSIS")
        print("=" * 70)

        start_time = time.time()

        # Run stability maps
        results = compute_stability_maps_all_stages(
            parameter_pairs=self.config["stability_pairs"],
            stages=self.config["stages"],
            mode=self.config["mode"],
            n_processes=self.config["n_processes"],
        )

        elapsed = time.time() - start_time
        print(f"\nStability analysis completed in {elapsed:.1f} seconds")

        # Save results
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        mode_suffix = self.config["mode"]
        results_file = output_dir / f"stability_maps_{mode_suffix}.pkl"
        metadata = make_run_metadata(
            stages=self.config["stages"],
            params={"mode": mode_suffix, "pairs": self.config["stability_pairs"]},
        )
        save_with_version(results, str(results_file), metadata=metadata)
        print(f"Results saved to: {results_file}")

        return results

    def run_gain_analysis(self) -> dict:
        """Run gain map and spectrum analysis.

        Returns:
            Dictionary with 'maps' and 'spectra' keys containing respective results
        """
        print("\n" + "=" * 70)
        print("GAIN ANALYSIS")
        print("=" * 70)

        start_time = time.time()

        # Run 2D gain maps
        print("\n>>> Running 2D Gain Maps...")
        map_results = compute_gain_maps_all_stages(
            parameter_pairs=self.config["gain_pairs"],
            stages=self.config["stages"],
            mode=self.config["mode"],
            n_processes=self.config["n_processes"],
        )

        # Run 1D gain spectra
        print("\n>>> Running 1D Gain Spectra...")
        spectrum_results = compute_gain_spectra_all_stages(
            parameter_keys=self.config["spectrum_sweeps"],
            stages=self.config["stages"],
            n_processes=self.config["n_processes"],
        )

        elapsed = time.time() - start_time
        print(f"\nGain analysis completed in {elapsed:.1f} seconds")

        # Save results
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        mode_suffix = self.config["mode"]

        # Save maps
        maps_file = output_dir / f"gain_maps_{mode_suffix}.pkl"
        map_metadata = make_run_metadata(
            stages=self.config["stages"],
            params={"mode": mode_suffix, "pairs": self.config["gain_pairs"]},
        )
        save_with_version(map_results, str(maps_file), metadata=map_metadata)
        print(f"Gain maps saved to: {maps_file}")

        # Save spectra
        spectra_file = output_dir / "gain_spectra.pkl"
        spectra_metadata = make_run_metadata(
            stages=self.config["stages"],
            params={"sweeps": self.config["spectrum_sweeps"]},
        )
        save_with_version(spectrum_results, str(spectra_file), metadata=spectra_metadata)
        print(f"Gain spectra saved to: {spectra_file}")

        return {"maps": map_results, "spectra": spectrum_results}

    def run_all(self) -> dict:
        """Run complete bifurcation analysis pipeline.

        Returns:
            Dictionary with 'stability', 'gain_maps', and 'gain_spectra' results
        """
        print("\n" + "=" * 70)
        print("BIFURCATION ANALYSIS - COMPLETE PIPELINE")
        print("=" * 70)
        print(f"Stages: {', '.join(self.config['stages'])}")
        print(f"Mode: {self.config['mode']}")
        print(f"Output: {self.config['output_dir']}")

        overall_start = time.time()

        # Run stability analysis
        stability_results = self.run_stability_analysis()

        # Run gain analysis
        gain_results = self.run_gain_analysis()

        total_time = time.time() - overall_start

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.1f} seconds")

        return {
            "stability": stability_results,
            "gain_maps": gain_results["maps"],
            "gain_spectra": gain_results["spectra"],
        }


def main():
    """CLI entry point for bifurcation analysis."""
    parser = argparse.ArgumentParser(
        description="Bifurcation analysis for cortical circuit development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis with default parameters
  python -m src.analysis.bifurcation

  # Run only stability analysis
  python -m src.analysis.bifurcation --analysis stability

  # Run with fixed ratio mode
  python -m src.analysis.bifurcation --mode fixed_ratio

  # Run for specific stages
  python -m src.analysis.bifurcation --stages P0 P5

  # Specify number of parallel processes
  python -m src.analysis.bifurcation --n-processes 4
        """,
    )

    parser.add_argument(
        "--analysis",
        choices=["all", "stability", "gain", "gain_maps", "gain_spectra"],
        default="all",
        help="Type of analysis to run (default: all)",
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["P0", "P5", "P10", "P15"],
        default=["P0", "P5", "P10", "P15"],
        help="Developmental stages to analyze (default: all)",
    )

    parser.add_argument(
        "--mode",
        choices=["fixed_absolute", "fixed_ratio"],
        default="fixed_absolute",
        help="Parameter range mode (default: fixed_absolute)",
    )

    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: cpu_count - 1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})",
    )

    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")

    args = parser.parse_args()

    # Build configuration
    config = {
        "stages": args.stages,
        "mode": args.mode,
        "n_processes": args.n_processes,
        "stability_pairs": DEFAULT_STABILITY_PAIRS,
        "gain_pairs": DEFAULT_GAIN_PAIRS,
        "spectrum_sweeps": DEFAULT_SPECTRUM_SWEEPS,
        "output_dir": args.output_dir,
    }

    # Initialize analyzer
    analyzer = BifurcationPipeline(config)

    # Run requested analysis
    try:
        if args.analysis == "all":
            results = analyzer.run_all()
        elif args.analysis == "stability":
            results = {"stability": analyzer.run_stability_analysis()}
        elif args.analysis == "gain":
            results = {"gain": analyzer.run_gain_analysis()}
        elif args.analysis == "gain_maps":
            # Run only 2D maps
            config_maps = config.copy()
            config_maps["spectrum_sweeps"] = []
            BifurcationPipeline(config_maps)
            print("\n" + "=" * 70)
            print("GAIN MAPS ONLY")
            print("=" * 70)
            map_results = compute_gain_maps_all_stages(
                parameter_pairs=config["gain_pairs"],
                stages=config["stages"],
                mode=config["mode"],
                n_processes=config["n_processes"],
            )
            output_dir = Path(config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            maps_file = output_dir / f'gain_maps_{config["mode"]}.pkl'
            map_metadata = make_run_metadata(
                stages=config["stages"],
                params={"mode": config["mode"], "pairs": config["gain_pairs"]},
            )
            save_with_version(map_results, str(maps_file), metadata=map_metadata)
            print(f"Gain maps saved to: {maps_file}")
            results = {"gain_maps": map_results}
        elif args.analysis == "gain_spectra":
            # Run only 1D spectra
            print("\n" + "=" * 70)
            print("GAIN SPECTRA ONLY")
            print("=" * 70)
            spectrum_results = compute_gain_spectra_all_stages(
                parameter_keys=config["spectrum_sweeps"],
                stages=config["stages"],
                n_processes=config["n_processes"],
            )
            output_dir = Path(config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            spectra_file = output_dir / "gain_spectra.pkl"
            spectra_metadata = make_run_metadata(
                stages=config["stages"],
                params={"sweeps": config["spectrum_sweeps"]},
            )
            save_with_version(spectrum_results, str(spectra_file), metadata=spectra_metadata)
            print(f"Gain spectra saved to: {spectra_file}")
            results = {"gain_spectra": spectrum_results}

        # Generate visualizations if requested
        if not args.no_viz:
            print("\n" + "=" * 70)
            print("GENERATING VISUALIZATIONS")
            print("=" * 70)

            from .visualizer import BifurcationVisualizer

            visualizer = BifurcationVisualizer()
            visualizer.generate_all_figures(results, mode=args.mode)

            print("\nVisualization complete!")

        print("\n" + "=" * 70)
        print("BIFURCATION ANALYSIS COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:
        print("\nERROR: Analysis failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
