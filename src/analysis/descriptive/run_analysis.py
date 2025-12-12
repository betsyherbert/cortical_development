"""Run descriptive analysis pipeline."""

import time
from typing import Any

from .activity_analysis import DescriptiveAnalysis
from .visualizer import ActivityVisualizer


def run_descriptive_analysis() -> dict[str, Any]:
    """Execute complete descriptive analysis pipeline.

    Returns:
        Dict containing analysis results for all developmental stages.
    """
    print("=" * 70)
    print("Starting Descriptive Activity Analysis")
    print("=" * 70)

    start_time = time.time()

    # Run analysis
    analyzer = DescriptiveAnalysis()
    results = analyzer.run_analysis()

    analysis_time = time.time() - start_time
    print(f"\nData analysis completed in {analysis_time:.1f} seconds")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = ActivityVisualizer()
    visualizer.generate_all_plots(results)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print("Descriptive analysis complete!")

    return results


if __name__ == "__main__":
    run_descriptive_analysis()
