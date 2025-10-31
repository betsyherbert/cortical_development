"""Run perturbation analysis with optimized cross-stage visualization."""

from .perturbation_analysis import PerturbationAnalysis
from .config import DEVELOPMENTAL_STAGES, PERTURBATION_TYPES, REGIMES, LAYERS


def run_perturbation_analysis():
    """Execute complete perturbation analysis pipeline."""
    print("=" * 70)
    print("Starting Perturbation Analysis")
    print("=" * 70)
    
    analysis = PerturbationAnalysis()
    results = analysis.run_analysis()
    
    # Print summary
    print("\nAnalysis Summary:")
    print("-" * 40)
    
    # Calculate figures: column-wise (1 per perturbation type) + layer-wise (1 per perturbation type per layer)
    figures_per_snapshot = len(PERTURBATION_TYPES) * (1 + len(LAYERS))
    
    total_figures = 0
    for regime, regime_results in results.items():
        n_snapshots = len(regime_results)
        figures_this_regime = n_snapshots * figures_per_snapshot
        total_figures += figures_this_regime
        
        print(f"{regime.title()} regime: {n_snapshots} snapshots × {figures_per_snapshot} figures = {figures_this_regime} figures")
    
    print(f"\nTotal figures generated: {total_figures}")
    print(f"Developmental stages tested: {', '.join(DEVELOPMENTAL_STAGES)}")
    print(f"Perturbation types: {', '.join(PERTURBATION_TYPES)}")
    print(f"Analysis complete!")
    
    return results


if __name__ == "__main__":
    run_perturbation_analysis() 