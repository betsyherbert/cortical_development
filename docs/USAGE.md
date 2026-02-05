# Analysis Pipeline Usage Guide

This guide provides quick reference for running the three analysis pipelines: bifurcation, stability, and descriptive analysis.

## Quick Start

All pipelines can be run via their CLI entry points:

```bash
python -m src.analysis.bifurcation
python -m src.analysis.stability
python -m src.analysis.descriptive
```

## Bifurcation Analysis

Analyzes network stability and gain through eigenvalue analysis and parameter space scanning.

### Basic Usage

```bash
# Run complete analysis (all stages, all analyses, both modes)
python -m src.analysis.bifurcation

# Run only stability analysis
python -m src.analysis.bifurcation --analysis stability

# Run only gain maps
python -m src.analysis.bifurcation --analysis gain_maps

# Run only gain spectra
python -m src.analysis.bifurcation --analysis gain_spectra

# Run maturity analysis
python -m src.analysis.bifurcation --analysis maturity
```

### Options

- `--analysis`: Type of analysis (`all`, `stability`, `gain`, `gain_maps`, `gain_spectra`, `maturity`)
- `--stages`: Developmental stages to analyze (`P0`, `P5`, `P10`, `P15`)
- `--mode`: Parameter range mode (`fixed_absolute`, `fixed_ratio`, `all`). Default: `all` (runs both modes)
- `--n-processes`: Number of parallel processes (default: cpu_count - 1)
- `--output-dir`: Custom output directory
- `--no-viz`: Skip visualization generation

### Examples

```bash
# Run for specific stages
python -m src.analysis.bifurcation --stages P0 P5

# Use fixed ratio mode only
python -m src.analysis.bifurcation --mode fixed_ratio

# Use fixed absolute mode only
python -m src.analysis.bifurcation --mode fixed_absolute

# Skip visualization
python -m src.analysis.bifurcation --no-viz

# Custom output directory
python -m src.analysis.bifurcation --output-dir /path/to/outputs
```

### Output

Results are saved to `outputs/bifurcation/`:
- `stability_maps_{mode}.pkl` - Stability analysis results
- `gain_maps_{mode}.pkl` - 2D gain maps
- `gain_spectra.pkl` - 1D gain spectra

## Stability Analysis

Analyzes local network stability by computing Jacobian eigenvalues for patches of neurons.

### Basic Usage

```bash
# Run complete analysis (all stages)
python -m src.analysis.stability
```

### Options

- `--stages`: Developmental stages to analyze (`P0`, `P5`, `P10`, `P15`)
- `--global`: Run whole-network (global) stability analysis instead of patch-based
- `--output-dir`: Custom output directory
- `--no-viz`: Skip visualization generation

### Examples

```bash
# Run for specific stages
python -m src.analysis.stability --stages P0 P5

# Whole-network stability analysis (saves to stability_analysis_global_results.pkl)
python -m src.analysis.stability --global

# Skip visualization
python -m src.analysis.stability --no-viz

# Custom output directory
python -m src.analysis.stability --output-dir /path/to/outputs
```

### Output

Results are saved to `outputs/stability/`:
- `stability_analysis_results.pkl` - Patch-based analysis results (default run)
- `stability_analysis_global_results.pkl` - Whole-network analysis results (when using `--global`)
- `snapshots/` - Snapshot data directory
- `summary/` - Visualization figures

## Descriptive Analysis

Analyzes network activity patterns (firing rates, correlations, synchronous events) over time.

### Basic Usage

```bash
# Run complete analysis (all stages)
python -m src.analysis.descriptive
```

### Options

- `--stages`: Developmental stages to analyze (`P0`, `P5`, `P10`, `P15`)
- `--output-dir`: Custom output directory
- `--no-viz`: Skip visualization generation

### Examples

```bash
# Run for specific stages
python -m src.analysis.descriptive --stages P0 P5

# Skip visualization
python -m src.analysis.descriptive --no-viz

# Custom output directory
python -m src.analysis.descriptive --output-dir /path/to/outputs
```

### Output

Results are saved to `outputs/descriptive/`:
- `descriptive_analysis_results.pkl` - Complete analysis results
- Various visualization figures (SVG format)

## Common Options

All three pipelines support:

- `--stages`: Select developmental stages (`P0`, `P5`, `P10`, `P15`)
- `--output-dir`: Custom output directory (defaults to `outputs/<module_name>/`)
- `--no-viz`: Skip visualization generation

## Getting Help

Get detailed help for any pipeline:

```bash
python -m src.analysis.bifurcation --help
python -m src.analysis.stability --help
python -m src.analysis.descriptive --help
```

## Programmatic Usage

You can also use the pipeline classes directly in Python:

```python
from src.analysis.bifurcation import BifurcationPipeline
from src.analysis.stability import StabilityPipeline
from src.analysis.descriptive import DescriptivePipeline

# Example: Run bifurcation analysis programmatically
config = {
    "stages": ["P0", "P5"],
    "mode": "fixed_absolute",
    "output_dir": "outputs/bifurcation"
}
pipeline = BifurcationPipeline(config)
results = pipeline.run_all()
pipeline.save_results(results)

# Example: Run stability analysis programmatically
config = {
    "stages": ["P0", "P5"],
    "output_dir": "outputs/stability"
}
pipeline = StabilityPipeline(config)
results = pipeline.run()
pipeline.save_results(results)
pipeline.generate_visualizations(results)

# Example: Run descriptive analysis programmatically
config = {
    "stages": ["P0", "P5"],
    "output_dir": "outputs/descriptive"
}
pipeline = DescriptivePipeline(config)
results = pipeline.run()
pipeline.save_results(results)
pipeline.generate_visualizations(results)
```

### Simulation core (no dashboard imports)

For programmatic access to the simulation without importing Dash/dashboard code, import
`CorticalSimulation` from `src.simulation`:

```python
from src.model.config import seed_random
from src.simulation import CorticalSimulation

seed_random(4)  # callers own seeding for reproducibility
sim = CorticalSimulation()
activities = sim.update()
```

Note: the analysis pipelines seed explicitly and save the used seed in the result metadata.

## Output Format

All pipelines save results using `save_with_version()` which wraps data in a versioned structure:

```python
{
    "version": "1.0",              # Format version
    "package_version": "x.y.z",     # Package version
    "timestamp": "2024-01-01 12:00:00",  # ISO timestamp
    "metadata": {                    # Run metadata
        "seed": 4,
        "stages": ["P0", "P5"],
        "params": {...}
    },
    "data": {...}                   # Actual results
}
```

Load results using:

```python
from src.analysis.common import load_with_version

loaded = load_with_version("outputs/bifurcation/stability_maps_fixed_absolute.pkl")
results = loaded["data"]
metadata = loaded["metadata"]
```

