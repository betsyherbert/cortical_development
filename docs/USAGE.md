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
# Run complete analysis (all stages, all analyses)
python -m src.analysis.bifurcation

# Run only stability analysis
python -m src.analysis.bifurcation --analysis stability

# Run only gain maps
python -m src.analysis.bifurcation --analysis gain_maps

# Run only gain spectra
python -m src.analysis.bifurcation --analysis gain_spectra
```

### Options

- `--analysis`: Type of analysis (`all`, `stability`, `gain`, `gain_maps`, `gain_spectra`)
- `--stages`: Developmental stages to analyze (`P0`, `P5`, `P10`, `P15`)
- `--mode`: Parameter range mode (`fixed_absolute`, `fixed_ratio`)
- `--n-processes`: Number of parallel processes (default: cpu_count - 1)
- `--output-dir`: Custom output directory
- `--no-viz`: Skip visualization generation

### Examples

```bash
# Run for specific stages
python -m src.analysis.bifurcation --stages P0 P5

# Use fixed ratio mode
python -m src.analysis.bifurcation --mode fixed_ratio

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
- `--output-dir`: Custom output directory
- `--no-viz`: Skip visualization generation

### Examples

```bash
# Run for specific stages
python -m src.analysis.stability --stages P0 P5

# Skip visualization
python -m src.analysis.stability --no-viz

# Custom output directory
python -m src.analysis.stability --output-dir /path/to/outputs
```

### Output

Results are saved to `outputs/stability/`:
- `stability_analysis_results.pkl` - Complete analysis results
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

