# Descriptive Activity Analysis

This module generates comprehensive descriptive plots of network activity across developmental stages.

## Overview

Analyzes cortical circuit activity patterns across four developmental timepoints (P4, P8, P12, P16) and generates:

### Timeseries Plots (4×3 subplots)
- **Activity Heatmaps**: Spatial activity evolution over time, layers stacked vertically
- **Average Firing Rates**: Mean firing rates across all layers for each cell type
- **Active Cell Fractions**: Fraction of cells above activity threshold over time

### Developmental Trends (1×3 subplots)  
- **Pairwise Correlations**: Correlation development by cell type, layer, and total network
- **Synchronous Events**: Large synchronous event frequency by cell type, layer, and total network

## Usage

### Run Complete Analysis
```bash
# From project root
python -m src.analysis.descriptive
```

### Programmatic Usage
```python
from src.analysis.descriptive import DescriptiveAnalysis, ActivityVisualizer

# Run analysis
analyzer = DescriptiveAnalysis()
results = analyzer.run_analysis()

# Generate plots
visualizer = ActivityVisualizer() 
visualizer.generate_all_plots(results)
```

## Configuration

Edit parameters in `config.py`:
- `simulation_duration`: Simulation length (default: 10.0 seconds)
- `activity_threshold`: Threshold for "active" cells (default: 0.2)
- `synchronous_event_threshold`: Threshold for synchronous events (default: 20% of cells)
- `sampling_interval`: Data sampling frequency (default: 10.0 ms)
- `AVERAGE_FIRING_RATE_YLIM`: Y-axis limits for average firing rate plots (default: None for auto-scaling)
  - Set to `[min, max]` for fixed limits (e.g., `[0, 1.0]`)
  - Set to `None` for automatic scaling based on data

## Output

Plots saved to: `outputs/descriptive/`
- `activity_heatmaps.svg`
- `average_firing_rates.svg` 
- `active_cell_fractions.svg`
- `correlation_trends.svg`
- `synchronous_event_trends.svg`

## Analysis Details

- **Duration**: 5 seconds per developmental stage (500 timepoints at 10ms intervals)
- **Stages**: P4, P8, P12, P16 with stage-appropriate thalamic input
- **Consistent Scaling**: All plots use global min/max for fair comparison across ages
- **Execution Time**: ~20-25 seconds total 