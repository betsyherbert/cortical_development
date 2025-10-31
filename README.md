# Cortical Circuit Development Simulator

An interactive simulation of early postnatal cortical circuit development, focusing on the barrel cortex. This project implements a multi-layer, spatially-extended firing rate model with real-time visualization using Dash.

![Simulation Preview](docs/simulation_preview.png)

## Model Overview

The simulation implements a firing rate model that includes:
- Multiple cortical layers (L2/3, L4, L5)
- Three cell types per layer (Excitatory, SST-expressing, PV-expressing interneurons)
- Thalamic input layer with intrinsic and sensory-driven activity
- Spatially-extended connectivity with Gaussian profiles
- Real-time interactive visualization with adjustable parameters

### Neural Dynamics

Each neuron follows a firing rate equation:
```
τ dV_i/dt = -V_i + ΣW_ij r_j + I^ext_i + noise
```
where `r_i = ReLU(V_i) = max(0, V_i)`

## Features

- **Interactive Dashboard**: Real-time visualization of neural activity in all populations
- **Adjustable Parameters**: Control connectivity strengths and thalamic input characteristics
- **Efficient Implementation**: Optimized for performance with vectorized operations
- **Modular Architecture**: Well-organized codebase for ease of extension and modification
- **Configurable**: Centralized configuration for easy parameter adjustments

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py        # Centralized configuration
│   │   ├── neurons.py       # Neural dynamics implementation
│   │   ├── connectivity.py  # Connectivity matrices and profiles
│   │   └── thalamus.py      # Thalamic input generation
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── dashboard.py     # Interactive Dash application
│   └── main.py              # Main entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cortical-circuit-simulator
```

2. Create and activate the virtual environment:
```bash
# On macOS/Linux
python3 -m venv dev_env
source dev_env/bin/activate

# On Windows
python -m venv dev_env
.\dev_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Simulation

Start the simulation with default parameters (port 8050):
```bash
python src/main.py
```

For custom configurations (e.g., change port or enable debug):
```bash
python src/main.py --port 8052 --debug
```

### Command-line Arguments

- `--port`: Specify the port for the Dash server (default: 8050)
- `--debug`: Run in debug mode with hot-reloading

### Interactive Controls

- **Neuron parameters**: Time constants and gains per cell type
- **Connection widths**: Thalamic and outgoing sigmas per cell type
- **Strength scaling**: Per cell-type scaling of all outgoing connections (incl. thalamus)
- **Sparsity**: Per cell-type sparsity of connections
- **Thalamic input balance**: `alpha` between intrinsic and sensory
- **External input (noise)**: Mean, std, and correlation per cell type
- **Connection matrix**: Click any cell to adjust that connection's strength via a slider
- **Pause/Resume**: Toggle simulation updates

## Customization

### Modifying Parameters

Core simulation parameters can be adjusted in `src/model/config.py`, including:
- Grid size
- Time constants
- Connectivity profiles
- Visualization settings
- Random seed (for reproducibility)

### Random Seed Management

The simulation uses a centralized random seed management system for reproducibility:
- Default seed is defined in `src/model/config.py` as `RANDOM_SEED`
- Use `seed_random()` from `src.model.config` to set the seed
- The seed is automatically set when creating a `CorticalSimulation` instance
- The seed is reset when calling `simulation.reset()`

```python
from src.model.config import seed_random

# Use default seed from config
seed_random()

# Use custom seed
seed_random(42)
```

### Extending the Model

To add new cell types or connection patterns:
1. Add new parameters to `config.py`
2. Update the relevant model classes in `neurons.py` and `connectivity.py`
3. Modify the visualization in `dashboard.py` to display the new elements

## Scientific Background

This model is based on established principles in computational neuroscience, particularly firing rate models of cortical dynamics. The implementation follows a mean-field approach where each unit represents the average activity of a local population of neurons.

The connectivity patterns reflect known neuroanatomical patterns in the mammalian cortex, with layer-specific and cell-type-specific connection profiles. The thalamic input model captures both intrinsic oscillatory activity (common in early development) and sensory-driven activity patterns.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 