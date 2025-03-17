# Cortical Circuit Development Simulator

An interactive simulation of early postnatal cortical circuit development, focusing on the barrel cortex. This project implements a multi-layer, spatially-extended firing rate model with real-time visualization.

## Model Overview

The simulation includes:
- Multiple cortical layers (L2/3, L4, L5)
- Three cell types per layer (Excitatory, SST, PV)
- Thalamic input layer
- Spatially-extended connectivity with Gaussian profiles
- Real-time interactive visualization

### Neural Dynamics

Each neuron follows a firing rate equation:
```math
τ dV_i/dt = -V_i + ΣW_ij r_j + I^ext_i
```
where r_i = ReLU(V_i) = max(0, V_i)

## Setup

1. Create and activate the virtual environment:
```bash
python3 -m venv dev_env
source dev_env/bin/activate  # On Unix/MacOS
# or
.\dev_env\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python src/main.py
```

The interactive dashboard will open in your default web browser, allowing you to:
- Visualize neural activity in real-time
- Adjust connectivity parameters
- Modify simulation parameters

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── neurons.py
│   │   ├── connectivity.py
│   │   └── thalamus.py
│   ├── visualization/
│   │   └── dashboard.py
│   └── main.py
├── requirements.txt
└── README.md
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 