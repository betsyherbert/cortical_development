# Cortical Circuit Development Model: Technical Description

## High-Level Overview

This codebase implements a spatially-extended firing rate model of early postnatal cortical circuit development, with specific focus on the barrel cortex. The model combines computational neuroscience principles with developmental biology to simulate the emergence of cortical circuits during the first postnatal week.

The simulation captures the interplay between:
1. **Multi-layer cortical architecture** (L2/3, L4, L5)
2. **Cell-type specific dynamics** (Excitatory, SST+, PV+ interneurons)
3. **Thalamic input patterns** (intrinsic bursts and sensory-driven activity)
4. **Spatial connectivity patterns** with Gaussian profiles
5. **Developmental progression** through parameter presets

## Mathematical Framework

### Core Neural Dynamics

The fundamental equation governing each neural population follows a firing rate model:

```
τ_c dV_c(x,y,t)/dt = -V_c(x,y,t) + Σ_s W_sc * r_s(x,y,t) + I^ext_c(x,y,t) + η_c(x,y,t)
```

Where:
- `V_c(x,y,t)` is the membrane potential of cell type `c` at spatial location `(x,y)` and time `t`
- `τ_c` is the membrane time constant (cell-type specific)
- `r_c(x,y,t) = ReLU(g_c * V_c(x,y,t)) = max(0, g_c * V_c(x,y,t))` is the firing rate
- `g_c` is the gain parameter for cell type `c`
- `W_sc` represents the connectivity weight matrix from source cell type `s` to target cell type `c`
- `I^ext_c(x,y,t)` is external input (primarily thalamic)
- `η_c(x,y,t)` is correlated noise

### Cell Types and Layers

The model implements three cell types across three cortical layers:

**Cell Types:**
- **E (Excitatory)**: Pyramidal neurons with typical cortical excitatory properties
- **SST (Somatostatin-expressing)**: Inhibitory interneurons targeting dendrites
- **PV (Parvalbumin-expressing)**: Fast-spiking inhibitory interneurons

**Layers:**
- **L2/3 (Layer 2/3)**: Superficial cortical layer
- **L4 (Layer 4)**: Primary input layer from thalamus
- **L5 (Layer 5)**: Deep cortical layer

### Spatial Connectivity

Connections between populations follow spatially-structured patterns with Gaussian profiles:

```
W_sc(x,y,x',y') = A_sc * G_σsc(x-x', y-y') * M_sc(x,y,x',y')
```

Where:
- `A_sc` is the connection amplitude between source type `s` and target type `c`
- `G_σsc(dx,dy) = exp(-0.5 * (dx² + dy²) / σ_sc²) / (2π * σ_sc²)` is the normalized Gaussian profile
- `σ_sc` is the spatial spread of connections
- `M_sc` is a binary sparsity mask determining which connections exist

### Noise Model

The model implements spatially and temporally correlated noise using an Ornstein-Uhlenbeck process:

```
dη_c(x,y,t) = -(η_c(x,y,t) - μ_c) * dt/τ_noise + σ_c * √(2*dt/τ_noise) * dW_c(x,y,t)
```

With both private and shared noise components:
```
η_c(x,y,t) = μ_c + σ_c * [√(1-c_c) * η_private_c(x,y,t) + √(c_c) * η_shared(x,y,t)]
```

Where:
- `μ_c` is the mean noise level
- `σ_c` is the noise standard deviation
- `c_c` is the correlation coefficient controlling shared vs. private noise
- `τ_noise = 20ms` is the noise correlation time constant

## Thalamic Input Model

The thalamic input combines two components representing different activity patterns during early development:

### Intrinsic Bursts
Spontaneous wave-like activity patterns:
```
I_intrinsic(x,y,t) = Σ_i A_i(t) * G_σi(x-x_i, y-y_i) * cos(φ_i(t))
```

Where each burst `i` has:
- Gaussian spatial profile with center `(x_i, y_i)` and width `σ_i`
- Temporal envelope `A_i(t)` with oscillatory component
- Phase `φ_i(t)` for wave-like propagation

### Sensory-Driven Activity
Localized stimulus-evoked responses:
```
I_sensory(x,y,t) = max_i [A_i(t) * G_σi(x-x_i, y-y_i)]
```

### Combined Thalamic Input
```
I_thalamic(x,y,t) = (1-α) * I_intrinsic(x,y,t) + α * I_sensory(x,y,t)
```

Where `α ∈ [0,1]` controls the balance between intrinsic and sensory-driven activity.

## Implementation Architecture

### Core Classes

1. **NoiseGenerator**: Implements correlated Ornstein-Uhlenbeck noise
2. **NeuralLayer**: Manages dynamics for a single cortical layer
3. **CorticalCircuit**: Integrates multiple layers with connectivity
4. **ThalamicInput**: Generates realistic thalamic activity patterns
5. **LayerConnectivity**: Manages spatial connectivity matrices
6. **ConnectivityProfile**: Optimized Gaussian profile computation

### Key Modules

The codebase is organized into several key modules:

1. **model/config.py**: Central configuration module containing:
   - Core simulation parameters (grid size, time steps, etc.)
   - Network structure definitions (cell types, layers)
   - Default parameter values for all components
   - Visualization settings

2. **model/neurons.py**: Core neural dynamics implementation:
   - NeuralLayer class for single-layer dynamics
   - CorticalCircuit class for multi-layer integration
   - Rate model implementation with spatial connectivity
   - Noise generation and integration

3. **model/thalamus.py**: Thalamic input generation:
   - Intrinsic burst pattern generation
   - Sensory-driven activity simulation
   - Spatial and temporal profile computation
   - Input mixing and modulation

4. **model/presets.py**: Development stage presets:
   - Parameter sets for different developmental timepoints
   - Connection strength configurations
   - Spatial connectivity patterns
   - Cell-type specific properties

5. **visualization/dashboard.py**: Interactive visualization:
   - Real-time activity plotting
   - Parameter control interface
   - Connection strength visualization
   - Development stage selection

6. **main.py**: Application entry point:
   - Simulation initialization and control
   - Component integration
   - Runtime parameter management
   - Update loop coordination


### Spatial Discretization

The model uses a 2D square grid (default 20×20) representing cortical space. Each grid point corresponds to a local population of neurons of each cell type.

### Temporal Integration

The simulation uses explicit Euler integration with:
- Time step: `dt = 1.5ms`
- Multiple integration steps per visualization update for stability
- Configurable integration steps for performance tuning