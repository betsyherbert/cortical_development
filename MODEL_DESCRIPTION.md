# Cortical Circuit Development Model: Technical Description

## High-Level Overview

This codebase implements a spatially-extended firing rate model of early postnatal cortical circuit development, with specific focus on the barrel cortex. The model combines computational neuroscience principles with developmental biology to simulate the emergence of cortical circuits during the first postnatal week.

The simulation captures the interplay between:
1. **Multi-layer cortical architecture** (L2/3, L4, L5)
2. **Cell-type specific dynamics** (Excitatory, SST+, PV+ interneurons)
3. **Thalamic input patterns** (intrinsic bursts and sensory-driven activity)
4. **Spatial connectivity patterns** with Gaussian profiles
5. **Developmental progression** through parameter presets

## Key Simulation Parameters

The simulation behavior is controlled by several key parameters defined in `model/config.py`:

**Grid and Spatial:**
- `GRID_SIZE = 20`: Number of grid points in each dimension (20×20 grid)
- `ANATOMICAL_GRID_SIZE = 1000.0`: Physical size of cortical patch in μm (1000×1000 μm)
- Grid scale: 50 μm per grid unit (automatically derived)

**Temporal:**
- `DT = 3`: Time step in milliseconds for Euler integration
- `INTEGRATION_STEPS = 6`: Integration steps per update cycle
- `VISUALIZATION_STEPS = 5`: Simulation steps per visualization update
- `UPDATE_INTERVAL = 50`: Dashboard refresh interval in wall-clock milliseconds

**Reproducibility:**
- `RANDOM_SEED = 9`: Global random seed for reproducible simulations

**Thalamic Input Defaults:**
- `THALAMIC_INTRINSIC_SIGMA = 100.0`: Spatial spread of intrinsic bursts (μm)
- `THALAMIC_INTRINSIC_DURATION = 30.0`: Mean duration of intrinsic bursts (ms)
- `THALAMIC_INTRINSIC_INTERVAL = 20.0`: Mean interval between bursts (ms)
- `THALAMIC_INTRINSIC_AMP = 3.0`: Mean amplitude of intrinsic bursts
- `THALAMIC_SENSORY_SIGMA = 25.0`: Spatial spread of sensory inputs (μm)
- `THALAMIC_SENSORY_DURATION = 10.0`: Mean duration of sensory bursts (ms)
- `THALAMIC_SENSORY_INTERVAL = 10.0`: Mean interval between sensory bursts (ms)
- `THALAMIC_SENSORY_AMP = 1.0`: Mean amplitude of sensory bursts

**Preset-Dependent Parameters:**
- `THALAMIC_ALPHA`: Balance between intrinsic (0) and sensory (1) activity (varies: 0.1→0.8)
- Connection strengths, time constants, and scaling factors (defined per preset)

## Mathematical Framework

### Core Neural Dynamics

The fundamental equation governing each neural population follows a firing rate model:

```
τ_c dV_c(x,y,t)/dt = -V_c(x,y,t) + Σ_s W_sc * r_s(x,y,t) + I^ext_c(x,y,t) + I^bg_c
```

Where:
- `V_c(x,y,t)` is the membrane potential of cell type `c` at spatial location `(x,y)` and time `t`
- `τ_c` is the membrane time constant (cell-type specific, typically 5-25 ms)
- `r_c(x,y,t) = ReLU(g_c * V_c(x,y,t)) = max(0, g_c * V_c(x,y,t))` is the firing rate
- `g_c` is the gain parameter for cell type `c` (typically fixed at 1.0)
- `W_sc` represents the connectivity weight matrix from source cell type `s` to target cell type `c`
- `I^ext_c(x,y,t)` is external input (primarily thalamic)
- `I^bg_c` is constant background input for cell type `c`

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

**Connectivity Constraints:**
- SST interneurons do not form connections onto other SST interneurons (SST→SST connections are excluded)
- All other pairwise connections between cell types are implemented

### Strength Scaling System

The model implements a two-level connectivity strength system:

1. **Base Connection Amplitudes** (`A_sc`): Specific to each source-target pair
2. **Cell-Type Scaling Factors** (`S_s`): Multiplicative factors applied to all outgoing connections from source cell type `s`

This allows independent control of:
- Individual connection specificity (via base amplitudes)
- Overall cell-type activity levels (via scaling factors)

The effective connection strength is: `W_effective = A_sc × S_s`

Typical scaling factor ranges:
- E cells: 0.8-4.0 (increases during development)
- SST cells: 0.5-1.5 (moderate increase)
- PV cells: 0.5-3.5 (late developmental increase)
- Thalamus: 1.0-4.0 (increases with sensory refinement)

### Spatial Connectivity

Connections between populations follow spatially-structured patterns with Gaussian profiles:

```
W_sc(x,y,x',y') = A_sc * S_s * G_σsc(x-x', y-y')
```

Where:
- `A_sc` is the base connection amplitude between source type `s` and target type `c`
- `S_s` is the strength scaling factor for source cell type `s` (applies to all outgoing connections)
- `G_σsc(dx,dy) = exp(-0.5 * (dx² + dy²) / σ_sc²) / Σ exp(...)` is the normalized Gaussian profile
- `σ_sc` is the spatial spread of connections (specified in μm, anatomical units)

**Important**: All spatial parameters (σ values) are specified in μm (micrometers) representing anatomical distances. These are automatically converted to grid units internally based on the anatomical grid size (default 1000 μm × 1000 μm for a 20×20 grid).

## Thalamic Input Model

The thalamic input combines two components representing different activity patterns during early development:

### Intrinsic Bursts
Spontaneous wave-like activity patterns:
```
I_intrinsic(x,y,t) = max(0, Σ_i A_i * T_i(t) * G_σi(x-x_i, y-y_i) * cos(φ_i))
```

Where each burst `i` has:
- Gaussian spatial profile with center `(x_i, y_i)` and width `σ_i` (in μm)
- Temporal envelope `T_i(t)` with oscillatory component and smooth rise/fall
- Phase `φ_i` for wave-like propagation and interference patterns
- Amplitude `A_i` (randomized around mean THALAMIC_INTRINSIC_AMP)
- Duration (randomized around mean THALAMIC_INTRINSIC_DURATION)

### Sensory-Driven Activity
Localized stimulus-evoked responses:
```
I_sensory(x,y,t) = max_i [A_i * T_i(t) * G_σi(x-x_i, y-y_i)]
```

Where:
- Multiple sensory bursts are combined using maximum (not summation)
- Each burst has amplitude `A_i` (randomized around mean THALAMIC_SENSORY_AMP)
- Temporal profile `T_i(t)` with smooth rise/fall (no oscillations)
- Narrower spatial width (σ ≈ 25 μm) compared to intrinsic bursts

### Combined Thalamic Input
```
I_thalamic(x,y,t) = (1-α) * I_intrinsic(x,y,t) + α * I_sensory(x,y,t)
```

Where `α ∈ [0,1]` controls the balance between intrinsic and sensory-driven activity.

## Developmental Presets

The model includes four developmental stage presets that capture key changes during early postnatal development:

### P0 (Postnatal Day 0 - Early 1st Week)
- **Weak connections**: E scaling 0.8, SST/PV scaling 0.5
- **Broad connectivity**: Outgoing widths ~300 μm
- **Intrinsic-dominated**: α = 0.1 (90% intrinsic thalamic activity)
- **Slow dynamics**: E tau 15 ms, SST/PV tau 25 ms
- **Sparse inhibition**: Most inhibitory connections weak or absent

### P5 (Postnatal Day 5 - Late 1st Week)
- **Strengthening connections**: E scaling 1.5, SST scaling 1.5, PV scaling 1.0
- **Refined connectivity**: E widths 125 μm, SST/PV widths 300 μm
- **Mixed activity**: α = 0.3 (70% intrinsic, 30% sensory)
- **Moderate dynamics**: E tau 14 ms, SST/PV tau 20 ms
- **Emerging inhibition**: Inhibitory connections starting to develop

### P10 (Postnatal Day 10 - Mid 2nd Week)
- **Strong connections**: E scaling 2.5, SST/PV scaling 1.5
- **Sharpening connectivity**: E widths 140 μm, SST widths 75 μm, PV widths 100 μm
- **Sensory-dominated**: α = 0.6 (40% intrinsic, 60% sensory)
- **Fast E dynamics**: E tau 9 ms, SST/PV tau 20 ms
- **Functional inhibition**: Inhibitory connections strengthening

### P15 (Postnatal Day 15 - Late 2nd Week)
- **Mature-like connections**: E scaling 4.0, PV scaling 3.5, SST scaling 1.5
- **Narrow connectivity**: E widths 40 μm, SST widths 50 μm, PV widths 60 μm
- **Sensory-driven**: α = 0.8 (20% intrinsic, 80% sensory)
- **Fast dynamics**: E tau 7 ms, SST tau 10 ms, PV tau 5 ms
- **Strong inhibition**: PV inhibition becomes dominant

**Key Developmental Trends:**
1. Excitatory connections strengthen and sharpen spatially
2. PV inhibition develops later but becomes very strong
3. Thalamic input shifts from intrinsic to sensory-driven
4. Neural time constants decrease (faster dynamics)
5. Background inputs may increase for excitatory cells

## Implementation Architecture

### Core Classes

1. **NeuralLayer**: Manages dynamics for a single cortical layer with three cell types
2. **CorticalCircuit**: Integrates multiple layers (L2/3, L4, L5) with full connectivity
3. **ThalamicInput**: Generates realistic thalamic activity patterns (intrinsic + sensory)
4. **LayerConnectivity**: Manages spatial connectivity matrices for all layer connections
5. **ConnectivityProfile**: Optimized Gaussian profile computation with caching
6. **CorticalSimulation**: Main simulation class integrating all components
7. **DashboardApp**: Interactive visualization and control interface

### Key Modules

The codebase is organized into several key modules:

1. **model/config.py**: Central configuration module containing:
   - Core simulation parameters (grid size, time steps, anatomical dimensions)
   - Network structure definitions (cell types, layers, connection lists)
   - Spatial scale conversion functions (μm ↔ grid units)
   - Layer-specific connectivity parameters (amplitudes and sigmas in μm)
   - Visualization settings and color schemes
   - Random seed management for reproducibility

2. **model/neurons.py**: Core neural dynamics implementation:
   - `NeuralLayer` class for single-layer dynamics with cell-type specific parameters
   - `CorticalCircuit` class for multi-layer integration
   - Euler integration of firing rate equations
   - ReLU activation with configurable gain
   - Background input and time constant management

3. **model/connectivity.py**: Spatial connectivity implementation:
   - `ConnectivityProfile` class for efficient Gaussian profile computation and caching
   - `LayerConnectivity` class managing all inter-layer and intra-layer connections
   - Weight matrix computation and caching for all connection types
   - Strength scaling system applying multiplicative factors to outgoing connections
   - Spatial parameter handling in anatomical units (μm)

4. **model/thalamus.py**: Thalamic input generation:
   - `ThalamicInput` class generating realistic activity patterns
   - Intrinsic burst pattern generation with wave-like propagation
   - Sensory-driven localized activity simulation
   - Spatial profiles in anatomical units (μm)
   - Temporal envelope computation with smooth transitions

5. **model/presets.py**: Development stage presets:
   - Parameter sets for P0, P5, P10, and P15 developmental timepoints
   - Connection strength matrices for all layer-to-layer connections
   - Spatial connectivity widths (outgoing_widths, thalamic_widths in μm)
   - Strength scaling factors per cell type
   - Time constants and background inputs per developmental stage

6. **visualization/dashboard.py**: Interactive visualization:
   - Real-time activity heatmaps for all layers and cell types
   - Parameter control interface (time constants, gains, background inputs)
   - Connection strength visualization and editing
   - Development stage preset selection
   - Strength scaling controls

7. **main.py**: Application entry point:
   - `CorticalSimulation` class integrating circuit and thalamic input
   - Simulation initialization with reproducible random seeding
   - Parameter management interface
   - Dashboard integration and server startup


### Spatial Discretization

The model uses a 2D square grid (default 20×20) representing a cortical patch of configurable anatomical size (default 1000 μm × 1000 μm). Each grid point corresponds to a local population of neurons of each cell type.

**Spatial Scale Conversion:**
- Grid scale: anatomical_size / grid_size (default: 50 μm per grid unit)
- All spatial parameters (connection widths, thalamic input profiles) are specified in μm
- Automatic conversion to grid units happens internally for computation
- This allows consistent anatomical interpretation regardless of grid resolution

### Temporal Integration

The simulation uses explicit Euler integration with:
- Time step: `dt = 3ms` (configurable via `DT` parameter)
- Multiple integration steps per visualization update for stability (default: 6 steps)
- Additional visualization steps control update frequency (default: 5 steps)
- Configurable parameters for balancing simulation speed vs temporal resolution