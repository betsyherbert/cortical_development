"""Neural dynamics module for the cortical circuit simulation."""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from .connectivity import LayerConnectivity
from .config import (
    GRID_SIZE, DT, NEURAL_TAU, NOISE_AMPLITUDE,
    INTEGRATION_STEPS, CELL_TYPES, LAYERS
)


class NeuralLayer:
    """
    Implements neural population dynamics for a single cortical layer.
    
    Each layer contains three cell types: excitatory (E), somatostatin-expressing (SST),
    and parvalbumin-expressing (PV) interneurons, each with their own dynamics.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE, tau: float = NEURAL_TAU, dt: float = DT):
        """
        Initialize a neural layer with specified grid size and time constant.
        
        Args:
            grid_size: Size of the square grid
            tau: Time constant in milliseconds
            dt: Time step in milliseconds
        """
        self.grid_size = grid_size
        self.tau = tau
        self.dt = dt
        
        # Initialize membrane potentials and firing rates for all cell types
        self.V = {
            'E': np.zeros((grid_size, grid_size)),
            'SST': np.zeros((grid_size, grid_size)),
            'PV': np.zeros((grid_size, grid_size))
        }
        
        # Initialize firing rates
        self.r = {
            'E': np.zeros((grid_size, grid_size)),
            'SST': np.zeros((grid_size, grid_size)),
            'PV': np.zeros((grid_size, grid_size))
        }
        
        # Noise amplitude for dynamics
        self.noise_amplitude = NOISE_AMPLITUDE

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function: max(0, x)."""
        return np.maximum(0, x)

    def update(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update neural dynamics for one time step.
        
        Args:
            inputs: Dictionary mapping cell types to their input currents
            
        Returns:
            Dictionary of updated firing rates for each cell type
        """
        # Add noise to all inputs
        noise = {
            cell_type: self.noise_amplitude * np.random.randn(*input_curr.shape)
            for cell_type, input_curr in inputs.items()
        }
        
        # Update membrane potentials using Euler method (vectorized)
        for cell_type in CELL_TYPES:
            if cell_type in inputs:
                # Calculate voltage change: dV = (-V + I + noise) * dt/tau
                dV = (-self.V[cell_type] + inputs[cell_type] + noise[cell_type]) * (self.dt / self.tau)
                self.V[cell_type] += dV
                
                # Update firing rates with ReLU activation
                self.r[cell_type] = self.relu(self.V[cell_type])
        
        # Return current firing rates
        return self.r

    def reset(self) -> None:
        """Reset all state variables to zero."""
        for cell_type in CELL_TYPES:
            self.V[cell_type].fill(0)
            self.r[cell_type].fill(0)


class CorticalCircuit:
    """
    Integrates multiple neural layers into a complete cortical circuit.
    
    This class manages the connections between layers and handles the overall
    circuit dynamics, including inputs from thalamus.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE):
        """
        Initialize the full cortical circuit with all layers.
        
        Args:
            grid_size: Size of the square grid
        """
        self.grid_size = grid_size
        
        # Initialize cortical layers
        self.layers = {
            'L23': NeuralLayer(grid_size),
            'L4': NeuralLayer(grid_size),
            'L5': NeuralLayer(grid_size)
        }
        
        # Initialize connectivity
        self.connectivity = LayerConnectivity(grid_size)
        
        # Initialize thalamic input (will be set from outside)
        self.thalamus = np.zeros((grid_size, grid_size))
    
    def get_layer_activities(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get current activities of all populations.
        
        Returns:
            Dictionary containing firing rates of all populations
        """
        activities = {
            layer_name: self.layers[layer_name].r.copy()
            for layer_name in LAYERS
        }
        activities['thalamus'] = self.thalamus
        
        return activities
    
    def reset(self) -> None:
        """Reset all layers to their initial state."""
        for layer in self.layers.values():
            layer.reset()
        self.thalamus = np.zeros((self.grid_size, self.grid_size))

    def update(self, n_steps: int = INTEGRATION_STEPS) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Update the circuit state for multiple time steps.
        
        Args:
            n_steps: Number of integration steps to perform
            
        Returns:
            Dictionary containing all population activities
        """
        grid_shape = (self.grid_size, self.grid_size)
        thalamus_flat = self.thalamus.flatten()
        
        # Precompute thalamic inputs once outside the integration loop
        thalamic_inputs = {}
        for target_layer in LAYERS:
            thalamic_inputs[target_layer] = {}
            for target_cell in CELL_TYPES:
                thal_input = self.connectivity.compute_input(
                    'thalamus', target_cell, thalamus_flat,
                    source_layer='thalamus', target_layer=target_layer
                )
                thalamic_inputs[target_layer][target_cell] = thal_input.reshape(grid_shape)
        
        # Run multiple steps for smoother dynamics
        for _ in range(n_steps):
            # Process each layer
            for target_layer in LAYERS:
                # Initialize inputs for this layer
                layer_inputs = {cell_type: np.zeros(grid_shape) for cell_type in CELL_TYPES}
                
                # Add pre-computed thalamic inputs to this layer
                for target_cell in CELL_TYPES:
                    layer_inputs[target_cell] += thalamic_inputs[target_layer][target_cell]
                
                # Add inputs from all layers to this layer
                for source_layer in LAYERS:
                    for source_cell in CELL_TYPES:
                        # Get the flattened rates from source layer
                        source_rates = self.layers[source_layer].r[source_cell].flatten()
                        
                        # Compute input to each cell type in target layer
                        for target_cell in CELL_TYPES:
                            # Skip SST->SST which doesn't exist in the base model
                            if source_cell == 'SST' and target_cell == 'SST':
                                continue
                                
                            # Compute input
                            input_curr = self.connectivity.compute_input(
                                source_cell, target_cell, source_rates,
                                source_layer=source_layer, target_layer=target_layer
                            )
                            
                            # Add to the target layer inputs
                            layer_inputs[target_cell] += input_curr.reshape(grid_shape)
                
                # Update this layer
                self.layers[target_layer].update(layer_inputs)
        
        # Return current activities
        return self.get_layer_activities() 