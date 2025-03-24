"""Neural dynamics module for the cortical circuit simulation."""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from .connectivity import LayerConnectivity
from .config import (
    GRID_SIZE, DT, NOISE_AMPLITUDE,
    INTEGRATION_STEPS, CELL_TYPES, LAYERS
)

# Default firing threshold (0 for standard ReLU)
FIRING_THRESHOLD = 0.0


class NeuralLayer:
    """
    Implements neural population dynamics for a single cortical layer.
    
    Each layer contains three cell types: excitatory (E), somatostatin-expressing (SST),
    and parvalbumin-expressing (PV) interneurons, each with their own dynamics.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE, dt: float = DT, threshold: float = FIRING_THRESHOLD):
        """
        Initialize a neural layer with specified grid size and time constant.
        
        Args:
            grid_size: Size of the square grid
            dt: Time step in milliseconds
            threshold: Firing threshold (default value, can be overridden per cell type)
        """
        self.grid_size = grid_size
        # Initialize separate time constants for each cell type
        self.tau = {
            'E': 80.0,    # Default time constant for E cells
            'SST': 60.0,  # Default time constant for SST cells
            'PV': 40.0    # Default time constant for PV cells
        }
        # Initialize separate firing thresholds for each cell type
        self.threshold = {
            'E': threshold,
            'SST': threshold,
            'PV': threshold
        }
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

    def relu(self, x: np.ndarray, threshold: float = FIRING_THRESHOLD) -> np.ndarray:
        """ReLU activation function with threshold: max(0, x - threshold)."""
        # More efficient single-step calculation
        return np.maximum(0, x - threshold)

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
                # Using cell-type specific time constant
                dV = (-self.V[cell_type] + inputs[cell_type] + noise[cell_type]) * (self.dt / self.tau[cell_type])
                self.V[cell_type] += dV
                
                # Update firing rates with ReLU activation and cell-type specific threshold
                self.r[cell_type] = self.relu(self.V[cell_type], self.threshold[cell_type])
        
        # Return current firing rates
        return self.r

    def reset(self) -> None:
        """Reset neural state variables to initial state, while preserving parameters."""
        # Only reset state variables (V and r), not parameters (tau and threshold)
        for cell_type in CELL_TYPES:
            self.V[cell_type].fill(0)
            self.r[cell_type].fill(0)

    def set_time_constant(self, cell_type: str, tau: float) -> None:
        """
        Set the membrane time constant for a specific cell type.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            tau: New time constant value in milliseconds
        """
        if cell_type in CELL_TYPES:
            self.tau[cell_type] = tau
            
    def get_time_constants(self) -> Dict[str, float]:
        """
        Get current time constant values for all cell types.
        
        Returns:
            Dictionary mapping cell types to their time constants
        """
        return self.tau.copy()
        
    def set_firing_threshold(self, cell_type: str, threshold: float) -> None:
        """
        Set the firing threshold for a specific cell type.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            threshold: New threshold value
        """
        if cell_type in CELL_TYPES:
            self.threshold[cell_type] = threshold
            
    def get_firing_thresholds(self) -> Dict[str, float]:
        """
        Get current firing threshold values for all cell types.
        
        Returns:
            Dictionary mapping cell types to their firing thresholds
        """
        return self.threshold.copy()


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
        
        # Run multiple steps for smoother dynamics
        for _ in range(n_steps):
            # Process each layer
            for target_layer in LAYERS:
                # Initialize inputs for this layer
                layer_inputs = {cell_type: np.zeros(grid_shape) for cell_type in CELL_TYPES}
                
                # Add thalamic inputs
                for target_cell in CELL_TYPES:
                    conn_key = f'thalamus_to_{target_layer}_{target_cell}'
                    if conn_key in self.connectivity.layer_params:
                        layer_inputs[target_cell] += (
                            self.connectivity.layer_params[conn_key]['amplitude'] * self.thalamus
                        )
                
                # Add inputs from all layers
                for source_layer in LAYERS:
                    for source_cell in CELL_TYPES:
                        source_rates = self.layers[source_layer].r[source_cell].flatten()
                        
                        for target_cell in CELL_TYPES:
                            # Skip SST->SST which doesn't exist
                            if source_cell == 'SST' and target_cell == 'SST':
                                continue
                                
                            input_curr = self.connectivity.compute_input(
                                source_cell, target_cell, source_rates,
                                source_layer=source_layer, target_layer=target_layer
                            )
                            layer_inputs[target_cell] += input_curr.reshape(grid_shape)
                
                # Update this layer
                self.layers[target_layer].update(layer_inputs)
        
        # Return current activities
        return self.get_layer_activities()
        
    def set_time_constant(self, cell_type: str, tau: float) -> None:
        """
        Set the membrane time constant for a specific cell type across all layers.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            tau: New time constant value in milliseconds
        """
        for layer in self.layers.values():
            layer.set_time_constant(cell_type, tau)
            
    def get_time_constants(self) -> Dict[str, float]:
        """
        Get current time constant values for all cell types (from L23 layer).
        
        Returns:
            Dictionary mapping cell types to their time constants
        """
        # Return values from L23 layer as they're synced across all layers
        return self.layers['L23'].get_time_constants()
        
    def set_firing_threshold(self, cell_type: str, threshold: float) -> None:
        """
        Set the firing threshold for a specific cell type across all layers.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV') 
            threshold: New threshold value
        """
        for layer in self.layers.values():
            layer.set_firing_threshold(cell_type, threshold)
            
    def get_firing_thresholds(self) -> Dict[str, float]:
        """
        Get current firing threshold values for all cell types (from L23 layer).
        
        Returns:
            Dictionary mapping cell types to their firing thresholds
        """
        # Return values from L23 layer as they're synced across all layers
        return self.layers['L23'].get_firing_thresholds() 