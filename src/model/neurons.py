"""Neural dynamics module for the cortical circuit simulation.

This module implements the core neural dynamics for the cortical circuit model.
It provides classes for simulating individual neural layers and their interactions,
with support for different cell types and their specific properties.
"""

from typing import Dict
import numpy as np
from .connectivity import LayerConnectivity
from .config import (
    GRID_SIZE, DT,
    INTEGRATION_STEPS, CELL_TYPES, LAYERS,
    INITIAL_BACKGROUND_INPUT
)


class NeuralLayer:
    """
    Implements neural population dynamics for a single cortical layer.
    
    Each layer contains three cell types:
    - Excitatory (E): Regular spiking pyramidal neurons
    - Somatostatin-expressing (SST): Inhibitory interneurons targeting dendrites
    - Parvalbumin-expressing (PV): Fast-spiking inhibitory interneurons
    
    Each cell type has its own dynamics characterized by:
    - Membrane time constant (tau)
    - Input gain
    - Firing rate nonlinearity (ReLU)
    - Constant background input
    """
    
    def __init__(self, grid_size: int = GRID_SIZE, dt: float = DT, gain: float = 1.0):
        """
        Initialize a neural layer with specified grid size and time constant.
        
        Args:
            grid_size: Size of the square grid
            dt: Time step in milliseconds
            gain: Gain parameter (default value, can be overridden per cell type)
        """
        self.grid_size = grid_size
        # Initialize separate time constants for each cell type
        self.tau = {
            'E': 80.0,    # Default time constant for E cells
            'SST': 60.0,  # Default time constant for SST cells
            'PV': 40.0    # Default time constant for PV cells
        }
        # Initialize separate gains for each cell type
        self.gain = {
            'E': gain,
            'SST': gain,
            'PV': gain
        }
        # Initialize background input for each cell type
        self.background_input = {
            'E': INITIAL_BACKGROUND_INPUT['E'],
            'SST': INITIAL_BACKGROUND_INPUT['SST'],
            'PV': INITIAL_BACKGROUND_INPUT['PV']
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

    def relu(self, x: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """ReLU activation function with gain: max(0, gain * x)."""
        return np.maximum(0, gain * x)

    def update(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update neural dynamics for one time step.
        
        Args:
            inputs: Dictionary mapping cell types to their input currents
            
        Returns:
            Dictionary of updated firing rates for each cell type
        """
        # Update membrane potentials using Euler method (vectorized)
        for cell_type in CELL_TYPES:
            if cell_type in inputs:
                # Add constant background input to the input current
                total_input = inputs[cell_type] + self.background_input[cell_type]
                
                # Using cell-type specific time constant
                dV = (-self.V[cell_type] + total_input) * (self.dt / self.tau[cell_type])
                self.V[cell_type] += dV
                
                # Update firing rates with ReLU activation and cell-type specific gain
                self.r[cell_type] = self.relu(self.V[cell_type], self.gain[cell_type])
        
        # Return current firing rates
        return self.r

    def reset(self) -> None:
        """Reset neural state variables to initial state, while preserving parameters."""
        # Reset state variables (V and r)
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
        
    def set_gain(self, cell_type: str, gain: float) -> None:
        """
        Set the gain for a specific cell type.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            gain: New gain value
        """
        if cell_type in CELL_TYPES:
            self.gain[cell_type] = gain
            
    def get_gains(self) -> Dict[str, float]:
        """
        Get current gain values for all cell types.
        
        Returns:
            Dictionary mapping cell types to their gains
        """
        return self.gain.copy()
        
    def set_background_input(self, cell_type: str, value: float) -> None:
        """
        Set the background input for a specific cell type.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            value: New background input value
        """
        if cell_type in CELL_TYPES:
            self.background_input[cell_type] = value
            
    def get_background_input(self, cell_type: str) -> float:
        """
        Get the current background input for a specific cell type.
        
        Args:
            cell_type: The cell type to query ('E', 'SST', or 'PV')
            
        Returns:
            Background input value
        """
        if cell_type in CELL_TYPES:
            return self.background_input[cell_type]
        return 0.0


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
                
                # Add thalamic inputs - using weight matrices to properly apply strength scaling
                for target_cell in CELL_TYPES:
                    # Create the connection tuple key - for thalamic connections, source_cell is None
                    conn_key = ('thalamus', None, target_layer, target_cell)
                    if conn_key in self.connectivity.W:
                        # Get the weight matrix with scaling already applied
                        weight_matrix = self.connectivity.W[conn_key]
                        
                        # Apply the weight matrix to thalamic rates
                        thalamic_rates = self.thalamus.flatten()
                        input_curr = weight_matrix @ thalamic_rates
                        layer_inputs[target_cell] += input_curr.reshape(grid_shape)
                
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
        Get current time constant values for all cell types (from L4 layer).
        
        Returns:
            Dictionary mapping cell types to their time constants
        """
        return self.layers['L4'].get_time_constants()
        
    def set_gain(self, cell_type: str, gain: float) -> None:
        """
        Set the gain for a specific cell type across all layers.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            gain: New gain value
        """
        for layer in self.layers.values():
            layer.set_gain(cell_type, gain)
            
    def get_gains(self) -> Dict[str, float]:
        """
        Get current gain values for all cell types (from L4 layer).
        
        Returns:
            Dictionary mapping cell types to their gains
        """
        return self.layers['L4'].get_gains()
        
    def set_background_input(self, cell_type: str, value: float) -> None:
        """
        Set background input for a specific cell type across all layers.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            value: New background input value
        """
        for layer in self.layers.values():
            layer.set_background_input(cell_type, value)
            
    def get_background_input(self, cell_type: str) -> float:
        """
        Get current background input for a specific cell type (from L4 layer).
        
        Args:
            cell_type: The cell type to query ('E', 'SST', or 'PV')
            
        Returns:
            Background input value
        """
        return self.layers['L4'].get_background_input(cell_type) 