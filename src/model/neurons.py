import numpy as np
from typing import Tuple, List, Optional

class NeuralLayer:
    def __init__(self, grid_size: int = 10, tau: float = 50.0):
        """
        Initialize a neural layer with specified grid size and time constant.
        
        Args:
            grid_size: Size of the square grid (default: 10)
            tau: Time constant in milliseconds (default: 50.0)
        """
        self.grid_size = grid_size
        self.tau = tau
        self.dt = 1.0  # 1ms integration time step
        
        # Initialize membrane potentials and firing rates
        self.V_e = np.zeros((grid_size, grid_size))  # Excitatory
        self.V_sst = np.zeros((grid_size, grid_size))  # SST
        self.V_pv = np.zeros((grid_size, grid_size))  # PV
        
        # Initialize firing rates
        self.r_e = np.zeros_like(self.V_e)
        self.r_sst = np.zeros_like(self.V_sst)
        self.r_pv = np.zeros_like(self.V_pv)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def update(self, I_e: np.ndarray, I_sst: np.ndarray, I_pv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update neural dynamics for one time step.
        
        Args:
            I_e: Input current to excitatory neurons
            I_sst: Input current to SST neurons
            I_pv: Input current to PV neurons
            
        Returns:
            Tuple of firing rates (r_e, r_sst, r_pv)
        """
        # Update membrane potentials using Euler method
        dV_e = (-self.V_e + I_e) * (self.dt / self.tau)
        dV_sst = (-self.V_sst + I_sst) * (self.dt / self.tau)
        dV_pv = (-self.V_pv + I_pv) * (self.dt / self.tau)
        
        self.V_e += dV_e
        self.V_sst += dV_sst
        self.V_pv += dV_pv
        
        # Update firing rates
        self.r_e = self.relu(self.V_e)
        self.r_sst = self.relu(self.V_sst)
        self.r_pv = self.relu(self.V_pv)
        
        return self.r_e, self.r_sst, self.r_pv

    def reset(self) -> None:
        """Reset all state variables to zero."""
        self.V_e.fill(0)
        self.V_sst.fill(0)
        self.V_pv.fill(0)
        self.r_e.fill(0)
        self.r_sst.fill(0)
        self.r_pv.fill(0)

class CorticalCircuit:
    def __init__(self, grid_size: int = 10):
        """
        Initialize the full cortical circuit with all layers.
        
        Args:
            grid_size: Size of the square grid (default: 10)
        """
        # Initialize cortical layers
        self.L23 = NeuralLayer(grid_size)
        self.L4 = NeuralLayer(grid_size)
        self.L5 = NeuralLayer(grid_size)
        
        # Initialize thalamic layer (only firing rates, no dynamics)
        self.thalamus = np.zeros((grid_size, grid_size))
        
        self.grid_size = grid_size
    
    def get_layer_activities(self) -> dict:
        """
        Get current activities of all populations.
        
        Returns:
            Dictionary containing firing rates of all populations
        """
        return {
            'L23': {'E': self.L23.r_e, 'SST': self.L23.r_sst, 'PV': self.L23.r_pv},
            'L4': {'E': self.L4.r_e, 'SST': self.L4.r_sst, 'PV': self.L4.r_pv},
            'L5': {'E': self.L5.r_e, 'SST': self.L5.r_sst, 'PV': self.L5.r_pv},
            'thalamus': self.thalamus
        }
    
    def reset(self) -> None:
        """Reset all layers to their initial state."""
        self.L23.reset()
        self.L4.reset()
        self.L5.reset()
        self.thalamus.fill(0) 