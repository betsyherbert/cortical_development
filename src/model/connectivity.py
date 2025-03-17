import numpy as np
from typing import Tuple, Optional

class ConnectivityProfile:
    def __init__(self, grid_size: int = 10):
        """
        Initialize connectivity profiles for the neural circuit.
        
        Args:
            grid_size: Size of the square grid (default: 10)
        """
        self.grid_size = grid_size
        self._cache = {}  # Cache for computed Gaussian profiles
        
        # Create coordinate meshgrid for distance calculations
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
    
    def gaussian_profile(self, sigma: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Compute a 2D Gaussian profile.
        
        Args:
            sigma: Width of the Gaussian
            center: Optional center coordinates (x, y). If None, uses grid center.
            
        Returns:
            2D array containing the Gaussian profile
        """
        # Use cached profile if available
        cache_key = (sigma, center if center else 'center')
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if center is None:
            center = (self.grid_size // 2, self.grid_size // 2)
        
        x, y = self.coords
        d_squared = (x - center[0])**2 + (y - center[1])**2
        profile = np.exp(-0.5 * d_squared / sigma**2)
        
        # Normalize
        profile /= profile.sum()
        
        # Cache the result
        self._cache[cache_key] = profile
        return profile

    def compute_weight_matrix(self, amplitude: float, sigma: float,
                            source_size: Tuple[int, int],
                            target_size: Tuple[int, int]) -> np.ndarray:
        """
        Compute weight matrix between two populations.
        
        Args:
            amplitude: Connection strength
            sigma: Width of the Gaussian profile
            source_size: Size of source population grid
            target_size: Size of target population grid
            
        Returns:
            Weight matrix shaped (target_neurons, source_neurons)
        """
        weights = np.zeros((np.prod(target_size), np.prod(source_size)))
        
        for i in range(target_size[0]):
            for j in range(target_size[1]):
                target_idx = i * target_size[1] + j
                profile = self.gaussian_profile(sigma, center=(i, j))
                weights[target_idx, :] = amplitude * profile.flatten()
        
        return weights

class LayerConnectivity:
    def __init__(self, grid_size: int = 10):
        """
        Initialize connectivity matrices for all layer connections.
        
        Args:
            grid_size: Size of the square grid (default: 10)
        """
        self.grid_size = grid_size
        self.profile = ConnectivityProfile(grid_size)
        
        # Default connection parameters
        self.default_params = {
            'E_to_E': {'amplitude': 0.2, 'sigma': 2.0},
            'E_to_SST': {'amplitude': 0.1, 'sigma': 2.0},
            'E_to_PV': {'amplitude': 0.15, 'sigma': 2.0},
            'SST_to_E': {'amplitude': -0.1, 'sigma': 3.0},
            'SST_to_PV': {'amplitude': -0.05, 'sigma': 3.0},
            'PV_to_E': {'amplitude': -0.15, 'sigma': 1.5},
            'PV_to_SST': {'amplitude': -0.1, 'sigma': 1.5},
            'PV_to_PV': {'amplitude': -0.1, 'sigma': 1.5},
            'thalamus_to_E': {'amplitude': 0.15, 'sigma': 2.0},
            'thalamus_to_SST': {'amplitude': 0.1, 'sigma': 2.0},
            'thalamus_to_PV': {'amplitude': 0.12, 'sigma': 2.0}
        }
        
        # Initialize weight matrices
        self.update_weights()
    
    def update_weights(self, params: Optional[dict] = None) -> None:
        """
        Update all weight matrices based on provided parameters.
        
        Args:
            params: Optional dictionary of connection parameters to update
        """
        if params:
            self.default_params.update(params)
        
        # Compute all weight matrices
        size = (self.grid_size, self.grid_size)
        
        # Within-layer connectivity
        self.W = {
            'E_to_E': self.profile.compute_weight_matrix(
                self.default_params['E_to_E']['amplitude'],
                self.default_params['E_to_E']['sigma'],
                size, size
            ),
            'E_to_SST': self.profile.compute_weight_matrix(
                self.default_params['E_to_SST']['amplitude'],
                self.default_params['E_to_SST']['sigma'],
                size, size
            ),
            # ... (similar for other connections)
        }
        
    def compute_input(self, source: str, target: str, rates: np.ndarray) -> np.ndarray:
        """
        Compute input from one population to another.
        
        Args:
            source: Source population type ('E', 'SST', 'PV', 'thalamus')
            target: Target population type ('E', 'SST', 'PV')
            rates: Firing rates of source population
            
        Returns:
            Input currents to target population
        """
        connection_key = f'{source}_to_{target}'
        if connection_key not in self.W:
            return np.zeros_like(rates)
            
        return self.W[connection_key] @ rates.flatten() 