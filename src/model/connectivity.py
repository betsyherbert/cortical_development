"""Connectivity module for the cortical circuit simulation."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .config import (
    GRID_SIZE, CONNECTIVITY_PARAMS, CELL_TYPES, CONNECTIONS,
    LAYER_CONNECTIVITY_PARAMS, LAYER_CONNECTIONS, LAYERS
)


class ConnectivityProfile:
    """
    Handles the computation and caching of spatial connectivity profiles.
    
    This class efficiently computes 2D Gaussian profiles for neural connectivity
    and caches results to avoid redundant calculations.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE):
        """
        Initialize connectivity profiles.
        
        Args:
            grid_size: Size of the square grid
        """
        self.grid_size = grid_size
        self._cache = {}  # Cache for computed Gaussian profiles
        self._matrix_cache = {}  # Cache for computed weight matrices
        
        # Pre-compute coordinate meshgrid once for efficiency
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
        
        # Pre-compute center coordinates
        self.center = (self.grid_size // 2, self.grid_size // 2)
        
        # Pre-compute squared distances from center for common case
        x, y = self.coords
        self.center_d_squared = (x - self.center[0])**2 + (y - self.center[1])**2
    
    def gaussian_profile(self, sigma: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Compute a 2D Gaussian profile with efficient caching.
        
        Args:
            sigma: Width of the Gaussian
            center: Optional center coordinates (x, y). If None, uses grid center.
            
        Returns:
            2D array containing the normalized Gaussian profile
        """
        # Use cached profile if available
        cache_key = (sigma, center if center else 'center')
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Use pre-computed distance matrix for center case
        if center is None or center == self.center:
            d_squared = self.center_d_squared
        else:
            # Calculate squared distance from specified center
            x, y = self.coords
            d_squared = (x - center[0])**2 + (y - center[1])**2
        
        # Compute Gaussian profile
        profile = np.exp(-0.5 * d_squared / sigma**2)
        
        # Normalize to sum to 1
        profile /= profile.sum()
        
        # Cache the result
        self._cache[cache_key] = profile
        return profile

    def compute_weight_matrix(self, amplitude: float, sigma: float,
                            source_size: Tuple[int, int],
                            target_size: Tuple[int, int]) -> np.ndarray:
        """
        Compute weight matrix between two neural populations.
        
        Args:
            amplitude: Connection strength
            sigma: Width of the Gaussian profile
            source_size: Size of source population grid
            target_size: Size of target population grid
            
        Returns:
            2D connection weight matrix (target_neurons x source_neurons)
        """
        # Cache key for this specific weight matrix
        cache_key = (amplitude, sigma, source_size, target_size)
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]
            
        # Get Gaussian profile
        profile = self.gaussian_profile(sigma)
        
        # Special case for same-size grids (most common case)
        if source_size == target_size == (self.grid_size, self.grid_size):
            # Reshape to connection matrix
            source_neurons = self.grid_size * self.grid_size
            target_neurons = source_neurons
            
            # Create toeplitz-like connectivity matrix for 2D grid
            W = np.zeros((target_neurons, source_neurons))
            
            for i in range(target_neurons):
                # Calculate target neuron's position
                y_tgt, x_tgt = i // self.grid_size, i % self.grid_size
                
                # Shift the profile to be centered at this target neuron
                shifted_profile = np.roll(profile, (x_tgt - self.center[0], y_tgt - self.center[1]), axis=(0, 1))
                
                # Flatten and store in weight matrix
                W[i, :] = shifted_profile.flatten()
            
            # Scale by amplitude
            W *= amplitude
            
        else:
            # For different sized grids, compute the full matrix (less common case)
            source_neurons = source_size[0] * source_size[1]
            target_neurons = target_size[0] * target_size[1]
            
            W = np.zeros((target_neurons, source_neurons))
            
            # Less optimized implementation for different grid sizes
            for i in range(target_neurons):
                for j in range(source_neurons):
                    # Calculate positions
                    y_tgt, x_tgt = i // target_size[0], i % target_size[0]
                    y_src, x_src = j // source_size[0], j % source_size[0]
                    
                    # Calculate distance
                    dist_sq = (x_tgt - x_src)**2 + (y_tgt - y_src)**2
                    
                    # Compute weight using Gaussian profile
                    W[i, j] = amplitude * np.exp(-0.5 * dist_sq / sigma**2)
            
            # Normalize each row
            row_sums = W.sum(axis=1, keepdims=True)
            W = W / row_sums
        
        # Cache the computed matrix
        self._matrix_cache[cache_key] = W
        return W


class LayerConnectivity:
    """
    Manages connectivity matrices for all layer connections.
    
    This class handles the creation and updating of weight matrices for connections
    between different neural populations.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE):
        """
        Initialize connectivity matrices for all layer connections.
        
        Args:
            grid_size: Size of the square grid
        """
        self.grid_size = grid_size
        self.profile = ConnectivityProfile(grid_size)
        
        # Initialize with default parameters
        self.default_params = CONNECTIVITY_PARAMS.copy()
        self.layer_params = LAYER_CONNECTIVITY_PARAMS.copy()
        
        # Weight matrices dictionary
        # Format: (source_layer, source_cell, target_layer, target_cell)
        self.W = {}
        
        # Initialize weight matrices
        self.update_weights()
    
    def update_weights(self, params: Optional[Dict[str, Dict[str, float]]] = None, 
                     layer_params: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """
        Update all weight matrices based on provided parameters.
        
        Args:
            params: Optional dictionary of traditional connection parameters to update
            layer_params: Optional dictionary of layer-specific connection parameters to update
        """
        # Update traditional parameters if provided
        if params:
            for conn_key, conn_params in params.items():
                if conn_key in self.default_params:
                    self.default_params[conn_key].update(conn_params)
        
        # Update layer-specific parameters if provided
        if layer_params:
            for conn_key, conn_params in layer_params.items():
                if conn_key in self.layer_params:
                    self.layer_params[conn_key].update(conn_params)
        
        # Compute the grid size tuple
        size = (self.grid_size, self.grid_size)
        
        # Build all layer-specific connection matrices
        for conn in LAYER_CONNECTIONS:
            source_layer, source_cell, target_layer, target_cell = conn
            
            # Generate the connection key format {source_layer}_{source_cell}_to_{target_layer}_{target_cell}
            # For thalamic connections, the format is thalamus_to_{target_layer}_{target_cell}
            if source_layer == 'thalamus':
                conn_key = f'thalamus_to_{target_layer}_{target_cell}'
            else:
                conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
            
            # Check if this connection has parameters defined
            if conn_key in self.layer_params:
                params = self.layer_params[conn_key]
                # Create weight matrix
                self.W[conn] = self.profile.compute_weight_matrix(
                    params['amplitude'], params['sigma'], size, size
                )
        
        # For backward compatibility, also build traditional connections
        for source, target in CONNECTIONS:
            conn_key = f'{source}_to_{target}'
            if conn_key in self.default_params:
                # Store under a special key for backward compatibility
                self.W[('any', source, 'any', target)] = self.profile.compute_weight_matrix(
                    self.default_params[conn_key]['amplitude'],
                    self.default_params[conn_key]['sigma'],
                    size, size
                )
                
        # Also add traditional thalamic connections for backward compatibility
        for target in CELL_TYPES:
            conn_key = f'thalamus_to_{target}'
            if conn_key in self.default_params:
                self.W[('thalamus', None, 'any', target)] = self.profile.compute_weight_matrix(
                    self.default_params[conn_key]['amplitude'],
                    self.default_params[conn_key]['sigma'],
                    size, size
                )
    
    def compute_input(self, source: str, target: str, rates: np.ndarray,
                    source_layer: str = 'any', target_layer: str = 'any') -> np.ndarray:
        """
        Compute input from one population to another.
        
        Args:
            source: Source population type ('E', 'SST', 'PV', 'thalamus')
            target: Target population type ('E', 'SST', 'PV')
            rates: Firing rates of source population
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            target_layer: Target layer ('L23', 'L4', 'L5')
            
        Returns:
            Input currents to target population
        """
        # Create connection key tuple
        conn_key = (source_layer, source, target_layer, target)
        
        # Direct lookup with the exact layer-specific key
        if conn_key in self.W:
            return self.W[conn_key] @ rates
        
        # Prepare fallback keys in order of specificity
        fallback_keys = [
            (source_layer, source, 'any', target),
            ('any', source, target_layer, target),
            ('any', source, 'any', target)
        ]
        
        if source == 'thalamus':
            fallback_keys = [('thalamus', None, 'any', target)]
            
        # Try each fallback key
        for key in fallback_keys:
            if key in self.W:
                return self.W[key] @ rates
                
        # If no connection exists, return zeros
        return np.zeros_like(rates)
    
    def get_connection_strength(self, source_layer: str, source_cell: str, 
                              target_layer: str, target_cell: str) -> float:
        """
        Get the current connection strength between two populations.
        
        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            
        Returns:
            Connection amplitude or 0 if connection doesn't exist
        """
        # Generate the connection key
        if source_layer == 'thalamus':
            conn_key = f'thalamus_to_{target_layer}_{target_cell}'
        else:
            conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
        
        # Check if this connection has parameters defined
        if conn_key in self.layer_params:
            return self.layer_params[conn_key]['amplitude']
        
        return 0.0

    def set_connection_strength(self, source_layer: str, source_cell: str, 
                              target_layer: str, target_cell: str, 
                              amplitude: float) -> None:
        """
        Set the connection strength between two populations.
        
        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            amplitude: Connection strength
        """
        # Generate the connection key
        if source_layer == 'thalamus':
            conn_key = f'thalamus_to_{target_layer}_{target_cell}'
        else:
            conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
        
        # Check if this connection has parameters defined
        if conn_key in self.layer_params:
            # Update the parameter
            self.layer_params[conn_key]['amplitude'] = amplitude
            
            # Update the weight matrix
            size = (self.grid_size, self.grid_size)
            sigma = self.layer_params[conn_key]['sigma']
            
            # Create the connection tuple key
            tuple_key = (source_layer, source_cell, target_layer, target_cell)
            
            # Update the weight matrix
            self.W[tuple_key] = self.profile.compute_weight_matrix(
                amplitude, sigma, size, size
            ) 