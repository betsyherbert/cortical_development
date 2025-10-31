"""Connectivity module for the cortical circuit simulation."""

from typing import Dict, Optional, Tuple
import numpy as np

from .config import (
    GRID_SIZE, LAYER_CONNECTIVITY_PARAMS,
    INITIAL_STRENGTH_SCALING, INITIAL_SPARSITY
)

class ConnectivityProfile:
    """
    Handles the computation and caching of spatial connectivity profiles.
    
    This class efficiently computes and caches 2D Gaussian profiles and weight matrices
    for neural connectivity. It uses pre-computed coordinate grids and distance matrices
    to optimize repeated calculations.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE):
        """
        Initialize connectivity profiles with optimized caching.
        
        Args:
            grid_size: Size of the square grid
        """
        self.grid_size = grid_size
        self._profile_cache = {}  # Cache for Gaussian profiles
        self._matrix_cache = {}  # Cache for weight matrices
        
        # Pre-compute coordinate meshgrid and center coordinates
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
        self.center = (grid_size // 2, grid_size // 2)
        
        # Pre-compute squared distances from center
        self.center_d_squared = (x - self.center[0])**2 + (y - self.center[1])**2
        
        # Pre-compute common grid sizes for weight matrices
        self.common_size = (grid_size * grid_size, grid_size * grid_size)
    
    def _compute_gaussian(self, d_squared: np.ndarray, sigma: float) -> np.ndarray:
        """
        Compute Gaussian profile from squared distances.
        
        Args:
            d_squared: Array of squared distances
            sigma: Width of the Gaussian
            
        Returns:
            Normalized Gaussian profile
        """
        profile = np.exp(-0.5 * d_squared / sigma**2)
        return profile / profile.sum()
    
    def gaussian_profile(self, sigma: float, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get a cached 2D Gaussian profile or compute if not available.
        
        Args:
            sigma: Width of the Gaussian
            center: Optional center coordinates (x, y). If None, uses grid center.
            
        Returns:
            2D array containing the normalized Gaussian profile
        """
        # Use cached profile if available
        cache_key = (sigma, center if center else 'center')
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]
        
        # Use pre-computed distance matrix for center case
        if center is None or center == self.center:
            d_squared = self.center_d_squared
        else:
            # Calculate squared distance from specified center
            x, y = self.coords
            d_squared = (x - center[0])**2 + (y - center[1])**2
        
        # Compute and cache Gaussian profile
        profile = self._compute_gaussian(d_squared, sigma)
        self._profile_cache[cache_key] = profile
        return profile

    def compute_weight_matrix(self, amplitude: float, sigma: float,
                            source_size: Tuple[int, int],
                            target_size: Tuple[int, int]) -> np.ndarray:
        """
        Get a cached weight matrix or compute if not available.
        
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
        
        # Optimize for the common case of same-size grids
        if source_size == target_size == (self.grid_size, self.grid_size):
            W = self._compute_same_size_weights(profile, amplitude)
        else:
            W = self._compute_different_size_weights(profile, amplitude, source_size, target_size)
        
        # Cache and return the weight matrix
        self._matrix_cache[cache_key] = W
        return W
    
    def _compute_same_size_weights(self, profile: np.ndarray, amplitude: float) -> np.ndarray:
        """Compute weight matrix for same-size grids."""
        W = np.zeros(self.common_size)
        
        for i in range(self.grid_size * self.grid_size):
            # Calculate target neuron's position
            y_tgt, x_tgt = i // self.grid_size, i % self.grid_size
            
            # Shift the profile to be centered at this target neuron
            shifted_profile = np.roll(profile, 
                                    (x_tgt - self.center[0], y_tgt - self.center[1]), 
                                    axis=(0, 1))
            
            # Store in weight matrix
            W[i, :] = shifted_profile.flatten()
        
        return W * amplitude
    
    def _compute_different_size_weights(self, profile: np.ndarray, amplitude: float,
                                      source_size: Tuple[int, int],
                                      target_size: Tuple[int, int]) -> np.ndarray:
        """Compute weight matrix for different-size grids."""
        source_neurons = source_size[0] * source_size[1]
        target_neurons = target_size[0] * target_size[1]
        
        W = np.zeros((target_neurons, source_neurons))
        
        for i in range(target_neurons):
            for j in range(source_neurons):
                # Calculate positions
                y_tgt, x_tgt = i // target_size[0], i % target_size[0]
                y_src, x_src = j // source_size[0], j % source_size[0]
                
                # Calculate distance and weight
                dist_sq = (x_tgt - x_src)**2 + (y_tgt - y_src)**2
                W[i, j] = amplitude * np.exp(-0.5 * dist_sq / profile.shape[0]**2)
        
        # Normalize rows
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / row_sums
        
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
        
        # Initialize with layer parameters
        self.layer_params = LAYER_CONNECTIVITY_PARAMS.copy()
        
        # Weight matrices dictionary
        # Format: (source_layer, source_cell, target_layer, target_cell)
        self.W = {}
        
        # Initialize strength scaling factors and sparsity factors
        self.strength_scaling = INITIAL_STRENGTH_SCALING.copy()
        self.sparsity = INITIAL_SPARSITY.copy()
        
        # Cache for sparsity masks to avoid regenerating them every time
        self._sparsity_masks = {}
        
        # Use the global random state for consistency with centralized seed management
        
        # Initialize weight matrices
        self.update_weights()
    
    def _get_sparsity_mask(self, shape, cell_type_for_scaling):
        """
        Get a cached sparsity mask or generate a new one if needed.
        
        Args:
            shape: Shape of the weight matrix
            cell_type_for_scaling: Cell type for which to create the mask
            
        Returns:
            Binary mask for applying sparsity
        """
        sparsity_level = self.sparsity[cell_type_for_scaling]
        cache_key = (shape, cell_type_for_scaling, sparsity_level)
        
        # Return cached mask if available
        if cache_key in self._sparsity_masks:
            return self._sparsity_masks[cache_key]
        
        # Generate a new mask if sparsity is less than 1.0
        if sparsity_level < 1.0:
            mask = np.random.random(shape) < sparsity_level
        else:
            # For full connectivity, use a mask of all ones (more efficient)
            mask = np.ones(shape, dtype=bool)
            
        # Cache the mask
        self._sparsity_masks[cache_key] = mask
        return mask
    
    def update_weights(self, layer_params: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """
        Update weight matrices based on connection parameters.
        
        Args:
            layer_params: Optional dictionary of layer-specific connection parameters
        """
        if layer_params is not None:
            self.layer_params = layer_params.copy()
        
        # Clear existing weight matrices
        self.W.clear()
        
        # Create weight matrices for each layer-specific connection
        for conn_key, params in self.layer_params.items():
            # Split the connection key into its components
            # Format: {source_layer}_{source_cell}_to_{target_layer}_{target_cell}
            parts = conn_key.split('_to_')
            if len(parts) != 2:
                continue  # Skip invalid connection keys
                
            source_part, target_part = parts
            
            # Split source and target parts
            source_parts = source_part.split('_')
            target_parts = target_part.split('_')
            
            if len(source_parts) != 2 or len(target_parts) != 2:
                continue  # Skip invalid connection keys
                
            source_layer, source_cell = source_parts
            target_layer, target_cell = target_parts
            
            # Handle thalamic connections
            if source_layer == 'thalamus':
                source_cell = None
                cell_type_for_scaling = 'thalamus'
            else:
                cell_type_for_scaling = source_cell
            
            # Apply strength scaling factor
            scaled_amplitude = params['amplitude'] * self.strength_scaling[cell_type_for_scaling]
            
            # Create weight matrix
            weight_matrix = self.profile.compute_weight_matrix(
                scaled_amplitude,
                params['sigma'],
                (self.grid_size, self.grid_size),
                (self.grid_size, self.grid_size)
            )
            
            # Apply sparsity factor
            if self.sparsity[cell_type_for_scaling] < 1.0:
                # Get or create sparsity mask for this connection
                sparsity_mask = self._get_sparsity_mask(weight_matrix.shape, cell_type_for_scaling)
                weight_matrix = weight_matrix * sparsity_mask
            
            # Store weight matrix
            self.W[(source_layer, source_cell, target_layer, target_cell)] = weight_matrix
    
    def compute_input(self, source: str, target: str, rates: np.ndarray,
                    source_layer: str = 'any', target_layer: str = 'any') -> np.ndarray:
        """
        Compute input from one population to another.
        
        Args:
            source: Source population type ('E', 'SST', 'PV', 'thalamus')
            target: Target population type ('E', 'SST', 'PV')
            rates: Firing rates of source population
            source_layer: Layer of source population ('L23', 'L4', 'L5', 'thalamus', 'any')
            target_layer: Layer of target population ('L23', 'L4', 'L5', 'any')
            
        Returns:
            Input current from source to target population
        """
        # Find the layer-specific connection
        conn_key = (source_layer, source, target_layer, target)
        
        # If connection exists, use it
        if conn_key in self.W:
            weight_matrix = self.W[conn_key]
            input_current = weight_matrix @ rates
        else:
            input_current = np.zeros_like(rates)
            
        return input_current
    
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
            cell_type_for_scaling = 'thalamus'
        else:
            conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
            cell_type_for_scaling = source_cell
        
        # Create connection parameter entry if it doesn't exist
        if conn_key not in self.layer_params:
            # Determine appropriate sigma based on cell type patterns
            if source_cell == 'E' or source_layer == 'thalamus':
                sigma = 2.0  # Default for excitatory connections
            elif source_cell == 'SST':
                sigma = 3.0  # Default for SST connections (wider)
            elif source_cell == 'PV':
                sigma = 1.5  # Default for PV connections (narrower)
            else:
                sigma = 2.0  # Default fallback
                
            # Add the new connection parameters
            self.layer_params[conn_key] = {'amplitude': 0.0, 'sigma': sigma}
            
        # Update the parameter
        self.layer_params[conn_key]['amplitude'] = amplitude
            
        # Get the sigma value
        sigma = self.layer_params[conn_key]['sigma']
        
        # Apply strength scaling factor
        scaled_amplitude = amplitude * self.strength_scaling[cell_type_for_scaling]
            
        # Create the connection tuple key
        tuple_key = (source_layer, source_cell, target_layer, target_cell)
        
        # Create the weight matrix with scaled amplitude
        weight_matrix = self.profile.compute_weight_matrix(
            scaled_amplitude, sigma, (self.grid_size, self.grid_size), (self.grid_size, self.grid_size)
        )
        
        # Apply sparsity factor
        if self.sparsity[cell_type_for_scaling] < 1.0:
            # Get or create sparsity mask for this connection
            sparsity_mask = self._get_sparsity_mask(weight_matrix.shape, cell_type_for_scaling)
            weight_matrix = weight_matrix * sparsity_mask
            
        # Store weight matrix
        self.W[tuple_key] = weight_matrix

    def get_connection_sigma(self, source_layer: str, source_cell: str, 
                           target_layer: str, target_cell: str) -> float:
        """
        Get the current connection width (sigma) between two populations.
        
        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            
        Returns:
            Connection sigma or default value if connection doesn't exist
        """
        # Generate the connection key
        if source_layer == 'thalamus':
            conn_key = f'thalamus_to_{target_layer}_{target_cell}'
        else:
            conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
        
        # Check if this connection has parameters defined
        if conn_key in self.layer_params:
            return self.layer_params[conn_key]['sigma']
        
        # Determine default sigma based on cell type patterns if connection doesn't exist
        if source_cell == 'E' or source_layer == 'thalamus':
            return 2.0  # Default for excitatory connections
        elif source_cell == 'SST':
            return 3.0  # Default for SST connections (wider)
        elif source_cell == 'PV':
            return 1.5  # Default for PV connections (narrower)
        
        return 2.0  # Default fallback

    def set_connection_sigma(self, source_layer: str, source_cell: str, 
                           target_layer: str, target_cell: str, 
                           sigma: float) -> None:
        """
        Set the connection width (sigma) between two populations.
        
        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            sigma: Connection width (Gaussian sigma)
        """
        # Generate the connection key
        if source_layer == 'thalamus':
            conn_key = f'thalamus_to_{target_layer}_{target_cell}'
            cell_type_for_scaling = 'thalamus'
        else:
            conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
            cell_type_for_scaling = source_cell
        
        # Create connection parameter entry if it doesn't exist
        if conn_key not in self.layer_params:
            # Determine appropriate amplitude based on common patterns
            if source_cell == 'E' or source_layer == 'thalamus':
                amplitude = 0.2  # Default for excitatory
            elif source_cell == 'SST' or source_cell == 'PV':
                amplitude = -0.1  # Default for inhibitory
            else:
                amplitude = 0.0  # Default fallback
                
            # Add the new connection parameters
            self.layer_params[conn_key] = {'amplitude': amplitude, 'sigma': sigma}
        else:
            # Update the parameter
            self.layer_params[conn_key]['sigma'] = sigma
            
        # Get current raw amplitude
        amplitude = self.layer_params[conn_key]['amplitude']
        
        # Apply strength scaling factor
        scaled_amplitude = amplitude * self.strength_scaling[cell_type_for_scaling]
            
        # Create the connection tuple key
        tuple_key = (source_layer, source_cell, target_layer, target_cell)
        
        # Create the weight matrix with scaled amplitude
        weight_matrix = self.profile.compute_weight_matrix(
            scaled_amplitude, sigma, (self.grid_size, self.grid_size), (self.grid_size, self.grid_size)
        )
        
        # Apply sparsity factor
        if self.sparsity[cell_type_for_scaling] < 1.0:
            # Get or create sparsity mask for this connection
            sparsity_mask = self._get_sparsity_mask(weight_matrix.shape, cell_type_for_scaling)
            weight_matrix = weight_matrix * sparsity_mask
            
        # Store weight matrix
        self.W[tuple_key] = weight_matrix

    def get_strength_scaling(self, cell_type: str) -> float:
        """
        Get the current strength scaling factor for a cell type.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            
        Returns:
            Current strength scaling factor
        """
        return self.strength_scaling.get(cell_type, 1.0)
    
    def set_strength_scaling(self, cell_type: str, scaling: float) -> None:
        """
        Set the strength scaling factor for a cell type and update weights.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            scaling: New strength scaling factor
        """
        # Only update if the value actually changed
        if self.strength_scaling.get(cell_type, 1.0) != scaling:
            self.strength_scaling[cell_type] = scaling
            self.update_weights()
    
    def get_sparsity(self, cell_type: str) -> float:
        """
        Get the current sparsity factor for a cell type.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            
        Returns:
            Current sparsity factor (1 = all connections, 0 = no connections)
        """
        return self.sparsity.get(cell_type, 1.0)
    
    def set_sparsity(self, cell_type: str, sparsity: float) -> None:
        """
        Set the sparsity factor for a cell type and update weights.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            sparsity: New sparsity factor (1 = all connections, 0 = no connections)
        """
        # Only update if the value actually changed
        if self.sparsity.get(cell_type, 1.0) != sparsity:
            self.sparsity[cell_type] = sparsity
            
            # Clear cached masks for this cell type when sparsity changes
            keys_to_remove = [k for k in self._sparsity_masks if k[1] == cell_type]
            for key in keys_to_remove:
                del self._sparsity_masks[key]
                
            self.update_weights()

    def apply_preset(self, preset: dict) -> None:
        """
        Apply a preset configuration to initialize all connection parameters.
        
        Args:
            preset: Dictionary containing connection strengths, widths, scaling, and sparsity
        """
        # Set strength scaling factors
        if 'strength_scaling' in preset:
            for cell_type, scaling in preset['strength_scaling'].items():
                self.strength_scaling[cell_type] = scaling
        
        # Set sparsity factors
        if 'sparsity' in preset:
            for cell_type, sparsity_val in preset['sparsity'].items():
                self.sparsity[cell_type] = sparsity_val
        
        # Set connection strengths
        if 'connection_strengths' in preset:
            for conn_name, strength in preset['connection_strengths'].items():
                # Parse connection name to extract components
                if conn_name.startswith('thalamus_to_'):
                    # Thalamic connection: thalamus_to_L23_E
                    parts = conn_name.split('_')
                    target_layer = parts[2]
                    target_cell = parts[3]
                    self.set_connection_strength('thalamus', None, target_layer, target_cell, strength)
                elif '_to_' in conn_name:
                    # Layer-to-layer connection: L23_E_to_L4_SST
                    source_part, target_part = conn_name.split('_to_')
                    source_parts = source_part.split('_')
                    target_parts = target_part.split('_')
                    
                    source_layer = source_parts[0]
                    source_cell = source_parts[1]
                    target_layer = target_parts[0]
                    target_cell = target_parts[1]
                    
                    self.set_connection_strength(source_layer, source_cell, target_layer, target_cell, strength)
        
        # Set connection widths based on preset defaults
        if 'outgoing_widths' in preset:
            for cell_type, width in preset['outgoing_widths'].items():
                # Set outgoing connection widths for this cell type
                for target_layer in ['L23', 'L4', 'L5']:
                    for target_cell in ['E', 'SST', 'PV']:
                        for source_layer in ['L23', 'L4', 'L5']:
                            # Skip SST->SST connections which don't exist
                            if cell_type == 'SST' and target_cell == 'SST':
                                continue
                            
                            conn_key = f'{source_layer}_{cell_type}_to_{target_layer}_{target_cell}'
                            if conn_key in preset.get('connection_strengths', {}):
                                self.set_connection_sigma(source_layer, cell_type, target_layer, target_cell, width)
        
        # Set thalamic input widths
        if 'thalamic_widths' in preset:
            for cell_type, width in preset['thalamic_widths'].items():
                for target_layer in ['L23', 'L4', 'L5']:
                    conn_key = f'thalamus_to_{target_layer}_{cell_type}'
                    if conn_key in preset.get('connection_strengths', {}):
                        self.set_connection_sigma('thalamus', None, target_layer, cell_type, width)
        
        # Update all weights with new parameters
        self.update_weights() 