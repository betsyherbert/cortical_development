"""Connectivity module for the cortical circuit simulation.

Note: All sigma (spatial width) parameters are in μm (anatomical units).
The conversion to grid units happens internally during Gaussian computation.
"""

import numpy as np

from .config import (
    ANATOMICAL_GRID_SIZE,
    GRID_SIZE,
    INITIAL_STRENGTH_SCALING,
    LAYER_CONNECTIVITY_PARAMS,
)


class ConnectivityProfile:
    """
    Handles the computation and caching of spatial connectivity profiles.

    This class efficiently computes and caches 2D Gaussian profiles and weight matrices
    for neural connectivity. It uses pre-computed coordinate grids and distance matrices
    to optimize repeated calculations.

    Note: All sigma parameters are expected in μm (anatomical units) and are
    converted internally to grid units for computation.
    """

    def __init__(
        self, grid_size: int = GRID_SIZE, anatomical_grid_size: float = ANATOMICAL_GRID_SIZE
    ):
        """
        Initialize connectivity profiles with optimized caching.

        Args:
            grid_size: Number of grid points in each dimension
            anatomical_grid_size: Anatomical size of the grid in μm
        """
        self.grid_size = grid_size
        self.anatomical_grid_size = anatomical_grid_size
        self.grid_scale = anatomical_grid_size / grid_size  # μm per grid unit

        self._profile_cache = {}  # Cache for Gaussian profiles
        self._matrix_cache = {}  # Cache for weight matrices

        # Pre-compute coordinate meshgrid and center coordinates
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
        self.center = (grid_size // 2, grid_size // 2)

        # Pre-compute squared distances from center
        self.center_d_squared = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2

        # Pre-compute common grid sizes for weight matrices
        self.common_size = (grid_size * grid_size, grid_size * grid_size)

    def _compute_gaussian(self, d_squared: np.ndarray, sigma_grid: float) -> np.ndarray:
        """
        Compute Gaussian profile from squared distances.

        Args:
            d_squared: Array of squared distances (in grid units squared)
            sigma_grid: Width of the Gaussian in grid units

        Returns:
            Normalized Gaussian profile
        """
        profile = np.exp(-0.5 * d_squared / sigma_grid**2)
        return profile / profile.sum()

    def gaussian_profile(
        self, sigma_um: float, center: tuple[int, int] | None = None
    ) -> np.ndarray:
        """
        Get a cached 2D Gaussian profile or compute if not available.

        Args:
            sigma_um: Width of the Gaussian in μm (anatomical units)
            center: Optional center coordinates (x, y) in grid units. If None, uses grid center.

        Returns:
            2D array containing the normalized Gaussian profile
        """
        # Convert sigma from μm to grid units
        sigma_grid = sigma_um / self.grid_scale

        # Use cached profile if available (cache key uses μm for consistency)
        cache_key = (sigma_um, center if center else "center")
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        # Use pre-computed distance matrix for center case
        if center is None or center == self.center:
            d_squared = self.center_d_squared
        else:
            # Calculate squared distance from specified center
            x, y = self.coords
            d_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2

        # Compute and cache Gaussian profile
        profile = self._compute_gaussian(d_squared, sigma_grid)
        self._profile_cache[cache_key] = profile
        return profile

    def compute_weight_matrix(
        self,
        amplitude: float,
        sigma_um: float,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Get a cached weight matrix or compute if not available.

        Args:
            amplitude: Connection strength
            sigma_um: Width of the Gaussian profile in μm (anatomical units)
            source_size: Size of source population grid (in grid points)
            target_size: Size of target population grid (in grid points)

        Returns:
            2D connection weight matrix (target_neurons x source_neurons)
        """
        # Cache key for this specific weight matrix
        cache_key = (amplitude, sigma_um, source_size, target_size)
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]

        # Get Gaussian profile (sigma conversion happens inside gaussian_profile)
        profile = self.gaussian_profile(sigma_um)

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
            shifted_profile = np.roll(
                profile, (x_tgt - self.center[0], y_tgt - self.center[1]), axis=(0, 1)
            )

            # Store in weight matrix
            W[i, :] = shifted_profile.flatten()

        return W * amplitude

    def _compute_different_size_weights(
        self,
        profile: np.ndarray,
        amplitude: float,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> np.ndarray:
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
                dist_sq = (x_tgt - x_src) ** 2 + (y_tgt - y_src) ** 2
                W[i, j] = amplitude * np.exp(-0.5 * dist_sq / profile.shape[0] ** 2)

        # Normalize rows
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / row_sums

        return W


class LayerConnectivity:
    """
    Manages connectivity matrices for all layer connections.

    This class handles the creation and updating of weight matrices for connections
    between different neural populations.

    Note: All sigma (spatial width) parameters are in μm (anatomical units).
    """

    def __init__(
        self, grid_size: int = GRID_SIZE, anatomical_grid_size: float = ANATOMICAL_GRID_SIZE
    ):
        """
        Initialize connectivity matrices for all layer connections.

        Args:
            grid_size: Number of grid points in each dimension
            anatomical_grid_size: Anatomical size of the grid in μm
        """
        self.grid_size = grid_size
        self.anatomical_grid_size = anatomical_grid_size
        self.grid_scale = anatomical_grid_size / grid_size  # μm per grid unit
        self.profile = ConnectivityProfile(grid_size, anatomical_grid_size)

        # Initialize with layer parameters (sigma values are in μm)
        self.layer_params = LAYER_CONNECTIVITY_PARAMS.copy()

        # Weight matrices dictionary
        # Format: (source_layer, source_cell, target_layer, target_cell)
        self.W = {}

        # Initialize strength scaling factors
        self.strength_scaling = INITIAL_STRENGTH_SCALING.copy()

        # Use the global random state for consistency with centralized seed management

        # Initialize weight matrices
        self.update_weights()

    def update_weights(self, layer_params: dict[str, dict[str, float]] | None = None) -> None:
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
            parts = conn_key.split("_to_")
            if len(parts) != 2:
                continue  # Skip invalid connection keys

            source_part, target_part = parts

            # Split source and target parts
            source_parts = source_part.split("_")
            target_parts = target_part.split("_")

            if len(source_parts) != 2 or len(target_parts) != 2:
                continue  # Skip invalid connection keys

            source_layer, source_cell = source_parts
            target_layer, target_cell = target_parts

            # Handle thalamic connections
            if source_layer == "thalamus":
                source_cell = None
                cell_type_for_scaling = "thalamus"
            else:
                cell_type_for_scaling = source_cell

            # Apply strength scaling factor
            scaled_amplitude = params["amplitude"] * self.strength_scaling[cell_type_for_scaling]

            # Create weight matrix
            weight_matrix = self.profile.compute_weight_matrix(
                scaled_amplitude,
                params["sigma"],
                (self.grid_size, self.grid_size),
                (self.grid_size, self.grid_size),
            )

            # Store weight matrix
            self.W[(source_layer, source_cell, target_layer, target_cell)] = weight_matrix

    def compute_input(
        self,
        source: str,
        target: str,
        rates: np.ndarray,
        source_layer: str = "any",
        target_layer: str = "any",
    ) -> np.ndarray:
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

    def get_connection_strength(
        self, source_layer: str, source_cell: str, target_layer: str, target_cell: str
    ) -> float:
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
        if source_layer == "thalamus":
            conn_key = f"thalamus_to_{target_layer}_{target_cell}"
        else:
            conn_key = f"{source_layer}_{source_cell}_to_{target_layer}_{target_cell}"

        # Check if this connection has parameters defined
        if conn_key in self.layer_params:
            return self.layer_params[conn_key]["amplitude"]

        return 0.0

    def get_scaled_connection_strength(
        self, source_layer: str, source_cell: str, target_layer: str, target_cell: str
    ) -> float:
        """
        Get the strength-scaled connection strength between two populations.

        This returns amplitude * strength_scaling[source_cell_type], which is the
        actual effective connection strength used in the simulation.

        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')

        Returns:
            Scaled connection amplitude or 0 if connection doesn't exist
        """
        # Get raw amplitude
        raw_amplitude = self.get_connection_strength(
            source_layer, source_cell, target_layer, target_cell
        )

        # Get scaling factor for source cell type
        if source_layer == "thalamus":
            scaling = self.strength_scaling.get("thalamus", 1.0)
        else:
            scaling = self.strength_scaling.get(source_cell, 1.0)

        return raw_amplitude * scaling

    def get_all_connection_strengths(self) -> dict:
        """
        Get all raw (unscaled) connection strengths in the network.

        Returns:
            Dictionary mapping connection keys to their raw amplitudes
        """
        raw_strengths = {}

        for conn_key, params in self.layer_params.items():
            raw_strengths[conn_key] = params["amplitude"]

        return raw_strengths

    def get_all_scaled_strengths(self) -> dict:
        """
        Get all scaled connection strengths in the network.

        Returns:
            Dictionary mapping connection keys to their scaled strengths
        """
        scaled_strengths = {}

        for conn_key in self.layer_params:
            # Parse connection key
            parts = conn_key.split("_to_")
            source_part, target_part = parts

            if source_part == "thalamus":
                source_layer = "thalamus"
                source_cell = None
                target_parts = target_part.split("_")
                target_layer = target_parts[0]
                target_cell = target_parts[1]
            else:
                source_parts = source_part.split("_")
                target_parts = target_part.split("_")
                source_layer = source_parts[0]
                source_cell = source_parts[1]
                target_layer = target_parts[0]
                target_cell = target_parts[1]

            scaled_strengths[conn_key] = self.get_scaled_connection_strength(
                source_layer, source_cell, target_layer, target_cell
            )

        return scaled_strengths

    def get_all_sigmas(self) -> dict:
        """
        Get all connection widths (sigmas) in the network.

        Returns:
            Dictionary mapping connection keys to their sigma values
        """
        sigmas = {}

        for conn_key, params in self.layer_params.items():
            sigmas[conn_key] = params["sigma"]

        return sigmas

    def get_scaled_strength_range(self) -> tuple:
        """
        Get the min and max scaled connection strengths across all connections.

        Returns:
            Tuple of (min_strength, max_strength)
        """
        scaled_strengths = self.get_all_scaled_strengths()

        if not scaled_strengths:
            return (0.0, 0.0)

        values = list(scaled_strengths.values())
        return (min(values), max(values))

    def set_connection_strength(
        self,
        source_layer: str,
        source_cell: str,
        target_layer: str,
        target_cell: str,
        amplitude: float,
    ) -> None:
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
        if source_layer == "thalamus":
            conn_key = f"thalamus_to_{target_layer}_{target_cell}"
            cell_type_for_scaling = "thalamus"
        else:
            conn_key = f"{source_layer}_{source_cell}_to_{target_layer}_{target_cell}"
            cell_type_for_scaling = source_cell

        # Create connection parameter entry if it doesn't exist
        if conn_key not in self.layer_params:
            # Determine appropriate sigma (in μm) based on cell type patterns
            if source_cell == "E" or source_layer == "thalamus":
                sigma = 100.0  # Default for excitatory connections (μm)
            elif source_cell == "SST":
                sigma = 150.0  # Default for SST connections (wider, μm)
            elif source_cell == "PV":
                sigma = 75.0  # Default for PV connections (narrower, μm)
            else:
                sigma = 100.0  # Default fallback (μm)

            # Add the new connection parameters
            self.layer_params[conn_key] = {"amplitude": 0.0, "sigma": sigma}

        # Update the parameter
        self.layer_params[conn_key]["amplitude"] = amplitude

        # Get the sigma value
        sigma = self.layer_params[conn_key]["sigma"]

        # Apply strength scaling factor
        scaled_amplitude = amplitude * self.strength_scaling[cell_type_for_scaling]

        # Create the connection tuple key
        tuple_key = (source_layer, source_cell, target_layer, target_cell)

        # Create the weight matrix with scaled amplitude
        weight_matrix = self.profile.compute_weight_matrix(
            scaled_amplitude,
            sigma,
            (self.grid_size, self.grid_size),
            (self.grid_size, self.grid_size),
        )

        # Store weight matrix
        self.W[tuple_key] = weight_matrix

    def get_connection_sigma(
        self, source_layer: str, source_cell: str, target_layer: str, target_cell: str
    ) -> float:
        """
        Get the current connection width (sigma) between two populations.

        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')

        Returns:
            Connection sigma in μm, or default value if connection doesn't exist
        """
        # Generate the connection key
        if source_layer == "thalamus":
            conn_key = f"thalamus_to_{target_layer}_{target_cell}"
        else:
            conn_key = f"{source_layer}_{source_cell}_to_{target_layer}_{target_cell}"

        # Check if this connection has parameters defined
        if conn_key in self.layer_params:
            return self.layer_params[conn_key]["sigma"]

        # Determine default sigma (in μm) based on cell type patterns if connection doesn't exist
        if source_cell == "E" or source_layer == "thalamus":
            return 100.0  # Default for excitatory connections (μm)
        elif source_cell == "SST":
            return 150.0  # Default for SST connections (wider, μm)
        elif source_cell == "PV":
            return 75.0  # Default for PV connections (narrower, μm)

        return 100.0  # Default fallback (μm)

    def set_connection_sigma(
        self,
        source_layer: str,
        source_cell: str,
        target_layer: str,
        target_cell: str,
        sigma_um: float,
    ) -> None:
        """
        Set the connection width (sigma) between two populations.

        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            sigma_um: Connection width (Gaussian sigma) in μm
        """
        # Generate the connection key
        if source_layer == "thalamus":
            conn_key = f"thalamus_to_{target_layer}_{target_cell}"
            cell_type_for_scaling = "thalamus"
        else:
            conn_key = f"{source_layer}_{source_cell}_to_{target_layer}_{target_cell}"
            cell_type_for_scaling = source_cell

        # Create connection parameter entry if it doesn't exist
        if conn_key not in self.layer_params:
            # Determine appropriate amplitude based on common patterns
            if source_cell == "E" or source_layer == "thalamus":
                amplitude = 0.2  # Default for excitatory
            elif source_cell == "SST" or source_cell == "PV":
                amplitude = -0.1  # Default for inhibitory
            else:
                amplitude = 0.0  # Default fallback

            # Add the new connection parameters
            self.layer_params[conn_key] = {"amplitude": amplitude, "sigma": sigma_um}
        else:
            # Update the parameter
            self.layer_params[conn_key]["sigma"] = sigma_um

        # Get current raw amplitude
        amplitude = self.layer_params[conn_key]["amplitude"]

        # Apply strength scaling factor
        scaled_amplitude = amplitude * self.strength_scaling[cell_type_for_scaling]

        # Create the connection tuple key
        tuple_key = (source_layer, source_cell, target_layer, target_cell)

        # Create the weight matrix with scaled amplitude (sigma is in μm)
        weight_matrix = self.profile.compute_weight_matrix(
            scaled_amplitude,
            sigma_um,
            (self.grid_size, self.grid_size),
            (self.grid_size, self.grid_size),
        )

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

    def get_all_strength_scaling(self) -> dict:
        """
        Get all current strength scaling factors.

        Returns:
            Dictionary mapping cell types to their strength scaling factors
        """
        return self.strength_scaling.copy()

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

    def apply_preset(self, preset: dict) -> None:
        """
        Apply a preset configuration to initialize all connection parameters.

        Args:
            preset: Dictionary containing connection strengths, widths, and scaling
        """
        # Set strength scaling factors
        if "strength_scaling" in preset:
            for cell_type, scaling in preset["strength_scaling"].items():
                self.strength_scaling[cell_type] = scaling

        # Set connection strengths
        if "connection_strengths" in preset:
            for conn_name, strength in preset["connection_strengths"].items():
                # Parse connection name to extract components
                if conn_name.startswith("thalamus_to_"):
                    # Thalamic connection: thalamus_to_L23_E
                    parts = conn_name.split("_")
                    target_layer = parts[2]
                    target_cell = parts[3]
                    self.set_connection_strength(
                        "thalamus", None, target_layer, target_cell, strength
                    )
                elif "_to_" in conn_name:
                    # Layer-to-layer connection: L23_E_to_L4_SST
                    source_part, target_part = conn_name.split("_to_")
                    source_parts = source_part.split("_")
                    target_parts = target_part.split("_")

                    source_layer = source_parts[0]
                    source_cell = source_parts[1]
                    target_layer = target_parts[0]
                    target_cell = target_parts[1]

                    self.set_connection_strength(
                        source_layer, source_cell, target_layer, target_cell, strength
                    )

        # Set connection widths based on preset defaults
        if "outgoing_widths" in preset:
            for cell_type, width in preset["outgoing_widths"].items():
                # Set outgoing connection widths for this cell type
                for target_layer in ["L23", "L4", "L5"]:
                    for target_cell in ["E", "SST", "PV"]:
                        for source_layer in ["L23", "L4", "L5"]:
                            # Skip SST->SST connections which don't exist
                            if cell_type == "SST" and target_cell == "SST":
                                continue

                            conn_key = f"{source_layer}_{cell_type}_to_{target_layer}_{target_cell}"
                            if conn_key in preset.get("connection_strengths", {}):
                                self.set_connection_sigma(
                                    source_layer, cell_type, target_layer, target_cell, width
                                )

        # Set thalamic input widths
        if "thalamic_widths" in preset:
            for cell_type, width in preset["thalamic_widths"].items():
                for target_layer in ["L23", "L4", "L5"]:
                    conn_key = f"thalamus_to_{target_layer}_{cell_type}"
                    if conn_key in preset.get("connection_strengths", {}):
                        self.set_connection_sigma("thalamus", None, target_layer, cell_type, width)

        # Update all weights with new parameters
        self.update_weights()
