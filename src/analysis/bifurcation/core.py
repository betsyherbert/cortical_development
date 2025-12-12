"""Core infrastructure for bifurcation analysis.

This module provides the fundamental classes and utilities for network stability
and gain analysis, including network parameter extraction, steady state finding,
and stability computation through eigenvalue analysis.

Note: All spatial parameters (sigma, wavelength) are in μm (anatomical units).
Wavenumber k is in cycles/μm.
"""

import numpy as np

from .config import ANALYSIS_PARAMS

# ============================================================================
# Utility Functions
# ============================================================================


def set_nested_value(d: dict, path: list[str], value: float) -> None:
    """Set nested dictionary value via path.

    Args:
        d: Dictionary to modify
        path: List of keys representing path (e.g., ['time_constants', 'E'])
        value: Value to set
    """
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def get_nested_value(d: dict, path: list[str]) -> float:
    """Get nested dictionary value via path.

    Args:
        d: Dictionary to access
        path: List of keys representing path

    Returns:
        Value at the specified path
    """
    for key in path:
        d = d[key]
    return d


def compute_B_fourier(
    network: "NetworkModel", k_squared: float, anatomical_grid_size: float
) -> np.ndarray:
    """Compute thalamic input B(k) in Fourier space with Gaussian spatial filtering.

    Args:
        network: NetworkModel instance containing thalamic parameters
        k_squared: Square of the wavenumber k (in mode number units)
        anatomical_grid_size: Anatomical size of the grid in μm

    Returns:
        B(k): Thalamic input vector in Fourier space (length = number of populations)
    """
    total_pops = len(network.thalamic_strengths)
    B_k = np.zeros(total_pops)

    for i in range(total_pops):
        # Normalize thalamic width by anatomical grid size
        # thalamic_widths are in μm, anatomical_grid_size is in μm, so ratio is dimensionless
        sigma_thal_i = network.thalamic_widths[i] / anatomical_grid_size
        # Apply Gaussian spatial filtering: B[i] = strength * exp(-2π²k²σ²)
        B_k[i] = network.thalamic_strengths[i] * np.exp(-2 * np.pi**2 * k_squared * sigma_thal_i**2)

    return B_k


# ============================================================================
# Core Classes
# ============================================================================


class NetworkModel:
    """Handles network dynamics for single or multiple layers with E-SST-PV populations."""

    def __init__(self, preset: dict, layers: list = None):
        """Initialize network from preset.

        Args:
            preset: Developmental preset dictionary (e.g., P0_PRESET)
            layers: List of layers to analyze (e.g., ['L4'] or ['L23', 'L4', 'L5'])
                   Defaults to ['L4'] for backward compatibility
        """
        # Handle backward compatibility: convert single layer string to list
        if layers is None:
            layers = ["L4"]
        elif isinstance(layers, str):
            layers = [layers]

        self.layers = layers
        self.preset = preset
        self.pop_names = ["E", "SST", "PV"]

        # Extract parameters
        self._extract_time_constants()
        self._extract_gains()
        self._extract_connection_strengths()
        self._extract_spatial_scales()
        self._extract_thalamic_strengths()
        self._extract_thalamic_widths()

        # Baseline input: noise mean provides constant baseline input in the simulation
        # For spatially uniform steady state, we use the noise mean as mu
        self._extract_baseline_input()

    def _extract_time_constants(self):
        """Extract time constants for each population, repeated for each layer."""
        tau_per_layer = np.array(
            [
                self.preset["time_constants"]["E"],
                self.preset["time_constants"]["SST"],
                self.preset["time_constants"]["PV"],
            ]
        )
        self.tau = np.tile(tau_per_layer, len(self.layers))

    def _extract_gains(self):
        """Extract gains for each population, repeated for each layer.

        Gains are always 1.0 for all cell types.
        """
        gain_per_layer = np.array([1.0, 1.0, 1.0])
        self.gain = np.tile(gain_per_layer, len(self.layers))

    def _extract_connection_strengths(self):
        """Extract connection strengths for all layer pairs, building block-structured matrix.

        Strength scaling is applied column-wise (per source cell type).
        Connection strengths from presets include signs (inhibitory = negative).
        """
        scaling = (
            self.preset["strength_scaling"]["E"],
            self.preset["strength_scaling"]["SST"],
            self.preset["strength_scaling"]["PV"],
        )

        n_layers = len(self.layers)
        n = 3  # E, SST, PV
        total_pops = n_layers * n
        self.A = np.zeros((total_pops, total_pops))

        for i_tgt, layer_tgt in enumerate(self.layers):
            for i_src, layer_src in enumerate(self.layers):
                block = self._extract_layer_pair_block(layer_src, layer_tgt, *scaling)
                row_start = i_tgt * n
                col_start = i_src * n
                self.A[row_start : row_start + n, col_start : col_start + n] = block

    def _extract_layer_pair_block(
        self,
        layer_src: str,
        layer_tgt: str,
        scaling_e: float,
        scaling_sst: float,
        scaling_pv: float,
    ) -> np.ndarray:
        """Extract connection strength block for one layer pair.

        Connection strengths are read directly from preset with signs included.
        Inhibitory connections (SST→*, PV→*) should be negative in presets.
        Strength scaling is column-wise (applied per source cell type).

        Args:
            layer_src: Source layer name
            layer_tgt: Target layer name
            scaling_e: Scaling factor for E source connections
            scaling_sst: Scaling factor for SST source connections
            scaling_pv: Scaling factor for PV source connections

        Returns:
            3×3 connection strength block (E, SST, PV)
        """

        def get_raw(source, target):
            key = f"{layer_src}_{source}_to_{layer_tgt}_{target}"
            return self.preset["connection_strengths"].get(key, 0.0)

        scalings = [scaling_e, scaling_sst, scaling_pv]
        sources = ["E", "SST", "PV"]
        targets = ["E", "SST", "PV"]

        block = np.zeros((3, 3))
        for i, target in enumerate(targets):
            for j, source in enumerate(sources):
                block[i, j] = get_raw(source, target) * scalings[j]

        return block

    def _extract_spatial_scales(self):
        """Extract spatial scales (sigma) for connections, using source population's width for all its connections."""
        sigma_per_pop = np.array(
            [
                self.preset["outgoing_widths"]["E"],
                self.preset["outgoing_widths"]["SST"],
                self.preset["outgoing_widths"]["PV"],
            ]
        )

        n_layers = len(self.layers)
        n = 3  # E, SST, PV
        total_pops = n_layers * n
        self.sigma = np.zeros((total_pops, total_pops))

        # For each source population (column), use its sigma for all targets (rows)
        for i_src in range(n_layers):
            for j_pop in range(n):
                col_idx = i_src * n + j_pop
                self.sigma[:, col_idx] = sigma_per_pop[j_pop]

    def _extract_baseline_input(self):
        """Extract baseline input from background_input parameters, repeated for each layer."""
        mu_per_layer = np.array(
            [
                self.preset["background_input"]["E"],
                self.preset["background_input"]["SST"],
                self.preset["background_input"]["PV"],
            ]
        )
        self.mu = np.tile(mu_per_layer, len(self.layers))

    def _extract_thalamic_strengths(self):
        """Extract thalamic connection strengths for each population in each layer."""
        # Get thalamic scaling factor
        thalamic_scaling = self.preset["strength_scaling"].get("thalamus", 1.0)

        # Build thalamic strengths for each layer and population
        thalamic_per_layer = []
        for layer in self.layers:
            thal_e = self.preset["connection_strengths"].get(f"thalamus_to_{layer}_E", 0.0)
            thal_sst = self.preset["connection_strengths"].get(f"thalamus_to_{layer}_SST", 0.0)
            thal_pv = self.preset["connection_strengths"].get(f"thalamus_to_{layer}_PV", 0.0)
            thalamic_per_layer.extend([thal_e, thal_sst, thal_pv])

        self.thalamic_strengths = np.array(thalamic_per_layer) * thalamic_scaling

    def _extract_thalamic_widths(self):
        """Extract thalamic spatial widths for each population in each layer."""
        thalamic_widths_per_layer = np.array(
            [
                self.preset["thalamic_widths"]["E"],
                self.preset["thalamic_widths"]["SST"],
                self.preset["thalamic_widths"]["PV"],
            ]
        )
        self.thalamic_widths = np.tile(thalamic_widths_per_layer, len(self.layers))

    def compute_thalamic_input(self, input_magnitude: float) -> np.ndarray:
        """Compute spatially-averaged thalamic drive.

        Args:
            input_magnitude: Scalar magnitude of thalamic input

        Returns:
            Array of thalamic drive for each population (same shape as mu)
        """
        return self.thalamic_strengths * input_magnitude

    def get_parameters(self) -> dict:
        """Get all network parameters for display."""
        full_pop_names = [f"{layer}_{pop}" for layer in self.layers for pop in self.pop_names]

        return {
            "n_populations": 3,  # Always 3: E, SST, PV
            "n_layers": len(self.layers),
            "layers": self.layers.copy(),
            "pop_names": self.pop_names,
            "full_pop_names": full_pop_names,
            "tau": self.tau.copy(),
            "gain": self.gain.copy(),
            "A": self.A.copy(),
            "sigma": self.sigma.copy(),
            "mu": self.mu.copy(),
            "thalamic_strengths": self.thalamic_strengths.copy(),
            "thalamic_widths": self.thalamic_widths.copy(),
        }


class SteadyStateFinder:
    """Finds the spatially uniform steady state of the network."""

    def __init__(self, network: NetworkModel, tol: float = None, max_iters: int = None):
        """Initialize steady state finder.

        Args:
            network: NetworkModel instance
            tol: Convergence tolerance (defaults to config value)
            max_iters: Maximum iterations (defaults to config value)
        """
        self.network = network
        self.tol = tol if tol is not None else ANALYSIS_PARAMS["tolerance"]
        self.max_iters = max_iters if max_iters is not None else ANALYSIS_PARAMS["max_iters"]

    def find_steady_state(self, thalamic_input: np.ndarray | None = None) -> tuple[np.ndarray, str]:
        """Find the spatially uniform steady state.

        For thalamic-driven mode, this computes the spatially uniform mean firing rate
        during thalamic input epochs. This captures the mean depolarization but not
        the spatial structure of bursts (which would require full spatial dynamics).

        Args:
            thalamic_input: Optional array of thalamic drive for each population.
                           If None, finds silent fixed point (no external input).

        Returns:
            (r_star, status) tuple where r_star is array of firing rates.
            status is one of: 'converged', 'not_converged', 'diverged'
        """
        n = len(self.network.tau)
        r = np.ones(n) * 0.1

        external_input = (
            self.network.mu if thalamic_input is None else self.network.mu + thalamic_input
        )

        alpha = 0.05  # Conservative fixed damping for stability near criticality

        for _ in range(self.max_iters):
            input_vec = self.network.A @ r + external_input
            r_new_raw = np.maximum(0.0, self.network.gain * input_vec)

            if np.any(r_new_raw > 1e10):
                return np.zeros(n), "diverged"

            r_new = (1 - alpha) * r + alpha * r_new_raw

            if np.all(np.abs(r_new - r) < self.tol):
                return r_new, "converged"

            r = r_new

        return np.clip(r, 0, 1e10), "not_converged"


class StabilityAnalyzer:
    """Computes network stability through eigenvalue analysis."""

    def __init__(
        self,
        network: NetworkModel,
        steady_state: np.ndarray,
        n_modes: int = None,
        threshold: float = 1e-10,
    ):
        """Initialize stability analyzer.

        Args:
            network: NetworkModel instance
            steady_state: Array of steady state firing rates
            n_modes: Number of Fourier modes to scan in each direction (defaults to config value)
            threshold: Threshold for determining active units (default: 1e-10)
        """
        self.network = network
        self.r_star = steady_state
        self.n_modes = n_modes if n_modes is not None else ANALYSIS_PARAMS["n_modes"]
        self.threshold = threshold

        # For ReLU, effective gains depend on operating point:
        # - Active units (r* > threshold) have full gain
        # - Inactive units (r* ≈ 0) have zero gain
        self.g_eff = self._compute_effective_gains()

    def _compute_effective_gains(self) -> np.ndarray:
        """Compute effective gains based on operating point.

        For ReLU nonlinearity, the linearization around an operating point
        has gain equal to the slope only for active units.

        Returns:
            Array of effective gains for each population
        """
        g_eff = np.zeros_like(self.network.gain)
        active_mask = self.r_star > self.threshold
        g_eff[active_mask] = self.network.gain[active_mask]
        return g_eff

    def build_jacobian(self, n1: int, n2: int) -> np.ndarray:
        """Build n×n Jacobian matrix for Fourier mode (n1, n2).

        Args:
            n1: Fourier mode index in first dimension
            n2: Fourier mode index in second dimension

        Returns:
            n×n Jacobian matrix (n = total populations across all layers)
        """
        k_squared = n1**2 + n2**2
        anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]

        n = len(self.network.tau)
        J = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Normalize spatial scale: σ_norm = σ_μm / L_μm
                sigma_ij = self.network.sigma[i, j] / anatomical_grid_size

                # Fourier transform of Gaussian connectivity: exp(-2π²k²σ²)
                w_tilde = self.network.A[i, j] * np.exp(-2 * np.pi**2 * k_squared * sigma_ij**2)

                # Jacobian element: -δ_ij/τ_i + (g_i * w_tilde) / τ_i
                delta_term = -1.0 / self.network.tau[i] if i == j else 0.0
                J[i, j] = delta_term + (self.g_eff[i] * w_tilde) / self.network.tau[i]

        return J

    def compute_stability(
        self, verbose: bool = False
    ) -> tuple[float, tuple[int, int], float, float]:
        """Compute stability by finding maximum real part of eigenvalues.

        Args:
            verbose: If True, print diagnostic information

        Returns:
            (distance_to_instability, critical_mode, critical_k, wavelength) tuple
            - critical_k: Critical wavenumber in cycles/μm
            - wavelength: Critical wavelength in μm (= anatomical_grid_size / k_mode, or inf if k=0)
        """
        grid_size = ANALYSIS_PARAMS["grid_size"]
        anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
        n_modes_effective = min(self.n_modes, int(0.6 * grid_size))

        if verbose:
            print(
                f"  [Stability] Scanning modes from -{n_modes_effective} to +{n_modes_effective} (clamped from {self.n_modes})"
            )
            print(
                f"  [Stability] Active populations: {np.sum(self.r_star > 1e-10)}/{len(self.r_star)}"
            )
            print(f"  [Stability] g_eff non-zero: {np.sum(self.g_eff > 0)}/{len(self.g_eff)}")
            print(f"  [Stability] g_eff values: {self.g_eff}")

        max_real_eigenvalue = -np.inf
        critical_mode = (0, 0)
        critical_k_mode = 0.0

        for n1 in range(-n_modes_effective, n_modes_effective + 1):
            for n2 in range(-n_modes_effective, n_modes_effective + 1):
                J = self.build_jacobian(n1, n2)
                max_real = np.max(np.linalg.eigvals(J).real)

                if max_real > max_real_eigenvalue:
                    max_real_eigenvalue = max_real
                    critical_mode = (n1, n2)
                    critical_k_mode = np.sqrt(n1**2 + n2**2)

        distance_to_instability = -max_real_eigenvalue
        critical_k = critical_k_mode / anatomical_grid_size
        wavelength = anatomical_grid_size / critical_k_mode if critical_k_mode > 0 else np.inf

        if verbose:
            print(
                f"  [Stability] Critical mode: {critical_mode}, k={critical_k:.6f} cycles/μm, λ*={wavelength:.1f} μm"
            )
            print(f"  [Stability] Max Re(λ): {max_real_eigenvalue:.6f}")
            print(f"  [Stability] Distance to instability: {distance_to_instability:.6f}")

        return distance_to_instability, critical_mode, critical_k, wavelength
