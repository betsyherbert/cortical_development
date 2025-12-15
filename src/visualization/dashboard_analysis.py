"""Analysis computation delegate for the dashboard.

This module contains the DashboardAnalysis class which handles stability
and gain computations for the dashboard. It is owned by DashboardApp
and accesses simulation state through the dashboard reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.analysis.bifurcation import NetworkModel, StabilityAnalyzer
from src.analysis.bifurcation.config import ANALYSIS_PARAMS

if TYPE_CHECKING:
    from src.visualization.dashboard import DashboardApp

logger = logging.getLogger(__name__)


class DashboardAnalysis:
    """Delegate object for stability and gain analysis computations.

    This class encapsulates the expensive analysis computations that were
    previously embedded in DashboardApp. It maintains a reference to the
    dashboard to access simulation state.

    Attributes:
        dashboard: Reference to the parent DashboardApp instance
    """

    def __init__(self, dashboard: "DashboardApp"):
        """Initialize the analysis delegate.

        Args:
            dashboard: Parent DashboardApp instance
        """
        self.dashboard = dashboard

    def _get_steady_state(self) -> np.ndarray | None:
        """Extract spatial mean rates from current simulation state.

        Returns:
            Array of 9 mean rates or None if network not active
        """
        return self.dashboard._extract_mean_rates_from_simulation()

    def _get_population_indices(self, selected_pops: list[str]) -> np.ndarray:
        """Get indices for selected populations.

        Args:
            selected_pops: List of population IDs (e.g., ['L23_E', 'L4_SST'])

        Returns:
            Array of indices corresponding to the selected populations
        """
        return self.dashboard._get_population_indices(selected_pops)

    def compute_B_fourier(
        self, network: NetworkModel, k_squared: float, anatomical_grid_size: float
    ) -> np.ndarray:
        """Compute thalamic input B(k) in Fourier space with Gaussian spatial filtering.

        Args:
            network: NetworkModel instance containing thalamic parameters
            k_squared: Square of the wavenumber k (mode number)
            anatomical_grid_size: Anatomical grid size in micrometers

        Returns:
            B(k): Thalamic input vector in Fourier space (length = number of populations)
        """
        total_pops = len(network.thalamic_strengths)
        B_k = np.zeros(total_pops)

        for i in range(total_pops):
            # Normalize thalamic width by anatomical grid size
            sigma_thal_i = network.thalamic_widths[i] / anatomical_grid_size
            # Apply Gaussian spatial filtering: B[i] = strength * exp(-2*pi^2*k^2*sigma^2)
            B_k[i] = network.thalamic_strengths[i] * np.exp(
                -2 * np.pi**2 * k_squared * sigma_thal_i**2
            )

        return B_k

    def _build_exp_cache(
        self, network: NetworkModel, k_squared_set: set, anatomical_grid_size: float
    ) -> dict[int, np.ndarray]:
        """Build cache of exponential factors for all k^2 values.

        Args:
            network: NetworkModel instance
            k_squared_set: Set of k^2 values to cache
            anatomical_grid_size: Grid size in micrometers

        Returns:
            Dictionary mapping k^2 to exponential factor matrices
        """
        total_pops = len(network.tau)
        exp_cache = {}

        for k_squared in k_squared_set:
            exp_cache[k_squared] = np.zeros((total_pops, total_pops))
            for i in range(total_pops):
                for j in range(total_pops):
                    sigma_ij = network.sigma[i, j] / anatomical_grid_size
                    exp_cache[k_squared][i, j] = np.exp(
                        -2 * np.pi**2 * k_squared * sigma_ij**2
                    )

        return exp_cache

    def _build_jacobian(
        self,
        network: NetworkModel,
        analyzer: StabilityAnalyzer,
        exp_factors: np.ndarray,
    ) -> np.ndarray:
        """Build Jacobian matrix using pre-computed exponential factors.

        Args:
            network: NetworkModel instance
            analyzer: StabilityAnalyzer with effective gains
            exp_factors: Pre-computed exponential factors for this k^2

        Returns:
            Jacobian matrix
        """
        total_pops = len(network.tau)
        J = np.zeros((total_pops, total_pops))

        for i in range(total_pops):
            for j in range(total_pops):
                w_tilde = network.A[i, j] * exp_factors[i, j]
                if i == j:
                    J[i, j] = (
                        -1.0 / network.tau[i]
                        + (analyzer.g_eff[i] * w_tilde) / network.tau[i]
                    )
                else:
                    J[i, j] = (analyzer.g_eff[i] * w_tilde) / network.tau[i]

        return J

    def _get_k_squared_set(self, n_modes_effective: int) -> set[int]:
        """Get set of unique k^2 values for positive quadrant scan.

        Args:
            n_modes_effective: Effective number of modes to scan

        Returns:
            Set of k^2 values
        """
        k_squared_set = set()
        for n1 in range(0, n_modes_effective + 1):
            for n2 in range(0, n_modes_effective + 1):
                k_squared_set.add(n1**2 + n2**2)
        return k_squared_set

    def compute_stability_spectrum(
        self, preset: dict, selected_pops: list | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute stability spectrum (max Re(lambda) vs k) for current network state.

        Linearizes around the current spatial-mean activity from the running simulation.
        Optimized: scans only positive quadrant and caches exponentials.

        Args:
            preset: Preset dictionary with current network parameters
            selected_pops: Optional list of population IDs to analyze.
                          If None, analyzes full network.

        Returns:
            Tuple of (k_values, max_real_eigenvalues, eigenvalues_at_max_k, k_max)
        """
        try:
            # Extract current spatial mean rates from simulation
            steady_state = self._get_steady_state()

            # Check if network is active
            if steady_state is None:
                return np.array([]), np.array([]), np.array([]), 0.0

            # Create network model for full network (all 3 layers)
            network = NetworkModel(preset, layers=["L23", "L4", "L5"])

            # Create stability analyzer with current simulation state
            analyzer = StabilityAnalyzer(network, steady_state)

            # Get analysis parameters
            n_modes = ANALYSIS_PARAMS["n_modes"]
            grid_size = ANALYSIS_PARAMS["grid_size"]
            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
            n_modes_effective = min(n_modes, int(0.6 * grid_size))

            # Pre-compute all unique k^2 values and cache exponentials
            k_squared_set = self._get_k_squared_set(n_modes_effective)
            exp_cache = self._build_exp_cache(network, k_squared_set, anatomical_grid_size)

            # Dictionary to store results by k^2
            results_by_k2 = {}

            # Scan only positive quadrant (reduces work by ~4x)
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)

                    # Skip if k > n_modes
                    if k > n_modes:
                        continue

                    # Build Jacobian using cached exponentials
                    J = self._build_jacobian(network, analyzer, exp_cache[k_squared])

                    # Extract subset if selected_pops is provided
                    if selected_pops is not None and len(selected_pops) > 0:
                        indices = self._get_population_indices(selected_pops)
                        if len(indices) == 0:
                            continue
                        J_subset = J[np.ix_(indices, indices)]
                        eigenvalues = np.linalg.eigvals(J_subset)
                    else:
                        eigenvalues = np.linalg.eigvals(J)

                    max_real = np.max(eigenvalues.real)

                    # Store or update max real eigenvalue for this k
                    if k_squared not in results_by_k2:
                        results_by_k2[k_squared] = {
                            "k": k,
                            "max_real": max_real,
                            "eigenvalues": eigenvalues,
                        }
                    else:
                        if max_real > results_by_k2[k_squared]["max_real"]:
                            results_by_k2[k_squared]["max_real"] = max_real
                            results_by_k2[k_squared]["eigenvalues"] = eigenvalues

            # Sort by k and extract arrays
            sorted_results = sorted(results_by_k2.values(), key=lambda x: x["k"])
            k_values = np.array([r["k"] for r in sorted_results])
            max_real_values = np.array([r["max_real"] for r in sorted_results])

            # Find k with maximum instability
            if len(max_real_values) > 0:
                max_idx = np.argmax(max_real_values)
                k_max = k_values[max_idx]
                eigenvalues_at_max_k = sorted_results[max_idx]["eigenvalues"]
            else:
                k_max = 0.0
                eigenvalues_at_max_k = np.array([])

            return k_values, max_real_values, eigenvalues_at_max_k, k_max

        except Exception:
            logger.exception("Error computing stability spectrum")
            return np.array([]), np.array([]), np.array([]), 0.0

    def compute_static_gain(
        self, preset: dict, selected_pops: list | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute static spatial gain curve G(k) = ||-J(k)^(-1) B(k)||.

        Shows how strongly each spatial frequency is amplified by the cortical circuit.

        Args:
            preset: Network preset dictionary
            selected_pops: Optional list of population IDs to analyze.

        Returns:
            (k_values, gain_values): Arrays of k and corresponding gains
        """
        try:
            # Get steady state from running simulation
            steady_state = self._get_steady_state()
            if steady_state is None:
                return np.array([]), np.array([])

            # Build network model and analyzer
            network = NetworkModel(preset, layers=["L23", "L4", "L5"])
            analyzer = StabilityAnalyzer(network, steady_state)

            # Get analysis parameters
            n_modes = ANALYSIS_PARAMS["n_modes"]
            grid_size = ANALYSIS_PARAMS["grid_size"]
            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
            n_modes_effective = min(n_modes, int(0.6 * grid_size))

            # Pre-compute exponentials
            k_squared_set = self._get_k_squared_set(n_modes_effective)
            exp_cache = self._build_exp_cache(network, k_squared_set, anatomical_grid_size)

            # Aggregate results by k^2
            results_by_k2 = {}

            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)
                    if k > n_modes:
                        continue

                    # Build Jacobian J(k)
                    J = self._build_jacobian(network, analyzer, exp_cache[k_squared])

                    # Compute B(k) with thalamic spatial filtering
                    B_k = self.compute_B_fourier(network, k_squared, anatomical_grid_size)

                    # Extract subset if selected_pops is provided
                    if selected_pops is not None and len(selected_pops) > 0:
                        indices = self._get_population_indices(selected_pops)
                        if len(indices) == 0:
                            continue
                        J_to_use = J[np.ix_(indices, indices)]
                        B_to_use = B_k[indices]
                    else:
                        J_to_use = J
                        B_to_use = B_k

                    # Check if B(k) is non-zero
                    if np.linalg.norm(B_to_use) < 1e-10:
                        continue

                    # Compute gain: G(k) = ||-J(k)^(-1) B(k)||
                    try:
                        J_inv_B = np.linalg.solve(-J_to_use, B_to_use)
                        gain = np.linalg.norm(J_inv_B)

                        # Store maximum gain across degenerate modes
                        if k_squared not in results_by_k2:
                            results_by_k2[k_squared] = {"k": k, "gain": gain}
                        else:
                            results_by_k2[k_squared]["gain"] = max(
                                results_by_k2[k_squared]["gain"], gain
                            )
                    except np.linalg.LinAlgError:
                        continue

            # Sort by k and extract results
            sorted_results = sorted(results_by_k2.values(), key=lambda x: x["k"])
            k_values = np.array([r["k"] for r in sorted_results])
            gain_values = np.array([r["gain"] for r in sorted_results])

            return k_values, gain_values

        except Exception:
            logger.exception("Error computing static gain")
            return np.array([]), np.array([])

    def compute_spatiotemporal_gain(
        self, preset: dict, selected_pops: list | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spatiotemporal amplification map A(k,omega) = ||(i*omega*I - J(k))^(-1) B(k)||.

        Shows which spatial (k) and temporal (omega) frequencies the circuit amplifies most.

        Args:
            preset: Network preset dictionary
            selected_pops: Optional list of population IDs to analyze.

        Returns:
            (k_values, omega_values, gain_matrix): Arrays of k, omega, and gain[k,omega]
        """
        try:
            # Get steady state from running simulation
            steady_state = self._get_steady_state()
            if steady_state is None:
                return np.array([]), np.array([]), np.array([])

            # Build network model and analyzer
            network = NetworkModel(preset, layers=["L23", "L4", "L5"])
            analyzer = StabilityAnalyzer(network, steady_state)

            # Get analysis parameters
            n_modes = ANALYSIS_PARAMS["n_modes"]
            grid_size = ANALYSIS_PARAMS["grid_size"]
            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
            n_modes_effective = min(n_modes, int(0.6 * grid_size))
            total_pops = len(network.tau)

            # Define temporal frequency range (0-1 Hz)
            omega_values = np.linspace(0, 1, 21)

            # Pre-compute exponentials
            k_squared_set = self._get_k_squared_set(n_modes_effective)
            exp_cache = self._build_exp_cache(network, k_squared_set, anatomical_grid_size)

            # Aggregate k values
            k_values_dict = {}
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)
                    if k > n_modes:
                        continue
                    if k_squared not in k_values_dict:
                        k_values_dict[k_squared] = k

            # Sort k values
            sorted_k_squared = sorted(k_values_dict.keys())
            k_values = np.array([k_values_dict[k2] for k2 in sorted_k_squared])

            # Initialize gain matrix
            gain_matrix = np.zeros((len(k_values), len(omega_values)))

            # Compute gain for each (k, omega)
            for k_idx, k_squared in enumerate(sorted_k_squared):
                # Build Jacobian J(k)
                J = self._build_jacobian(network, analyzer, exp_cache[k_squared])

                # Compute B(k) with thalamic spatial filtering
                B_k = self.compute_B_fourier(network, k_squared, anatomical_grid_size)

                # Extract subset if selected_pops is provided
                if selected_pops is not None and len(selected_pops) > 0:
                    indices = self._get_population_indices(selected_pops)
                    if len(indices) == 0:
                        continue
                    J_to_use = J[np.ix_(indices, indices)]
                    B_to_use = B_k[indices]
                    n_pops_subset = len(indices)
                else:
                    J_to_use = J
                    B_to_use = B_k
                    n_pops_subset = total_pops

                # Check if B(k) is non-zero
                if np.linalg.norm(B_to_use) < 1e-10:
                    continue

                # For each temporal frequency omega
                for omega_idx, omega in enumerate(omega_values):
                    omega_rad = 2 * np.pi * omega

                    # Compute (i*omega*I - J(k))
                    M = 1j * omega_rad * np.eye(n_pops_subset) - J_to_use

                    # Compute A(k,omega) = ||(i*omega*I - J(k))^(-1) B(k)||
                    try:
                        M_inv_B = np.linalg.solve(M, B_to_use)
                        gain = np.linalg.norm(M_inv_B)
                        gain_matrix[k_idx, omega_idx] = gain
                    except np.linalg.LinAlgError:
                        gain_matrix[k_idx, omega_idx] = 0.0

            return k_values, omega_values, gain_matrix

        except Exception:
            logger.exception("Error computing spatiotemporal gain")
            return np.array([]), np.array([]), np.array([])

