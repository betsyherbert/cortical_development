"""Computational helpers for the cortical simulation dashboard.

This module contains expensive computation functions extracted from callbacks.
Keep simulation/analysis logic here, not in callback handlers.

The DashboardApp class calls these helpers from its callbacks.
"""

import numpy as np

from src.analysis.timing import timed

# =============================================================================
# Correlation Computations
# =============================================================================


def compute_group_correlation(corr_matrix: np.ndarray, indices: np.ndarray) -> float:
    """Compute mean correlation for a group of populations.

    Args:
        corr_matrix: Full correlation matrix
        indices: Indices of populations in the group

    Returns:
        Mean pairwise correlation within the group (excluding diagonal)
    """
    if len(indices) < 2:
        return 0.0

    submatrix = corr_matrix[np.ix_(indices, indices)]
    # Exclude diagonal (self-correlations)
    mask = ~np.eye(len(indices), dtype=bool)
    return float(np.mean(submatrix[mask]))


def compute_activity_correlation_matrix(
    activity_buffer: list[np.ndarray],
    sample_rate: int = 4,
) -> np.ndarray | None:
    """Compute correlation matrix from activity buffer.

    Args:
        activity_buffer: List of flattened activity arrays over time
        sample_rate: Sample every Nth cell to reduce matrix size

    Returns:
        Correlation matrix or None if insufficient data
    """
    if len(activity_buffer) < 2:
        return None

    # Stack and transpose: each column is one cell's time series
    data_matrix = np.array(activity_buffer)  # shape: (time, cells)

    # Subsample cells for efficiency
    n_cells = data_matrix.shape[1]
    cell_indices = np.arange(0, n_cells, sample_rate)
    data_subsampled = data_matrix[:, cell_indices]

    # Compute correlation
    if data_subsampled.shape[0] > 1:
        corr_matrix = np.corrcoef(data_subsampled.T)
        # Replace NaN with 0 (constant cells)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        return corr_matrix

    return None


# =============================================================================
# Synchronous Event Detection
# =============================================================================


def count_synchronous_events(
    activities: dict[str, dict[str, np.ndarray]],
    layers: list[str],
    cell_types: list[str],
    activity_threshold: float = 0.1,
    sync_threshold: float = 0.1,
) -> dict[str, dict[str, int]]:
    """Count synchronous events in each layer and cell type.

    A synchronous event occurs when the fraction of cells above threshold
    exceeds the sync_threshold.

    Args:
        activities: Nested dict of activities[layer][cell_type] = 2D array
        layers: List of layer names
        cell_types: List of cell type names
        activity_threshold: Minimum activity to count a cell as active
        sync_threshold: Fraction of cells that must be active for a sync event

    Returns:
        Dictionary with 'by_layer' and 'by_celltype' counts
    """
    events = {
        "by_layer": {layer: 0 for layer in layers},
        "by_celltype": {ct: 0 for ct in cell_types},
    }

    for layer in layers:
        for cell_type in cell_types:
            activity = activities.get(layer, {}).get(cell_type)
            if activity is None:
                continue

            # Count fraction of active cells
            n_active = np.sum(activity > activity_threshold)
            n_total = activity.size
            if n_total > 0:
                fraction_active = n_active / n_total
                if fraction_active >= sync_threshold:
                    events["by_layer"][layer] += 1
                    events["by_celltype"][cell_type] += 1

    return events


# =============================================================================
# Jacobian and Stability Computations
# =============================================================================


@timed
def build_jacobian_cached(
    A: np.ndarray,
    tau: np.ndarray,
    g_eff: np.ndarray,
    exp_factors: np.ndarray,
) -> np.ndarray:
    """Build Jacobian matrix using pre-computed exponential factors.

    This is a pure function for computing the Jacobian at a specific k-mode.

    Args:
        A: Connectivity matrix (n_pops x n_pops)
        tau: Time constants for each population
        g_eff: Effective gains at linearization point
        exp_factors: Pre-computed exp(-2π²k²σ²) factors

    Returns:
        Jacobian matrix
    """
    n_pops = len(tau)
    J = np.zeros((n_pops, n_pops))

    for i in range(n_pops):
        for j in range(n_pops):
            w_tilde = A[i, j] * exp_factors[i, j]
            if i == j:
                J[i, j] = -1.0 / tau[i] + (g_eff[i] * w_tilde) / tau[i]
            else:
                J[i, j] = (g_eff[i] * w_tilde) / tau[i]

    return J


def compute_exponential_cache(
    sigma: np.ndarray,
    k_squared_values: set[int],
    anatomical_grid_size: float,
) -> dict[int, np.ndarray]:
    """Pre-compute exponential factors for all unique k² values.

    Args:
        sigma: Spatial spread matrix (n_pops x n_pops) in μm
        k_squared_values: Set of unique k² values to cache
        anatomical_grid_size: Grid size in μm for normalization

    Returns:
        Dictionary mapping k² to exponential factor matrices
    """
    n_pops = sigma.shape[0]
    exp_cache = {}

    for k_squared in k_squared_values:
        exp_cache[k_squared] = np.zeros((n_pops, n_pops))
        for i in range(n_pops):
            for j in range(n_pops):
                sigma_ij = sigma[i, j] / anatomical_grid_size
                exp_cache[k_squared][i, j] = np.exp(-2 * np.pi**2 * k_squared * sigma_ij**2)

    return exp_cache


def get_unique_k_squared_values(n_modes: int) -> set[int]:
    """Get all unique k² values for the positive quadrant scan.

    Args:
        n_modes: Maximum mode number to scan

    Returns:
        Set of unique k² values
    """
    k_squared_set = set()
    for n1 in range(0, n_modes + 1):
        for n2 in range(0, n_modes + 1):
            k_squared_set.add(n1**2 + n2**2)
    return k_squared_set


# =============================================================================
# Gain Computations
# =============================================================================


def compute_b_fourier(
    thalamic_weights: np.ndarray,
    thalamic_sigma: np.ndarray,
    k_squared: float,
    anatomical_grid_size: float,
) -> np.ndarray:
    """Compute thalamic input B(k) in Fourier space with Gaussian filtering.

    Args:
        thalamic_weights: Thalamic connection weights to each population
        thalamic_sigma: Thalamic spread widths for each population (μm)
        k_squared: Square of the wavenumber k
        anatomical_grid_size: Grid size in μm

    Returns:
        B(k) vector: Thalamic input in Fourier space
    """
    n_pops = len(thalamic_weights)
    B_k = np.zeros(n_pops)

    for i in range(n_pops):
        sigma_thal = thalamic_sigma[i] / anatomical_grid_size
        spatial_filter = np.exp(-2 * np.pi**2 * k_squared * sigma_thal**2)
        B_k[i] = thalamic_weights[i] * spatial_filter

    return B_k


def compute_static_gain_at_k(
    J: np.ndarray,
    B_k: np.ndarray,
    tau: np.ndarray,
    g_eff: np.ndarray,
) -> float:
    """Compute static gain G(k) = ||(-J)^{-1} * diag(g_eff/tau) * B(k)||.

    Args:
        J: Jacobian matrix at this k
        B_k: Thalamic input vector at this k
        tau: Time constants
        g_eff: Effective gains

    Returns:
        Static gain value (L2 norm of response)
    """
    try:
        neg_J_inv = np.linalg.inv(-J)
        diag_g_tau = np.diag(g_eff / tau)
        response = neg_J_inv @ diag_g_tau @ B_k
        return float(np.linalg.norm(response))
    except np.linalg.LinAlgError:
        return 0.0


# =============================================================================
# Mean Rate Extraction
# =============================================================================


def extract_spatial_mean_rates(
    activities: dict[str, dict[str, np.ndarray]],
    layers: list[str] = None,
    cell_types: list[str] = None,
) -> np.ndarray | None:
    """Extract spatial mean rates from activity dictionary.

    Args:
        activities: Nested dict of activities[layer][cell_type] = 2D array
        layers: List of layers (default: ['L23', 'L4', 'L5'])
        cell_types: List of cell types (default: ['E', 'SST', 'PV'])

    Returns:
        Array of mean rates ordered by (layer, cell_type) or None if inactive
    """
    if layers is None:
        layers = ["L23", "L4", "L5"]
    if cell_types is None:
        cell_types = ["E", "SST", "PV"]

    mean_rates = []
    for layer in layers:
        for cell_type in cell_types:
            activity = activities.get(layer, {}).get(cell_type)
            if activity is None:
                return None
            mean_rates.append(activity.mean())

    mean_rates = np.array(mean_rates)

    # Check if network is active
    if np.all(mean_rates < 0.01):
        return None

    return mean_rates

