"""Core analysis class for descriptive network activity analysis."""

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from src.analysis.common import DEVELOPMENTAL_STAGES, PRESETS
from src.simulation import CorticalSimulation
from src.model.config import DT, GRID_SIZE, INTEGRATION_STEPS, grid_to_um

from .config import ANALYSIS_PARAMS, CELL_TYPES, LAYERS


def compute_structural_ei_balance(preset: dict[str, Any]) -> dict[str, Any]:
    """Compute structural E-I balance from preset connection strengths.

    Calculates the balance of excitatory vs inhibitory input to E cells,
    using connection amplitudes scaled by strength_scaling factors.

    Excitation sources: E cells (all layers) + thalamus
    Inhibition sources: SST cells + PV cells (all layers)

    Args:
        preset: Developmental preset dictionary containing connection_strengths
                and strength_scaling

    Returns:
        Dictionary with:
        - by_inhibitory_celltype: {"SST": total, "PV": total} - raw inhibition magnitudes
        - by_layer: {"L23": ratio, "L4": ratio, "L5": ratio} - E-I ratio per layer
        - total: float - network-wide E-I ratio
        - excitation_total: float - total excitatory input to E cells
        - inhibition_total: float - total inhibitory input to E cells
    """
    conn_strengths = preset["connection_strengths"]
    scaling = preset["strength_scaling"]

    layers = ["L23", "L4", "L5"]

    # Initialize accumulators
    excitation_by_layer = {layer: 0.0 for layer in layers}
    inhibition_by_layer = {layer: 0.0 for layer in layers}
    inhibition_by_celltype = {"SST": 0.0, "PV": 0.0}

    # Process all connections targeting E cells
    for conn_key, amplitude in conn_strengths.items():
        # Skip non-E targets
        if not conn_key.endswith("_E"):
            continue

        # Parse connection key to get source and target info
        if conn_key.startswith("thalamus_to_"):
            # Thalamic connection: thalamus_to_L4_E
            parts = conn_key.split("_")
            target_layer = parts[2]
            source_type = "thalamus"
            scaled_strength = abs(amplitude) * scaling.get("thalamus", 1.0)
            excitation_by_layer[target_layer] += scaled_strength

        elif "_to_" in conn_key:
            # Cortical connection: L4_E_to_L23_E or L4_SST_to_L23_E
            source_part, target_part = conn_key.split("_to_")
            source_parts = source_part.split("_")
            target_parts = target_part.split("_")

            source_layer = source_parts[0]
            source_cell = source_parts[1]
            target_layer = target_parts[0]
            target_cell = target_parts[1]

            # Only process connections to E cells
            if target_cell != "E":
                continue

            scaled_strength = abs(amplitude) * scaling.get(source_cell, 1.0)

            if source_cell == "E":
                excitation_by_layer[target_layer] += scaled_strength
            elif source_cell in ("SST", "PV"):
                inhibition_by_layer[target_layer] += scaled_strength
                inhibition_by_celltype[source_cell] += scaled_strength

    # Calculate E-I ratio per layer (avoid division by zero)
    ei_ratio_by_layer = {}
    for layer in layers:
        exc = excitation_by_layer[layer]
        inh = inhibition_by_layer[layer]
        if inh > 0:
            ei_ratio_by_layer[layer] = exc / inh
        else:
            ei_ratio_by_layer[layer] = float("inf") if exc > 0 else 1.0

    # Calculate total network E-I ratio
    total_excitation = sum(excitation_by_layer.values())
    total_inhibition = sum(inhibition_by_layer.values())

    if total_inhibition > 0:
        total_ratio = total_excitation / total_inhibition
    else:
        total_ratio = float("inf") if total_excitation > 0 else 1.0

    return {
        "by_inhibitory_celltype": inhibition_by_celltype,
        "by_layer": ei_ratio_by_layer,
        "total": total_ratio,
        "excitation_total": total_excitation,
        "inhibition_total": total_inhibition,
    }


class DescriptiveAnalysis:
    """Analyzes network activity patterns across developmental stages.

    Timing semantics:
    - All durations and intervals are in simulation-time milliseconds.
    - Each simulation.update() advances time by `update_dt_ms = DT * INTEGRATION_STEPS`.
    - Samples are recorded at the closest achievable times to the target sampling grid.
    """

    def __init__(self) -> None:
        """Initialize the analysis with simulation parameters."""
        self.simulation = CorticalSimulation()

        # The time quantum: each update() advances this many ms of simulation time
        self.update_dt_ms = DT * INTEGRATION_STEPS

        # Calculate sampling parameters
        self._setup_sampling_parameters()

        # Print concise timing info
        print(
            f"Analysis setup: warmup={self.warmup_ms:.1f}ms, duration={self.duration_ms:.1f}ms, "
            f"target_interval={self.sampling_interval_ms:.1f}ms"
        )
        print(
            f"  Update quantum: {self.update_dt_ms:.1f}ms (DT={DT}ms × INTEGRATION_STEPS={INTEGRATION_STEPS})"
        )
        print(f"  Expected samples: ~{self.n_samples} (recorded at nearest achievable times)")

    def _setup_sampling_parameters(self) -> None:
        """Set up timing and sampling parameters.

        All times are in simulation-time milliseconds.
        """
        # Convert config durations (in seconds) to milliseconds
        self.warmup_ms = ANALYSIS_PARAMS["warmup_duration"] * 1000
        self.duration_ms = ANALYSIS_PARAMS["simulation_duration"] * 1000

        # Target sampling interval in ms (from config)
        self.sampling_interval_ms = ANALYSIS_PARAMS["sampling_interval"]

        # Estimate number of samples (actual count depends on time accumulator)
        self.n_samples = int(self.duration_ms / self.sampling_interval_ms)

    def run_simulation_for_stage(self, stage_name: str) -> dict[str, Any]:
        """Run simulation for a single developmental stage and collect timeseries."""
        print(f"Running simulation for {stage_name}...")

        # Apply preset and reset simulation
        preset = PRESETS[stage_name]
        self.simulation.apply_preset(preset)
        self.simulation.reset()

        # Get thalamic alpha for this stage
        alpha = preset["thalamic_alpha"]

        # Initialize timeseries data structure
        timeseries = self._initialize_timeseries_data(stage_name, alpha)

        # Collect data over time
        self._collect_timeseries_data(timeseries, alpha)

        return timeseries

    def _initialize_timeseries_data(self, stage_name: str, alpha: float) -> dict[str, Any]:
        """Initialize the timeseries data structure."""
        return {
            "activities": [],  # List of activity dictionaries at each timepoint
            "time": [],
            "stage": stage_name,
            "alpha": alpha,
        }

    def _collect_timeseries_data(self, timeseries: dict[str, Any], alpha: float) -> None:
        """Collect timeseries data by running simulation.

        Uses a time-accumulator scheme:
        - Warmup: advance simulation while elapsed_ms < warmup_ms
        - Recording: advance until reaching next sample time, then record
        """
        elapsed_ms = 0.0

        # Warmup period: let network settle to equilibrium
        if self.warmup_ms > 0:
            n_warmup_updates = int(np.ceil(self.warmup_ms / self.update_dt_ms))
            for _ in tqdm(
                range(n_warmup_updates),
                desc=f"  {timeseries['stage']} warmup",
                unit="updates",
                leave=False,
            ):
                self.simulation.update(alpha=alpha)
                elapsed_ms += self.update_dt_ms

        # Reset elapsed time for recording phase (warmup doesn't count toward recorded time)
        elapsed_ms = 0.0
        next_sample_time_ms = 0.0

        # Data collection period using time accumulator
        with tqdm(
            total=self.n_samples,
            desc=f"  {timeseries['stage']} recording",
            unit="samples",
            leave=False,
        ) as pbar:
            while len(timeseries["activities"]) < self.n_samples:
                # Advance simulation by one update quantum
                activities = self.simulation.update(alpha=alpha)
                elapsed_ms += self.update_dt_ms

                # Check if we've reached or passed the next sample time
                if elapsed_ms >= next_sample_time_ms:
                    # Record sample at the current (closest achievable) time
                    timeseries["activities"].append(activities.copy())
                    timeseries["time"].append(elapsed_ms)

                    # Advance to next target sample time
                    next_sample_time_ms += self.sampling_interval_ms
                    pbar.update(1)

                # Safety: stop if we've simulated far beyond expected duration
                if elapsed_ms > self.duration_ms + self.sampling_interval_ms:
                    break

    def process_activity_data(self, timeseries_data: dict[str, Any]) -> dict[str, Any]:
        """Process timeseries data to extract key metrics."""
        activities = timeseries_data["activities"]
        stage = timeseries_data["stage"]

        # Initialize processed data structure
        processed = self._initialize_processed_data(timeseries_data)

        # Process each timepoint
        threshold = ANALYSIS_PARAMS["activity_threshold"]

        for activity_dict in activities:
            self._process_single_timepoint(activity_dict, processed, threshold)

        # Convert lists to arrays for efficiency
        self._convert_to_arrays(processed)

        # Compute functional E-I balance (activity-weighted)
        preset = PRESETS[stage]
        processed["functional_ei_balance"] = self._compute_functional_ei_balance(
            processed, preset
        )

        return processed

    def _initialize_processed_data(self, timeseries_data: dict[str, Any]) -> dict[str, Any]:
        """Initialize the processed data structure."""
        return {
            "stage": timeseries_data["stage"],
            "time": timeseries_data["time"],
            "average_rates": {layer: {cell: [] for cell in CELL_TYPES} for layer in LAYERS},
            "active_fractions": {layer: {cell: [] for cell in CELL_TYPES} for layer in LAYERS},
            "spatial_activities": {layer: {cell: [] for cell in CELL_TYPES} for layer in LAYERS},
        }

    def _process_single_timepoint(
        self, activity_dict: dict[str, Any], processed: dict[str, Any], threshold: float
    ) -> None:
        """Process activity data for a single timepoint."""
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                activity = activity_dict[layer][cell_type]

                # Average firing rate across space
                avg_rate = np.mean(activity)
                processed["average_rates"][layer][cell_type].append(avg_rate)

                # Fraction of active cells
                active_fraction = np.mean(activity > threshold)
                processed["active_fractions"][layer][cell_type].append(active_fraction)

                # Store flattened spatial activity for correlation analysis
                processed["spatial_activities"][layer][cell_type].append(activity.flatten())

    def _convert_to_arrays(self, processed: dict[str, Any]) -> None:
        """Convert lists to numpy arrays for efficient processing."""
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                processed["average_rates"][layer][cell_type] = np.array(
                    processed["average_rates"][layer][cell_type]
                )
                processed["active_fractions"][layer][cell_type] = np.array(
                    processed["active_fractions"][layer][cell_type]
                )
                processed["spatial_activities"][layer][cell_type] = np.array(
                    processed["spatial_activities"][layer][cell_type]
                )

    def _compute_functional_ei_balance(
        self, processed: dict[str, Any], preset: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute functional E-I balance from activity data and connection strengths.

        Calculates activity-weighted inputs to E cells:
        Contribution = mean_activity × connection_strength

        Args:
            processed: Processed activity data with average_rates
            preset: Developmental preset with connection strengths and scaling

        Returns:
            Dictionary with:
            - by_inhibitory_celltype: mean/std of SST and PV inhibition
            - by_layer: mean/std of total inhibition per layer
            - total: mean/std of network-wide inhibition
        """
        conn_strengths = preset["connection_strengths"]
        scaling = preset["strength_scaling"]

        # Get mean activities across time for each population
        mean_activities = {}
        for layer in LAYERS:
            mean_activities[layer] = {}
            for cell_type in CELL_TYPES:
                # Average across both time and space
                mean_activities[layer][cell_type] = np.mean(
                    processed["average_rates"][layer][cell_type]
                )

        # Initialize timeseries accumulators for each timepoint
        n_timepoints = len(processed["time"])
        inhibition_timeseries = {
            "SST": np.zeros(n_timepoints),
            "PV": np.zeros(n_timepoints),
            "by_layer": {layer: np.zeros(n_timepoints) for layer in LAYERS},
        }

        # Calculate activity-weighted inhibition at each timepoint
        for t in range(n_timepoints):
            for layer in LAYERS:
                layer_inhibition = 0.0

                # SST contribution to E cells in this layer
                sst_activity = processed["average_rates"][layer]["SST"][t]
                for source_layer in LAYERS:
                    conn_key = f"{source_layer}_SST_to_{layer}_E"
                    if conn_key in conn_strengths:
                        strength = abs(conn_strengths[conn_key]) * scaling.get("SST", 1.0)
                        contribution = sst_activity * strength
                        inhibition_timeseries["SST"][t] += contribution
                        layer_inhibition += contribution

                # PV contribution to E cells in this layer
                pv_activity = processed["average_rates"][layer]["PV"][t]
                for source_layer in LAYERS:
                    conn_key = f"{source_layer}_PV_to_{layer}_E"
                    if conn_key in conn_strengths:
                        strength = abs(conn_strengths[conn_key]) * scaling.get("PV", 1.0)
                        contribution = pv_activity * strength
                        inhibition_timeseries["PV"][t] += contribution
                        layer_inhibition += contribution

                inhibition_timeseries["by_layer"][layer][t] = layer_inhibition

        # Compute summary statistics
        return {
            "by_inhibitory_celltype": {
                "SST": {
                    "mean": np.mean(inhibition_timeseries["SST"]),
                    "std": np.std(inhibition_timeseries["SST"]),
                },
                "PV": {
                    "mean": np.mean(inhibition_timeseries["PV"]),
                    "std": np.std(inhibition_timeseries["PV"]),
                },
            },
            "by_layer": {
                layer: {
                    "mean": np.mean(inhibition_timeseries["by_layer"][layer]),
                    "std": np.std(inhibition_timeseries["by_layer"][layer]),
                }
                for layer in LAYERS
            },
            "total": {
                "mean": np.mean(
                    inhibition_timeseries["SST"] + inhibition_timeseries["PV"]
                ),
                "std": np.std(
                    inhibition_timeseries["SST"] + inhibition_timeseries["PV"]
                ),
            },
        }

    def calculate_correlations_and_events(self, processed_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate pairwise correlations and synchronous events."""
        # Collect all cell timeseries for correlation analysis
        all_cells, _cell_labels, layer_labels, celltype_labels = self._collect_all_cells(
            processed_data
        )

        # Calculate correlations using per-group approach (faster than global matrix)
        correlations = self._calculate_correlations(all_cells, celltype_labels, layer_labels)

        # Calculate synchronous events (use full data)
        sync_events = self._calculate_synchronous_events(all_cells, celltype_labels, layer_labels)

        return {"correlations": correlations, "synchronous_events": sync_events}

    def calculate_dimensionality(self, processed_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate network dimensionality using normalized participation ratio.
        
        Uses PCA to compute eigenvalues, then calculates participation ratio:
        PR = (Σλᵢ)² / Σ(λᵢ²), normalized by number of cells.
        
        Args:
            processed_data: Processed activity data with spatial_activities
            
        Returns:
            Dictionary with:
            - total: normalized PR for entire network
            - by_celltype: normalized PR for each cell type
            - by_layer: normalized PR for each layer
        """
        # Collect all cell timeseries (use all timepoints, not just active)
        all_cells, _cell_labels, layer_labels, celltype_labels = self._collect_all_cells(
            processed_data
        )
        min_rate = ANALYSIS_PARAMS["min_mean_rate"]

        # Calculate dimensionality for total network
        total_dim = (
            np.nan
            if np.mean(all_cells) < min_rate
            else self._calculate_participation_ratio(all_cells)
        )

        # Calculate dimensionality by cell type
        dim_by_celltype = {}
        for cell_type in CELL_TYPES:
            cell_indices = [i for i, ct in enumerate(celltype_labels) if ct == cell_type]
            if len(cell_indices) > 1:
                celltype_data = all_cells[:, cell_indices]
                dim_by_celltype[cell_type] = (
                    np.nan
                    if np.mean(celltype_data) < min_rate
                    else self._calculate_participation_ratio(celltype_data)
                )
            else:
                dim_by_celltype[cell_type] = np.nan

        # Calculate dimensionality by layer
        dim_by_layer = {}
        for layer in LAYERS:
            layer_indices = [i for i, l in enumerate(layer_labels) if l == layer]
            if len(layer_indices) > 1:
                layer_data = all_cells[:, layer_indices]
                dim_by_layer[layer] = (
                    np.nan
                    if np.mean(layer_data) < min_rate
                    else self._calculate_participation_ratio(layer_data)
                )
            else:
                dim_by_layer[layer] = np.nan
        
        return {
            "total": total_dim,
            "by_celltype": dim_by_celltype,
            "by_layer": dim_by_layer,
        }
    
    def _calculate_participation_ratio(self, data: np.ndarray) -> float:
        """Calculate normalized participation ratio from PCA eigenvalues.
        
        Args:
            data: Activity matrix (n_timepoints × n_cells)
            
        Returns:
            Normalized participation ratio (0-1 range)
        """
        n_timepoints, n_cells = data.shape
        
        # Need at least 2 timepoints for PCA
        if n_timepoints < 2 or n_cells < 1:
            return np.nan
        
        # Center the data (subtract mean across time for each cell)
        data_centered = data - np.mean(data, axis=0, keepdims=True)
        
        # Perform SVD to get eigenvalues (singular values squared = eigenvalues)
        # Use economic SVD for efficiency
        try:
            _, s, _ = np.linalg.svd(data_centered, full_matrices=False)
            eigenvalues = s**2 / (n_timepoints - 1)  # Convert singular values to eigenvalues
        except np.linalg.LinAlgError:
            return np.nan
        
        # Filter out near-zero eigenvalues
        threshold = ANALYSIS_PARAMS["dimensionality_min_variance"]
        eigenvalues_filtered = eigenvalues[eigenvalues > threshold]
        
        if len(eigenvalues_filtered) == 0:
            return 0.0
        
        # Calculate participation ratio: PR = (Σλᵢ)² / Σ(λᵢ²)
        sum_eig = np.sum(eigenvalues_filtered)
        sum_eig_sq = np.sum(eigenvalues_filtered**2)
        
        if sum_eig_sq == 0:
            return 0.0
        
        pr = (sum_eig**2) / sum_eig_sq
        
        # Normalize by number of cells (so PR ranges from 0 to 1)
        normalized_pr = pr / n_cells
        
        return float(normalized_pr)

    def calculate_spatial_correlations(self, processed_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate spatial correlation functions C(r) and correlation length ξ.

        For each layer and cell type, computes pairwise correlations between
        spatial locations as a function of distance, then extracts the
        correlation length (distance where C(r) drops to C(0)/e).

        Args:
            processed_data: Processed activity data with spatial_activities

        Returns:
            Dictionary with C(r) curves and correlation lengths for each
            layer/cell type combination
        """
        n_bins = ANALYSIS_PARAMS["spatial_correlation_bins"]

        # Precompute distance matrix for the grid (once, reuse for all)
        distance_matrix = self._compute_distance_matrix(GRID_SIZE)
        max_distance = np.max(distance_matrix)

        # Create distance bins that extend to the maximum distance
        # Use n_bins bins from 0 to max_distance
        bin_edges = np.linspace(0, max_distance, n_bins + 1)
        bin_centers_grid = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create distance array: start at 0, use bin centers, end at max_distance
        # This ensures curves extend from x=0 to x=max_distance
        distances_grid = np.concatenate([[0.0], bin_centers_grid, [max_distance]])
        distances_um = grid_to_um(distances_grid)
        min_rate = ANALYSIS_PARAMS["min_mean_rate"]

        result = {}
        for layer in LAYERS:
            result[layer] = {}
            for cell_type in CELL_TYPES:
                # Get spatial activity: (n_timepoints, n_cells) where n_cells = grid_size^2
                spatial_data = processed_data["spatial_activities"][layer][cell_type]

                if np.mean(spatial_data) < min_rate:
                    result[layer][cell_type] = {
                        "distances_um": distances_um,
                        "correlations": np.full_like(distances_um, np.nan),
                        "xi_um": np.nan,
                    }
                    continue

                # Compute pairwise correlations between spatial locations
                # Transpose so columns are timepoints, rows are cells
                corr_matrix = np.corrcoef(spatial_data.T)

                # Bin correlations by distance
                correlations_binned = self._bin_correlations_by_distance(
                    corr_matrix, distance_matrix, bin_edges
                )
                
                # Calculate correlation at exactly distance=0 (diagonal elements)
                corr_at_zero = np.nanmean(np.diag(corr_matrix))
                
                # For the last point, use the correlation in the final bin
                # This ensures the curve extends to max_distance
                corr_at_max = correlations_binned[-1] if len(correlations_binned) > 0 else np.nan
                
                # Build full correlation array: start at 0, use bin centers, end at max
                correlations_with_endpoints = np.concatenate([[corr_at_zero], correlations_binned, [corr_at_max]])

                # Extract correlation length (use full distances array including endpoints)
                xi_grid = self._extract_correlation_length(
                    distances_grid, correlations_with_endpoints
                )
                xi_um = grid_to_um(xi_grid) if not np.isnan(xi_grid) else np.nan

                result[layer][cell_type] = {
                    "distances_um": distances_um,
                    "correlations": correlations_with_endpoints,
                    "xi_um": xi_um,
                }

        return {"spatial_correlations": result}

    def _compute_distance_matrix(self, grid_size: int) -> np.ndarray:
        """Compute Euclidean distance matrix between all grid points.

        Args:
            grid_size: Size of the square grid (grid_size × grid_size)

        Returns:
            Distance matrix of shape (n_cells, n_cells) where n_cells = grid_size^2
        """
        n_cells = grid_size * grid_size

        # Create coordinate arrays for each cell
        coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])

        # Compute pairwise Euclidean distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=2))

        return distance_matrix

    def _bin_correlations_by_distance(
        self,
        corr_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        bin_edges: np.ndarray,
    ) -> np.ndarray:
        """Bin pairwise correlations by distance.

        Args:
            corr_matrix: Pairwise correlation matrix (n_cells × n_cells)
            distance_matrix: Pairwise distance matrix (n_cells × n_cells)
            bin_edges: Edges of distance bins

        Returns:
            Array of mean correlations for each distance bin
        """
        n_bins = len(bin_edges) - 1
        correlations_binned = np.zeros(n_bins)

        # Use upper triangle to avoid counting pairs twice
        triu_indices = np.triu_indices_from(corr_matrix, k=1)
        distances_flat = distance_matrix[triu_indices]
        correlations_flat = corr_matrix[triu_indices]

        # Bin the correlations
        for i in range(n_bins):
            mask = (distances_flat >= bin_edges[i]) & (distances_flat < bin_edges[i + 1])
            if np.any(mask):
                correlations_binned[i] = np.nanmean(correlations_flat[mask])
            else:
                correlations_binned[i] = np.nan

        return correlations_binned

    def _extract_correlation_length(
        self, distances: np.ndarray, correlations: np.ndarray
    ) -> float:
        """Extract correlation length ξ from C(r) curve.

        Correlation length is defined as the distance where C(r) drops to C(0)/e.

        Args:
            distances: Array of distance bin centers (grid units)
            correlations: Array of mean correlations at each distance

        Returns:
            Correlation length in grid units, or NaN if cannot be determined
        """
        # Find C(0) - correlation at smallest distance
        valid_mask = ~np.isnan(correlations)
        if not np.any(valid_mask):
            return np.nan

        # Use first valid correlation as C(0)
        first_valid_idx = np.argmax(valid_mask)
        c0 = correlations[first_valid_idx]

        if c0 <= 0:
            return np.nan

        # Threshold is C(0)/e
        threshold = c0 / np.e

        # Find where correlation drops below threshold
        for i in range(first_valid_idx, len(correlations)):
            if not np.isnan(correlations[i]) and correlations[i] <= threshold:
                # Interpolate to get more precise crossing point
                if i > first_valid_idx and not np.isnan(correlations[i - 1]):
                    # Linear interpolation between i-1 and i
                    c_prev = correlations[i - 1]
                    c_curr = correlations[i]
                    d_prev = distances[i - 1]
                    d_curr = distances[i]
                    # Solve: threshold = c_prev + (c_curr - c_prev) * (xi - d_prev) / (d_curr - d_prev)
                    if c_curr != c_prev:
                        xi = d_prev + (threshold - c_prev) * (d_curr - d_prev) / (c_curr - c_prev)
                        return float(xi)
                return float(distances[i])

        # Never crossed threshold - return max distance (indicates long-range correlation)
        return float(distances[-1])

    def _collect_all_cells(
        self, processed_data: dict[str, Any]
    ) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """Collect all cell timeseries for correlation analysis."""
        all_cells = []
        cell_labels = []
        layer_labels = []
        celltype_labels = []

        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                # Each row is a timepoint, each column is a spatial location
                spatial_data = processed_data["spatial_activities"][layer][cell_type]
                _n_timepoints, n_locations = spatial_data.shape

                # Add each location as a separate "cell"
                for loc in range(n_locations):
                    all_cells.append(spatial_data[:, loc])
                    cell_labels.append(f"{layer}_{cell_type}_{loc}")
                    layer_labels.append(layer)
                    celltype_labels.append(cell_type)

        # Convert to array (n_timepoints x n_cells)
        return np.array(all_cells).T, cell_labels, layer_labels, celltype_labels

    def _calculate_correlations(
        self, all_cells: np.ndarray, celltype_labels: list[str], layer_labels: list[str]
    ) -> dict[str, Any]:
        """Calculate average correlations by different groupings.
        
        Uses per-group correlation matrices (faster than one global matrix).
        This matches the dashboard's approach for consistency.
        
        Args:
            all_cells: Activity data (n_timepoints, n_cells)
            celltype_labels: Cell type for each cell
            layer_labels: Layer for each cell
            
        Returns:
            Dictionary with total, by_celltype, and by_layer correlations
        """
        def compute_group_correlation(data: np.ndarray) -> float:
            """Compute mean pairwise correlation for a group.

            Best practice: return NaN when correlation is undefined (too few
            points, constant data). Do not replace NaN with 0, since 0 denotes
            no linear relationship; NaN denotes cannot compute.
            """
            if data.shape[1] < 2 or data.shape[0] < 2:
                return np.nan

            # Transpose for corrcoef: each row = one cell's timeseries
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(data.T)

            if corr_matrix.shape[0] <= 1:
                return np.nan
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            # nanmean: average only defined pairs; if all NaN (e.g. constant data) -> NaN
            mean_corr = np.nanmean(corr_matrix[mask])
            return np.nan if not np.isfinite(mean_corr) else float(mean_corr)

        min_rate = ANALYSIS_PARAMS["min_mean_rate"]

        def gated_correlation(data: np.ndarray) -> float:
            """Return NaN if group mean rate is below threshold; else compute correlation."""
            if np.mean(data) < min_rate:
                return np.nan
            return compute_group_correlation(data)

        # Calculate total correlation (all cells)
        avg_correlation_total = gated_correlation(all_cells)

        # Calculate correlations by cell type
        avg_correlation_by_celltype = {}
        for cell_type in CELL_TYPES:
            cell_indices = [i for i, ct in enumerate(celltype_labels) if ct == cell_type]
            if len(cell_indices) > 1:
                celltype_data = all_cells[:, cell_indices]
                avg_correlation_by_celltype[cell_type] = gated_correlation(celltype_data)
            else:
                avg_correlation_by_celltype[cell_type] = np.nan

        # Calculate correlations by layer
        avg_correlation_by_layer = {}
        for layer in LAYERS:
            layer_indices = [i for i, l in enumerate(layer_labels) if l == layer]
            if len(layer_indices) > 1:
                layer_data = all_cells[:, layer_indices]
                avg_correlation_by_layer[layer] = gated_correlation(layer_data)
            else:
                avg_correlation_by_layer[layer] = np.nan

        return {
            "total": avg_correlation_total,
            "by_celltype": avg_correlation_by_celltype,
            "by_layer": avg_correlation_by_layer,
        }

    def _calculate_synchronous_events(
        self, all_cells: np.ndarray, celltype_labels: list[str], layer_labels: list[str]
    ) -> dict[str, Any]:
        """Calculate synchronous events for different groupings."""
        threshold = ANALYSIS_PARAMS["synchronous_event_threshold"]
        activity_threshold = ANALYSIS_PARAMS["activity_threshold"]

        sync_events = {
            "total": self._count_network_events(all_cells, threshold, activity_threshold),
            "by_celltype": self._count_events_by_celltype(
                all_cells, celltype_labels, threshold, activity_threshold
            ),
            "by_layer": self._count_events_by_layer(
                all_cells, layer_labels, threshold, activity_threshold
            ),
        }

        return sync_events

    def _count_network_events(
        self, all_cells: np.ndarray, threshold: float, activity_threshold: float
    ) -> int:
        """Count synchronous events across the entire network."""
        count = 0
        for t in range(all_cells.shape[0]):
            active_fraction = np.mean(all_cells[t] > activity_threshold)
            if active_fraction > threshold:
                count += 1
        return count

    def _count_events_by_celltype(
        self,
        all_cells: np.ndarray,
        celltype_labels: list[str],
        threshold: float,
        activity_threshold: float,
    ) -> dict[str, int]:
        """Count synchronous events by cell type."""
        events_by_celltype = {ct: 0 for ct in CELL_TYPES}

        for cell_type in CELL_TYPES:
            cell_indices = [i for i, ct in enumerate(celltype_labels) if ct == cell_type]
            celltype_data = all_cells[:, cell_indices]

            for t in range(celltype_data.shape[0]):
                active_fraction = np.mean(celltype_data[t] > activity_threshold)
                if active_fraction > threshold:
                    events_by_celltype[cell_type] += 1

        return events_by_celltype

    def _count_events_by_layer(
        self,
        all_cells: np.ndarray,
        layer_labels: list[str],
        threshold: float,
        activity_threshold: float,
    ) -> dict[str, int]:
        """Count synchronous events by layer."""
        events_by_layer = {l: 0 for l in LAYERS}

        for layer in LAYERS:
            layer_indices = [i for i, l in enumerate(layer_labels) if l == layer]
            layer_data = all_cells[:, layer_indices]

            for t in range(layer_data.shape[0]):
                active_fraction = np.mean(layer_data[t] > activity_threshold)
                if active_fraction > threshold:
                    events_by_layer[layer] += 1

        return events_by_layer

    def compute_structural_ei_balance_all_stages(self) -> dict[str, dict[str, Any]]:
        """Compute structural E-I balance for all developmental stages.

        This is a static analysis of preset parameters - no simulation needed.

        Returns:
            Dictionary mapping stage names to their E-I balance results
        """
        ei_balance_results = {}
        for stage in DEVELOPMENTAL_STAGES:
            preset = PRESETS[stage]
            ei_balance_results[stage] = compute_structural_ei_balance(preset)
        return ei_balance_results

    def run_analysis(self) -> dict[str, Any]:
        """Run complete analysis across all developmental stages."""
        print("Starting descriptive activity analysis...")
        start_time = time.time()

        results = {}

        # Compute structural E-I balance (no simulation needed)
        print("\nComputing structural E-I balance...")
        ei_balance = self.compute_structural_ei_balance_all_stages()

        # Progress bar for developmental stages
        for stage in tqdm(DEVELOPMENTAL_STAGES, desc="Processing stages", unit="stage"):
            print(f"\n=== Processing {stage} ===")

            # Run simulation and collect timeseries
            timeseries = self.run_simulation_for_stage(stage)

            # Process activity data
            processed = self.process_activity_data(timeseries)

            # Calculate correlations and events
            correlations_events = self.calculate_correlations_and_events(processed)

            # Calculate dimensionality
            dimensionality = self.calculate_dimensionality(processed)

            # Calculate spatial correlations C(r) and correlation length
            spatial_corr = self.calculate_spatial_correlations(processed)

            # Combine results (including E-I balance for this stage)
            results[stage] = {
                **processed,
                **correlations_events,
                **spatial_corr,
                "dimensionality": dimensionality,
                "structural_ei_balance": ei_balance[stage],
            }

        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.1f} seconds")

        return results
