"""Core analysis class for descriptive network activity analysis."""

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from src.main import CorticalSimulation
from src.model.config import DT, seed_random

from .config import ANALYSIS_PARAMS, CELL_TYPES, DEVELOPMENTAL_STAGES, LAYERS, PRESETS


class DescriptiveAnalysis:
    """Analyzes network activity patterns across developmental stages."""

    def __init__(self) -> None:
        """Initialize the analysis with simulation parameters."""
        self.simulation = CorticalSimulation()
        self.dt = DT  # From model config

        # Calculate sampling parameters
        self._setup_sampling_parameters()

        print(
            f"Analysis setup: {self.warmup_ms}ms warmup + {self._get_duration_ms()}ms duration, "
            f"{self.n_timepoints} timepoints, sampling every {self.sampling_interval}ms"
        )

    def _setup_sampling_parameters(self) -> None:
        """Set up timing and sampling parameters."""
        duration_ms = self._get_duration_ms()
        self.warmup_ms = ANALYSIS_PARAMS["warmup_duration"] * 1000
        self.sampling_interval = ANALYSIS_PARAMS["sampling_interval"]
        self.n_warmup_steps = int(self.warmup_ms / self.sampling_interval)
        self.n_timepoints = int(duration_ms / self.sampling_interval)
        self.steps_per_sample = int(self.sampling_interval / self.dt)

    def _get_duration_ms(self) -> float:
        """Get simulation duration in milliseconds."""
        return ANALYSIS_PARAMS["simulation_duration"] * 1000

    def _apply_preset(self, preset: dict[str, Any]) -> None:
        """Apply developmental stage preset to simulation."""
        self._update_connection_strengths(preset)
        self._update_scaling(preset)
        self._update_time_constants(preset)
        self._update_background_input(preset)
        self._update_thalamic_widths(preset)
        self._update_outgoing_widths(preset)
        self._update_thalamic_params(preset)

    def _update_connection_strengths(self, preset: dict[str, Any]) -> None:
        """Update connection strengths from preset."""
        if "connection_strengths" not in preset:
            return

        for conn_key, strength in preset["connection_strengths"].items():
            source_layer, source_cell, target_layer, target_cell = self._parse_connection_key(
                conn_key
            )
            self.simulation.connectivity.set_connection_strength(
                source_layer, source_cell, target_layer, target_cell, strength
            )

    def _update_scaling(self, preset: dict[str, Any]) -> None:
        """Update strength scaling parameters."""
        if "strength_scaling" in preset:
            for cell_type, scaling in preset["strength_scaling"].items():
                self.simulation.set_strength_scaling(cell_type, scaling)

    def _update_time_constants(self, preset: dict[str, Any]) -> None:
        """Update time constants from preset."""
        if "time_constants" in preset:
            for cell_type, tau in preset["time_constants"].items():
                self.simulation.set_time_constant(cell_type, tau)

    def _update_background_input(self, preset: dict[str, Any]) -> None:
        """Update background input from preset."""
        if "background_input" not in preset:
            return

        for cell_type, value in preset["background_input"].items():
            self.simulation.set_background_input(cell_type, value)

    def _update_thalamic_widths(self, preset: dict[str, Any]) -> None:
        """Update thalamic connection widths (sigma values)."""
        if "thalamic_widths" not in preset:
            return

        for cell_type, sigma in preset["thalamic_widths"].items():
            for target_layer in LAYERS:
                self.simulation.set_connection_sigma(
                    "thalamus", None, target_layer, cell_type, sigma
                )

    def _update_thalamic_params(self, preset: dict[str, Any]) -> None:
        """Update thalamic developmental parameters."""
        self.simulation.update_thalamic_params(preset)

    def _update_outgoing_widths(self, preset: dict[str, Any]) -> None:
        """Update outgoing connection widths (sigma values for lateral connections)."""
        if "outgoing_widths" not in preset:
            return

        for cell_type, sigma in preset["outgoing_widths"].items():
            for source_layer in LAYERS:
                for target_layer in LAYERS:
                    for target_cell in CELL_TYPES:
                        self.simulation.set_connection_sigma(
                            source_layer, cell_type, target_layer, target_cell, sigma
                        )

    def _parse_connection_key(self, conn_key: str) -> tuple[str, str, str, str]:
        """Parse connection key into source and target components."""
        parts = conn_key.split("_to_")
        source_parts = parts[0].split("_")
        target_parts = parts[1].split("_")

        if source_parts[0] == "thalamus":
            return "thalamus", None, target_parts[0], target_parts[1]
        else:
            return source_parts[0], source_parts[1], target_parts[0], target_parts[1]

    def run_simulation_for_stage(self, stage_name: str) -> dict[str, Any]:
        """Run simulation for a single developmental stage and collect timeseries."""
        print(f"Running simulation for {stage_name}...")

        # Apply preset and reset simulation
        preset = PRESETS[stage_name]
        self._apply_preset(preset)
        self.simulation.reset()

        # Set random seed for reproducibility
        seed_random()

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
        """Collect timeseries data by running simulation."""
        # Warmup period: let network settle to equilibrium
        if self.n_warmup_steps > 0:
            for _ in tqdm(
                range(self.n_warmup_steps),
                desc=f"  {timeseries['stage']} warmup",
                unit="steps",
                leave=False,
            ):
                self.simulation.update(alpha=alpha)

        # Data collection period
        for t_idx in tqdm(
            range(self.n_timepoints),
            desc=f"  {timeseries['stage']} recording",
            unit="steps",
            leave=False,
        ):
            # Update simulation
            activities = self.simulation.update(alpha=alpha)

            # Store activities and time
            timeseries["activities"].append(activities.copy())
            timeseries["time"].append(t_idx * self.sampling_interval)

    def process_activity_data(self, timeseries_data: dict[str, Any]) -> dict[str, Any]:
        """Process timeseries data to extract key metrics."""
        activities = timeseries_data["activities"]

        # Initialize processed data structure
        processed = self._initialize_processed_data(timeseries_data)

        # Process each timepoint
        threshold = ANALYSIS_PARAMS["activity_threshold"]

        for activity_dict in activities:
            self._process_single_timepoint(activity_dict, processed, threshold)

        # Convert lists to arrays for efficiency
        self._convert_to_arrays(processed)

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

    def calculate_correlations_and_events(self, processed_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate pairwise correlations and synchronous events."""
        # Collect all cell timeseries for correlation analysis
        all_cells, _cell_labels, layer_labels, celltype_labels = self._collect_all_cells(
            processed_data
        )

        # Filter to active periods only (exclude timepoints where network is mostly silent)
        threshold = ANALYSIS_PARAMS["correlation_activity_threshold"]
        active_timepoints = np.mean(all_cells, axis=1) > threshold
        all_cells_active = all_cells[active_timepoints, :]

        # Calculate correlation matrix (only on active periods)
        if all_cells_active.shape[0] > 1:
            corr_matrix = np.corrcoef(all_cells_active.T)
        else:
            # Fallback: if no active periods, return NaN correlations
            corr_matrix = np.full((all_cells.shape[1], all_cells.shape[1]), np.nan)

        # Calculate correlations
        correlations = self._calculate_correlations(corr_matrix, celltype_labels, layer_labels)

        # Calculate synchronous events (use full data)
        sync_events = self._calculate_synchronous_events(all_cells, celltype_labels, layer_labels)

        return {"correlations": correlations, "synchronous_events": sync_events}

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
        self, corr_matrix: np.ndarray, celltype_labels: list[str], layer_labels: list[str]
    ) -> dict[str, Any]:
        """Calculate average correlations by different groupings."""
        # Extract upper triangle for total average correlation
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        all_correlations = corr_matrix[mask]
        avg_correlation_total = np.nanmean(all_correlations)

        # Calculate correlations by cell type
        avg_correlation_by_celltype = {}
        for cell_type in CELL_TYPES:
            cell_indices = [i for i, ct in enumerate(celltype_labels) if ct == cell_type]
            if len(cell_indices) > 1:
                ct_corr = corr_matrix[np.ix_(cell_indices, cell_indices)]
                ct_mask = np.triu(np.ones_like(ct_corr, dtype=bool), k=1)
                avg_correlation_by_celltype[cell_type] = np.nanmean(ct_corr[ct_mask])

        # Calculate correlations by layer
        avg_correlation_by_layer = {}
        for layer in LAYERS:
            layer_indices = [i for i, l in enumerate(layer_labels) if l == layer]
            if len(layer_indices) > 1:
                l_corr = corr_matrix[np.ix_(layer_indices, layer_indices)]
                l_mask = np.triu(np.ones_like(l_corr, dtype=bool), k=1)
                avg_correlation_by_layer[layer] = np.nanmean(l_corr[l_mask])

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

    def run_analysis(self) -> dict[str, Any]:
        """Run complete analysis across all developmental stages."""
        print("Starting descriptive activity analysis...")
        start_time = time.time()

        results = {}

        # Progress bar for developmental stages
        for stage in tqdm(DEVELOPMENTAL_STAGES, desc="Processing stages", unit="stage"):
            print(f"\n=== Processing {stage} ===")

            # Run simulation and collect timeseries
            timeseries = self.run_simulation_for_stage(stage)

            # Process activity data
            processed = self.process_activity_data(timeseries)

            # Calculate correlations and events
            correlations_events = self.calculate_correlations_and_events(processed)

            # Combine results
            results[stage] = {**processed, **correlations_events}

        total_time = time.time() - start_time
        print(f"\nAnalysis completed in {total_time:.1f} seconds")

        return results
