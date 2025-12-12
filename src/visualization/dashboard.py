"""Dashboard module for visualizing the cortical circuit simulation."""


import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import ALL, MATCH, Input, Output, State

from src.analysis.bifurcation import NetworkModel, StabilityAnalyzer
from src.model.config import (
    CELL_ACTIVITY_COLORS,
    CELL_COLORS,
    CELL_TYPES,
    CONNECTIONS,
    INITIAL_STRENGTH_SCALING,
    LAYER_CONNECTIVITY_PARAMS,
    LAYER_NAMES,
    LAYERS,
    THALAMIC_ALPHA,
    UPDATE_INTERVAL,
)
from src.model.config import (
    LAYER_COLORS as MODEL_LAYER_COLORS,
)
from src.model.presets import P0_PRESET, P5_PRESET, P10_PRESET, P15_PRESET

# Import layout components from dedicated module
from src.visualization.dashboard_layout import (
    AXIS_FONT_SIZE,
    CELL_STYLE,
    CONTROL_PANEL_STYLE,
    GRAPH_CONFIG,
    HEADER_STYLE,
    LAYER_COLORS,
    SLIDER_CONTAINER_STYLE,
    SUBTITLE_FONT_SIZE,
    TITLE_FONT_SIZE,
    create_control_panel,
    create_grid_info_boxes,
    create_preset_buttons,
)

# Import plot helpers from dedicated module
from src.visualization.dashboard_plots import (
    HEATMAP_ZMAX,
    HEATMAP_ZMIN,
    create_empty_message_figure,
    create_heatmap_figure,
    create_initial_correlation_figure,
    create_initial_event_figure,
)

# Import utility helpers from dedicated module
from src.visualization.dashboard_utils import (
    SLIDER_HIDDEN_STYLE,
    SLIDER_POPUP_STYLE,
    format_analysis_display,
    get_triggered_id,
    get_triggered_value,
    is_valid_click,
    no_update_tuple,
    parse_connection_cell_id,
    parse_pattern_match_id,
)

# Correlation plot constants
CORRELATION_WINDOW_MS = 10000  # Rolling window: 10 seconds
CORRELATION_DISPLAY_SECONDS = 10  # Display window: 10 seconds
CORRELATION_UPDATE_INTERVAL = 20  # Update every 20 frames (~1s) to reduce computational load
CORRELATION_CELL_SAMPLE_RATE = 4  # Sample every 4th cell to reduce matrix size
CORRELATION_HISTORY_LENGTH = int(CORRELATION_WINDOW_MS / UPDATE_INTERVAL)

# Synchronous event tracking constants (reuse correlation timing)
SYNCHRONOUS_EVENT_THRESHOLD = 0.1  # From descriptive analysis
ACTIVITY_THRESHOLD = 0.1  # From descriptive analysis


class ConnectionKeyUtils:
    """Utility for parsing and building connection keys."""

    @staticmethod
    def parse(conn_key: str):
        """Parse connection key to (source_layer, source_cell, target_layer, target_cell).

        Args:
            conn_key: String like 'L23_E_to_L4_PV' or 'thalamus_to_L4_E'

        Returns:
            Tuple of (source_layer, source_cell, target_layer, target_cell)
        """
        parts = conn_key.split("_to_")
        source_parts = parts[0].split("_")
        target_parts = parts[1].split("_")

        if source_parts[0] == "thalamus":
            return "thalamus", None, target_parts[0], target_parts[1]
        return source_parts[0], source_parts[1], target_parts[0], target_parts[1]

    @staticmethod
    def build(source_layer, source_cell, target_layer, target_cell):
        """Build connection key from components.

        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', 'Th', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')

        Returns:
            Connection key string
        """
        if source_layer in ["Th", "thalamus"]:
            return f"thalamus_to_{target_layer}_{target_cell}"
        return f"{source_layer}_{source_cell}_to_{target_layer}_{target_cell}"


class DashboardApp:
    """
    Dashboard application for visualizing and controlling the neural simulation.

    This class creates an interactive Dash application that displays real-time
    neural activity and provides controls for adjusting simulation parameters.

    Layout components and styling are imported from dashboard_layout module.
    """

    def __init__(self, simulation, update_interval: int = UPDATE_INTERVAL):
        """
        Initialize the dashboard application.

        Args:
            simulation: CorticalSimulation instance
            update_interval: Update interval in milliseconds
        """
        self.simulation = simulation
        self.update_interval = update_interval

        # Flags to prevent concurrent expensive computations
        self._computing_stability = False
        self._computing_forced_response = False

        # Initialize the Dash app with light theme
        self.app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True
        )

        # Add custom CSS for sliders and light mode styling
        self.app.index_string = """
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        background-color: #f8f9fa !important;
                    }
                    .custom-slider .rc-slider-track {
                        background-color: #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-rail {
                        background-color: #ddd !important;
                    }
                    .custom-slider .rc-slider-handle {
                        border-color: #2c3e50 !important;
                        background-color: #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-handle:hover {
                        border-color: #34495e !important;
                    }
                    .custom-slider .rc-slider-handle:active {
                        border-color: #2c3e50 !important;
                        box-shadow: 0 0 5px #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-dot {
                        border-color: #ccc !important;
                        background-color: #ccc !important;
                    }
                    .custom-slider .rc-slider-dot-active {
                        border-color: #2c3e50 !important;
                        background-color: #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-mark {
                        width: 100% !important;
                    }
                    .custom-slider .rc-slider-mark-text {
                        color: #333 !important;
                    }
                    .custom-slider .rc-slider-mark-text:last-child {
                        transform: translateX(-100%) !important;
                    }

                    /* Force pointer cursor on clickable heatmaps */
                    .clickable-heatmap,
                    .clickable-heatmap * {
                        cursor: pointer !important;
                    }
                    .clickable-heatmap .plotly,
                    .clickable-heatmap .plotly *,
                    .clickable-heatmap .main-svg,
                    .clickable-heatmap .main-svg * {
                        cursor: pointer !important;
                    }
                    .clickable-heatmap .nsewdrag,
                    .clickable-heatmap .drag {
                        cursor: pointer !important;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        """

        # Pre-create all figures for better performance
        self.figures = {}
        self._initialize_figures()

        # Activity history for temporal averaging in spectrum computation
        self.activity_history = []  # List of recent mean_rates arrays
        self.history_length = 1  # How many snapshots to average over

        # Correlation tracking
        self.correlation_activity_buffer = []
        self.correlation_time_series = {
            "by_layer": {layer: [] for layer in LAYERS},
            "by_celltype": {cell_type: [] for cell_type in CELL_TYPES},
        }
        self.simulation_time = 0.0

        # Synchronous event tracking
        self.event_time_series = {
            "by_layer": {layer: [] for layer in LAYERS},
            "by_celltype": {cell_type: [] for cell_type in CELL_TYPES},
        }

        # Define common outputs for preset callbacks
        self._PRESET_OUTPUTS = [
            Output("tau-e-slider", "value"),
            Output("tau-sst-slider", "value"),
            Output("tau-pv-slider", "value"),
            Output("background-input-e-slider", "value"),
            Output("background-input-sst-slider", "value"),
            Output("background-input-pv-slider", "value"),
            Output("thalamic-width-e-slider", "value"),
            Output("thalamic-width-sst-slider", "value"),
            Output("thalamic-width-pv-slider", "value"),
            Output("outgoing-width-e-slider", "value"),
            Output("outgoing-width-sst-slider", "value"),
            Output("outgoing-width-pv-slider", "value"),
            Output("strength-scaling-e-slider", "value"),
            Output("strength-scaling-sst-slider", "value"),
            Output("strength-scaling-pv-slider", "value"),
            Output("strength-scaling-thalamus-slider", "value"),
            Output("alpha-slider", "value"),
            Output("connection-matrix-container", "children"),
        ]

        # Set up the layout and callbacks
        self.setup_layout()
        self.setup_callbacks()

    def build_current_preset(self) -> dict:
        """
        Build a preset dictionary from current simulation parameters.

        Returns:
            Dictionary compatible with NetworkModel initialization
        """
        # Get all current parameters from simulation
        connection_strengths = self.simulation.connectivity.get_all_connection_strengths()
        strength_scaling = self.simulation.connectivity.get_all_strength_scaling()
        time_constants = self.simulation.circuit.get_time_constants()
        gains = self.simulation.circuit.get_gains()
        background_input = self.simulation.circuit.get_all_background_inputs()
        all_sigmas = self.simulation.connectivity.get_all_sigmas()

        # Extract outgoing widths per cell type (use first occurrence for each source cell type)
        outgoing_widths = {}
        thalamic_widths = {}

        for conn_key, sigma in all_sigmas.items():
            if conn_key.startswith("thalamus_to_"):
                # Thalamic connection: extract target cell type
                parts = conn_key.split("_")
                target_cell = parts[3]  # thalamus_to_L4_E -> 'E'
                if target_cell not in thalamic_widths:
                    thalamic_widths[target_cell] = sigma
            else:
                # Regular connection: extract source cell type
                source_part = conn_key.split("_to_")[0]
                source_cell = source_part.split("_")[1]  # L4_E_to_L5_SST -> 'E'
                if source_cell not in outgoing_widths:
                    outgoing_widths[source_cell] = sigma

        # Build preset dictionary
        preset = {
            "connection_strengths": connection_strengths,
            "strength_scaling": strength_scaling,
            "time_constants": time_constants,
            "gains": gains,
            "background_input": background_input,
            "outgoing_widths": outgoing_widths,
            "thalamic_widths": thalamic_widths,
            "thalamic_alpha": THALAMIC_ALPHA,  # Not used in bifurcation analysis but included
        }

        return preset

    def _get_population_indices(self, selected_pops: list) -> np.ndarray:
        """
        Map population IDs like 'L23_E' to 0-8 indices.

        Args:
            selected_pops: List of population IDs (e.g., ['L23_E', 'L4_SST'])

        Returns:
            Array of indices corresponding to the selected populations
        """
        all_pops = [f"{layer}_{cell}" for layer in LAYERS for cell in CELL_TYPES]
        return np.array([all_pops.index(pop) for pop in selected_pops if pop in all_pops])

    def _format_population_title(self, selected_pops: list | None) -> str:
        """
        Format a title suffix showing selected populations.

        Args:
            selected_pops: List of population IDs or None for full network

        Returns:
            Title string (e.g., " Full Network" or ": L23_E + L4_SST")
        """
        if selected_pops is None or len(selected_pops) == 9:
            return " full network"
        elif len(selected_pops) == 0:
            return ""
        elif len(selected_pops) <= 4:
            return ": " + " + ".join(selected_pops)
        else:
            return f" {len(selected_pops)} populations"

    def _extract_mean_rates_from_simulation(self) -> np.ndarray | None:
        """
        Extract spatial mean rates from current simulation state with temporal averaging.

        Returns temporal average over recent history for smoother spectrum computation.

        Returns:
            Array of 9 mean rates [L23_E, L23_SST, L23_PV, L4_E, L4_SST, L4_PV, L5_E, L5_SST, L5_PV]
            or None if network is not yet active (all rates < threshold)
        """
        # Get current layer activities
        activities = self.simulation.circuit.get_layer_activities()

        # Extract spatial mean for each population
        mean_rates = []
        for layer in ["L23", "L4", "L5"]:
            for cell_type in ["E", "SST", "PV"]:
                spatial_mean = activities[layer][cell_type].mean()
                mean_rates.append(spatial_mean)

        mean_rates = np.array(mean_rates)

        # Check if network is active (instantaneous)
        if np.all(mean_rates < 1e-10):
            return None  # Network not yet active

        # Add to history buffer
        self.activity_history.append(mean_rates)
        if len(self.activity_history) > self.history_length:
            self.activity_history.pop(0)

        # Return temporal average if enough history, otherwise instantaneous
        if len(self.activity_history) >= 3:
            # Use temporal average for smoother spectrum
            averaged_rates = np.mean(self.activity_history, axis=0)
            # Check if averaged rates are still active
            if np.all(averaged_rates < 1e-10):
                return None
            return averaged_rates
        else:
            # Not enough history yet, use instantaneous
            return mean_rates

    def _update_time_series_data(self, results, time_series_dict):
        """Update time series data with new results and trim old data."""
        if results is None:
            return
        cutoff_time = self.simulation_time - CORRELATION_DISPLAY_SECONDS
        for group_key in ["by_layer", "by_celltype"]:
            for item, value in results[group_key].items():
                time_series_dict[group_key][item].append((self.simulation_time, value))
                time_series_dict[group_key][item] = [
                    (t, v) for t, v in time_series_dict[group_key][item] if t >= cutoff_time
                ]

    def _update_time_series_figure(self, fig, items, group_key, time_series_dict):
        """Update a time series figure with current data."""
        with fig.batch_update():
            for i, item in enumerate(items):
                time_series = time_series_dict[group_key][item]
                if time_series:
                    times, values = zip(*time_series, strict=False)
                    fig.data[i]["x"] = times
                    fig.data[i]["y"] = values

            x_max = max(self.simulation_time, CORRELATION_DISPLAY_SECONDS)
            x_min = max(0, self.simulation_time - CORRELATION_DISPLAY_SECONDS)
            fig.update_xaxes(range=[x_min, x_max])

    def _update_correlation_figures(self):
        """Update correlation figures with current time series data."""
        fig_layer = self.figures["correlation-by-layer"]
        fig_celltype = self.figures["correlation-by-celltype"]
        self._update_time_series_figure(fig_layer, LAYERS, "by_layer", self.correlation_time_series)
        self._update_time_series_figure(
            fig_celltype, CELL_TYPES, "by_celltype", self.correlation_time_series
        )
        return fig_layer, fig_celltype

    def _update_event_figures(self):
        """Update synchronous event figures with current time series data."""
        fig_layer = self.figures["events-by-layer"]
        fig_celltype = self.figures["events-by-celltype"]
        self._update_time_series_figure(fig_layer, LAYERS, "by_layer", self.event_time_series)
        self._update_time_series_figure(
            fig_celltype, CELL_TYPES, "by_celltype", self.event_time_series
        )
        return fig_layer, fig_celltype

    def _compute_group_correlation(self, corr_matrix, indices):
        """Helper to compute average correlation for a group of cells."""
        if len(indices) <= 1:
            return 0.0
        submatrix = corr_matrix[np.ix_(indices, indices)]
        mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
        mean_corr = np.mean(submatrix[mask])
        return 0.0 if not np.isfinite(mean_corr) else float(mean_corr)

    def _compute_rolling_correlations(self):
        """Compute pairwise correlations over the rolling activity window."""
        if len(self.correlation_activity_buffer) < CORRELATION_HISTORY_LENGTH:
            return None

        # Collect cell data from buffer - organize by group for efficient computation
        # Sample cells to reduce computational load
        layer_data = {layer: [] for layer in LAYERS}
        celltype_data = {cell_type: [] for cell_type in CELL_TYPES}

        for snapshot in self.correlation_activity_buffer[-CORRELATION_HISTORY_LENGTH:]:
            # Collect by layer (all cell types in each layer)
            for layer in LAYERS:
                cells_at_t = []
                for cell_type in CELL_TYPES:
                    cells = snapshot[layer][cell_type].flatten()
                    # Sample every Nth cell to reduce matrix size
                    cells_at_t.extend(cells[::CORRELATION_CELL_SAMPLE_RATE])
                layer_data[layer].append(cells_at_t)

            # Collect by celltype (all layers for each cell type)
            for cell_type in CELL_TYPES:
                cells_at_t = []
                for layer in LAYERS:
                    cells = snapshot[layer][cell_type].flatten()
                    # Sample every Nth cell to reduce matrix size
                    cells_at_t.extend(cells[::CORRELATION_CELL_SAMPLE_RATE])
                celltype_data[cell_type].append(cells_at_t)

        # Compute correlations within each group (much more efficient than all cells at once)
        def compute_group_corr(data_list):
            """Compute average pairwise correlation for a group."""
            if not data_list or len(data_list) < 2:
                return 0.0
            data_array = np.array(data_list)  # Shape: (num_timepoints, num_cells_in_group)
            # Return zeros if group inactive
            if np.max(data_array) < 1e-6:
                return 0.0
            # Skip if too few cells (can't compute meaningful correlation)
            if data_array.shape[1] < 2:
                return 0.0
            # Compute correlation matrix: (num_cells, num_timepoints)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(data_array.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            # Extract upper triangle (excluding diagonal) and compute mean
            if corr_matrix.shape[0] <= 1:
                return 0.0
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_corr = np.mean(corr_matrix[mask])
            return 0.0 if not np.isfinite(mean_corr) else float(mean_corr)

        return {
            "by_layer": {layer: compute_group_corr(layer_data[layer]) for layer in LAYERS},
            "by_celltype": {
                cell_type: compute_group_corr(celltype_data[cell_type]) for cell_type in CELL_TYPES
            },
        }

    def _compute_synchronous_events(self):
        """Compute synchronous event rates over the rolling activity window."""
        if len(self.correlation_activity_buffer) < CORRELATION_HISTORY_LENGTH:
            return None

        # Build labels once (same for all snapshots)
        layer_labels, celltype_labels = [], []
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                layer_labels.append(layer)
                celltype_labels.append(cell_type)

        # Collect cell data from buffer
        all_cells = []
        for snapshot in self.correlation_activity_buffer[-CORRELATION_HISTORY_LENGTH:]:
            cells_at_t = []
            for layer in LAYERS:
                for cell_type in CELL_TYPES:
                    cells_at_t.extend(snapshot[layer][cell_type].flatten())
            all_cells.append(cells_at_t)

        all_cells = np.array(all_cells)
        window_seconds = CORRELATION_WINDOW_MS / 1000.0

        def count_events(data):
            return (
                sum(
                    1
                    for t in range(len(data))
                    if np.mean(data[t] > ACTIVITY_THRESHOLD) > SYNCHRONOUS_EVENT_THRESHOLD
                )
                / window_seconds
            )

        return {
            "by_layer": {
                layer: count_events(
                    all_cells[:, np.array([i for i, l in enumerate(layer_labels) if l == layer])]
                )
                for layer in LAYERS
            },
            "by_celltype": {
                cell_type: count_events(
                    all_cells[
                        :, np.array([i for i, ct in enumerate(celltype_labels) if ct == cell_type])
                    ]
                )
                for cell_type in CELL_TYPES
            },
        }

    def compute_stability_spectrum(self, preset: dict, selected_pops: list | None = None) -> tuple:
        """
        Compute stability spectrum (max Re(λ) vs k) for current network state.

        Linearizes around the current spatial-mean activity from the running simulation.
        Optimized: scans only positive quadrant and caches exponentials.

        Args:
            preset: Preset dictionary with current network parameters
            selected_pops: Optional list of population IDs to analyze (e.g., ['L23_E', 'L4_SST']).
                          If None, analyzes full network. If provided, analyzes subset in isolation.

        Returns:
            Tuple of (k_values, max_real_eigenvalues, eigenvalues_at_max_k, k_max)
            - k_values: Array of k values
            - max_real_eigenvalues: Array of max real parts for each k
            - eigenvalues_at_max_k: Complex array of all eigenvalues at k with max instability
            - k_max: The k value where maximum instability occurs
        """
        try:
            # Extract current spatial mean rates from simulation
            steady_state = self._extract_mean_rates_from_simulation()

            # Check if network is active
            if steady_state is None:
                # Network not yet active - return empty arrays
                return np.array([]), np.array([]), np.array([]), 0.0

            # Create network model for full network (all 3 layers)
            network = NetworkModel(preset, layers=["L23", "L4", "L5"])

            # Create stability analyzer with current simulation state
            analyzer = StabilityAnalyzer(network, steady_state)

            # Scan k modes and collect max Re(λ) for each k
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            n_modes = ANALYSIS_PARAMS["n_modes"]
            grid_size = ANALYSIS_PARAMS["grid_size"]
            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]  # μm
            n_modes_effective = min(n_modes, int(0.6 * grid_size))

            total_pops = len(network.tau)

            # Pre-compute all unique k² values and cache exponentials
            # Only scan positive quadrant: n1 >= 0, n2 >= 0
            k_squared_set = set()
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared_set.add(n1**2 + n2**2)

            # Build cache: exp_cache[k²][i,j] = exp(-2π²k²σ²_ij)
            # sigma values are in μm, normalize by anatomical_grid_size (also in μm)
            exp_cache = {}
            for k_squared in k_squared_set:
                exp_cache[k_squared] = np.zeros((total_pops, total_pops))
                for i in range(total_pops):
                    for j in range(total_pops):
                        sigma_ij = network.sigma[i, j] / anatomical_grid_size
                        exp_cache[k_squared][i, j] = np.exp(-2 * np.pi**2 * k_squared * sigma_ij**2)

            # Dictionary to store results by k^2
            results_by_k2 = {}

            # Scan only positive quadrant (reduces work by ~4×)
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)

                    # Skip if k > n_modes (keep k in [0, n_modes])
                    if k > n_modes:
                        continue

                    # Build Jacobian using cached exponentials
                    J = np.zeros((total_pops, total_pops))
                    exp_factors = exp_cache[k_squared]

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

                    # Extract subset if selected_pops is provided
                    if selected_pops is not None and len(selected_pops) > 0:
                        indices = self._get_population_indices(selected_pops)
                        if len(indices) == 0:
                            continue  # Skip if no valid populations
                        J_subset = J[np.ix_(indices, indices)]
                        eigenvalues = np.linalg.eigvals(J_subset)
                    else:
                        # Full network analysis
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
                        # Update if this has a larger max real part
                        if max_real > results_by_k2[k_squared]["max_real"]:
                            results_by_k2[k_squared]["max_real"] = max_real
                            results_by_k2[k_squared]["eigenvalues"] = eigenvalues

            # Sort by k and extract arrays
            sorted_results = sorted(results_by_k2.values(), key=lambda x: x["k"])
            k_values = np.array([r["k"] for r in sorted_results])
            max_real_values = np.array([r["max_real"] for r in sorted_results])

            # Find k with maximum instability (most positive real part)
            if len(max_real_values) > 0:
                max_idx = np.argmax(max_real_values)
                k_max = k_values[max_idx]
                eigenvalues_at_max_k = sorted_results[max_idx]["eigenvalues"]
            else:
                k_max = 0.0
                eigenvalues_at_max_k = np.array([])

            return k_values, max_real_values, eigenvalues_at_max_k, k_max

        except Exception as e:
            print(f"Error computing stability spectrum: {e}")
            return np.array([]), np.array([]), np.array([]), 0.0

    def compute_B_fourier(
        self, network: "NetworkModel", k_squared: float, anatomical_grid_size: float
    ) -> np.ndarray:
        """
        Compute thalamic input B(k) in Fourier space with Gaussian spatial filtering.

        Args:
            network: NetworkModel instance containing thalamic parameters
            k_squared: Square of the wavenumber k (mode number)
            anatomical_grid_size: Anatomical grid size in μm

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
            B_k[i] = network.thalamic_strengths[i] * np.exp(
                -2 * np.pi**2 * k_squared * sigma_thal_i**2
            )

        return B_k

    def compute_static_gain(self, preset: dict, selected_pops: list | None = None) -> tuple:
        """
        Compute static spatial gain curve G(k) = ||−J(k)^(−1) B(k)||.

        Shows how strongly each spatial frequency is amplified by the cortical circuit.

        Args:
            preset: Network preset dictionary
            selected_pops: Optional list of population IDs to analyze (e.g., ['L23_E', 'L4_SST']).
                          If None, analyzes full network. If provided, analyzes subset in isolation.

        Returns:
            (k_values, gain_values): Arrays of k and corresponding gains
        """
        try:
            # Get steady state from running simulation
            steady_state = self._extract_mean_rates_from_simulation()
            if steady_state is None:
                return np.array([]), np.array([])

            # Build network model and analyzer
            network = NetworkModel(preset, layers=["L23", "L4", "L5"])
            analyzer = StabilityAnalyzer(network, steady_state)

            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            n_modes = ANALYSIS_PARAMS["n_modes"]
            grid_size = ANALYSIS_PARAMS["grid_size"]
            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]  # μm
            n_modes_effective = min(n_modes, int(0.6 * grid_size))
            total_pops = len(network.tau)

            # Cache exponentials for cortical connections (reuse from stability spectrum)
            k_squared_set = set()
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared_set.add(n1**2 + n2**2)

            # Cache for cortical connection exponentials
            # sigma values are in μm, normalize by anatomical_grid_size (also in μm)
            exp_cache = {}
            for k_squared in k_squared_set:
                exp_cache[k_squared] = np.zeros((total_pops, total_pops))
                for i in range(total_pops):
                    for j in range(total_pops):
                        sigma_ij = network.sigma[i, j] / anatomical_grid_size
                        exp_cache[k_squared][i, j] = np.exp(-2 * np.pi**2 * k_squared * sigma_ij**2)

            # Aggregate results by k² (handle degenerate modes)
            results_by_k2 = {}

            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)
                    if k > n_modes:
                        continue

                    # Build Jacobian J(k)
                    J = np.zeros((total_pops, total_pops))
                    exp_factors = exp_cache[k_squared]

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

                    # Compute B(k) with thalamic spatial filtering
                    B_k = self.compute_B_fourier(network, k_squared, anatomical_grid_size)

                    # Extract subset if selected_pops is provided
                    if selected_pops is not None and len(selected_pops) > 0:
                        indices = self._get_population_indices(selected_pops)
                        if len(indices) == 0:
                            continue  # Skip if no valid populations
                        J_subset = J[np.ix_(indices, indices)]
                        B_subset = B_k[indices]
                        J_to_use = J_subset
                        B_to_use = B_subset
                    else:
                        J_to_use = J
                        B_to_use = B_k

                    # Check if B(k) is non-zero
                    if np.linalg.norm(B_to_use) < 1e-10:
                        continue

                    # Compute gain: G(k) = ||−J(k)^(−1) B(k)||
                    try:
                        # Compute -J(k)^(-1) @ B(k)
                        J_inv_B = np.linalg.solve(-J_to_use, B_to_use)
                        # Spectral norm (largest singular value)
                        gain = np.linalg.norm(J_inv_B)

                        # Store maximum gain across degenerate modes
                        if k_squared not in results_by_k2:
                            results_by_k2[k_squared] = {"k": k, "gain": gain}
                        else:
                            results_by_k2[k_squared]["gain"] = max(
                                results_by_k2[k_squared]["gain"], gain
                            )
                    except np.linalg.LinAlgError:
                        # Singular matrix - skip this mode
                        continue

            # Sort by k and extract results
            sorted_results = sorted(results_by_k2.values(), key=lambda x: x["k"])
            k_values = np.array([r["k"] for r in sorted_results])
            gain_values = np.array([r["gain"] for r in sorted_results])

            return k_values, gain_values

        except Exception as e:
            print(f"Error computing static gain: {e}")
            return np.array([]), np.array([])

    def compute_spatiotemporal_gain(self, preset: dict, selected_pops: list | None = None) -> tuple:
        """
        Compute spatiotemporal amplification map A(k,ω) = ||(iωI − J(k))^(−1) B(k)||.

        Shows which spatial (k) and temporal (ω) frequencies the circuit amplifies most.

        Args:
            preset: Network preset dictionary
            selected_pops: Optional list of population IDs to analyze (e.g., ['L23_E', 'L4_SST']).
                          If None, analyzes full network. If provided, analyzes subset in isolation.

        Returns:
            (k_values, omega_values, gain_matrix): Arrays of k, ω, and gain[k,ω]
        """
        try:
            # Get steady state from running simulation
            steady_state = self._extract_mean_rates_from_simulation()
            if steady_state is None:
                return np.array([]), np.array([]), np.array([])

            # Build network model and analyzer
            network = NetworkModel(preset, layers=["L23", "L4", "L5"])
            analyzer = StabilityAnalyzer(network, steady_state)

            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            n_modes = ANALYSIS_PARAMS["n_modes"]
            grid_size = ANALYSIS_PARAMS["grid_size"]
            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]  # μm
            n_modes_effective = min(n_modes, int(0.6 * grid_size))
            total_pops = len(network.tau)

            # Define temporal frequency range (0-1 Hz)
            omega_values = np.linspace(0, 1, 21)  # 21 samples for smooth heatmap up to 1 Hz

            # Cache exponentials for cortical connections
            # sigma values are in μm, normalize by anatomical_grid_size (also in μm)
            k_squared_set = set()
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared_set.add(n1**2 + n2**2)

            exp_cache = {}
            for k_squared in k_squared_set:
                exp_cache[k_squared] = np.zeros((total_pops, total_pops))
                for i in range(total_pops):
                    for j in range(total_pops):
                        sigma_ij = network.sigma[i, j] / anatomical_grid_size
                        exp_cache[k_squared][i, j] = np.exp(-2 * np.pi**2 * k_squared * sigma_ij**2)

            # Aggregate k values (just scan positive quadrant for efficiency)
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

            # Compute gain for each (k, ω)
            for k_idx, k_squared in enumerate(sorted_k_squared):
                # Build Jacobian J(k)
                J = np.zeros((total_pops, total_pops))
                exp_factors = exp_cache[k_squared]

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

                # Compute B(k) with thalamic spatial filtering
                B_k = self.compute_B_fourier(network, k_squared, anatomical_grid_size)

                # Extract subset if selected_pops is provided
                if selected_pops is not None and len(selected_pops) > 0:
                    indices = self._get_population_indices(selected_pops)
                    if len(indices) == 0:
                        continue  # Skip if no valid populations
                    J_subset = J[np.ix_(indices, indices)]
                    B_subset = B_k[indices]
                    J_to_use = J_subset
                    B_to_use = B_subset
                    n_pops_subset = len(indices)
                else:
                    J_to_use = J
                    B_to_use = B_k
                    n_pops_subset = total_pops

                # Check if B(k) is non-zero
                if np.linalg.norm(B_to_use) < 1e-10:
                    continue

                # For each temporal frequency ω
                for omega_idx, omega in enumerate(omega_values):
                    # Convert Hz to rad/s
                    omega_rad = 2 * np.pi * omega

                    # Compute (iωI - J(k))
                    M = 1j * omega_rad * np.eye(n_pops_subset) - J_to_use

                    # Compute A(k,ω) = ||(iωI − J(k))^(−1) B(k)||
                    try:
                        M_inv_B = np.linalg.solve(M, B_to_use)
                        # Spectral norm (use norm for complex vectors)
                        gain = np.linalg.norm(M_inv_B)
                        gain_matrix[k_idx, omega_idx] = gain
                    except np.linalg.LinAlgError:
                        # Singular matrix - set to 0
                        gain_matrix[k_idx, omega_idx] = 0.0

            return k_values, omega_values, gain_matrix

        except Exception as e:
            print(f"Error computing spatiotemporal gain: {e}")
            return np.array([]), np.array([]), np.array([])

    def create_stability_spectrum_figure(
        self, k_values: np.ndarray, max_real_values: np.ndarray
    ) -> go.Figure:
        """
        Create Plotly figure for stability spectrum (max Re(λ) vs wavelength).

        Args:
            k_values: Array of k values (mode numbers, dimensionless)
            max_real_values: Array of max real eigenvalues for each k
            selected_pops: Optional list of selected population IDs for title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        highlight_data = None

        # Check if we have data
        if len(k_values) > 0 and len(max_real_values) > 0:
            # Convert k (mode number) to wavelength: λ = anatomical_grid_size / k (μm)
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]

            # Filter out k=0 first to avoid division by zero
            nonzero_mask = k_values > 0
            k_values_nonzero = k_values[nonzero_mask]
            max_real_values_nonzero = max_real_values[nonzero_mask]

            if len(k_values_nonzero) > 0:
                # Now safe to divide
                wavelength_values_finite = anatomical_grid_size / k_values_nonzero
                max_real_values_finite = max_real_values_nonzero

                # Add main spectrum line
                fig.add_trace(
                    go.Scatter(
                        x=wavelength_values_finite,
                        y=max_real_values_finite,
                        mode="lines",
                        name="max Re(λ)",
                        line=dict(color="#2c3e50", width=2),
                        hovertemplate="L=%{x:.0f} μm<br>max Re(λ)=%{y:.3f}<extra></extra>",
                        showlegend=False,
                    )
                )

                # Add horizontal line at y=0 (stability boundary)
                fig.add_trace(
                    go.Scatter(
                        x=[wavelength_values_finite.min(), wavelength_values_finite.max()],
                        y=[0, 0],
                        mode="lines",
                        name="Stability boundary",
                        line=dict(color="gray", width=2, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # Determine y-axis range
                y_min = min(max_real_values_finite.min(), -0.5)
                y_max = max(max_real_values_finite.max(), 0.5)
                y_range = y_max - y_min
                y_padding = y_range * 0.1

                # Determine dominant mode (strict maximum)
                max_idx = int(np.argmax(max_real_values_finite))
                max_value = max_real_values_finite[max_idx]
                if np.sum(np.isclose(max_real_values_finite, max_value)) == 1:
                    highlight_wavelength = wavelength_values_finite[max_idx]
                    highlight_color = "#e74c3c" if max_value > 0 else "#7f8c8d"
                    highlight_data = (highlight_wavelength, max_value, highlight_color)

                # Determine wavelength range dynamically from data
                wavelength_min = wavelength_values_finite.min() * 0.9  # Add 10% padding
                wavelength_max = wavelength_values_finite.max() * 1.1
            else:
                y_min, y_max, y_padding = -0.5, 0.5, 0
                n_modes = ANALYSIS_PARAMS["n_modes"]
                wavelength_min, wavelength_max = (
                    anatomical_grid_size / n_modes,
                    anatomical_grid_size,
                )
                fig.add_annotation(
                    text="Network not yet active (run simulation to see spectrum)",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
                )
        else:
            # No data - show empty plot with message
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
            n_modes = ANALYSIS_PARAMS["n_modes"]
            y_min, y_max, y_padding = -0.5, 0.5, 0
            wavelength_min, wavelength_max = anatomical_grid_size / n_modes, anatomical_grid_size
            fig.add_annotation(
                text="Network not yet active (run simulation to see spectrum)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
            )

        # Add highlight marker if applicable
        if highlight_data is not None:
            highlight_wavelength, highlight_val, highlight_color = highlight_data
            fig.add_trace(
                go.Scatter(
                    x=[highlight_wavelength],
                    y=[highlight_val],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color=highlight_color,
                        symbol="star",
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="Dominant L=%{x:.0f} μm<br>max Re(λ)=%{y:.3f}<extra></extra>",
                    showlegend=False,
                    cliponaxis=False,
                )
            )

        # Generate title
        title_text = "Stability Spectrum"

        fig.update_layout(
            title=dict(
                text=title_text, x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)
            ),
            xaxis=dict(
                title=dict(text="Wavelength (μm)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=True,
                gridcolor="#e0e0e0",
                zeroline=False,
                range=[wavelength_min, wavelength_max],
            ),
            yaxis=dict(
                title=dict(text="max Re(λ)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=True,
                gridcolor="#e0e0e0",
                zeroline=True,
                zerolinecolor="gray",
                zerolinewidth=1,
                range=[y_min - y_padding, y_max + y_padding] if len(k_values) > 0 else [-0.5, 0.5],
            ),
            margin=dict(l=50, r=25, t=35, b=40),
            height=280,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
            showlegend=False,
        )

        return fig

    def create_eigenvalue_spectrum_figure(self, eigenvalues: np.ndarray, k_max: float) -> go.Figure:
        """
        Create Plotly figure for eigenvalue spectrum in the complex plane.

        Args:
            eigenvalues: Complex array of eigenvalues to plot
            k_max: The k value (mode number, dimensionless) at which these eigenvalues were computed
            selected_pops: Optional list of selected population IDs for title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Check if we have data
        if len(eigenvalues) > 0:
            # Extract real and imaginary parts
            real_parts = eigenvalues.real
            imag_parts = eigenvalues.imag

            # Add eigenvalue scatter plot
            fig.add_trace(
                go.Scatter(
                    x=real_parts,
                    y=imag_parts,
                    mode="markers",
                    name="Eigenvalues",
                    marker=dict(
                        size=8,
                        color=real_parts,  # Color by real part
                        colorscale="balance",  # Balance colormap
                        cmin=-0.4,
                        cmax=0.4,
                        opacity=1.0,
                        line=dict(color="black", width=0.5),
                        showscale=False,
                    ),
                    hovertemplate="Re(λ)=%{x:.3f}<br>Im(λ)=%{y:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

            # Add vertical line at Re(λ) = 0 (stability boundary)
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[-0.4, 0.4],
                    mode="lines",
                    name="Stability boundary",
                    line=dict(color="gray", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        else:
            # No data - show empty plot with message
            fig.add_annotation(
                text="Network not yet active",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
            )

        # Convert k (mode number) to wavelength and add annotation in top right corner (always show)
        from src.analysis.bifurcation.config import ANALYSIS_PARAMS

        anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]

        wavelength_max = (anatomical_grid_size / k_max) if k_max > 0 else np.inf
        if np.isfinite(wavelength_max):
            wavelength_text = f"L = {wavelength_max:.0f} μm"
        else:
            wavelength_text = "L = ∞"

        fig.add_annotation(
            text=wavelength_text,
            xref="paper",
            yref="paper",
            x=0.999,
            y=0.999,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=AXIS_FONT_SIZE, color="black"),
            bgcolor="rgba(220, 220, 220, 0.9)",  # Pale grey background
            bordercolor="rgba(180, 180, 180, 0.8)",
            borderwidth=1,
            borderpad=3,
        )

        # Generate title
        title_text = "Eigenvalue Spectrum"

        fig.update_layout(
            title=dict(
                text=title_text, x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)
            ),
            xaxis=dict(
                title=dict(text="Re(λ)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=True,
                gridcolor="#e0e0e0",
                zeroline=True,
                zerolinecolor="gray",
                zerolinewidth=2,
                range=[-0.4, 0.4],
                tickvals=[-0.4, 0, 0.4],
            ),
            yaxis=dict(
                title=dict(text="Im(λ)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=True,
                gridcolor="#e0e0e0",
                zeroline=True,
                zerolinecolor="#e0e0e0",
                zerolinewidth=1,
                range=[-0.4, 0.4],
                tickvals=[-0.4, 0, 0.4],
            ),
            margin=dict(l=50, r=25, t=35, b=40),
            height=280,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
            showlegend=False,
        )

        return fig

    def create_static_gain_figure(self, k_values: np.ndarray, gain_values: np.ndarray) -> go.Figure:
        """
        Create Plotly figure for static spatial gain G(λ).

        Args:
            k_values: Array of k values (mode numbers, dimensionless)
            gain_values: Array of gain values for each k
            selected_pops: Optional list of selected population IDs for title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        highlight_data = None

        # Check if we have data
        if len(k_values) > 0 and len(gain_values) > 0:
            # Convert k (mode number) to wavelength: λ = anatomical_grid_size / k (μm)
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]

            # Filter out k=0 first to avoid division by zero
            nonzero_mask = k_values > 0
            k_values_nonzero = k_values[nonzero_mask]
            gain_values_nonzero = gain_values[nonzero_mask]

            if len(k_values_nonzero) > 0:
                # Now safe to divide
                wavelength_values_finite = anatomical_grid_size / k_values_nonzero
                gain_values_finite = gain_values_nonzero

                # Add main gain curve
                fig.add_trace(
                    go.Scatter(
                        x=wavelength_values_finite,
                        y=gain_values_finite,
                        mode="lines",
                        name="G(L)",
                        line=dict(color="#2c3e50", width=2),
                        hovertemplate="L=%{x:.0f} μm<br>Gain=%{y:.2f}<extra></extra>",
                        showlegend=False,
                    )
                )

                # Determine y-axis range
                y_min = 0
                y_max = max(gain_values_finite.max() * 1.1, 1.0)

                # Determine dominant gain mode (unique maximum)
                max_idx = int(np.argmax(gain_values_finite))
                max_value = gain_values_finite[max_idx]
                if np.sum(np.isclose(gain_values_finite, max_value)) == 1:
                    highlight_data = (wavelength_values_finite[max_idx], max_value)

                # Determine wavelength range dynamically from data
                wavelength_min = wavelength_values_finite.min() * 0.9  # Add 10% padding
                wavelength_max = wavelength_values_finite.max() * 1.1
            else:
                y_min, y_max = 0, 10
                n_modes = ANALYSIS_PARAMS["n_modes"]
                wavelength_min, wavelength_max = (
                    anatomical_grid_size / n_modes,
                    anatomical_grid_size,
                )
                fig.add_annotation(
                    text="Network not yet active (run simulation to see gain)",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
                )
        else:
            # No data - show empty plot with message
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
            n_modes = ANALYSIS_PARAMS["n_modes"]
            y_min, y_max = 0, 10
            wavelength_min, wavelength_max = anatomical_grid_size / n_modes, anatomical_grid_size
            fig.add_annotation(
                text="Network not yet active (run simulation to see gain)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
            )

        # Add highlight marker if applicable
        if highlight_data is not None:
            highlight_wavelength, highlight_val = highlight_data
            fig.add_trace(
                go.Scatter(
                    x=[highlight_wavelength],
                    y=[highlight_val],
                    mode="markers",
                    marker=dict(
                        size=11, color="#7f8c8d", symbol="star", line=dict(color="#ffffff", width=1)
                    ),
                    hovertemplate="Dominant L=%{x:.0f} μm<br>Gain=%{y:.2f}<extra></extra>",
                    showlegend=False,
                    cliponaxis=False,
                )
            )

        # Generate title
        title_text = "Static Gain"

        fig.update_layout(
            title=dict(
                text=title_text, x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)
            ),
            xaxis=dict(
                title=dict(text="Wavelength (μm)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=True,
                gridcolor="#e0e0e0",
                zeroline=False,
                range=[wavelength_min, wavelength_max],
            ),
            yaxis=dict(
                title=dict(text="Gain G(L)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=True,
                gridcolor="#e0e0e0",
                zeroline=False,
                range=[y_min, y_max],
            ),
            margin=dict(l=50, r=25, t=35, b=40),
            height=280,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
            showlegend=False,
        )

        return fig

    def create_spatiotemporal_gain_figure(
        self, k_values: np.ndarray, omega_values: np.ndarray, gain_matrix: np.ndarray
    ) -> go.Figure:
        """
        Create Plotly figure for spatiotemporal amplification map A(λ,ω).

        Args:
            k_values: Array of spatial frequencies k (mode numbers, dimensionless)
            omega_values: Array of temporal frequencies ω (Hz)
            gain_matrix: 2D array of gain values [k_idx, omega_idx]
            selected_pops: Optional list of selected population IDs for title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Check if we have data
        if len(k_values) > 0 and len(omega_values) > 0 and gain_matrix.size > 0:
            # Convert k (mode number) to wavelength: λ = anatomical_grid_size / k (μm)
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]

            # Filter out k=0 first to avoid division by zero
            nonzero_mask = k_values > 0
            k_values_nonzero = k_values[nonzero_mask]
            gain_matrix_nonzero = gain_matrix[nonzero_mask, :]

            if len(k_values_nonzero) > 0:
                # Now safe to divide
                wavelength_values_finite = anatomical_grid_size / k_values_nonzero
                gain_matrix_finite = gain_matrix_nonzero
                # Flip gain matrix left-right since wavelength is inverse of k
                gain_matrix_flipped = np.flipud(gain_matrix_finite)
                wavelength_min = wavelength_values_finite.min()
                wavelength_max = wavelength_values_finite.max()

                # Create heatmap
                fig.add_trace(
                    go.Heatmap(
                        x=wavelength_values_finite[
                            ::-1
                        ],  # Reverse so larger wavelengths are on left
                        y=omega_values,
                        z=gain_matrix_flipped.T,  # Transpose so wavelength is on x-axis
                        colorscale="Hot",
                        colorbar=dict(
                            title=dict(
                                text="Amplification", side="right", font=dict(size=AXIS_FONT_SIZE)
                            ),
                            tickfont=dict(size=AXIS_FONT_SIZE),
                            len=1.0,
                            thickness=12,
                        ),
                        hovertemplate="L=%{x:.0f} μm<br>ω=%{y:.2f} Hz<br>Gain=%{z:.2f}<extra></extra>",
                    )
                )

                # Determine wavelength range dynamically from data
                wavelength_range = [wavelength_min * 0.9, wavelength_max * 1.1]
            else:
                from src.analysis.bifurcation.config import ANALYSIS_PARAMS

                anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
                n_modes = ANALYSIS_PARAMS["n_modes"]
                wavelength_range = [anatomical_grid_size / n_modes, anatomical_grid_size]
                fig.add_annotation(
                    text="Network not yet active (run simulation to see amplification)",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
                )
        else:
            # No data - show empty plot with message
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS

            anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
            n_modes = ANALYSIS_PARAMS["n_modes"]
            wavelength_range = [anatomical_grid_size / n_modes, anatomical_grid_size]
            fig.add_annotation(
                text="Network not yet active (run simulation to see amplification)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
            )

        # Generate title
        title_text = "Spatiotemporal Gain"

        fig.update_layout(
            title=dict(
                text=title_text, x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)
            ),
            xaxis=dict(
                title=dict(text="Wavelength (μm)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=False,
                range=wavelength_range,
            ),
            yaxis=dict(
                title=dict(text="Temporal freq ω (Hz)", font=dict(size=AXIS_FONT_SIZE)),
                tickfont=dict(size=AXIS_FONT_SIZE),
                showgrid=False,
                range=[0, 1],
            ),
            margin=dict(l=50, r=25, t=35, b=40),
            height=280,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig

    def _initialize_correlation_figures(self):
        """Initialize correlation line plot figures."""
        self.figures["correlation-by-layer"] = create_initial_correlation_figure(
            LAYERS, MODEL_LAYER_COLORS, CORRELATION_DISPLAY_SECONDS
        )
        self.figures["correlation-by-celltype"] = create_initial_correlation_figure(
            CELL_TYPES, CELL_COLORS, CORRELATION_DISPLAY_SECONDS
        )

    def _initialize_event_figures(self):
        """Initialize synchronous event line plot figures."""
        self.figures["events-by-layer"] = create_initial_event_figure(
            LAYERS, MODEL_LAYER_COLORS, CORRELATION_DISPLAY_SECONDS
        )
        self.figures["events-by-celltype"] = create_initial_event_figure(
            CELL_TYPES, CELL_COLORS, CORRELATION_DISPLAY_SECONDS
        )

    def _initialize_figures(self):
        """Pre-create all heatmap figures for better performance."""
        # Initialize with zeros
        empty_data = np.zeros((self.simulation.grid_size, self.simulation.grid_size))

        # Create figures for all cell types in all layers
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                fig_id = f"graph-{layer}-{cell_type}"
                self.figures[fig_id] = self.create_heatmap(empty_data, cell_type)

        # Create thalamus figure
        self.figures["graph-thalamus"] = self.create_heatmap(empty_data, "thalamus")

        # Create correlation figures
        self._initialize_correlation_figures()

        # Create event figures
        self._initialize_event_figures()

    def _create_grid_info_boxes(self):
        """Create info boxes showing grid parameters."""
        return create_grid_info_boxes()

    def _create_preset_buttons(self):
        """Create the preset buttons row."""
        # Note: Uses 'mb-3' instead of 'mb-4' in the extracted function
        # This preserves the original spacing behavior
        result = create_preset_buttons()
        result.className = "mb-4"
        return result

    def _create_thalamus_visualization(self):
        """Create the thalamus visualization row."""
        return html.Div(
            [
                dbc.Row(
                    [
                        # Thalamus label
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Th",
                                            style={
                                                "margin": "0",
                                                "whiteSpace": "nowrap",  # Prevent text wrapping
                                                "fontSize": f"{SUBTITLE_FONT_SIZE}px",
                                                "fontWeight": "600",
                                            },
                                        )
                                    ],
                                    style={
                                        "display": "flex",
                                        "justifyContent": "flex-end",  # Align to the right
                                        "paddingRight": "60px",  # Match the spacing of other layer labels
                                        "height": "100%",
                                        "alignItems": "center",
                                    },
                                )
                            ],
                            width=2,
                        ),
                        # Thalamus heatmap
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="graph-thalamus",
                                                    figure=self.figures["graph-thalamus"],
                                                    config=GRAPH_CONFIG,
                                                )
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "border": "3px solid #7f8c8d",
                                            },
                                        )
                                    ],
                                    style={"display": "flex", "justifyContent": "center"},
                                )
                            ],
                            width=10,
                        ),
                    ],
                    className="align-items-center",
                )
            ],
            className="mt-2",
        )

    def _create_activity_visualization(self):
        """Create the activity visualization section."""
        return dbc.Col(
            [
                # Add more top padding to shift visualization down
                html.Div(style={"height": "10px"}),
                # Grid info boxes
                self._create_grid_info_boxes(),
                # Preset Buttons
                self._create_preset_buttons(),
                # Layer visualizations
                *[self.create_layer_row(layer) for layer in LAYERS],
                # Thalamus visualization
                self._create_thalamus_visualization(),
                # Selected populations display
                html.Div(
                    [
                        dbc.Row(
                            [
                                # Empty column to match heatmap label width
                                dbc.Col(width=2),
                                # Populations display container
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="selected-populations-display",
                                            children="Click heatmaps to select populations for analysis",
                                            style={
                                                "textAlign": "center",
                                                "padding": "8px 14px",
                                                "backgroundColor": "#e0e0e0",  # Light grey
                                                "borderColor": "#e0e0e0",
                                                "borderRadius": "4px",
                                                "fontSize": f"{SUBTITLE_FONT_SIZE}px",
                                                "color": "black",  # Black text
                                                "fontWeight": "500",
                                            },
                                        )
                                    ],
                                    width=10,
                                ),
                            ]
                        )
                    ],
                    className="mt-2",
                ),
                # Correlation plots
                html.Div(
                    [
                        # Title
                        dbc.Row(
                            [
                                dbc.Col(width=2),  # Empty column for alignment
                                dbc.Col(
                                    [
                                        html.H5(
                                            "Pairwise Correlation",
                                            className="mb-2 text-center",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                                "fontSize": f"{TITLE_FONT_SIZE}px",
                                                "fontWeight": "600",
                                            },
                                        )
                                    ],
                                    width=10,
                                ),
                            ]
                        ),
                        # Plots row
                        dbc.Row(
                            [
                                dbc.Col(width=2),  # Empty column for alignment
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="correlation-by-layer",
                                                            figure=self.figures[
                                                                "correlation-by-layer"
                                                            ],
                                                            config=GRAPH_CONFIG,
                                                            style={"height": "100%"},
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="correlation-by-celltype",
                                                            figure=self.figures[
                                                                "correlation-by-celltype"
                                                            ],
                                                            config=GRAPH_CONFIG,
                                                            style={"height": "100%"},
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        )
                                    ],
                                    width=10,
                                ),
                            ]
                        ),
                    ],
                    className="mt-1",
                ),
                # Synchronous event plots
                html.Div(
                    [
                        # Title
                        dbc.Row(
                            [
                                dbc.Col(width=2),  # Empty column for alignment
                                dbc.Col(
                                    [
                                        html.H5(
                                            "Large Synchronous Events",
                                            className="mb-2 text-center",
                                            style={
                                                "textAlign": "center",
                                                "marginTop": "20px",
                                                "fontSize": f"{TITLE_FONT_SIZE}px",
                                                "fontWeight": "600",
                                            },
                                        )
                                    ],
                                    width=10,
                                ),
                            ]
                        ),
                        # Plots row
                        dbc.Row(
                            [
                                dbc.Col(width=2),  # Empty column for alignment
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="events-by-layer",
                                                            figure=self.figures["events-by-layer"],
                                                            config=GRAPH_CONFIG,
                                                            style={"height": "100%"},
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Graph(
                                                            id="events-by-celltype",
                                                            figure=self.figures[
                                                                "events-by-celltype"
                                                            ],
                                                            config=GRAPH_CONFIG,
                                                            style={"height": "100%"},
                                                        )
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        )
                                    ],
                                    width=10,
                                ),
                            ]
                        ),
                    ],
                    className="mt-1",
                ),
            ],
            width=4,
            className="px-4",
        )

    def _create_connectivity_matrix(self):
        """Create the connectivity matrix section."""
        return dbc.Col(
            [
                # Connection Strength Matrix
                html.Div(
                    [
                        html.H5(
                            "Connection Strengths",
                            className="mb-3 text-center",
                            style={
                                "textAlign": "center",
                                "width": "85%",  # Match matrix container width
                                "margin": "0 auto",  # Center the title
                                "paddingLeft": "50px",  # Match matrix container padding
                                "fontSize": f"{TITLE_FONT_SIZE}px",
                                "fontWeight": "600",
                            },
                        ),
                        # Connection Matrix Container
                        html.Div(
                            self.create_connection_matrix(),
                            id="connection-matrix-container",
                            style={
                                "position": "relative",
                                "display": "flex",
                                "justifyContent": "center",
                                "width": "85%",  # Further reduce width to prevent overlap
                                "margin": "0 auto",  # Center the container
                                "paddingLeft": "50px",  # Add left padding to shift matrix right
                            },
                        ),
                        # Hover Activated Slider Container (initially hidden)
                        html.Div(
                            id="slider-container",
                            style={"display": "none", **SLIDER_CONTAINER_STYLE},
                        ),
                    ],
                    className="mb-3",
                ),
                # Stability Analysis (Spectrum and Eigenvalues)
                html.Div(
                    [
                        html.H5(
                            "Stability Analysis",
                            className="mb-3 text-center",
                            style={
                                "textAlign": "center",
                                "width": "95%",
                                "margin": "0 auto",
                                "fontSize": f"{TITLE_FONT_SIZE}px",
                                "fontWeight": "600",
                            },
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="stability-spectrum-graph",
                                            figure=self.create_stability_spectrum_figure(
                                                np.array([]), np.array([])
                                            ),
                                            config=GRAPH_CONFIG,
                                            style={
                                                "width": "100%",
                                                "margin": "0 auto",
                                                "height": "300px",
                                            },
                                        )
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="eigenvalue-spectrum-graph",
                                            figure=self.create_eigenvalue_spectrum_figure(
                                                np.array([]), 0.0
                                            ),
                                            config=GRAPH_CONFIG,
                                            style={
                                                "width": "100%",
                                                "margin": "0 auto",
                                                "height": "300px",
                                            },
                                        )
                                    ],
                                    width=6,
                                ),
                            ],
                            style={"width": "95%", "margin": "0 auto"},
                        ),
                    ],
                    className="mt-4",
                ),
                # Forced Response Analysis
                html.Div(
                    [
                        html.H5(
                            "Forced Response Analysis",
                            className="mb-3 text-center",
                            style={
                                "textAlign": "center",
                                "width": "95%",
                                "margin": "0 auto",
                                "fontSize": f"{TITLE_FONT_SIZE}px",
                                "fontWeight": "600",
                            },
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="static-gain-graph",
                                            figure=self.create_static_gain_figure(
                                                np.array([]), np.array([])
                                            ),
                                            config=GRAPH_CONFIG,
                                            style={
                                                "width": "100%",
                                                "margin": "0 auto",
                                                "height": "300px",
                                            },
                                        )
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="spatiotemporal-gain-graph",
                                            figure=self.create_spatiotemporal_gain_figure(
                                                np.array([]), np.array([]), np.array([])
                                            ),
                                            config=GRAPH_CONFIG,
                                            style={
                                                "width": "100%",
                                                "margin": "0 auto",
                                                "height": "300px",
                                            },
                                        )
                                    ],
                                    width=6,
                                ),
                            ],
                            style={"width": "95%", "margin": "0 auto"},
                        ),
                    ],
                    className="mt-4",
                ),
            ],
            width=4,
            className="px-5",
        )

    def setup_layout(self):
        """Set up the dashboard layout."""
        # Add interval component for updates
        interval = dcc.Interval(
            id="interval-component", interval=self.update_interval, n_intervals=0, disabled=False
        )

        # Add slower interval for stability spectrum updates
        spectrum_interval = dcc.Interval(
            id="spectrum-interval",
            interval=1000,  # How often to update spectrum (1 second for expensive computations)
            n_intervals=0,
            disabled=False,
        )

        # Store component for currently selected cell
        selected_cell = dcc.Store(id="selected-cell", data=None)

        # Store component for selected populations for analysis
        selected_populations = dcc.Store(
            id="selected-populations",
            data=[
                "L23_E",
                "L23_SST",
                "L23_PV",
                "L4_E",
                "L4_SST",
                "L4_PV",
                "L5_E",
                "L5_SST",
                "L5_PV",
            ],
        )

        # Hidden button for resetting slider state
        reset_btn = html.Button(id="reset-slider-state-btn", style={"display": "none"}, n_clicks=0)

        self.app.layout = dbc.Container(
            [
                # Utility components
                interval,
                spectrum_interval,
                selected_cell,
                selected_populations,
                reset_btn,
                # Main content: three columns
                dbc.Row(
                    [
                        # Left column: activity visualization
                        self._create_activity_visualization(),
                        # Middle column: connectivity matrix
                        self._create_connectivity_matrix(),
                        # Right column: Control panel
                        dbc.Col(
                            [
                                # Container for control sliders
                                html.Div(
                                    self.create_control_panel(),
                                    className="control-panel-column",
                                    style=CONTROL_PANEL_STYLE,
                                )
                            ],
                            width=4,
                            className="px-4",
                        ),
                    ],
                    className="g-0",
                ),  # Remove gutters from main row
            ],
            fluid=True,
            className="py-3",
        )

    def create_layer_row(self, layer: str) -> dbc.Row:
        """Create a row for a single cortical layer with cell types as columns."""
        ordered_cell_types = ["SST", "E", "PV"]

        return html.Div(
            dbc.Row(
                [
                    # Layer label - ensure it's properly positioned to the right
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        LAYER_NAMES[layer],
                                        style={
                                            "margin": "0",
                                            "whiteSpace": "nowrap",  # Prevent text wrapping
                                            "fontSize": f"{SUBTITLE_FONT_SIZE}px",
                                            "fontWeight": "600",
                                        },
                                    )
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "flex-end",  # Align to the right
                                    "paddingRight": "60px",  # Large spacing from heatmaps
                                    "height": "100%",
                                    "alignItems": "center",
                                },
                            )
                        ],
                        width=2,
                    ),
                    # Cell type columns
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id={"type": "graph", "id": f"{layer}_{cell_type}"},
                                                figure=self.figures[f"graph-{layer}-{cell_type}"],
                                                config=GRAPH_CONFIG,
                                                className="clickable-heatmap",
                                            )
                                        ],
                                        id={
                                            "type": "graph-container",
                                            "id": f"{layer}_{cell_type}",
                                        },
                                        n_clicks=0,
                                        style={
                                            "display": "inline-block",
                                            "border": "3px solid #7f8c8d",
                                            "transition": "border-color 0.2s",
                                            "cursor": "pointer",
                                        },
                                    )
                                    for cell_type in ordered_cell_types
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "center",
                                    "gap": "15px",
                                    "width": "100%",
                                },
                            )
                        ],
                        width=10,
                    ),
                ],
                className="align-items-center",
                style={"height": "155px"},
            ),
            className="mb-4",
        )

    def create_heatmap(self, data: np.ndarray, cell_type: str) -> go.Figure:
        """Create a heatmap figure for the given neural activity data."""
        return create_heatmap_figure(data, cell_type)

    def _parse_connection_key(self, conn_key):
        """Parse a connection key into source and target components.

        Args:
            conn_key: String in format 'source_to_target' where source and target
                     can be either 'thalamus' or 'layer_celltype'

        Returns:
            Tuple of (source_layer, source_cell, target_layer, target_cell)
        """
        return ConnectionKeyUtils.parse(conn_key)

    def _apply_preset(self, preset):
        """Apply a preset configuration to the simulation."""
        # Update all connection strengths
        for conn_key, strength in preset["connection_strengths"].items():
            # Parse the connection key and update strength
            source_layer, source_cell, target_layer, target_cell = self._parse_connection_key(
                conn_key
            )
            self.simulation.connectivity.set_connection_strength(
                source_layer, source_cell, target_layer, target_cell, strength
            )

        # Update strength scaling factors if present in the preset
        if "strength_scaling" in preset:
            for cell_type, scaling in preset["strength_scaling"].items():
                self.simulation.set_strength_scaling(cell_type, scaling)

        # Update background input if present in the preset
        if "background_input" in preset:
            for cell_type, value in preset["background_input"].items():
                self.simulation.set_background_input(cell_type, value)

        # Update thalamic developmental parameters
        self.simulation.update_thalamic_params(preset)

    def get_connection_key(self, source_layer, source_cell, target_layer, target_cell):
        """Generate a connection key based on source and target information."""
        return ConnectionKeyUtils.build(source_layer, source_cell, target_layer, target_cell)

    def get_max_scaled_strength_magnitude(self):
        """Get the maximum absolute scaled connection strength across all presets.

        This is used to normalize colors across developmental stages for fair comparison.

        Returns:
            Maximum absolute scaled strength value
        """
        max_magnitude = 0.0

        # Check all developmental presets
        for preset in [P0_PRESET, P5_PRESET, P10_PRESET, P15_PRESET]:
            for conn_key, raw_strength in preset["connection_strengths"].items():
                # Parse to get source cell type
                parts = conn_key.split("_to_")
                source_part = parts[0]

                if source_part == "thalamus":
                    scaling = preset["strength_scaling"].get("thalamus", 1.0)
                else:
                    source_cell = source_part.split("_")[1]
                    scaling = preset["strength_scaling"].get(source_cell, 1.0)

                scaled_strength = raw_strength * scaling
                max_magnitude = max(max_magnitude, abs(scaled_strength))

        return max_magnitude

    def create_connection_matrix(self) -> html.Div:
        """Create a matrix visualization of all layer and cell type connections."""
        # Define the labels/indices for the matrix
        all_populations = [(layer, cell_type) for layer in LAYERS for cell_type in CELL_TYPES]
        all_populations.append(("Th", None))  # Add thalamus

        # Create the main header row with layer spans
        main_header_cells = [html.Th("", colSpan=2, style=HEADER_STYLE)]

        # Add layer headers that span 3 columns each (for E, SST, PV)
        for layer in LAYERS:
            main_header_cells.append(
                html.Th(
                    LAYER_NAMES[layer],
                    className="text-center fw-bold",
                    colSpan=3,  # Span all cell types
                    style={
                        **HEADER_STYLE,
                        "backgroundColor": (
                            LAYER_COLORS["L4"] if layer == "L4" else LAYER_COLORS["default"]
                        ),
                        "color": "#2c3e50",
                        "padding": "10px 5px",
                        "fontSize": "0.9rem",
                        "fontWeight": "600",
                    },
                )
            )

        # Create sub-header row for cell types
        sub_header_cells = [html.Th("", style=HEADER_STYLE) for _ in range(2)]

        # Add all cell types under their respective layers
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                # Get base color from CELL_ACTIVITY_COLORS with 0.2 opacity
                header_color = CELL_ACTIVITY_COLORS[cell_type]["bg"](0.2)
                sub_header_cells.append(
                    html.Th(
                        cell_type,
                        className="text-center",
                        style={
                            **HEADER_STYLE,
                            "backgroundColor": header_color,
                            "color": "#2c3e50",
                            "padding": "8px 5px",
                            "fontSize": "0.9rem",
                            "fontWeight": "500",
                            "borderRight": "1px solid #ddd" if cell_type == "PV" else "none",
                        },
                    )
                )

        # Generate matrix rows
        rows = []
        current_layer = None
        for source in all_populations:
            source_layer, source_cell = source

            # Check if we're starting a new layer group
            is_new_layer = current_layer != source_layer
            if is_new_layer:
                current_layer = source_layer

            # Create row header with layer spanning if first cell in layer
            if is_new_layer:
                layer_name = "Thalamus" if source_layer == "Th" else LAYER_NAMES[source_layer]
                layer_cells_count = 1 if source_layer == "Th" else len(CELL_TYPES)

                bg_color = (
                    LAYER_COLORS["L4"]
                    if source_layer == "L4"
                    else (
                        LAYER_COLORS["transparent"]
                        if source_layer == "Th"
                        else LAYER_COLORS["default"]
                    )
                )

                row_header = html.Th(
                    layer_name,
                    className="fw-bold",
                    rowSpan=layer_cells_count,
                    style={
                        **HEADER_STYLE,
                        "backgroundColor": bg_color,
                        "color": "#2c3e50",
                        "textAlign": "center",
                        "verticalAlign": "middle",
                        "padding": "10px 5px",
                        "height": "100%",
                        "fontSize": "0.9rem",
                        "fontWeight": "600",
                    },
                )
            else:
                row_header = None

            # Create cell type header with 0.2 opacity
            header_color = (
                CELL_ACTIVITY_COLORS.get(source_cell, {"bg": lambda x: "transparent"})["bg"](0.2)
                if source_cell
                else "transparent"
            )
            cell_type_header = html.Th(
                source_cell or "",
                className="text-center",
                style={
                    **HEADER_STYLE,
                    "backgroundColor": header_color,
                    "color": "#2c3e50",
                    "padding": "5px",
                    "fontSize": "0.9rem",
                    "fontWeight": "500",
                },
            )

            # Create data cells
            cells = []
            # Get max magnitude for normalization across all presets
            max_magnitude = self.get_max_scaled_strength_magnitude()

            for target_layer in LAYERS:
                for target_cell in CELL_TYPES:
                    # Skip thalamus to thalamus connections
                    if source_layer == "Th" and target_layer == "Th":
                        cells.append(
                            html.Td(
                                "",
                                className="text-center",
                                style={**CELL_STYLE, "backgroundColor": "#f8f9fa"},
                            )
                        )
                        continue

                    # Get scaled connection strength for display
                    value = self.get_connection_value(
                        source_layer, source_cell, target_layer, target_cell, scaled=True
                    )

                    # Determine cell colors based on scaled connection strength and source cell type
                    bg_color, hover_color = self._get_connection_colors(
                        source_layer, source_cell, value, max_magnitude
                    )

                    # Create cell with unique ID for callbacks
                    cell_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"
                    cells.append(
                        html.Td(
                            f"{value:.2f}",  # Show 2 decimals for scaled values
                            id={"type": "connection-cell", "id": cell_id},
                            className="connection-cell text-center",
                            style={
                                **CELL_STYLE,
                                "backgroundColor": bg_color,
                                "cursor": "pointer",
                                "transition": "background-color 0.2s",
                                "padding": "5px",
                                "fontSize": "0.8rem",
                                "borderRight": "1px solid #ddd" if target_cell == "PV" else "none",
                                "color": "#2c3e50",
                            },
                            **{"data-highlight-color": hover_color},
                        )
                    )

            # Create row with header (if needed) and cells
            is_last_in_layer = (
                source_layer != "Th" and source_cell == "PV"
            ) or source_layer == "Th"
            row_style = {"marginLeft": "0", "marginRight": "0"}
            if is_last_in_layer:
                row_style["borderBottom"] = "1px solid #ddd"

            row_cells = [
                cell for cell in [row_header, cell_type_header, *cells] if cell is not None
            ]
            rows.append(html.Tr(row_cells, style=row_style))

        # Create table with colorbar
        # Use symmetric range based on max magnitude across all presets
        colorbar_max = max_magnitude  # Already computed above for color normalization

        return html.Div(
            [
                html.Div(
                    [
                        # Connection matrix table
                        html.Div(
                            html.Table(
                                [html.Tr(main_header_cells), html.Tr(sub_header_cells), *rows],
                                className="table connection-matrix",
                                style={
                                    "tableLayout": "fixed",
                                    "fontSize": "0.8rem",
                                    "borderCollapse": "collapse",
                                    "width": "auto",
                                    "margin": "0",
                                    "borderSpacing": "0",
                                    "border": "none",
                                },
                            ),
                            style={"display": "flex", "flexDirection": "column"},
                        ),
                        # Colorbar
                        html.Div(
                            [
                                # Top label (max)
                                html.Div(
                                    f"+{colorbar_max:.1f}",
                                    style={
                                        "fontSize": "0.7rem",
                                        "textAlign": "center",
                                        "marginBottom": "2px",
                                        "color": "#2c3e50",
                                        "fontWeight": "500",
                                    },
                                ),
                                # Gradient bar (excitatory to inhibitory)
                                html.Div(
                                    style={
                                        "width": "25px",
                                        "flex": "1",
                                        "background": "linear-gradient(to bottom, #4292c2, rgba(200, 200, 200, 0.2), #D91B12)",
                                        "border": "1px solid #ddd",
                                        "borderRadius": "3px",
                                        "position": "relative",
                                    },
                                    children=[
                                        # Zero label (positioned in middle)
                                        html.Div(
                                            "0.0",
                                            style={
                                                "fontSize": "0.7rem",
                                                "textAlign": "center",
                                                "color": "#2c3e50",
                                                "fontWeight": "500",
                                                "position": "absolute",
                                                "top": "50%",
                                                "left": "50%",
                                                "transform": "translate(-50%, -50%)",
                                                "backgroundColor": "white",
                                                "padding": "0 2px",
                                            },
                                        )
                                    ],
                                ),
                                # Bottom label (min)
                                html.Div(
                                    f"-{colorbar_max:.1f}",
                                    style={
                                        "fontSize": "0.7rem",
                                        "textAlign": "center",
                                        "marginTop": "2px",
                                        "color": "#2c3e50",
                                        "fontWeight": "500",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "marginLeft": "15px",
                                "alignSelf": "stretch",
                            },
                        ),
                    ],
                    style={"display": "flex", "alignItems": "stretch", "justifyContent": "center"},
                )
            ]
        )

    def _get_connection_colors(self, source_layer, source_cell, value, max_magnitude=None):
        """Get background and hover colors for a connection based on source and value.

        Args:
            source_layer: Source layer ('thalamus' or layer name)
            source_cell: Source cell type (E, SST, PV, or None for thalamus)
            value: Connection strength value (scaled)
            max_magnitude: Maximum absolute value for normalization (if None, uses 1.0)

        Returns:
            Tuple of (background_color, hover_color)
        """
        if max_magnitude is None or max_magnitude == 0:
            max_magnitude = 1.0

        if source_layer == "thalamus" or source_layer == "Th":
            # For thalamic connections, always use E color and only positive values
            if value > 0:
                intensity = min(abs(value) / max_magnitude, 1.0) * 0.7
                bg_color = CELL_ACTIVITY_COLORS["E"]["bg"](intensity)
                hover_color = CELL_ACTIVITY_COLORS["E"]["hover"](intensity)
            else:
                bg_color = CELL_ACTIVITY_COLORS["inactive"]["bg"]
                hover_color = CELL_ACTIVITY_COLORS["inactive"]["hover"]
        else:
            # For cell-type specific connections
            if value != 0:
                intensity = min(abs(value) / max_magnitude, 1.0) * 0.7
                if source_cell in ["PV", "SST"]:
                    # For inhibitory cells: use their color for negative values, E color for positive
                    if value < 0:
                        bg_color = CELL_ACTIVITY_COLORS[source_cell]["bg"](intensity)
                        hover_color = CELL_ACTIVITY_COLORS[source_cell]["hover"](intensity)
                    else:
                        bg_color = CELL_ACTIVITY_COLORS["E"]["bg"](intensity)
                        hover_color = CELL_ACTIVITY_COLORS["E"]["hover"](intensity)
                else:  # E cells
                    # For E cells: only show color for positive values
                    if value > 0:
                        bg_color = CELL_ACTIVITY_COLORS["E"]["bg"](intensity)
                        hover_color = CELL_ACTIVITY_COLORS["E"]["hover"](intensity)
                    else:
                        bg_color = CELL_ACTIVITY_COLORS["inactive"]["bg"]
                        hover_color = CELL_ACTIVITY_COLORS["inactive"]["hover"]
            else:
                bg_color = CELL_ACTIVITY_COLORS["inactive"]["bg"]
                hover_color = CELL_ACTIVITY_COLORS["inactive"]["bg"]

        return bg_color, hover_color

    def create_slider_for_cell(self, source_layer, source_cell, target_layer, target_cell, value):
        """Create a slider component for a connection cell."""
        # Set slider range based on excitatory/inhibitory type
        is_excitatory = source_cell == "E" or source_layer == "Th"
        slider_min = 0 if is_excitatory else -1.0
        slider_max = 1.0

        # Create unique ID for slider
        slider_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"

        return html.Div(
            [
                html.Div(
                    f"{source_layer}"
                    + (f"-{source_cell}" if source_cell else "")
                    + f" → {target_layer}-{target_cell}",
                    style={"marginBottom": "5px", "textAlign": "center"},
                ),
                dcc.Slider(
                    id={"type": "matrix-slider", "id": slider_id},
                    min=slider_min,
                    max=slider_max,
                    step=0.1,
                    value=value,
                    marks={
                        slider_min: f"{slider_min:.1f}",
                        0: "0",
                        slider_max / 2: f"{slider_max/2:.1f}",
                        slider_max: f"{slider_max:.1f}",
                    },
                ),
                html.Div(
                    id={"type": "slider-value", "id": slider_id},
                    style={"marginTop": "5px", "textAlign": "center"},
                ),
            ]
        )

    def _get_preset_values(self, preset):
        """Helper function to get values from a preset object or dictionary."""
        values = {
            "tau_e": preset["time_constants"]["E"],
            "tau_sst": preset["time_constants"]["SST"],
            "tau_pv": preset["time_constants"]["PV"],
            "background_input_e": preset["background_input"]["E"],
            "background_input_sst": preset["background_input"]["SST"],
            "background_input_pv": preset["background_input"]["PV"],
            "sigma_thal_e": preset["thalamic_widths"]["E"],
            "sigma_thal_sst": preset["thalamic_widths"]["SST"],
            "sigma_thal_pv": preset["thalamic_widths"]["PV"],
            "sigma_e_out": preset["outgoing_widths"]["E"],
            "sigma_sst_out": preset["outgoing_widths"]["SST"],
            "sigma_pv_out": preset["outgoing_widths"]["PV"],
            "strength_e": preset["strength_scaling"]["E"],
            "strength_sst": preset["strength_scaling"]["SST"],
            "strength_pv": preset["strength_scaling"]["PV"],
            "strength_thal": preset["strength_scaling"]["thalamus"],
            "alpha": preset["thalamic_alpha"],
        }
        return values

    def _create_preset_callback(self, preset_name, preset_obj, allow_duplicate=False):
        """Helper function to create a preset callback."""
        outputs = (
            [
                Output(id, prop, allow_duplicate=allow_duplicate)
                for id, prop in [
                    (o.component_id, o.component_property) for o in self._PRESET_OUTPUTS
                ]
            ]
            if allow_duplicate
            else self._PRESET_OUTPUTS
        )

        @self.app.callback(
            outputs, Input(f"{preset_name}-preset-button", "n_clicks"), prevent_initial_call=True
        )
        def apply_preset_callback(n_clicks):  # pylint: disable=unused-argument
            """Apply the preset configuration."""
            # Apply the preset using the generic apply_preset function
            self._apply_preset(preset_obj)

            # Get values from the preset
            values = self._get_preset_values(preset_obj)

            # Return all values in the expected order
            return (
                values["tau_e"],
                values["tau_sst"],
                values["tau_pv"],
                values["background_input_e"],
                values["background_input_sst"],
                values["background_input_pv"],
                values["sigma_thal_e"],
                values["sigma_thal_sst"],
                values["sigma_thal_pv"],
                values["sigma_e_out"],
                values["sigma_sst_out"],
                values["sigma_pv_out"],
                values["strength_e"],
                values["strength_sst"],
                values["strength_pv"],
                values["strength_thal"],
                values["alpha"],
                self.create_connection_matrix(),
            )

        return apply_preset_callback

    def get_connection_value(
        self, source_layer, source_cell, target_layer, target_cell, scaled=False
    ):
        """Get the current connection strength value.

        Args:
            source_layer: Source layer
            source_cell: Source cell type
            target_layer: Target layer
            target_cell: Target cell type
            scaled: If True, return strength-scaled value; if False, return raw amplitude

        Returns:
            Connection strength (raw or scaled)
        """
        try:
            # Get connection strength
            conn_key = self.get_connection_key(source_layer, source_cell, target_layer, target_cell)

            # First try to get the value from the simulation connectivity
            if hasattr(self, "simulation") and hasattr(self.simulation, "connectivity"):
                # Convert 'Th' to 'thalamus' for the simulation API
                source_layer_sim = "thalamus" if source_layer == "Th" else source_layer

                if scaled:
                    return self.simulation.connectivity.get_scaled_connection_strength(
                        source_layer_sim, source_cell, target_layer, target_cell
                    )
                else:
                    return self.simulation.connectivity.get_connection_strength(
                        source_layer_sim, source_cell, target_layer, target_cell
                    )

            # Fall back to config-based lookup
            if conn_key in LAYER_CONNECTIVITY_PARAMS:
                raw_value = LAYER_CONNECTIVITY_PARAMS[conn_key]["amplitude"]
                if scaled:
                    # Apply strength scaling from config
                    if source_layer == "Th" or source_layer == "thalamus":
                        scaling = INITIAL_STRENGTH_SCALING.get("thalamus", 1.0)
                    else:
                        scaling = INITIAL_STRENGTH_SCALING.get(source_cell, 1.0)
                    return raw_value * scaling
                return raw_value

            # Default to 0 if not found
            return 0.0
        except (AttributeError, KeyError) as e:
            print(f"Error getting connection from simulation: {e!s}")
            return 0.0

    def setup_callbacks(self):
        """Set up the dashboard callbacks for interactivity."""
        # Add callbacks for preset buttons
        self._create_preset_callback("P0", P0_PRESET)
        self._create_preset_callback("P5", P5_PRESET, allow_duplicate=True)
        self._create_preset_callback("P10", P10_PRESET, allow_duplicate=True)
        self._create_preset_callback("P15", P15_PRESET, allow_duplicate=True)

        # Initialize slider container (hidden)
        @self.app.callback(
            [
                Output("slider-container", "style"),
                Output("slider-container", "children"),
                Output("selected-cell", "data"),
            ],
            [Input("connection-matrix-container", "children")],
            [State("selected-cell", "data")],
        )
        def initialize_slider_container(_, current_data):  # pylint: disable=unused-argument
            """Initialize the slider container as hidden when the dashboard loads."""
            return SLIDER_HIDDEN_STYLE, [], None

        # Handle cell clicks to show the slider
        @self.app.callback(
            [
                Output("slider-container", "style", allow_duplicate=True),
                Output("slider-container", "children", allow_duplicate=True),
                Output("selected-cell", "data", allow_duplicate=True),
            ],
            [Input({"type": "connection-cell", "id": ALL}, "n_clicks")],
            [State("selected-cell", "data")],
            prevent_initial_call=True,
        )
        def handle_cell_click(clicks, current_data):  # pylint: disable=unused-argument
            """Show the connection strength slider when a matrix cell is clicked."""
            # Early return if not a valid click
            triggered_prop_id = get_triggered_id()
            if not triggered_prop_id or not is_valid_click(get_triggered_value()):
                return no_update_tuple(3)

            # Parse the pattern-match ID
            cell_data = parse_pattern_match_id(triggered_prop_id)
            if not cell_data or "id" not in cell_data:
                return no_update_tuple(3)

            clicked_id = cell_data["id"]

            # Parse connection info from the cell ID
            parsed = parse_connection_cell_id(clicked_id)
            if not parsed:
                print(f"Invalid cell ID format: {clicked_id}")
                return no_update_tuple(3)

            source_layer, source_cell, target_layer, target_cell = parsed

            # Get current connection value and create slider
            value = self.get_connection_value(source_layer, source_cell, target_layer, target_cell)
            slider = self.create_slider_for_cell(
                source_layer, source_cell, target_layer, target_cell, value
            )

            # Build connection state data
            connection_data = {
                "source_layer": source_layer,
                "source_cell": source_cell,
                "target_layer": target_layer,
                "target_cell": target_cell,
                "slider_id": clicked_id,
            }

            return SLIDER_POPUP_STYLE, slider, connection_data

        # Update connection strength when slider changes
        @self.app.callback(
            Output({"type": "slider-value", "id": MATCH}, "children"),
            Input({"type": "matrix-slider", "id": MATCH}, "value"),
            State("selected-cell", "data"),
        )
        def update_connection_value(value, connection_data):
            """Update the connection strength value display and simulation when slider changes."""
            if not connection_data:
                return ""

            try:
                # Update connection in simulation
                source_layer = connection_data["source_layer"]
                source_cell = connection_data["source_cell"]
                target_layer = connection_data["target_layer"]
                target_cell = connection_data["target_cell"]

                # Handle thalamus special case
                if source_layer == "Th":
                    source_layer = "thalamus"

                # Update simulation connection strength
                self.simulation.connectivity.set_connection_strength(
                    source_layer, source_cell, target_layer, target_cell, value
                )

                return f"Value: {value:.1f}"
            except (KeyError, AttributeError, ValueError) as e:
                print(f"Error updating connection value: {e!s}")
                return f"Error: {e!s}"

        # Update connection cell in matrix when slider changes
        @self.app.callback(
            [
                Output({"type": "connection-cell", "id": MATCH}, "children"),
                Output({"type": "connection-cell", "id": MATCH}, "style"),
                Output({"type": "connection-cell", "id": MATCH}, "data-highlight-color"),
            ],
            Input({"type": "matrix-slider", "id": MATCH}, "value"),
            [
                State({"type": "connection-cell", "id": MATCH}, "style"),
                State({"type": "connection-cell", "id": MATCH}, "id"),
                State("selected-cell", "data"),
            ],
        )
        def update_matrix_cell(
            raw_value, current_style, cell_id, connection_data
        ):  # pylint: disable=unused-argument
            """Update the matrix cell appearance and value when the slider changes."""
            if raw_value is None:
                return no_update_tuple(3)

            try:
                # Parse cell ID from the dictionary
                cell_id_str = cell_id["id"]
                parsed = parse_connection_cell_id(cell_id_str)
                if not parsed:
                    return no_update_tuple(3)

                source_layer, source_cell, target_layer, target_cell = parsed

                # Convert to thalamus if needed
                source_layer_sim = "thalamus" if source_layer == "Th" else source_layer

                # Get the scaled value to display (raw_value * strength_scaling)
                if hasattr(self, "simulation") and hasattr(self.simulation, "connectivity"):
                    scaled_value = self.simulation.connectivity.get_scaled_connection_strength(
                        source_layer_sim, source_cell, target_layer, target_cell
                    )
                else:
                    # Fallback: compute scaled value manually
                    if source_layer_sim == "thalamus":
                        scaling = INITIAL_STRENGTH_SCALING.get("thalamus", 1.0)
                    else:
                        scaling = INITIAL_STRENGTH_SCALING.get(source_cell, 1.0)
                    scaled_value = raw_value * scaling

                # Get max magnitude for color normalization
                max_magnitude = self.get_max_scaled_strength_magnitude()

                # Determine cell colors based on scaled connection strength and source cell type
                bg_color, hover_color = self._get_connection_colors(
                    source_layer, source_cell, scaled_value, max_magnitude
                )

                # Update style with new background color while preserving other styles
                updated_style = {
                    **CELL_STYLE,
                    "backgroundColor": bg_color,
                    "cursor": "pointer",
                    "transition": "background-color 0.2s",
                    "padding": "5px",
                    "fontSize": "0.8rem",
                    "borderRight": "1px solid #ddd" if target_cell == "PV" else "none",
                    "color": "#2c3e50",
                }

                # Return updated text (scaled value), style, and hover color
                return f"{scaled_value:.2f}", updated_style, hover_color
            except (KeyError, ValueError) as e:
                print(f"Error updating matrix cell: {e!s}")
                return no_update_tuple(3)

        # Reset the slider when clicking the reset button
        @self.app.callback(
            [
                Output("slider-container", "style", allow_duplicate=True),
                Output("slider-container", "children", allow_duplicate=True),
                Output("selected-cell", "data", allow_duplicate=True),
            ],
            Input("reset-slider-state-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def reset_slider_state(n_clicks):  # pylint: disable=unused-argument
            """Reset the slider state when clicking outside the slider or matrix."""
            return SLIDER_HIDDEN_STYLE, [], None

        # JavaScript to position the slider near the clicked cell
        self.app.clientside_callback(
            """
            function(styles, children, data) {
                // Skip if no data or slider is hidden
                if (!data || styles.display === 'none') {
                    return window.dash_clientside.no_update;
                }

                // Find the cell that was clicked
                const cellId = data.slider_id;
                const cell = document.querySelector(`[id*="${cellId}"]`);

                if (cell) {
                    const rect = cell.getBoundingClientRect();
                    const sliderContainer = document.getElementById('slider-container');

                    if (sliderContainer) {
                        // Position below the cell
                        sliderContainer.style.top = (rect.bottom + window.scrollY + 5) + 'px';
                        sliderContainer.style.left = (rect.left + window.scrollX - 75) + 'px';

                        // Highlight the active cell
                        document.querySelectorAll('.connection-cell').forEach(c => c.style.outline = 'none');
                        cell.style.outline = '2px solid white';

                        // Clean up any existing click handler
                        if (window.outsideClickHandler) {
                            document.removeEventListener('click', window.outsideClickHandler);
                            window.outsideClickHandler = null;
                        }

                        // Create new document click handler for closing the slider
                        window.outsideClickHandler = function(e) {
                            if (!e.target.closest('.connection-cell') &&
                                !e.target.closest('#slider-container')) {

                                // Reset with the button
                                const resetBtn = document.getElementById('reset-slider-state-btn');
                                if (resetBtn) resetBtn.click();

                                // Clear active cell highlighting
                                document.querySelectorAll('.connection-cell').forEach(c => {
                                    c.style.outline = 'none';
                                });
                            }
                        };

                        // Add handler with slight delay to avoid immediate trigger
                        setTimeout(() => {
                            document.addEventListener('click', window.outsideClickHandler);
                        }, 50);
                    }
                }

                return window.dash_clientside.no_update;
            }
            """,
            Output("slider-container", "id"),
            [
                Input("slider-container", "style"),
                Input("slider-container", "children"),
                Input("selected-cell", "data"),
            ],
        )

        # Add callback for updating connectivity widths in simulation
        @self.app.callback(
            [Output("interval-component", "n_intervals", allow_duplicate=True)],
            [
                Input("thalamic-width-e-slider", "value"),
                Input("thalamic-width-sst-slider", "value"),
                Input("thalamic-width-pv-slider", "value"),
                Input("outgoing-width-e-slider", "value"),
                Input("outgoing-width-sst-slider", "value"),
                Input("outgoing-width-pv-slider", "value"),
            ],
            [State("interval-component", "n_intervals")],
            prevent_initial_call=True,
        )
        def update_connectivity_parameters(
            sigma_thal_e,
            sigma_thal_sst,
            sigma_thal_pv,
            sigma_e_out,
            sigma_sst_out,
            sigma_pv_out,
            n_intervals,
        ):
            """Update all connectivity parameters in the simulation."""
            # Update thalamic connections
            thalamic_params = [("E", sigma_thal_e), ("SST", sigma_thal_sst), ("PV", sigma_thal_pv)]
            for layer in LAYERS:
                for cell_type, sigma in thalamic_params:
                    self.simulation.set_connection_sigma("thalamus", None, layer, cell_type, sigma)

            # Update cell type outgoing connections
            outgoing_params = [
                ("E", sigma_e_out, CELL_TYPES),  # E connects to all cell types
                ("SST", sigma_sst_out, ["E", "PV"]),  # SST only connects to E and PV
                ("PV", sigma_pv_out, CELL_TYPES),  # PV connects to all cell types
            ]

            for source_layer in LAYERS:
                for target_layer in LAYERS:
                    for source_cell, sigma, target_cells in outgoing_params:
                        for target_cell in target_cells:
                            if (source_cell, target_cell) in CONNECTIONS:
                                self.simulation.set_connection_sigma(
                                    source_layer, source_cell, target_layer, target_cell, sigma
                                )

            # Return unchanged intervals to not disrupt the update loop
            return [n_intervals]

        # Update the graphs with neural activity
        @self.app.callback(
            # Outputs: all graph figures
            [
                Output({"type": "graph", "id": f"{layer}_{cell_type}"}, "figure")
                for layer in LAYERS
                for cell_type in CELL_TYPES
            ]
            + [
                Output("graph-thalamus", "figure"),
                Output("correlation-by-layer", "figure"),
                Output("correlation-by-celltype", "figure"),
                Output("events-by-layer", "figure"),
                Output("events-by-celltype", "figure"),
            ],
            # Input: only interval trigger (parameter sliders update state via separate callbacks)
            [Input("interval-component", "n_intervals")],
            # States: all parameters and pause button state
            [
                State("alpha-slider", "value"),
                State("tau-e-slider", "value"),
                State("tau-sst-slider", "value"),
                State("tau-pv-slider", "value"),
                State("thalamic-width-e-slider", "value"),
                State("thalamic-width-sst-slider", "value"),
                State("thalamic-width-pv-slider", "value"),
                State("outgoing-width-e-slider", "value"),
                State("outgoing-width-sst-slider", "value"),
                State("outgoing-width-pv-slider", "value"),
                State("pause-button", "n_clicks"),
            ],
        )
        def update_graphs(
            n_intervals,
            alpha,
            tau_e,
            tau_sst,
            tau_pv,  # pylint: disable=unused-argument
            sigma_thal_e,
            sigma_thal_sst,
            sigma_thal_pv,
            sigma_e_out,
            sigma_sst_out,
            sigma_pv_out,
            pause_clicks,
        ):
            """Update all graphs based on current simulation state."""
            # Check if simulation is paused
            is_paused = pause_clicks is not None and pause_clicks % 2 == 1
            if is_paused:
                # Return current figures without updating if paused
                return [
                    self.figures[f"graph-{layer}-{cell_type}"]
                    for layer in LAYERS
                    for cell_type in CELL_TYPES
                ] + [
                    self.figures["graph-thalamus"],
                    self.figures["correlation-by-layer"],
                    self.figures["correlation-by-celltype"],
                    self.figures["events-by-layer"],
                    self.figures["events-by-celltype"],
                ]

            # Update neuron parameters
            self.simulation.set_time_constant("E", tau_e)
            self.simulation.set_time_constant("SST", tau_sst)
            self.simulation.set_time_constant("PV", tau_pv)
            # Note: gains are set by presets, not sliders

            # Update all connectivity widths
            for layer in LAYERS:
                # Update thalamic inputs
                self.simulation.set_connection_sigma("thalamus", None, layer, "E", sigma_thal_e)
                self.simulation.set_connection_sigma("thalamus", None, layer, "SST", sigma_thal_sst)
                self.simulation.set_connection_sigma("thalamus", None, layer, "PV", sigma_thal_pv)

                # Update outgoing connections for each source layer
                for source_layer in LAYERS:
                    for source_cell, sigma in [
                        ("E", sigma_e_out),
                        ("SST", sigma_sst_out),
                        ("PV", sigma_pv_out),
                    ]:
                        for target_cell in CELL_TYPES:
                            if (source_cell, target_cell) in CONNECTIONS:
                                self.simulation.set_connection_sigma(
                                    source_layer, source_cell, layer, target_cell, sigma
                                )

            # Update simulation state
            activities = self.simulation.update(alpha=alpha)

            # Update all figures
            updated_figures = []

            # Update each layer-cell type figure
            for layer in LAYERS:
                for cell_type in CELL_TYPES:
                    fig_id = f"graph-{layer}-{cell_type}"
                    fig = self.figures[fig_id]

                    # Update the figure data
                    with fig.batch_update():
                        # Get the layer's activity for this cell type
                        data = activities[layer][cell_type].reshape(
                            self.simulation.grid_size, self.simulation.grid_size
                        )
                        fig.data[0]["z"] = data

                        # Keep consistent scaling for fair comparison across heatmaps
                        fig.update_traces(zmin=HEATMAP_ZMIN, zmax=HEATMAP_ZMAX)

                    updated_figures.append(fig)

            # Update thalamus figure
            thal_fig = self.figures["graph-thalamus"]
            with thal_fig.batch_update():
                thal_data = activities["thalamus"]
                thal_fig.data[0]["z"] = thal_data
                # Keep consistent scaling for fair comparison with cortical heatmaps
                thal_fig.update_traces(zmin=HEATMAP_ZMIN, zmax=HEATMAP_ZMAX)

            updated_figures.append(thal_fig)

            # Update correlation tracking
            self.simulation_time += UPDATE_INTERVAL / 1000.0
            self.correlation_activity_buffer.append(activities)
            # Keep buffer size limited to prevent memory issues
            max_buffer_size = CORRELATION_HISTORY_LENGTH + 10
            if len(self.correlation_activity_buffer) > max_buffer_size:
                # Remove oldest entries to maintain buffer size
                self.correlation_activity_buffer = self.correlation_activity_buffer[
                    -max_buffer_size:
                ]

            # Compute correlations and events every N updates for efficiency
            if n_intervals % CORRELATION_UPDATE_INTERVAL == 0:
                self._update_time_series_data(
                    self._compute_rolling_correlations(), self.correlation_time_series
                )
                self._update_time_series_data(
                    self._compute_synchronous_events(), self.event_time_series
                )

            # Update correlation figures (always, for smooth rendering)
            corr_fig_layer, corr_fig_celltype = self._update_correlation_figures()
            updated_figures.extend([corr_fig_layer, corr_fig_celltype])

            # Update event figures
            event_fig_layer, event_fig_celltype = self._update_event_figures()
            updated_figures.extend([event_fig_layer, event_fig_celltype])

            return updated_figures

        # Toggle simulation pause state
        @self.app.callback(
            Output("interval-component", "disabled"), [Input("pause-button", "n_clicks")]
        )
        def toggle_simulation(n_clicks):
            return n_clicks is not None and n_clicks % 2 == 1

        # Add callback for updating strength scaling factors
        @self.app.callback(
            [
                Output("interval-component", "n_intervals", allow_duplicate=True),
                Output("connection-matrix-container", "children", allow_duplicate=True),
            ],
            [
                Input("strength-scaling-e-slider", "value"),
                Input("strength-scaling-sst-slider", "value"),
                Input("strength-scaling-pv-slider", "value"),
                Input("strength-scaling-thalamus-slider", "value"),
            ],
            [State("interval-component", "n_intervals")],
            prevent_initial_call=True,
        )
        def update_strength_scaling_parameters(
            e_scaling, sst_scaling, pv_scaling, thalamus_scaling, n_intervals
        ):
            """Update all strength scaling parameters in the simulation."""
            # Update strength scaling parameters
            self.simulation.set_strength_scaling("E", e_scaling)
            self.simulation.set_strength_scaling("SST", sst_scaling)
            self.simulation.set_strength_scaling("PV", pv_scaling)
            self.simulation.set_strength_scaling("thalamus", thalamus_scaling)

            # Regenerate the connection matrix with updated scaled values
            updated_matrix = self.create_connection_matrix()

            # Return unchanged intervals and updated matrix
            return [n_intervals, updated_matrix]

        # Add callback for updating background input parameters
        @self.app.callback(
            [Output("interval-component", "n_intervals", allow_duplicate=True)],
            [
                Input("background-input-e-slider", "value"),
                Input("background-input-sst-slider", "value"),
                Input("background-input-pv-slider", "value"),
            ],
            [State("interval-component", "n_intervals")],
            prevent_initial_call=True,
        )
        def update_background_input_parameters(bg_e, bg_sst, bg_pv, n_intervals):
            """Update all background input parameters in the simulation."""
            # Update background input for each cell type
            self.simulation.set_background_input("E", bg_e)
            self.simulation.set_background_input("SST", bg_sst)
            self.simulation.set_background_input("PV", bg_pv)

            # Return unchanged intervals to not disrupt the update loop
            return [n_intervals]

        # Update stability spectrum and eigenvalue spectrum periodically only
        # Remove parameter inputs to prevent multiple rapid firings on preset changes
        @self.app.callback(
            [
                Output("stability-spectrum-graph", "figure"),
                Output("eigenvalue-spectrum-graph", "figure"),
            ],
            [Input("spectrum-interval", "n_intervals"), Input("selected-populations", "data")],
        )
        def update_stability_spectrum(
            n_intervals, selected_pops
        ):  # pylint: disable=unused-argument
            """Update the stability spectrum and eigenvalue spectrum graphs periodically."""
            # Prevent concurrent computation
            if self._computing_stability:
                return no_update_tuple(2)

            try:
                self._computing_stability = True

                # Check if any populations are selected
                if not selected_pops or len(selected_pops) == 0:
                    fig = create_empty_message_figure("Select populations by clicking heatmaps")
                    return fig, fig

                # Build preset from current simulation state
                preset = self.build_current_preset()

                # Compute stability spectrum for selected populations
                k_values, max_real_values, eigenvalues_at_max_k, k_max = (
                    self.compute_stability_spectrum(preset, selected_pops)
                )

                # Create both figures
                stability_fig = self.create_stability_spectrum_figure(k_values, max_real_values)
                eigenvalue_fig = self.create_eigenvalue_spectrum_figure(eigenvalues_at_max_k, k_max)

                return stability_fig, eigenvalue_fig

            except Exception as e:
                print(f"Error updating stability spectrum: {e}")
                # Return empty figures on error
                empty_stability = self.create_stability_spectrum_figure(np.array([]), np.array([]))
                empty_eigenvalue = self.create_eigenvalue_spectrum_figure(np.array([]), 0.0)
                return empty_stability, empty_eigenvalue
            finally:
                self._computing_stability = False

        # Update forced response graphs periodically only
        # Remove parameter inputs to prevent multiple rapid firings on preset changes
        @self.app.callback(
            [Output("static-gain-graph", "figure"), Output("spatiotemporal-gain-graph", "figure")],
            [Input("spectrum-interval", "n_intervals"), Input("selected-populations", "data")],
        )
        def update_forced_response(n_intervals, selected_pops):  # pylint: disable=unused-argument
            """Update the forced response graphs periodically."""
            # Prevent concurrent computation
            if self._computing_forced_response:
                return no_update_tuple(2)

            try:
                self._computing_forced_response = True

                # Check if any populations are selected
                if not selected_pops or len(selected_pops) == 0:
                    fig = create_empty_message_figure("Select populations by clicking heatmaps")
                    return fig, fig

                # Build preset from current simulation state
                preset = self.build_current_preset()

                # Compute static gain for selected populations
                k_values_static, gain_values = self.compute_static_gain(preset, selected_pops)
                static_fig = self.create_static_gain_figure(k_values_static, gain_values)

                # Compute spatiotemporal gain for selected populations
                k_values_st, omega_values, gain_matrix = self.compute_spatiotemporal_gain(
                    preset, selected_pops
                )
                spatiotemporal_fig = self.create_spatiotemporal_gain_figure(
                    k_values_st, omega_values, gain_matrix
                )

                return static_fig, spatiotemporal_fig

            except Exception as e:
                print(f"Error updating forced response: {e}")
                # Return empty figures on error
                empty_static = self.create_static_gain_figure(np.array([]), np.array([]))
                empty_st = self.create_spatiotemporal_gain_figure(
                    np.array([]), np.array([]), np.array([])
                )
                return empty_static, empty_st
            finally:
                self._computing_forced_response = False

        # Handle heatmap clicks for population selection using n_clicks on containers
        @self.app.callback(
            [
                Output("selected-populations", "data", allow_duplicate=True),
                Output({"type": "graph-container", "id": ALL}, "style"),
            ],
            [Input({"type": "graph-container", "id": ALL}, "n_clicks")],
            [
                State("selected-populations", "data"),
                State({"type": "graph-container", "id": ALL}, "id"),
            ],
            prevent_initial_call=True,
        )
        def toggle_population_selection(
            n_clicks_list, selected_pops, container_ids
        ):  # pylint: disable=unused-argument
            """Toggle population selection when clicking on heatmaps."""
            # Parse the triggered input
            triggered_prop_id = get_triggered_id()
            if not triggered_prop_id or "n_clicks" not in triggered_prop_id:
                return no_update_tuple(2)

            # Extract the population ID from the pattern-match callback
            triggered_dict = parse_pattern_match_id(triggered_prop_id)
            if not triggered_dict or "id" not in triggered_dict:
                return no_update_tuple(2)

            pop_id = triggered_dict["id"]

            # Toggle the population in the selection
            selected_pops = list(selected_pops) if selected_pops else []
            if pop_id in selected_pops:
                selected_pops.remove(pop_id)
            else:
                selected_pops.append(pop_id)

            # Update styles for all graph containers
            updated_styles = []
            for container_id in container_ids:
                pop_id_for_style = container_id["id"]
                if pop_id_for_style in selected_pops:
                    # Selected: grey border
                    style = {
                        "display": "inline-block",
                        "border": "3px solid #7f8c8d",
                        "transition": "border-color 0.2s",
                        "cursor": "pointer",
                    }
                else:
                    # Not selected: transparent border
                    style = {
                        "display": "inline-block",
                        "border": "3px solid transparent",
                        "transition": "border-color 0.2s",
                        "cursor": "pointer",
                    }
                updated_styles.append(style)

            return selected_pops, updated_styles

        # Update the selected populations display text
        @self.app.callback(
            Output("selected-populations-display", "children"),
            [Input("selected-populations", "data")],
        )
        def update_selected_populations_display(selected_pops):
            """Update the display showing which populations are selected for analysis."""
            return format_analysis_display(selected_pops)

    def create_control_panel(self):
        """Create the control panel with all sliders and controls."""
        return create_control_panel()

    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard application."""
        self.app.run_server(
            debug=debug, port=port, threaded=True, dev_tools_silence_routes_logging=True
        )
