"""Dashboard module for visualizing the cortical circuit simulation."""

import logging

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

from src.analysis.bifurcation.config import ANALYSIS_PARAMS
from src.analysis.descriptive.config import HEATMAP_VMAX
from src.model.config import (
    CELL_ACTIVITY_COLORS,
    CELL_COLORS,
    CELL_TYPES,
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
    create_eigenvalue_spectrum_figure,
    create_heatmap_figure,
    create_initial_correlation_figure,
    create_initial_event_figure,
    create_stability_spectrum_figure,
    create_static_gain_figure,
    create_spatiotemporal_gain_figure,
)

# Import utility helpers from dedicated module
from src.visualization.dashboard_utils import (
    build_connection_key,
    parse_connection_key,
)

# Import callback registration modules
from src.visualization import (
    dashboard_callbacks_analysis,
    dashboard_callbacks_connectivity,
    dashboard_callbacks_core,
    dashboard_callbacks_presets,
)

# Import centralized dashboard config
from src.visualization.dashboard_config import (
    ACTIVITY_THRESHOLD,
    CORRELATION_CELL_SAMPLE_RATE,
    CORRELATION_DISPLAY_SECONDS,
    CORRELATION_HISTORY_LENGTH,
    min_mean_rate,
    CORRELATION_WINDOW_MS,
    SYNCHRONOUS_EVENT_THRESHOLD,
)

# Import analysis delegate
from src.visualization.dashboard_analysis import DashboardAnalysis

logger = logging.getLogger(__name__)


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
            "total": [],
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

        # Initialize analysis delegate for stability/gain computations
        self.analysis = DashboardAnalysis(self)

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
            "thalamic_alpha": self.simulation.preset.get("thalamic_alpha", THALAMIC_ALPHA),
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
        # Handle "total" if present
        if "total" in results and "total" in time_series_dict:
            time_series_dict["total"].append((self.simulation_time, results["total"]))
            time_series_dict["total"] = [
                (t, v) for t, v in time_series_dict["total"] if t >= cutoff_time
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
            """Compute average pairwise correlation for a group.

            Best practice: return NaN when correlation is undefined (too few
            points, constant data). Do not replace NaN with 0, since 0 denotes
            no linear relationship; NaN denotes cannot compute.
            """
            if not data_list or len(data_list) < 2:
                return np.nan
            data_array = np.array(data_list)  # Shape: (num_timepoints, num_cells_in_group)
            if data_array.shape[1] < 2:
                return np.nan
            if np.mean(data_array) < min_mean_rate:
                return np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(data_array.T)
            if corr_matrix.shape[0] <= 1:
                return np.nan
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            # nanmean: average only defined pairs; if all NaN (e.g. constant data) -> NaN
            mean_corr = np.nanmean(corr_matrix[mask])
            return np.nan if not np.isfinite(mean_corr) else float(mean_corr)

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

        # all_cells columns are ordered: for layer in LAYERS, for cell_type in CELL_TYPES,
        # with each population flattened (grid_size^2 cells). We need per-cell column
        # indices for each layer and each cell type, not population indices.
        grid_size = self.simulation.grid_size
        n = grid_size * grid_size  # cells per population

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

        def col_indices_layer(layer):
            """Column indices in all_cells for all cells in this layer."""
            layer_idx = LAYERS.index(layer)
            start = layer_idx * 3 * n
            end = (layer_idx + 1) * 3 * n
            return np.arange(start, end)

        def col_indices_celltype(cell_type):
            """Column indices in all_cells for all cells of this type across layers."""
            ct_idx = CELL_TYPES.index(cell_type)
            return np.concatenate(
                [np.arange((p * 3 + ct_idx) * n, (p * 3 + ct_idx) * n + n) for p in range(3)]
            )

        return {
            "by_layer": {
                layer: count_events(all_cells[:, col_indices_layer(layer)]) for layer in LAYERS
            },
            "by_celltype": {
                cell_type: count_events(all_cells[:, col_indices_celltype(cell_type)])
                for cell_type in CELL_TYPES
            },
            "total": count_events(all_cells),
        }

    def compute_stability_spectrum(self, preset: dict, selected_pops: list | None = None) -> tuple:
        """Compute stability spectrum (max Re(lambda) vs k) for current network state.

        Delegates to DashboardAnalysis.

        Args:
            preset: Preset dictionary with current network parameters
            selected_pops: Optional list of population IDs to analyze.

        Returns:
            Tuple of (k_values, max_real_eigenvalues, eigenvalues_at_max_k, k_max)
        """
        return self.analysis.compute_stability_spectrum(preset, selected_pops)

    def compute_static_gain(self, preset: dict, selected_pops: list | None = None) -> tuple:
        """Compute static spatial gain curve G(k) = ||-J(k)^(-1) B(k)||.

        Delegates to DashboardAnalysis.

        Args:
            preset: Network preset dictionary
            selected_pops: Optional list of population IDs to analyze.

        Returns:
            (k_values, gain_values): Arrays of k and corresponding gains
        """
        return self.analysis.compute_static_gain(preset, selected_pops)

    def compute_spatiotemporal_gain(self, preset: dict, selected_pops: list | None = None) -> tuple:
        """Compute spatiotemporal amplification map A(k,omega).

        Delegates to DashboardAnalysis.

        Args:
            preset: Network preset dictionary
            selected_pops: Optional list of population IDs to analyze.

        Returns:
            (k_values, omega_values, gain_matrix): Arrays of k, omega, and gain[k,omega]
        """
        return self.analysis.compute_spatiotemporal_gain(preset, selected_pops)

    def create_stability_spectrum_figure(
        self, k_values: np.ndarray, max_real_values: np.ndarray
    ) -> go.Figure:
        """Create Plotly figure for stability spectrum.

        Delegates to dashboard_plots.

        Args:
            k_values: Array of k values (mode numbers, dimensionless)
            max_real_values: Array of max real eigenvalues for each k

        Returns:
            Plotly Figure object
        """
        anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
        n_modes = ANALYSIS_PARAMS["n_modes"]
        return create_stability_spectrum_figure(
            k_values, max_real_values, anatomical_grid_size, n_modes
        )

    def create_eigenvalue_spectrum_figure(
        self, eigenvalues: np.ndarray, k_max: float
    ) -> go.Figure:
        """Create Plotly figure for eigenvalue spectrum in the complex plane.

        Delegates to dashboard_plots.

        Args:
            eigenvalues: Complex array of eigenvalues to plot
            k_max: The k value at which these eigenvalues were computed

        Returns:
            Plotly Figure object
        """
        return create_eigenvalue_spectrum_figure(eigenvalues, k_max)

    def create_static_gain_figure(
        self, k_values: np.ndarray, gain_values: np.ndarray
    ) -> go.Figure:
        """Create Plotly figure for static spatial gain G(k).

        Delegates to dashboard_plots.

        Args:
            k_values: Array of k values (mode numbers, dimensionless)
            gain_values: Array of gain values for each k

        Returns:
            Plotly Figure object
        """
        anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
        return create_static_gain_figure(k_values, gain_values, anatomical_grid_size)

    def create_spatiotemporal_gain_figure(
        self, k_values: np.ndarray, omega_values: np.ndarray, gain_matrix: np.ndarray
    ) -> go.Figure:
        """Create Plotly figure for spatiotemporal amplification map.

        Delegates to dashboard_plots.

        Args:
            k_values: Array of spatial frequencies k
            omega_values: Array of temporal frequencies omega (Hz)
            gain_matrix: 2D array of gain values [k_idx, omega_idx]

        Returns:
            Plotly Figure object
        """
        anatomical_grid_size = ANALYSIS_PARAMS["anatomical_grid_size"]
        return create_spatiotemporal_gain_figure(
            k_values, omega_values, gain_matrix, anatomical_grid_size
        )

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

    def _create_activity_colorbar(self):
        """Create a combined vertical colorbar for SST, E, PV cell types.

        Returns:
            html.Div containing three vertical gradient bars (SST|E|PV) with
            scale labels 0 to HEATMAP_VMAX and a vertical "Firing rate (Hz)" label.
        """
        ordered_cell_types = ["SST", "E", "PV"]
        colors = {ct: CELL_COLORS[ct] for ct in ordered_cell_types}

        # Height to span all three layer rows: 3 × 155px + 2 × 24px (mb-4 between)
        COLORBAR_HEIGHT = 513

        # Create three vertical gradient bars side by side
        gradient_bars = [
            html.Div(
                style={
                    "width": "12px",
                    "flex": "1",
                    "background": f"linear-gradient(to top, white, {colors[ct]})",
                    "borderLeft": "1px solid #aaa" if i > 0 else "none",
                }
            )
            for i, ct in enumerate(ordered_cell_types)
        ]

        # Format vmax label (show as decimal if < 1, otherwise integer)
        vmax_label = f"{HEATMAP_VMAX:.1f}" if HEATMAP_VMAX < 1 else f"{int(HEATMAP_VMAX)}"

        # Scale labels to the right of the bar: 0.5 at top, 0 at bottom
        scale_labels = html.Div(
            [
                html.Div(vmax_label, style={"fontSize": "11px", "color": "#555"}),
                html.Div("0", style={"fontSize": "11px", "color": "#555"}),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "space-between",
                "height": f"{COLORBAR_HEIGHT}px",
                "paddingLeft": "4px",
            },
        )

        gradient_block = html.Div(
            gradient_bars,
            style={
                "display": "flex",
                "height": f"{COLORBAR_HEIGHT}px",
                "border": "1px solid #aaa",
            },
        )

        colorbar_block = html.Div(
            [gradient_block, scale_labels],
            style={"display": "flex", "alignItems": "stretch"},
        )

        # Vertical label to the right of the colorbar
        vertical_label = html.Div(
            "Firing rate (Hz)",
            style={
                "writingMode": "vertical-rl",
                "height": f"{COLORBAR_HEIGHT}px",
                "fontSize": "12px",
                "color": "#555",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "marginLeft": "10px",
                "whiteSpace": "nowrap",
            },
        )

        return html.Div(
            [colorbar_block, vertical_label],
            style={
                "display": "flex",
                "alignItems": "center",
                "marginLeft": "0px",
            },
        )

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
                # Layer visualizations with colorbar on the right
                html.Div(
                    [
                        # Layer rows container
                        html.Div(
                            [self.create_layer_row(layer) for layer in LAYERS],
                            style={"flex": "1"},
                        ),
                        # Combined colorbar for SST/E/PV
                        self._create_activity_colorbar(),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "flex-start",
                    },
                ),
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
        return parse_connection_key(conn_key)

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

        # Note: Thalamic burst statistics are controlled by alpha parameter
        # passed to simulation.update(), not by preset parameters

    def get_connection_key(self, source_layer, source_cell, target_layer, target_cell):
        """Generate a connection key based on source and target information."""
        return build_connection_key(source_layer, source_cell, target_layer, target_cell)

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
        """Get the current connection strength value from live simulation.

        Args:
            source_layer: Source layer
            source_cell: Source cell type
            target_layer: Target layer
            target_cell: Target cell type
            scaled: If True, return strength-scaled value; if False, return raw amplitude

        Returns:
            Connection strength (raw or scaled)
        """
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

    def setup_callbacks(self):
        """Set up the dashboard callbacks for interactivity.

        Dispatches to dedicated callback modules for better organization:
        - dashboard_callbacks_core: interval updates, pause/play
        - dashboard_callbacks_presets: P0/P5/P10/P15 preset buttons
        - dashboard_callbacks_connectivity: connection matrix interactions
        - dashboard_callbacks_analysis: stability/gain spectra, population selection
        """
        dashboard_callbacks_core.register_callbacks(self.app, self)
        dashboard_callbacks_presets.register_callbacks(self.app, self)
        dashboard_callbacks_connectivity.register_callbacks(self.app, self)
        dashboard_callbacks_analysis.register_callbacks(self.app, self)

    def create_control_panel(self):
        """Create the control panel with all sliders and controls."""
        return create_control_panel()

    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard application."""
        self.app.run_server(
            debug=debug, port=port, threaded=True, dev_tools_silence_routes_logging=True
        )
