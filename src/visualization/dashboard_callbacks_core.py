"""Core dashboard callbacks: interval updates, pause/play, and figure refresh.

This module handles the main simulation loop and activity visualization updates.
"""

from dash import Input, Output, State

from src.model.config import CELL_TYPES, CONNECTIONS, LAYERS, UPDATE_INTERVAL
from src.visualization.dashboard_config import (
    CORRELATION_HISTORY_LENGTH,
    CORRELATION_UPDATE_INTERVAL,
)
from src.visualization.dashboard_plots import HEATMAP_ZMAX, HEATMAP_ZMIN
from src.visualization.dashboard_utils import SLIDER_HIDDEN_STYLE


def register_callbacks(app, dashboard):
    """Register core callbacks for simulation updates and pause/play.

    Args:
        app: Dash application instance
        dashboard: DashboardApp instance containing simulation and figures
    """
    # Initialize slider container (hidden)
    @app.callback(
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

    # Update the graphs with neural activity
    @app.callback(
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
        # Input: only interval trigger
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
                dashboard.figures[f"graph-{layer}-{cell_type}"]
                for layer in LAYERS
                for cell_type in CELL_TYPES
            ] + [
                dashboard.figures["graph-thalamus"],
                dashboard.figures["correlation-by-layer"],
                dashboard.figures["correlation-by-celltype"],
                dashboard.figures["events-by-layer"],
                dashboard.figures["events-by-celltype"],
            ]

        # Update neuron parameters
        dashboard.simulation.set_time_constant("E", tau_e)
        dashboard.simulation.set_time_constant("SST", tau_sst)
        dashboard.simulation.set_time_constant("PV", tau_pv)

        # Update all connectivity widths
        for layer in LAYERS:
            # Update thalamic inputs
            dashboard.simulation.set_connection_sigma(
                "thalamus", None, layer, "E", sigma_thal_e
            )
            dashboard.simulation.set_connection_sigma(
                "thalamus", None, layer, "SST", sigma_thal_sst
            )
            dashboard.simulation.set_connection_sigma(
                "thalamus", None, layer, "PV", sigma_thal_pv
            )

            # Update outgoing connections for each source layer
            for source_layer in LAYERS:
                for source_cell, sigma in [
                    ("E", sigma_e_out),
                    ("SST", sigma_sst_out),
                    ("PV", sigma_pv_out),
                ]:
                    for target_cell in CELL_TYPES:
                        if (source_cell, target_cell) in CONNECTIONS:
                            dashboard.simulation.set_connection_sigma(
                                source_layer, source_cell, layer, target_cell, sigma
                            )

        # Update simulation state
        activities = dashboard.simulation.update(alpha=alpha)

        # Update all figures
        updated_figures = []

        # Update each layer-cell type figure
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                fig_id = f"graph-{layer}-{cell_type}"
                fig = dashboard.figures[fig_id]

                # Update the figure data
                with fig.batch_update():
                    # Get the layer's activity for this cell type
                    data = activities[layer][cell_type].reshape(
                        dashboard.simulation.grid_size, dashboard.simulation.grid_size
                    )
                    fig.data[0]["z"] = data

                    # Keep consistent scaling for fair comparison across heatmaps
                    fig.update_traces(zmin=HEATMAP_ZMIN, zmax=HEATMAP_ZMAX)

                updated_figures.append(fig)

        # Update thalamus figure
        thal_fig = dashboard.figures["graph-thalamus"]
        with thal_fig.batch_update():
            thal_data = activities["thalamus"]
            thal_fig.data[0]["z"] = thal_data
            thal_fig.update_traces(zmin=HEATMAP_ZMIN, zmax=HEATMAP_ZMAX)

        updated_figures.append(thal_fig)

        # Update correlation tracking
        dashboard.simulation_time += UPDATE_INTERVAL / 1000.0
        dashboard.correlation_activity_buffer.append(activities)
        # Keep buffer size limited to prevent memory issues
        max_buffer_size = CORRELATION_HISTORY_LENGTH + 10
        if len(dashboard.correlation_activity_buffer) > max_buffer_size:
            dashboard.correlation_activity_buffer = dashboard.correlation_activity_buffer[
                -max_buffer_size:
            ]

        # Compute correlations and events every N updates for efficiency
        if n_intervals % CORRELATION_UPDATE_INTERVAL == 0:
            dashboard._update_time_series_data(
                dashboard._compute_rolling_correlations(), dashboard.correlation_time_series
            )
            dashboard._update_time_series_data(
                dashboard._compute_synchronous_events(), dashboard.event_time_series
            )

        # Update correlation figures (always, for smooth rendering)
        corr_fig_layer, corr_fig_celltype = dashboard._update_correlation_figures()
        updated_figures.extend([corr_fig_layer, corr_fig_celltype])

        # Update event figures
        event_fig_layer, event_fig_celltype = dashboard._update_event_figures()
        updated_figures.extend([event_fig_layer, event_fig_celltype])

        return updated_figures

    # Toggle simulation pause state
    @app.callback(
        Output("interval-component", "disabled"), [Input("pause-button", "n_clicks")]
    )
    def toggle_simulation(n_clicks):
        """Toggle simulation pause state based on button clicks."""
        return n_clicks is not None and n_clicks % 2 == 1

