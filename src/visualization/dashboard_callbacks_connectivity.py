"""Connectivity callbacks for connection matrix interactions.

This module handles:
- Connection cell clicks and slider popup
- Connection strength updates
- Connectivity parameter updates (sigma widths, strength scaling, background input)
"""

import logging

from dash import Input, Output, State
from dash.dependencies import ALL, MATCH

from src.model.config import CELL_TYPES, CONNECTIONS, LAYERS
from src.visualization.dashboard_layout import CELL_STYLE
from src.visualization.dashboard_utils import (
    SLIDER_HIDDEN_STYLE,
    SLIDER_POPUP_STYLE,
    get_triggered_id,
    get_triggered_value,
    is_valid_click,
    no_update_tuple,
    parse_connection_cell_id,
    parse_pattern_match_id,
)

logger = logging.getLogger(__name__)

# Fallback strength scaling (used when simulation not available)
INITIAL_STRENGTH_SCALING = {"E": 1.0, "SST": 1.0, "PV": 1.0, "thalamus": 1.0}


def register_callbacks(app, dashboard):
    """Register connectivity-related callbacks.

    Args:
        app: Dash application instance
        dashboard: DashboardApp instance containing simulation and connection methods
    """
    # Handle cell clicks to show the slider
    @app.callback(
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
            logger.warning("Invalid cell ID format: %s", clicked_id)
            return no_update_tuple(3)

        source_layer, source_cell, target_layer, target_cell = parsed

        # Get current connection value and create slider
        value = dashboard.get_connection_value(source_layer, source_cell, target_layer, target_cell)
        slider = dashboard.create_slider_for_cell(
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
    @app.callback(
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
            dashboard.simulation.connectivity.set_connection_strength(
                source_layer, source_cell, target_layer, target_cell, value
            )

            return f"Value: {value:.1f}"
        except (KeyError, AttributeError, ValueError):
            logger.exception("Error updating connection value")
            return "Error"

    # Update connection cell in matrix when slider changes
    @app.callback(
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
            if hasattr(dashboard, "simulation") and hasattr(dashboard.simulation, "connectivity"):
                scaled_value = dashboard.simulation.connectivity.get_scaled_connection_strength(
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
            max_magnitude = dashboard.get_max_scaled_strength_magnitude()

            # Determine cell colors based on scaled connection strength and source cell type
            bg_color, hover_color = dashboard._get_connection_colors(
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
        except (KeyError, ValueError):
            logger.exception("Error updating matrix cell")
            return no_update_tuple(3)

    # Reset the slider when clicking the reset button
    @app.callback(
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
    app.clientside_callback(
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

    # Update connectivity widths in simulation
    @app.callback(
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
                dashboard.simulation.set_connection_sigma("thalamus", None, layer, cell_type, sigma)

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
                            dashboard.simulation.set_connection_sigma(
                                source_layer, source_cell, target_layer, target_cell, sigma
                            )

        # Return unchanged intervals to not disrupt the update loop
        return [n_intervals]

    # Update strength scaling factors
    @app.callback(
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
        dashboard.simulation.set_strength_scaling("E", e_scaling)
        dashboard.simulation.set_strength_scaling("SST", sst_scaling)
        dashboard.simulation.set_strength_scaling("PV", pv_scaling)
        dashboard.simulation.set_strength_scaling("thalamus", thalamus_scaling)

        # Regenerate the connection matrix with updated scaled values
        updated_matrix = dashboard.create_connection_matrix()

        # Return unchanged intervals and updated matrix
        return [n_intervals, updated_matrix]

    # Update background input parameters
    @app.callback(
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
        dashboard.simulation.set_background_input("E", bg_e)
        dashboard.simulation.set_background_input("SST", bg_sst)
        dashboard.simulation.set_background_input("PV", bg_pv)

        # Return unchanged intervals to not disrupt the update loop
        return [n_intervals]

