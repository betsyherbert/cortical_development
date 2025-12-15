"""Analysis callbacks for stability/gain spectra and population selection.

This module handles:
- Stability spectrum and eigenvalue updates
- Static and spatiotemporal gain updates
- Population selection via heatmap clicks
"""

import logging

import numpy as np
from dash import Input, Output, State
from dash.dependencies import ALL

from src.visualization.dashboard_plots import create_empty_message_figure
from src.visualization.dashboard_utils import (
    format_analysis_display,
    get_triggered_id,
    no_update_tuple,
    parse_pattern_match_id,
)

logger = logging.getLogger(__name__)


def register_callbacks(app, dashboard):
    """Register analysis-related callbacks.

    Args:
        app: Dash application instance
        dashboard: DashboardApp instance containing analysis methods and figures
    """
    # Update stability spectrum and eigenvalue spectrum periodically
    @app.callback(
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
        if dashboard._computing_stability:
            return no_update_tuple(2)

        try:
            dashboard._computing_stability = True

            # Check if any populations are selected
            if not selected_pops or len(selected_pops) == 0:
                fig = create_empty_message_figure("Select populations by clicking heatmaps")
                return fig, fig

            # Build preset from current simulation state
            preset = dashboard.build_current_preset()

            # Compute stability spectrum for selected populations
            k_values, max_real_values, eigenvalues_at_max_k, k_max = (
                dashboard.compute_stability_spectrum(preset, selected_pops)
            )

            # Create both figures
            stability_fig = dashboard.create_stability_spectrum_figure(k_values, max_real_values)
            eigenvalue_fig = dashboard.create_eigenvalue_spectrum_figure(eigenvalues_at_max_k, k_max)

            return stability_fig, eigenvalue_fig

        except Exception:
            logger.exception("Error updating stability spectrum")
            # Return empty figures on error
            empty_stability = dashboard.create_stability_spectrum_figure(np.array([]), np.array([]))
            empty_eigenvalue = dashboard.create_eigenvalue_spectrum_figure(np.array([]), 0.0)
            return empty_stability, empty_eigenvalue
        finally:
            dashboard._computing_stability = False

    # Update forced response graphs periodically
    @app.callback(
        [Output("static-gain-graph", "figure"), Output("spatiotemporal-gain-graph", "figure")],
        [Input("spectrum-interval", "n_intervals"), Input("selected-populations", "data")],
    )
    def update_forced_response(n_intervals, selected_pops):  # pylint: disable=unused-argument
        """Update the forced response graphs periodically."""
        # Prevent concurrent computation
        if dashboard._computing_forced_response:
            return no_update_tuple(2)

        try:
            dashboard._computing_forced_response = True

            # Check if any populations are selected
            if not selected_pops or len(selected_pops) == 0:
                fig = create_empty_message_figure("Select populations by clicking heatmaps")
                return fig, fig

            # Build preset from current simulation state
            preset = dashboard.build_current_preset()

            # Compute static gain for selected populations
            k_values_static, gain_values = dashboard.compute_static_gain(preset, selected_pops)
            static_fig = dashboard.create_static_gain_figure(k_values_static, gain_values)

            # Compute spatiotemporal gain for selected populations
            k_values_st, omega_values, gain_matrix = dashboard.compute_spatiotemporal_gain(
                preset, selected_pops
            )
            spatiotemporal_fig = dashboard.create_spatiotemporal_gain_figure(
                k_values_st, omega_values, gain_matrix
            )

            return static_fig, spatiotemporal_fig

        except Exception:
            logger.exception("Error updating forced response")
            # Return empty figures on error
            empty_static = dashboard.create_static_gain_figure(np.array([]), np.array([]))
            empty_st = dashboard.create_spatiotemporal_gain_figure(
                np.array([]), np.array([]), np.array([])
            )
            return empty_static, empty_st
        finally:
            dashboard._computing_forced_response = False

    # Handle heatmap clicks for population selection using n_clicks on containers
    @app.callback(
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
    @app.callback(
        Output("selected-populations-display", "children"),
        [Input("selected-populations", "data")],
    )
    def update_selected_populations_display(selected_pops):
        """Update the display showing which populations are selected for analysis."""
        return format_analysis_display(selected_pops)

