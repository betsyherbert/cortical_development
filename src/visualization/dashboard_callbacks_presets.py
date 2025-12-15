"""Preset button callbacks for developmental stages (P0/P5/P10/P15).

This module handles the preset button interactions and slider synchronization.
"""

from dash import Input, Output

from src.model.presets import P0_PRESET, P5_PRESET, P10_PRESET, P15_PRESET


def register_callbacks(app, dashboard):
    """Register preset button callbacks.

    Args:
        app: Dash application instance
        dashboard: DashboardApp instance containing simulation and preset methods
    """
    # Register all preset callbacks
    _create_preset_callback(app, dashboard, "P0", P0_PRESET)
    _create_preset_callback(app, dashboard, "P5", P5_PRESET, allow_duplicate=True)
    _create_preset_callback(app, dashboard, "P10", P10_PRESET, allow_duplicate=True)
    _create_preset_callback(app, dashboard, "P15", P15_PRESET, allow_duplicate=True)


def _create_preset_callback(app, dashboard, preset_name, preset_obj, allow_duplicate=False):
    """Helper function to create a preset callback.

    Args:
        app: Dash application instance
        dashboard: DashboardApp instance
        preset_name: Name of the preset (e.g., 'P0', 'P5')
        preset_obj: Preset dictionary with configuration values
        allow_duplicate: Whether to allow duplicate outputs
    """
    outputs = (
        [
            Output(id, prop, allow_duplicate=allow_duplicate)
            for id, prop in [
                (o.component_id, o.component_property) for o in dashboard._PRESET_OUTPUTS
            ]
        ]
        if allow_duplicate
        else dashboard._PRESET_OUTPUTS
    )

    @app.callback(
        outputs, Input(f"{preset_name}-preset-button", "n_clicks"), prevent_initial_call=True
    )
    def apply_preset_callback(n_clicks):  # pylint: disable=unused-argument
        """Apply the preset configuration."""
        # Apply the preset using the dashboard's apply_preset method
        dashboard._apply_preset(preset_obj)

        # Get values from the preset
        values = dashboard._get_preset_values(preset_obj)

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
            dashboard.create_connection_matrix(),
        )

    return apply_preset_callback

