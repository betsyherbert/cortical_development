"""Layout components and styling constants for the cortical simulation dashboard.

This module contains:
- Style constants (sizes, colors, common CSS dicts)
- Pure layout builder functions (no simulation state dependencies)
- Slider configuration parameters

The DashboardApp class imports from here to keep layout concerns separated.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.model.config import (
    ANATOMICAL_GRID_SIZE,
    CELL_TYPES,
    GRID_SIZE,
    INITIAL_BACKGROUND_INPUT,
    INITIAL_OUTGOING_WIDTHS,
    INITIAL_STRENGTH_SCALING,
    INITIAL_THALAMIC_WIDTHS,
    INITIAL_TIME_CONSTANTS,
    THALAMIC_ALPHA,
)

# =============================================================================
# Size Constants
# =============================================================================
CELL_SIZE = 40  # Size for data cells in pixels
HEADER_HEIGHT = 40  # Height for headers in pixels
HEADER_WIDTH = 40  # Width for all header columns

# Font size constants for consistent aesthetics
TITLE_FONT_SIZE = 17  # Main section titles (H5 elements)
SUBTITLE_FONT_SIZE = 14  # Plot titles and subtitles
AXIS_FONT_SIZE = 13  # Axis titles, tick labels, and legend text

# =============================================================================
# Style Dictionaries
# =============================================================================

HEADER_STYLE = {
    "border": "none",
    "width": f"{HEADER_WIDTH}px",
    "height": f"{HEADER_HEIGHT}px",
    "minWidth": f"{HEADER_WIDTH}px",
    "maxWidth": f"{HEADER_WIDTH}px",
}

CELL_STYLE = {
    "width": f"{CELL_SIZE}px",
    "height": f"{CELL_SIZE}px",
    "minWidth": f"{CELL_SIZE}px",
    "maxWidth": f"{CELL_SIZE}px",
    "minHeight": f"{CELL_SIZE}px",
    "maxHeight": f"{CELL_SIZE}px",
}

# Colors for light mode
LAYER_COLORS = {
    "L4": "rgba(52, 73, 94, 0.15)",
    "default": "rgba(149, 165, 166, 0.15)",
    "transparent": "transparent",
}

MAIN_HEADER_STYLE = {
    **HEADER_STYLE,
    "backgroundColor": LAYER_COLORS["default"],
    "color": "#2c3e50",
    "padding": "10px 5px",
    "fontSize": "0.9rem",
    "fontWeight": "600",
}

LAYER_HEADER_STYLE = {
    **MAIN_HEADER_STYLE,
    "backgroundColor": LAYER_COLORS["L4"],
}

CELL_TYPE_HEADER_STYLE = {
    **HEADER_STYLE,
    "color": "#2c3e50",
    "padding": "8px 5px",
    "fontSize": "0.9rem",
    "fontWeight": "500",
}

ROW_HEADER_STYLE = {
    **HEADER_STYLE,
    "color": "#2c3e50",
    "textAlign": "center",
    "verticalAlign": "middle",
    "padding": "10px 5px",
    "height": "100%",
    "fontSize": "0.9rem",
    "fontWeight": "600",
}

CONTROL_PANEL_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "10px",
    "padding": "15px",
    "border": "1px solid #ddd",
}

SLIDER_CONTAINER_STYLE = {
    "backgroundColor": "rgba(255, 255, 255, 0.95)",
    "padding": "10px",
    "border": "1px solid #ccc",
    "borderRadius": "5px",
    "zIndex": "1000",
    "width": "200px",
    "position": "absolute",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
}

# Graph configuration
GRAPH_CONFIG = {"displayModeBar": False}

GRAPH_LAYOUT = {
    "margin": dict(l=0, r=0, t=0, b=0),
    "height": 150,
    "width": 150,
    "dragmode": False,
    "clickmode": "event",
    "hovermode": False,
    "xaxis": dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        scaleanchor="y",
        scaleratio=1,
        fixedrange=True,
    ),
    "yaxis": dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True,
    ),
}

# =============================================================================
# Slider Configuration Parameters
# =============================================================================

SLIDER_STYLE = {
    "tooltip": {"placement": "bottom", "always_visible": False},
    "className": "custom-slider",
}

TIME_CONSTANT_PARAMS = {
    "min_val": 1.0,
    "max_val": 60.0,
    "step": 1.0,
    "marks": {1: "1", 20: "20", 40: "40", 60: "60"},
}

GAIN_PARAMS = {
    "min_val": 0.4,
    "max_val": 1.0,
    "step": 0.1,
    "marks": {0.4: "0.4", 0.6: "0.6", 0.8: "0.8", 1.0: "1"},
}

WIDTH_PARAMS = {
    "min_val": 5.0,
    "max_val": 400.0,
    "step": 5.0,
    "marks": {i: f"{i}" for i in range(0, 401, 100)},
}

STRENGTH_SCALING_PARAMS = {
    "min_val": 0.0,
    "max_val": 10.0,
    "step": 0.1,
    "marks": {i: f"{i}" for i in range(0, 11, 2)},
}

BACKGROUND_INPUT_PARAMS = {
    "min_val": 0.0,
    "max_val": 0.4,
    "step": 0.05,
    "marks": {0.0: "0.0", 0.2: "0.2", 0.4: "0.4"},
}


# =============================================================================
# Pure Layout Builder Functions
# =============================================================================


def create_grid_info_boxes() -> html.Div:
    """Create info boxes showing grid parameters.

    Returns:
        Dash Div component with grid info boxes
    """
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(width=2),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    # Anatomical grid size box
                                    html.Div(
                                        [
                                            html.Div(
                                                "Anatomical Grid Size",
                                                style={
                                                    "fontSize": "11px",
                                                    "fontWeight": "600",
                                                    "color": "#34495e",
                                                },
                                            ),
                                            html.Div(
                                                f"{ANATOMICAL_GRID_SIZE:.0f} × {ANATOMICAL_GRID_SIZE:.0f} μm",
                                                style={
                                                    "fontSize": "13px",
                                                    "fontWeight": "bold",
                                                    "color": "#2c3e50",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "inline-block",
                                            "padding": "8px 16px",
                                            "backgroundColor": "#ecf0f1",
                                            "borderRadius": "4px",
                                            "marginRight": "15px",
                                            "textAlign": "center",
                                        },
                                    ),
                                    # Neurons per grid box
                                    html.Div(
                                        [
                                            html.Div(
                                                "Neurons Per Grid",
                                                style={
                                                    "fontSize": "11px",
                                                    "fontWeight": "600",
                                                    "color": "#34495e",
                                                },
                                            ),
                                            html.Div(
                                                f"{GRID_SIZE} × {GRID_SIZE} = {GRID_SIZE * GRID_SIZE}",
                                                style={
                                                    "fontSize": "13px",
                                                    "fontWeight": "bold",
                                                    "color": "#2c3e50",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "inline-block",
                                            "padding": "8px 16px",
                                            "backgroundColor": "#ecf0f1",
                                            "borderRadius": "4px",
                                            "textAlign": "center",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "flex-start",
                                    "marginLeft": "20px",
                                },
                            )
                        ],
                        width=10,
                    ),
                ]
            )
        ],
        className="mb-3",
    )


def create_preset_buttons() -> html.Div:
    """Create the developmental stage preset buttons row.

    Returns:
        Dash Div component with P0/P5/P10/P15 buttons
    """
    button_style = {
        "backgroundColor": "#2c3e50",
        "borderColor": "#2c3e50",
    }

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(width=2),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Button(
                                        stage,
                                        id=f"{stage}-preset-button",
                                        color="dark",
                                        className="mx-2 px-3",
                                        style=button_style,
                                    )
                                    for stage in ["P0", "P5", "P10", "P15"]
                                ],
                                style={"display": "flex", "justifyContent": "center"},
                            )
                        ],
                        width=10,
                    ),
                ]
            )
        ],
        className="mb-3",
    )


def create_slider(
    id_prefix: str,
    cell_type: str,
    min_val: float,
    max_val: float,
    step: float,
    initial_value: float,
    marks: dict,
) -> dcc.Slider:
    """Create a slider with consistent styling.

    Args:
        id_prefix: Prefix for the slider ID (e.g., 'tau', 'background-input')
        cell_type: Cell type for the slider (e.g., 'E', 'SST', 'PV')
        min_val: Minimum slider value
        max_val: Maximum slider value
        step: Step size
        initial_value: Initial slider value
        marks: Dictionary of mark positions to labels

    Returns:
        Dash Slider component
    """
    return dcc.Slider(
        id=f"{id_prefix}-{cell_type.lower()}-slider",
        min=min_val,
        max=max_val,
        step=step,
        value=initial_value,
        marks=marks,
        **SLIDER_STYLE,
    )


def create_parameter_row(cell_type: str) -> dbc.Row:
    """Create a row of sliders for a cell type's parameters.

    Args:
        cell_type: Cell type ('E', 'SST', or 'PV')

    Returns:
        Dash Row component with time constant and background input sliders
    """
    return dbc.Row(
        [
            dbc.Col(
                html.Strong(cell_type),
                width=1,
                className="d-flex align-items-center",
                style={"paddingRight": "5px"},
            ),
            dbc.Col(
                create_slider(
                    id_prefix="tau",
                    cell_type=cell_type,
                    initial_value=INITIAL_TIME_CONSTANTS[cell_type],
                    **TIME_CONSTANT_PARAMS,
                ),
                width=5,
                style={"paddingRight": "5px"},
            ),
            dbc.Col(
                create_slider(
                    id_prefix="background-input",
                    cell_type=cell_type,
                    initial_value=INITIAL_BACKGROUND_INPUT[cell_type],
                    **BACKGROUND_INPUT_PARAMS,
                ),
                width=5,
            ),
        ],
        className="mb-1",
    )


def create_parameter_sliders() -> html.Div:
    """Create the neural parameter sliders section.

    Returns:
        Dash Div with time constant and background input sliders for all cell types
    """
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col("", width=1),
                    dbc.Col(html.Div("Time Constant (ms)", className="text-center"), width=5),
                    dbc.Col(html.Div("Background Input", className="text-center"), width=5),
                ],
                className="mb-1",
            ),
            *[create_parameter_row(cell_type) for cell_type in CELL_TYPES],
        ]
    )


def create_connectivity_row(cell_type: str) -> dbc.Row:
    """Create a row of sliders for a cell type's connectivity parameters.

    Args:
        cell_type: Cell type ('E', 'SST', or 'PV')

    Returns:
        Dash Row component with thalamic and outgoing width sliders
    """
    return dbc.Row(
        [
            dbc.Col(
                html.Strong(cell_type),
                width=1,
                className="d-flex align-items-center",
                style={"paddingRight": "5px"},
            ),
            dbc.Col(
                create_slider(
                    id_prefix="thalamic-width",
                    cell_type=cell_type,
                    initial_value=INITIAL_THALAMIC_WIDTHS[cell_type],
                    **WIDTH_PARAMS,
                ),
                width=5,
                style={"paddingRight": "5px"},
            ),
            dbc.Col(
                create_slider(
                    id_prefix="outgoing-width",
                    cell_type=cell_type,
                    initial_value=INITIAL_OUTGOING_WIDTHS[cell_type],
                    **WIDTH_PARAMS,
                ),
                width=5,
            ),
        ],
        className="mb-1",
    )


def create_connectivity_sliders() -> html.Div:
    """Create the connectivity width sliders section.

    Returns:
        Dash Div with thalamic and outgoing width sliders for all cell types
    """
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col("", width=1),
                    dbc.Col(html.Div("Thalamic", className="text-center"), width=5),
                    dbc.Col(html.Div("Outgoing", className="text-center"), width=5),
                ],
                className="mb-1",
            ),
            *[create_connectivity_row(cell_type) for cell_type in CELL_TYPES],
        ]
    )


def create_strength_scaling_row(cell_type: str) -> dbc.Row:
    """Create a row for a cell type's strength scaling parameter.

    Args:
        cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')

    Returns:
        Dash Row component with strength scaling slider
    """
    label = cell_type if cell_type != "thalamus" else "TC"
    return dbc.Row(
        [
            dbc.Col(
                html.Strong(label),
                width=1,
                className="d-flex align-items-center",
                style={"paddingRight": "5px"},
            ),
            dbc.Col(
                create_slider(
                    id_prefix="strength-scaling",
                    cell_type=cell_type.lower(),
                    initial_value=INITIAL_STRENGTH_SCALING[cell_type],
                    **STRENGTH_SCALING_PARAMS,
                ),
                width=11,
            ),
        ],
        className="mb-1",
    )


def create_strength_scaling_sliders() -> html.Div:
    """Create the connection strength scaling sliders section.

    Returns:
        Dash Div with strength scaling sliders for all cell types and thalamus
    """
    return html.Div(
        [
            *[create_strength_scaling_row(cell_type) for cell_type in CELL_TYPES],
            create_strength_scaling_row("thalamus"),
        ]
    )


def create_input_controls() -> dbc.Row:
    """Create the input control sliders for intrinsic/sensory balance.

    Returns:
        Dash Row with alpha slider for thalamic input balance
    """
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col("Intrinsic", className="text-start", width=6),
                            dbc.Col("Sensory", className="text-end", width=6),
                        ],
                        className="mb-2",
                    ),
                    dcc.Slider(
                        id="alpha-slider",
                        min=0,
                        max=1,
                        step=0.1,
                        value=THALAMIC_ALPHA,
                        marks={i / 10: f"{i/10:.1f}" for i in range(11)},
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="custom-slider",
                    ),
                ]
            )
        ]
    )


def create_control_panel() -> html.Div:
    """Create the control panel with all sliders and controls.

    Returns:
        Dash Div with the complete control panel layout
    """
    return html.Div(
        [
            # Section: Neuron parameters
            html.Div(
                [
                    html.H5(
                        "Neuron Parameters",
                        className="text-center",
                        style={"fontSize": f"{TITLE_FONT_SIZE}px", "fontWeight": "600"},
                    ),
                    create_parameter_sliders(),
                ],
                className="mb-3",
            ),
            # Section: Connectivity widths
            html.Div(
                [
                    html.H5(
                        "Connection Widths (μm)",
                        className="text-center",
                        style={"fontSize": f"{TITLE_FONT_SIZE}px", "fontWeight": "600"},
                    ),
                    create_connectivity_sliders(),
                    html.Div([html.Hr()], className="my-3"),
                    html.H5(
                        "Strength Scaling",
                        className="text-center",
                        style={"fontSize": f"{TITLE_FONT_SIZE}px", "fontWeight": "600"},
                    ),
                    create_strength_scaling_sliders(),
                    html.Div([html.Hr()], className="my-3"),
                    html.H5(
                        "Thalamic Input",
                        className="text-center",
                        style={"fontSize": f"{TITLE_FONT_SIZE}px", "fontWeight": "600"},
                    ),
                    create_input_controls(),
                ],
                className="mb-3",
            ),
            # Pause/Play control
            html.Div(
                [dbc.Button("Pause", id="pause-button", color="secondary", className="me-md-2")],
                className="mt-4",
            ),
        ]
    )

