"""Dashboard module for visualizing the cortical circuit simulation."""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from typing import Dict, List, Tuple
import json

from model.config import (
    COLORMAPS, UPDATE_INTERVAL, CELL_TYPES, LAYERS, LAYER_NAMES, 
    THALAMIC_SCALING, LAYER_CONNECTIONS,
    LAYER_CONNECTIVITY_PARAMS, THALAMIC_ALPHA, CONNECTIONS,
    GRID_SIZE, INITIAL_THALAMIC_WIDTHS, INITIAL_OUTGOING_WIDTHS,
    INITIAL_TIME_CONSTANTS, INITIAL_FIRING_THRESHOLDS
)
from model.neurons import FIRING_THRESHOLD
from model.presets import P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET


class DashboardApp:
    """
    Dashboard application for visualizing and controlling the neural simulation.
    
    This class creates an interactive Dash application that displays real-time
    neural activity and provides controls for adjusting simulation parameters.
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
        
        # Initialize the Dash app with dark theme
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
        
        # Add custom CSS for sliders
        self.app.index_string = '''
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
                    .custom-slider .rc-slider-track {
                        background-color: white !important;
                    }
                    .custom-slider .rc-slider-rail {
                        background-color: #555 !important;
                    }
                    .custom-slider .rc-slider-handle {
                        border-color: white !important;
                        background-color: white !important;
                    }
                    .custom-slider .rc-slider-handle:hover {
                        border-color: white !important;
                    }
                    .custom-slider .rc-slider-handle:active {
                        border-color: white !important;
                        box-shadow: 0 0 5px white !important;
                    }
                    .custom-slider .rc-slider-dot {
                        border-color: #888 !important;
                        background-color: #888 !important;
                    }
                    .custom-slider .rc-slider-dot-active {
                        border-color: white !important;
                        background-color: white !important;
                    }
                    /* Make the control panel take less width */
                    .control-panel-column {
                        padding-left: 1rem !important;
                        padding-right: 1rem !important;
                        margin-left: -2rem !important; 
                        max-width: 600px !important;
                    }
                    /* Reduce left margin in left column labels */
                    .cell-type-label {
                        padding-right: 1rem !important;
                    }
                    /* Remove rightmost and bottommost borders */
                    .connection-matrix tr:last-child td, 
                    .connection-matrix tr:last-child th {
                        border-bottom: none !important;
                    }
                    .connection-matrix tr td:last-child, 
                    .connection-matrix tr th:last-child {
                        border-right: none !important;
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
        '''
        
        # Pre-create all figures for better performance
        self.figures = {}
        self._initialize_figures()
        
        # Set up the layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def _initialize_figures(self):
        """Pre-create all heatmap figures for better performance."""
        # Initialize with zeros
        empty_data = np.zeros((self.simulation.grid_size, self.simulation.grid_size))
        
        # Create figures for all cell types in all layers
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                fig_id = f'graph-{layer}-{cell_type}'
                self.figures[fig_id] = self.create_heatmap(empty_data, cell_type)
        
        # Create thalamus figure
        self.figures['graph-thalamus'] = self.create_heatmap(empty_data, 'thalamus')
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        # Add interval component for updates
        interval = dcc.Interval(
            id='interval-component',
            interval=self.update_interval,
            n_intervals=0,
            disabled=False
        )
        
        self.app.layout = dbc.Container([
            # Interval component for updates
            interval,
            
            # Store component for currently selected cell
            dcc.Store(id='selected-cell', data=None),
            
            # Hidden button for resetting slider state
            html.Button(
                id='reset-slider-state-btn',
                style={'display': 'none'},
                n_clicks=0
            ),
            
            # Main content: visualization and controls
            dbc.Row([
                # Left column: activity visualization
                dbc.Col([
                    # Add more top padding to shift visualization down
                    html.Div(style={"height": "40px"}),
                    
                    # One row per layer (L2/3, L4, L5)
                    *[self.create_layer_row(layer) for layer in LAYERS],
                    
                    # Thalamus visualization
                    html.Div(
                        dbc.Row([
                            # Thalamus label
                            dbc.Col([
                                html.H4("Thalamus", 
                                       style={"position": "relative", "top": "50%", "transform": "translateY(-50%)", 
                                             "textAlign": "right", "paddingRight": "20px", "margin": "0"})
                            ], width=2),
                            
                            # Container for centered thalamus
                            dbc.Col([
                                html.Div([
                                    # Empty space to match SST position
                                    html.Div(style={"width": "180px", "display": "inline-block"}),
                                    # Thalamus heatmap (aligned with E)
                                    html.Div(
                                        dcc.Graph(
                                            id='graph-thalamus',
                                            figure=self.figures['graph-thalamus'],
                                            config={'displayModeBar': False}
                                        ),
                                        style={"display": "inline-block"}
                                    ),
                                    # Empty space to match PV position
                                    html.Div(style={"width": "180px", "display": "inline-block"})
                                ], style={
                                    "display": "flex",
                                    "justifyContent": "center",
                                    "gap": "20px",  # Match the gap from layer rows
                                    "width": "100%"
                                })
                            ], width=10)
                        ], style={"height": "180px"}),
                        className="mt-5"  # Match vertical spacing with other layers
                    )
                ], width=7, style={"paddingLeft": "0rem", "paddingRight": "1rem"}),  # Adjust left column width and padding
                
                # Right column: control panel
                dbc.Col([
                    # Preset Buttons (moved here)
                    html.Div([
                        html.Div([
                            dbc.Button("P4", id="p4-preset-button", color="dark", 
                                    className="mx-2 px-3", 
                                    style={"backgroundColor": "#2c3e50", "borderColor": "#2c3e50"}),
                            dbc.Button("P8", id="p8-preset-button", color="dark", 
                                    className="mx-2 px-3", 
                                    style={"backgroundColor": "#2c3e50", "borderColor": "#2c3e50"}),
                            dbc.Button("P12", id="p12-preset-button", color="dark", 
                                     className="mx-2 px-3", 
                                     style={"backgroundColor": "#2c3e50", "borderColor": "#2c3e50"}),
                            dbc.Button("P16", id="p16-preset-button", color="dark", 
                                     className="mx-2 px-3", 
                                     style={"backgroundColor": "#2c3e50", "borderColor": "#2c3e50"})
                        ], style={"display": "flex", "justifyContent": "center"})
                    ], className="mb-4"),
                    
                    # Connection Strength Matrix
                    html.Div([
                        html.H5("Connection Strengths", 
                              className="mb-4 text-center",
                              style={"textAlign": "center", "width": "100%"}),
                        
                        # Connection Matrix Container
                        html.Div(
                            self.create_connection_matrix(),
                            id="connection-matrix-container",
                            style={"position": "relative", "display": "flex", "justifyContent": "center"}
                        ),
                        
                        # Hover Activated Slider Container (initially hidden)
                        html.Div(
                            id="slider-container",
                            style={
                                "position": "absolute", 
                                "display": "none",
                                "backgroundColor": "rgba(50, 50, 50, 0.9)",
                                "padding": "10px",
                                "border": "1px solid #444",
                                "borderRadius": "5px",
                                "zIndex": "1000",
                                "width": "200px"
                            }
                        )
                    ], className="mb-3"),
                    
                    # Time constants and firing thresholds section
                    html.Div([
                        html.Div([
                            dbc.Row([
                                # Empty header for label column
                                dbc.Col("", width=2),
                                # Header for time constants column
                                dbc.Col("Time Constants (ms)", className="text-center", width=5),
                                # Header for firing thresholds column
                                dbc.Col("Firing Thresholds", className="text-center", width=5),
                            ], className="mb-2", style={"marginLeft": "0", "marginRight": "0"}),
                            
                            # E cells row
                            dbc.Row([
                                # Cell type label
                                dbc.Col(html.Strong("E"), width=2, className="d-flex align-items-center justify-content-end"),
                                # Time constant slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='tau-e-slider',
                                        min=1.0,
                                        max=100.0,
                                        step=1.0,
                                        value=INITIAL_TIME_CONSTANTS['E'],
                                        marks={i: f"{i}" for i in range(20, 101, 20)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingRight": "15px"}
                                ),
                                # Firing threshold slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='threshold-e-slider',
                                        min=0.0,
                                        max=0.5,
                                        step=0.01,
                                        value=INITIAL_FIRING_THRESHOLDS['E'],
                                        marks={i/10: f"{i/10:.1f}" for i in range(0, 6, 1)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingLeft": "15px"}
                                ),
                            ], className="mb-3", style={"marginLeft": "0", "marginRight": "0"}),
                            
                            # SST cells row
                            dbc.Row([
                                # Cell type label
                                dbc.Col(html.Strong("SST"), width=2, className="d-flex align-items-center justify-content-end"),
                                # Time constant slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='tau-sst-slider',
                                        min=1.0,
                                        max=100.0,
                                        step=1.0,
                                        value=INITIAL_TIME_CONSTANTS['SST'],
                                        marks={i: f"{i}" for i in range(20, 101, 20)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingRight": "15px"}
                                ),
                                # Firing threshold slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='threshold-sst-slider',
                                        min=0.0,
                                        max=0.5,
                                        step=0.01,
                                        value=INITIAL_FIRING_THRESHOLDS['SST'],
                                        marks={i/10: f"{i/10:.1f}" for i in range(0, 6, 1)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingLeft": "15px"}
                                ),
                            ], className="mb-3", style={"marginLeft": "0", "marginRight": "0"}),
                            
                            # PV cells row
                            dbc.Row([
                                # Cell type label
                                dbc.Col(html.Strong("PV"), width=2, className="d-flex align-items-center justify-content-end"),
                                # Time constant slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='tau-pv-slider',
                                        min=1.0,
                                        max=100.0,
                                        step=1.0,
                                        value=INITIAL_TIME_CONSTANTS['PV'],
                                        marks={i: f"{i}" for i in range(20, 101, 20)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingRight": "15px"}
                                ),
                                # Firing threshold slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='threshold-pv-slider',
                                        min=0.0,
                                        max=0.5,
                                        step=0.01,
                                        value=INITIAL_FIRING_THRESHOLDS['PV'],
                                        marks={i/10: f"{i/10:.1f}" for i in range(0, 6, 1)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5, 
                                    style={"paddingLeft": "15px"}
                                ),
                            ], style={"marginLeft": "0", "marginRight": "0"}),
                        ]),
                    ], className="mb-4"),
                    
                    # Connectivity width section
                    html.Div([
                        html.Div([
                            dbc.Row([
                                # Empty header for label column
                                dbc.Col("", width=2),
                                # Header for thalamic connections column
                                dbc.Col("Thalamic Input Width", className="text-center", width=5),
                                # Header for outgoing connections column
                                dbc.Col("Outgoing Width", className="text-center", width=5),
                            ], className="mb-2", style={"marginLeft": "0", "marginRight": "0"}),
                            
                            # E row
                            dbc.Row([
                                # Cell type label
                                dbc.Col(html.Strong("E"), width=2, className="d-flex align-items-center justify-content-end"),
                                # Thalamic input width slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='thalamic-width-e-slider',
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        value=INITIAL_THALAMIC_WIDTHS['E'],
                                        marks={i: f"{i}" for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingRight": "15px"}
                                ),
                                # Outgoing width slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='outgoing-width-e-slider',
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        value=INITIAL_OUTGOING_WIDTHS['E'],
                                        marks={i: f"{i}" for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingLeft": "15px"}
                                ),
                            ], className="mb-3", style={"marginLeft": "0", "marginRight": "0"}),
                            
                            # SST row
                            dbc.Row([
                                # Cell type label
                                dbc.Col(html.Strong("SST"), width=2, className="d-flex align-items-center justify-content-end"),
                                # Thalamic input width slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='thalamic-width-sst-slider',
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        value=INITIAL_THALAMIC_WIDTHS['SST'],
                                        marks={i: f"{i}" for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingRight": "15px"}
                                ),
                                # Outgoing width slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='outgoing-width-sst-slider',
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        value=INITIAL_OUTGOING_WIDTHS['SST'],
                                        marks={i: f"{i}" for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingLeft": "15px"}
                                ),
                            ], className="mb-3", style={"marginLeft": "0", "marginRight": "0"}),
                            
                            # PV row
                            dbc.Row([
                                # Cell type label
                                dbc.Col(html.Strong("PV"), width=2, className="d-flex align-items-center justify-content-end"),
                                # Thalamic input width slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='thalamic-width-pv-slider',
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        value=INITIAL_THALAMIC_WIDTHS['PV'],
                                        marks={i: f"{i}" for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingRight": "15px"}
                                ),
                                # Outgoing width slider
                                dbc.Col(
                                    dcc.Slider(
                                        id='outgoing-width-pv-slider',
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        value=INITIAL_OUTGOING_WIDTHS['PV'],
                                        marks={i: f"{i}" for i in range(0, 11, 2)},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="custom-slider"
                                    ),
                                    width=5,
                                    style={"paddingLeft": "15px"}
                                ),
                            ], style={"marginLeft": "0", "marginRight": "0"}),
                        ]),
                    ], className="mb-4"),
                    
                    # Thalamic input controls
                    html.Div([
                        dbc.Row([
                            # Empty header for label column
                            dbc.Col("", width=2),
                            # Header for thalamic input slider
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col("Intrinsic", className="text-start", width=6),
                                    dbc.Col("Sensory", className="text-end", width=6),
                                ], className="mb-1")
                            ], width=10),
                        ], className="mb-2", style={"marginLeft": "0", "marginRight": "0"}),
                        
                        dbc.Row([
                            # Label
                            dbc.Col(html.Strong("Input"), width=2, className="d-flex align-items-center justify-content-end"),
                            # Alpha slider
                            dbc.Col(
                                dcc.Slider(
                                    id='alpha-slider',
                                    min=0, max=1, step=0.1, value=THALAMIC_ALPHA,
                                    marks={i/10: f"{i/10:.1f}" for i in range(11)},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    className="custom-slider"
                                ),
                                width=10,
                                style={"paddingLeft": "15px", "paddingRight": "15px"}
                            )
                        ], style={"marginLeft": "0", "marginRight": "0"}),
                    ], className="mb-4"),
                    
                    # Simulation controls
                    html.Div([
                        # Center the pause/resume button
                        html.Div([
                            dbc.Button(
                                "Pause/Resume", id="pause-button",
                                color="primary"
                            )
                        ], style={"display": "flex", "justifyContent": "center"})
                    ])
                ], width=5, className="ps-4 control-panel-column")  # Adjust control panel column width
            ], className="g-4 px-4"),  # Add horizontal padding to row
        ], fluid=True, className="py-3 px-0")  # Added more padding to container
    
    def create_layer_row(self, layer: str) -> dbc.Row:
        """Create a row for a single cortical layer with cell types as columns."""
        # Define the cell type order (SST, E, PV)
        ordered_cell_types = ['SST', 'E', 'PV']
        
        return html.Div(
            dbc.Row([
                # Layer label
                dbc.Col([
                    html.H4(LAYER_NAMES[layer], 
                           style={"position": "relative", "top": "50%", "transform": "translateY(-50%)", 
                                 "textAlign": "right", "paddingRight": "20px", "margin": "0"})
                ], width=2),
                
                # Cell type columns (SST, E, PV from left to right)
                dbc.Col([
                    html.Div([
                        html.Div(
                            dcc.Graph(
                                id=f'graph-{layer}-{cell_type}',
                                figure=self.figures[f'graph-{layer}-{cell_type}'],
                                config={'displayModeBar': False}
                            ),
                            style={"display": "inline-block"}
                        ) for cell_type in ordered_cell_types
                    ], style={
                        "display": "flex",
                        "justifyContent": "center",
                        "gap": "20px",  # Small horizontal gap
                        "width": "100%"
                    })
                ], width=10)
            ], style={"height": "180px"}),
            className="mb-5"  # Large vertical gap between layers
        )

    def create_heatmap(self, data: np.ndarray, cell_type: str) -> go.Figure:
        """Create a heatmap figure for the given neural activity data."""
        colorscale = COLORMAPS.get(cell_type, [[0, 'black'], [1, 'gray']])
        
        # Set appropriate range for each cell type
        if cell_type == 'thalamus':
            zmax = THALAMIC_SCALING
        elif cell_type == 'E':
            zmax = 0.8  # More sensitive to E cell activity
        else:  # SST and PV
            zmax = 0.8  # More sensitive to inhibitory cell activity
        
        return go.Figure(
            data=[go.Heatmap(
                z=data,
                colorscale=colorscale,
                showscale=False,
                hoverinfo='none',  # Disable hover info for performance
                zmin=0,
                zmax=zmax
            )],
            layout=go.Layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=180,  # Reduced height
                width=180,  # Reduced width
                dragmode=False,
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    scaleanchor="y",  # Force square aspect ratio
                    scaleratio=1
                ),
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False
                )
            )
        )
    
    def get_connection_key(self, source_layer, source_cell, target_layer, target_cell):
        """Generate a connection key based on source and target information."""
        if source_layer == 'Th':
            return f'thalamus_to_{target_layer}_{target_cell}'
        else:
            return f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
    
    def get_connection_value(self, source_layer, source_cell, target_layer, target_cell):
        """Get the current connection strength value."""
        conn_key = self.get_connection_key(source_layer, source_cell, target_layer, target_cell)
        
        # First try to get the value from the simulation connectivity
        if hasattr(self, 'simulation') and hasattr(self.simulation, 'connectivity'):
            # Convert 'Th' to 'thalamus' for the simulation API
            source_layer_sim = 'thalamus' if source_layer == 'Th' else source_layer
            try:
                return self.simulation.connectivity.get_connection_strength(
                    source_layer_sim, source_cell, target_layer, target_cell
                )
            except Exception as e:
                print(f"Error getting connection from simulation: {e}")
                
        # Fall back to config-based lookup
        if conn_key in LAYER_CONNECTIVITY_PARAMS:
            return LAYER_CONNECTIVITY_PARAMS[conn_key]['amplitude']
            
        # Default to 0 if not found
        return 0.0

    def create_connection_matrix(self) -> html.Div:
        """Create a matrix visualization of all layer and cell type connections."""
        # Define consistent cell dimensions
        CELL_SIZE = 50  # Size for data cells in pixels
        HEADER_HEIGHT = 40  # Height for headers in pixels
        HEADER_WIDTH = 40  # Width for all header columns
        
        # Define the labels/indices for the matrix
        all_populations = [(layer, cell_type) for layer in LAYERS for cell_type in CELL_TYPES]
        all_populations.append(('Th', None))  # Add thalamus
        
        # Create the main header row with layer spans (removed "To" text)
        main_header_cells = [
            # Empty cell for top-left corner (matches header width)
            html.Th("", colSpan=2, style={
                "border": "none", 
                "width": f"{HEADER_WIDTH}px", 
                "height": f"{HEADER_HEIGHT}px",
                "minWidth": f"{HEADER_WIDTH}px",
                "maxWidth": f"{HEADER_WIDTH}px"
            })
        ]
        
        # Add layer headers that span 3 columns each (for E, SST, PV)
        for layer in LAYERS:
            # Different background colors based on layer
            bg_color = "rgba(180, 180, 180, 0.3)" if layer == "L4" else "rgba(180, 180, 180, 0.15)"
            
            main_header_cells.append(
                html.Th(
                    LAYER_NAMES[layer], 
                    className="text-center fw-bold",
                    colSpan=3,  # Span all cell types
                    style={
                        "backgroundColor": bg_color, 
                        "color": "white",  # White text for layer headers
                        "padding": "10px 5px",
                        "fontSize": "0.9rem",  # Consistent font size
                        "height": f"{HEADER_HEIGHT}px",  # Consistent height
                        "minHeight": f"{HEADER_HEIGHT}px",
                        "maxHeight": f"{HEADER_HEIGHT}px"
                    }
                )
            )
        
        main_header_row = html.Tr(main_header_cells)
        
        # Create the cell type sub-header row
        sub_header_cells = [
            # Empty cell to align with the layer column (matches header width)
            html.Th("", style={
                "border": "none", 
                "width": f"{HEADER_WIDTH}px", 
                "height": f"{HEADER_HEIGHT}px",
                "minWidth": f"{HEADER_WIDTH}px",
                "maxWidth": f"{HEADER_WIDTH}px"
            }),
            # Empty cell to align with the cell type column (matches cell type header width)
            html.Th("", style={
                "border": "none", 
                "width": f"{HEADER_WIDTH}px", 
                "height": f"{HEADER_HEIGHT}px",
                "minWidth": f"{HEADER_WIDTH}px",
                "maxWidth": f"{HEADER_WIDTH}px"
            })
        ]
        
        # Add all cell types under their respective layers
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                # Set background colors based on cell type (darker colors)
                bg_color = "#103a5b" if cell_type == "E" else \
                          "#752f00" if cell_type == "SST" else \
                          "#661111"  # PV
                
                # Add right border to last cell in each layer
                right_border = "1px solid #555" if cell_type == "PV" else "none"
                
                sub_header_cells.append(
                    html.Th(
                        cell_type,
                        className="text-center",
                        style={
                            "backgroundColor": bg_color, 
                            "color": "white",
                            "padding": "8px 5px",
                            "width": f"{HEADER_WIDTH}px",
                            "height": f"{HEADER_HEIGHT}px",  # Consistent height
                            "minWidth": f"{HEADER_WIDTH}px",
                            "maxWidth": f"{HEADER_WIDTH}px",
                            "minHeight": f"{HEADER_HEIGHT}px",
                            "maxHeight": f"{HEADER_HEIGHT}px",
                            "fontSize": "0.9rem",  # Consistent font size
                            "borderRight": right_border
                        }
                    )
                )
        
        sub_header_row = html.Tr(sub_header_cells)
        
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
                layer_name = "Thalamus" if source_layer == 'Th' else LAYER_NAMES[source_layer]
                layer_cells_count = 1 if source_layer == 'Th' else len(CELL_TYPES)
                
                # Different background colors based on layer
                bg_color = "rgba(180, 180, 180, 0.3)" if source_layer == "L4" else \
                          "rgba(180, 180, 180, 0.15)" if source_layer != "Th" else \
                          "transparent"  # Transparent for Thalamus
                
                row_header = html.Th(
                    layer_name,
                    className="fw-bold",
                    rowSpan=layer_cells_count,
                    style={
                        "backgroundColor": bg_color, 
                        "color": "white",  # White text for all layer headers
                        "textAlign": "center",
                        "verticalAlign": "middle",
                        "padding": "10px 5px",
                        "width": f"{HEADER_WIDTH}px",
                        "minWidth": f"{HEADER_WIDTH}px",
                        "maxWidth": f"{HEADER_WIDTH}px",
                        "height": "100%",
                        "fontSize": "0.9rem"  # Consistent font size
                    }
                )
            else:
                # No placeholder needed anymore due to structure change
                row_header = None
            
            # Create the cell type header for this row
            cell_type_bg = "#103a5b" if source_cell == "E" else \
                         "#752f00" if source_cell == "SST" else \
                         "#661111" if source_cell == "PV" else \
                         "transparent"  # Transparent for Thalamus
            
            cell_type_header = html.Th(
                source_cell or "",
                className="text-center",
                style={
                    "backgroundColor": cell_type_bg,
                    "color": "white",
                    "width": f"{HEADER_WIDTH}px",
                    "height": f"{CELL_SIZE}px",  # Make cells more square
                    "minWidth": f"{HEADER_WIDTH}px",
                    "maxWidth": f"{HEADER_WIDTH}px",
                    "minHeight": f"{CELL_SIZE}px", 
                    "maxHeight": f"{CELL_SIZE}px",
                    "padding": "5px",
                    "fontSize": "0.9rem"  # Consistent font size
                }
            )
            
            # Create data cells
            cells = []
            for target_layer in LAYERS:
                for target_cell in CELL_TYPES:
                    # Skip certain connection types
                    if (source_layer == 'Th' and target_layer == 'Th'):
                        cells.append(html.Td(
                            "",
                            className="text-center",
                            style={
                                "backgroundColor": "#1a1a1a",
                                "width": f"{CELL_SIZE}px",
                                "height": f"{CELL_SIZE}px",  # Make cells more square
                                "minWidth": f"{CELL_SIZE}px",
                                "maxWidth": f"{CELL_SIZE}px",
                                "minHeight": f"{CELL_SIZE}px",
                                "maxHeight": f"{CELL_SIZE}px"
                            }
                        ))
                        continue
                    
                    # Get connection strength
                    value = self.get_connection_value(source_layer, source_cell, target_layer, target_cell)
                    
                    # Create cell with background color based on strength
                    if value > 0:
                        intensity = min(value / 1.0, 1.0) * 0.7
                        bg_color = f"rgba(0, 120, 215, {intensity})"
                        hover_color = f"rgba(0, 150, 255, {intensity + 0.2})"
                    elif value < 0:
                        intensity = min(abs(value) / 1.0, 1.0) * 0.7
                        bg_color = f"rgba(215, 0, 0, {intensity})"
                        hover_color = f"rgba(255, 0, 0, {intensity + 0.2})"
                    else:
                        bg_color = "rgba(80, 80, 80, 0.1)"
                        hover_color = "rgba(100, 100, 100, 0.3)"
                    
                    # Determine if this cell is at a layer boundary (add border to right side of last cell in layer)
                    right_border = "1px solid #555" if target_cell == "PV" else "none"
                    
                    # Create cell with unique ID for callbacks
                    cell_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"
                    cells.append(html.Td(
                        f"{value:.1f}",
                        id={'type': 'connection-cell', 'id': cell_id},
                        className="connection-cell text-center",
                        style={
                            "backgroundColor": bg_color,
                            "cursor": "pointer",
                            "transition": "background-color 0.2s",
                            "width": f"{CELL_SIZE}px",
                            "height": f"{CELL_SIZE}px",
                            "minWidth": f"{CELL_SIZE}px",
                            "maxWidth": f"{CELL_SIZE}px",
                            "minHeight": f"{CELL_SIZE}px",
                            "maxHeight": f"{CELL_SIZE}px",
                            "padding": "5px",
                            "fontSize": "0.8rem",  # Consistent font size for values
                            "borderRight": right_border
                        },
                        **{
                            'data-highlight-color': hover_color
                        }
                    ))
            
            # Create row with header (if needed) and cells
            # Add bottom border to rows at layer boundaries
            is_last_in_layer = (source_layer != 'Th' and source_cell == 'PV') or \
                             (source_layer == 'Th')
            
            bottom_border = {"borderBottom": "1px solid #555"} if is_last_in_layer else {}
            
            if row_header:
                row_style = {"marginLeft": "0", "marginRight": "0", **bottom_border}
                rows.append(html.Tr([row_header, cell_type_header] + cells, style=row_style))
            else:
                row_style = {"marginLeft": "0", "marginRight": "0", **bottom_border}
                rows.append(html.Tr([cell_type_header] + cells, style=row_style))
        
        # Create table
        return html.Div([
            html.Table(
                [main_header_row, sub_header_row] + rows,
                className="table connection-matrix",
                style={
                    "tableLayout": "fixed", 
                    "fontSize": "0.8rem",
                    "borderCollapse": "collapse",
                    "width": "auto",
                    "margin": "0 auto",
                    "borderSpacing": "0",
                    "border": "none"  # Remove outermost border
                }
            )
        ], style={
            # Add custom CSS to override any remaining table borders
            "& table tr:last-child td, & table tr:last-child th": {
                "borderBottom": "none"
            },
            "& table tr td:last-child, & table tr th:last-child": {
                "borderRight": "none"
            }
        })  # Removed the horizontal scrolling

    def create_slider_for_cell(self, source_layer, source_cell, target_layer, target_cell, value):
        """Create a slider component for a connection cell."""
        # Set slider range based on excitatory/inhibitory type
        is_excitatory = source_cell == 'E' or source_layer == 'Th'
        slider_min = 0 if is_excitatory else -2.0
        slider_max = 2.0
        
        # Create unique ID for slider
        slider_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"
        
        return html.Div([
            html.Div(
                f"{source_layer}" + (f"-{source_cell}" if source_cell else "") + 
                f" → {target_layer}-{target_cell}", 
                style={"marginBottom": "5px", "textAlign": "center"}
            ),
            dcc.Slider(
                id={'type': 'matrix-slider', 'id': slider_id},
                min=slider_min, max=slider_max, step=0.1,
                value=value,
                marks={
                    slider_min: f"{slider_min:.1f}",
                    0: "0",
                    slider_max/2: f"{slider_max/2:.1f}",
                    slider_max: f"{slider_max:.1f}"
                }
            ),
            html.Div(
                id={'type': 'slider-value', 'id': slider_id}, 
                style={"marginTop": "5px", "textAlign": "center"}
            )
        ])

    def setup_callbacks(self):
        """Set up the dashboard callbacks for interactivity."""
        # Create a generic preset application function to avoid code duplication
        def apply_preset(preset):
            # Update all connection strengths
            for conn_key, strength in preset['connection_strengths'].items():
                # Parse the connection key to get source and target info
                parts = conn_key.split('_to_')
                source_parts = parts[0].split('_')
                target_parts = parts[1].split('_')
                
                if source_parts[0] == 'thalamus':
                    source_layer = 'thalamus'
                    source_cell = None
                    target_layer = target_parts[0]
                    target_cell = target_parts[1]
                else:
                    source_layer = source_parts[0]
                    source_cell = source_parts[1]
                    target_layer = target_parts[0]
                    target_cell = target_parts[1]
                
                # Update the connection strength in the simulation
                self.simulation.connectivity.set_connection_strength(
                    source_layer, source_cell, target_layer, target_cell, strength
                )
            
            # Regenerate the connection matrix to reflect the updated values
            updated_matrix = self.create_connection_matrix()
                
            return [
                # Time constants
                preset['time_constants']['E'],
                preset['time_constants']['SST'],
                preset['time_constants']['PV'],
                # Firing thresholds
                preset['firing_thresholds']['E'],
                preset['firing_thresholds']['SST'],
                preset['firing_thresholds']['PV'],
                # Thalamic widths
                preset['thalamic_widths']['E'],
                preset['thalamic_widths']['SST'],
                preset['thalamic_widths']['PV'],
                # Outgoing widths
                preset['outgoing_widths']['E'],
                preset['outgoing_widths']['SST'],
                preset['outgoing_widths']['PV'],
                # Thalamic alpha
                preset['thalamic_alpha'],
                # Updated connection matrix
                updated_matrix
            ]
        
        # Add callback for P4 preset button
        @self.app.callback(
            [
                Output('tau-e-slider', 'value'),
                Output('tau-sst-slider', 'value'),
                Output('tau-pv-slider', 'value'),
                Output('threshold-e-slider', 'value'),
                Output('threshold-sst-slider', 'value'),
                Output('threshold-pv-slider', 'value'),
                Output('thalamic-width-e-slider', 'value'),
                Output('thalamic-width-sst-slider', 'value'),
                Output('thalamic-width-pv-slider', 'value'),
                Output('outgoing-width-e-slider', 'value'),
                Output('outgoing-width-sst-slider', 'value'),
                Output('outgoing-width-pv-slider', 'value'),
                Output('alpha-slider', 'value'),
                Output('connection-matrix-container', 'children')
            ],
            Input('p4-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p4_preset(n_clicks):
            """Apply the P4 preset values to all parameters."""
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate
            return apply_preset(P4_PRESET)
        
        # Add callback for P8 preset button
        @self.app.callback(
            [
                Output('tau-e-slider', 'value', allow_duplicate=True),
                Output('tau-sst-slider', 'value', allow_duplicate=True),
                Output('tau-pv-slider', 'value', allow_duplicate=True),
                Output('threshold-e-slider', 'value', allow_duplicate=True),
                Output('threshold-sst-slider', 'value', allow_duplicate=True),
                Output('threshold-pv-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-e-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-sst-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-pv-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-e-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-sst-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-pv-slider', 'value', allow_duplicate=True),
                Output('alpha-slider', 'value', allow_duplicate=True),
                Output('connection-matrix-container', 'children', allow_duplicate=True)
            ],
            Input('p8-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p8_preset(n_clicks):
            """Apply the P8 preset values to all parameters."""
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate
            return apply_preset(P8_PRESET)
        
        # Add callback for P12 preset button
        @self.app.callback(
            [
                Output('tau-e-slider', 'value', allow_duplicate=True),
                Output('tau-sst-slider', 'value', allow_duplicate=True),
                Output('tau-pv-slider', 'value', allow_duplicate=True),
                Output('threshold-e-slider', 'value', allow_duplicate=True),
                Output('threshold-sst-slider', 'value', allow_duplicate=True),
                Output('threshold-pv-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-e-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-sst-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-pv-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-e-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-sst-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-pv-slider', 'value', allow_duplicate=True),
                Output('alpha-slider', 'value', allow_duplicate=True),
                Output('connection-matrix-container', 'children', allow_duplicate=True)
            ],
            Input('p12-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p12_preset(n_clicks):
            """Apply the P12 preset values to all parameters."""
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate
            return apply_preset(P12_PRESET)
        
        # Add callback for P16 preset button
        @self.app.callback(
            [
                Output('tau-e-slider', 'value', allow_duplicate=True),
                Output('tau-sst-slider', 'value', allow_duplicate=True),
                Output('tau-pv-slider', 'value', allow_duplicate=True),
                Output('threshold-e-slider', 'value', allow_duplicate=True),
                Output('threshold-sst-slider', 'value', allow_duplicate=True),
                Output('threshold-pv-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-e-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-sst-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-pv-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-e-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-sst-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-pv-slider', 'value', allow_duplicate=True),
                Output('alpha-slider', 'value', allow_duplicate=True),
                Output('connection-matrix-container', 'children', allow_duplicate=True)
            ],
            Input('p16-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p16_preset(n_clicks):
            """Apply the P16 preset values to all parameters."""
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate
            return apply_preset(P16_PRESET)

        # Initialize slider container (hidden)
        @self.app.callback(
            [Output('slider-container', 'style'),
             Output('slider-container', 'children'),
             Output('selected-cell', 'data')],
            [Input('connection-matrix-container', 'children')],
            [State('selected-cell', 'data')]
        )
        def initialize_slider_container(_, current_data):
            """Initialize the slider container as hidden when the dashboard loads."""
            return {'display': 'none', 'position': 'absolute'}, [], None
            
        # Handle cell clicks to show the slider
        @self.app.callback(
            [Output('slider-container', 'style', allow_duplicate=True),
             Output('slider-container', 'children', allow_duplicate=True),
             Output('selected-cell', 'data', allow_duplicate=True)],
            [Input({'type': 'connection-cell', 'id': ALL}, 'n_clicks')],
            [State('selected-cell', 'data')],
            prevent_initial_call=True
        )
        def handle_cell_click(clicks, current_data):
            """Show the connection strength slider when a matrix cell is clicked."""
            try:
                # Get the context that triggered the callback
                ctx = dash.callback_context
                if not ctx.triggered:
                    return {'display': 'none', 'position': 'absolute'}, [], None
                
                # Get the ID of the clicked cell
                triggered_prop_id = ctx.triggered[0]['prop_id']
                cell_data = json.loads(triggered_prop_id.split('.')[0])
                clicked_id = cell_data['id']
                
                # Extract connection info from the ID
                parts = clicked_id.split('-')
                if len(parts) < 4:
                    print(f"Invalid cell ID format: {clicked_id}")
                    return {'display': 'none', 'position': 'absolute'}, [], None
                
                source_layer = parts[0]
                source_cell = parts[1] if parts[1] != "None" else None
                target_layer = parts[2]
                target_cell = parts[3]
                
                # Get current connection value
                value = self.get_connection_value(source_layer, source_cell, target_layer, target_cell)
                
                # Create slider component
                slider = self.create_slider_for_cell(
                    source_layer, source_cell, target_layer, target_cell, value
                )
                
                # Create connection data for state
                connection_data = {
                    "source_layer": source_layer,
                    "source_cell": source_cell,
                    "target_layer": target_layer,
                    "target_cell": target_cell,
                    "slider_id": clicked_id
                }
                
                # Return with initial position - exact positioning will be handled by clientside JS
                return {
                    'display': 'block',
                    'backgroundColor': 'rgba(50, 50, 50, 0.9)',
                    'padding': '10px',
                    'border': '1px solid #444',
                    'borderRadius': '5px',
                    'zIndex': '1000',
                    'width': '200px',
                    'position': 'absolute',
                    'top': '0px',
                    'left': '0px'
                }, slider, connection_data
            except Exception as e:
                print(f"Error handling cell click: {e}")
                return {'display': 'none', 'position': 'absolute'}, [], None
        
        # Update connection strength when slider changes
        @self.app.callback(
            Output({'type': 'slider-value', 'id': MATCH}, 'children'),
            Input({'type': 'matrix-slider', 'id': MATCH}, 'value'),
            State('selected-cell', 'data')
        )
        def update_connection_value(value, connection_data):
            """Update the connection strength value display and simulation when slider changes."""
            if not connection_data:
                return ""
            
            try:
                # Update connection in simulation
                source_layer = connection_data['source_layer'] 
                source_cell = connection_data['source_cell']
                target_layer = connection_data['target_layer']
                target_cell = connection_data['target_cell']
                
                # Handle thalamus special case
                if source_layer == 'Th':
                    source_layer = 'thalamus'
                
                # Update simulation connection strength
                self.simulation.connectivity.set_connection_strength(
                    source_layer, source_cell, target_layer, target_cell, value
                )
                
                return f"Value: {value:.1f}"
            except Exception as e:
                print(f"Error updating connection value: {e}")
                return f"Error: {str(e)}"
        
        # Update connection cell in matrix when slider changes
        @self.app.callback(
            [Output({'type': 'connection-cell', 'id': MATCH}, 'children'),
             Output({'type': 'connection-cell', 'id': MATCH}, 'style'),
             Output({'type': 'connection-cell', 'id': MATCH}, 'data-highlight-color')],
            Input({'type': 'matrix-slider', 'id': MATCH}, 'value'),
            [State({'type': 'connection-cell', 'id': MATCH}, 'style'),
             State({'type': 'connection-cell', 'id': MATCH}, 'id')]
        )
        def update_matrix_cell(value, current_style, cell_id):
            """Update the matrix cell appearance and value when the slider changes."""
            if value is None:
                # No change if value is None
                return dash.no_update, dash.no_update, dash.no_update
            
            try:
                # Determine color based on connection value
                if value > 0:
                    intensity = min(value / 1.0, 1.0) * 0.7
                    bg_color = f"rgba(0, 120, 215, {intensity})"
                    hover_color = f"rgba(0, 150, 255, {intensity + 0.2})"
                elif value < 0:
                    intensity = min(abs(value) / 1.0, 1.0) * 0.7
                    bg_color = f"rgba(215, 0, 0, {intensity})"
                    hover_color = f"rgba(255, 0, 0, {intensity + 0.2})"
                else:
                    bg_color = "rgba(80, 80, 80, 0.1)"
                    hover_color = "rgba(100, 100, 100, 0.3)"
                
                # Update style with new background color
                updated_style = {**current_style, "backgroundColor": bg_color}
                
                # Return updated text, style, and hover color
                return f"{value:.1f}", updated_style, hover_color
            except Exception as e:
                print(f"Error updating matrix cell: {e}")
                return dash.no_update, dash.no_update, dash.no_update
        
        # Reset the slider when clicking the reset button
        @self.app.callback(
            [Output('slider-container', 'style', allow_duplicate=True),
             Output('slider-container', 'children', allow_duplicate=True),
             Output('selected-cell', 'data', allow_duplicate=True)],
            Input('reset-slider-state-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def reset_slider_state(n_clicks):
            """Reset the slider state when clicking outside the slider or matrix."""
            return {'display': 'none', 'position': 'absolute'}, [], None
        
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
            Output('slider-container', 'id'),
            [Input('slider-container', 'style'),
             Input('slider-container', 'children'),
             Input('selected-cell', 'data')],
        )
        
        # Add callback for updating time constants and thresholds in simulation
        @self.app.callback(
            [Output('interval-component', 'n_intervals')],  # Dummy output to trigger update
            [Input('tau-e-slider', 'value'),
             Input('tau-sst-slider', 'value'),
             Input('tau-pv-slider', 'value'),
             Input('threshold-e-slider', 'value'),
             Input('threshold-sst-slider', 'value'),
             Input('threshold-pv-slider', 'value')],
            [State('interval-component', 'n_intervals')]
        )
        def update_neuron_parameters(tau_e, tau_sst, tau_pv, threshold_e, threshold_sst, threshold_pv, n_intervals):
            # No need to update parameters here since it's already handled in update_graphs
            # This callback is kept just to maintain the slider interactivity and force a refresh
            # when the sliders are moved
            
            # Return unchanged intervals to not disrupt the update loop
            return [n_intervals]
        
        # Add callback for updating connectivity widths in simulation
        @self.app.callback(
            [Output('interval-component', 'n_intervals', allow_duplicate=True)],
            [Input('thalamic-width-e-slider', 'value'),
             Input('thalamic-width-sst-slider', 'value'),
             Input('thalamic-width-pv-slider', 'value'),
             Input('outgoing-width-e-slider', 'value'),
             Input('outgoing-width-sst-slider', 'value'),
             Input('outgoing-width-pv-slider', 'value')],
            [State('interval-component', 'n_intervals')],
            prevent_initial_call=True
        )
        def update_connectivity_parameters(sigma_thal_e, sigma_thal_sst, sigma_thal_pv, 
                                          sigma_e_out, sigma_sst_out, sigma_pv_out, n_intervals):
            # Update thalamic connections for all layers
            for layer in LAYERS:
                # Set Thalamus -> E connections
                self.simulation.set_connection_sigma('thalamus', None, layer, 'E', sigma_thal_e)
                
                # Set Thalamus -> SST connections
                self.simulation.set_connection_sigma('thalamus', None, layer, 'SST', sigma_thal_sst)
                
                # Set Thalamus -> PV connections
                self.simulation.set_connection_sigma('thalamus', None, layer, 'PV', sigma_thal_pv)
            
            # Update all E outgoing connections
            for source_layer in LAYERS:
                for target_layer in LAYERS:
                    for target_cell in CELL_TYPES:
                        # Check if this connection exists in the model
                        if ('E', target_cell) in CONNECTIONS:
                            self.simulation.set_connection_sigma(source_layer, 'E', target_layer, target_cell, sigma_e_out)
                            
            # Update all SST outgoing connections
            for source_layer in LAYERS:
                for target_layer in LAYERS:
                    # SST connections only go to E and PV cells (not to SST)
                    if ('SST', 'E') in CONNECTIONS:
                        self.simulation.set_connection_sigma(source_layer, 'SST', target_layer, 'E', sigma_sst_out)
                    if ('SST', 'PV') in CONNECTIONS:
                        self.simulation.set_connection_sigma(source_layer, 'SST', target_layer, 'PV', sigma_sst_out)
                
            # Update all PV outgoing connections
            for source_layer in LAYERS:
                for target_layer in LAYERS:
                    for target_cell in CELL_TYPES:
                        # Check if this connection exists in the model
                        if ('PV', target_cell) in CONNECTIONS:
                            self.simulation.set_connection_sigma(source_layer, 'PV', target_layer, target_cell, sigma_pv_out)
            
            # Return unchanged intervals to not disrupt the update loop
            return [n_intervals]
        
        # Update the graphs with neural activity
        @self.app.callback(
            # Outputs: all graph figures
            [Output(f'graph-{layer}-{cell_type}', 'figure')
             for layer in LAYERS
             for cell_type in CELL_TYPES] +
            [Output('graph-thalamus', 'figure')],
            
            # Inputs: interval trigger, alpha slider, time constant sliders, threshold sliders and connectivity width sliders
            [Input('interval-component', 'n_intervals'),
             Input('alpha-slider', 'value'),
             Input('tau-e-slider', 'value'),
             Input('tau-sst-slider', 'value'),
             Input('tau-pv-slider', 'value'),
             Input('threshold-e-slider', 'value'),
             Input('threshold-sst-slider', 'value'),
             Input('threshold-pv-slider', 'value'),
             Input('thalamic-width-e-slider', 'value'),
             Input('thalamic-width-sst-slider', 'value'),
             Input('thalamic-width-pv-slider', 'value'),
             Input('outgoing-width-e-slider', 'value'),
             Input('outgoing-width-sst-slider', 'value'),
             Input('outgoing-width-pv-slider', 'value')],
            
            # States: pause button state
            [State('pause-button', 'n_clicks')]
        )
        def update_graphs(n_intervals, alpha, tau_e, tau_sst, tau_pv, threshold_e, threshold_sst, threshold_pv, 
                         sigma_thal_e, sigma_thal_sst, sigma_thal_pv, sigma_e_out, sigma_sst_out, sigma_pv_out, 
                         pause_clicks):
            # Check if simulation is paused
            if pause_clicks is not None and pause_clicks % 2 == 1:
                # If paused, return current figures without updates
                return list(self.figures.values())
            
            # Update neural parameters
            self.simulation.set_time_constant('E', tau_e)
            self.simulation.set_time_constant('SST', tau_sst)
            self.simulation.set_time_constant('PV', tau_pv)
            
            self.simulation.set_firing_threshold('E', threshold_e)
            self.simulation.set_firing_threshold('SST', threshold_sst)
            self.simulation.set_firing_threshold('PV', threshold_pv)
            
            try:
                # Update simulation state with new alpha value
                activities = self.simulation.update(alpha=alpha)
                
                # Update all figures with current activity
                updated_figures = []
                
                # Update all neural population figures
                for layer in LAYERS:
                    for cell_type in CELL_TYPES:
                        fig_id = f'graph-{layer}-{cell_type}'
                        fig = self.figures[fig_id]
                        
                        # Update the heatmap data
                        fig.data[0].z = activities[layer][cell_type]
                        updated_figures.append(fig)
                
                # Update thalamus figure
                thalamus_fig = self.figures['graph-thalamus']
                thalamus_fig.data[0].z = activities['thalamus']
                updated_figures.append(thalamus_fig)
                
                return updated_figures
            except Exception as e:
                print(f"Error updating graphs: {e}")
                # Return unchanged figures on error
                return list(self.figures.values())
        
        # Toggle simulation pause state
        @self.app.callback(
            Output('interval-component', 'disabled'),
            [Input('pause-button', 'n_clicks')]
        )
        def toggle_simulation(n_clicks):
            return n_clicks is not None and n_clicks % 2 == 1
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard application."""
        self.app.run_server(
            debug=debug, 
            port=port, 
            threaded=True,
            dev_tools_silence_routes_logging=True
        ) 