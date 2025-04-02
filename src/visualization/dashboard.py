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
    INITIAL_TIME_CONSTANTS, INITIAL_GAINS, CELL_COLORS, CELL_ACTIVITY_COLORS,
    INITIAL_STRENGTH_SCALING, INITIAL_SPARSITY
)
from model.neurons import GAIN
from model.presets import P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET

# Constants for styling
CELL_SIZE = 40  # Size for data cells in pixels
HEADER_HEIGHT = 40  # Height for headers in pixels
HEADER_WIDTH = 40  # Width for all header columns

# Common styles
HEADER_STYLE = {
    "border": "none",
    "width": f"{HEADER_WIDTH}px",
    "height": f"{HEADER_HEIGHT}px",
    "minWidth": f"{HEADER_WIDTH}px",
    "maxWidth": f"{HEADER_WIDTH}px"
}

CELL_STYLE = {
    "width": f"{CELL_SIZE}px",
    "height": f"{CELL_SIZE}px",
    "minWidth": f"{CELL_SIZE}px",
    "maxWidth": f"{CELL_SIZE}px",
    "minHeight": f"{CELL_SIZE}px",
    "maxHeight": f"{CELL_SIZE}px"
}

# Colors
LAYER_COLORS = {
    "L4": "rgba(180, 180, 180, 0.3)",
    "default": "rgba(180, 180, 180, 0.15)",
    "transparent": "transparent"
}

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
        
        # Store component for currently selected cell
        selected_cell = dcc.Store(id='selected-cell', data=None)
        
        # Hidden button for resetting slider state
        reset_btn = html.Button(
            id='reset-slider-state-btn',
            style={'display': 'none'},
            n_clicks=0
        )
        
        # Define column widths for the three-column layout
        LEFT_COL_WIDTH = 4  # Activity heatmaps
        MIDDLE_COL_WIDTH = 4  # Connectivity matrix
        RIGHT_COL_WIDTH = 4  # Control sliders
        
        self.app.layout = dbc.Container([
            # Interval component for updates
            interval,
            
            # Store component for currently selected cell
            selected_cell,
            
            # Hidden button for resetting slider state
            reset_btn,
            
            # Main content: three columns
            dbc.Row([
                # Left column: activity visualization
                dbc.Col([
                    # Add more top padding to shift visualization down
                    html.Div(style={"height": "20px"}),
                    
                    # Preset Buttons - centered container
                    html.Div([
                        dbc.Row([
                            # Empty column to match heatmap label width
                            dbc.Col(width=2),
                            # Buttons container
                            dbc.Col([
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
                            ], width=10)
                        ])
                    ], className="mb-4"),
                    
                    # One row per layer (L2/3, L4, L5)
                    *[self.create_layer_row(layer) for layer in LAYERS],
                    
                    # Thalamus visualization - centered container
                    html.Div([
                        dbc.Row([
                            # Thalamus label
                            dbc.Col([
                                html.Div([
                                    html.H6("TC",
                                           style={
                                               "margin": "0",
                                               "whiteSpace": "nowrap"  # Prevent text wrapping
                                           })
                                ], style={
                                    "display": "flex",
                                    "justifyContent": "flex-end",  # Align to the right
                                    "paddingRight": "60px",  # Match the spacing of other layer labels
                                    "height": "100%",
                                    "alignItems": "center"
                                })
                            ], width=2),
                            
                            # Thalamus heatmap
                            dbc.Col([
                                html.Div([
                                    dcc.Graph(
                                        id='graph-thalamus',
                                        figure=self.figures['graph-thalamus'],
                                        config={'displayModeBar': False}
                                    )
                                ], style={"display": "flex", "justifyContent": "center"})
                            ], width=10)
                        ], className="align-items-center")
                    ], className="mt-2")
                ], width=LEFT_COL_WIDTH, className="px-4"),  # Increased padding
                
                # Middle column: connectivity matrix
                dbc.Col([
                    # Connection Strength Matrix
                    html.Div([
                        html.H5("Connection Strengths", 
                              className="mb-3 text-center",  # Reduced from mb-4 to mb-3
                              style={
                                  "textAlign": "center",
                                  "width": "85%",  # Match matrix container width
                                  "margin": "0 auto",  # Center the title
                                  "paddingLeft": "50px"  # Match matrix container padding
                              }),
                        
                        # Connection Matrix Container with reduced width
                        html.Div(
                            self.create_connection_matrix(),
                            id="connection-matrix-container",
                            style={
                                "position": "relative", 
                                "display": "flex", 
                                "justifyContent": "center",
                                "width": "85%",  # Further reduce width to prevent overlap
                                "margin": "0 auto",  # Center the container
                                "paddingLeft": "50px"  # Add left padding to shift matrix right
                            }
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
                ], width=MIDDLE_COL_WIDTH, className="px-5"),  # Increased padding further
                    
                # Right column: Control panel
                dbc.Col([
                    # Container for control sliders
                    html.Div(
                        self.create_control_panel(),
                        className="control-panel-column",
                        style={
                            "backgroundColor": "#28323f", 
                            "borderRadius": "10px", 
                            "padding": "15px"
                        }
                    )
                ], width=RIGHT_COL_WIDTH, className="px-4")
            ], className="g-0"),  # Remove gutters from main row
        ], fluid=True, className="py-3")
    
    def create_layer_row(self, layer: str) -> dbc.Row:
        """Create a row for a single cortical layer with cell types as columns."""
        ordered_cell_types = ['SST', 'E', 'PV']
        
        return html.Div(
            dbc.Row([
                # Layer label - ensure it's properly positioned to the right
                dbc.Col([
                    html.Div([
                        html.H6(LAYER_NAMES[layer],
                               style={
                                   "margin": "0",
                                   "whiteSpace": "nowrap"  # Prevent text wrapping
                               })
                    ], style={
                        "display": "flex",
                        "justifyContent": "flex-end",  # Align to the right
                        "paddingRight": "60px",  # Large spacing from heatmaps
                        "height": "100%",
                        "alignItems": "center"
                    })
                ], width=2),
                
                # Cell type columns
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
                        "gap": "15px",
                        "width": "100%"
                    })
                ], width=10)
            ], className="align-items-center", style={"height": "118px"}),
            className="mb-5"
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
                height=150,  # Reduced height
                width=150,  # Reduced width
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
        # Define the labels/indices for the matrix
        all_populations = [(layer, cell_type) for layer in LAYERS for cell_type in CELL_TYPES]
        all_populations.append(('Th', None))  # Add thalamus
        
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
                        "backgroundColor": LAYER_COLORS["L4"] if layer == "L4" else LAYER_COLORS["default"],
                        "color": "white",
                        "padding": "10px 5px",
                        "fontSize": "0.9rem"
                    }
                )
            )
        
        # Create sub-header row for cell types
        sub_header_cells = [html.Th("", style=HEADER_STYLE) for _ in range(2)]
        
        # Add all cell types under their respective layers
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                # Get base color from CELL_ACTIVITY_COLORS with 0.2 opacity
                header_color = CELL_ACTIVITY_COLORS[cell_type]['bg'](0.2)
                sub_header_cells.append(
                    html.Th(
                        cell_type,
                        className="text-center",
                        style={
                            **HEADER_STYLE,
                            "backgroundColor": header_color,
                            "color": "white",
                            "padding": "8px 5px",
                            "fontSize": "0.9rem",
                            "borderRight": "1px solid #555" if cell_type == "PV" else "none"
                        }
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
                layer_name = "Thalamus" if source_layer == 'Th' else LAYER_NAMES[source_layer]
                layer_cells_count = 1 if source_layer == 'Th' else len(CELL_TYPES)
                
                bg_color = (LAYER_COLORS["L4"] if source_layer == "L4" else 
                           LAYER_COLORS["transparent"] if source_layer == "Th" else 
                           LAYER_COLORS["default"])
                
                row_header = html.Th(
                    layer_name,
                    className="fw-bold",
                    rowSpan=layer_cells_count,
                    style={
                        **HEADER_STYLE,
                        "backgroundColor": bg_color,
                        "color": "white",
                        "textAlign": "center",
                        "verticalAlign": "middle",
                        "padding": "10px 5px",
                        "height": "100%",
                        "fontSize": "0.9rem"
                    }
                )
            else:
                row_header = None
            
            # Create cell type header with 0.2 opacity
            header_color = CELL_ACTIVITY_COLORS.get(source_cell, {'bg': lambda x: "transparent"})['bg'](0.2) if source_cell else "transparent"
            cell_type_header = html.Th(
                source_cell or "",
                className="text-center",
                style={
                    **HEADER_STYLE,
                    "backgroundColor": header_color,
                    "color": "white",
                    "padding": "5px",
                    "fontSize": "0.9rem"
                }
            )
            
            # Create data cells
            cells = []
            for target_layer in LAYERS:
                for target_cell in CELL_TYPES:
                    # Skip thalamus to thalamus connections
                    if source_layer == 'Th' and target_layer == 'Th':
                        cells.append(html.Td("", className="text-center", style={
                            **CELL_STYLE,
                            "backgroundColor": "#1a1a1a"
                        }))
                        continue
                    
                    # Get connection strength
                    value = self.get_connection_value(source_layer, source_cell, target_layer, target_cell)
                    
                    # Determine cell colors based on connection strength and source cell type
                    if source_layer == 'Th':
                        # For thalamic connections, always use E color and only positive values
                        if value > 0:
                            intensity = min(value / 1.0, 1.0) * 0.7
                            bg_color = CELL_ACTIVITY_COLORS['E']['bg'](intensity)
                            hover_color = CELL_ACTIVITY_COLORS['E']['hover'](intensity)
                        else:
                            bg_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                            hover_color = CELL_ACTIVITY_COLORS['inactive']['hover']
                    else:
                        # For cell-type specific connections
                        if value != 0:
                            intensity = min(abs(value) / 1.0, 1.0) * 0.7
                            if source_cell in ['PV', 'SST']:
                                # For inhibitory cells: use their color for negative values, E color for positive
                                if value < 0:
                                    bg_color = CELL_ACTIVITY_COLORS[source_cell]['bg'](intensity)
                                    hover_color = CELL_ACTIVITY_COLORS[source_cell]['hover'](intensity)
                                else:
                                    bg_color = CELL_ACTIVITY_COLORS['E']['bg'](intensity)
                                    hover_color = CELL_ACTIVITY_COLORS['E']['hover'](intensity)
                            else:  # E cells
                                # For E cells: only show color for positive values
                                if value > 0:
                                    bg_color = CELL_ACTIVITY_COLORS['E']['bg'](intensity)
                                    hover_color = CELL_ACTIVITY_COLORS['E']['hover'](intensity)
                                else:
                                    bg_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                                    hover_color = CELL_ACTIVITY_COLORS['inactive']['hover']
                        else:
                            bg_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                            hover_color = CELL_ACTIVITY_COLORS['inactive']['hover']
                    
                    # Create cell with unique ID for callbacks
                    cell_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"
                    cells.append(html.Td(
                        f"{value:.1f}",
                        id={'type': 'connection-cell', 'id': cell_id},
                        className="connection-cell text-center",
                        style={
                            **CELL_STYLE,
                            "backgroundColor": bg_color,
                            "cursor": "pointer",
                            "transition": "background-color 0.2s",
                            "padding": "5px",
                            "fontSize": "0.8rem",
                            "borderRight": "1px solid #555" if target_cell == "PV" else "none"
                        },
                        **{'data-highlight-color': hover_color}
                    ))
            
            # Create row with header (if needed) and cells
            is_last_in_layer = (source_layer != 'Th' and source_cell == 'PV') or source_layer == 'Th'
            row_style = {"marginLeft": "0", "marginRight": "0"}
            if is_last_in_layer:
                row_style["borderBottom"] = "1px solid #555"
            
            row_cells = [cell for cell in [row_header, cell_type_header] + cells if cell is not None]
            rows.append(html.Tr(row_cells, style=row_style))
        
        # Create table
        return html.Div([
            html.Table(
                [html.Tr(main_header_cells), html.Tr(sub_header_cells)] + rows,
                className="table connection-matrix",
                style={
                    "tableLayout": "fixed",
                    "fontSize": "0.8rem",
                    "borderCollapse": "collapse",
                    "width": "auto",
                    "margin": "0 auto",
                    "borderSpacing": "0",
                    "border": "none"
                }
            )
        ])

    def create_slider_for_cell(self, source_layer, source_cell, target_layer, target_cell, value):
        """Create a slider component for a connection cell."""
        # Set slider range based on excitatory/inhibitory type
        is_excitatory = source_cell == 'E' or source_layer == 'Th'
        slider_min = 0 if is_excitatory else -1.0
        slider_max = 1.0
        
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
            
            # Update strength scaling factors if present in the preset
            if 'strength_scaling' in preset:
                for cell_type, scaling in preset['strength_scaling'].items():
                    self.simulation.set_strength_scaling(cell_type, scaling)
                    
            # Update sparsity factors if present in the preset
            if 'sparsity' in preset:
                for cell_type, sparsity in preset['sparsity'].items():
                    self.simulation.set_sparsity(cell_type, sparsity)
            
            # Regenerate the connection matrix to reflect the updated values
            updated_matrix = self.create_connection_matrix()
                
            return [
                # Time constants
                preset['time_constants']['E'],
                preset['time_constants']['SST'],
                preset['time_constants']['PV'],
                # Gains
                preset['gains']['E'],
                preset['gains']['SST'],
                preset['gains']['PV'],
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
                Output('gain-e-slider', 'value'),
                Output('gain-sst-slider', 'value'),
                Output('gain-pv-slider', 'value'),
                Output('thalamic-width-e-slider', 'value'),
                Output('thalamic-width-sst-slider', 'value'),
                Output('thalamic-width-pv-slider', 'value'),
                Output('outgoing-width-e-slider', 'value'),
                Output('outgoing-width-sst-slider', 'value'),
                Output('outgoing-width-pv-slider', 'value'),
                Output('strength-scaling-e-slider', 'value'),
                Output('strength-scaling-sst-slider', 'value'),
                Output('strength-scaling-pv-slider', 'value'),
                Output('strength-scaling-thalamus-slider', 'value'),
                Output('sparsity-e-slider', 'value'),
                Output('sparsity-sst-slider', 'value'),
                Output('sparsity-pv-slider', 'value'),
                Output('sparsity-thalamus-slider', 'value'),
                Output('alpha-slider', 'value'),
                Output('connection-matrix-container', 'children')
            ],
            Input('p4-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p4_preset(n_clicks):
            """Apply the P4 preset configuration."""
            # Apply the preset
            apply_preset(P4_PRESET)
            
            # Get values from the preset object
            try:
                tau_e = P4_PRESET.get_tau('E')
                tau_sst = P4_PRESET.get_tau('SST')
                tau_pv = P4_PRESET.get_tau('PV')
                gain_e = P4_PRESET.get_gain('E')
                gain_sst = P4_PRESET.get_gain('SST')
                gain_pv = P4_PRESET.get_gain('PV')
                sigma_thal_e = P4_PRESET.get_sigma('thalamus', 'E')
                sigma_thal_sst = P4_PRESET.get_sigma('thalamus', 'SST')
                sigma_thal_pv = P4_PRESET.get_sigma('thalamus', 'PV')
                sigma_e_out = P4_PRESET.get_outgoing_sigma('E')
                sigma_sst_out = P4_PRESET.get_outgoing_sigma('SST')
                sigma_pv_out = P4_PRESET.get_outgoing_sigma('PV')
                strength_e = P4_PRESET.get('strength_scaling', {}).get('E', 1.0)
                strength_sst = P4_PRESET.get('strength_scaling', {}).get('SST', 1.0)
                strength_pv = P4_PRESET.get('strength_scaling', {}).get('PV', 1.0)
                strength_thal = P4_PRESET.get('strength_scaling', {}).get('thalamus', 1.0)
                sparsity_e = P4_PRESET.get('sparsity', {}).get('E', 1.0)
                sparsity_sst = P4_PRESET.get('sparsity', {}).get('SST', 1.0)
                sparsity_pv = P4_PRESET.get('sparsity', {}).get('PV', 1.0)
                sparsity_thal = P4_PRESET.get('sparsity', {}).get('thalamus', 1.0)
                alpha = P4_PRESET.get_thalamic_alpha()
            except (AttributeError, TypeError):
                # If the preset is a dictionary, access values directly
                tau_e = P4_PRESET['time_constants']['E']
                tau_sst = P4_PRESET['time_constants']['SST']
                tau_pv = P4_PRESET['time_constants']['PV']
                gain_e = P4_PRESET['gains']['E']
                gain_sst = P4_PRESET['gains']['SST']
                gain_pv = P4_PRESET['gains']['PV']
                sigma_thal_e = P4_PRESET['thalamic_widths']['E']
                sigma_thal_sst = P4_PRESET['thalamic_widths']['SST']
                sigma_thal_pv = P4_PRESET['thalamic_widths']['PV']
                sigma_e_out = P4_PRESET['outgoing_widths']['E']
                sigma_sst_out = P4_PRESET['outgoing_widths']['SST']
                sigma_pv_out = P4_PRESET['outgoing_widths']['PV']
                strength_e = P4_PRESET['strength_scaling']['E']
                strength_sst = P4_PRESET['strength_scaling']['SST']
                strength_pv = P4_PRESET['strength_scaling']['PV']
                strength_thal = P4_PRESET['strength_scaling']['thalamus']
                sparsity_e = P4_PRESET['sparsity']['E']
                sparsity_sst = P4_PRESET['sparsity']['SST']
                sparsity_pv = P4_PRESET['sparsity']['PV']
                sparsity_thal = P4_PRESET['sparsity']['thalamus']
                alpha = P4_PRESET['thalamic_alpha']
                
            # Return all values, including our new scaling and sparsity factors
            return (tau_e, tau_sst, tau_pv,
                   gain_e, gain_sst, gain_pv,
                   sigma_thal_e, sigma_thal_sst, sigma_thal_pv,
                   sigma_e_out, sigma_sst_out, sigma_pv_out,
                   strength_e, strength_sst, strength_pv, strength_thal,
                   sparsity_e, sparsity_sst, sparsity_pv, sparsity_thal,
                   alpha, self.create_connection_matrix())
        
        # Add callback for P8 preset button
        @self.app.callback(
            [
                Output('tau-e-slider', 'value', allow_duplicate=True),
                Output('tau-sst-slider', 'value', allow_duplicate=True),
                Output('tau-pv-slider', 'value', allow_duplicate=True),
                Output('gain-e-slider', 'value', allow_duplicate=True),
                Output('gain-sst-slider', 'value', allow_duplicate=True),
                Output('gain-pv-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-e-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-sst-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-pv-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-e-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-sst-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-pv-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-e-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-sst-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-pv-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-thalamus-slider', 'value', allow_duplicate=True),
                Output('sparsity-e-slider', 'value', allow_duplicate=True),
                Output('sparsity-sst-slider', 'value', allow_duplicate=True),
                Output('sparsity-pv-slider', 'value', allow_duplicate=True),
                Output('sparsity-thalamus-slider', 'value', allow_duplicate=True),
                Output('alpha-slider', 'value', allow_duplicate=True),
                Output('connection-matrix-container', 'children', allow_duplicate=True)
            ],
            Input('p8-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p8_preset(n_clicks):
            """Apply the P8 preset configuration."""
            # Apply the preset
            apply_preset(P8_PRESET)
            
            # Get values from the preset object
            try:
                tau_e = P8_PRESET.get_tau('E')
                tau_sst = P8_PRESET.get_tau('SST')
                tau_pv = P8_PRESET.get_tau('PV')
                gain_e = P8_PRESET.get_gain('E')
                gain_sst = P8_PRESET.get_gain('SST')
                gain_pv = P8_PRESET.get_gain('PV')
                sigma_thal_e = P8_PRESET.get_sigma('thalamus', 'E')
                sigma_thal_sst = P8_PRESET.get_sigma('thalamus', 'SST')
                sigma_thal_pv = P8_PRESET.get_sigma('thalamus', 'PV')
                sigma_e_out = P8_PRESET.get_outgoing_sigma('E')
                sigma_sst_out = P8_PRESET.get_outgoing_sigma('SST')
                sigma_pv_out = P8_PRESET.get_outgoing_sigma('PV')
                strength_e = P8_PRESET.get('strength_scaling', {}).get('E', 1.0)
                strength_sst = P8_PRESET.get('strength_scaling', {}).get('SST', 1.0)
                strength_pv = P8_PRESET.get('strength_scaling', {}).get('PV', 1.0)
                strength_thal = P8_PRESET.get('strength_scaling', {}).get('thalamus', 1.0)
                sparsity_e = P8_PRESET.get('sparsity', {}).get('E', 1.0)
                sparsity_sst = P8_PRESET.get('sparsity', {}).get('SST', 1.0)
                sparsity_pv = P8_PRESET.get('sparsity', {}).get('PV', 1.0)
                sparsity_thal = P8_PRESET.get('sparsity', {}).get('thalamus', 1.0)
                alpha = P8_PRESET.get_thalamic_alpha()
            except (AttributeError, TypeError):
                # If the preset is a dictionary, access values directly
                tau_e = P8_PRESET['time_constants']['E']
                tau_sst = P8_PRESET['time_constants']['SST']
                tau_pv = P8_PRESET['time_constants']['PV']
                gain_e = P8_PRESET['gains']['E']
                gain_sst = P8_PRESET['gains']['SST']
                gain_pv = P8_PRESET['gains']['PV']
                sigma_thal_e = P8_PRESET['thalamic_widths']['E']
                sigma_thal_sst = P8_PRESET['thalamic_widths']['SST']
                sigma_thal_pv = P8_PRESET['thalamic_widths']['PV']
                sigma_e_out = P8_PRESET['outgoing_widths']['E']
                sigma_sst_out = P8_PRESET['outgoing_widths']['SST']
                sigma_pv_out = P8_PRESET['outgoing_widths']['PV']
                strength_e = P8_PRESET['strength_scaling']['E']
                strength_sst = P8_PRESET['strength_scaling']['SST']
                strength_pv = P8_PRESET['strength_scaling']['PV']
                strength_thal = P8_PRESET['strength_scaling']['thalamus']
                sparsity_e = P8_PRESET['sparsity']['E']
                sparsity_sst = P8_PRESET['sparsity']['SST']
                sparsity_pv = P8_PRESET['sparsity']['PV']
                sparsity_thal = P8_PRESET['sparsity']['thalamus']
                alpha = P8_PRESET['thalamic_alpha']
                
            # Return all values, including our new scaling and sparsity factors
            return (tau_e, tau_sst, tau_pv,
                   gain_e, gain_sst, gain_pv,
                   sigma_thal_e, sigma_thal_sst, sigma_thal_pv,
                   sigma_e_out, sigma_sst_out, sigma_pv_out,
                   strength_e, strength_sst, strength_pv, strength_thal,
                   sparsity_e, sparsity_sst, sparsity_pv, sparsity_thal,
                   alpha, self.create_connection_matrix())
        
        # Add callback for P12 preset button
        @self.app.callback(
            [
                Output('tau-e-slider', 'value', allow_duplicate=True),
                Output('tau-sst-slider', 'value', allow_duplicate=True),
                Output('tau-pv-slider', 'value', allow_duplicate=True),
                Output('gain-e-slider', 'value', allow_duplicate=True),
                Output('gain-sst-slider', 'value', allow_duplicate=True),
                Output('gain-pv-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-e-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-sst-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-pv-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-e-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-sst-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-pv-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-e-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-sst-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-pv-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-thalamus-slider', 'value', allow_duplicate=True),
                Output('sparsity-e-slider', 'value', allow_duplicate=True),
                Output('sparsity-sst-slider', 'value', allow_duplicate=True),
                Output('sparsity-pv-slider', 'value', allow_duplicate=True),
                Output('sparsity-thalamus-slider', 'value', allow_duplicate=True),
                Output('alpha-slider', 'value', allow_duplicate=True),
                Output('connection-matrix-container', 'children', allow_duplicate=True)
            ],
            Input('p12-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p12_preset(n_clicks):
            """Apply the P12 preset configuration."""
            # Apply the preset
            apply_preset(P12_PRESET)
            
            # Get values from the preset object
            try:
                tau_e = P12_PRESET.get_tau('E')
                tau_sst = P12_PRESET.get_tau('SST')
                tau_pv = P12_PRESET.get_tau('PV')
                gain_e = P12_PRESET.get_gain('E')
                gain_sst = P12_PRESET.get_gain('SST')
                gain_pv = P12_PRESET.get_gain('PV')
                sigma_thal_e = P12_PRESET.get_sigma('thalamus', 'E')
                sigma_thal_sst = P12_PRESET.get_sigma('thalamus', 'SST')
                sigma_thal_pv = P12_PRESET.get_sigma('thalamus', 'PV')
                sigma_e_out = P12_PRESET.get_outgoing_sigma('E')
                sigma_sst_out = P12_PRESET.get_outgoing_sigma('SST')
                sigma_pv_out = P12_PRESET.get_outgoing_sigma('PV')
                strength_e = P12_PRESET.get('strength_scaling', {}).get('E', 1.0)
                strength_sst = P12_PRESET.get('strength_scaling', {}).get('SST', 1.0)
                strength_pv = P12_PRESET.get('strength_scaling', {}).get('PV', 1.0)
                strength_thal = P12_PRESET.get('strength_scaling', {}).get('thalamus', 1.0)
                sparsity_e = P12_PRESET.get('sparsity', {}).get('E', 1.0)
                sparsity_sst = P12_PRESET.get('sparsity', {}).get('SST', 1.0)
                sparsity_pv = P12_PRESET.get('sparsity', {}).get('PV', 1.0)
                sparsity_thal = P12_PRESET.get('sparsity', {}).get('thalamus', 1.0)
                alpha = P12_PRESET.get_thalamic_alpha()
            except (AttributeError, TypeError):
                # If the preset is a dictionary, access values directly
                tau_e = P12_PRESET['time_constants']['E']
                tau_sst = P12_PRESET['time_constants']['SST']
                tau_pv = P12_PRESET['time_constants']['PV']
                gain_e = P12_PRESET['gains']['E']
                gain_sst = P12_PRESET['gains']['SST']
                gain_pv = P12_PRESET['gains']['PV']
                sigma_thal_e = P12_PRESET['thalamic_widths']['E']
                sigma_thal_sst = P12_PRESET['thalamic_widths']['SST']
                sigma_thal_pv = P12_PRESET['thalamic_widths']['PV']
                sigma_e_out = P12_PRESET['outgoing_widths']['E']
                sigma_sst_out = P12_PRESET['outgoing_widths']['SST']
                sigma_pv_out = P12_PRESET['outgoing_widths']['PV']
                strength_e = P12_PRESET['strength_scaling']['E']
                strength_sst = P12_PRESET['strength_scaling']['SST']
                strength_pv = P12_PRESET['strength_scaling']['PV']
                strength_thal = P12_PRESET['strength_scaling']['thalamus']
                sparsity_e = P12_PRESET['sparsity']['E']
                sparsity_sst = P12_PRESET['sparsity']['SST']
                sparsity_pv = P12_PRESET['sparsity']['PV']
                sparsity_thal = P12_PRESET['sparsity']['thalamus']
                alpha = P12_PRESET['thalamic_alpha']
                
            # Return all values, including our new scaling and sparsity factors
            return (tau_e, tau_sst, tau_pv,
                   gain_e, gain_sst, gain_pv,
                   sigma_thal_e, sigma_thal_sst, sigma_thal_pv,
                   sigma_e_out, sigma_sst_out, sigma_pv_out,
                   strength_e, strength_sst, strength_pv, strength_thal,
                   sparsity_e, sparsity_sst, sparsity_pv, sparsity_thal,
                   alpha, self.create_connection_matrix())
        
        # Add callback for P16 preset button
        @self.app.callback(
            [
                Output('tau-e-slider', 'value', allow_duplicate=True),
                Output('tau-sst-slider', 'value', allow_duplicate=True),
                Output('tau-pv-slider', 'value', allow_duplicate=True),
                Output('gain-e-slider', 'value', allow_duplicate=True),
                Output('gain-sst-slider', 'value', allow_duplicate=True),
                Output('gain-pv-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-e-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-sst-slider', 'value', allow_duplicate=True),
                Output('thalamic-width-pv-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-e-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-sst-slider', 'value', allow_duplicate=True),
                Output('outgoing-width-pv-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-e-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-sst-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-pv-slider', 'value', allow_duplicate=True),
                Output('strength-scaling-thalamus-slider', 'value', allow_duplicate=True),
                Output('sparsity-e-slider', 'value', allow_duplicate=True),
                Output('sparsity-sst-slider', 'value', allow_duplicate=True),
                Output('sparsity-pv-slider', 'value', allow_duplicate=True),
                Output('sparsity-thalamus-slider', 'value', allow_duplicate=True),
                Output('alpha-slider', 'value', allow_duplicate=True),
                Output('connection-matrix-container', 'children', allow_duplicate=True)
            ],
            Input('p16-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_p16_preset(n_clicks):
            """Apply the P16 preset configuration."""
            # Apply the preset
            apply_preset(P16_PRESET)
            
            # Get values from the preset object
            try:
                tau_e = P16_PRESET.get_tau('E')
                tau_sst = P16_PRESET.get_tau('SST')
                tau_pv = P16_PRESET.get_tau('PV')
                gain_e = P16_PRESET.get_gain('E')
                gain_sst = P16_PRESET.get_gain('SST')
                gain_pv = P16_PRESET.get_gain('PV')
                sigma_thal_e = P16_PRESET.get_sigma('thalamus', 'E')
                sigma_thal_sst = P16_PRESET.get_sigma('thalamus', 'SST')
                sigma_thal_pv = P16_PRESET.get_sigma('thalamus', 'PV')
                sigma_e_out = P16_PRESET.get_outgoing_sigma('E')
                sigma_sst_out = P16_PRESET.get_outgoing_sigma('SST')
                sigma_pv_out = P16_PRESET.get_outgoing_sigma('PV')
                strength_e = P16_PRESET.get('strength_scaling', {}).get('E', 1.0)
                strength_sst = P16_PRESET.get('strength_scaling', {}).get('SST', 1.0)
                strength_pv = P16_PRESET.get('strength_scaling', {}).get('PV', 1.0)
                strength_thal = P16_PRESET.get('strength_scaling', {}).get('thalamus', 1.0)
                sparsity_e = P16_PRESET.get('sparsity', {}).get('E', 1.0)
                sparsity_sst = P16_PRESET.get('sparsity', {}).get('SST', 1.0)
                sparsity_pv = P16_PRESET.get('sparsity', {}).get('PV', 1.0)
                sparsity_thal = P16_PRESET.get('sparsity', {}).get('thalamus', 1.0)
                alpha = P16_PRESET.get_thalamic_alpha()
            except (AttributeError, TypeError):
                # If the preset is a dictionary, access values directly
                tau_e = P16_PRESET['time_constants']['E']
                tau_sst = P16_PRESET['time_constants']['SST']
                tau_pv = P16_PRESET['time_constants']['PV']
                gain_e = P16_PRESET['gains']['E']
                gain_sst = P16_PRESET['gains']['SST']
                gain_pv = P16_PRESET['gains']['PV']
                sigma_thal_e = P16_PRESET['thalamic_widths']['E']
                sigma_thal_sst = P16_PRESET['thalamic_widths']['SST']
                sigma_thal_pv = P16_PRESET['thalamic_widths']['PV']
                sigma_e_out = P16_PRESET['outgoing_widths']['E']
                sigma_sst_out = P16_PRESET['outgoing_widths']['SST']
                sigma_pv_out = P16_PRESET['outgoing_widths']['PV']
                strength_e = P16_PRESET['strength_scaling']['E']
                strength_sst = P16_PRESET['strength_scaling']['SST']
                strength_pv = P16_PRESET['strength_scaling']['PV']
                strength_thal = P16_PRESET['strength_scaling']['thalamus']
                sparsity_e = P16_PRESET['sparsity']['E']
                sparsity_sst = P16_PRESET['sparsity']['SST']
                sparsity_pv = P16_PRESET['sparsity']['PV']
                sparsity_thal = P16_PRESET['sparsity']['thalamus']
                alpha = P16_PRESET['thalamic_alpha']
                
            # Return all values, including our new scaling and sparsity factors
            return (tau_e, tau_sst, tau_pv,
                   gain_e, gain_sst, gain_pv,
                   sigma_thal_e, sigma_thal_sst, sigma_thal_pv,
                   sigma_e_out, sigma_sst_out, sigma_pv_out,
                   strength_e, strength_sst, strength_pv, strength_thal,
                   sparsity_e, sparsity_sst, sparsity_pv, sparsity_thal,
                   alpha, self.create_connection_matrix())
        
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
                # Parse cell ID from the dictionary
                cell_id_str = cell_id['id']  # Extract the ID string from the dictionary
                source_layer, source_cell, target_layer, target_cell = cell_id_str.split('-')
                source_cell = None if source_cell == 'None' else source_cell
                
                # Determine cell colors based on connection strength and source cell type
                if source_layer == 'Th':
                    # For thalamic connections, always use E color and only positive values
                    if value > 0:
                        intensity = min(value / 1.0, 1.0) * 0.7
                        bg_color = CELL_ACTIVITY_COLORS['E']['bg'](intensity)
                        hover_color = CELL_ACTIVITY_COLORS['E']['hover'](intensity)
                    else:
                        bg_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                        hover_color = CELL_ACTIVITY_COLORS['inactive']['hover']
                else:
                    # For cell-type specific connections
                    if value != 0:
                        intensity = min(abs(value) / 1.0, 1.0) * 0.7
                        if source_cell in ['PV', 'SST']:
                            # For inhibitory cells: use their color for negative values, E color for positive
                            if value < 0:
                                bg_color = CELL_ACTIVITY_COLORS[source_cell]['bg'](intensity)
                                hover_color = CELL_ACTIVITY_COLORS[source_cell]['hover'](intensity)
                            else:
                                bg_color = CELL_ACTIVITY_COLORS['E']['bg'](intensity)
                                hover_color = CELL_ACTIVITY_COLORS['E']['hover'](intensity)
                        else:  # E cells
                            # For E cells: only show color for positive values
                            if value > 0:
                                bg_color = CELL_ACTIVITY_COLORS['E']['bg'](intensity)
                                hover_color = CELL_ACTIVITY_COLORS['E']['hover'](intensity)
                            else:
                                bg_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                                hover_color = CELL_ACTIVITY_COLORS['inactive']['hover']
                    else:
                        bg_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                        hover_color = CELL_ACTIVITY_COLORS['inactive']['hover']
                
                # Update style with new background color while preserving other styles
                updated_style = {
                    **CELL_STYLE,
                    "backgroundColor": bg_color,
                    "cursor": "pointer",
                    "transition": "background-color 0.2s",
                    "padding": "5px",
                    "fontSize": "0.8rem",
                    "borderRight": "1px solid #555" if target_cell == "PV" else "none"
                }
                
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
        
        # Add callback for updating time constants and gains in simulation
        @self.app.callback(
            [Output('interval-component', 'n_intervals')],  # Dummy output to trigger update
            [Input('tau-e-slider', 'value'),
             Input('tau-sst-slider', 'value'),
             Input('tau-pv-slider', 'value'),
             Input('gain-e-slider', 'value'),
             Input('gain-sst-slider', 'value'),
             Input('gain-pv-slider', 'value')],
            [State('interval-component', 'n_intervals')]
        )
        def update_neuron_parameters(tau_e, tau_sst, tau_pv, gain_e, gain_sst, gain_pv, n_intervals):
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
            """Update all connectivity parameters in the simulation."""
            # Update thalamic connections
            thalamic_params = [
                ('E', sigma_thal_e),
                ('SST', sigma_thal_sst),
                ('PV', sigma_thal_pv)
            ]
            for layer in LAYERS:
                for cell_type, sigma in thalamic_params:
                    self.simulation.set_connection_sigma('thalamus', None, layer, cell_type, sigma)
            
            # Update cell type outgoing connections
            outgoing_params = [
                ('E', sigma_e_out, CELL_TYPES),  # E connects to all cell types
                ('SST', sigma_sst_out, ['E', 'PV']),  # SST only connects to E and PV
                ('PV', sigma_pv_out, CELL_TYPES)  # PV connects to all cell types
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
            [Output(f'graph-{layer}-{cell_type}', 'figure')
             for layer in LAYERS
             for cell_type in CELL_TYPES] +
            [Output('graph-thalamus', 'figure')],
            
            # Inputs: interval trigger, alpha slider, time constant sliders, gain sliders, connectivity width sliders
            # Remove the strength scaling and sparsity sliders from the inputs
            [Input('interval-component', 'n_intervals'),
             Input('alpha-slider', 'value'),
             Input('tau-e-slider', 'value'),
             Input('tau-sst-slider', 'value'),
             Input('tau-pv-slider', 'value'),
             Input('gain-e-slider', 'value'),
             Input('gain-sst-slider', 'value'),
             Input('gain-pv-slider', 'value'),
             Input('thalamic-width-e-slider', 'value'),
             Input('thalamic-width-sst-slider', 'value'),
             Input('thalamic-width-pv-slider', 'value'),
             Input('outgoing-width-e-slider', 'value'),
             Input('outgoing-width-sst-slider', 'value'),
             Input('outgoing-width-pv-slider', 'value')],
            
            # States: pause button state
            [State('pause-button', 'n_clicks')]
        )
        def update_graphs(n_intervals, alpha, tau_e, tau_sst, tau_pv, gain_e, gain_sst, gain_pv, 
                         sigma_thal_e, sigma_thal_sst, sigma_thal_pv, sigma_e_out, sigma_sst_out, sigma_pv_out,
                         pause_clicks):
            """Update all graphs based on current slider values."""
            # Check if simulation is paused
            is_paused = pause_clicks is not None and pause_clicks % 2 == 1
            if is_paused:
                # Return current figures without updating if paused
                return [self.figures[f'graph-{layer}-{cell_type}'] 
                        for layer in LAYERS
                        for cell_type in CELL_TYPES] + [self.figures['graph-thalamus']]
            
            # Update neuron parameters
            self.simulation.set_time_constant('E', tau_e)
            self.simulation.set_time_constant('SST', tau_sst)
            self.simulation.set_time_constant('PV', tau_pv)
            self.simulation.set_gain('E', gain_e)
            self.simulation.set_gain('SST', gain_sst)
            self.simulation.set_gain('PV', gain_pv)
            
            # Update all connectivity widths
            for layer in LAYERS:
                # Update thalamic inputs
                self.simulation.set_connection_sigma('thalamus', None, layer, 'E', sigma_thal_e)
                self.simulation.set_connection_sigma('thalamus', None, layer, 'SST', sigma_thal_sst)
                self.simulation.set_connection_sigma('thalamus', None, layer, 'PV', sigma_thal_pv)
                
                # Update outgoing connections for each source layer
                for source_layer in LAYERS:
                    for source_cell, sigma in [('E', sigma_e_out), ('SST', sigma_sst_out), ('PV', sigma_pv_out)]:
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
                    fig_id = f'graph-{layer}-{cell_type}'
                    fig = self.figures[fig_id]
                    
                    # Update the figure data
                    with fig.batch_update():
                        # Get the layer's activity for this cell type
                        data = activities[layer][cell_type].reshape(self.simulation.grid_size, self.simulation.grid_size)
                        fig.data[0]['z'] = data
                        
                        # Update colorscale max for better contrast
                        max_val = max(data.max(), 0.5)
                        fig.update_traces(zmax=max_val)
                        
                    updated_figures.append(fig)
            
            # Update thalamus figure
            thal_fig = self.figures['graph-thalamus']
            with thal_fig.batch_update():
                thal_data = activities['thalamus']
                thal_fig.data[0]['z'] = thal_data
                max_val = max(thal_data.max(), 0.5)
                thal_fig.update_traces(zmax=max_val)
            
            updated_figures.append(thal_fig)
            
            return updated_figures
        
        # Toggle simulation pause state
        @self.app.callback(
            Output('interval-component', 'disabled'),
            [Input('pause-button', 'n_clicks')]
        )
        def toggle_simulation(n_clicks):
            return n_clicks is not None and n_clicks % 2 == 1
        
        # Add callback for updating strength scaling factors
        @self.app.callback(
            [Output('interval-component', 'n_intervals', allow_duplicate=True)],
            [Input('strength-scaling-e-slider', 'value'),
             Input('strength-scaling-sst-slider', 'value'),
             Input('strength-scaling-pv-slider', 'value'),
             Input('strength-scaling-thalamus-slider', 'value')],
            [State('interval-component', 'n_intervals')],
            prevent_initial_call=True
        )
        def update_strength_scaling_parameters(e_scaling, sst_scaling, pv_scaling, thalamus_scaling, n_intervals):
            """Update all strength scaling parameters in the simulation."""
            # Update strength scaling parameters
            self.simulation.set_strength_scaling('E', e_scaling)
            self.simulation.set_strength_scaling('SST', sst_scaling)
            self.simulation.set_strength_scaling('PV', pv_scaling)
            self.simulation.set_strength_scaling('thalamus', thalamus_scaling)
            
            # Return unchanged intervals to not disrupt the update loop
            return [n_intervals]
        
        # Add callback for updating sparsity factors
        @self.app.callback(
            [Output('interval-component', 'n_intervals', allow_duplicate=True)],
            [Input('sparsity-e-slider', 'value'),
             Input('sparsity-sst-slider', 'value'),
             Input('sparsity-pv-slider', 'value'),
             Input('sparsity-thalamus-slider', 'value')],
            [State('interval-component', 'n_intervals')],
            prevent_initial_call=True
        )
        def update_sparsity_parameters(e_sparsity, sst_sparsity, pv_sparsity, thalamus_sparsity, n_intervals):
            """Update all sparsity parameters in the simulation."""
            # Update sparsity parameters
            self.simulation.set_sparsity('E', e_sparsity)
            self.simulation.set_sparsity('SST', sst_sparsity)
            self.simulation.set_sparsity('PV', pv_sparsity)
            self.simulation.set_sparsity('thalamus', thalamus_sparsity)
            
            # Return unchanged intervals to not disrupt the update loop
            return [n_intervals]
    
    def create_parameter_sliders(self):
        """Create the neural parameter sliders section."""
        return html.Div([
            # Headers row
            dbc.Row([
                dbc.Col("", width=1),
                dbc.Col(html.Div("Time Constant", className="text-center"), width=5),
                dbc.Col(html.Div("Gain", className="text-center"), width=5),
            ], className="mb-1"),
            
            # Parameter rows
            *[self._create_parameter_row(cell_type) for cell_type in CELL_TYPES]
        ])

    def _create_parameter_row(self, cell_type):
        """Create a row of sliders for a cell type's parameters."""
        return dbc.Row([
            # Cell type label
            dbc.Col(html.Strong(cell_type), width=1, className="d-flex align-items-center", style={"paddingRight": "5px"}),
            # Time constant slider
            dbc.Col(self._create_slider(
                id_prefix='tau',
                cell_type=cell_type,
                min_val=1.0,
                max_val=100.0,
                step=1.0,
                initial_value=INITIAL_TIME_CONSTANTS[cell_type],
                marks={i: f"{i}" for i in range(20, 101, 20)}
            ), width=5, style={"paddingRight": "5px"}),
            # Gain slider
            dbc.Col(self._create_slider(
                id_prefix='gain',
                cell_type=cell_type,
                min_val=0,
                max_val=1.0,
                step=0.1,
                initial_value=INITIAL_GAINS[cell_type],
                marks={i/10: f"{i/10:.1f}" for i in range(2, 11, 2)}
            ), width=5)
        ], className="mb-1")

    def create_connectivity_sliders(self):
        """Create the connectivity width sliders section."""
        return html.Div([
            # Headers row
            dbc.Row([
                dbc.Col("", width=1),
                dbc.Col(html.Div("Thalamic", className="text-center"), width=5),
                dbc.Col(html.Div("Outgoing", className="text-center"), width=5),
            ], className="mb-1"),
            
            # Connectivity rows
            *[self._create_connectivity_row(cell_type) for cell_type in CELL_TYPES]
        ])

    def _create_connectivity_row(self, cell_type):
        """Create a row of sliders for a cell type's connectivity parameters."""
        return dbc.Row([
            # Cell type label
            dbc.Col(html.Strong(cell_type), width=1, className="d-flex align-items-center", style={"paddingRight": "5px"}),
            # Thalamic width slider
            dbc.Col(self._create_slider(
                id_prefix='thalamic-width',
                cell_type=cell_type,
                min_val=0.1,
                max_val=10.0,
                step=0.1,
                initial_value=INITIAL_THALAMIC_WIDTHS[cell_type],
                marks={i: f"{i}" for i in range(0, 11, 2)}
            ), width=5, style={"paddingRight": "5px"}),
            # Outgoing width slider
            dbc.Col(self._create_slider(
                id_prefix='outgoing-width',
                cell_type=cell_type,
                min_val=0.1,
                max_val=10.0,
                step=0.1,
                initial_value=INITIAL_OUTGOING_WIDTHS[cell_type],
                marks={i: f"{i}" for i in range(0, 11, 2)}
            ), width=5)
        ], className="mb-1")
        
    def create_strength_scaling_sliders(self):
        """Create the connection strength scaling sliders section."""
        return html.Div([
            # Headers row
            dbc.Row([
                dbc.Col("", width=1),
                dbc.Col(html.Div("Strength Scaling", className="text-center"), width=11),
            ], className="mb-1"),
            
            # Strength scaling rows
            *[self._create_strength_scaling_row(cell_type) for cell_type in CELL_TYPES],
            
            # Add thalamus strength scaling slider
            self._create_strength_scaling_row('thalamus')
        ])

    def _create_strength_scaling_row(self, cell_type):
        """Create a row for a cell type's strength scaling parameter."""
        return dbc.Row([
            # Cell type label
            dbc.Col(html.Strong(cell_type if cell_type != 'thalamus' else 'TC'), 
                   width=1, className="d-flex align-items-center", style={"paddingRight": "5px"}),
            # Strength scaling slider
            dbc.Col(self._create_slider(
                id_prefix='strength-scaling',
                cell_type=cell_type.lower(),
                min_val=0.0,
                max_val=5.0,
                step=0.1,
                initial_value=INITIAL_STRENGTH_SCALING[cell_type],
                marks={i: f"{i}" for i in range(0, 6)}
            ), width=11)
        ], className="mb-1")
        
    def create_sparsity_sliders(self):
        """Create the connection sparsity sliders section."""
        return html.Div([
            # Headers row
            dbc.Row([
                dbc.Col("", width=1),
                dbc.Col(html.Div("Sparsity", className="text-center"), width=11),
            ], className="mb-1"),
            
            # Sparsity rows
            *[self._create_sparsity_row(cell_type) for cell_type in CELL_TYPES],
            
            # Add thalamus sparsity slider
            self._create_sparsity_row('thalamus')
        ])

    def _create_sparsity_row(self, cell_type):
        """Create a row for a cell type's sparsity parameter."""
        return dbc.Row([
            # Cell type label
            dbc.Col(html.Strong(cell_type if cell_type != 'thalamus' else 'TC'), 
                   width=1, className="d-flex align-items-center", style={"paddingRight": "5px"}),
            # Sparsity slider
            dbc.Col(self._create_slider(
                id_prefix='sparsity',
                cell_type=cell_type.lower(),
                min_val=0.0,
                max_val=1.0,
                step=0.05,
                initial_value=INITIAL_SPARSITY[cell_type],
                marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)}
            ), width=11)
        ], className="mb-1")

    def _create_slider(self, id_prefix, cell_type, min_val, max_val, step, initial_value, marks):
        """Create a slider with consistent styling."""
        return dcc.Slider(
            id=f'{id_prefix}-{cell_type.lower()}-slider',
            min=min_val,
            max=max_val,
            step=step,
            value=initial_value,
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": False},
            className="custom-slider"
        )

    def create_input_controls(self):
        """Create the input control sliders section for balancing intrinsic and sensory inputs."""
        return dbc.Row([
            dbc.Col([
                # Balance labels
                dbc.Row([
                    dbc.Col("Intrinsic", className="text-start", width=6),
                    dbc.Col("Sensory", className="text-end", width=6),
                ], className="mb-2"),
                # Alpha slider - use direct ID since it's not cell-type specific
                dcc.Slider(
                    id='alpha-slider',  # Changed back to original ID
                    min=0,
                    max=1,
                    step=0.1,
                    value=THALAMIC_ALPHA,
                    marks={i/10: f"{i/10:.1f}" for i in range(11)},
                    tooltip={"placement": "bottom", "always_visible": False},
                    className="custom-slider"
                )
            ])
        ])
    
    def create_control_panel(self):
        """Create the control panel with all sliders and controls."""
        return html.Div([
            # Section: Presynaptic parameters
            html.Div([
                html.H5("Presynaptic Parameters"),
                self.create_parameter_sliders()
            ], className="mb-3"),
            
            # Section: Connectivity parameters
            html.Div([
                html.H5("Connectivity Parameters"),
                self.create_connectivity_sliders(),
                
                # New sections for strength scaling and sparsity
                html.Div([html.Hr()], className="my-3"),
                self.create_strength_scaling_sliders(),
                
                html.Div([html.Hr()], className="my-3"),
                self.create_sparsity_sliders()
            ], className="mb-3"),
            
            # Section: Input controls
            html.Div([
                html.H5("Thalamic Input Balance"),
                self.create_input_controls()
            ], className="mb-3"),
            
            # Pause/Play control
            html.Div([
                dbc.Button(
                    "Pause", 
                    id="pause-button", 
                    color="secondary", 
                    className="me-md-2"
                )
            ], className="mt-4")
        ])
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard application."""
        self.app.run_server(
            debug=debug, 
            port=port, 
            threaded=True,
            dev_tools_silence_routes_logging=True
        ) 