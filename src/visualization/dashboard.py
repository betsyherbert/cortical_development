"""Dashboard module for visualizing the cortical circuit simulation."""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from typing import Dict, List, Tuple
import time
import json

from model.config import (
    COLORMAPS, UPDATE_INTERVAL, CELL_TYPES, LAYERS, LAYER_NAMES, 
    THALAMIC_SCALING, VISUALIZATION_STEPS, LAYER_CONNECTIONS,
    LAYER_CONNECTIVITY_PARAMS
)


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
                ], width=8),
                
                # Right column: control panel
                dbc.Col([
                    html.H4("Control Panel", className="text-center mb-2"),
                    
                    # Connection Strength Matrix
                    html.Div([
                        html.H5("Connection Strengths", className="mb-2"),
                        
                        # Connection Matrix Container
                        html.Div(
                            self.create_connection_matrix(),
                            id="connection-matrix-container",
                            style={"position": "relative"}
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
                    
                    # Thalamic input controls
                    html.Div([
                        html.H5("Thalamic Input", className="mb-2"),
                        dbc.Label("Intrinsic/Sensory Balance (α)", className="mb-1"),
                        dcc.Slider(
                            id='alpha-slider',
                            min=0, max=1, step=0.1, value=0.7,
                            marks={i/10: f"{i/10:.1f}" for i in range(11)}
                        )
                    ], className="mb-3"),
                    
                    # Simulation controls
                    html.Div([
                        html.H5("Simulation Control", className="mb-2"),
                        dbc.Button(
                            "Reset", id="reset-button",
                            color="warning", className="me-2"
                        ),
                        dbc.Button(
                            "Pause/Resume", id="pause-button",
                            color="primary"
                        )
                    ])
                ], width=4, className="ps-4")
            ], className="g-2"),
        ], fluid=True, className="py-2")
    
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
        if conn_key in LAYER_CONNECTIVITY_PARAMS:
            return LAYER_CONNECTIVITY_PARAMS[conn_key]['amplitude']
        return 0.0

    def create_connection_matrix(self) -> html.Div:
        """Create a matrix visualization of all layer and cell type connections."""
        # Define the labels/indices for the matrix
        all_populations = [(layer, cell_type) for layer in LAYERS for cell_type in CELL_TYPES]
        all_populations.append(('Th', None))  # Add thalamus
        
        # Create the matrix table header (To...)
        header_cells = [html.Th("From", className="text-center")] + [
            html.Th([
                html.Div([
                    html.Div(LAYER_NAMES[layer] if layer != 'Th' else "Thalamus", className="fw-bold"),
                    html.Div(cell_type or "")
                ], className="text-center")
            ]) 
            for layer, cell_type in all_populations if layer != 'Th' or cell_type is None
        ]
        
        header_row = html.Tr(header_cells)
        
        # Generate matrix rows
        rows = []
        for source in all_populations:
            source_layer, source_cell = source
            display_layer = "Thalamus" if source_layer == 'Th' else LAYER_NAMES[source_layer]
            
            # Create row header (From...)
            row_header = html.Th([
                html.Div([
                    html.Div(display_layer, className="fw-bold"),
                    html.Div(source_cell or "")
                ], className="text-end")
            ])
            
            # Create data cells
            cells = []
            for target in all_populations:
                target_layer, target_cell = target
                
                # Skip certain connection types
                if (source_layer == 'Th' and target_layer == 'Th') or \
                   (target_layer == 'Th') or \
                   (source_cell == 'SST' and target_cell == 'SST'):
                    cells.append(html.Td(
                        "",
                        className="text-center",
                        style={"backgroundColor": "#1a1a1a"}
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
                
                # Create cell with unique ID for callbacks
                cell_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"
                cells.append(html.Td(
                    f"{value:.1f}",
                    id={'type': 'connection-cell', 'id': cell_id},
                    className="connection-cell text-center",
                    style={
                        "backgroundColor": bg_color,
                        "cursor": "pointer",
                        "transition": "background-color 0.2s"
                    },
                    **{
                        'data-highlight-color': hover_color
                    }
                ))
            
            # Add row to table
            rows.append(html.Tr([row_header] + cells))
        
        # Create table
        return html.Div([
            html.Table(
                [header_row] + rows,
                className="table table-bordered connection-matrix",
                style={"tableLayout": "fixed", "fontSize": "0.8rem"}
            )
        ])

    def create_slider_for_cell(self, source_layer, source_cell, target_layer, target_cell, value):
        """Create a slider component for a connection cell."""
        # Set slider range based on excitatory/inhibitory type
        is_excitatory = source_cell == 'E' or source_layer == 'Th'
        slider_min = 0 if is_excitatory else -5.0
        slider_max = 5.0
        
        # Create unique ID for slider
        slider_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"
        
        return html.Div([
            html.Div(
                f"{source_layer}" + (f"-{source_cell}" if source_cell else "") + 
                f" → {target_layer}-{target_cell}: {value:.1f}", 
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
        # Initialize slider container (hidden)
        @self.app.callback(
            [Output('slider-container', 'style'),
             Output('slider-container', 'children'),
             Output('selected-cell', 'data')],
            [Input('connection-matrix-container', 'children')],
            [State('selected-cell', 'data')]
        )
        def initialize_slider_container(_, current_data):
            return {'display': 'none'}, [], None
            
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
            try:
                # Get the context that triggered the callback
                ctx = dash.callback_context
                if not ctx.triggered:
                    return {'display': 'none'}, [], None
                
                # Get the ID of the clicked cell
                triggered_prop_id = ctx.triggered[0]['prop_id']
                cell_data = json.loads(triggered_prop_id.split('.')[0])
                clicked_id = cell_data['id']
                
                # Extract connection info from the ID
                parts = clicked_id.split('-')
                if len(parts) < 4:
                    print(f"Invalid cell ID format: {clicked_id}")
                    return {'display': 'none'}, [], None
                
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
                
                # Return the slider container with display block but no position
                return {
                    'display': 'block',
                    'backgroundColor': 'rgba(50, 50, 50, 0.9)',
                    'padding': '10px',
                    'border': '1px solid #444',
                    'borderRadius': '5px',
                    'zIndex': '1000',
                    'width': '200px',
                    'position': 'absolute'
                }, slider, connection_data
            except Exception as e:
                print(f"Error handling cell click: {e}")
                return {'display': 'none'}, [], None
        
        # Update connection strength when slider changes
        @self.app.callback(
            Output({'type': 'slider-value', 'id': MATCH}, 'children'),
            Input({'type': 'matrix-slider', 'id': MATCH}, 'value'),
            State('selected-cell', 'data')
        )
        def update_connection_value(value, connection_data):
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
        
        # JavaScript to handle slider positioning and cell interactions
        self.app.clientside_callback(
            """
            function(n_clicks) {
                // Function to position the slider relative to a cell
                function positionSlider(cell) {
                    const rect = cell.getBoundingClientRect();
                    const sliderContainer = document.getElementById('slider-container');
                    if (!sliderContainer) return;
                    
                    // Position the slider immediately
                    sliderContainer.style.top = (rect.bottom + window.scrollY + 5) + 'px';
                    sliderContainer.style.left = (rect.left + window.scrollX - 75) + 'px';
                    
                    // Highlight the active cell
                    document.querySelectorAll('.connection-cell').forEach(c => c.style.outline = 'none');
                    cell.style.outline = '2px solid white';
                }
                
                // Set up event handlers for connection cells
                const cells = document.querySelectorAll('.connection-cell');
                cells.forEach(cell => {
                    // Remove any existing click listeners to prevent duplicates
                    cell.removeEventListener('click', cell._clickHandler);
                    
                    // Create and store new click handler
                    cell._clickHandler = function(e) {
                        positionSlider(this);
                    };
                    
                    // Add the click handler
                    cell.addEventListener('click', cell._clickHandler);
                });
                
                // Handle the initial click if this callback was triggered by a cell click
                const ctx = window.dash_clientside.callback_context;
                if (ctx && ctx.triggered && ctx.triggered.length > 0) {
                    const triggerId = ctx.triggered[0].prop_id;
                    if (triggerId.includes('connection-cell')) {
                        const cellId = JSON.parse(triggerId.split('.')[0]).id;
                        const clickedCell = document.querySelector(`[id*="${cellId}"]`);
                        if (clickedCell) {
                            positionSlider(clickedCell);  // Position immediately without setTimeout
                        }
                    }
                }
                
                // Hide slider when clicking outside the matrix or slider
                document.addEventListener('click', function(e) {
                    if (!e.target.closest('.connection-cell') && 
                        !e.target.closest('#slider-container')) {
                        const sliderContainer = document.getElementById('slider-container');
                        if (sliderContainer) sliderContainer.style.display = 'none';
                        
                        cells.forEach(c => c.style.outline = 'none');
                    }
                });
                
                return window.dash_clientside.no_update;
            }
            """,
            Output('connection-matrix-container', 'n_clicks'),
            [Input('connection-matrix-container', 'id'),
             Input({'type': 'connection-cell', 'id': ALL}, 'n_clicks')],
        )
        
        # Update visualization based on simulation state
        @self.app.callback(
            # Outputs: all graph figures
            [Output(f'graph-{layer}-{cell_type}', 'figure')
             for layer in LAYERS
             for cell_type in CELL_TYPES] +
            [Output('graph-thalamus', 'figure')],
            
            # Inputs: interval trigger and alpha slider
            [Input('interval-component', 'n_intervals'),
             Input('alpha-slider', 'value')],
            
            # States: pause button state
            [State('pause-button', 'n_clicks')]
        )
        def update_graphs(n_intervals, alpha, pause_clicks):
            # Check if simulation is paused
            is_paused = pause_clicks and pause_clicks % 2 == 1
            
            if not is_paused:
                try:
                    # Run simulation update
                    activities = self.simulation.update(alpha)
                    
                    # Update figures efficiently by only changing the z data
                    figures = []
                    for layer in LAYERS:
                        for cell_type in CELL_TYPES:
                            fig_id = f'graph-{layer}-{cell_type}'
                            self.figures[fig_id].data[0].z = activities[layer][cell_type]
                            figures.append(self.figures[fig_id])
                    
                    # Update thalamus figure
                    self.figures['graph-thalamus'].data[0].z = activities['thalamus']
                    figures.append(self.figures['graph-thalamus'])
                    
                    return figures
                except Exception as e:
                    print(f"Error updating graphs: {e}")
                    return [dash.no_update] * (len(LAYERS) * len(CELL_TYPES) + 1)
            
            # If paused, don't update
            return [dash.no_update] * (len(LAYERS) * len(CELL_TYPES) + 1)
        
        # Toggle simulation pause state
        @self.app.callback(
            Output('interval-component', 'disabled'),
            [Input('pause-button', 'n_clicks')]
        )
        def toggle_simulation(n_clicks):
            return n_clicks is not None and n_clicks % 2 == 1
        
        # Reset simulation
        @self.app.callback(
            Output('interval-component', 'n_intervals'),
            [Input('reset-button', 'n_clicks')]
        )
        def reset_simulation(n_clicks):
            if n_clicks is not None:
                self.simulation.reset()
            return 0
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard application."""
        self.app.run_server(
            debug=debug, 
            port=port, 
            threaded=True,
            dev_tools_silence_routes_logging=True
        ) 