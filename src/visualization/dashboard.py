"""Dashboard module for visualizing the cortical circuit simulation."""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from typing import Dict, List, Tuple
import time

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
            # Prevent validation errors when dynamically generating callbacks
            suppress_callback_exceptions=True
        )
        
        # Pre-create all figures for better performance
        self.figures = {}
        self._initialize_figures()
        
        # Set up the layout
        self.setup_layout()
        
        # Set up callbacks
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
                    
                    # Connection strength controls in tabs
                    html.Div([
                        html.H5("Connection Strengths", className="mb-2"),
                        
                        # Tab navigation for connection types
                        dbc.Tabs([
                            # Within-layer connections
                            dbc.Tab(
                                self.create_within_layer_controls(),
                                label="Within-Layer", 
                                tab_id="tab-within-layer"
                            ),
                            
                            # Cross-layer connections
                            dbc.Tab(
                                self.create_cross_layer_controls(),
                                label="Cross-Layer", 
                                tab_id="tab-cross-layer"
                            ),
                            
                            # Thalamic connections
                            dbc.Tab(
                                self.create_thalamic_controls(),
                                label="Thalamic", 
                                tab_id="tab-thalamic"
                            )
                        ], id="conn-tabs", active_tab="tab-within-layer")
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
        """
        Create a row for a single cortical layer with cell types as columns.
        
        Args:
            layer: Layer identifier (e.g., 'L23', 'L4', 'L5')
            
        Returns:
            A Bootstrap row component containing the layer visualization
        """
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
        """
        Create a heatmap figure for the given neural activity data.
        
        Args:
            data: 2D array of activity values
            cell_type: Cell type for determining colormap
            
        Returns:
            Plotly figure object
        """
        colorscale = COLORMAPS.get(cell_type, [[0, 'black'], [1, 'gray']])
        
        # Set appropriate range for each cell type
        # Using lower zmax values to make activity more visible
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
                # Fix color range to prevent flashing
                zmin=0,
                zmax=zmax
            )],
            layout=go.Layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=180,  # Reduced height
                width=180,  # Reduced width
                # Optimize for performance by disabling interactions
                dragmode=False,
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    scaleanchor="y",  # Force square aspect ratio
                    scaleratio=1      # Force square aspect ratio
                ),
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False
                )
            )
        )
    
    def create_within_layer_controls(self) -> html.Div:
        """
        Create controls for within-layer connection parameters.
        
        Returns:
            Dash component with within-layer connection controls
        """
        # Filter for within-layer connections
        within_layer_controls = []
        
        # Create one accordion item per layer with sliders inside
        accordion_items = []
        
        for layer in LAYERS:
            # Create sliders for this layer's connections
            sliders = []
            
            for source_cell in CELL_TYPES:
                for target_cell in CELL_TYPES:
                    # Skip connections that don't exist (e.g., SST->SST)
                    if source_cell == 'SST' and target_cell == 'SST':
                        continue
                    
                    # Create connection key
                    conn_key = f'{layer}_{source_cell}_to_{layer}_{target_cell}'
                    
                    # Get default value if it exists
                    if conn_key in LAYER_CONNECTIVITY_PARAMS:
                        default_value = LAYER_CONNECTIVITY_PARAMS[conn_key]['amplitude']
                    else:
                        # Skip connections that aren't defined
                        continue
                    
                    # Set slider range based on excitatory/inhibitory type
                    is_excitatory = source_cell == 'E'
                    slider_min = 0 if is_excitatory else -0.5
                    slider_max = 0.5 if is_excitatory else 0
                    
                    # Create slider component
                    sliders.append(html.Div([
                        dbc.Label(f"{source_cell} → {target_cell}", className="mb-0"),
                        dcc.Slider(
                            id=f'slider-{layer}-{source_cell}-{layer}-{target_cell}',
                            min=slider_min, max=slider_max, step=0.01,
                            value=default_value,
                            marks={
                                slider_min: f"{slider_min:.1f}",
                                slider_min/2 + slider_max/2: f"{(slider_min/2 + slider_max/2):.1f}",
                                slider_max: f"{slider_max:.1f}"
                            }
                        )
                    ], className="mb-2"))
            
            # Create accordion item for this layer
            accordion_items.append(
                dbc.AccordionItem(
                    sliders,
                    title=f"{LAYER_NAMES[layer]} Connections",
                    item_id=f"layer-{layer}"
                )
            )
        
        # Return the accordion with all layers
        return html.Div([
            dbc.Accordion(
                accordion_items,
                start_collapsed=True,
                always_open=True
            )
        ])

    def create_cross_layer_controls(self) -> html.Div:
        """
        Create controls for cross-layer connection parameters.
        
        Returns:
            Dash component with cross-layer connection controls
        """
        # Create one accordion item per pair of layers
        accordion_items = []
        
        for source_layer in LAYERS:
            for target_layer in LAYERS:
                # Skip within-layer connections as they're in a different tab
                if source_layer == target_layer:
                    continue
                
                # Create sliders for connections between these layers
                sliders = []
                
                # Only excitatory neurons project between layers
                source_cell = 'E'
                for target_cell in CELL_TYPES:
                    # Create connection key
                    conn_key = f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'
                    
                    # Get default value if it exists
                    if conn_key in LAYER_CONNECTIVITY_PARAMS:
                        default_value = LAYER_CONNECTIVITY_PARAMS[conn_key]['amplitude']
                    else:
                        # Skip connections that aren't defined
                        continue
                    
                    # Create slider component
                    sliders.append(html.Div([
                        dbc.Label(f"{source_cell} → {target_cell}", className="mb-0"),
                        dcc.Slider(
                            id=f'slider-{source_layer}-{source_cell}-{target_layer}-{target_cell}',
                            min=0, max=0.3, step=0.01,
                            value=default_value,
                            marks={0: "0", 0.15: "0.15", 0.3: "0.3"}
                        )
                    ], className="mb-2"))
                
                # Only add accordion item if there are sliders
                if sliders:
                    accordion_items.append(
                        dbc.AccordionItem(
                            sliders,
                            title=f"{LAYER_NAMES[source_layer]} → {LAYER_NAMES[target_layer]}",
                            item_id=f"layers-{source_layer}-{target_layer}"
                        )
                    )
        
        # Return the accordion with all layer pairs
        return html.Div([
            dbc.Accordion(
                accordion_items,
                start_collapsed=True,
                always_open=True
            )
        ])

    def create_thalamic_controls(self) -> html.Div:
        """
        Create controls for thalamic connection parameters.
        
        Returns:
            Dash component with thalamic connection controls
        """
        # Create one accordion item per target layer
        accordion_items = []
        
        for target_layer in LAYERS:
            # Create sliders for thalamic connections to this layer
            sliders = []
            
            for target_cell in CELL_TYPES:
                # Create connection key
                conn_key = f'thalamus_to_{target_layer}_{target_cell}'
                
                # Get default value if it exists
                if conn_key in LAYER_CONNECTIVITY_PARAMS:
                    default_value = LAYER_CONNECTIVITY_PARAMS[conn_key]['amplitude']
                else:
                    # Skip connections that aren't defined
                    continue
                
                # Create slider component
                sliders.append(html.Div([
                    dbc.Label(f"Thalamus → {target_cell}", className="mb-0"),
                    dcc.Slider(
                        id=f'slider-thalamus-None-{target_layer}-{target_cell}',
                        min=0, max=0.3, step=0.01,
                        value=default_value,
                        marks={0: "0", 0.15: "0.15", 0.3: "0.3"}
                    )
                ], className="mb-2"))
            
            # Create accordion item for this layer
            accordion_items.append(
                dbc.AccordionItem(
                    sliders,
                    title=f"Thalamus → {LAYER_NAMES[target_layer]}",
                    item_id=f"thal-{target_layer}"
                )
            )
        
        # Return the accordion with all layers
        return html.Div([
            dbc.Accordion(
                accordion_items,
                start_collapsed=True,
                always_open=True
            )
        ])
    
    def collect_layer_connection_ids(self) -> List[str]:
        """
        Collect all slider IDs for layer-specific connections.
        
        Returns:
            List of slider IDs for all layer-specific connections
        """
        slider_ids = []
        
        # Within-layer connections
        for layer in LAYERS:
            for source_cell in CELL_TYPES:
                for target_cell in CELL_TYPES:
                    # Skip connections that don't exist (e.g., SST->SST)
                    if source_cell == 'SST' and target_cell == 'SST':
                        continue
                    
                    slider_ids.append(f'slider-{layer}-{source_cell}-{layer}-{target_cell}')
        
        # Cross-layer connections (E cells only)
        for source_layer in LAYERS:
            for target_layer in LAYERS:
                if source_layer != target_layer:  # Skip within-layer
                    for target_cell in CELL_TYPES:
                        slider_ids.append(f'slider-{source_layer}-E-{target_layer}-{target_cell}')
        
        # Thalamic connections
        for target_layer in LAYERS:
            for target_cell in CELL_TYPES:
                slider_ids.append(f'slider-thalamus-None-{target_layer}-{target_cell}')
        
        return slider_ids
    
    def setup_callbacks(self):
        """Set up the dashboard callbacks for interactivity."""
        # Collect all slider IDs for layer-specific connections
        slider_ids = self.collect_layer_connection_ids()
        
        @self.app.callback(
            # Outputs: all graph figures
            [Output(f'graph-{layer}-{cell_type}', 'figure')
             for layer in LAYERS
             for cell_type in CELL_TYPES] +
            [Output('graph-thalamus', 'figure')],
            
            # Inputs: interval trigger and all sliders
            [Input('interval-component', 'n_intervals')] +
            [Input(slider_id, 'value') for slider_id in slider_ids] +
            [Input('alpha-slider', 'value')],
            
            # States: pause button state
            [State('pause-button', 'n_clicks')]
        )
        def update_graphs(n_intervals, *args):
            # Extract parameters from args
            connection_params = args[:-2]  # All but last two args
            alpha = args[-2] if args[-2] is not None else 0.7  # Second to last arg, default to 0.7
            pause_clicks = args[-1]  # Last arg
            
            # Check if simulation is paused
            is_paused = pause_clicks and pause_clicks % 2 == 1
            
            if not is_paused:
                try:
                    # Update connection weights only if they've changed
                    ctx = dash.callback_context
                    if ctx.triggered:
                        prop_id = ctx.triggered[0]['prop_id']
                        if 'slider' in prop_id:
                            self.update_connections(slider_ids, connection_params)
                    
                    # Run simulation update
                    activities = self.simulation.update(alpha)
                    
                    # Update figures efficiently by only changing the z data
                    figures = []
                    for layer in LAYERS:
                        for cell_type in CELL_TYPES:
                            fig_id = f'graph-{layer}-{cell_type}'
                            # Update only the z data without recreating the figure
                            self.figures[fig_id].data[0].z = activities[layer][cell_type]
                            figures.append(self.figures[fig_id])
                    
                    # Update thalamus figure
                    self.figures['graph-thalamus'].data[0].z = activities['thalamus']
                    figures.append(self.figures['graph-thalamus'])
                    
                    return figures
                except Exception as e:
                    # On error, don't update
                    return [dash.no_update] * (len(LAYERS) * len(CELL_TYPES) + 1)
            
            # If paused, don't update
            return [dash.no_update] * (len(LAYERS) * len(CELL_TYPES) + 1)
        
        @self.app.callback(
            Output('interval-component', 'disabled'),
            [Input('pause-button', 'n_clicks')]
        )
        def toggle_simulation(n_clicks):
            """Toggle the simulation between running and paused states."""
            if n_clicks is None:
                return False
            return n_clicks % 2 == 1
        
        @self.app.callback(
            Output('interval-component', 'n_intervals'),
            [Input('reset-button', 'n_clicks')]
        )
        def reset_simulation(n_clicks):
            """Reset the simulation to its initial state."""
            if n_clicks is not None:
                self.simulation.reset()
            return 0
    
    def update_connections(self, slider_ids: List[str], params: Tuple[float, ...]):
        """
        Update connection weights based on slider values.
        
        Args:
            slider_ids: List of slider IDs
            params: Tuple of connection strength values from sliders
        """
        # Map slider IDs to their values
        slider_values = dict(zip(slider_ids, params))
        
        # Update layer-specific connection strengths
        for slider_id, value in slider_values.items():
            # Parse the slider ID to get connection information
            # Format: slider-{source_layer}-{source_cell}-{target_layer}-{target_cell}
            parts = slider_id.split('-')
            if len(parts) >= 5:
                source_layer = parts[1]
                source_cell = parts[2]
                target_layer = parts[3]
                target_cell = parts[4]
                
                # Update connection strength in the simulation
                self.simulation.connectivity.set_connection_strength(
                    source_layer, source_cell, target_layer, target_cell, value
                )
    
    def run(self, debug: bool = True, port: int = 8050):
        """
        Run the dashboard application.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        self.app.run_server(
            debug=debug, 
            port=port, 
            threaded=True,
            dev_tools_silence_routes_logging=True
        ) 