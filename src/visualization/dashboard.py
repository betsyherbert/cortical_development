import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from typing import Dict, List, Tuple
import time

# Custom color scales
COLORMAPS = {
    'E': [[0, 'black'], [1, 'blue']],
    'SST': [[0, 'black'], [1, 'orange']],
    'PV': [[0, 'black'], [1, 'red']],
    'thalamus': [[0, 'black'], [1, 'white']]
}

class DashboardApp:
    def __init__(self, circuit, update_interval: int = 100):
        """
        Initialize the dashboard application.
        
        Args:
            circuit: CorticalCircuit instance
            update_interval: Update interval in milliseconds
        """
        self.circuit = circuit
        self.update_interval = update_interval
        
        # Initialize the Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        
        # Set up the layout
        self.setup_layout()
        
        # Set up callbacks
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Cortical Circuit Simulation", 
                           className="text-center mb-4")
                ])
            ]),
            
            # Activity visualization
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.H4(f"Layer {layer}", className="text-center")
                            ]) for layer in ['L2/3', 'L4', 'L5']
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id=f'graph-{layer}-{cell_type}',
                                    figure=self.create_heatmap(np.zeros((10, 10)), cell_type)
                                ) for cell_type in ['E', 'SST', 'PV']
                            ]) for layer in ['L23', 'L4', 'L5']
                        ])
                    ]),
                    # Thalamus visualization
                    dbc.Row([
                        dbc.Col([
                            html.H4("Thalamus", className="text-center"),
                            dcc.Graph(
                                id='graph-thalamus',
                                figure=self.create_heatmap(np.zeros((10, 10)), 'thalamus')
                            )
                        ], width=4, className="mx-auto")
                    ])
                ], width=8),
                
                # Control panel
                dbc.Col([
                    html.H4("Control Panel", className="text-center mb-4"),
                    
                    # Connection strength controls
                    html.Div([
                        html.H5("Connection Strengths"),
                        *self.create_connection_sliders()
                    ], className="mb-4"),
                    
                    # Thalamic input controls
                    html.Div([
                        html.H5("Thalamic Input"),
                        dbc.Label("Intrinsic/Sensory Balance (α)"),
                        dcc.Slider(
                            id='alpha-slider',
                            min=0, max=1, step=0.1, value=0.7,
                            marks={i/10: str(i/10) for i in range(11)}
                        )
                    ], className="mb-4"),
                    
                    # Simulation controls
                    html.Div([
                        html.H5("Simulation Control"),
                        dbc.Button(
                            "Reset", id="reset-button",
                            color="warning", className="me-2"
                        ),
                        dbc.Button(
                            "Pause/Resume", id="pause-button",
                            color="primary"
                        )
                    ])
                ], width=4)
            ]),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ], fluid=True)
    
    def create_heatmap(self, data: np.ndarray, cell_type: str) -> go.Figure:
        """Create a heatmap figure for the given data."""
        return go.Figure(
            data=[go.Heatmap(
                z=data,
                colorscale=COLORMAPS[cell_type],
                showscale=False
            )],
            layout=go.Layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=200
            )
        )
    
    def create_connection_sliders(self) -> List[html.Div]:
        """Create sliders for connection strengths."""
        sliders = []
        connections = [
            ('E', 'E'), ('E', 'SST'), ('E', 'PV'),
            ('SST', 'E'), ('SST', 'PV'),
            ('PV', 'E'), ('PV', 'SST'), ('PV', 'PV')
        ]
        
        for source, target in connections:
            sliders.append(html.Div([
                dbc.Label(f"{source} → {target}"),
                dcc.Slider(
                    id=f'slider-{source}-{target}',
                    min=-1, max=1, step=0.1,
                    value=0.2 if source == 'E' else -0.1,
                    marks={i/2: str(i/2) for i in range(-2, 3)}
                )
            ], className="mb-2"))
        
        return sliders
    
    def setup_callbacks(self):
        """Set up the dashboard callbacks."""
        @self.app.callback(
            [Output(f'graph-{layer}-{cell_type}', 'figure')
             for layer in ['L23', 'L4', 'L5']
             for cell_type in ['E', 'SST', 'PV']] +
            [Output('graph-thalamus', 'figure')],
            [Input('interval-component', 'n_intervals')] +
            [Input(f'slider-{source}-{target}', 'value')
             for source, target in [
                 ('E', 'E'), ('E', 'SST'), ('E', 'PV'),
                 ('SST', 'E'), ('SST', 'PV'),
                 ('PV', 'E'), ('PV', 'SST'), ('PV', 'PV')
             ]] +
            [Input('alpha-slider', 'value')],
            [State('pause-button', 'n_clicks')]
        )
        def update_graphs(n_intervals, *args):
            # Extract parameters
            connection_params = args[:-2]
            alpha = args[-2]
            is_paused = args[-1] and args[-1] % 2 == 1
            
            if not is_paused:
                # Update connection weights
                self.update_connections(connection_params)
                
                # Update simulation
                activities = self.circuit.update(alpha)
                
                # Create figures
                figures = []
                for layer in ['L23', 'L4', 'L5']:
                    for cell_type in ['E', 'SST', 'PV']:
                        data = activities[layer][cell_type]
                        figures.append(self.create_heatmap(data, cell_type))
                
                # Add thalamus figure
                figures.append(self.create_heatmap(
                    activities['thalamus'], 'thalamus'
                ))
                
                return figures
            
            # If paused, return current figures
            return [dash.no_update] * 10
        
        @self.app.callback(
            Output('interval-component', 'disabled'),
            [Input('pause-button', 'n_clicks')]
        )
        def toggle_simulation(n_clicks):
            if n_clicks is None:
                return False
            return n_clicks % 2 == 1
        
        @self.app.callback(
            Output('interval-component', 'n_intervals'),
            [Input('reset-button', 'n_clicks')]
        )
        def reset_simulation(n_clicks):
            if n_clicks is not None:
                self.circuit.reset()
            return 0
    
    def update_connections(self, params: Tuple[float, ...]):
        """Update connection weights based on slider values."""
        connections = [
            ('E', 'E'), ('E', 'SST'), ('E', 'PV'),
            ('SST', 'E'), ('SST', 'PV'),
            ('PV', 'E'), ('PV', 'SST'), ('PV', 'PV')
        ]
        
        updates = {}
        for (source, target), value in zip(connections, params):
            updates[f'{source}_to_{target}'] = {
                'amplitude': value,
                'sigma': self.circuit.connectivity.default_params[f'{source}_to_{target}']['sigma']
            }
        
        self.circuit.connectivity.update_weights(updates)
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard application."""
        self.app.run_server(debug=debug, port=port) 