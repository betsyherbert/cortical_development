"""Dashboard module for visualizing the cortical circuit simulation."""

import json
from typing import Optional
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc

from src.model.config import (
    COLORMAPS, UPDATE_INTERVAL, CELL_TYPES, LAYERS, LAYER_NAMES,
    LAYER_CONNECTIVITY_PARAMS, THALAMIC_ALPHA, CONNECTIONS,
    INITIAL_THALAMIC_WIDTHS, INITIAL_OUTGOING_WIDTHS, 
    INITIAL_TIME_CONSTANTS, CELL_ACTIVITY_COLORS,
    INITIAL_STRENGTH_SCALING, INITIAL_BACKGROUND_INPUT
)
from src.model.presets import P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET
from src.analysis.bifurcation.bifurcation_analysis import (
    NetworkModel, StabilityAnalyzer
)

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

# Colors for light mode
LAYER_COLORS = {
    "L4": "rgba(52, 73, 94, 0.15)",
    "default": "rgba(149, 165, 166, 0.15)",
    "transparent": "transparent"
}

# Table header styles for light mode
MAIN_HEADER_STYLE = {
    **HEADER_STYLE,
    "backgroundColor": LAYER_COLORS["default"],
    "color": "#2c3e50",
    "padding": "10px 5px",
    "fontSize": "0.9rem",
    "fontWeight": "600"
}

LAYER_HEADER_STYLE = {
    **MAIN_HEADER_STYLE,
    "backgroundColor": LAYER_COLORS["L4"]  # Will be overridden for non-L4 layers
}

CELL_TYPE_HEADER_STYLE = {
    **HEADER_STYLE,
    "color": "#2c3e50",
    "padding": "8px 5px",
    "fontSize": "0.9rem",
    "fontWeight": "500"
}

ROW_HEADER_STYLE = {
    **HEADER_STYLE,
    "color": "#2c3e50",
    "textAlign": "center",
    "verticalAlign": "middle",
    "padding": "10px 5px",
    "height": "100%",
    "fontSize": "0.9rem",
    "fontWeight": "600"
}

# Common layout styles
CONTROL_PANEL_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "10px",
    "padding": "15px",
    "border": "1px solid #ddd"
}

SLIDER_CONTAINER_STYLE = {
    "backgroundColor": "rgba(255, 255, 255, 0.95)",
    "padding": "10px",
    "border": "1px solid #ccc",
    "borderRadius": "5px",
    "zIndex": "1000",
    "width": "200px",
    "position": "absolute",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"
}

# Graph configuration
GRAPH_CONFIG = {'displayModeBar': False}

# Heatmap scaling constants for fair comparison across all cell types
HEATMAP_ZMIN = 0.0  # Minimum activity value
HEATMAP_ZMAX = 1.0  # Maximum activity value (allows for some headroom)

GRAPH_LAYOUT = {
    "margin": dict(l=0, r=0, t=0, b=0),
    "height": 150,  # Reduced height
    "width": 150,  # Reduced width
    "dragmode": False,
    "xaxis": dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        scaleanchor="y",  # Force square aspect ratio
        scaleratio=1
    ),
    "yaxis": dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False
    )
}

class ConnectionKeyUtils:
    """Utility for parsing and building connection keys."""
    
    @staticmethod
    def parse(conn_key: str):
        """Parse connection key to (source_layer, source_cell, target_layer, target_cell).
        
        Args:
            conn_key: String like 'L23_E_to_L4_PV' or 'thalamus_to_L4_E'
            
        Returns:
            Tuple of (source_layer, source_cell, target_layer, target_cell)
        """
        parts = conn_key.split('_to_')
        source_parts = parts[0].split('_')
        target_parts = parts[1].split('_')
        
        if source_parts[0] == 'thalamus':
            return 'thalamus', None, target_parts[0], target_parts[1]
        return source_parts[0], source_parts[1], target_parts[0], target_parts[1]
    
    @staticmethod
    def build(source_layer, source_cell, target_layer, target_cell):
        """Build connection key from components.
        
        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', 'Th', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            
        Returns:
            Connection key string
        """
        if source_layer in ['Th', 'thalamus']:
            return f'thalamus_to_{target_layer}_{target_cell}'
        return f'{source_layer}_{source_cell}_to_{target_layer}_{target_cell}'

class DashboardApp:
    """
    Dashboard application for visualizing and controlling the neural simulation.
    
    This class creates an interactive Dash application that displays real-time
    neural activity and provides controls for adjusting simulation parameters.
    """
    
    # Common slider styles and parameters
    SLIDER_STYLE = {
        "tooltip": {"placement": "bottom", "always_visible": False},
        "className": "custom-slider"
    }
    
    TIME_CONSTANT_PARAMS = {
        "min_val": 1.0,
        "max_val": 40.0,
        "step": 1.0,
        "marks": {1: "1", 10: "10", 20: "20", 30: "30", 40: "40"}
    }
    
    GAIN_PARAMS = {
        "min_val": 0.4,
        "max_val": 1.0,
        "step": 0.1,
        "marks": {0.4: "0.4", 0.6: "0.6", 0.8: "0.8", 1.0: "1"}
    }
    
    WIDTH_PARAMS = {
        "min_val": 0.1,
        "max_val": 8.0,
        "step": 0.1,
        "marks": {i: f"{i}" for i in range(0, 9, 2)}
    }
    
    STRENGTH_SCALING_PARAMS = {
        "min_val": 0.0,
        "max_val": 5.0,
        "step": 0.1,
        "marks": {i: f"{i}" for i in range(0, 6)}
    }
    
    BACKGROUND_INPUT_PARAMS = {
        "min_val": 0.0,
        "max_val": 0.4,
        "step": 0.05,
        "marks": {
            0.0: "0.0",
            0.2: "0.2",
            0.4: "0.4"
        }
    }
    
    def __init__(self, simulation, update_interval: int = UPDATE_INTERVAL):
        """
        Initialize the dashboard application.
        
        Args:
            simulation: CorticalSimulation instance
            update_interval: Update interval in milliseconds
        """
        self.simulation = simulation
        self.update_interval = update_interval
        
        # Initialize the Dash app with light theme
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.FLATLY],
            suppress_callback_exceptions=True
        )
        
        # Add custom CSS for sliders and light mode styling
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
                    body {
                        background-color: #f8f9fa !important;
                    }
                    .custom-slider .rc-slider-track {
                        background-color: #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-rail {
                        background-color: #ddd !important;
                    }
                    .custom-slider .rc-slider-handle {
                        border-color: #2c3e50 !important;
                        background-color: #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-handle:hover {
                        border-color: #34495e !important;
                    }
                    .custom-slider .rc-slider-handle:active {
                        border-color: #2c3e50 !important;
                        box-shadow: 0 0 5px #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-dot {
                        border-color: #ccc !important;
                        background-color: #ccc !important;
                    }
                    .custom-slider .rc-slider-dot-active {
                        border-color: #2c3e50 !important;
                        background-color: #2c3e50 !important;
                    }
                    .custom-slider .rc-slider-mark {
                        width: 100% !important;
                    }
                    .custom-slider .rc-slider-mark-text {
                        color: #333 !important;
                    }
                    .custom-slider .rc-slider-mark-text:last-child {
                        transform: translateX(-100%) !important;
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
        
        # Activity history for temporal averaging in spectrum computation
        self.activity_history = []  # List of recent mean_rates arrays
        self.history_length = 1    # How many snapshots to average over
        
        # Define common outputs for preset callbacks
        self._PRESET_OUTPUTS = [
            Output('tau-e-slider', 'value'),
            Output('tau-sst-slider', 'value'),
            Output('tau-pv-slider', 'value'),
            Output('background-input-e-slider', 'value'),
            Output('background-input-sst-slider', 'value'),
            Output('background-input-pv-slider', 'value'),
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
            Output('alpha-slider', 'value'),
            Output('connection-matrix-container', 'children')
        ]
        
        # Set up the layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def build_current_preset(self) -> dict:
        """
        Build a preset dictionary from current simulation parameters.
        
        Returns:
            Dictionary compatible with NetworkModel initialization
        """
        # Get all current parameters from simulation
        connection_strengths = self.simulation.connectivity.get_all_connection_strengths()
        strength_scaling = self.simulation.connectivity.get_all_strength_scaling()
        time_constants = self.simulation.circuit.get_time_constants()
        gains = self.simulation.circuit.get_gains()
        background_input = self.simulation.circuit.get_all_background_inputs()
        all_sigmas = self.simulation.connectivity.get_all_sigmas()
        
        # Extract outgoing widths per cell type (use first occurrence for each source cell type)
        outgoing_widths = {}
        thalamic_widths = {}
        
        for conn_key, sigma in all_sigmas.items():
            if conn_key.startswith('thalamus_to_'):
                # Thalamic connection: extract target cell type
                parts = conn_key.split('_')
                target_cell = parts[3]  # thalamus_to_L4_E -> 'E'
                if target_cell not in thalamic_widths:
                    thalamic_widths[target_cell] = sigma
            else:
                # Regular connection: extract source cell type
                source_part = conn_key.split('_to_')[0]
                source_cell = source_part.split('_')[1]  # L4_E_to_L5_SST -> 'E'
                if source_cell not in outgoing_widths:
                    outgoing_widths[source_cell] = sigma
        
        # Build preset dictionary
        preset = {
            'connection_strengths': connection_strengths,
            'strength_scaling': strength_scaling,
            'time_constants': time_constants,
            'gains': gains,
            'background_input': background_input,
            'outgoing_widths': outgoing_widths,
            'thalamic_widths': thalamic_widths,
            'thalamic_alpha': THALAMIC_ALPHA  # Not used in bifurcation analysis but included
        }
        
        return preset
    
    def _extract_mean_rates_from_simulation(self) -> Optional[np.ndarray]:
        """
        Extract spatial mean rates from current simulation state with temporal averaging.
        
        Returns temporal average over recent history for smoother spectrum computation.
        
        Returns:
            Array of 9 mean rates [L23_E, L23_SST, L23_PV, L4_E, L4_SST, L4_PV, L5_E, L5_SST, L5_PV]
            or None if network is not yet active (all rates < threshold)
        """
        # Get current layer activities
        activities = self.simulation.circuit.get_layer_activities()
        
        # Extract spatial mean for each population
        mean_rates = []
        for layer in ['L23', 'L4', 'L5']:
            for cell_type in ['E', 'SST', 'PV']:
                spatial_mean = activities[layer][cell_type].mean()
                mean_rates.append(spatial_mean)
        
        mean_rates = np.array(mean_rates)
        
        # Check if network is active (instantaneous)
        if np.all(mean_rates < 1e-10):
            return None  # Network not yet active
        
        # Add to history buffer
        self.activity_history.append(mean_rates)
        if len(self.activity_history) > self.history_length:
            self.activity_history.pop(0)
        
        # Return temporal average if enough history, otherwise instantaneous
        if len(self.activity_history) >= 3:
            # Use temporal average for smoother spectrum
            averaged_rates = np.mean(self.activity_history, axis=0)
            # Check if averaged rates are still active
            if np.all(averaged_rates < 1e-10):
                return None
            return averaged_rates
        else:
            # Not enough history yet, use instantaneous
            return mean_rates
    
    def compute_stability_spectrum(self, preset: dict) -> tuple:
        """
        Compute stability spectrum (max Re(λ) vs k) for current network state.
        
        Linearizes around the current spatial-mean activity from the running simulation.
        Optimized: scans only positive quadrant and caches exponentials.
        
        Args:
            preset: Preset dictionary with current network parameters
            
        Returns:
            Tuple of (k_values, max_real_eigenvalues) arrays
        """
        try:
            # Extract current spatial mean rates from simulation
            steady_state = self._extract_mean_rates_from_simulation()
            
            # Check if network is active
            if steady_state is None:
                # Network not yet active - return empty arrays
                return np.array([]), np.array([])
            
            # Create network model for full network (all 3 layers)
            network = NetworkModel(preset, layers=['L23', 'L4', 'L5'])
            
            # Create stability analyzer with current simulation state
            analyzer = StabilityAnalyzer(network, steady_state)
            
            # Scan k modes and collect max Re(λ) for each k
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS
            n_modes = ANALYSIS_PARAMS['n_modes']
            grid_size = ANALYSIS_PARAMS['grid_size']
            domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)
            n_modes_effective = min(n_modes, int(0.6 * domain_length))
            
            total_pops = len(network.tau)
            
            # Pre-compute all unique k² values and cache exponentials
            # Only scan positive quadrant: n1 >= 0, n2 >= 0
            k_squared_set = set()
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared_set.add(n1**2 + n2**2)
            
            # Build cache: exp_cache[k²][i,j] = exp(-2π²k²σ²_ij)
            exp_cache = {}
            for k_squared in k_squared_set:
                exp_cache[k_squared] = np.zeros((total_pops, total_pops))
                for i in range(total_pops):
                    for j in range(total_pops):
                        sigma_ij = network.sigma[i, j] / domain_length
                        exp_cache[k_squared][i, j] = np.exp(
                            -2 * np.pi**2 * k_squared * sigma_ij**2
                        )
            
            # Dictionary to store results by k^2
            results_by_k2 = {}
            
            # Scan only positive quadrant (reduces work by ~4×)
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)
                    
                    # Skip if k > n_modes (keep k in [0, n_modes])
                    if k > n_modes:
                        continue
                    
                    # Build Jacobian using cached exponentials
                    J = np.zeros((total_pops, total_pops))
                    exp_factors = exp_cache[k_squared]
                    
                    for i in range(total_pops):
                        for j in range(total_pops):
                            w_tilde = network.A[i, j] * exp_factors[i, j]
                            if i == j:
                                J[i, j] = (-1.0 / network.tau[i] + 
                                          (analyzer.g_eff[i] * w_tilde) / network.tau[i])
                            else:
                                J[i, j] = (analyzer.g_eff[i] * w_tilde) / network.tau[i]
                    
                    # Compute eigenvalues
                    eigenvalues = np.linalg.eigvals(J)
                    max_real = np.max(eigenvalues.real)
                    
                    # Store or update max real eigenvalue for this k
                    if k_squared not in results_by_k2:
                        results_by_k2[k_squared] = {'k': k, 'max_real': max_real}
                    else:
                        results_by_k2[k_squared]['max_real'] = max(
                            results_by_k2[k_squared]['max_real'], max_real
                        )
            
            # Sort by k and extract arrays
            sorted_results = sorted(results_by_k2.values(), key=lambda x: x['k'])
            k_values = np.array([r['k'] for r in sorted_results])
            max_real_values = np.array([r['max_real'] for r in sorted_results])
            
            return k_values, max_real_values
            
        except Exception as e:
            print(f"Error computing stability spectrum: {e}")
            return np.array([]), np.array([])
    
    def compute_B_fourier(self, network: 'NetworkModel', k_squared: float, domain_length: float) -> np.ndarray:
        """
        Compute thalamic input B(k) in Fourier space with Gaussian spatial filtering.
        
        Args:
            network: NetworkModel instance containing thalamic parameters
            k_squared: Square of the wavenumber k
            domain_length: Domain length for normalization
            
        Returns:
            B(k): Thalamic input vector in Fourier space (length = number of populations)
        """
        total_pops = len(network.thalamic_strengths)
        B_k = np.zeros(total_pops)
        
        for i in range(total_pops):
            # Normalize thalamic width by domain length
            sigma_thal_i = network.thalamic_widths[i] / domain_length
            # Apply Gaussian spatial filtering: B[i] = strength * exp(-2π²k²σ²)
            B_k[i] = network.thalamic_strengths[i] * np.exp(-2 * np.pi**2 * k_squared * sigma_thal_i**2)
        
        return B_k
    
    def compute_static_gain(self, preset: dict) -> tuple:
        """
        Compute static spatial gain curve G(k) = ||−J(k)^(−1) B(k)||.
        
        Shows how strongly each spatial frequency is amplified by the cortical circuit.
        
        Args:
            preset: Network preset dictionary
            
        Returns:
            (k_values, gain_values): Arrays of k and corresponding gains
        """
        try:
            # Get steady state from running simulation
            steady_state = self._extract_mean_rates_from_simulation()
            if steady_state is None:
                return np.array([]), np.array([])
            
            # Build network model and analyzer
            network = NetworkModel(preset, layers=['L23', 'L4', 'L5'])
            analyzer = StabilityAnalyzer(network, steady_state)
            
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS
            n_modes = ANALYSIS_PARAMS['n_modes']
            grid_size = ANALYSIS_PARAMS['grid_size']
            domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)
            n_modes_effective = min(n_modes, int(0.6 * domain_length))
            total_pops = len(network.tau)
            
            # Cache exponentials for cortical connections (reuse from stability spectrum)
            k_squared_set = set()
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared_set.add(n1**2 + n2**2)
            
            # Cache for cortical connection exponentials
            exp_cache = {}
            for k_squared in k_squared_set:
                exp_cache[k_squared] = np.zeros((total_pops, total_pops))
                for i in range(total_pops):
                    for j in range(total_pops):
                        sigma_ij = network.sigma[i, j] / domain_length
                        exp_cache[k_squared][i, j] = np.exp(
                            -2 * np.pi**2 * k_squared * sigma_ij**2
                        )
            
            # Aggregate results by k² (handle degenerate modes)
            results_by_k2 = {}
            
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)
                    if k > n_modes:
                        continue
                    
                    # Build Jacobian J(k)
                    J = np.zeros((total_pops, total_pops))
                    exp_factors = exp_cache[k_squared]
                    
                    for i in range(total_pops):
                        for j in range(total_pops):
                            w_tilde = network.A[i, j] * exp_factors[i, j]
                            if i == j:
                                J[i, j] = (-1.0 / network.tau[i] + 
                                          (analyzer.g_eff[i] * w_tilde) / network.tau[i])
                            else:
                                J[i, j] = (analyzer.g_eff[i] * w_tilde) / network.tau[i]
                    
                    # Compute B(k) with thalamic spatial filtering
                    B_k = self.compute_B_fourier(network, k_squared, domain_length)
                    
                    # Check if B(k) is non-zero
                    if np.linalg.norm(B_k) < 1e-10:
                        continue
                    
                    # Compute gain: G(k) = ||−J(k)^(−1) B(k)||
                    try:
                        # Compute -J(k)^(-1) @ B(k)
                        J_inv_B = np.linalg.solve(-J, B_k)
                        # Spectral norm (largest singular value)
                        gain = np.linalg.norm(J_inv_B)
                        
                        # Store maximum gain across degenerate modes
                        if k_squared not in results_by_k2:
                            results_by_k2[k_squared] = {'k': k, 'gain': gain}
                        else:
                            results_by_k2[k_squared]['gain'] = max(
                                results_by_k2[k_squared]['gain'], gain
                            )
                    except np.linalg.LinAlgError:
                        # Singular matrix - skip this mode
                        continue
            
            # Sort by k and extract results
            sorted_results = sorted(results_by_k2.values(), key=lambda x: x['k'])
            k_values = np.array([r['k'] for r in sorted_results])
            gain_values = np.array([r['gain'] for r in sorted_results])
            
            return k_values, gain_values
            
        except Exception as e:
            print(f"Error computing static gain: {e}")
            return np.array([]), np.array([])
    
    def compute_spatiotemporal_gain(self, preset: dict) -> tuple:
        """
        Compute spatiotemporal amplification map A(k,ω) = ||(iωI − J(k))^(−1) B(k)||.
        
        Shows which spatial (k) and temporal (ω) frequencies the circuit amplifies most.
        
        Args:
            preset: Network preset dictionary
            
        Returns:
            (k_values, omega_values, gain_matrix): Arrays of k, ω, and gain[k,ω]
        """
        try:
            # Get steady state from running simulation
            steady_state = self._extract_mean_rates_from_simulation()
            if steady_state is None:
                return np.array([]), np.array([]), np.array([])
            
            # Build network model and analyzer
            network = NetworkModel(preset, layers=['L23', 'L4', 'L5'])
            analyzer = StabilityAnalyzer(network, steady_state)
            
            from src.analysis.bifurcation.config import ANALYSIS_PARAMS
            n_modes = ANALYSIS_PARAMS['n_modes']
            grid_size = ANALYSIS_PARAMS['grid_size']
            domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)
            n_modes_effective = min(n_modes, int(0.6 * domain_length))
            total_pops = len(network.tau)
            
            # Define temporal frequency range (0-1 Hz)
            omega_values = np.linspace(0, 1, 21)  # 21 samples for smooth heatmap up to 1 Hz
            
            # Cache exponentials for cortical connections
            k_squared_set = set()
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared_set.add(n1**2 + n2**2)
            
            exp_cache = {}
            for k_squared in k_squared_set:
                exp_cache[k_squared] = np.zeros((total_pops, total_pops))
                for i in range(total_pops):
                    for j in range(total_pops):
                        sigma_ij = network.sigma[i, j] / domain_length
                        exp_cache[k_squared][i, j] = np.exp(
                            -2 * np.pi**2 * k_squared * sigma_ij**2
                        )
            
            # Aggregate k values (just scan positive quadrant for efficiency)
            k_values_dict = {}
            for n1 in range(0, n_modes_effective + 1):
                for n2 in range(0, n_modes_effective + 1):
                    k_squared = n1**2 + n2**2
                    k = np.sqrt(k_squared)
                    if k > n_modes:
                        continue
                    if k_squared not in k_values_dict:
                        k_values_dict[k_squared] = k
            
            # Sort k values
            sorted_k_squared = sorted(k_values_dict.keys())
            k_values = np.array([k_values_dict[k2] for k2 in sorted_k_squared])
            
            # Initialize gain matrix
            gain_matrix = np.zeros((len(k_values), len(omega_values)))
            
            # Compute gain for each (k, ω)
            for k_idx, k_squared in enumerate(sorted_k_squared):
                # Build Jacobian J(k)
                J = np.zeros((total_pops, total_pops))
                exp_factors = exp_cache[k_squared]
                
                for i in range(total_pops):
                    for j in range(total_pops):
                        w_tilde = network.A[i, j] * exp_factors[i, j]
                        if i == j:
                            J[i, j] = (-1.0 / network.tau[i] + 
                                      (analyzer.g_eff[i] * w_tilde) / network.tau[i])
                        else:
                            J[i, j] = (analyzer.g_eff[i] * w_tilde) / network.tau[i]
                
                # Compute B(k) with thalamic spatial filtering
                B_k = self.compute_B_fourier(network, k_squared, domain_length)
                
                # Check if B(k) is non-zero
                if np.linalg.norm(B_k) < 1e-10:
                    continue
                
                # For each temporal frequency ω
                for omega_idx, omega in enumerate(omega_values):
                    # Convert Hz to rad/s
                    omega_rad = 2 * np.pi * omega
                    
                    # Compute (iωI - J(k))
                    M = 1j * omega_rad * np.eye(total_pops) - J
                    
                    # Compute A(k,ω) = ||(iωI − J(k))^(−1) B(k)||
                    try:
                        M_inv_B = np.linalg.solve(M, B_k)
                        # Spectral norm (use norm for complex vectors)
                        gain = np.linalg.norm(M_inv_B)
                        gain_matrix[k_idx, omega_idx] = gain
                    except np.linalg.LinAlgError:
                        # Singular matrix - set to 0
                        gain_matrix[k_idx, omega_idx] = 0.0
            
            return k_values, omega_values, gain_matrix
            
        except Exception as e:
            print(f"Error computing spatiotemporal gain: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def create_stability_spectrum_figure(self, k_values: np.ndarray, max_real_values: np.ndarray) -> go.Figure:
        """
        Create Plotly figure for stability spectrum (max Re(λ) vs k).
        
        Args:
            k_values: Array of k values (wave numbers)
            max_real_values: Array of max real eigenvalues for each k
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        highlight_data = None

        # Check if we have data
        if len(k_values) > 0 and len(max_real_values) > 0:
            # Add main spectrum line
            fig.add_trace(go.Scatter(
                x=k_values,
                y=max_real_values,
                mode='lines',
                name='max Re(λ)',
                line=dict(color='#2c3e50', width=2),
                hovertemplate='k=%{x:.2f}<br>max Re(λ)=%{y:.3f}<extra></extra>',
                showlegend=False
            ))
            
            # Add horizontal line at y=0 (stability boundary)
            fig.add_trace(go.Scatter(
                x=[k_values.min(), k_values.max()],
                y=[0, 0],
                mode='lines',
                name='Stability boundary',
                line=dict(color='gray', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Determine y-axis range
            y_min = min(max_real_values.min(), -0.5)
            y_max = max(max_real_values.max(), 0.5)
            y_range = y_max - y_min
            y_padding = y_range * 0.1

            # Determine dominant mode (strict maximum)
            max_idx = int(np.argmax(max_real_values))
            max_value = max_real_values[max_idx]
            if np.sum(np.isclose(max_real_values, max_value)) == 1:
                highlight_k = k_values[max_idx]
                highlight_color = '#e74c3c' if max_value > 0 else '#7f8c8d'
                highlight_data = (highlight_k, max_value, highlight_color)
        else:
            # No data - show empty plot with message
            y_min, y_max, y_padding = -0.5, 0.5, 0
            fig.add_annotation(
                text="Network not yet active (run simulation to see spectrum)",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color='gray')
            )
        
        # Add highlight marker if applicable
        if highlight_data is not None:
            highlight_k, highlight_val, highlight_color = highlight_data
            fig.add_trace(go.Scatter(
                x=[highlight_k],
                y=[highlight_val],
                mode='markers',
                marker=dict(size=11, color=highlight_color, symbol='star', line=dict(color='#ffffff', width=1)),
                hovertemplate='Dominant k=%{x:.2f}<br>max Re(λ)=%{y:.3f}<extra></extra>',
                showlegend=False,
                cliponaxis=False
            ))

        # Configure layout with dashboard styling
        from src.analysis.bifurcation.config import ANALYSIS_PARAMS
        n_modes = ANALYSIS_PARAMS['n_modes']
        
        fig.update_layout(
            xaxis=dict(
                title='Spatial freq k',
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=False,
                range=[0, n_modes] if len(k_values) > 0 else [0, 10]
            ),
            yaxis=dict(
                title='max Re(λ)',
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1,
                range=[y_min - y_padding, y_max + y_padding] if len(k_values) > 0 else [-0.5, 0.5]
            ),
            margin=dict(l=45, r=20, t=12, b=35),
            height=200,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            showlegend=False
        )
        
        return fig
    
    def create_static_gain_figure(self, k_values: np.ndarray, gain_values: np.ndarray) -> go.Figure:
        """
        Create Plotly figure for static spatial gain G(k).
        
        Args:
            k_values: Array of k values (wave numbers)
            gain_values: Array of gain values for each k
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        highlight_data = None

        # Check if we have data
        if len(k_values) > 0 and len(gain_values) > 0:
            # Add main gain curve
            fig.add_trace(go.Scatter(
                x=k_values,
                y=gain_values,
                mode='lines',
                name='G(k)',
                line=dict(color='#2c3e50', width=2),
                hovertemplate='k=%{x:.2f}<br>Gain=%{y:.2f}<extra></extra>',
                showlegend=False
            ))
            
            # Determine y-axis range
            y_min = 0
            y_max = max(gain_values.max() * 1.1, 1.0)

            # Determine dominant gain mode (unique maximum)
            max_idx = int(np.argmax(gain_values))
            max_value = gain_values[max_idx]
            if np.sum(np.isclose(gain_values, max_value)) == 1:
                highlight_data = (k_values[max_idx], max_value)
        else:
            # No data - show empty plot with message
            y_min, y_max = 0, 10
            fig.add_annotation(
                text="Network not yet active (run simulation to see gain)",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color='gray')
            )

        # Add highlight marker if applicable
        if highlight_data is not None:
            highlight_k, highlight_val = highlight_data
            fig.add_trace(go.Scatter(
                x=[highlight_k],
                y=[highlight_val],
                mode='markers',
                marker=dict(size=11, color='#7f8c8d', symbol='star', line=dict(color='#ffffff', width=1)),
                hovertemplate='Dominant k=%{x:.2f}<br>Gain=%{y:.2f}<extra></extra>',
                showlegend=False,
                cliponaxis=False
            ))
        
        # Configure layout
        from src.analysis.bifurcation.config import ANALYSIS_PARAMS
        n_modes = ANALYSIS_PARAMS['n_modes']
        
        fig.update_layout(
            xaxis=dict(
                title=' k',
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=False,
                range=[0, n_modes] if len(k_values) > 0 else [0, 10]
            ),
            yaxis=dict(
                title='Gain G(k)',
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=False,
                range=[y_min, y_max]
            ),
            margin=dict(l=45, r=20, t=12, b=35),
            height=200,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            showlegend=False
        )
        
        return fig
    
    def create_spatiotemporal_gain_figure(self, k_values: np.ndarray, omega_values: np.ndarray, 
                                         gain_matrix: np.ndarray) -> go.Figure:
        """
        Create Plotly figure for spatiotemporal amplification map A(k,ω).
        
        Args:
            k_values: Array of spatial frequencies k
            omega_values: Array of temporal frequencies ω (Hz)
            gain_matrix: 2D array of gain values [k_idx, omega_idx]
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Check if we have data
        if len(k_values) > 0 and len(omega_values) > 0 and gain_matrix.size > 0:
            # Create heatmap
            fig.add_trace(go.Heatmap(
                x=k_values,
                y=omega_values,
                z=gain_matrix.T,  # Transpose so k is on x-axis
                colorscale='Hot',
                colorbar=dict(
                    title=dict(
                        text='Amplification',
                        side='right',
                        font=dict(size=12)
                    ),
                    len=1.0,
                    thickness=12
                ),
                hovertemplate='k=%{x:.2f}<br>ω=%{y:.2f} Hz<br>Gain=%{z:.2f}<extra></extra>'
            ))
        else:
            # No data - show empty plot with message
            fig.add_annotation(
                text="Network not yet active (run simulation to see amplification)",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color='gray')
            )
        
        # Configure layout
        from src.analysis.bifurcation.config import ANALYSIS_PARAMS
        n_modes = ANALYSIS_PARAMS['n_modes']
        
        fig.update_layout(
            xaxis=dict(
                title='Spatial freq k',
                showgrid=False,
                range=[0, n_modes] if len(k_values) > 0 else [0, 10]
            ),
            yaxis=dict(
                title='Temporal freq ω (Hz)',
                showgrid=False,
                range=[0, 1]
            ),
            margin=dict(l=45, r=20, t=12, b=35),
            height=200,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
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
    
    def _create_preset_buttons(self):
        """Create the preset buttons row."""
        return html.Div([
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
        ], className="mb-4")

    def _create_thalamus_visualization(self):
        """Create the thalamus visualization row."""
        return html.Div([
            dbc.Row([
                # Thalamus label
                dbc.Col([
                    html.Div([
                        html.H6("Th",
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
                            config=GRAPH_CONFIG
                        )
                    ], style={"display": "flex", "justifyContent": "center"})
                ], width=10)
            ], className="align-items-center")
        ], className="mt-2")

    def _create_activity_visualization(self):
        """Create the activity visualization section."""
        return dbc.Col([
            # Add more top padding to shift visualization down
            html.Div(style={"height": "20px"}),
            
            # Preset Buttons
            self._create_preset_buttons(),
            
            # Layer visualizations
            *[self.create_layer_row(layer) for layer in LAYERS],
            
            # Thalamus visualization
            self._create_thalamus_visualization()
        ], width=4, className="px-4")

    def _create_connectivity_matrix(self):
        """Create the connectivity matrix section."""
        return dbc.Col([
            # Connection Strength Matrix
            html.Div([
                html.H5("Connection Strengths", 
                       className="mb-3 text-center",
                       style={
                           "textAlign": "center",
                           "width": "85%",  # Match matrix container width
                           "margin": "0 auto",  # Center the title
                           "paddingLeft": "50px"  # Match matrix container padding
                       }),
                
                # Connection Matrix Container
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
                    style={"display": "none", **SLIDER_CONTAINER_STYLE}
                )
            ], className="mb-3"),
            
            # Stability Spectrum Graph
            html.Div([
                html.H5("Stability Analysis", 
                       className="mb-3 text-center",
                       style={
                           "textAlign": "center",
                           "width": "85%",
                           "margin": "0 auto"
                       }),
                dcc.Graph(
                    id='stability-spectrum-graph',
                    figure=self.create_stability_spectrum_figure(np.array([]), np.array([])),
                    config=GRAPH_CONFIG,
                    style={"width": "85%", "margin": "0 auto", "height": "210px"}
                )
            ], className="mt-4"),
            
            # Forced Response Analysis
            html.Div([
                html.H5("Forced Response Analysis", 
                       className="mb-3 text-center",
                       style={
                           "textAlign": "center",
                           "width": "85%",
                           "margin": "0 auto"
                       }),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='static-gain-graph',
                            figure=self.create_static_gain_figure(np.array([]), np.array([])),
                            config=GRAPH_CONFIG,
                            style={"width": "95%", "margin": "0 auto", "height": "210px"}
                        )
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(
                            id='spatiotemporal-gain-graph',
                            figure=self.create_spatiotemporal_gain_figure(np.array([]), np.array([]), np.array([])),
                            config=GRAPH_CONFIG,
                            style={"width": "95%", "margin": "0 auto", "height": "210px"}
                        )
                    ], width=6)
                ], style={"width": "95%", "margin": "0 auto"})
            ], className="mt-4")
        ], width=4, className="px-5")

    def setup_layout(self):
        """Set up the dashboard layout."""
        # Add interval component for updates
        interval = dcc.Interval(
            id='interval-component',
            interval=self.update_interval,
            n_intervals=0,
            disabled=False
        )

        # Add slower interval for stability spectrum updates
        spectrum_interval = dcc.Interval(
            id='spectrum-interval',
            interval=200,  # How often to update spectrum
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
        
        self.app.layout = dbc.Container([
            # Utility components
            interval,
            spectrum_interval,
            selected_cell,
            reset_btn,
            
            # Main content: three columns
            dbc.Row([
                # Left column: activity visualization
                self._create_activity_visualization(),
                
                # Middle column: connectivity matrix
                self._create_connectivity_matrix(),
                    
                # Right column: Control panel
                dbc.Col([
                    # Container for control sliders
                    html.Div(
                        self.create_control_panel(),
                        className="control-panel-column",
                        style=CONTROL_PANEL_STYLE
                    )
                ], width=4, className="px-4")
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
                                config=GRAPH_CONFIG
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
        colorscale = COLORMAPS.get(cell_type, [[0, 'black'], [1, 'white']])
        
        return go.Figure(
            data=[go.Heatmap(
                z=data,
                colorscale=colorscale,
                showscale=False,
                hoverinfo='none',  # Disable hover info for performance
                zmin=HEATMAP_ZMIN,
                zmax=HEATMAP_ZMAX
            )],
            layout=GRAPH_LAYOUT
        )
    
    def _parse_connection_key(self, conn_key):
        """Parse a connection key into source and target components.
        
        Args:
            conn_key: String in format 'source_to_target' where source and target 
                     can be either 'thalamus' or 'layer_celltype'
                     
        Returns:
            Tuple of (source_layer, source_cell, target_layer, target_cell)
        """
        return ConnectionKeyUtils.parse(conn_key)

    def _apply_preset(self, preset):
        """Apply a preset configuration to the simulation."""
        # Update all connection strengths
        for conn_key, strength in preset['connection_strengths'].items():
            # Parse the connection key and update strength
            source_layer, source_cell, target_layer, target_cell = self._parse_connection_key(conn_key)
            self.simulation.connectivity.set_connection_strength(
                source_layer, source_cell, target_layer, target_cell, strength
            )
        
        # Update strength scaling factors if present in the preset
        if 'strength_scaling' in preset:
            for cell_type, scaling in preset['strength_scaling'].items():
                self.simulation.set_strength_scaling(cell_type, scaling)
        
        # Update gains if present in the preset
        if 'gains' in preset:
            for cell_type, gain in preset['gains'].items():
                self.simulation.set_gain(cell_type, gain)
        
        # Update background input if present in the preset
        if 'background_input' in preset:
            for cell_type, value in preset['background_input'].items():
                self.simulation.set_background_input(cell_type, value)
                
    def get_connection_key(self, source_layer, source_cell, target_layer, target_cell):
        """Generate a connection key based on source and target information."""
        return ConnectionKeyUtils.build(source_layer, source_cell, target_layer, target_cell)

    def get_max_scaled_strength_magnitude(self):
        """Get the maximum absolute scaled connection strength across all presets.
        
        This is used to normalize colors across developmental stages for fair comparison.
        
        Returns:
            Maximum absolute scaled strength value
        """
        max_magnitude = 0.0
        
        # Check all developmental presets
        for preset in [P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET]:
            for conn_key, raw_strength in preset['connection_strengths'].items():
                # Parse to get source cell type
                parts = conn_key.split('_to_')
                source_part = parts[0]
                
                if source_part == 'thalamus':
                    scaling = preset['strength_scaling'].get('thalamus', 1.0)
                else:
                    source_cell = source_part.split('_')[1]
                    scaling = preset['strength_scaling'].get(source_cell, 1.0)
                
                scaled_strength = raw_strength * scaling
                max_magnitude = max(max_magnitude, abs(scaled_strength))
        
        return max_magnitude

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
                        "color": "#2c3e50",
                        "padding": "10px 5px",
                        "fontSize": "0.9rem",
                        "fontWeight": "600"
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
                            "color": "#2c3e50",
                            "padding": "8px 5px",
                            "fontSize": "0.9rem",
                            "fontWeight": "500",
                            "borderRight": "1px solid #ddd" if cell_type == "PV" else "none"
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
                        "color": "#2c3e50",
                        "textAlign": "center",
                        "verticalAlign": "middle",
                        "padding": "10px 5px",
                        "height": "100%",
                        "fontSize": "0.9rem",
                        "fontWeight": "600"
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
                    "color": "#2c3e50",
                    "padding": "5px",
                    "fontSize": "0.9rem",
                    "fontWeight": "500"
                }
            )
            
            # Create data cells
            cells = []
            # Get max magnitude for normalization across all presets
            max_magnitude = self.get_max_scaled_strength_magnitude()
            
            for target_layer in LAYERS:
                for target_cell in CELL_TYPES:
                    # Skip thalamus to thalamus connections
                    if source_layer == 'Th' and target_layer == 'Th':
                        cells.append(html.Td("", className="text-center", style={
                            **CELL_STYLE,
                            "backgroundColor": "#f8f9fa"
                        }))
                        continue
                    
                    # Get scaled connection strength for display
                    value = self.get_connection_value(source_layer, source_cell, target_layer, target_cell, scaled=True)
                    
                    # Determine cell colors based on scaled connection strength and source cell type
                    bg_color, hover_color = self._get_connection_colors(source_layer, source_cell, value, max_magnitude)
                    
                    # Create cell with unique ID for callbacks
                    cell_id = f"{source_layer}-{source_cell or 'None'}-{target_layer}-{target_cell}"
                    cells.append(html.Td(
                        f"{value:.2f}",  # Show 2 decimals for scaled values
                        id={'type': 'connection-cell', 'id': cell_id},
                        className="connection-cell text-center",
                        style={
                            **CELL_STYLE,
                            "backgroundColor": bg_color,
                            "cursor": "pointer",
                            "transition": "background-color 0.2s",
                            "padding": "5px",
                            "fontSize": "0.8rem",
                            "borderRight": "1px solid #ddd" if target_cell == "PV" else "none",
                            "color": "#2c3e50"
                        },
                        **{'data-highlight-color': hover_color}
                    ))
            
            # Create row with header (if needed) and cells
            is_last_in_layer = (source_layer != 'Th' and source_cell == 'PV') or source_layer == 'Th'
            row_style = {"marginLeft": "0", "marginRight": "0"}
            if is_last_in_layer:
                row_style["borderBottom"] = "1px solid #ddd"
            
            row_cells = [cell for cell in [row_header, cell_type_header] + cells if cell is not None]
            rows.append(html.Tr(row_cells, style=row_style))
        
        # Create table with colorbar
        # Use symmetric range based on max magnitude across all presets
        colorbar_max = max_magnitude  # Already computed above for color normalization
        
        return html.Div([
            html.Div([
                # Connection matrix table
                html.Div(
                    html.Table(
                        [html.Tr(main_header_cells), html.Tr(sub_header_cells)] + rows,
                        className="table connection-matrix",
                        style={
                            "tableLayout": "fixed",
                            "fontSize": "0.8rem",
                            "borderCollapse": "collapse",
                            "width": "auto",
                            "margin": "0",
                            "borderSpacing": "0",
                            "border": "none"
                        }
                    ),
                    style={"display": "flex", "flexDirection": "column"}
                ),
                # Colorbar
                html.Div([
                    # Top label (max)
                    html.Div(
                        f"+{colorbar_max:.1f}",
                        style={
                            "fontSize": "0.7rem",
                            "textAlign": "center",
                            "marginBottom": "2px",
                            "color": "#2c3e50",
                            "fontWeight": "500"
                        }
                    ),
                    # Gradient bar (excitatory to inhibitory)
                    html.Div(
                        style={
                            "width": "25px",
                            "flex": "1",
                            "background": "linear-gradient(to bottom, #4292c2, rgba(200, 200, 200, 0.2), #D91B12)",
                            "border": "1px solid #ddd",
                            "borderRadius": "3px",
                            "position": "relative"
                        },
                        children=[
                            # Zero label (positioned in middle)
                            html.Div(
                                "0.0",
                                style={
                                    "fontSize": "0.7rem",
                                    "textAlign": "center",
                                    "color": "#2c3e50",
                                    "fontWeight": "500",
                                    "position": "absolute",
                                    "top": "50%",
                                    "left": "50%",
                                    "transform": "translate(-50%, -50%)",
                                    "backgroundColor": "white",
                                    "padding": "0 2px"
                                }
                            )
                        ]
                    ),
                    # Bottom label (min)
                    html.Div(
                        f"-{colorbar_max:.1f}",
                        style={
                            "fontSize": "0.7rem",
                            "textAlign": "center",
                            "marginTop": "2px",
                            "color": "#2c3e50",
                            "fontWeight": "500"
                        }
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "marginLeft": "15px",
                    "alignSelf": "stretch"
                })
            ], style={
                "display": "flex",
                "alignItems": "stretch",
                "justifyContent": "center"
            })
        ])

    def _get_connection_colors(self, source_layer, source_cell, value, max_magnitude=None):
        """Get background and hover colors for a connection based on source and value.
        
        Args:
            source_layer: Source layer ('thalamus' or layer name)
            source_cell: Source cell type (E, SST, PV, or None for thalamus)
            value: Connection strength value (scaled)
            max_magnitude: Maximum absolute value for normalization (if None, uses 1.0)
            
        Returns:
            Tuple of (background_color, hover_color)
        """
        if max_magnitude is None or max_magnitude == 0:
            max_magnitude = 1.0
            
        if source_layer == 'thalamus' or source_layer == 'Th':
            # For thalamic connections, always use E color and only positive values
            if value > 0:
                intensity = min(abs(value) / max_magnitude, 1.0) * 0.7
                bg_color = CELL_ACTIVITY_COLORS['E']['bg'](intensity)
                hover_color = CELL_ACTIVITY_COLORS['E']['hover'](intensity)
            else:
                bg_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                hover_color = CELL_ACTIVITY_COLORS['inactive']['hover']
        else:
            # For cell-type specific connections
            if value != 0:
                intensity = min(abs(value) / max_magnitude, 1.0) * 0.7
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
                hover_color = CELL_ACTIVITY_COLORS['inactive']['bg']
                
        return bg_color, hover_color

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

    def _get_preset_values(self, preset):
        """Helper function to get values from a preset object or dictionary."""
        values = {
            'tau_e': preset['time_constants']['E'],
            'tau_sst': preset['time_constants']['SST'],
            'tau_pv': preset['time_constants']['PV'],
            'background_input_e': preset['background_input']['E'],
            'background_input_sst': preset['background_input']['SST'],
            'background_input_pv': preset['background_input']['PV'],
            'sigma_thal_e': preset['thalamic_widths']['E'],
            'sigma_thal_sst': preset['thalamic_widths']['SST'],
            'sigma_thal_pv': preset['thalamic_widths']['PV'],
            'sigma_e_out': preset['outgoing_widths']['E'],
            'sigma_sst_out': preset['outgoing_widths']['SST'],
            'sigma_pv_out': preset['outgoing_widths']['PV'],
            'strength_e': preset['strength_scaling']['E'],
            'strength_sst': preset['strength_scaling']['SST'],
            'strength_pv': preset['strength_scaling']['PV'],
            'strength_thal': preset['strength_scaling']['thalamus'],
            'alpha': preset['thalamic_alpha']
        }
        return values

    def _create_preset_callback(self, preset_name, preset_obj, allow_duplicate=False):
        """Helper function to create a preset callback."""
        outputs = [
            Output(id, prop, allow_duplicate=allow_duplicate) 
            for id, prop in [(o.component_id, o.component_property) for o in self._PRESET_OUTPUTS]
        ] if allow_duplicate else self._PRESET_OUTPUTS

        @self.app.callback(
            outputs,
            Input(f'{preset_name}-preset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def apply_preset_callback(n_clicks):  # pylint: disable=unused-argument
            """Apply the preset configuration."""
            # Apply the preset using the generic apply_preset function
            self._apply_preset(preset_obj)
            
            # Get values from the preset
            values = self._get_preset_values(preset_obj)
            
            # Return all values in the expected order
            return (
                values['tau_e'], values['tau_sst'], values['tau_pv'],
                values['background_input_e'], values['background_input_sst'], values['background_input_pv'],
                values['sigma_thal_e'], values['sigma_thal_sst'], values['sigma_thal_pv'],
                values['sigma_e_out'], values['sigma_sst_out'], values['sigma_pv_out'],
                values['strength_e'], values['strength_sst'], values['strength_pv'], values['strength_thal'],
                values['alpha'], self.create_connection_matrix()
            )
        
        return apply_preset_callback

    def get_connection_value(self, source_layer, source_cell, target_layer, target_cell, scaled=False):
        """Get the current connection strength value.
        
        Args:
            source_layer: Source layer
            source_cell: Source cell type
            target_layer: Target layer
            target_cell: Target cell type
            scaled: If True, return strength-scaled value; if False, return raw amplitude
            
        Returns:
            Connection strength (raw or scaled)
        """
        try:
            # Get connection strength
            conn_key = self.get_connection_key(source_layer, source_cell, target_layer, target_cell)
            
            # First try to get the value from the simulation connectivity
            if hasattr(self, 'simulation') and hasattr(self.simulation, 'connectivity'):
                # Convert 'Th' to 'thalamus' for the simulation API
                source_layer_sim = 'thalamus' if source_layer == 'Th' else source_layer
                
                if scaled:
                    return self.simulation.connectivity.get_scaled_connection_strength(
                        source_layer_sim, source_cell, target_layer, target_cell
                    )
                else:
                    return self.simulation.connectivity.get_connection_strength(
                        source_layer_sim, source_cell, target_layer, target_cell
                    )
                
            # Fall back to config-based lookup
            if conn_key in LAYER_CONNECTIVITY_PARAMS:
                raw_value = LAYER_CONNECTIVITY_PARAMS[conn_key]['amplitude']
                if scaled:
                    # Apply strength scaling from config
                    if source_layer == 'Th' or source_layer == 'thalamus':
                        scaling = INITIAL_STRENGTH_SCALING.get('thalamus', 1.0)
                    else:
                        scaling = INITIAL_STRENGTH_SCALING.get(source_cell, 1.0)
                    return raw_value * scaling
                return raw_value
                
            # Default to 0 if not found
            return 0.0
        except (AttributeError, KeyError) as e:
            print(f"Error getting connection from simulation: {str(e)}")
            return 0.0

    def setup_callbacks(self):
        """Set up the dashboard callbacks for interactivity."""
        # Add callbacks for preset buttons
        self._create_preset_callback('p4', P4_PRESET)
        self._create_preset_callback('p8', P8_PRESET, allow_duplicate=True)
        self._create_preset_callback('p12', P12_PRESET, allow_duplicate=True)
        self._create_preset_callback('p16', P16_PRESET, allow_duplicate=True)
        
        # Initialize slider container (hidden)
        @self.app.callback(
            [Output('slider-container', 'style'),
             Output('slider-container', 'children'),
             Output('selected-cell', 'data')],
            [Input('connection-matrix-container', 'children')],
            [State('selected-cell', 'data')]
        )
        def initialize_slider_container(_, current_data):  # pylint: disable=unused-argument
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
        def handle_cell_click(clicks, current_data):  # pylint: disable=unused-argument
            """Show the connection strength slider when a matrix cell is clicked."""
            try:
                # Get the context that triggered the callback
                ctx = dash.callback_context
                if not ctx.triggered:
                    return dash.no_update, dash.no_update, dash.no_update
                
                # Get the ID of the clicked cell
                triggered_prop_id = ctx.triggered[0]['prop_id']
                
                # Check if this is actually a click (not just matrix recreation)
                # If the value is None or the prop_id doesn't contain valid data, ignore it
                triggered_value = ctx.triggered[0]['value']
                if triggered_value is None or triggered_value == 0:
                    return dash.no_update, dash.no_update, dash.no_update
                
                cell_data = json.loads(triggered_prop_id.split('.')[0])
                clicked_id = cell_data['id']
                
                # Extract connection info from the ID
                parts = clicked_id.split('-')
                if len(parts) < 4:
                    print(f"Invalid cell ID format: {clicked_id}")
                    return dash.no_update, dash.no_update, dash.no_update
                
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
                    'backgroundColor': 'rgba(255, 255, 255, 0.95)',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'borderRadius': '5px',
                    'zIndex': '1000',
                    'width': '200px',
                    'position': 'absolute',
                    'top': '0px',
                    'left': '0px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.15)',
                    'color': '#2c3e50'
                }, slider, connection_data
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error handling cell click: {str(e)}")
                return dash.no_update, dash.no_update, dash.no_update
        
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
            except (KeyError, AttributeError, ValueError) as e:
                print(f"Error updating connection value: {str(e)}")
                return f"Error: {str(e)}"
        
        # Update connection cell in matrix when slider changes
        @self.app.callback(
            [Output({'type': 'connection-cell', 'id': MATCH}, 'children'),
             Output({'type': 'connection-cell', 'id': MATCH}, 'style'),
             Output({'type': 'connection-cell', 'id': MATCH}, 'data-highlight-color')],
            Input({'type': 'matrix-slider', 'id': MATCH}, 'value'),
            [State({'type': 'connection-cell', 'id': MATCH}, 'style'),
             State({'type': 'connection-cell', 'id': MATCH}, 'id'),
             State('selected-cell', 'data')]
        )
        def update_matrix_cell(raw_value, current_style, cell_id, connection_data):  # pylint: disable=unused-argument
            """Update the matrix cell appearance and value when the slider changes."""
            if raw_value is None:
                # No change if value is None
                return dash.no_update, dash.no_update, dash.no_update
            
            try:
                # Parse cell ID from the dictionary
                cell_id_str = cell_id['id']  # Extract the ID string from the dictionary
                parts = cell_id_str.split('-')
                source_layer = parts[0]
                source_cell = parts[1] if parts[1] != 'None' else None
                target_layer = parts[2]
                target_cell = parts[3]
                
                # Convert to thalamus if needed
                source_layer_sim = 'thalamus' if source_layer == 'Th' else source_layer
                
                # Get the scaled value to display (raw_value * strength_scaling)
                if hasattr(self, 'simulation') and hasattr(self.simulation, 'connectivity'):
                    scaled_value = self.simulation.connectivity.get_scaled_connection_strength(
                        source_layer_sim, source_cell, target_layer, target_cell
                    )
                else:
                    # Fallback: compute scaled value manually
                    if source_layer_sim == 'thalamus':
                        scaling = INITIAL_STRENGTH_SCALING.get('thalamus', 1.0)
                    else:
                        scaling = INITIAL_STRENGTH_SCALING.get(source_cell, 1.0)
                    scaled_value = raw_value * scaling
                
                # Get max magnitude for color normalization
                max_magnitude = self.get_max_scaled_strength_magnitude()
                
                # Determine cell colors based on scaled connection strength and source cell type
                bg_color, hover_color = self._get_connection_colors(source_layer, source_cell, scaled_value, max_magnitude)
                
                # Update style with new background color while preserving other styles
                updated_style = {
                    **CELL_STYLE,
                    "backgroundColor": bg_color,
                    "cursor": "pointer",
                    "transition": "background-color 0.2s",
                    "padding": "5px",
                    "fontSize": "0.8rem",
                    "borderRight": "1px solid #ddd" if target_cell == "PV" else "none",
                    "color": "#2c3e50"
                }
                
                # Return updated text (scaled value), style, and hover color
                return f"{scaled_value:.2f}", updated_style, hover_color
            except (KeyError, ValueError) as e:
                print(f"Error updating matrix cell: {str(e)}")
                return dash.no_update, dash.no_update, dash.no_update
        
        # Reset the slider when clicking the reset button
        @self.app.callback(
            [Output('slider-container', 'style', allow_duplicate=True),
             Output('slider-container', 'children', allow_duplicate=True),
             Output('selected-cell', 'data', allow_duplicate=True)],
            Input('reset-slider-state-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def reset_slider_state(n_clicks):  # pylint: disable=unused-argument
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
            
            # Inputs: interval trigger, alpha slider, time constant sliders, connectivity width sliders
            [Input('interval-component', 'n_intervals'),
             Input('alpha-slider', 'value'),
             Input('tau-e-slider', 'value'),
             Input('tau-sst-slider', 'value'),
             Input('tau-pv-slider', 'value'),
             Input('thalamic-width-e-slider', 'value'),
             Input('thalamic-width-sst-slider', 'value'),
             Input('thalamic-width-pv-slider', 'value'),
             Input('outgoing-width-e-slider', 'value'),
             Input('outgoing-width-sst-slider', 'value'),
             Input('outgoing-width-pv-slider', 'value')],
            
            # States: pause button state
            [State('pause-button', 'n_clicks')]
        )
        def update_graphs(n_intervals, alpha, tau_e, tau_sst, tau_pv, # pylint: disable=unused-argument
                         sigma_thal_e, sigma_thal_sst, sigma_thal_pv, 
                         sigma_e_out, sigma_sst_out, sigma_pv_out, pause_clicks):  
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
            # Note: gains are set by presets, not sliders
            
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
                        
                        # Keep consistent scaling for fair comparison across heatmaps
                        fig.update_traces(zmin=HEATMAP_ZMIN, zmax=HEATMAP_ZMAX)
                        
                    updated_figures.append(fig)
            
            # Update thalamus figure
            thal_fig = self.figures['graph-thalamus']
            with thal_fig.batch_update():
                thal_data = activities['thalamus']
                thal_fig.data[0]['z'] = thal_data
                # Keep consistent scaling for fair comparison with cortical heatmaps
                thal_fig.update_traces(zmin=HEATMAP_ZMIN, zmax=HEATMAP_ZMAX)
            
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
            [Output('interval-component', 'n_intervals', allow_duplicate=True),
             Output('connection-matrix-container', 'children', allow_duplicate=True)],
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
            
            # Regenerate the connection matrix with updated scaled values
            updated_matrix = self.create_connection_matrix()
            
            # Return unchanged intervals and updated matrix
            return [n_intervals, updated_matrix]
        
        # Add callback for updating background input parameters
        @self.app.callback(
            [Output('interval-component', 'n_intervals', allow_duplicate=True)],
            [Input('background-input-e-slider', 'value'),
             Input('background-input-sst-slider', 'value'),
             Input('background-input-pv-slider', 'value')],
            [State('interval-component', 'n_intervals')],
            prevent_initial_call=True
        )
        def update_background_input_parameters(bg_e, bg_sst, bg_pv, n_intervals):
            """Update all background input parameters in the simulation."""
            # Update background input for each cell type
            self.simulation.set_background_input('E', bg_e)
            self.simulation.set_background_input('SST', bg_sst)
            self.simulation.set_background_input('PV', bg_pv)
            
            # Return unchanged intervals to not disrupt the update loop
            return [n_intervals]
        
        # Update stability spectrum when parameters change or periodically
        @self.app.callback(
            Output('stability-spectrum-graph', 'figure'),
            [Input('spectrum-interval', 'n_intervals'),
             Input('tau-e-slider', 'value'),
             Input('tau-sst-slider', 'value'),
             Input('tau-pv-slider', 'value'),
             Input('background-input-e-slider', 'value'),
             Input('background-input-sst-slider', 'value'),
             Input('background-input-pv-slider', 'value'),
             Input('thalamic-width-e-slider', 'value'),
             Input('thalamic-width-sst-slider', 'value'),
             Input('thalamic-width-pv-slider', 'value'),
             Input('outgoing-width-e-slider', 'value'),
             Input('outgoing-width-sst-slider', 'value'),
             Input('outgoing-width-pv-slider', 'value'),
             Input('strength-scaling-e-slider', 'value'),
             Input('strength-scaling-sst-slider', 'value'),
             Input('strength-scaling-pv-slider', 'value'),
             Input('strength-scaling-thalamus-slider', 'value'),
             Input({'type': 'matrix-slider', 'id': ALL}, 'value')]
        )
        def update_stability_spectrum(*args):  # pylint: disable=unused-argument
            """Update the stability spectrum graph when parameters change or periodically."""
            try:
                # Build preset from current simulation state
                preset = self.build_current_preset()
                
                # Compute stability spectrum
                k_values, max_real_values = self.compute_stability_spectrum(preset)
                
                # Create and return figure
                return self.create_stability_spectrum_figure(k_values, max_real_values)
                
            except Exception as e:
                print(f"Error updating stability spectrum: {e}")
                # Return empty figure on error
                return self.create_stability_spectrum_figure(np.array([]), np.array([]))
        
        # Update forced response graphs when parameters change or periodically
        @self.app.callback(
            [Output('static-gain-graph', 'figure'),
             Output('spatiotemporal-gain-graph', 'figure')],
            [Input('spectrum-interval', 'n_intervals'),
             Input('tau-e-slider', 'value'),
             Input('tau-sst-slider', 'value'),
             Input('tau-pv-slider', 'value'),
             Input('background-input-e-slider', 'value'),
             Input('background-input-sst-slider', 'value'),
             Input('background-input-pv-slider', 'value'),
             Input('thalamic-width-e-slider', 'value'),
             Input('thalamic-width-sst-slider', 'value'),
             Input('thalamic-width-pv-slider', 'value'),
             Input('outgoing-width-e-slider', 'value'),
             Input('outgoing-width-sst-slider', 'value'),
             Input('outgoing-width-pv-slider', 'value'),
             Input('strength-scaling-e-slider', 'value'),
             Input('strength-scaling-sst-slider', 'value'),
             Input('strength-scaling-pv-slider', 'value'),
             Input('strength-scaling-thalamus-slider', 'value'),
             Input({'type': 'matrix-slider', 'id': ALL}, 'value')]
        )
        def update_forced_response(*args):  # pylint: disable=unused-argument
            """Update the forced response graphs when parameters change or periodically."""
            try:
                # Build preset from current simulation state
                preset = self.build_current_preset()
                
                # Compute static gain
                k_values_static, gain_values = self.compute_static_gain(preset)
                static_fig = self.create_static_gain_figure(k_values_static, gain_values)
                
                # Compute spatiotemporal gain
                k_values_st, omega_values, gain_matrix = self.compute_spatiotemporal_gain(preset)
                spatiotemporal_fig = self.create_spatiotemporal_gain_figure(k_values_st, omega_values, gain_matrix)
                
                return static_fig, spatiotemporal_fig
                
            except Exception as e:
                print(f"Error updating forced response: {e}")
                # Return empty figures on error
                empty_static = self.create_static_gain_figure(np.array([]), np.array([]))
                empty_st = self.create_spatiotemporal_gain_figure(np.array([]), np.array([]), np.array([]))
                return empty_static, empty_st
    
    def create_parameter_sliders(self):
        """Create the neural parameter sliders section."""
        return html.Div([
            # Headers row
            dbc.Row([
                dbc.Col("", width=1),
                dbc.Col(html.Div("Time Constant", className="text-center"), width=5),
                dbc.Col(html.Div("Background Input", className="text-center"), width=5),
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
                initial_value=INITIAL_TIME_CONSTANTS[cell_type],
                **self.TIME_CONSTANT_PARAMS
            ), width=5, style={"paddingRight": "5px"}),
            # Background input slider
            dbc.Col(self._create_slider(
                id_prefix='background-input',
                cell_type=cell_type,
                initial_value=INITIAL_BACKGROUND_INPUT[cell_type],
                **self.BACKGROUND_INPUT_PARAMS
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
                initial_value=INITIAL_THALAMIC_WIDTHS[cell_type],
                **self.WIDTH_PARAMS
            ), width=5, style={"paddingRight": "5px"}),
            # Outgoing width slider
            dbc.Col(self._create_slider(
                id_prefix='outgoing-width',
                cell_type=cell_type,
                initial_value=INITIAL_OUTGOING_WIDTHS[cell_type],
                **self.WIDTH_PARAMS
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
                initial_value=INITIAL_STRENGTH_SCALING[cell_type],
                **self.STRENGTH_SCALING_PARAMS
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
            **self.SLIDER_STYLE
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
            # Section: Neuron parameters
            html.Div([
                html.H5("Neuron Parameters", className="text-center"),
                self.create_parameter_sliders()
            ], className="mb-3"),
            
            # Section: Connectivity widths
            html.Div([
                html.H5("Connection Widths", className="text-center"),
                self.create_connectivity_sliders(),
                
                # Section for strength scaling
                html.Div([html.Hr()], className="my-3"),
                html.H5("Strength Scaling", className="text-center"),
                self.create_strength_scaling_sliders(),
                
                # Section for thalamic input balance
                html.Div([html.Hr()], className="my-3"),
                html.H5("Thalamic Input", className="text-center"),
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