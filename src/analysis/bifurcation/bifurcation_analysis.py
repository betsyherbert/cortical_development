"""Core bifurcation analysis implementation.

This module implements linear stability analysis for network models,
computing steady states and determining network stability through eigenvalue analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from .config import ANALYSIS_PARAMS


class NetworkModel:
    """Handles network dynamics for single or multiple layers, supporting both 2-pop (E-I) and 3-pop (E-SST-PV) models."""
    
    def __init__(self, preset: Dict, layers: list = None, n_populations: int = 3):
        """
        Initialize network from preset.
        
        Args:
            preset: Developmental preset dictionary (e.g., P4_PRESET)
            layers: List of layers to analyze (e.g., ['L4'] or ['L23', 'L4', 'L5'])
                   Defaults to ['L4'] for backward compatibility
            n_populations: Number of populations (2 for E-I, 3 for E-SST-PV)
        """
        # Handle backward compatibility: convert single layer string to list
        if layers is None:
            layers = ['L4']
        elif isinstance(layers, str):
            layers = [layers]
        
        self.layers = layers
        self.preset = preset
        self.n_populations = n_populations
        
        if n_populations == 2:
            self.pop_names = ['E', 'I']
        elif n_populations == 3:
            self.pop_names = ['E', 'SST', 'PV']
        else:
            raise ValueError(f"n_populations must be 2 or 3, got {n_populations}")
        
        # Extract parameters
        self._extract_time_constants()
        self._extract_gains()
        self._extract_connection_strengths()
        self._extract_spatial_scales()
        
        # Baseline input: noise mean provides constant baseline input in the simulation
        # For spatially uniform steady state, we use the noise mean as mu
        self._extract_baseline_input()
    
    def _extract_time_constants(self):
        """Extract time constants for each population, repeated for each layer."""
        tau_e = self.preset['time_constants']['E']
        
        if self.n_populations == 2:
            # Collapse SST and PV into generic I
            tau_sst = self.preset['time_constants']['SST']
            tau_pv = self.preset['time_constants']['PV']
            tau_i = (tau_sst + tau_pv) / 2.0
            tau_per_layer = np.array([tau_e, tau_i])
        else:
            tau_sst = self.preset['time_constants']['SST']
            tau_pv = self.preset['time_constants']['PV']
            tau_per_layer = np.array([tau_e, tau_sst, tau_pv])
        
        # Repeat for each layer
        self.tau = np.tile(tau_per_layer, len(self.layers))
    
    def _extract_gains(self):
        """Extract gains for each population, repeated for each layer."""
        g_e = self.preset['gains']['E']
        
        if self.n_populations == 2:
            # Collapse SST and PV into generic I
            g_sst = self.preset['gains']['SST']
            g_pv = self.preset['gains']['PV']
            g_i = (g_sst + g_pv) / 2.0
            gain_per_layer = np.array([g_e, g_i])
        else:
            g_sst = self.preset['gains']['SST']
            g_pv = self.preset['gains']['PV']
            gain_per_layer = np.array([g_e, g_sst, g_pv])
        
        # Repeat for each layer
        self.gain = np.tile(gain_per_layer, len(self.layers))
    
    def _extract_connection_strengths(self):
        """Extract connection strengths for all layer pairs, building block-structured matrix."""
        scaling_e = self.preset['strength_scaling']['E']
        scaling_sst = self.preset['strength_scaling']['SST']
        scaling_pv = self.preset['strength_scaling']['PV']
        
        n_layers = len(self.layers)
        n = self.n_populations
        total_pops = n_layers * n
        
        # Initialize full connection matrix
        self.A = np.zeros((total_pops, total_pops))
        
        # Build block for each layer pair
        for i_tgt, layer_tgt in enumerate(self.layers):
            for i_src, layer_src in enumerate(self.layers):
                # Extract connection block for this layer pair
                block = self._extract_layer_pair_block(layer_src, layer_tgt, scaling_e, scaling_sst, scaling_pv)
                
                # Insert block into full matrix
                row_start = i_tgt * n
                row_end = row_start + n
                col_start = i_src * n
                col_end = col_start + n
                self.A[row_start:row_end, col_start:col_end] = block
    
    def _extract_layer_pair_block(self, layer_src: str, layer_tgt: str, 
                                   scaling_e: float, scaling_sst: float, scaling_pv: float) -> np.ndarray:
        """Extract connection strength block for one layer pair."""
        def get_raw(source, target):
            key = f'{layer_src}_{source}_to_{layer_tgt}_{target}'
            return self.preset['connection_strengths'].get(key, 0.0)
        
        if self.n_populations == 2:
            # Collapse to E-I
            raw_ee = get_raw('E', 'E')
            raw_esst = get_raw('E', 'SST')
            raw_epv = get_raw('E', 'PV')
            raw_sste = get_raw('SST', 'E')
            raw_pve = get_raw('PV', 'E')
            raw_sstpv = get_raw('SST', 'PV')
            raw_pvsst = get_raw('PV', 'SST')
            raw_pvpv = get_raw('PV', 'PV')
            
            # Apply scaling and collapse
            A_ee = raw_ee * scaling_e
            A_ei = (raw_esst * scaling_e + raw_epv * scaling_e) / 2.0
            A_ie = (raw_sste * scaling_sst + raw_pve * scaling_pv) / 2.0
            A_ii = (raw_sstpv * scaling_sst + raw_pvsst * scaling_pv + raw_pvpv * scaling_pv) / 3.0
            
            return np.array([[A_ee, A_ei], [A_ie, A_ii]])
        else:
            # Full 3-population model
            A_ee = get_raw('E', 'E') * scaling_e
            A_esst = get_raw('E', 'SST') * scaling_e
            A_epv = get_raw('E', 'PV') * scaling_e
            A_sste = get_raw('SST', 'E') * scaling_sst
            A_sstsst = 0.0  # SST doesn't connect to SST
            A_sstpv = get_raw('SST', 'PV') * scaling_sst
            A_pve = get_raw('PV', 'E') * scaling_pv
            A_pvsst = get_raw('PV', 'SST') * scaling_pv
            A_pvpv = get_raw('PV', 'PV') * scaling_pv
            
            return np.array([
                [A_ee, A_esst, A_epv],
                [A_sste, A_sstsst, A_sstpv],
                [A_pve, A_pvsst, A_pvpv]
            ])
    
    def _extract_spatial_scales(self):
        """Extract spatial scales (sigma) for connections, using source population's width for all its connections."""
        if self.n_populations == 2:
            sigma_e = self.preset['outgoing_widths']['E']
            sigma_sst = self.preset['outgoing_widths']['SST']
            sigma_pv = self.preset['outgoing_widths']['PV']
            sigma_i = (sigma_sst + sigma_pv) / 2.0
            sigma_per_pop = np.array([sigma_e, sigma_i])
        else:
            # Full 3-population model
            sigma_e = self.preset['outgoing_widths']['E']
            sigma_sst = self.preset['outgoing_widths']['SST']
            sigma_pv = self.preset['outgoing_widths']['PV']
            sigma_per_pop = np.array([sigma_e, sigma_sst, sigma_pv])
        
        # Build full matrix: use source population's width for all its connections (including inter-layer)
        n_layers = len(self.layers)
        n = self.n_populations
        total_pops = n_layers * n
        self.sigma = np.zeros((total_pops, total_pops))
        
        # For each source population (column), use its sigma for all targets (rows)
        for i_src in range(n_layers):
            for j_pop in range(n):
                col_idx = i_src * n + j_pop
                sigma_src = sigma_per_pop[j_pop]
                # Apply to all target populations
                for i_tgt in range(n_layers):
                    for i_pop in range(n):
                        row_idx = i_tgt * n + i_pop
                        self.sigma[row_idx, col_idx] = sigma_src
    
    def _extract_baseline_input(self):
        """Extract baseline input from noise mean parameters, repeated for each layer."""
        # In the simulation, noise mean provides a constant baseline input
        # Noise formula: noise = mean + std * (private + shared components)
        # For steady state analysis, we use the mean component as baseline
        if self.n_populations == 2:
            # Collapse SST and PV noise means
            mean_sst = self.preset['noise_params']['SST']['mean']
            mean_pv = self.preset['noise_params']['PV']['mean']
            mean_i = (mean_sst + mean_pv) / 2.0
            mean_e = self.preset['noise_params']['E']['mean']
            mu_per_layer = np.array([mean_e, mean_i])
        else:
            # Full 3-population model
            mean_e = self.preset['noise_params']['E']['mean']
            mean_sst = self.preset['noise_params']['SST']['mean']
            mean_pv = self.preset['noise_params']['PV']['mean']
            mu_per_layer = np.array([mean_e, mean_sst, mean_pv])
        
        # Repeat for each layer
        self.mu = np.tile(mu_per_layer, len(self.layers))
    
    def get_parameters(self) -> Dict:
        """Get all network parameters for display."""
        # Build full population names (layer + cell type)
        full_pop_names = []
        for layer in self.layers:
            for pop in self.pop_names:
                full_pop_names.append(f'{layer}_{pop}')
        
        params = {
            'n_populations': self.n_populations,
            'n_layers': len(self.layers),
            'layers': self.layers.copy(),
            'pop_names': self.pop_names,
            'full_pop_names': full_pop_names,
            'tau': self.tau.copy(),
            'gain': self.gain.copy(),
            'A': self.A.copy(),
            'sigma': self.sigma.copy(),
            'mu': self.mu.copy()
        }
        return params


# Backward compatibility alias
SimplifiedNetwork = NetworkModel


class SteadyStateFinder:
    """Finds the spatially uniform steady state of the network."""
    
    def __init__(self, network: NetworkModel, tol: float = None, max_iters: int = None):
        """
        Initialize steady state finder.
        
        Args:
            network: NetworkModel instance
            tol: Convergence tolerance (defaults to config value)
            max_iters: Maximum iterations (defaults to config value)
        """
        self.network = network
        self.tol = tol if tol is not None else ANALYSIS_PARAMS['tolerance']
        self.max_iters = max_iters if max_iters is not None else ANALYSIS_PARAMS['max_iters']
    
    def find_steady_state(self) -> Tuple[np.ndarray, bool]:
        """
        Find the spatially uniform steady state.
        
        Returns:
            (r_star, converged) tuple where r_star is array of firing rates
        """
        # Total number of populations (all layers)
        n = len(self.network.tau)
        # Initial guess: small positive rates
        r = np.ones(n) * 0.1
        
        for iteration in range(self.max_iters):
            # Compute input: A @ r + mu
            input_vec = self.network.A @ r + self.network.mu
            # Apply ReLU: r = max(0, gain * input)
            r_new = np.maximum(0.0, self.network.gain * input_vec)
            
            # Check for divergence
            if np.any(r_new > 1e10):
                # Network is diverging - return zero state
                return np.zeros(n), False
            
            # Check convergence
            change = np.abs(r_new - r)
            if np.all(change < self.tol):
                return r_new, True
            
            # Check if oscillating or stuck after some iterations
            if iteration > 10:
                if np.all(change < self.tol * 10):
                    return r_new, True
            
            r = r_new
        
        # Did not converge - return final values
        r_final = np.clip(r, 0, 1e10)
        return r_final, False


class StabilityAnalyzer:
    """Computes network stability through eigenvalue analysis."""
    
    def __init__(self, network: NetworkModel, steady_state: np.ndarray, 
                 n_modes: int = None):
        """
        Initialize stability analyzer.
        
        Args:
            network: NetworkModel instance
            steady_state: Array of steady state firing rates
            n_modes: Number of Fourier modes to scan in each direction (defaults to config value)
        """
        self.network = network
        self.r_star = steady_state
        self.n_modes = n_modes if n_modes is not None else ANALYSIS_PARAMS['n_modes']
        
        # For ReLU, use full gains (linear stability analysis around steady state)
        self.g_eff = self.network.gain.copy()
    
    def build_jacobian(self, n1: int, n2: int) -> np.ndarray:
        """
        Build n×n Jacobian matrix for Fourier mode (n1, n2).
        
        Args:
            n1: Fourier mode index in first dimension
            n2: Fourier mode index in second dimension
            
        Returns:
            n×n Jacobian matrix (n = total populations across all layers)
        """
        k_squared = n1**2 + n2**2
        grid_size = ANALYSIS_PARAMS['grid_size']
        scale_factor = 1.0 / grid_size
        
        # Total number of populations (all layers)
        n = len(self.network.tau)
        J = np.zeros((n, n))
        
        for i in range(n):  # target
            for j in range(n):  # source
                # Get spatial scale for this connection (source population's width)
                sigma_ij = self.network.sigma[i, j]
                
                # Fourier transform of Gaussian connectivity
                w_tilde = self.network.A[i, j] * np.exp(
                    -2 * np.pi**2 * k_squared * sigma_ij**2 * scale_factor**2
                )
                
                # Jacobian element: -delta_ij/tau_i + (g_i * w_tilde) / tau_i
                if i == j:
                    J[i, j] = -1.0 / self.network.tau[i] + (self.g_eff[i] * w_tilde) / self.network.tau[i]
                else:
                    J[i, j] = (self.g_eff[i] * w_tilde) / self.network.tau[i]
        
        return J
    
    def compute_stability(self) -> Tuple[float, Tuple[int, int], float]:
        """
        Compute stability by finding maximum real part of eigenvalues.
        
        Returns:
            (distance_to_instability, critical_mode, critical_k) tuple
        """
        max_real_eigenvalue = -np.inf
        critical_mode = (0, 0)
        critical_k = 0.0
        
        # Scan Fourier modes
        for n1 in range(-self.n_modes, self.n_modes + 1):
            for n2 in range(-self.n_modes, self.n_modes + 1):
                J = self.build_jacobian(n1, n2)
                eigenvalues = np.linalg.eigvals(J)
                max_real = np.max(eigenvalues.real)
                
                if max_real > max_real_eigenvalue:
                    max_real_eigenvalue = max_real
                    critical_mode = (n1, n2)
                    critical_k = np.sqrt(n1**2 + n2**2)
        
        # Distance to instability: negative of max eigenvalue
        distance_to_instability = -max_real_eigenvalue
        
        return distance_to_instability, critical_mode, critical_k
