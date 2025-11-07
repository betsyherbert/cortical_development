"""Core bifurcation analysis implementation.

This module implements linear stability analysis for network models,
computing steady states and determining network stability through eigenvalue analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from .config import ANALYSIS_PARAMS


class NetworkModel:
    """Handles network dynamics for single or multiple layers with E-SST-PV populations."""
    
    def __init__(self, preset: Dict, layers: list = None):
        """
        Initialize network from preset.
        
        Args:
            preset: Developmental preset dictionary (e.g., P4_PRESET)
            layers: List of layers to analyze (e.g., ['L4'] or ['L23', 'L4', 'L5'])
                   Defaults to ['L4'] for backward compatibility
        """
        # Handle backward compatibility: convert single layer string to list
        if layers is None:
            layers = ['L4']
        elif isinstance(layers, str):
            layers = [layers]
        
        self.layers = layers
        self.preset = preset
        self.pop_names = ['E', 'SST', 'PV']
        
        # Extract parameters
        self._extract_time_constants()
        self._extract_gains()
        self._extract_connection_strengths()
        self._extract_spatial_scales()
        self._extract_thalamic_strengths()
        self._extract_thalamic_widths()
        
        # Baseline input: noise mean provides constant baseline input in the simulation
        # For spatially uniform steady state, we use the noise mean as mu
        self._extract_baseline_input()
    
    def _extract_time_constants(self):
        """Extract time constants for each population, repeated for each layer."""
        tau_e = self.preset['time_constants']['E']
        tau_sst = self.preset['time_constants']['SST']
        tau_pv = self.preset['time_constants']['PV']
        tau_per_layer = np.array([tau_e, tau_sst, tau_pv])
        
        # Repeat for each layer
        self.tau = np.tile(tau_per_layer, len(self.layers))
    
    def _extract_gains(self):
        """Extract gains for each population, repeated for each layer."""
        g_e = self.preset['gains']['E']
        g_sst = self.preset['gains']['SST']
        g_pv = self.preset['gains']['PV']
        gain_per_layer = np.array([g_e, g_sst, g_pv])
        
        # Repeat for each layer
        self.gain = np.tile(gain_per_layer, len(self.layers))
    
    def _extract_connection_strengths(self):
        """
        Extract connection strengths for all layer pairs, building block-structured matrix.
        
        Strength scaling is applied column-wise (per source cell type).
        Connection strengths from presets include signs (inhibitory = negative).
        """
        scaling_e = self.preset['strength_scaling']['E']
        scaling_sst = self.preset['strength_scaling']['SST']
        scaling_pv = self.preset['strength_scaling']['PV']
        
        n_layers = len(self.layers)
        n = 3  # E, SST, PV
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
        """
        Extract connection strength block for one layer pair.
        
        Connection strengths are read directly from preset with signs included.
        Inhibitory connections (SST→*, PV→*) should be negative in presets.
        Strength scaling is column-wise (applied per source cell type).
        
        Args:
            layer_src: Source layer name
            layer_tgt: Target layer name
            scaling_e: Scaling factor for E (excitatory) source connections
            scaling_sst: Scaling factor for SST (inhibitory) source connections
            scaling_pv: Scaling factor for PV (inhibitory) source connections
            
        Returns:
            3×3 connection strength block (E, SST, PV)
        """
        def get_raw(source, target):
            key = f'{layer_src}_{source}_to_{layer_tgt}_{target}'
            return self.preset['connection_strengths'].get(key, 0.0)
        
        A_ee = get_raw('E', 'E') * scaling_e
        A_esst = get_raw('E', 'SST') * scaling_e
        A_epv = get_raw('E', 'PV') * scaling_e
        A_sste = get_raw('SST', 'E') * scaling_sst
        A_sstsst = get_raw('SST', 'SST') * scaling_sst  # Typically zero, but read from preset
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
        sigma_e = self.preset['outgoing_widths']['E']
        sigma_sst = self.preset['outgoing_widths']['SST']
        sigma_pv = self.preset['outgoing_widths']['PV']
        sigma_per_pop = np.array([sigma_e, sigma_sst, sigma_pv])
        
        # Build full matrix: use source population's width for all its connections (including inter-layer)
        n_layers = len(self.layers)
        n = 3  # E, SST, PV
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
        """Extract baseline input from background_input parameters, repeated for each layer."""
        # In the simulation, each cell type receives a constant background input
        # For steady state analysis, we use these background input values as baseline
        mean_e = self.preset['background_input']['E']
        mean_sst = self.preset['background_input']['SST']
        mean_pv = self.preset['background_input']['PV']
        mu_per_layer = np.array([mean_e, mean_sst, mean_pv])
        
        # Repeat for each layer
        self.mu = np.tile(mu_per_layer, len(self.layers))
    
    def _extract_thalamic_strengths(self):
        """Extract thalamic connection strengths for each population in each layer."""
        # Get thalamic scaling factor
        thalamic_scaling = self.preset['strength_scaling'].get('thalamus', 1.0)
        
        # Build thalamic strengths for each layer and population
        thalamic_per_layer = []
        for layer in self.layers:
            thal_e = self.preset['connection_strengths'].get(f'thalamus_to_{layer}_E', 0.0)
            thal_sst = self.preset['connection_strengths'].get(f'thalamus_to_{layer}_SST', 0.0)
            thal_pv = self.preset['connection_strengths'].get(f'thalamus_to_{layer}_PV', 0.0)
            thalamic_per_layer.extend([thal_e, thal_sst, thal_pv])
        
        self.thalamic_strengths = np.array(thalamic_per_layer) * thalamic_scaling
    
    def _extract_thalamic_widths(self):
        """Extract thalamic spatial widths for each population in each layer."""
        # Get thalamic widths from preset
        thal_width_e = self.preset['thalamic_widths']['E']
        thal_width_sst = self.preset['thalamic_widths']['SST']
        thal_width_pv = self.preset['thalamic_widths']['PV']
        thalamic_widths_per_layer = np.array([thal_width_e, thal_width_sst, thal_width_pv])
        
        # Repeat for each layer
        self.thalamic_widths = np.tile(thalamic_widths_per_layer, len(self.layers))
    
    def compute_thalamic_input(self, input_magnitude: float) -> np.ndarray:
        """
        Compute spatially-averaged thalamic drive.
        
        Args:
            input_magnitude: Scalar magnitude of thalamic input
            
        Returns:
            Array of thalamic drive for each population (same shape as mu)
        """
        return self.thalamic_strengths * input_magnitude
    
    def get_parameters(self) -> Dict:
        """Get all network parameters for display."""
        # Build full population names (layer + cell type)
        full_pop_names = []
        for layer in self.layers:
            for pop in self.pop_names:
                full_pop_names.append(f'{layer}_{pop}')
        
        params = {
            'n_populations': 3,  # Always 3: E, SST, PV
            'n_layers': len(self.layers),
            'layers': self.layers.copy(),
            'pop_names': self.pop_names,
            'full_pop_names': full_pop_names,
            'tau': self.tau.copy(),
            'gain': self.gain.copy(),
            'A': self.A.copy(),
            'sigma': self.sigma.copy(),
            'mu': self.mu.copy(),
            'thalamic_strengths': self.thalamic_strengths.copy(),
            'thalamic_widths': self.thalamic_widths.copy()
        }
        return params


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
    
    def find_steady_state(self, thalamic_input: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
        """
        Find the spatially uniform steady state.
        
        For thalamic-driven mode, this computes the spatially uniform mean firing rate
        during thalamic input epochs. This captures the mean depolarization but not
        the spatial structure of bursts (which would require full spatial dynamics).
        
        Args:
            thalamic_input: Optional array of thalamic drive for each population.
                           If None, finds silent fixed point (no external input).
                           
        Returns:
            (r_star, status) tuple where r_star is array of firing rates.
            status is one of: 'converged', 'not_converged', 'diverged'
        """
        # Total number of populations (all layers)
        n = len(self.network.tau)
        # Initial guess: small positive rates
        r = np.ones(n) * 0.1
        
        # Determine total external input
        if thalamic_input is None:
            external_input = self.network.mu
        else:
            external_input = self.network.mu + thalamic_input
        
        # Damping parameter for stability near criticality
        alpha = 0.25
        
        for _ in range(self.max_iters):
            # Compute input: A @ r + external_input
            input_vec = self.network.A @ r + external_input
            # Apply ReLU: r = max(0, gain * input)
            r_new_raw = np.maximum(0.0, self.network.gain * input_vec)
            
            # Check for divergence
            if np.any(r_new_raw > 1e10):
                # Network is diverging - return zero state with clear status
                return np.zeros(n), 'diverged'
            
            # Apply damping for stability near criticality: r ← (1-α)r + αr_new
            r_new = (1 - alpha) * r + alpha * r_new_raw
            
            # Check convergence (strict criterion)
            change = np.abs(r_new - r)
            if np.all(change < self.tol):
                return r_new, 'converged'
            
            r = r_new
        
        # Did not converge - return final values
        r_final = np.clip(r, 0, 1e10)
        return r_final, 'not_converged'


class StabilityAnalyzer:
    """Computes network stability through eigenvalue analysis."""
    
    def __init__(self, network: NetworkModel, steady_state: np.ndarray, 
                 n_modes: int = None, threshold: float = 1e-10):
        """
        Initialize stability analyzer.
        
        Args:
            network: NetworkModel instance
            steady_state: Array of steady state firing rates
            n_modes: Number of Fourier modes to scan in each direction (defaults to config value)
            threshold: Threshold for determining active units (default: 1e-10)
        """
        self.network = network
        self.r_star = steady_state
        self.n_modes = n_modes if n_modes is not None else ANALYSIS_PARAMS['n_modes']
        self.threshold = threshold
        
        # For ReLU, effective gains depend on operating point:
        # - Active units (r* > threshold) have full gain
        # - Inactive units (r* ≈ 0) have zero gain
        self.g_eff = self._compute_effective_gains()
    
    def _compute_effective_gains(self) -> np.ndarray:
        """
        Compute effective gains based on operating point.
        
        For ReLU nonlinearity, the linearization around an operating point
        has gain equal to the slope only for active units.
        
        Returns:
            Array of effective gains for each population
        """
        g_eff = np.zeros_like(self.network.gain)
        active_mask = self.r_star > self.threshold
        g_eff[active_mask] = self.network.gain[active_mask]
        return g_eff
    
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
        domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)  # Use grid_size as fallback for backward compatibility
        
        # Total number of populations (all layers)
        n = len(self.network.tau)
        J = np.zeros((n, n))
        
        for i in range(n):  # target
            for j in range(n):  # source
                # Get spatial scale for this connection (source population's width)
                # σ values in presets are in grid cells (e.g., 2-3 cells on a 20×20 grid)
                # Normalize by grid_size to convert to physical units: σ_phys = σ_cells / grid_size
                # For Fourier transform: exp(-2π²k²σ_phys²) with k = sqrt(n1²+n2²)
                sigma_ij = self.network.sigma[i, j] / domain_length
                
                # Fourier transform of Gaussian connectivity
                # For domain [0,L]², wave number k corresponds to exp(2πi k·x / L)
                # Gaussian kernel exp(-|x|²/(2σ²)) has Fourier transform exp(-2π²k²σ²)
                w_tilde = self.network.A[i, j] * np.exp(
                    -2 * np.pi**2 * k_squared * sigma_ij**2
                )
                
                # Jacobian element: -delta_ij/tau_i + (g_i * w_tilde) / tau_i
                if i == j:
                    J[i, j] = -1.0 / self.network.tau[i] + (self.g_eff[i] * w_tilde) / self.network.tau[i]
                else:
                    J[i, j] = (self.g_eff[i] * w_tilde) / self.network.tau[i]
        
        return J
    
    def compute_stability(self, verbose: bool = False) -> Tuple[float, Tuple[int, int], float, float]:
        """
        Compute stability by finding maximum real part of eigenvalues.
        
        Args:
            verbose: If True, print diagnostic information
        
        Returns:
            (distance_to_instability, critical_mode, critical_k, wavelength) tuple
            wavelength = domain_length / critical_k (or inf if k=0)
        """
        # Clamp mode scan range based on domain_length (Nyquist limit)
        domain_length = ANALYSIS_PARAMS.get('domain_length', ANALYSIS_PARAMS['grid_size'])
        n_modes_effective = min(self.n_modes, int(0.6 * domain_length))
        
        if verbose:
            print(f"  [Stability] Scanning modes from -{n_modes_effective} to +{n_modes_effective} (clamped from {self.n_modes})")
            print(f"  [Stability] Active populations: {np.sum(self.r_star > 1e-10)}/{len(self.r_star)}")
            print(f"  [Stability] g_eff non-zero: {np.sum(self.g_eff > 0)}/{len(self.g_eff)}")
            print(f"  [Stability] g_eff values: {self.g_eff}")
        
        max_real_eigenvalue = -np.inf
        critical_mode = (0, 0)
        critical_k = 0.0
        
        # Scan Fourier modes (clamped range)
        for n1 in range(-n_modes_effective, n_modes_effective + 1):
            for n2 in range(-n_modes_effective, n_modes_effective + 1):
                J = self.build_jacobian(n1, n2)
                eigenvalues = np.linalg.eigvals(J)
                max_real = np.max(eigenvalues.real)
                
                if max_real > max_real_eigenvalue:
                    max_real_eigenvalue = max_real
                    critical_mode = (n1, n2)
                    critical_k = np.sqrt(n1**2 + n2**2)
        
        # Distance to instability: negative of max eigenvalue
        distance_to_instability = -max_real_eigenvalue
        
        # Compute wavelength: λ* = L / k
        wavelength = domain_length / critical_k if critical_k > 0 else np.inf
        
        if verbose:
            print(f"  [Stability] Critical mode: {critical_mode}, k={critical_k:.4f}, λ*={wavelength:.4f}")
            print(f"  [Stability] Max Re(λ): {max_real_eigenvalue:.6f}")
            print(f"  [Stability] Distance to instability: {distance_to_instability:.6f}")
        
        return distance_to_instability, critical_mode, critical_k, wavelength
    
    def compute_forced_response(self, B: np.ndarray, verbose: bool = False) -> Dict:
        """
        Compute forced response gain by finding the Fourier mode with maximum static gain.
        
        The forced response gain is computed as ||J(k)⁻¹ @ B_norm||, where B_norm is
        the normalized thalamic input vector and J(k) is the Jacobian for Fourier mode k.
        
        Args:
            B: Thalamic input vector (array of connection strengths)
            verbose: If True, print diagnostic information
        
        Returns:
            Dictionary containing forced response diagnostics:
            - 'max_gain': Maximum forced response gain across all modes
            - 'critical_mode': (n₁, n₂) tuple of the mode with maximum gain
            - 'critical_k': k value (sqrt(n₁² + n₂²)) of the critical mode
            - 'max_condition': Maximum condition number encountered
            - 'k_values': Sorted array of unique k magnitudes evaluated
            - 'gain_profile': Gain for each k (maximum across degenerate modes)
            - 'max_real_profile': Max real part of eigenvalues for each k
        """
        # Normalize B once per analysis call
        B_norm_vec = np.linalg.norm(B)
        if B_norm_vec < 1e-10:
            # Zero thalamic input - return NaN values
            if verbose:
                print(f"  [ForcedResponse] Zero thalamic input (||B||={B_norm_vec:.2e}), skipping analysis")
            return {
                'max_gain': np.nan,
                'critical_mode': (0, 0),
                'critical_k': 0.0,
                'max_condition': np.nan,
                'k_values': np.array([]),
                'gain_profile': np.array([]),
                'max_real_profile': np.array([])
            }
        
        B_norm = B / B_norm_vec
        
        # Clamp mode scan range based on domain_length (same as compute_stability)
        domain_length = ANALYSIS_PARAMS.get('domain_length', ANALYSIS_PARAMS['grid_size'])
        n_modes_effective = min(self.n_modes, int(0.6 * domain_length))
        
        if verbose:
            print(f"  [ForcedResponse] Scanning modes from -{n_modes_effective} to +{n_modes_effective}")
            print(f"  [ForcedResponse] B norm: {B_norm_vec:.6f}, normalized B shape: {B_norm.shape}")
        
        max_gain = -np.inf
        critical_mode = (0, 0)
        critical_k = 0.0
        max_condition = 0.0

        # Aggregate results per k^2 (to handle degeneracy of Fourier modes sharing same magnitude)
        profile_by_k2: Dict[int, Dict[str, float]] = {}
        
        # Scan Fourier modes (clamped range)
        for n1 in range(-n_modes_effective, n_modes_effective + 1):
            for n2 in range(-n_modes_effective, n_modes_effective + 1):
                # Build Jacobian for this mode
                Jk = self.build_jacobian(n1, n2)
                
                # Compute condition number for diagnostics
                cond_Jk = np.linalg.cond(Jk)
                if cond_Jk > max_condition:
                    max_condition = cond_Jk
                
                # Compute J(k)⁻¹ @ B_norm using pseudo-inverse (handles singular matrices)
                try:
                    eigenvalues = np.linalg.eigvals(Jk)
                    max_real_mode = np.max(eigenvalues.real)

                    Jk_inv_B = np.linalg.pinv(Jk) @ B_norm
                    gain_k = np.linalg.norm(Jk_inv_B)

                    if gain_k > max_gain:
                        max_gain = gain_k
                        critical_mode = (n1, n2)
                        critical_k = np.sqrt(n1**2 + n2**2)

                    k_squared = n1**2 + n2**2
                    if k_squared not in profile_by_k2:
                        profile_by_k2[k_squared] = {
                            'k': np.sqrt(k_squared),
                            'gain': gain_k,
                            'max_real': max_real_mode
                        }
                    else:
                        profile_entry = profile_by_k2[k_squared]
                        profile_entry['gain'] = max(profile_entry['gain'], gain_k)
                        profile_entry['max_real'] = max(profile_entry['max_real'], max_real_mode)
                except np.linalg.LinAlgError:
                    # If pseudo-inverse fails, skip this mode
                    if verbose:
                        print(f"  [ForcedResponse] Warning: Failed to compute pseudo-inverse for mode ({n1}, {n2})")
                    continue
        
        # Sort profiles by k magnitude
        if profile_by_k2:
            sorted_items = sorted(profile_by_k2.values(), key=lambda item: item['k'])
            k_values = np.array([item['k'] for item in sorted_items])
            gain_profile = np.array([item['gain'] for item in sorted_items])
            max_real_profile = np.array([item['max_real'] for item in sorted_items])
        else:
            k_values = np.array([])
            gain_profile = np.array([])
            max_real_profile = np.array([])

        if verbose:
            print(f"  [ForcedResponse] Critical mode: {critical_mode}, k={critical_k:.4f}")
            print(f"  [ForcedResponse] Max gain: {max_gain:.6f}")
            print(f"  [ForcedResponse] Max condition number: {max_condition:.2e}")
            if k_values.size > 0:
                print(f"  [ForcedResponse] Evaluated {k_values.size} unique k values")

        return {
            'max_gain': max_gain,
            'critical_mode': critical_mode,
            'critical_k': critical_k,
            'max_condition': max_condition,
            'k_values': k_values,
            'gain_profile': gain_profile,
            'max_real_profile': max_real_profile
        }