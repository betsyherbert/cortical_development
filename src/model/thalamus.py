"""Thalamic input module for the cortical circuit simulation.

This module generates realistic thalamic activity patterns that combine:
1. Intrinsic bursts - ongoing random burst activity with wave-like properties
2. Sensory inputs - localized stimulus-driven responses

Note: All spatial parameters (sigma) are in μm (anatomical units).
"""

import numpy as np
from typing import Tuple, List, Dict
from .config import (
    GRID_SIZE, DT, ANATOMICAL_GRID_SIZE,
    THALAMIC_INTRINSIC_SIGMA, THALAMIC_INTRINSIC_DURATION,
    THALAMIC_INTRINSIC_INTERVAL, THALAMIC_INTRINSIC_AMP,
    THALAMIC_SENSORY_SIGMA, THALAMIC_SENSORY_DURATION,
    THALAMIC_SENSORY_INTERVAL, THALAMIC_SENSORY_AMP,
    THALAMIC_ALPHA
)


class ThalamicInput:
    """Generates thalamic activity patterns combining intrinsic and sensory components.
    
    Note: All sigma (spatial width) parameters are in μm (anatomical units) and are
    converted internally to grid units for computation.
    """
    
    def __init__(self, grid_size: int = GRID_SIZE, dt: float = DT,
                 anatomical_grid_size: float = ANATOMICAL_GRID_SIZE):
        """Initialize the thalamic input generator.
        
        Args:
            grid_size: Number of grid points in each dimension
            dt: Time step in milliseconds
            anatomical_grid_size: Anatomical size of the grid in μm
        """
        self.grid_size = grid_size
        self.dt = dt
        self.t = 0.0
        self.anatomical_grid_size = anatomical_grid_size
        self.grid_scale = anatomical_grid_size / grid_size  # μm per grid unit
        
        # Set up spatial coordinate grid
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
        self._spatial_cache = {}
        
        # Initialize burst lists
        self.intrinsic_bursts = []
        self.sensory_bursts = []
        self.max_intrinsic_bursts = 8
        self.max_sensory_bursts = 3
        
        # Pre-compute components
        self._prepare_gaussian_components()
    

    
    def _prepare_gaussian_components(self):
        """Pre-compute coordinate arrays for faster Gaussian calculation."""
        self.x_coords, self.y_coords = self.coords
        self.grid_center = (self.grid_size // 2, self.grid_size // 2)
    
    def gaussian_spatial(self, center: Tuple[int, int], sigma_um: float) -> np.ndarray:
        """Generate a 2D Gaussian spatial profile with caching.
        
        Args:
            center: Center coordinates in grid units (x, y)
            sigma_um: Width of the Gaussian in μm (anatomical units)
            
        Returns:
            2D array containing the Gaussian profile
        """
        # Convert sigma from μm to grid units
        sigma_grid = sigma_um / self.grid_scale
        
        # Use μm value for cache key (for consistency)
        cache_key = (center, sigma_um)
        if cache_key in self._spatial_cache:
            return self._spatial_cache[cache_key]
            
        d_squared = (self.x_coords - center[0])**2 + (self.y_coords - center[1])**2
        profile = np.exp(-0.5 * d_squared / sigma_grid**2)
        
        self._spatial_cache[cache_key] = profile
        return profile
    
    def _compute_temporal_profile(self, time_since_start: float, duration: float, 
                                phase: float = 0.0, n_oscillations: float = 2.0) -> float:
        """Compute temporal profile with oscillations and smooth transitions."""
        normalized_time = time_since_start / duration
        
        # Base envelope function
        if normalized_time < 0.3:
            envelope = normalized_time / 0.3
        elif normalized_time > 0.7:
            envelope = 1.0 - (normalized_time - 0.7) / 0.3
        else:
            envelope = 1.0
            
        # Add oscillations
        oscillation = np.sin(2 * np.pi * n_oscillations * normalized_time + phase)
        
        return envelope * (0.5 + 0.5 * oscillation)
    
    def _generate_burst_activity(self, bursts: List[Dict], is_intrinsic: bool = False) -> np.ndarray:
        """Generate activity from a list of bursts.
        
        Args:
            bursts: List of burst dictionaries
            is_intrinsic: Whether these are intrinsic bursts (affects combination method)
            
        Returns:
            Combined activity pattern
        """
        activity = np.zeros((self.grid_size, self.grid_size))
        active_bursts = []
        
        for burst in bursts:
            time_since_start = self.t - burst['start_time']
            
            if time_since_start < burst['duration']:
                # Calculate temporal profile
                if is_intrinsic:
                    temporal = self._compute_temporal_profile(
                        time_since_start, 
                        burst['duration'],
                        burst['phase'],
                        burst['n_oscillations']
                    )
                    # Phase-dependent combination for intrinsic bursts
                    burst_activity = burst['amplitude'] * temporal * self.gaussian_spatial(burst['center'], burst['sigma'])
                    activity += burst_activity * np.cos(burst['phase'])
                else:
                    # Simple temporal profile for sensory bursts
                    normalized_time = time_since_start / burst['duration']
                    if normalized_time < 0.3:
                        temporal = normalized_time / 0.3
                    elif normalized_time > 0.7:
                        temporal = 1.0 - (normalized_time - 0.7) / 0.3
                    else:
                        temporal = 1.0
                    
                    burst_activity = burst['amplitude'] * temporal * self.gaussian_spatial(burst['center'], burst['sigma'])
                    activity = np.maximum(activity, burst_activity)
                
                active_bursts.append(burst)
        
        if is_intrinsic:
            activity = np.maximum(0, activity)  # Ensure non-negative activity
            
        return activity, active_bursts
    
    def generate_intrinsic(self) -> np.ndarray:
        """Generate intrinsic thalamic activity as random bursts with wave-like properties."""
        activity, self.intrinsic_bursts = self._generate_burst_activity(self.intrinsic_bursts, is_intrinsic=True)
        return activity
    
    def generate_sensory(self) -> np.ndarray:
        """Generate sensory thalamic activity as localized pulses."""
        activity, self.sensory_bursts = self._generate_burst_activity(self.sensory_bursts, is_intrinsic=False)
        return activity
    
    def _create_burst(self, center: Tuple[int, int], sigma_um: float, duration: float, 
                     amplitude: float, is_intrinsic: bool = False) -> Dict:
        """Create a new burst with appropriate parameters.
        
        Args:
            center: Center coordinates in grid units (x, y)
            sigma_um: Base spatial width in μm (will be randomized)
            duration: Base duration in ms (will be randomized)
            amplitude: Base amplitude (will be randomized)
            is_intrinsic: Whether this is an intrinsic burst
            
        Returns:
            Dictionary with burst parameters
        """
        burst = {
            'center': center,
            'sigma': sigma_um * np.random.uniform(0.3, 1.7),  # sigma in μm
            'duration': duration * np.random.uniform(0.5, 1.5),
            'amplitude': amplitude * np.random.uniform(0.5, 1.5),
            'start_time': self.t
        }
        
        if is_intrinsic:
            burst.update({
                'phase': np.random.uniform(0, 2 * np.pi),
                'n_oscillations': np.random.uniform(1.5, 3.0)
            })
            
        return burst
    
    def _maybe_add_burst(self, is_intrinsic: bool = False) -> None:
        """Add a new burst if conditions are met."""
        if is_intrinsic:
            max_bursts = self.max_intrinsic_bursts
            interval = THALAMIC_INTRINSIC_INTERVAL
            sigma = THALAMIC_INTRINSIC_SIGMA
            duration = THALAMIC_INTRINSIC_DURATION
            amplitude = THALAMIC_INTRINSIC_AMP
            bursts = self.intrinsic_bursts
        else:
            max_bursts = self.max_sensory_bursts
            interval = THALAMIC_SENSORY_INTERVAL
            sigma = THALAMIC_SENSORY_SIGMA
            duration = THALAMIC_SENSORY_DURATION
            amplitude = THALAMIC_SENSORY_AMP
            bursts = self.sensory_bursts
        
        if len(bursts) < max_bursts and np.random.random() < self.dt / interval:
            center = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            new_burst = self._create_burst(center, sigma, duration, amplitude, is_intrinsic)
            bursts.append(new_burst)
    
    def update(self, alpha: float = THALAMIC_ALPHA, n_steps: int = 10) -> np.ndarray:
        """Generate combined thalamic input for multiple time steps."""
        time_increment = self.dt * n_steps
        self.t += time_increment
        
        # Add new bursts
        for _ in range(n_steps):
            self._maybe_add_burst(is_intrinsic=True)
            self._maybe_add_burst(is_intrinsic=False)
        
        # Generate and combine activity
        intrinsic = self.generate_intrinsic()
        sensory = self.generate_sensory()
        combined = (1 - alpha) * intrinsic + alpha * sensory
        
        return combined
    
    def reset(self) -> None:
        """Reset the generator to initial state."""
        self.t = 0.0
        self.intrinsic_bursts = []
        self.sensory_bursts = []
        self._spatial_cache.clear() 