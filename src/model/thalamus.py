"""Thalamic input module for the cortical circuit simulation.

This module generates realistic thalamic activity patterns that combine:
1. Intrinsic oscillations - ongoing rhythmic activity
2. Sensory inputs - localized stimulus-driven responses
"""

import numpy as np
from typing import Tuple, List, Dict
from .config import (
    GRID_SIZE, DT, 
    THALAMIC_INTRINSIC_FREQ, THALAMIC_INTRINSIC_SIGMA,
    THALAMIC_SENSORY_SIGMA, THALAMIC_SENSORY_DURATION,
    THALAMIC_SENSORY_INTERVAL, THALAMIC_SCALING
)


class ThalamicInput:
    """Generates thalamic activity patterns combining intrinsic and sensory components."""
    
    def __init__(self, grid_size: int = GRID_SIZE, dt: float = DT):
        """Initialize the thalamic input generator.
        
        Args:
            grid_size: Size of the 2D grid for spatial patterns
            dt: Time step in milliseconds
        """
        self.grid_size = grid_size
        self.dt = dt
        self.t = 0.0
        
        # Set up spatial coordinate grid
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
        self._spatial_cache = {}
        
        # Initialize intrinsic oscillators
        self.n_oscillators = 5
        self.oscillator_phases = np.random.uniform(0, 2*np.pi, self.n_oscillators)
        self.oscillator_freqs = THALAMIC_INTRINSIC_FREQ * np.random.uniform(0.5, 1.5, self.n_oscillators)
        self.intrinsic_patterns = self._create_intrinsic_patterns()
        
        # Initialize sensory parameters
        self.sensory_sigma = THALAMIC_SENSORY_SIGMA
        self.sensory_duration = THALAMIC_SENSORY_DURATION
        self.sensory_interval = THALAMIC_SENSORY_INTERVAL
        self.sensory_bursts = []
        self.max_sensory_bursts = 3
        
        # Pre-compute components for frequently used calculations
        self._prepare_gaussian_components()
    
    def _prepare_gaussian_components(self):
        """Pre-compute coordinate arrays for faster Gaussian calculation."""
        self.x_coords, self.y_coords = self.coords
        # Pre-compute grid centers for common operations
        self.grid_center = (self.grid_size // 2, self.grid_size // 2)
    
    def _create_intrinsic_patterns(self) -> List[np.ndarray]:
        """Create spatial patterns for intrinsic activity using random Gaussian bumps."""
        patterns = []
        for _ in range(self.n_oscillators):
            pattern = np.zeros((self.grid_size, self.grid_size))
            n_bumps = np.random.randint(2, 8)
            
            for _ in range(n_bumps):
                cx = np.random.randint(0, self.grid_size)
                cy = np.random.randint(0, self.grid_size)
                sigma = THALAMIC_INTRINSIC_SIGMA * np.random.uniform(0.5, 1.5)
                weight = np.random.uniform(0.5, 1.0)
                
                x, y = self.coords
                d_squared = (x - cx)**2 + (y - cy)**2
                bump = weight * np.exp(-0.5 * d_squared / sigma**2)
                pattern += bump
            
            if np.max(pattern) > 0:
                pattern /= np.max(pattern)
            patterns.append(pattern)
        
        return patterns
    
    def gaussian_spatial(self, center: Tuple[int, int], sigma: float) -> np.ndarray:
        """Generate a 2D Gaussian spatial profile with caching."""
        cache_key = (center, sigma)
        if cache_key in self._spatial_cache:
            return self._spatial_cache[cache_key]
            
        d_squared = (self.x_coords - center[0])**2 + (self.y_coords - center[1])**2
        profile = np.exp(-0.5 * d_squared / sigma**2)
        
        self._spatial_cache[cache_key] = profile
        return profile
    
    def generate_intrinsic(self) -> np.ndarray:
        """Generate intrinsic thalamic activity by combining oscillating patterns."""
        # Vectorize amplitude calculation for all oscillators at once
        amplitudes = 0.5 * (1 + np.tanh(2 * np.sin(self.oscillator_phases)))
        
        # Multiply each pattern by its amplitude and sum
        activity = np.zeros((self.grid_size, self.grid_size))
        for i, pattern in enumerate(self.intrinsic_patterns):
            activity += amplitudes[i] * pattern
        
        # Normalize if needed
        max_val = np.max(activity)
        if max_val > 0:
            activity /= max_val
        
        return activity
    
    def generate_sensory(self) -> np.ndarray:
        """Generate sensory thalamic activity as localized pulses."""
        activity = np.zeros((self.grid_size, self.grid_size))
        active_bursts = []
        
        for burst in self.sensory_bursts:
            time_since_start = self.t - burst['start_time']
            
            if time_since_start < burst['duration']:
                # Calculate temporal profile
                normalized_time = time_since_start / burst['duration']
                if normalized_time < 0.3:
                    temporal = normalized_time / 0.3
                elif normalized_time > 0.7:
                    temporal = 1.0 - (normalized_time - 0.7) / 0.3
                else:
                    temporal = 1.0
                
                spatial = self.gaussian_spatial(burst['center'], burst['sigma'])
                burst_activity = burst['amplitude'] * temporal * spatial
                activity = np.maximum(activity, burst_activity)
                active_bursts.append(burst)
        
        self.sensory_bursts = active_bursts
        return activity
    
    def _update_oscillators(self) -> None:
        """Update oscillator phases based on their frequencies."""
        dt_sec = self.dt / 1000.0
        self.oscillator_phases += 2 * np.pi * self.oscillator_freqs * dt_sec
    
    def _maybe_add_sensory_burst(self) -> None:
        """Add a new sensory burst if conditions are met."""
        if (len(self.sensory_bursts) < self.max_sensory_bursts and 
            np.random.random() < self.dt / self.sensory_interval):
            
            new_burst = {
                'center': (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ),
                'sigma': self.sensory_sigma * np.random.uniform(0.3, 2.2),
                'duration': self.sensory_duration * np.random.uniform(0.6, 1.4),
                'amplitude': np.random.uniform(0.4, 1.4),
                'start_time': self.t
            }
            
            self.sensory_bursts.append(new_burst)
    
    def update(self, alpha: float = 0.7, n_steps: int = 10) -> np.ndarray:
        """Generate combined thalamic input for multiple time steps.
        
        Args:
            alpha: Weight for sensory component (0 = pure intrinsic, 1 = pure sensory)
            n_steps: Number of time steps to simulate
            
        Returns:
            Combined activity pattern
        """
        # Calculate time increment for entire update at once
        time_increment = self.dt * n_steps
        self.t += time_increment
        
        # Update oscillator phases based on their frequencies
        self.oscillator_phases += 2 * np.pi * self.oscillator_freqs * time_increment / 1000.0
        
        # Add sensory bursts
        for _ in range(n_steps):
            self._maybe_add_sensory_burst()
        
        # Generate final output
        intrinsic = self.generate_intrinsic()
        sensory = self.generate_sensory()
        
        # Combine intrinsic and sensory components
        combined = (1 - alpha) * intrinsic + alpha * sensory
        result = combined * THALAMIC_SCALING
        
        return result
    
    def reset(self) -> None:
        """Reset the generator to initial state."""
        self.t = 0.0
        self.oscillator_phases = np.random.uniform(0, 2*np.pi, self.n_oscillators)
        self.oscillator_freqs = THALAMIC_INTRINSIC_FREQ * np.random.uniform(0.6, 1.6, self.n_oscillators)
        self.intrinsic_patterns = self._create_intrinsic_patterns()
        self.sensory_bursts = []
        self._spatial_cache.clear() 