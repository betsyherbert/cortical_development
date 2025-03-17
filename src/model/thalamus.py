import numpy as np
from typing import Optional, Tuple

class ThalamicInput:
    def __init__(self, grid_size: int = 10, dt: float = 1.0):
        """
        Initialize thalamic input generator.
        
        Args:
            grid_size: Size of the square grid (default: 10)
            dt: Time step in milliseconds (default: 1.0)
        """
        self.grid_size = grid_size
        self.dt = dt
        
        # Parameters for intrinsic activity
        self.intrinsic_freq = 1.0  # Hz
        self.intrinsic_sigma = 5.0  # grid units
        self.intrinsic_phase = 0.0
        
        # Parameters for sensory activity
        self.sensory_sigma = 2.0  # grid units
        self.sensory_duration = 50.0  # ms
        self.sensory_interval = 500.0  # ms
        self.last_sensory_time = 0.0
        
        # Center coordinates for sensory input
        self.sensory_center = (grid_size // 2, grid_size // 2)
        
        # Create coordinate grid for spatial profiles
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
        
        # Initialize time
        self.t = 0.0
        
    def gaussian_spatial(self, center: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Generate a 2D Gaussian spatial profile.
        
        Args:
            center: Center coordinates (x, y)
            sigma: Width of the Gaussian
            
        Returns:
            2D array containing the spatial profile
        """
        x, y = self.coords
        d_squared = (x - center[0])**2 + (y - center[1])**2
        return np.exp(-0.5 * d_squared / sigma**2)
    
    def generate_intrinsic(self) -> np.ndarray:
        """
        Generate intrinsic thalamic activity.
        
        Returns:
            2D array containing the intrinsic activity
        """
        # Update phase
        self.intrinsic_phase += 2 * np.pi * self.intrinsic_freq * self.dt / 1000.0
        
        # Generate temporal oscillation
        amplitude = 0.5 * (1 + np.sin(self.intrinsic_phase))
        
        # Generate spatial profile
        center = (self.grid_size // 2, self.grid_size // 2)
        spatial = self.gaussian_spatial(center, self.intrinsic_sigma)
        
        return amplitude * spatial
    
    def generate_sensory(self) -> np.ndarray:
        """
        Generate sensory thalamic activity.
        
        Returns:
            2D array containing the sensory activity
        """
        # Check if it's time for a new sensory input
        time_since_last = self.t - self.last_sensory_time
        
        if time_since_last >= self.sensory_interval:
            self.last_sensory_time = self.t
            # Generate new random center near the middle
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            self.sensory_center = (
                self.grid_size // 2 + dx,
                self.grid_size // 2 + dy
            )
        
        # If within pulse duration, generate Gaussian pulse
        if time_since_last < self.sensory_duration:
            amplitude = np.exp(-0.5 * (time_since_last / (self.sensory_duration/3))**2)
            spatial = self.gaussian_spatial(self.sensory_center, self.sensory_sigma)
            return amplitude * spatial
        
        return np.zeros((self.grid_size, self.grid_size))
    
    def update(self, alpha: float = 0.7) -> np.ndarray:
        """
        Generate combined thalamic input.
        
        Args:
            alpha: Weight of intrinsic vs sensory activity (default: 0.7)
            
        Returns:
            2D array containing the combined thalamic activity
        """
        # Generate components
        intrinsic = self.generate_intrinsic()
        sensory = self.generate_sensory()
        
        # Combine components
        activity = alpha * intrinsic + (1 - alpha) * sensory
        
        # Update time
        self.t += self.dt
        
        return activity
    
    def reset(self) -> None:
        """Reset the thalamic input generator."""
        self.t = 0.0
        self.last_sensory_time = 0.0
        self.intrinsic_phase = 0.0
        self.sensory_center = (self.grid_size // 2, self.grid_size // 2) 