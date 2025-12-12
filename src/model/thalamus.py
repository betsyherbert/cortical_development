"""Thalamic input module for the cortical circuit simulation.

This module generates realistic thalamic activity patterns that combine:
1. Intrinsic bursts - ongoing random burst activity organized in spatial modules
2. Sensory inputs - localized stimulus-driven responses

Developmental changes:
- Spatial scales (sigma) vary by developmental stage
- Temporal scales (duration, interval) vary by developmental stage
- Burst centers organized in a modular lattice with jitter
- Smooth temporal bumps (raised cosine) replace oscillations

Note: All spatial parameters (sigma) are in μm (anatomical units).
"""

import numpy as np

from .config import (
    ANATOMICAL_GRID_SIZE,
    DT,
    GRID_SIZE,
    THALAMIC_ALPHA,
    THALAMIC_INTRINSIC_AMP,
    THALAMIC_JITTER_FACTOR,
    THALAMIC_N_MODULES_PER_DIM,
    THALAMIC_SENSORY_AMP,
)


class ThalamicInput:
    """Generates thalamic activity patterns combining intrinsic and sensory components.

    This class implements developmentally realistic thalamic input with:
    - Module-based spatial organization of burst centers
    - Developmental spatial scales (sigma in μm)
    - Developmental temporal scales (duration and inter-event intervals in ms)
    - Smooth temporal profiles (raised cosine bumps)
    - Fixed amplitudes with alpha mixing for intrinsic vs sensory balance

    Note: All sigma (spatial width) parameters are in μm (anatomical units) and are
    converted internally to grid units for computation.
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        dt: float = DT,
        anatomical_grid_size: float = ANATOMICAL_GRID_SIZE,
        thalamic_spatial_scales: dict | None = None,
        thalamic_temporal_scales: dict | None = None,
        thalamic_modules: dict | None = None,
    ):
        """Initialize the thalamic input generator.

        Args:
            grid_size: Number of grid points in each dimension
            dt: Time step in milliseconds
            anatomical_grid_size: Anatomical size of the grid in μm
            thalamic_spatial_scales: Dict with 'intrinsic_sigma_range' and 'sensory_sigma_range' (μm)
            thalamic_temporal_scales: Dict with duration and interval ranges (ms)
            thalamic_modules: Dict with 'n_modules_per_dim' and 'jitter_factor'
        """
        self.grid_size = grid_size
        self.dt = dt
        self.t = 0.0
        self.anatomical_grid_size = anatomical_grid_size
        self.grid_scale = anatomical_grid_size / grid_size  # μm per grid unit

        # Store developmental parameters (with defaults if not provided)
        if thalamic_spatial_scales is None:
            # Default to P0-like parameters
            thalamic_spatial_scales = {
                "intrinsic_sigma_range": (100.0, 150.0),
                "sensory_sigma_range": (100.0, 150.0),
            }
        if thalamic_temporal_scales is None:
            # Default to P0-like parameters
            thalamic_temporal_scales = {
                "intrinsic_duration_range": (200.0, 300.0),
                "intrinsic_interval_range": (2000.0, 5000.0),
                "sensory_duration_range": (150.0, 250.0),
                "sensory_interval_range": (2000.0, 5000.0),
            }
        if thalamic_modules is None:
            thalamic_modules = {
                "n_modules_per_dim": THALAMIC_N_MODULES_PER_DIM,
                "jitter_factor": THALAMIC_JITTER_FACTOR,
            }

        self.spatial_scales = thalamic_spatial_scales
        self.temporal_scales = thalamic_temporal_scales
        self.n_modules_per_dim = thalamic_modules["n_modules_per_dim"]
        self.jitter_factor = thalamic_modules["jitter_factor"]

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
        self._generate_module_lattice()

    def _prepare_gaussian_components(self):
        """Pre-compute coordinate arrays for faster Gaussian calculation."""
        self.x_coords, self.y_coords = self.coords
        self.grid_center = (self.grid_size // 2, self.grid_size // 2)

    def _generate_module_lattice(self):
        """Generate a regular lattice of module centers for burst organization.

        Creates a regular grid of n_modules_per_dim × n_modules_per_dim module centers
        distributed evenly across the spatial grid.
        """
        module_spacing = self.grid_size / self.n_modules_per_dim
        self.module_spacing = module_spacing

        # Generate centers at regular intervals
        centers = []
        for i in range(self.n_modules_per_dim):
            for j in range(self.n_modules_per_dim):
                # Center each module in its region
                x = (i + 0.5) * module_spacing
                y = (j + 0.5) * module_spacing
                centers.append((x, y))

        self.module_centers = centers
        self.jitter_sigma = self.jitter_factor * module_spacing

    def _pick_burst_center(self) -> tuple[float, float]:
        """Pick a burst center from the module lattice with Gaussian jitter.

        Returns:
            Tuple of (x, y) coordinates in grid units, clipped to grid bounds
        """
        # Randomly select a module center
        module_center = self.module_centers[np.random.randint(len(self.module_centers))]

        # Add Gaussian jitter
        jitter_x = np.random.normal(0, self.jitter_sigma)
        jitter_y = np.random.normal(0, self.jitter_sigma)

        # Apply jitter and clip to grid bounds
        x = np.clip(module_center[0] + jitter_x, 0, self.grid_size - 1)
        y = np.clip(module_center[1] + jitter_y, 0, self.grid_size - 1)

        return (x, y)

    def gaussian_spatial(self, center: tuple[float, float], sigma_um: float) -> np.ndarray:
        """Generate a 2D Gaussian spatial profile with caching.

        Args:
            center: Center coordinates in grid units (x, y) - can be float
            sigma_um: Width of the Gaussian in μm (anatomical units)

        Returns:
            2D array containing the Gaussian profile
        """
        # Convert sigma from μm to grid units
        sigma_grid = sigma_um / self.grid_scale

        # Use rounded center and μm value for cache key
        cache_key = (round(center[0], 2), round(center[1], 2), round(sigma_um, 2))
        if cache_key in self._spatial_cache:
            return self._spatial_cache[cache_key]

        d_squared = (self.x_coords - center[0]) ** 2 + (self.y_coords - center[1]) ** 2
        profile = np.exp(-0.5 * d_squared / sigma_grid**2)

        # Only cache if cache isn't too large
        if len(self._spatial_cache) < 1000:
            self._spatial_cache[cache_key] = profile

        return profile

    def _compute_temporal_profile(self, time_since_start: float, duration: float) -> float:
        """Compute smooth temporal profile using raised cosine bump.

        Implements: 0.5 * (1 + cos(π * (2t/duration - 1))) for t in [0, duration]
        Returns 0 outside this range.

        Args:
            time_since_start: Time elapsed since burst start (ms)
            duration: Total duration of burst (ms)

        Returns:
            Temporal amplitude factor [0, 1]
        """
        if time_since_start < 0 or time_since_start >= duration:
            return 0.0

        # Normalized time in [0, 1]
        t_norm = time_since_start / duration

        # Raised cosine: 0.5 * (1 + cos(π * (2t - 1)))
        # This creates a smooth bump that starts at 0, peaks at 0.5*duration, ends at 0
        temporal = 0.5 * (1.0 + np.cos(np.pi * (2.0 * t_norm - 1.0)))

        return temporal

    def _generate_burst_activity(
        self, bursts: list[dict], is_intrinsic: bool = False
    ) -> tuple[np.ndarray, list[dict]]:
        """Generate activity from a list of bursts.

        Args:
            bursts: List of burst dictionaries
            is_intrinsic: Whether these are intrinsic bursts (affects combination method)

        Returns:
            Tuple of (activity pattern, list of still-active bursts)
        """
        activity = np.zeros((self.grid_size, self.grid_size))
        active_bursts = []

        for burst in bursts:
            time_since_start = self.t - burst["start_time"]

            if 0 <= time_since_start < burst["duration"]:
                # Calculate temporal profile (same for both types now)
                temporal = self._compute_temporal_profile(time_since_start, burst["duration"])

                # Calculate spatial profile
                spatial = self.gaussian_spatial(burst["center"], burst["sigma"])

                # Combine temporal and spatial
                burst_activity = burst["amplitude"] * temporal * spatial

                # Different combination methods for intrinsic vs sensory
                if is_intrinsic:
                    # Intrinsic: simple summation
                    activity += burst_activity
                else:
                    # Sensory: maximum (winner-take-all)
                    activity = np.maximum(activity, burst_activity)

                active_bursts.append(burst)

        # Ensure non-negative activity
        activity = np.maximum(0, activity)

        return activity, active_bursts

    def generate_intrinsic(self) -> np.ndarray:
        """Generate intrinsic thalamic activity as random bursts organized in modules."""
        activity, self.intrinsic_bursts = self._generate_burst_activity(
            self.intrinsic_bursts, is_intrinsic=True
        )
        return activity

    def generate_sensory(self) -> np.ndarray:
        """Generate sensory thalamic activity as localized pulses."""
        activity, self.sensory_bursts = self._generate_burst_activity(
            self.sensory_bursts, is_intrinsic=False
        )
        return activity

    def _create_burst(self, is_intrinsic: bool = False) -> dict:
        """Create a new burst with developmental parameters.

        Samples spatial and temporal parameters from the developmental ranges
        specified in the preset.

        Args:
            is_intrinsic: Whether this is an intrinsic burst

        Returns:
            Dictionary with burst parameters
        """
        # Pick center from module lattice with jitter
        center = self._pick_burst_center()

        # Sample spatial scale from developmental range
        if is_intrinsic:
            sigma_range = self.spatial_scales["intrinsic_sigma_range"]
            duration_range = self.temporal_scales["intrinsic_duration_range"]
            amplitude = THALAMIC_INTRINSIC_AMP
        else:
            sigma_range = self.spatial_scales["sensory_sigma_range"]
            duration_range = self.temporal_scales["sensory_duration_range"]
            amplitude = THALAMIC_SENSORY_AMP

        # Sample from ranges
        sigma_um = np.random.uniform(sigma_range[0], sigma_range[1])
        duration = np.random.uniform(duration_range[0], duration_range[1])

        burst = {
            "center": center,
            "sigma": sigma_um,  # in μm
            "duration": duration,  # in ms
            "amplitude": amplitude,
            "start_time": self.t,
        }

        return burst

    def _maybe_add_burst(self, is_intrinsic: bool = False) -> None:
        """Add a new burst if conditions are met.

        Uses developmental interval parameters to determine burst timing.
        """
        if is_intrinsic:
            max_bursts = self.max_intrinsic_bursts
            interval_range = self.temporal_scales["intrinsic_interval_range"]
            bursts = self.intrinsic_bursts
        else:
            max_bursts = self.max_sensory_bursts
            interval_range = self.temporal_scales["sensory_interval_range"]
            bursts = self.sensory_bursts

        # Use mean interval for Poisson-like burst generation
        mean_interval = (interval_range[0] + interval_range[1]) / 2.0

        if len(bursts) < max_bursts and np.random.random() < self.dt / mean_interval:
            new_burst = self._create_burst(is_intrinsic=is_intrinsic)
            bursts.append(new_burst)

    def update(self, alpha: float = THALAMIC_ALPHA, n_steps: int = 10) -> np.ndarray:
        """Generate combined thalamic input for multiple time steps.

        Args:
            alpha: Balance between intrinsic (0) and sensory (1) activity
            n_steps: Number of integration steps to advance

        Returns:
            Combined thalamic activity pattern
        """
        time_increment = self.dt * n_steps
        self.t += time_increment

        # Add new bursts (check at each sub-step for proper timing)
        for _ in range(n_steps):
            self._maybe_add_burst(is_intrinsic=True)
            self._maybe_add_burst(is_intrinsic=False)

        # Generate and combine activity
        intrinsic = self.generate_intrinsic()
        sensory = self.generate_sensory()
        combined = (1 - alpha) * intrinsic + alpha * sensory

        return combined

    def update_developmental_params(
        self,
        thalamic_spatial_scales: dict | None = None,
        thalamic_temporal_scales: dict | None = None,
        thalamic_modules: dict | None = None,
    ) -> None:
        """Update developmental parameters (e.g., when switching presets).

        Args:
            thalamic_spatial_scales: New spatial scale ranges
            thalamic_temporal_scales: New temporal scale ranges
            thalamic_modules: New module configuration
        """
        if thalamic_spatial_scales is not None:
            self.spatial_scales = thalamic_spatial_scales

        if thalamic_temporal_scales is not None:
            self.temporal_scales = thalamic_temporal_scales

        if thalamic_modules is not None:
            self.n_modules_per_dim = thalamic_modules["n_modules_per_dim"]
            self.jitter_factor = thalamic_modules["jitter_factor"]
            # Regenerate module lattice with new parameters
            self._generate_module_lattice()

    def reset(self) -> None:
        """Reset the generator to initial state."""
        self.t = 0.0
        self.intrinsic_bursts = []
        self.sensory_bursts = []
        self._spatial_cache.clear()
