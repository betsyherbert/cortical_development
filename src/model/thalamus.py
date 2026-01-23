"""Thalamic input module for the cortical circuit simulation.

This module generates realistic thalamic activity patterns using a single
burst process whose statistics change smoothly with developmental time (alpha).

Developmental changes controlled by alpha ∈ [0,1]:
- Spatial scales (sigma) interpolate from broad (early) to narrow (late)
- Temporal scales (duration, interval) interpolate from long/sparse to short/frequent
- Jitter factor interpolates from low to high variability

Note: All spatial parameters (sigma) are in μm (anatomical units).
"""

import numpy as np

from .config import (
    ANATOMICAL_GRID_SIZE,
    DT,
    GRID_SIZE,
    THALAMIC_ALPHA,
    THALAMIC_AMP_RANGE,
    THALAMIC_N_MODULES_PER_DIM,
)


# Developmental endpoint parameters (early = alpha=0, late = alpha=1)
# These are derived from the original P0 intrinsic and P15 sensory parameters
SIGMA_RANGE_EARLY = (100.0, 300.0)  # μm - broad bursts at early development
SIGMA_RANGE_LATE = (10.0, 80.0)  # μm - narrow bursts at late development
DURATION_RANGE_EARLY = (200.0, 300.0)  # ms - long bursts at early development
DURATION_RANGE_LATE = (50.0, 150.0)  # ms - short bursts at late development
INTERVAL_RANGE_EARLY = (200.0, 400.0)  # ms - sparse bursts at early development
INTERVAL_RANGE_LATE = (50.0, 200.0)  # ms - frequent bursts at late development
JITTER_FACTOR_EARLY = 0.2  # low spatial variability
JITTER_FACTOR_LATE = 0.8  # high spatial variability


def _lerp(a: float, b: float, alpha: float) -> float:
    """Linear interpolation between a and b based on alpha."""
    return a * (1.0 - alpha) + b * alpha


def _lerp_tuple(
    early: tuple[float, float], late: tuple[float, float], alpha: float
) -> tuple[float, float]:
    """Linear interpolation between two (min, max) tuples."""
    return (_lerp(early[0], late[0], alpha), _lerp(early[1], late[1], alpha))


class ThalamicInput:
    """Generates thalamic activity patterns with alpha-controlled development.

    This class implements developmentally realistic thalamic input with:
    - Module-based spatial organization of burst centers
    - Alpha-controlled interpolation of spatial/temporal scales
    - Smooth temporal profiles (raised cosine bumps)
    - Single burst process with summation

    Note: All sigma (spatial width) parameters are in μm (anatomical units) and are
    converted internally to grid units for computation.
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        dt: float = DT,
        anatomical_grid_size: float = ANATOMICAL_GRID_SIZE,
        n_modules_per_dim: int = THALAMIC_N_MODULES_PER_DIM,
    ):
        """Initialize the thalamic input generator.

        Args:
            grid_size: Number of grid points in each dimension
            dt: Time step in milliseconds
            anatomical_grid_size: Anatomical size of the grid in μm
            n_modules_per_dim: Number of modules per dimension for burst organization
        """
        self.grid_size = grid_size
        self.dt = dt
        self.t = 0.0
        self.anatomical_grid_size = anatomical_grid_size
        self.grid_scale = anatomical_grid_size / grid_size  # μm per grid unit

        # Module lattice parameters
        self.n_modules_per_dim = n_modules_per_dim

        # Set up spatial coordinate grid
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        self.coords = np.stack([x, y])
        self._spatial_cache = {}

        # Single burst list
        self.bursts = []
        self.max_bursts = 10

        # Pre-compute components
        self._prepare_gaussian_components()
        self._generate_module_lattice()

    def _prepare_gaussian_components(self):
        """Pre-compute coordinate arrays for faster Gaussian calculation."""
        self.x_coords, self.y_coords = self.coords

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

    def _pick_burst_center(self, alpha: float) -> tuple[float, float]:
        """Pick a burst center from the module lattice with alpha-dependent jitter.

        Args:
            alpha: Developmental time parameter [0, 1]

        Returns:
            Tuple of (x, y) coordinates in grid units, clipped to grid bounds
        """
        # Randomly select a module center
        idx = int(np.random.randint(len(self.module_centers)))
        module_center = self.module_centers[idx]

        # Interpolate jitter factor based on alpha
        jitter_factor = _lerp(JITTER_FACTOR_EARLY, JITTER_FACTOR_LATE, alpha)
        jitter_sigma = jitter_factor * self.module_spacing

        # Add Gaussian jitter
        jitter_x = np.random.normal(0, jitter_sigma)
        jitter_y = np.random.normal(0, jitter_sigma)

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

    def _generate_burst_activity(self) -> tuple[np.ndarray, list[dict]]:
        """Generate activity from the burst list by summing all active bursts.

        Returns:
            Tuple of (activity pattern, list of still-active bursts)
        """
        activity = np.zeros((self.grid_size, self.grid_size))
        active_bursts = []

        for burst in self.bursts:
            time_since_start = self.t - burst["start_time"]

            if 0 <= time_since_start < burst["duration"]:
                # Calculate temporal profile
                temporal = self._compute_temporal_profile(time_since_start, burst["duration"])

                # Calculate spatial profile
                spatial = self.gaussian_spatial(burst["center"], burst["sigma"])

                # Combine temporal and spatial, sum across bursts
                burst_activity = burst["amplitude"] * temporal * spatial
                activity += burst_activity

                active_bursts.append(burst)

        # Ensure non-negative activity
        activity = np.maximum(0, activity)

        return activity, active_bursts

    def _create_burst(self, alpha: float) -> dict:
        """Create a new burst with alpha-interpolated parameters.

        Args:
            alpha: Developmental time parameter [0, 1]

        Returns:
            Dictionary with burst parameters
        """
        # Pick center from module lattice with alpha-dependent jitter
        center = self._pick_burst_center(alpha)

        # Interpolate parameter ranges based on alpha
        sigma_range = _lerp_tuple(SIGMA_RANGE_EARLY, SIGMA_RANGE_LATE, alpha)
        duration_range = _lerp_tuple(DURATION_RANGE_EARLY, DURATION_RANGE_LATE, alpha)

        # Sample from interpolated ranges
        sigma_um = np.random.uniform(sigma_range[0], sigma_range[1])
        duration = np.random.uniform(duration_range[0], duration_range[1])
        amplitude = np.random.uniform(THALAMIC_AMP_RANGE[0], THALAMIC_AMP_RANGE[1])

        burst = {
            "center": center,
            "sigma": sigma_um,  # in μm
            "duration": duration,  # in ms
            "amplitude": amplitude,  # varies per burst
            "start_time": self.t,
        }

        return burst

    def _maybe_add_burst(self, alpha: float) -> None:
        """Add a new burst probabilistically based on alpha-dependent interval.

        Args:
            alpha: Developmental time parameter [0, 1]
        """
        # Interpolate interval range based on alpha
        interval_range = _lerp_tuple(INTERVAL_RANGE_EARLY, INTERVAL_RANGE_LATE, alpha)
        mean_interval = (interval_range[0] + interval_range[1]) / 2.0

        # Poisson-like burst generation
        if len(self.bursts) < self.max_bursts and np.random.random() < self.dt / mean_interval:
            new_burst = self._create_burst(alpha)
            self.bursts.append(new_burst)

    def update(self, alpha: float = THALAMIC_ALPHA, n_steps: int = 10) -> np.ndarray:
        """Generate thalamic input for multiple time steps.

        Args:
            alpha: Developmental time parameter [0, 1] controlling burst statistics
            n_steps: Number of integration steps to advance

        Returns:
            Thalamic activity pattern (2D array)
        """
        time_increment = self.dt * n_steps
        self.t += time_increment

        # Add new bursts (check at each sub-step for proper timing)
        for _ in range(n_steps):
            self._maybe_add_burst(alpha)

        # Generate activity by summing all active bursts
        activity, self.bursts = self._generate_burst_activity()

        return activity

    def reset(self) -> None:
        """Reset the generator to initial state."""
        self.t = 0.0
        self.bursts = []
        self._spatial_cache.clear()
