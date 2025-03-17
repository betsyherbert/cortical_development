"""Main module for running the cortical circuit simulation."""

import argparse
from model.neurons import CorticalCircuit
from model.thalamus import ThalamicInput
from visualization.dashboard import DashboardApp
from model.config import GRID_SIZE, INTEGRATION_STEPS


class CorticalSimulation:
    """Main simulation class that integrates thalamic input with the cortical circuit."""
    
    def __init__(self, grid_size: int = GRID_SIZE):
        """Initialize the cortical simulation.
        
        Args:
            grid_size: Size of the square grid
        """
        self.grid_size = grid_size
        self.circuit = CorticalCircuit(grid_size)
        self.thalamus = ThalamicInput(grid_size)
        self._cache = {}
        
    @property
    def connectivity(self):
        """Access the circuit's connectivity configuration."""
        return self.circuit.connectivity
        
    def update(self, alpha: float = 0.7) -> dict:
        """Update the simulation state for one step.
        
        Args:
            alpha: Weight of intrinsic vs sensory thalamic activity (0-1)
            
        Returns:
            Dictionary containing all population activities
        """
        # Check if we already have a cached thalamic activity for this alpha
        # to avoid redundant computation when the UI is just requesting updates
        cache_key = f"alpha_{alpha:.3f}"
        use_cached = False
        
        # For visualization purposes, reuse the same thalamic input occasionally
        # to reduce computational load if alpha hasn't changed significantly
        if cache_key in self._cache:
            thalamic_activity, last_use = self._cache[cache_key]
            use_cached = last_use < 3  # Use cache for a few consecutive updates
            if use_cached:
                self._cache[cache_key] = (thalamic_activity, last_use + 1)
            
        if not use_cached:
            thalamic_activity = self.thalamus.update(alpha, n_steps=INTEGRATION_STEPS)
            self._cache = {cache_key: (thalamic_activity, 0)}  # Reset cache with new value
        
        self.circuit.thalamus = thalamic_activity
        activities = self.circuit.update(n_steps=INTEGRATION_STEPS)
        activities['thalamus'] = thalamic_activity
        
        return activities
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.circuit.reset()
        self.thalamus.reset()
        self._cache = {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the cortical circuit simulation')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()


def main():
    """Main function to run the simulation."""
    args = parse_arguments()
    sim = CorticalSimulation()
    app = DashboardApp(sim)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main() 