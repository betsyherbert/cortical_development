"""Main module for running the cortical circuit simulation."""

import argparse
from model.neurons import CorticalCircuit
from model.thalamus import ThalamicInput
from visualization.dashboard import DashboardApp
from model.config import (
    GRID_SIZE, INTEGRATION_STEPS, THALAMIC_ALPHA
)

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
        
    def update(self, alpha: float = THALAMIC_ALPHA) -> dict:
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
    
    def set_time_constant(self, cell_type: str, tau: float) -> None:
        """
        Set the membrane time constant for a specific cell type.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            tau: New time constant value in milliseconds
        """
        self.circuit.set_time_constant(cell_type, tau)
    
    def get_time_constants(self) -> dict:
        """
        Get current time constant values for all cell types.
        
        Returns:
            Dictionary mapping cell types to their time constants
        """
        return self.circuit.get_time_constants()
        
    def set_gain(self, cell_type: str, gain: float) -> None:
        """
        Set the gain for a specific cell type.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            gain: New gain value
        """
        self.circuit.set_gain(cell_type, gain)
    
    def get_gains(self) -> dict:
        """
        Get current gain values for all cell types.
        
        Returns:
            Dictionary mapping cell types to their gains
        """
        return self.circuit.get_gains()
        
    def set_connection_sigma(self, source_layer: str, source_cell: str, 
                           target_layer: str, target_cell: str, 
                           sigma: float) -> None:
        """
        Set the connection width (sigma) for a specific connection.
        
        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            sigma: Connection width (Gaussian sigma)
        """
        self.circuit.connectivity.set_connection_sigma(
            source_layer, source_cell, target_layer, target_cell, sigma
        )
        
    def get_connection_sigma(self, source_layer: str, source_cell: str, 
                           target_layer: str, target_cell: str) -> float:
        """
        Get the current connection width (sigma) for a specific connection.
        
        Args:
            source_layer: Source layer ('L23', 'L4', 'L5', or 'thalamus')
            source_cell: Source cell type ('E', 'SST', 'PV', or None for thalamus)
            target_layer: Target layer ('L23', 'L4', 'L5')
            target_cell: Target cell type ('E', 'SST', 'PV')
            
        Returns:
            Current connection sigma value
        """
        return self.circuit.connectivity.get_connection_sigma(
            source_layer, source_cell, target_layer, target_cell
        )
    
    def set_strength_scaling(self, cell_type: str, scaling: float) -> None:
        """
        Set the strength scaling factor for a cell type's outgoing connections.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            scaling: Strength scaling factor (0 to 5)
        """
        self.circuit.connectivity.set_strength_scaling(cell_type, scaling)
    
    def get_strength_scaling(self, cell_type: str) -> float:
        """
        Get the current strength scaling factor for a cell type's outgoing connections.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            
        Returns:
            Current strength scaling factor
        """
        return self.circuit.connectivity.get_strength_scaling(cell_type)
    
    def set_sparsity(self, cell_type: str, sparsity: float) -> None:
        """
        Set the sparsity factor for a cell type's outgoing connections.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            sparsity: Sparsity factor (0 to 1, where 1 means all connections present)
        """
        self.circuit.connectivity.set_sparsity(cell_type, sparsity)
    
    def get_sparsity(self, cell_type: str) -> float:
        """
        Get the current sparsity factor for a cell type's outgoing connections.
        
        Args:
            cell_type: Cell type ('E', 'SST', 'PV', or 'thalamus')
            
        Returns:
            Current sparsity factor
        """
        return self.circuit.connectivity.get_sparsity(cell_type)
        
    def set_noise_params(self, cell_type: str, mean: float, std: float, c: float) -> None:
        """
        Set the noise parameters for a specific cell type.
        
        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            mean: Mean of the noise
            std: Standard deviation of the noise
            c: Correlation coefficient
        """
        self.circuit.set_noise_parameters(cell_type, mean, std, c)


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

# Create the server variable that will be used by gunicorn
sim = CorticalSimulation()
app = DashboardApp(sim)
server = app.app  # Dash apps expose their Flask server through the app attribute

if __name__ == "__main__":
    main() 