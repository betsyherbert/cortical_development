from model.neurons import CorticalCircuit
from model.thalamus import ThalamicInput
from visualization.dashboard import DashboardApp
import numpy as np

class CorticalSimulation:
    def __init__(self, grid_size: int = 10):
        """
        Initialize the cortical simulation.
        
        Args:
            grid_size: Size of the square grid (default: 10)
        """
        self.grid_size = grid_size
        self.circuit = CorticalCircuit(grid_size)
        self.thalamus = ThalamicInput(grid_size)
        
    def update(self, alpha: float = 0.7) -> dict:
        """
        Update the simulation state.
        
        Args:
            alpha: Weight of intrinsic vs sensory thalamic activity
            
        Returns:
            Dictionary containing all population activities
        """
        # Update thalamic input
        thalamic_activity = self.thalamus.update(alpha)
        self.circuit.thalamus = thalamic_activity
        
        # Get current activities
        activities = self.circuit.get_layer_activities()
        activities['thalamus'] = thalamic_activity
        
        return activities
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.circuit.reset()
        self.thalamus.reset()

def main():
    """Main function to run the simulation."""
    # Create simulation
    sim = CorticalSimulation()
    
    # Create and run dashboard
    app = DashboardApp(sim)
    app.run(debug=True)

if __name__ == "__main__":
    main() 