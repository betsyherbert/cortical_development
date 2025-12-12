"""Main module for running the cortical circuit simulation."""

import argparse

from src.model.config import GRID_SIZE, INTEGRATION_STEPS, THALAMIC_ALPHA, seed_random
from src.model.neurons import CorticalCircuit
from src.model.presets import P0_PRESET
from src.model.thalamus import ThalamicInput
from src.visualization.dashboard import DashboardApp


class CorticalSimulation:
    """Main simulation class that integrates thalamic input with the cortical circuit."""

    def __init__(self, grid_size: int = GRID_SIZE, preset: dict = None):
        """Initialize the cortical simulation.

        Args:
            grid_size: Size of the square grid
            preset: Developmental preset dictionary (defaults to P0_PRESET)
        """
        self.grid_size = grid_size

        # Store preset (default to P0)
        if preset is None:
            preset = P0_PRESET
        self.preset = preset

        # Initialize circuit
        self.circuit = CorticalCircuit(grid_size)

        # Initialize thalamus with developmental parameters from preset
        self.thalamus = ThalamicInput(
            grid_size=grid_size,
            thalamic_spatial_scales=preset.get("thalamic_spatial_scales"),
            thalamic_temporal_scales=preset.get("thalamic_temporal_scales"),
            thalamic_modules=preset.get("thalamic_modules"),
        )

        # Set the random seed for reproducibility
        seed_random()

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
        thalamic_activity = self.thalamus.update(alpha, n_steps=INTEGRATION_STEPS)
        self.circuit.thalamus = thalamic_activity
        activities = self.circuit.update(n_steps=INTEGRATION_STEPS)
        activities["thalamus"] = thalamic_activity
        return activities

    def reset(self):
        """Reset the simulation to initial state."""
        # Reset the random seed to ensure reproducibility
        seed_random()
        self.circuit.reset()
        self.thalamus.reset()

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

    def set_connection_sigma(
        self, source_layer: str, source_cell: str, target_layer: str, target_cell: str, sigma: float
    ) -> None:
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

    def get_connection_sigma(
        self, source_layer: str, source_cell: str, target_layer: str, target_cell: str
    ) -> float:
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

    def set_background_input(self, cell_type: str, value: float) -> None:
        """
        Set the background input for a specific cell type.

        Args:
            cell_type: The cell type to update ('E', 'SST', or 'PV')
            value: Background input value
        """
        self.circuit.set_background_input(cell_type, value)

    def update_thalamic_params(self, preset: dict) -> None:
        """
        Update thalamic parameters from a new developmental preset.

        Args:
            preset: Developmental preset dictionary with thalamic parameters
        """
        self.preset = preset
        self.thalamus.update_developmental_params(
            thalamic_spatial_scales=preset.get("thalamic_spatial_scales"),
            thalamic_temporal_scales=preset.get("thalamic_temporal_scales"),
            thalamic_modules=preset.get("thalamic_modules"),
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the cortical circuit simulation")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


def main():
    """Main function to run the simulation."""
    args = parse_arguments()
    sim = CorticalSimulation()
    app = DashboardApp(sim)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
