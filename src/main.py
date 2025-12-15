"""Main module for running the cortical circuit simulation dashboard.

This is the CLI entrypoint for the interactive dashboard. For programmatic
use of the simulation, import CorticalSimulation from src.simulation.
"""

import argparse

from src.model.config import seed_random
from src.simulation import CorticalSimulation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the cortical circuit simulation")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


def main():
    """Main function to run the simulation."""
    # Import dashboard here to avoid import-time side effects when only
    # importing CorticalSimulation for analysis pipelines
    from src.visualization.dashboard import DashboardApp

    args = parse_arguments()

    # Seed RNG for reproducible dashboard sessions
    seed_random()

    sim = CorticalSimulation()
    app = DashboardApp(sim)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
