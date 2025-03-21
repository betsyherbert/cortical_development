#!/usr/bin/env python
"""Script to profile the simulation performance."""

from model.neurons import CorticalCircuit
from model.thalamus import ThalamicInput
from model.config import GRID_SIZE, DT, INTEGRATION_STEPS, THALAMIC_ALPHA
import time

def main():
    print("Initializing simulation...")
    
    # Create simulation components
    simulation = CorticalCircuit(GRID_SIZE)
    thalamic_input = ThalamicInput(GRID_SIZE, DT)
    
    # Run the simulation for a few steps
    print(f"Running performance test (grid size: {GRID_SIZE}x{GRID_SIZE}, steps: {INTEGRATION_STEPS})...")
    
    for i in range(5):
        print(f"\nIteration {i+1}:")
        # Generate thalamic input
        thal_start = time.time()
        simulation.thalamus = thalamic_input.update(alpha=THALAMIC_ALPHA)
        thal_time = time.time() - thal_start
        print(f"Thalamic input generation: {thal_time*1000:.2f} ms")
        
        # Update the circuit
        circuit_start = time.time()
        simulation.update()
        circuit_time = time.time() - circuit_start
        print(f"Circuit update: {circuit_time*1000:.2f} ms")
        
        # Total update time
        total_time = thal_time + circuit_time
        print(f"Total iteration time: {total_time*1000:.2f} ms")
        
        if total_time * 1000 > 50:  # Assuming a 50ms update interval
            print("WARNING: Update time exceeds typical refresh interval (50ms)!")
    
    print("\nPerformance profiling complete!")

if __name__ == "__main__":
    main() 