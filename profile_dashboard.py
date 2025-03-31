#!/usr/bin/env python
"""Script to profile the dashboard visualization performance."""

from model.neurons import CorticalCircuit
from model.thalamus import ThalamicInput
from model.config import (
    GRID_SIZE, DT, CELL_TYPES, LAYERS, COLORMAPS, THALAMIC_ALPHA,
    CELL_COLORS, CELL_ACTIVITY_COLORS
)
import plotly.graph_objects as go
import numpy as np
import time

def create_heatmap(data, cell_type):
    """Mimic the dashboard's heatmap creation function."""
    colorscale = COLORMAPS.get(cell_type, [[0, 'black'], [1, 'gray']])
    
    # Set appropriate range for each cell type
    if cell_type == 'thalamus':
        zmax = 2.0
    elif cell_type == 'E':
        zmax = 0.8
    else:  # SST and PV
        zmax = 0.8
    
    return go.Figure(
        data=[go.Heatmap(
            z=data,
            colorscale=colorscale,
            showscale=False,
            hoverinfo='none',
            zmin=0,
            zmax=zmax
        )],
        layout=go.Layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=150,
            width=150,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
    )

def main():
    print("Initializing simulation...")
    
    # Create simulation components
    simulation = CorticalCircuit(GRID_SIZE)
    thalamic_input = ThalamicInput(GRID_SIZE, DT)
    simulation.thalamus = thalamic_input.update(alpha=THALAMIC_ALPHA)
    
    # Run simulation once to get activities
    activities = simulation.update()
    
    # Profile heatmap creation
    print("\nProfiling heatmap creation:")
    start_time = time.time()
    figures = []
    
    for layer in LAYERS:
        for cell_type in CELL_TYPES:
            data = activities[layer][cell_type]
            fig_start = time.time()
            fig = create_heatmap(data, cell_type)
            fig_time = time.time() - fig_start
            print(f"  {layer}_{cell_type} heatmap creation: {fig_time*1000:.2f} ms")
            figures.append(fig)
    
    # Add thalamus figure
    fig_start = time.time()
    figures.append(create_heatmap(activities['thalamus'], 'thalamus'))
    fig_time = time.time() - fig_start
    print(f"  Thalamus heatmap creation: {fig_time*1000:.2f} ms")
    
    total_time = time.time() - start_time
    print(f"Total heatmap creation time: {total_time*1000:.2f} ms")
    
    # Simulate full update cycle
    print("\nSimulating full update cycle:")
    full_start = time.time()
    
    # Generate thalamic input
    thal_start = time.time()
    simulation.thalamus = thalamic_input.update(alpha=THALAMIC_ALPHA)
    thal_time = time.time() - thal_start
    print(f"1. Thalamic update: {thal_time*1000:.2f} ms")
    
    # Update neural circuit
    circuit_start = time.time()
    activities = simulation.update()
    circuit_time = time.time() - circuit_start
    print(f"2. Circuit update: {circuit_time*1000:.2f} ms")
    
    # Create all heatmaps
    heatmap_start = time.time()
    figures = []
    for layer in LAYERS:
        for cell_type in CELL_TYPES:
            figures.append(create_heatmap(activities[layer][cell_type], cell_type))
    figures.append(create_heatmap(activities['thalamus'], 'thalamus'))
    heatmap_time = time.time() - heatmap_start
    print(f"3. Heatmap creation: {heatmap_time*1000:.2f} ms")
    
    full_time = time.time() - full_start
    print(f"\nTotal update cycle time: {full_time*1000:.2f} ms")
    
    if full_time * 1000 > 50:  # Assuming 50ms update interval
        print(f"WARNING: Update cycle exceeds typical refresh interval (50ms) by {full_time*1000-50:.2f} ms!")
        print(f"Bottleneck: {'Heatmap creation' if heatmap_time > max(thal_time, circuit_time) else 'Neural simulation'}")
    
    print("\nPerformance profiling complete!")

if __name__ == "__main__":
    main() 