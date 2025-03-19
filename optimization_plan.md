# Neural Network Visualization Optimization Plan

Based on the performance profiling results, we've identified several bottlenecks affecting the smoothness of the visualization. Below is a detailed optimization plan targeting these issues.

## Primary Bottleneck: Plotly Heatmap Creation

The creation of Plotly heatmap figures is the most significant bottleneck, taking ~15-47ms depending on caching:

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| First figure creation | 32.50 | Initialization overhead |
| Subsequent figures | ~1.5 | Much faster after initialization |
| Total for 10 figures | 46.59 | Exceeds the 50ms target update interval |

### Recommendations:

1. **Reuse Plotly Figures Instead of Recreating**
   ```python
   # Instead of creating new figures on each update:
   fig = create_heatmap(data, cell_type)
   
   # Update existing figures using Plotly.update:
   fig.data[0].z = data  # Update just the data
   ```

2. **Use Efficient Update Patterns**
   - Create all figures once during initialization
   - Store them in a dictionary for easy access
   - Only update the `z` data property during callbacks
   - Use `uirevision` parameter to prevent unwanted redraws

3. **Implement Staggered Updates**
   - Update a subset of plots each frame
   - For example, update half the plots in one cycle, half in the next

## Secondary Bottleneck: Neural Simulation

The neural simulation takes ~10ms per update:

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Thalamic input | 0.09 | Very efficient |
| Circuit update | 10.12 | Main simulation cost |
| Total | 10.21 | ~40% of the update cycle |

### Recommendations:

1. **Optimize Matrix Operations**
   - Review connectivity computation loops
   - Consider using sparse matrices for connectivity
   - Memoize calculation results where possible

2. **Adjust Update Frequency**
   - Increase `INTEGRATION_STEPS` slightly for faster visual changes
   - Increase `DT` to speed up simulation time (with slightly less accuracy)

3. **Balance Computational Fidelity and Speed**
   - Add user options to reduce grid resolution (e.g., 15x15 instead of 20x20)
   - Consider multi-resolution approach: simulate at lower resolution, interpolate for display

## Dashboard Optimization

General improvements to the dashboard architecture:

1. **Implement Double Buffering**
   - Compute updates in a background thread or process
   - Swap buffers when update is complete to avoid visual stuttering

2. **Optimize Callback Structure**
   - Reduce the number of outputs in a single callback
   - Group outputs by update frequency (fast vs. slow)
   - Consider using clientside callbacks for UI elements

3. **Add Performance Controls**
   - Add a "performance mode" toggle that reduces visual fidelity for smoother operation
   - Provide grid size control in the dashboard
   - Add an FPS counter to help users optimize settings

## Implementation Priority

1. **High Priority** (immediate improvements):
   - Reuse Plotly figures instead of recreating them
   - Implement figure caching
   - Optimize connectivity computation with better caching

2. **Medium Priority** (significant improvements):
   - Add user controls for simulation parameters
   - Implement staggered updates
   - Review and optimize matrix operations

3. **Lower Priority** (nice-to-have):
   - Background thread processing
   - Multi-resolution simulation
   - UI performance optimizations 