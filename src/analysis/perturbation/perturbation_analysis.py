"""Core perturbation analysis implementation."""

import numpy as np
from typing import Dict, List, Tuple

from src.main import CorticalSimulation
from .config import ANALYSIS_PARAMS, DEVELOPMENTAL_STAGES, CELL_TYPES, LAYERS, PRESETS, PERTURBATION_TYPES, REGIMES
from src.model.config import DT


class PerturbationAnalysis:
    """Main class for perturbation analysis testing the paradoxical effect."""
    
    def __init__(self):
        """Initialize analysis with simulation instance."""
        self.simulation = CorticalSimulation()
        self.params = ANALYSIS_PARAMS
        
    def collect_snapshots_and_targets(self, stage: str) -> Dict[str, List[Dict]]:
        """Collect snapshots and identify perturbation targets for both regimes."""
        # Reset simulation and apply preset
        self.simulation.reset()
        self._apply_preset(PRESETS[stage])
        
        n_steps = int(self.params['duration'] * 1000 / DT)
        sample_interval = max(1, n_steps // 200)
        
        thalamic_buffer = []
        snapshots_targets = {'idle': [], 'driven': []}
        thresholds_computed = False
        idle_threshold = 0.0
        driven_threshold = 0.0
        
        for step in range(n_steps):
            activities = self.simulation.update()
            thalamic_sum = np.sum(activities['thalamus'])
            thalamic_buffer.append(thalamic_sum)
            
            # Compute thresholds after burn-in period
            if not thresholds_computed and step > n_steps // 4:
                idle_threshold = np.percentile(thalamic_buffer, self.params['percentiles'][0])
                driven_threshold = np.percentile(thalamic_buffer, self.params['percentiles'][1])
                thresholds_computed = True
            
            # Collect snapshots at intervals
            if thresholds_computed and step % sample_interval == 0:
                regime = None
                if thalamic_sum <= idle_threshold and len(snapshots_targets['idle']) < self.params['n_snapshots']:
                    regime = 'idle'
                elif thalamic_sum >= driven_threshold and len(snapshots_targets['driven']) < self.params['n_snapshots']:
                    regime = 'driven'
                
                if regime:
                    snapshot_data = self._capture_snapshot_with_targets(step, activities, regime)
                    snapshots_targets[regime].append(snapshot_data)
                
                # Early termination when enough snapshots collected
                if all(len(snapshots_targets[r]) >= self.params['n_snapshots'] for r in REGIMES):
                    break
        
        return snapshots_targets
    
    def _apply_preset(self, preset: Dict):
        """Apply developmental preset to simulation."""
        # Update connections
        for key, value in preset['connection_strengths'].items():
            self.simulation.circuit.connectivity.layer_params[key] = {'amplitude': value, 'sigma': 2.0}
        self.simulation.circuit.connectivity.update_weights()
        
        # Update cell properties  
        for cell_type in CELL_TYPES:
            if cell_type in preset['time_constants']:
                self.simulation.set_time_constant(cell_type, preset['time_constants'][cell_type])
            if cell_type in preset['gains']:
                self.simulation.set_gain(cell_type, preset['gains'][cell_type])
    
    def _capture_snapshot_with_targets(self, frame_idx: int, activities: Dict, regime: str) -> Dict:
        """Capture snapshot and identify target locations for both analysis types."""
        # Capture voltages only (rates not needed)
        voltages = {}
        for layer in LAYERS:
            voltages[layer] = {cell_type: self.simulation.circuit.layers[layer].V[cell_type].copy() 
                             for cell_type in CELL_TYPES}
        
        # Find target locations for both patch sizes
        layer_target = self._find_target_patch(activities['thalamus'], regime, self.params['layer_patch_size'])
        column_target = self._find_target_patch(activities['thalamus'], regime, self.params['column_patch_size'])
        
        return {
            'frame_idx': frame_idx,
            'voltages': voltages,
            'thalamic_input': activities['thalamus'].copy(),
            'targets': {'layer_wise': layer_target, 'column_wise': column_target},
            'time_constants': self.simulation.get_time_constants().copy(),
            'gains': self.simulation.get_gains().copy(),
            'regime': regime
        }
    
    def _find_target_patch(self, thalamic_input: np.ndarray, regime: str, patch_size: int) -> Tuple[int, int]:
        """Find optimal patch center based on thalamic activity."""
        boundary = self.params['boundary_exclude']
        region_size = self.params['target_region_size']
        
        # Find valid search area
        search_start = boundary
        search_end = self.simulation.grid_size - boundary - max(region_size, patch_size) + 1
        
        if search_end <= search_start:
            center = self.simulation.grid_size // 2
            return (center, center)
        
        # Search for optimal region
        best_activity = float('inf') if regime == 'idle' else float('-inf')
        best_center = (self.simulation.grid_size // 2, self.simulation.grid_size // 2)
        
        for x in range(search_start, search_end):
            for y in range(search_start, search_end):
                activity = np.mean(thalamic_input[x:x+region_size, y:y+region_size])
                if ((regime == 'idle' and activity < best_activity) or 
                    (regime == 'driven' and activity > best_activity)):
                    best_activity = activity
                    best_center = (x + region_size // 2, y + region_size // 2)
        
        # Ensure patch center allows full patch within bounds
        patch_half = patch_size // 2
        patch_boundary = boundary + patch_half
        max_center = self.simulation.grid_size - boundary - patch_half - 1
        
        final_x = np.clip(best_center[0], patch_boundary, max_center)
        final_y = np.clip(best_center[1], patch_boundary, max_center)
        
        return (final_x, final_y)
    
    def _should_perturb_cell_type(self, perturbation_type: str, cell_type: str) -> bool:
        """Check if a cell type should be perturbed based on perturbation type."""
        return ((perturbation_type == 'SST' and cell_type == 'SST') or
                (perturbation_type == 'PV' and cell_type == 'PV') or
                (perturbation_type == 'both' and cell_type in ['SST', 'PV']))
    
    def _calculate_response(self, baseline_rates: Dict, perturbed_rates: Dict) -> Dict:
        """Calculate response as difference between perturbed and baseline rates."""
        return {layer: {cell_type: perturbed_rates[layer][cell_type] - baseline_rates[layer][cell_type]
                       for cell_type in CELL_TYPES} for layer in LAYERS}
    
    def _apply_perturbation(self, perturbation_type: str, center: Tuple[int, int], patch_size: int, target_layer: str = None) -> Dict:
        """Generate perturbation input for specified cell types.
        
        Args:
            perturbation_type: Type of cells to perturb ('SST', 'PV', 'both')
            center: Center coordinates of perturbation patch
            patch_size: Size of perturbation patch
            target_layer: If specified, only perturb this layer (for layer-wise analysis)
        """
        perturbation = {}
        
        # Create spatial mask using slices
        half_size = patch_size // 2
        x_slice = slice(center[0] - half_size, center[0] + half_size)
        y_slice = slice(center[1] - half_size, center[1] + half_size)
        
        for layer in LAYERS:
            perturbation[layer] = {}
            for cell_type in CELL_TYPES:
                mask = np.zeros((self.simulation.grid_size, self.simulation.grid_size))
                
                # Apply perturbation based on type and layer constraint
                if self._should_perturb_cell_type(perturbation_type, cell_type):
                    # For layer-wise analysis, only perturb the specified layer
                    if target_layer is None or layer == target_layer:
                        mask[x_slice, y_slice] = self.params['perturbation_amplitude']
                
                perturbation[layer][cell_type] = mask
        
        return perturbation
    
    def _save_simulation_state(self) -> Dict:
        """Save complete simulation state including noise generators for frozen noise experiments."""
        state = {
            'voltages': {},
            'rates': {},
            'noise_states': {},
            'thalamic_state': {},
            'random_state': np.random.get_state()
        }
        
        # Save neural states for all layers
        for layer in LAYERS:
            layer_obj = self.simulation.circuit.layers[layer]
            state['voltages'][layer] = {cell_type: layer_obj.V[cell_type].copy() for cell_type in CELL_TYPES}
            state['rates'][layer] = {cell_type: layer_obj.r[cell_type].copy() for cell_type in CELL_TYPES}
            
            # Save noise generator states
            noise_gen = layer_obj.noise_generator
            state['noise_states'][layer] = {
                'private_noise': {cell_type: noise_gen.private_noise[cell_type].copy() for cell_type in CELL_TYPES},
                'shared_noise': noise_gen.shared_noise.copy(),
                'mean': noise_gen.mean.copy(),
                'std': noise_gen.std.copy(),
                'correlation': noise_gen.correlation.copy()
            }
        
        # Save thalamic state
        thalamic_obj = self.simulation.thalamus
        state['thalamic_state'] = {
            't': thalamic_obj.t,
            'intrinsic_bursts': [burst.copy() for burst in thalamic_obj.intrinsic_bursts],
            'sensory_bursts': [burst.copy() for burst in thalamic_obj.sensory_bursts]
        }
        
        return state
    
    def _restore_simulation_state(self, state: Dict) -> None:
        """Restore complete simulation state to ensure identical conditions for frozen noise experiments."""
        # Restore random state first
        np.random.set_state(state['random_state'])
        
        # Restore neural states for all layers
        for layer in LAYERS:
            layer_obj = self.simulation.circuit.layers[layer]
            
            # Restore voltages and rates
            for cell_type in CELL_TYPES:
                layer_obj.V[cell_type][:] = state['voltages'][layer][cell_type]
                layer_obj.r[cell_type][:] = state['rates'][layer][cell_type]
            
            # Restore noise generator states
            noise_gen = layer_obj.noise_generator
            noise_state = state['noise_states'][layer]
            for cell_type in CELL_TYPES:
                noise_gen.private_noise[cell_type][:] = noise_state['private_noise'][cell_type]
            noise_gen.shared_noise[:] = noise_state['shared_noise']
            noise_gen.mean = noise_state['mean'].copy()
            noise_gen.std = noise_state['std'].copy()
            noise_gen.correlation = noise_state['correlation'].copy()
        
        # Restore thalamic state
        thalamic_obj = self.simulation.thalamus
        thalamic_obj.t = state['thalamic_state']['t']
        thalamic_obj.intrinsic_bursts = [burst.copy() for burst in state['thalamic_state']['intrinsic_bursts']]
        thalamic_obj.sensory_bursts = [burst.copy() for burst in state['thalamic_state']['sensory_bursts']]
    
    def _run_simulation_with_perturbation(self, snapshot: Dict, perturbation: Dict = None) -> Dict:
        """Run simulation with optional perturbation and return averaged rates using frozen noise and thalamic input."""
        # Restore snapshot voltages
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                self.simulation.circuit.layers[layer].V[cell_type][:] = snapshot['voltages'][layer][cell_type]
        
        # Save the complete simulation state for frozen noise experiments
        initial_state = self._save_simulation_state()
        
        # Use frozen thalamic input from snapshot
        frozen_thalamic = snapshot['thalamic_input']
        
        # Calculate simulation steps
        dt = 1.5
        perturbation_steps = int(self.params['perturbation_duration'] / dt)
        delay_steps = int(self.params['post_perturbation_delay'] / dt)
        measurement_steps = int(self.params['measurement_window'] / dt)
        total_steps = perturbation_steps + delay_steps + measurement_steps
        
        # Initialize measurement collection
        measurement_data = {layer: {cell_type: [] for cell_type in CELL_TYPES} for layer in LAYERS}
        
        for step in range(total_steps):
            # Calculate inputs for all layers (similar to CorticalCircuit.update)
            grid_shape = (self.simulation.grid_size, self.simulation.grid_size)
            
            # Process each layer
            for target_layer in LAYERS:
                # Initialize inputs for this layer
                layer_inputs = {cell_type: np.zeros(grid_shape) for cell_type in CELL_TYPES}
                
                # Add thalamic inputs using frozen thalamic pattern
                for target_cell in CELL_TYPES:
                    conn_key = ('thalamus', None, target_layer, target_cell)
                    if conn_key in self.simulation.circuit.connectivity.W:
                        weight_matrix = self.simulation.circuit.connectivity.W[conn_key]
                        thalamic_rates = frozen_thalamic.flatten()
                        input_curr = weight_matrix @ thalamic_rates
                        layer_inputs[target_cell] += input_curr.reshape(grid_shape)
                
                # Add lateral connections between layers
                for source_layer in LAYERS:
                    source_rates = self.simulation.circuit.layers[source_layer].r
                    for source_cell in CELL_TYPES:
                        for target_cell in CELL_TYPES:
                            conn_key = (source_layer, source_cell, target_layer, target_cell)
                            if conn_key in self.simulation.circuit.connectivity.W:
                                weight_matrix = self.simulation.circuit.connectivity.W[conn_key]
                                source_rates_flat = source_rates[source_cell].flatten()
                                input_curr = weight_matrix @ source_rates_flat
                                layer_inputs[target_cell] += input_curr.reshape(grid_shape)
                
                # Add perturbation as input current during perturbation window
                if perturbation and step < perturbation_steps:
                    for cell_type in CELL_TYPES:
                        layer_inputs[cell_type] += perturbation[target_layer][cell_type]
                
                # Update this layer with calculated inputs
                self.simulation.circuit.layers[target_layer].update(layer_inputs)
            
            # Collect measurement data during measurement window
            if step >= perturbation_steps + delay_steps:
                for layer in LAYERS:
                    for cell_type in CELL_TYPES:
                        measurement_data[layer][cell_type].append(
                            self.simulation.circuit.layers[layer].r[cell_type].copy())
        
        # Return time-averaged rates
        return {layer: {cell_type: np.mean(measurement_data[layer][cell_type], axis=0)
                       for cell_type in CELL_TYPES} for layer in LAYERS}
    
    def _analyze_snapshot(self, snapshot: Dict, stage: str) -> Dict:
        """Analyze single snapshot across all perturbation types and analysis scales with frozen noise."""
        snapshot['stage'] = stage
        
        results = {
            'snapshot_info': {
                'frame_idx': snapshot['frame_idx'],
                'regime': snapshot['regime'],
                'targets': snapshot['targets'],
                'thalamic_input': snapshot['thalamic_input']
            }
        }
        
        # Restore snapshot state and save complete state for frozen noise experiments
        for layer in LAYERS:
            for cell_type in CELL_TYPES:
                self.simulation.circuit.layers[layer].V[cell_type][:] = snapshot['voltages'][layer][cell_type]
        
        frozen_state = self._save_simulation_state()
        
        # Run baseline simulation once with frozen state
        self._restore_simulation_state(frozen_state)
        baseline_rates = self._run_simulation_with_perturbation(snapshot)
        
        # Test each perturbation type
        for perturbation_type in PERTURBATION_TYPES:
            results[perturbation_type] = {}
            
            # Column-wise analysis (perturb all layers)
            target = snapshot['targets']['column_wise']
            perturbation = self._apply_perturbation(perturbation_type, target, self.params['column_patch_size'])
            
            # Restore frozen state before perturbed simulation
            self._restore_simulation_state(frozen_state)
            perturbed_rates = self._run_simulation_with_perturbation(snapshot, perturbation)
            response = self._calculate_response(baseline_rates, perturbed_rates)
            results[perturbation_type]['column_wise'] = {'target': target, 'response': response}
            
            # Layer-wise analysis (perturb one layer at a time)
            results[perturbation_type]['layer_wise'] = {}
            target = snapshot['targets']['layer_wise']
            
            for layer in LAYERS:
                perturbation = self._apply_perturbation(perturbation_type, target, self.params['layer_patch_size'], target_layer=layer)
                
                # Restore frozen state before each layer-specific perturbed simulation
                self._restore_simulation_state(frozen_state)
                perturbed_rates = self._run_simulation_with_perturbation(snapshot, perturbation)
                response = self._calculate_response(baseline_rates, perturbed_rates)
                results[perturbation_type]['layer_wise'][layer] = {'target': target, 'response': response}
        
        return results
    
    def run_analysis(self) -> Dict:
        """Run perturbation analysis with age-specific snapshots but cross-stage visualization."""
        print("Starting perturbation analysis...")
        
        from .visualizer import PerturbationVisualizer
        visualizer = PerturbationVisualizer()
        
        # Collect snapshots separately for each regime (using P4 as reference for consistency)
        print("Collecting snapshots...")
        self.simulation.reset()
        self._apply_preset(PRESETS['P4'])
        snapshots_targets = self.collect_snapshots_and_targets('P4')
        
        total_snapshots = sum(len(snapshots_targets.get(regime, [])) for regime in REGIMES)
        print(f"Collected {total_snapshots} snapshots across {len(REGIMES)} regimes")
        
        # Initialize complete results structure for developmental trends
        complete_results = {stage: {regime: {} for regime in REGIMES} for stage in DEVELOPMENTAL_STAGES}
        
        # Process each snapshot across all stages
        results_summary = {}
        
        for regime in REGIMES:
            if regime not in snapshots_targets:
                continue
                
            results_summary[regime] = {}
            
            for snap_idx, snapshot in enumerate(snapshots_targets[regime]):
                print(f"Processing {regime} snapshot {snap_idx + 1}/{len(snapshots_targets[regime])}...")
                
                # Collect results across all developmental stages
                snapshot_data = {}
                for stage in DEVELOPMENTAL_STAGES:
                    self._apply_preset(PRESETS[stage])
                    analysis_result = self._analyze_snapshot(snapshot, stage)
                    
                    # Store for individual snapshot figures
                    snapshot_data[stage] = {regime: {snap_idx: analysis_result}}
                    
                    # Store for developmental trend analysis
                    complete_results[stage][regime][snap_idx] = analysis_result
                
                # Generate figures immediately after collecting all stage data
                for perturbation_type in PERTURBATION_TYPES:
                    # Column-wise analysis (single figure)
                    visualizer.create_perturbation_figure(
                        snapshot_data, regime, snap_idx, perturbation_type, 'column_wise')
                    
                    # Layer-wise analysis (separate figure for each layer)
                    for layer in LAYERS:
                        visualizer.create_perturbation_figure(
                            snapshot_data, regime, snap_idx, perturbation_type, 'layer_wise', target_layer=layer)
                
                results_summary[regime][snap_idx] = f"Completed across all {len(DEVELOPMENTAL_STAGES)} stages"
                del snapshot_data  # Free memory for individual snapshot
        
        # Generate developmental trend plots using complete results
        print("\nGenerating developmental trend plots...")
        visualizer.generate_developmental_trends(complete_results)
        
        print("Perturbation analysis complete!")
        return results_summary 