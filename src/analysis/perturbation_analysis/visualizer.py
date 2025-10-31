"""Visualization for perturbation analysis results."""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

from .config import (OUTPUT_DIR, DEVELOPMENTAL_STAGES, CELL_TYPES, LAYERS, DPI, COLORBAR_PARAMS, 
                     ANALYSIS_PARAMS, FIGSIZE_TRENDS, FONT_SIZES_TRENDS, CELL_COLORS, LAYER_COLORS,
                     ERROR_BAR_ALPHA, LINE_WIDTH, MARKER_SIZE, SEM_FACTOR, PERTURBATION_CELL_TYPES,
                     REGIMES, PERTURBATION_TYPES)

# Font sizes to match stability analysis
FONT_SIZES = {
    'title': 16,
    'params': 11,
    'subtitle': 13,
}


class PerturbationVisualizer:
    """Handles visualization of perturbation analysis results."""
    
    def __init__(self):
        """Initialize visualizer and ensure output directory exists."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    def _should_show_border(self, perturbation_type: str, cell_type: str, layer: str, analysis_type: str, target_layer: str = None) -> bool:
        """Check if a red border should be shown for this cell type and layer."""
        # Check if this cell type is being perturbed
        cell_type_perturbed = ((perturbation_type == 'SST' and cell_type == 'SST') or
                              (perturbation_type == 'PV' and cell_type == 'PV') or
                              (perturbation_type == 'both' and cell_type in ['SST', 'PV']))
        
        if not cell_type_perturbed:
            return False
            
        # Check if this layer is being perturbed
        if analysis_type == 'column_wise':
            return True  # All layers for column-wise
        elif analysis_type == 'layer_wise' and layer == target_layer:
            return True  # Only target layer for layer-wise
        
        return False

    def calculate_developmental_metrics(self, results: Dict) -> Dict:
        """Calculate paradoxical effect metrics across developmental stages.
        
        Returns metrics organized as:
        metrics[regime][metric_type][grouping][stage] = value
        """
        metrics = {}
        
        for regime in REGIMES:
            metrics[regime] = {
                'paradoxical_magnitude': {'by_celltype': {}, 'by_layer': {}, 'total': {}},
                'normalized_index': {'by_celltype': {}, 'by_layer': {}, 'total': {}},
                'stabilization_strength': {'by_celltype': {}, 'by_layer': {}, 'total': {}}
            }
            
            # Initialize with empty lists for each category
            for celltype in PERTURBATION_CELL_TYPES:
                metrics[regime]['paradoxical_magnitude']['by_celltype'][celltype] = []
                metrics[regime]['normalized_index']['by_celltype'][celltype] = []
                metrics[regime]['stabilization_strength']['by_celltype'][celltype] = []
            
            for layer in LAYERS:
                metrics[regime]['paradoxical_magnitude']['by_layer'][layer] = []
                metrics[regime]['normalized_index']['by_layer'][layer] = []
                metrics[regime]['stabilization_strength']['by_layer'][layer] = []
            
            metrics[regime]['paradoxical_magnitude']['total'] = []
            metrics[regime]['normalized_index']['total'] = []
            metrics[regime]['stabilization_strength']['total'] = []
        
        # Calculate metrics for each developmental stage
        for stage in DEVELOPMENTAL_STAGES:
            stage_data = results[stage]
            
            for regime in REGIMES:
                if regime not in stage_data:
                    continue
                
                regime_data = stage_data[regime]
                stage_metrics = self._calculate_stage_metrics(regime_data, stage)
                
                # Store by cell type
                for celltype in PERTURBATION_CELL_TYPES:
                    if celltype in stage_metrics['by_celltype']:
                        metrics[regime]['paradoxical_magnitude']['by_celltype'][celltype].append(
                            stage_metrics['by_celltype'][celltype]['magnitude'])
                        metrics[regime]['normalized_index']['by_celltype'][celltype].append(
                            stage_metrics['by_celltype'][celltype]['normalized'])
                        metrics[regime]['stabilization_strength']['by_celltype'][celltype].append(
                            stage_metrics['by_celltype'][celltype]['strength'])
                
                # Store by layer
                for layer in LAYERS:
                    if layer in stage_metrics['by_layer']:
                        metrics[regime]['paradoxical_magnitude']['by_layer'][layer].append(
                            stage_metrics['by_layer'][layer]['magnitude'])
                        metrics[regime]['normalized_index']['by_layer'][layer].append(
                            stage_metrics['by_layer'][layer]['normalized'])
                        metrics[regime]['stabilization_strength']['by_layer'][layer].append(
                            stage_metrics['by_layer'][layer]['strength'])
                
                # Store total
                metrics[regime]['paradoxical_magnitude']['total'].append(stage_metrics['total']['magnitude'])
                metrics[regime]['normalized_index']['total'].append(stage_metrics['total']['normalized'])
                metrics[regime]['stabilization_strength']['total'].append(stage_metrics['total']['strength'])
        
        return metrics

    def _calculate_stage_metrics(self, regime_data: Dict, stage: str) -> Dict:
        """Calculate metrics for a single developmental stage and regime."""
        stage_metrics = {
            'by_celltype': {},
            'by_layer': {},
            'total': {'magnitude': [], 'normalized': [], 'strength': []}
        }
        
        # Get perturbation amplitude for strength calculation
        perturbation_amplitude = ANALYSIS_PARAMS['perturbation_amplitude']
        
        # Process each snapshot
        for snapshot_idx, snapshot_data in regime_data.items():
            if isinstance(snapshot_idx, int):  # Skip non-snapshot keys
                
                # Process each perturbation type (SST, PV only)
                for perturbation_type in PERTURBATION_CELL_TYPES:
                    if perturbation_type in snapshot_data:
                        
                        # Column-wise analysis
                        if 'column_wise' in snapshot_data[perturbation_type]:
                            response = snapshot_data[perturbation_type]['column_wise']['response']
                            self._extract_metrics_from_response(
                                response, perturbation_type, perturbation_amplitude, 
                                stage_metrics, 'column_wise')
                        
                        # Layer-wise analysis
                        if 'layer_wise' in snapshot_data[perturbation_type]:
                            for layer in LAYERS:
                                if layer in snapshot_data[perturbation_type]['layer_wise']:
                                    response = snapshot_data[perturbation_type]['layer_wise'][layer]['response']
                                    self._extract_metrics_from_response(
                                        response, perturbation_type, perturbation_amplitude,
                                        stage_metrics, 'layer_wise', target_layer=layer)
        
        # Average across snapshots for final stage values
        final_metrics = {
            'by_celltype': {},
            'by_layer': {},
            'total': {}
        }
        
        # Average by cell type
        for celltype in PERTURBATION_CELL_TYPES:
            if celltype in stage_metrics['by_celltype'] and stage_metrics['by_celltype'][celltype]['magnitude']:
                final_metrics['by_celltype'][celltype] = {
                    'magnitude': np.mean(stage_metrics['by_celltype'][celltype]['magnitude']),
                    'normalized': np.mean(stage_metrics['by_celltype'][celltype]['normalized']),
                    'strength': np.mean(stage_metrics['by_celltype'][celltype]['strength'])
                }
        
        # Average by layer
        for layer in LAYERS:
            if layer in stage_metrics['by_layer'] and stage_metrics['by_layer'][layer]['magnitude']:
                final_metrics['by_layer'][layer] = {
                    'magnitude': np.mean(stage_metrics['by_layer'][layer]['magnitude']),
                    'normalized': np.mean(stage_metrics['by_layer'][layer]['normalized']),
                    'strength': np.mean(stage_metrics['by_layer'][layer]['strength'])
                }
        
        # Average total
        if stage_metrics['total']['magnitude']:
            final_metrics['total'] = {
                'magnitude': np.mean(stage_metrics['total']['magnitude']),
                'normalized': np.mean(stage_metrics['total']['normalized']),
                'strength': np.mean(stage_metrics['total']['strength'])
            }
        
        return final_metrics

    def _extract_metrics_from_response(self, response: Dict, perturbation_type: str, 
                                     perturbation_amplitude: float, stage_metrics: Dict,
                                     analysis_type: str, target_layer: str = None) -> None:
        """Extract paradoxical effect metrics from response data."""
        
        # Initialize if needed
        if perturbation_type not in stage_metrics['by_celltype']:
            stage_metrics['by_celltype'][perturbation_type] = {
                'magnitude': [], 'normalized': [], 'strength': []
            }
        
        if target_layer and target_layer not in stage_metrics['by_layer']:
            stage_metrics['by_layer'][target_layer] = {
                'magnitude': [], 'normalized': [], 'strength': []
            }
        
        # Get baseline rates from simulation for normalization
        # For now, use a typical baseline rate - this could be improved by storing actual baseline
        baseline_rate = 0.1  # Typical baseline firing rate
        
        # Extract response for the perturbed cell type
        for layer in LAYERS:
            if layer in response:
                # Get response of the perturbed cell type (this is where paradoxical effect occurs)
                if perturbation_type in response[layer]:
                    cell_response = response[layer][perturbation_type]
                    
                    # Calculate mean response across spatial locations
                    magnitude = np.mean(cell_response)
                    
                    # Paradoxical effect magnitude (should be negative for true paradoxical effect)
                    stage_metrics['by_celltype'][perturbation_type]['magnitude'].append(magnitude)
                    
                    # Normalized paradoxical index
                    normalized = magnitude / baseline_rate if baseline_rate > 0 else 0
                    stage_metrics['by_celltype'][perturbation_type]['normalized'].append(normalized)
                    
                    # Inhibition stabilization strength (absolute magnitude relative to perturbation)
                    strength = abs(magnitude) / perturbation_amplitude
                    stage_metrics['by_celltype'][perturbation_type]['strength'].append(strength)
                    
                    # Add to layer-specific metrics if this is layer-wise analysis
                    if target_layer and layer == target_layer:
                        stage_metrics['by_layer'][target_layer]['magnitude'].append(magnitude)
                        stage_metrics['by_layer'][target_layer]['normalized'].append(normalized)
                        stage_metrics['by_layer'][target_layer]['strength'].append(strength)
                    
                    # Add to total metrics
                    stage_metrics['total']['magnitude'].append(magnitude)
                    stage_metrics['total']['normalized'].append(normalized)
                    stage_metrics['total']['strength'].append(strength)

    def _plot_trend_with_errorbars(self, ax: plt.Axes, x_pos: List[int], means: List[float], 
                                  color: str, label: str = None) -> None:
        """Plot trend line with error bars."""
        # Estimate SEM as fraction of mean for visualization
        sems = [abs(m) * SEM_FACTOR for m in means]
        
        ax.plot(x_pos, means, 'o-', color=color, label=label, 
               linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
        ax.fill_between(x_pos, [m-s for m,s in zip(means,sems)], [m+s for m,s in zip(means,sems)],
                       color=color, alpha=ERROR_BAR_ALPHA, linewidth=0)

    def _setup_trend_plot_axes(self, ax: plt.Axes, title: str, ylabel: str, 
                              x_pos: List[int], ylim: List[float] = None,
                              show_ylabel: bool = True) -> None:
        """Set up axes for trend plots."""
        ax.set_title(title, fontsize=FONT_SIZES_TRENDS['title'])
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=FONT_SIZES_TRENDS['ylabel'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES, fontsize=FONT_SIZES_TRENDS['tick_labels'])
        
        if ylim:
            ax.set_ylim(ylim)
        
        # Use scientific notation for y-axis to save space
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
        ax.yaxis.get_offset_text().set_fontsize(FONT_SIZES_TRENDS['tick_labels'])
        
        # Add horizontal line at zero for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    def _add_trend_legends(self, fig: plt.Figure, axes: np.ndarray) -> None:
        """Add legends to the right of trend plots."""
        plt.subplots_adjust(right=0.8)
        
        # Get legend handles and labels from both subplots
        handles1, labels1 = axes[0].get_legend_handles_labels()
        handles2, labels2 = axes[1].get_legend_handles_labels()
        
        if handles1:
            fig.legend(handles1, labels1, loc='center left', bbox_to_anchor=(0.82, 0.75), title='Cell Types')
        if handles2:
            fig.legend(handles2, labels2, loc='center left', bbox_to_anchor=(0.82, 0.25), title='Layers')

    def _calculate_global_ylim(self, metrics: Dict, metric_type: str, regime: str) -> List[float]:
        """Calculate global y-limits for consistent scaling across subplots."""
        all_values = []
        
        # Collect all values for this metric and regime
        for grouping in ['by_celltype', 'by_layer', 'total']:
            if grouping == 'total':
                all_values.extend(metrics[regime][metric_type][grouping])
            else:
                for key in metrics[regime][metric_type][grouping]:
                    all_values.extend(metrics[regime][metric_type][grouping][key])
        
        if not all_values:
            return [-0.1, 0.1]
        
        min_val = min(all_values)
        max_val = max(all_values)
        margin = max(abs(min_val), abs(max_val)) * 0.1
        
        return [min_val - margin, max_val + margin]

    def create_paradoxical_magnitude_trends(self, metrics: Dict) -> None:
        """Create developmental trends for paradoxical effect magnitude."""
        for regime in REGIMES:
            self._create_single_trend_figure(
                metrics, 'paradoxical_magnitude', regime,
                'Paradoxical Effect Magnitude', 'Change in Firing Rate (Hz)',
                f'paradoxical_magnitude_trends_{regime}.svg'
            )

    def create_normalized_index_trends(self, metrics: Dict) -> None:
        """Create developmental trends for normalized paradoxical index."""
        for regime in REGIMES:
            self._create_single_trend_figure(
                metrics, 'normalized_index', regime,
                'Normalized Paradoxical Index', 'Normalized Change',
                f'normalized_index_trends_{regime}.svg'
            )

    def create_stabilization_strength_trends(self, metrics: Dict) -> None:
        """Create developmental trends for inhibition stabilization strength."""
        for regime in REGIMES:
            self._create_single_trend_figure(
                metrics, 'stabilization_strength', regime,
                'Inhibition Stabilization Strength', 'Response/Perturbation Ratio',
                f'stabilization_strength_trends_{regime}.svg'
            )

    def _create_single_trend_figure(self, metrics: Dict, metric_type: str, regime: str,
                                   title: str, ylabel: str, filename: str) -> None:
        """Create a single trend figure with 1x3 subplot layout."""
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRENDS)
        fig.suptitle(f'{title} - {regime.title()} Regime', fontsize=FONT_SIZES_TRENDS['title'], fontweight='bold')
        
        x_pos = list(range(len(DEVELOPMENTAL_STAGES)))
        
        # Calculate global y-limits for consistency
        ylim = self._calculate_global_ylim(metrics, metric_type, regime)
        
        # Plot 1: By cell type
        for cell_type in PERTURBATION_CELL_TYPES:
            if cell_type in metrics[regime][metric_type]['by_celltype']:
                means = metrics[regime][metric_type]['by_celltype'][cell_type]
                if means:  # Only plot if we have data
                    self._plot_trend_with_errorbars(axes[0], x_pos, means, CELL_COLORS[cell_type], cell_type)
        
        self._setup_trend_plot_axes(axes[0], 'By Cell Type', ylabel, x_pos, ylim)
        
        # Plot 2: By layer
        for layer in LAYERS:
            if layer in metrics[regime][metric_type]['by_layer']:
                means = metrics[regime][metric_type]['by_layer'][layer]
                if means:  # Only plot if we have data
                    self._plot_trend_with_errorbars(axes[1], x_pos, means, LAYER_COLORS[layer], layer)
        
        self._setup_trend_plot_axes(axes[1], 'By Layer', '', x_pos, ylim, show_ylabel=False)
        
        # Plot 3: Total network
        if metrics[regime][metric_type]['total']:
            total_values = metrics[regime][metric_type]['total']
            self._plot_trend_with_errorbars(axes[2], x_pos, total_values, 'black')
        
        self._setup_trend_plot_axes(axes[2], 'Total Network', '', x_pos, ylim, show_ylabel=False)
        
        plt.tight_layout()
        self._add_trend_legends(fig, axes)
        
        # Save plot
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, format='svg', dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Saved developmental trend plot: {filename}")

    def generate_developmental_trends(self, results: Dict) -> None:
        """Generate all developmental trend plots."""
        print("Calculating developmental metrics...")
        metrics = self.calculate_developmental_metrics(results)
        
        print("Generating developmental trend plots...")
        self.create_paradoxical_magnitude_trends(metrics)
        self.create_normalized_index_trends(metrics)
        self.create_stabilization_strength_trends(metrics)
        
        print("Developmental trend visualization complete!")
        
    def create_perturbation_figure(self, results: Dict, regime: str, snapshot_idx: int, 
                                 perturbation_type: str, analysis_type: str, target_layer: str = None):
        """Create figure showing responses across all developmental stages.
        
        Args:
            results: Analysis results across all stages
            regime: Analysis regime ('driven' or 'idle')
            snapshot_idx: Index of the snapshot
            perturbation_type: Type of perturbation ('SST', 'PV', 'both')
            analysis_type: Type of analysis ('layer_wise' or 'column_wise')
            target_layer: For layer-wise analysis, the specific layer being perturbed
        """
        # Setup figure with increased height and 4 rows x 12 columns (4 stages x 3 cell types)
        fig = plt.figure(figsize=(24, 10))  # Increased height from 8 to 10
        
        # Create custom grid layout with spacing between supracolumns
        gs = fig.add_gridspec(4, 15, hspace=0.1, wspace=0.05,  # Reduced row spacing from 0.2 to 0.1
                             left=0.06, right=0.88, top=0.85, bottom=0.12)  # More space at top for age labels
        
        # Map columns to skip gaps at positions 3, 7, 11 (creating 4 supracolumns of 3 each)
        col_map = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]  # Skip 3, 7, 11
        axes = np.empty((4, 12), dtype=object)
        
        for row in range(4):
            for col_idx in range(12):
                axes[row, col_idx] = fig.add_subplot(gs[row, col_map[col_idx]])
        
        # Format perturbation type for display
        perturbation_display = {'SST': 'SST', 'PV': 'PV', 'both': 'PV & SST'}[perturbation_type]
        
        # Fix analysis type display (avoid "wisewise" issue)
        if analysis_type == 'layer_wise':
            analysis_display = 'Layer-wise'
        else:
            analysis_display = 'Column-wise'
        
        # Main title (bold, black) - include target layer in perturbation type for layer-wise analysis
        if analysis_type == 'layer_wise' and target_layer:
            perturbation_with_layer = f'{perturbation_display} {target_layer}'
            main_title = f'{analysis_display} perturbation   |   {regime.title()}   |   {perturbation_with_layer}'
        else:
            main_title = f'{analysis_display} perturbation   |   {regime.title()}   |   {perturbation_display}'
        fig.suptitle(main_title, fontsize=FONT_SIZES['title'], fontweight='bold', y=0.95)
        
        # Parameter subtitle (non-bold, grey)
        patch_size = ANALYSIS_PARAMS[f'{analysis_type.split("_")[0]}_patch_size']
        param_text = (f'snapshot {snapshot_idx + 1} | patch size {patch_size}×{patch_size} | '
                     f'amplitude {ANALYSIS_PARAMS["perturbation_amplitude"]} | '
                     f'duration {ANALYSIS_PARAMS["perturbation_duration"]:.0f}ms | '
                     f'delay {ANALYSIS_PARAMS["post_perturbation_delay"]:.0f}ms | '
                     f'window {ANALYSIS_PARAMS["measurement_window"]:.0f}ms')
        fig.text(0.5, 0.91, param_text, ha='center', va='center', 
                fontsize=FONT_SIZES['params'], color='grey', transform=fig.transFigure)
        
        # Get response data for all stages and perturbation target info
        stage_responses = {}
        thalamic_inputs = {}
        
        # Get target coordinates and patch size (same across all stages)
        first_stage_data = results[DEVELOPMENTAL_STAGES[0]][regime][snapshot_idx]
        target_coords = first_stage_data['snapshot_info']['targets'][analysis_type]
        patch_size = ANALYSIS_PARAMS[f'{analysis_type.split("_")[0]}_patch_size']
        
        for stage in DEVELOPMENTAL_STAGES:
            stage_data = results[stage][regime][snapshot_idx]
            
            # Handle different data structures for layer-wise vs column-wise
            if analysis_type == 'layer_wise' and target_layer:
                stage_responses[stage] = stage_data[perturbation_type][analysis_type][target_layer]['response']
            else:
                stage_responses[stage] = stage_data[perturbation_type][analysis_type]['response']
            
            thalamic_inputs[stage] = stage_data['snapshot_info']['thalamic_input']
        
        # Plot responses for each layer and stage
        for row_idx, layer in enumerate(LAYERS):
            for stage_idx, stage in enumerate(DEVELOPMENTAL_STAGES):
                for cell_idx, cell_type in enumerate(CELL_TYPES):
                    col_idx = stage_idx * 3 + cell_idx
                    ax = axes[row_idx, col_idx]
                    
                    response_data = stage_responses[stage][layer][cell_type]
                    
                    im = ax.imshow(response_data, cmap='RdBu_r', 
                                 vmin=COLORBAR_PARAMS['response_min'], 
                                 vmax=COLORBAR_PARAMS['response_max'])
                    
                    # Add red border around perturbation patch if this cell type and layer are perturbed
                    should_show_border = self._should_show_border(perturbation_type, cell_type, layer, analysis_type, target_layer)
                    
                    if should_show_border:
                        # Calculate rectangle coordinates (matplotlib uses bottom-left origin)
                        # target_coords are (x, y) center coordinates
                        half_size = patch_size // 2
                        rect_x = target_coords[1] - half_size  # target_coords[1] is y-coord, becomes x in plot
                        rect_y = target_coords[0] - half_size  # target_coords[0] is x-coord, becomes y in plot
                        
                        # Add red border rectangle
                        from matplotlib.patches import Rectangle
                        rect = Rectangle((rect_x, rect_y), patch_size, patch_size, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                    
                    # Add cell type labels only on top row
                    if row_idx == 0:
                        ax.set_title(cell_type, fontsize=10)
                    
                    # Add layer labels on first column
                    if col_idx == 0:
                        ax.set_ylabel(layer, fontsize=10, fontweight='bold')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        # Add age labels above SST columns (first column of each supracolumn)
        supracolumn_positions = [0, 3, 6, 9]  # First columns of each supracolumn (SST columns)
        for stage_idx, stage in enumerate(DEVELOPMENTAL_STAGES):
            sst_col_position = supracolumn_positions[stage_idx]
            x_pos = 0.06 + (col_map[sst_col_position] / 15) * 0.82  # Adjust for 15-column grid
            fig.text(x_pos, 0.88, f'$\\mathbf{{{stage}}}$', ha='center', va='center',  # Moved up from 0.86 to 0.88
                    fontsize=FONT_SIZES['subtitle'], fontweight='bold', transform=fig.transFigure)
        
        # Plot thalamic input in bottom row
        for stage_idx, stage in enumerate(DEVELOPMENTAL_STAGES):
            for cell_idx in range(3):  # Fill all 3 columns for consistency
                col_idx = stage_idx * 3 + cell_idx
                ax = axes[3, col_idx]
                
                if cell_idx == 1:  # Plot thalamic data in middle column only
                    thalamic_data = thalamic_inputs[stage]
                    thal_im = ax.imshow(thalamic_data, cmap='gray', vmin=0, vmax=1)  # Black-to-white, 0-1 range
                    if col_idx == 1:  # Add label to first occurrence
                        ax.set_ylabel('Thalamic Input', fontsize=10, fontweight='bold')
                else:
                    ax.axis('off')  # Hide unused subplots
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Calculate colorbar positions based on actual subplot positions
        # Response colorbar spans first 3 rows
        top_row_top = axes[0, 0].get_position().y1
        third_row_bottom = axes[2, 0].get_position().y0
        response_height = top_row_top - third_row_bottom
        
        # Thalamic colorbar spans bottom row only
        bottom_row_top = axes[3, 0].get_position().y1
        bottom_row_bottom = axes[3, 0].get_position().y0
        thalamic_height = bottom_row_top - bottom_row_bottom
        
        # Add response colorbar (spans first 3 rows) - made thinner
        cbar_ax1 = fig.add_axes([0.90, third_row_bottom, 0.015, response_height])  # Width reduced from 0.02 to 0.015
        response_cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r', 
                                    norm=plt.Normalize(COLORBAR_PARAMS['response_min'], 
                                                     COLORBAR_PARAMS['response_max'])), 
                                   cax=cbar_ax1, label='Change in response')
        # Format response colorbar ticks to 2 decimal places, fewer ticks
        response_ticks = [COLORBAR_PARAMS['response_min'], 0, COLORBAR_PARAMS['response_max']]
        response_cbar.set_ticks(response_ticks)
        response_cbar.set_ticklabels([f'{tick:.2f}' for tick in response_ticks])
        
        # Add thalamic colorbar (spans bottom row only) - made thinner
        cbar_ax2 = fig.add_axes([0.90, bottom_row_bottom, 0.015, thalamic_height])  # Width reduced from 0.02 to 0.015
        thalamic_cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='gray', 
                                    norm=plt.Normalize(0, 1)), 
                                   cax=cbar_ax2, label='Thalamic Input')
        # Format thalamic colorbar ticks to 2 decimal places, fewer ticks
        thalamic_ticks = [0, 0.5, 1]
        thalamic_cbar.set_ticks(thalamic_ticks)
        thalamic_cbar.set_ticklabels([f'{tick:.2f}' for tick in thalamic_ticks])
        
        # Save figure - include target layer for layer-wise analysis
        if analysis_type == 'layer_wise' and target_layer:
            filename = f'{perturbation_type}_{analysis_type}_{target_layer}_{regime}_snapshot{snapshot_idx + 1}.svg'
        else:
            filename = f'{perturbation_type}_{analysis_type}_{regime}_snapshot{snapshot_idx + 1}.svg'
        
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, format='svg', dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"      Saved: {filename}") 