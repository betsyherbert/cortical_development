"""Visualization module for bifurcation analysis results."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plots from showing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .config import (
    DEVELOPMENTAL_STAGES, OUTPUT_DIR, COLORMAP, FIGSIZE,
    ANALYSIS_PARAMS, STAGE_COLORS
)
from src.model.config import CELL_COLORS
from src.analysis.common import DPI


class BifurcationVisualizer:
    """Creates publication-quality visualizations for bifurcation analysis results."""
    
    def __init__(self):
        """Initialize visualizer with output directories and configuration."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.output_dir = OUTPUT_DIR
        self.dpi = DPI
        
    def plot_developmental_comparison(self, results: Dict[str, Dict]) -> None:
        """
        Create multi-panel figure comparing developmental stages.
        
        Args:
            results: Dictionary mapping stage names to analysis results
        """
        stages = list(results.keys())
        n_stages = len(stages)
        n_populations = 3  # Always E, SST, PV
        
        # Extract data (convert distance to max real eigenvalue)
        distances = [results[s]['distance'] for s in stages]
        max_real_eigenvalues = [-d for d in distances]  # Convert distance to max real eigenvalue
        critical_ks = [results[s]['critical_k'] for s in stages]
        
        # Extract connection strengths for heatmap
        network_params = results[stages[0]]['network_params']
        n_layers = network_params.get('n_layers', 1)
        pop_names = network_params['pop_names']  # ['E', 'SST', 'PV']
        layers_list = network_params.get('layers', ['L4'])
        
        # Prepare connection data based on network type
        if n_layers == 1:
            # Single layer: show all 3x3 connections
            conn_matrix = np.zeros((n_stages, 9))
            conn_labels = ['E→E', 'E→SST', 'E→PV', 'SST→E', 'SST→SST', 'SST→PV',
                           'PV→E', 'PV→SST', 'PV→PV']
            for i, stage in enumerate(stages):
                A = results[stage]['network_params']['A']
                conn_matrix[i, :] = A.flatten()
            full_conn_matrices = None
        else:
            # Multi-layer: show full connectivity matrices for each stage
            # Build full matrices: 9x9 for 3 layers × 3 populations
            full_conn_matrices = []
            for stage in stages:
                A = results[stage]['network_params']['A']
                full_conn_matrices.append(A.copy())
            
            # Create labels for full matrix: layer_population format
            conn_labels = []
            for layer in layers_list:
                for src_pop in pop_names:
                    for layer_tgt in layers_list:
                        for tgt_pop in pop_names:
                            conn_labels.append(f'{layer}_{src_pop}→{layer_tgt}_{tgt_pop}')
            
            conn_matrix = None
        
        # Extract time constants (show per population type, not per layer)
        tau_matrix = np.zeros((n_stages, n_populations))
        tau_labels = network_params['pop_names']
        for i, stage in enumerate(stages):
            tau = results[stage]['network_params']['tau']
            # Take first layer's time constants (they're repeated for each layer)
            tau_matrix[i, :] = tau[:n_populations]
        
        # Extract spatial scales (show per population type)
        sigma_matrix = np.zeros((n_stages, n_populations))
        for i, stage in enumerate(stages):
            sigma = results[stage]['network_params']['sigma']
            # Take first layer's spatial scales (use source population's width, which is column-wise)
            # For each population type, take its outgoing width from the first layer
            for j, pop in enumerate(tau_labels):
                # Column index for this population in first layer
                col_idx = j
                # Get the sigma value (should be same for all targets from this source)
                # Use any row from the first layer - they're all the same for a given source column
                if sigma.shape[0] > 0 and sigma.shape[1] > col_idx:
                    sigma_matrix[i, j] = sigma[0, col_idx]
                else:
                    sigma_matrix[i, j] = 0
        
        # Create figure with subplots
        # Adjust grid layout based on whether we show full matrices
        if n_layers > 1:
            # For full network: need more space for connection matrices
            # Use 4 rows: stability/spatial scale (row 0), full matrices (row 1-2), time/sigma (row 3)
            fig = plt.figure(figsize=(FIGSIZE[0], FIGSIZE[1] + 2))
            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.4, height_ratios=[1, 1.5, 1.5, 1])
        else:
            fig = plt.figure(figsize=FIGSIZE)
            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.4)
        
        # Panel: Stability Margin
        ax_a = fig.add_subplot(gs[0, 0])
        stage_colors = [STAGE_COLORS.get(s, '#666666') for s in stages]
        bars = ax_a.bar(range(n_stages), max_real_eigenvalues, color=stage_colors)
        ax_a.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax_a.set_xticks(range(n_stages))
        ax_a.set_xticklabels(stages)
        ax_a.set_ylabel('Max Re(λ)', fontsize=10)
        ax_a.set_title('Stability Margin', fontsize=10, fontweight='normal')
        ax_a.grid(True, alpha=0.3, axis='y')
        # Format tick labels to limit decimals and right-justify
        ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        for label in ax_a.get_yticklabels():
            label.set_ha('right')
        
        # Panel: Dominant Spatial Scale (line graph with circular markers)
        ax_b = fig.add_subplot(gs[0, 1])
        # Color code by pattern type: k=0 = global, k>0 = patterned
        x_pos = range(n_stages)
        for i, (k, stage) in enumerate(zip(critical_ks, stages)):
            color = '#8E44AD' if k == 0 else '#16A085'
            ax_b.plot(i, k, marker='o', markersize=8, color=color, 
                     markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        
        # Connect points with lines
        ax_b.plot(x_pos, critical_ks, linestyle='-', linewidth=1.5, 
                 color='gray', alpha=0.5, zorder=1)
        
        ax_b.set_xticks(range(n_stages))
        ax_b.set_xticklabels(stages)
        ax_b.set_ylabel('Critical k', fontsize=10)
        ax_b.set_title('Dominant Spatial Scale', fontsize=10, fontweight='normal')
        ax_b.grid(True, alpha=0.3, axis='y')
        # Format tick labels to limit decimals and right-justify
        ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        for label in ax_b.get_yticklabels():
            label.set_ha('right')
        
        # Add legend for pattern types (outside figure)
        global_patch = mpatches.Patch(color='#8E44AD', label='Global (k=0)')
        pattern_patch = mpatches.Patch(color='#16A085', label='Patterned (k>0)')
        ax_b.legend(handles=[global_patch, pattern_patch], 
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Compute global colorbar limits for fair comparison across all stages
        if n_layers == 1:
            conn_min = conn_matrix.min()
            conn_max = conn_matrix.max()
        else:
            # Find min/max across all full matrices
            conn_min = min([A.min() for A in full_conn_matrices])
            conn_max = max([A.max() for A in full_conn_matrices])
        
        # Panel: Connection strengths heatmap
        if n_layers == 1:
            # Single layer: show standard heatmap
            ax_c = fig.add_subplot(gs[1, :])
            im = ax_c.imshow(conn_matrix, aspect='auto', cmap=COLORMAP, 
                             interpolation='nearest', vmin=conn_min, vmax=conn_max)
            ax_c.set_xticks(range(len(conn_labels)))
            ax_c.set_xticklabels(conn_labels, rotation=45, ha='right', fontsize=9)
            ax_c.set_yticks(range(n_stages))
            ax_c.set_yticklabels(stages, fontsize=10)
            ax_c.set_title('Connection Strengths', fontsize=10, fontweight='normal')
            cbar = plt.colorbar(im, ax=ax_c)
            cbar.set_label('Strength', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        else:
            # Full network: show one full 9x9 matrix per stage
            # Create subplots for each stage
            for i, stage in enumerate(stages):
                ax_c = fig.add_subplot(gs[1 + (i // 2), i % 2])
                A_stage = full_conn_matrices[i]
                
                im = ax_c.imshow(A_stage, aspect='auto', cmap=COLORMAP, 
                                 interpolation='nearest', vmin=conn_min, vmax=conn_max)
                
                # Create tick labels: layer_population format
                tick_labels = []
                for layer in layers_list:
                    for pop in pop_names:
                        tick_labels.append(f'{layer}\n{pop}')
                
                # Set ticks and labels
                ax_c.set_xticks(range(len(tick_labels)))
                ax_c.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=7)
                ax_c.set_yticks(range(len(tick_labels)))
                ax_c.set_yticklabels(tick_labels, fontsize=7)
                ax_c.set_title(f'{stage} Connection Strengths', fontsize=9, fontweight='normal')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
                cbar.set_label('Strength', fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                
                # Add grid lines to separate layers
                for j in range(1, n_layers):
                    line_pos = j * n_populations - 0.5
                    ax_c.axhline(line_pos, color='white', linewidth=1.5, alpha=0.8)
                    ax_c.axvline(line_pos, color='white', linewidth=1.5, alpha=0.8)
        
        # Panel: Time constants and connection widths
        # Split into two side-by-side subplots for clarity
        # Adjust row index based on whether we show full matrices
        if n_layers > 1:
            # Full matrices take up rows 1-2, so time/sigma go to row 3
            ax_d_tau = fig.add_subplot(gs[3, 0])
            ax_d_sigma = fig.add_subplot(gs[3, 1])
        else:
            ax_d_tau = fig.add_subplot(gs[3, 0])
            ax_d_sigma = fig.add_subplot(gs[3, 1])
        x = np.arange(n_stages)
        width = 0.25
        
        # Time constants subplot
        for i, label in enumerate(tau_labels):
            color = CELL_COLORS.get(label, '#666666') if label in CELL_COLORS else '#666666'
            offset = (i - 1) * width  # n_populations = 3, so offset = (i - 1) * width
            ax_d_tau.bar(x + offset, tau_matrix[:, i], width, label=f'τ_{label}', color=color, alpha=0.7)
        ax_d_tau.set_xticks(x)
        ax_d_tau.set_xticklabels(stages, fontsize=10)
        ax_d_tau.set_ylabel('τ (ms)', fontsize=10)
        ax_d_tau.set_title('Time Constants', fontsize=10, fontweight='normal')
        ax_d_tau.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax_d_tau.grid(True, alpha=0.3, axis='y')
        # Format tick labels to limit decimals and right-justify
        ax_d_tau.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        for label in ax_d_tau.get_yticklabels():
            label.set_ha('right')
        
        # Connection widths subplot
        for i, label in enumerate(tau_labels):
            color = CELL_COLORS.get(label, '#999999') if label in CELL_COLORS else '#999999'
            offset = (i - 1) * width  # n_populations = 3, so offset = (i - 1) * width
            ax_d_sigma.bar(x + offset, sigma_matrix[:, i], width, 
                          label=f'σ_{label}', color=color, alpha=0.7)
        ax_d_sigma.set_xticks(x)
        ax_d_sigma.set_xticklabels(stages, fontsize=10)
        ax_d_sigma.set_ylabel('σ (Δx)', fontsize=10)
        ax_d_sigma.set_title('Connection Widths', fontsize=10, fontweight='normal')
        ax_d_sigma.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax_d_sigma.grid(True, alpha=0.3, axis='y')
        # Format tick labels to limit decimals and right-justify
        ax_d_sigma.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        for label in ax_d_sigma.get_yticklabels():
            label.set_ha('right')
        
        
        # Save figure as SVG with clear scope and mode
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine scope (layer or full network)
        layers = network_params.get('layers', ['L4'])
        scope = layers[0] if len(layers) == 1 else 'full'
        
        # Determine mode (silent, driven, or both)
        analysis_mode = results[stages[0]].get('analysis_mode', 'silent')
        if analysis_mode == 'both':
            mode_suffix = 'silent'  # Top-level fields are silent results
            mode_display = 'Silent'
        else:
            mode_suffix = analysis_mode
            mode_display = mode_suffix.capitalize()
        
        # Create meaningful title
        if scope == 'full':
            title = f'Full Network — {mode_display}'
        else:
            title = f'{scope} Layer — {mode_display}'
        
        # Update suptitle with meaningful title
        plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        
        # Adjust layout to accommodate legends outside
        plt.tight_layout(rect=[0, 0, 0.95, 0.98])
        
        output_path = Path(self.output_dir) / f'{scope}_development_{mode_suffix}.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def plot_eigenvalue_spectra(self, all_results: Dict[str, Dict], mode_override: Optional[str] = None) -> None:
        """
        Plot eigenvalue spectra for all stages in the complex plane.
        
        Args:
            all_results: Dictionary mapping stage names to analysis results
            mode_override: Optional mode override ('silent' or 'driven') when mode='both'
        """
        # Use square figure size with reasonable dimensions
        # Make it slightly smaller than FIGSIZE for better proportions
        square_size = min(FIGSIZE[0], FIGSIZE[1]) * 0.9
        fig, axes = plt.subplots(2, 2, figsize=(square_size, square_size))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        axes = axes.flatten()
        
        stages = ['P4', 'P8', 'P12', 'P16']
        
        # First pass: collect all eigenvalues to compute global xlim/ylim
        all_eigenvalues = []
        stage_results = []
        
        for idx, stage in enumerate(stages):
            if stage not in all_results:
                continue
                
            result = all_results[stage]
            network_params = result['network_params']
            total_pops = len(network_params['tau'])
            
            # Determine which steady state to use for visualization
            analysis_mode = result.get('analysis_mode', 'silent')
            if analysis_mode == 'both' and mode_override == 'driven' and 'driven' in result:
                steady_state = result['driven']['steady_state']
                critical_mode = result['driven']['critical_mode']
                critical_k = result['driven']['critical_k']
            else:
                steady_state = result['steady_state']
                critical_mode = result['critical_mode']
                critical_k = result['critical_k']
            
            # Compute effective gains
            threshold = 1e-10
            active_mask = steady_state > threshold
            gain_eff = network_params['gain'] * active_mask.astype(float)
            
            # Compute eigenvalues for critical mode
            n1_crit, n2_crit = critical_mode
            k_squared_crit = n1_crit**2 + n2_crit**2
            grid_size = ANALYSIS_PARAMS['grid_size']
            domain_length = ANALYSIS_PARAMS.get('domain_length', grid_size)
            
            J_crit = np.zeros((total_pops, total_pops))
            for i in range(total_pops):
                for j in range(total_pops):
                    sigma_ij = network_params['sigma'][i, j] / domain_length
                    w_tilde = network_params['A'][i, j] * np.exp(
                        -2 * np.pi**2 * k_squared_crit * sigma_ij**2
                    )
                    if i == j:
                        J_crit[i, j] = (-1.0 / network_params['tau'][i] + 
                                       (gain_eff[i] * w_tilde) / network_params['tau'][i])
                    else:
                        J_crit[i, j] = (gain_eff[i] * w_tilde) / network_params['tau'][i]
            
            eigenvalues = np.linalg.eigvals(J_crit)
            all_eigenvalues.append(eigenvalues)
            stage_results.append({
                'stage': stage,
                'eigenvalues': eigenvalues,
                'critical_k': critical_k,
                'ax': axes[idx]
            })
        
        # Compute global xlim/ylim with padding
        if all_eigenvalues:
            all_real = np.concatenate([eig.real for eig in all_eigenvalues])
            all_imag = np.concatenate([eig.imag for eig in all_eigenvalues])
            
            real_range = all_real.max() - all_real.min()
            imag_range = all_imag.max() - all_imag.min()
            
            # Ensure minimum range to avoid singular transformation
            min_range = 0.1
            if real_range < min_range:
                real_range = min_range
            if imag_range < min_range:
                imag_range = min_range
            
            x_padding = real_range * 0.1
            y_padding = imag_range * 0.1
            
            xlim = (all_real.min() - x_padding, all_real.max() + x_padding)
            ylim = (all_imag.min() - y_padding, all_imag.max() + y_padding)
            
            # Ensure symmetric limits
            x_max = max(abs(xlim[0]), abs(xlim[1]), min_range / 2)
            y_max = max(abs(ylim[0]), abs(ylim[1]), min_range / 2)
            
            # Use the larger range to ensure square limits
            max_range = max(x_max, y_max)
            xlim = (-max_range, max_range)
            ylim = (-max_range, max_range)
        else:
            xlim = (-1, 1)
            ylim = (-1, 1)
        
        # Second pass: plot with standardized axes
        for sr in stage_results:
            ax = sr['ax']
            stage = sr['stage']
            eigenvalues = sr['eigenvalues']
            critical_k = sr['critical_k']
            
            stage_color = STAGE_COLORS.get(stage, '#666666')
            ax.scatter(eigenvalues.real, eigenvalues.imag, s=50, 
                      c=stage_color, alpha=0.7)
            ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Re(λ)', fontsize=10)
            ax.set_ylabel('Im(λ)', fontsize=10)
            ax.set_title(f'{stage} (k={critical_k:.1f})', fontsize=10, fontweight='normal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # Force square aspect ratio with adjustable='box' to ensure true squares
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            # Format tick labels to limit decimals and right-justify y-axis
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            for label in ax.get_yticklabels():
                label.set_ha('right')
        
        # Determine scope (layer or full network) from first result
        first_result = all_results[list(all_results.keys())[0]]
        network_params = first_result['network_params']
        layers = network_params.get('layers', ['L4'])
        scope = layers[0] if len(layers) == 1 else 'full'
        
        # Determine mode (silent, driven, or both)
        analysis_mode = first_result.get('analysis_mode', 'silent')
        if analysis_mode == 'both':
            # Use mode_override if provided, otherwise default to silent
            mode_suffix = mode_override if mode_override else 'silent'
            mode_display = mode_override.capitalize() if mode_override else 'Silent'
        else:
            mode_suffix = analysis_mode
            mode_display = mode_suffix.capitalize()
        
        # Create meaningful title
        title = f'Eigenvalue Spectra - {mode_display} ({scope})'
        
        # Add clear top-level title
        plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        
        # Apply tight_layout first for spacing
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Manually adjust each subplot to be square
        for ax in axes:
            pos = ax.get_position()
            # Get current dimensions
            width = pos.x1 - pos.x0
            height = pos.y1 - pos.y0
            # Use the smaller dimension to ensure square
            size = min(width, height)
            # Center the square subplot
            center_x = (pos.x0 + pos.x1) / 2
            center_y = (pos.y0 + pos.y1) / 2
            # Set position to be centered square
            ax.set_position([center_x - size/2, center_y - size/2, size, size])
        
        # Save figure as SVG with clear scope and mode
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = Path(self.output_dir) / f'{scope}_spectrum_{mode_suffix}.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def plot_layer_coupling_comparison(self, l5_results: Dict[str, Dict],
                                       l4_results: Dict[str, Dict], 
                                       l23_results: Dict[str, Dict],
                                       full_results: Dict[str, Dict]) -> None:
        """
        Compare stability across layer isolation conditions.
        
        Args:
            l5_results: Results for L5-only analysis
            l4_results: Results for L4-only analysis
            l23_results: Results for L23-only analysis
            full_results: Results for full network (all layers) analysis
        """
        stages = ['P4', 'P8', 'P12', 'P16']
        
        # Convert distances to max real eigenvalues
        l5_max_real = [-l5_results[s]['distance'] for s in stages]
        l4_max_real = [-l4_results[s]['distance'] for s in stages]
        l23_max_real = [-l23_results[s]['distance'] for s in stages]
        full_max_real = [-full_results[s]['distance'] for s in stages]
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        x = np.arange(len(stages))
        width = 0.2
        
        # Layer colors
        layer_colors = {
            'L5': '#999999',
            'L4': '#555555',
            'L23': '#222222',
            'Full': '#4472c4'
        }
        
        bars1 = ax.bar(x - 1.5*width, l23_max_real, width, 
                      label='L23', color=layer_colors['L23'], alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, l4_max_real, width, 
                      label='L4', color=layer_colors['L4'], alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, l5_max_real, width, 
                      label='L5', color=layer_colors['L5'], alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, full_max_real, width, 
                      label='Full', color=layer_colors['Full'], alpha=0.8)
        
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Stage', fontsize=10)
        ax.set_ylabel('Max Re(λ)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Format tick labels to limit decimals and right-justify
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        for label in ax.get_yticklabels():
            label.set_ha('right')
        
        # Determine mode (silent, driven, or both)
        first_result = full_results[list(full_results.keys())[0]]
        analysis_mode = first_result.get('analysis_mode', 'silent')
        if analysis_mode == 'both':
            mode_suffix = 'silent'  # Top-level fields are silent results
            mode_display = 'Silent'
        else:
            mode_suffix = analysis_mode
            mode_display = mode_suffix.capitalize()
        
        # Create meaningful title
        title = f'Layer Comparison — {mode_display}'
        
        # Add clear top-level title
        plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        
        # Adjust layout to accommodate legend outside
        plt.tight_layout(rect=[0, 0, 0.95, 0.98])
        
        # Save figure as SVG with clear mode
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = Path(self.output_dir) / f'layers_comparison_{mode_suffix}.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
    
    def plot_forced_response_per_stage(self, results: Dict[str, Dict], stage_name: str) -> None:
        """
        Create separate figure per stage showing forced response gain.
        
        Args:
            results: Dictionary mapping stage names to analysis results
            stage_name: Specific stage to plot
        """
        if stage_name not in results:
            return
        
        result = results[stage_name]
        analysis_mode = result.get('analysis_mode', 'silent')
        
        # Determine which mode data to use
        if analysis_mode == 'both':
            modes_to_plot = ['silent', 'driven']
            silent_data = result.get('silent', {})
            driven_data = result.get('driven', {})
        elif analysis_mode == 'silent':
            modes_to_plot = ['silent']
            silent_data = result
            driven_data = None
        else:  # driven
            modes_to_plot = ['driven']
            silent_data = None
            driven_data = result
        
        # Determine scope
        network_params = result['network_params']
        layers = network_params.get('layers', ['L4'])
        scope = layers[0] if len(layers) == 1 else 'full'
        
        # Create figure with subplots
        n_modes = len(modes_to_plot)
        fig, axes = plt.subplots(1, n_modes, figsize=(FIGSIZE[0] * n_modes, FIGSIZE[1]))
        if n_modes == 1:
            axes = [axes]
        
        for idx, mode in enumerate(modes_to_plot):
            ax = axes[idx]
            
            # Get data for this mode
            if mode == 'silent':
                mode_data = silent_data
            else:
                mode_data = driven_data
            
            if mode_data is None:
                continue
            
            # Extract forced response data
            max_gain = mode_data.get('forced_response_max_gain', np.nan)
            critical_k = mode_data.get('forced_response_critical_k', 0.0)
            
            # Skip if NaN
            if np.isnan(max_gain):
                ax.text(0.5, 0.5, 'No thalamic input', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{stage_name} - {mode.capitalize()}', fontsize=10, fontweight='normal')
                continue
            
            # Plot bar with color-coded marker
            # Use viridis colormap for k* values
            # Normalize k* by a reasonable maximum (we'll use a fixed scale for per-stage plots)
            k_max = 20.0  # Reasonable maximum k for normalization
            k_normalized = min(critical_k / k_max, 1.0) if k_max > 0 else 0.0
            cmap = plt.get_cmap('viridis')
            color = cmap(k_normalized)
            
            ax.bar(0, max_gain, color=color, alpha=0.7, width=0.6)
            ax.set_ylabel('Max Gain', fontsize=10)
            ax.set_title(f'{stage_name} - {mode.capitalize()}\n(k*={critical_k:.2f})', fontsize=10, fontweight='normal')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            for label in ax.get_yticklabels():
                label.set_ha('right')
        
        plt.suptitle(f'Forced Response - {stage_name} ({scope})', fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure for each mode
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        for mode in modes_to_plot:
            output_path = Path(self.output_dir) / f'{scope}_forced_response_{stage_name}_{mode}.svg'
            # Save a copy of the figure for each mode
            # Since we have subplots, we save the full figure with all modes shown
            if n_modes > 1:
                # For 'both' mode, save the combined figure for each mode separately
                # by creating individual figures
                fig_single, ax_single = plt.subplots(1, 1, figsize=FIGSIZE)
                
                # Get data for this specific mode
                if mode == 'silent':
                    mode_data = silent_data
                else:
                    mode_data = driven_data
                
                if mode_data is not None:
                    max_gain = mode_data.get('forced_response_max_gain', np.nan)
                    critical_k = mode_data.get('forced_response_critical_k', 0.0)
                    
                    if not np.isnan(max_gain):
                        k_max = 20.0
                        k_normalized = min(critical_k / k_max, 1.0) if k_max > 0 else 0.0
                        cmap = plt.get_cmap('viridis')
                        color = cmap(k_normalized)
                        
                        ax_single.bar(0, max_gain, color=color, alpha=0.7, width=0.6)
                        ax_single.set_ylabel('Max Gain', fontsize=10)
                        ax_single.set_title(f'{stage_name} - {mode.capitalize()}\n(k*={critical_k:.2f})', fontsize=10, fontweight='normal')
                        ax_single.set_xticks([])
                        ax_single.grid(True, alpha=0.3, axis='y')
                        ax_single.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                        for label in ax_single.get_yticklabels():
                            label.set_ha('right')
                    else:
                        ax_single.text(0.5, 0.5, 'No thalamic input', ha='center', va='center', transform=ax_single.transAxes)
                        ax_single.set_title(f'{stage_name} - {mode.capitalize()}', fontsize=10, fontweight='normal')
                
                plt.suptitle(f'Forced Response - {stage_name} ({scope})', fontsize=12, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(output_path, format='svg', bbox_inches='tight')
                plt.close(fig_single)
            else:
                # Single mode, save the existing figure
                plt.savefig(output_path, format='svg', bbox_inches='tight')
        
        plt.close(fig)
    
    def plot_forced_response_development(self, results: Dict[str, Dict]) -> None:
        """
        Create bar plot showing forced response gain across developmental stages.
        
        Args:
            results: Dictionary mapping stage names to analysis results
        """
        stages = list(results.keys())
        
        # Determine scope and mode
        first_result = results[stages[0]]
        network_params = first_result['network_params']
        layers = network_params.get('layers', ['L4'])
        scope = layers[0] if len(layers) == 1 else 'full'
        analysis_mode = first_result.get('analysis_mode', 'silent')
        
        # Determine which modes to plot
        if analysis_mode == 'both':
            modes_to_plot = ['silent', 'driven']
            # For 'both' mode, we'll create separate figures, so we don't need the combined one
            fig = None
            axes = None
        elif analysis_mode == 'silent':
            modes_to_plot = ['silent']
            fig, axes = plt.subplots(1, 1, figsize=FIGSIZE)
            axes = [axes]
        else:  # driven
            modes_to_plot = ['driven']
            fig, axes = plt.subplots(1, 1, figsize=FIGSIZE)
            axes = [axes]
        
        # First pass: collect all k* values for normalization
        all_k_values = []
        for stage in stages:
            result = results[stage]
            if analysis_mode == 'both':
                for mode in ['silent', 'driven']:
                    mode_data = result.get(mode, {})
                    k_val = mode_data.get('forced_response_critical_k', 0.0)
                    if not np.isnan(k_val) and k_val > 0:
                        all_k_values.append(k_val)
            else:
                k_val = result.get('forced_response_critical_k', 0.0)
                if not np.isnan(k_val) and k_val > 0:
                    all_k_values.append(k_val)
        
        k_max = max(all_k_values) if all_k_values else 1.0
        
        # Second pass: collect all gains for consistent y-limits
        all_gains = []
        for stage in stages:
            result = results[stage]
            if analysis_mode == 'both':
                for mode in ['silent', 'driven']:
                    mode_data = result.get(mode, {})
                    gain = mode_data.get('forced_response_max_gain', np.nan)
                    if not np.isnan(gain):
                        all_gains.append(gain)
            else:
                gain = result.get('forced_response_max_gain', np.nan)
                if not np.isnan(gain):
                    all_gains.append(gain)
        
        y_min = 0.0
        y_max = max(all_gains) * 1.1 if all_gains else 1.0
        
        # Plot each mode
        # For 'both' mode, we create separate figures in the save section
        # For single mode, we plot on the existing axes
        if analysis_mode != 'both':
            # Single mode - plot on existing axes
            mode = modes_to_plot[0]
            ax = axes[0]
            
            # Extract data for this mode
            gains = []
            k_values = []
            valid_stages = []
            
            for stage in stages:
                result = results[stage]
                mode_data = result
                
                gain = mode_data.get('forced_response_max_gain', np.nan)
                k_val = mode_data.get('forced_response_critical_k', 0.0)
                
                if not np.isnan(gain):
                    gains.append(gain)
                    k_values.append(k_val)
                    valid_stages.append(stage)
            
            if gains:
                # Normalize k* values for color mapping
                k_normalized = [min(k / k_max, 1.0) if k_max > 0 else 0.0 for k in k_values]
                
                # Create color map
                cmap = plt.get_cmap('viridis')
                colors = [cmap(k_norm) for k_norm in k_normalized]
                
                # Plot bars
                x_pos = range(len(valid_stages))
                ax.bar(x_pos, gains, color=colors, alpha=0.7, width=0.6)
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=k_max))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('k*', fontsize=9)
                cbar.ax.tick_params(labelsize=8)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(valid_stages, fontsize=10)
                ax.set_ylabel('Max Gain', fontsize=10)
                ax.set_title(f'Forced Response - {mode.capitalize()}', fontsize=10, fontweight='normal')
                ax.set_ylim(y_min, y_max)
                ax.grid(True, alpha=0.3, axis='y')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                for label in ax.get_yticklabels():
                    label.set_ha('right')
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Forced Response - {mode.capitalize()}', fontsize=10, fontweight='normal')
            
            # Create meaningful title
            if scope == 'full':
                title = 'Forced Response Development - Full Network'
            else:
                title = f'Forced Response Development - {scope} Layer'
            
            plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # For 'both' mode, save separate files for each mode (matching spectrum plot pattern)
        # For single mode, save one file
        if analysis_mode == 'both':
            # Close the unused combined figure if it exists
            if fig is not None:
                plt.close(fig)
            # Save separate files for silent and driven modes
            for mode in modes_to_plot:
                # Create a single subplot figure for this mode
                fig_single, ax_single = plt.subplots(1, 1, figsize=FIGSIZE)
                
                # Extract data for this mode
                gains = []
                k_values = []
                valid_stages = []
                
                for stage in stages:
                    result = results[stage]
                    mode_data = result.get(mode, {})
                    gain = mode_data.get('forced_response_max_gain', np.nan)
                    k_val = mode_data.get('forced_response_critical_k', 0.0)
                    
                    if not np.isnan(gain):
                        gains.append(gain)
                        k_values.append(k_val)
                        valid_stages.append(stage)
                
                if gains:
                    # Normalize k* values for color mapping
                    k_normalized = [min(k / k_max, 1.0) if k_max > 0 else 0.0 for k in k_values]
                    cmap = plt.get_cmap('viridis')
                    colors = [cmap(k_norm) for k_norm in k_normalized]
                    
                    # Plot bars
                    x_pos = range(len(valid_stages))
                    ax_single.bar(x_pos, gains, color=colors, alpha=0.7, width=0.6)
                    
                    # Add colorbar
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=k_max))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax_single)
                    cbar.set_label('k*', fontsize=9)
                    cbar.ax.tick_params(labelsize=8)
                    
                    ax_single.set_xticks(x_pos)
                    ax_single.set_xticklabels(valid_stages, fontsize=10)
                    ax_single.set_ylabel('Max Gain', fontsize=10)
                    ax_single.set_title(f'Forced Response - {mode.capitalize()}', fontsize=10, fontweight='normal')
                    ax_single.set_ylim(y_min, y_max)
                    ax_single.grid(True, alpha=0.3, axis='y')
                    ax_single.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                    for label in ax_single.get_yticklabels():
                        label.set_ha('right')
                
                # Create meaningful title
                if scope == 'full':
                    title = f'Forced Response Development - Full Network ({mode.capitalize()})'
                else:
                    title = f'Forced Response Development - {scope} Layer ({mode.capitalize()})'
                
                plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                output_path = Path(self.output_dir) / f'{scope}_forced_response_{mode}.svg'
                plt.savefig(output_path, format='svg', bbox_inches='tight')
                plt.close(fig_single)
        else:
            # Save single mode figure
            output_path = Path(self.output_dir) / f'{scope}_forced_response_{analysis_mode}.svg'
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close(fig)