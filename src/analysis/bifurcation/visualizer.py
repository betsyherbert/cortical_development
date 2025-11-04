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
    DEVELOPMENTAL_STAGES, OUTPUT_DIR, COLORMAP, COLORBAR_PARAMS,
    FIGSIZE_DETAIL, FIGSIZE_COMPARISON, FIGSIZE_EIGENVALUE, FIGSIZE_COUPLING,
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
        
    def plot_developmental_comparison(self, results: Dict[str, Dict], n_populations: int) -> None:
        """
        Create multi-panel figure comparing developmental stages.
        
        Args:
            results: Dictionary mapping stage names to analysis results
            n_populations: Number of populations (2 or 3)
        """
        stages = list(results.keys())
        n_stages = len(stages)
        
        # Extract data (convert distance to max real eigenvalue)
        distances = [results[s]['distance'] for s in stages]
        max_real_eigenvalues = [-d for d in distances]  # Convert distance to max real eigenvalue
        critical_ks = [results[s]['critical_k'] for s in stages]
        
        # Extract connection strengths for heatmap
        # For multi-layer, show within-layer connections (diagonal blocks)
        network_params = results[stages[0]]['network_params']
        n_layers = network_params.get('n_layers', 1)
        
        if n_layers == 1:
            # Single layer: show all connections
            if n_populations == 2:
                conn_matrix = np.zeros((n_stages, 4))  # E->E, E->I, I->E, I->I
                for i, stage in enumerate(stages):
                    A = results[stage]['network_params']['A']
                    conn_matrix[i, 0] = A[0, 0]  # E->E
                    conn_matrix[i, 1] = A[0, 1]  # E->I
                    conn_matrix[i, 2] = A[1, 0]  # I->E
                    conn_matrix[i, 3] = A[1, 1]  # I->I
                conn_labels = ['E→E', 'E→I', 'I→E', 'I→I']
            else:
                conn_matrix = np.zeros((n_stages, 9))  # All 3x3 connections
                for i, stage in enumerate(stages):
                    A = results[stage]['network_params']['A']
                    conn_matrix[i, :] = A.flatten()
                conn_labels = ['E→E', 'E→SST', 'E→PV', 'SST→E', 'SST→SST', 'SST→PV',
                               'PV→E', 'PV→SST', 'PV→PV']
        else:
            # Multi-layer: show within-layer connections (first layer only for clarity)
            # Take diagonal block (within-layer connections) from first layer
            if n_populations == 2:
                conn_matrix = np.zeros((n_stages, 4))
                conn_labels = ['E→E', 'E→I', 'I→E', 'I→I']
                for i, stage in enumerate(stages):
                    A = results[stage]['network_params']['A']
                    # First layer's block (diagonal block)
                    conn_matrix[i, 0] = A[0, 0]
                    conn_matrix[i, 1] = A[0, 1]
                    conn_matrix[i, 2] = A[1, 0]
                    conn_matrix[i, 3] = A[1, 1]
            else:
                conn_matrix = np.zeros((n_stages, 9))
                conn_labels = ['E→E', 'E→SST', 'E→PV', 'SST→E', 'SST→SST', 'SST→PV',
                               'PV→E', 'PV→SST', 'PV→PV']
                for i, stage in enumerate(stages):
                    A = results[stage]['network_params']['A']
                    # First layer's block (diagonal block)
                    block = A[0:n_populations, 0:n_populations]
                    conn_matrix[i, :] = block.flatten()
        
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
        
        # Create figure with subplots (4 rows: A, B, C, D)
        fig = plt.figure(figsize=FIGSIZE_COMPARISON)
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Panel A: Max Real Eigenvalue
        ax_a = fig.add_subplot(gs[0, 0])
        stage_colors = [STAGE_COLORS.get(s, '#666666') for s in stages]
        bars = ax_a.bar(range(n_stages), max_real_eigenvalues, color=stage_colors)
        ax_a.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax_a.set_xticks(range(n_stages))
        ax_a.set_xticklabels(stages)
        ax_a.set_ylabel('Max Re(λ)')
        ax_a.set_title('A. Stability Across Development', fontweight='bold')
        ax_a.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, max_real_eigenvalues)):
            height = bar.get_height()
            ax_a.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:.4f}', ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=9)
        
        # Panel B: Critical mode k
        ax_b = fig.add_subplot(gs[0, 1])
        # Color code by pattern type: k=0 = global, k>0 = patterned
        bar_colors = ['#8E44AD' if k == 0 else '#16A085' for k in critical_ks]
        bars = ax_b.bar(range(n_stages), critical_ks, color=bar_colors)
        ax_b.set_xticks(range(n_stages))
        ax_b.set_xticklabels(stages)
        ax_b.set_ylabel('Critical Wave Number k')
        ax_b.set_title('B. Critical Spatial Mode', fontweight='bold')
        ax_b.grid(True, alpha=0.3, axis='y')
        
        # Add legend for pattern types
        global_patch = mpatches.Patch(color='#8E44AD', label='Global (k=0)')
        pattern_patch = mpatches.Patch(color='#16A085', label='Patterned (k>0)')
        ax_b.legend(handles=[global_patch, pattern_patch], loc='upper left', fontsize=8)
        
        # Add value labels
        for i, (bar, k) in enumerate(zip(bars, critical_ks)):
            height = bar.get_height()
            ax_b.text(bar.get_x() + bar.get_width()/2., height,
                     f'{k:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Panel C: Connection strengths heatmap
        ax_c = fig.add_subplot(gs[1, :])
        im = ax_c.imshow(conn_matrix, aspect='auto', cmap=COLORMAP, 
                         interpolation='nearest')
        ax_c.set_xticks(range(len(conn_labels)))
        ax_c.set_xticklabels(conn_labels, rotation=45, ha='right')
        ax_c.set_yticks(range(n_stages))
        ax_c.set_yticklabels(stages)
        title_c = 'C. Connection Strengths Across Stages (Within-Layer)' if n_layers > 1 else 'C. Connection Strengths Across Stages'
        ax_c.set_title(title_c, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax_c)
        cbar.set_label('Connection Strength')
        
        # Add text annotations
        for i in range(n_stages):
            for j in range(len(conn_labels)):
                text = ax_c.text(j, i, f'{conn_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=7)
        
        # Panel D: Time constants and spatial scales
        # Split into two side-by-side subplots for clarity
        ax_d_tau = fig.add_subplot(gs[3, 0])
        ax_d_sigma = fig.add_subplot(gs[3, 1])
        x = np.arange(n_stages)
        width = 0.25
        
        # Time constants subplot
        for i, label in enumerate(tau_labels):
            color = CELL_COLORS.get(label, '#666666') if label in CELL_COLORS else '#666666'
            offset = (i - (n_populations - 1) / 2) * width
            ax_d_tau.bar(x + offset, tau_matrix[:, i], width, label=f'τ_{label}', color=color, alpha=0.7)
        ax_d_tau.set_xticks(x)
        ax_d_tau.set_xticklabels(stages)
        ax_d_tau.set_ylabel('Time Constant (ms)', fontweight='bold')
        ax_d_tau.set_title('D. Time Constants', fontweight='bold')
        ax_d_tau.legend(loc='upper left', fontsize=8)
        ax_d_tau.grid(True, alpha=0.3, axis='y')
        
        # Spatial scales subplot
        for i, label in enumerate(tau_labels):
            color = CELL_COLORS.get(label, '#999999') if label in CELL_COLORS else '#999999'
            offset = (i - (n_populations - 1) / 2) * width
            ax_d_sigma.bar(x + offset, sigma_matrix[:, i], width, 
                          label=f'σ_{label}', color=color, alpha=0.7, edgecolor='black', linewidth=1)
        ax_d_sigma.set_xticks(x)
        ax_d_sigma.set_xticklabels(stages)
        ax_d_sigma.set_ylabel('Spatial Scale σ (grid units)', fontweight='bold')
        ax_d_sigma.set_title('E. Spatial Scales', fontweight='bold')
        ax_d_sigma.legend(loc='upper left', fontsize=8)
        ax_d_sigma.grid(True, alpha=0.3, axis='y')
        
        # Network description
        layers_desc = f"Full Network ({n_layers} layers × {n_populations} populations)" if n_layers > 1 else f"{network_params.get('layers', ['L4'])[0]} Network ({n_populations} populations)"
        
        plt.suptitle(f'Developmental Stability Analysis ({layers_desc})',
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure as SVG
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(self.output_dir) / f'stability_across_development_{n_populations}pop.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def plot_single_stage_detail(self, result: Dict, stage_name: str, n_populations: int) -> None:
        """
        Create detailed figure for a single developmental stage.
        
        Args:
            result: Analysis results for single stage
            stage_name: Name of developmental stage
            n_populations: Number of populations (2 or 3)
        """
        network_params = result['network_params']
        n_layers = network_params.get('n_layers', 1)
        layers = network_params.get('layers', ['L4'])
        full_pop_names = network_params.get('full_pop_names', network_params['pop_names'])
        total_pops = len(network_params['tau'])
        
        fig = plt.figure(figsize=FIGSIZE_DETAIL)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Connection strength matrix heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        A = network_params['A']
        im = ax1.imshow(A, cmap=COLORMAP, aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        if n_layers > 1:
            # Multi-layer: show block structure
            ax1.set_xticks(range(0, total_pops, n_populations))
            ax1.set_yticks(range(0, total_pops, n_populations))
            ax1.set_xticklabels(layers)
            ax1.set_yticklabels(layers)
            
            # Add minor ticks for individual populations
            ax1.set_xticks(range(total_pops), minor=True)
            ax1.set_yticks(range(total_pops), minor=True)
            ax1.grid(True, which='minor', color='white', linewidth=0.5, alpha=0.3)
            
            # Add block separators
            for i in range(1, n_layers):
                pos = i * n_populations - 0.5
                ax1.axhline(pos, color='black', linewidth=2)
                ax1.axvline(pos, color='black', linewidth=2)
            
            # Add text annotations (only show non-zero values or use smaller font for multi-layer)
            fontsize = 7 if total_pops > 6 else 9
            for i in range(total_pops):
                for j in range(total_pops):
                    if abs(A[i, j]) > 0.001:  # Only show significant connections
                        text = ax1.text(j, i, f'{A[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=fontsize)
        else:
            # Single layer: show population names with cell type colors
            ax1.set_xticks(range(n_populations))
            ax1.set_yticks(range(n_populations))
            pop_names = network_params['pop_names']
            ax1.set_xticklabels(pop_names)
            ax1.set_yticklabels(pop_names)
            # Color tick labels with cell type colors
            for i, pop_name in enumerate(pop_names):
                color = CELL_COLORS.get(pop_name, 'black')
                ax1.get_xticklabels()[i].set_color(color)
                ax1.get_yticklabels()[i].set_color(color)
            
            # Add text annotations
            for i in range(n_populations):
                for j in range(n_populations):
                    text = ax1.text(j, i, f'{A[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=10)
        
        title_1 = 'Connection Strength Matrix (Full Network)' if n_layers > 1 else 'Connection Strength Matrix'
        ax1.set_title(title_1, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Strength')
        
        # Panel 2: Eigenvalue spectrum (compute for all modes)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Compute eigenvalues for all modes to show spectrum
        n_modes = ANALYSIS_PARAMS['n_modes']
        all_eigenvalues = []
        critical_mode = result['critical_mode']
        critical_k = result['critical_k']
        
        for n1 in range(-n_modes, n_modes + 1):
            for n2 in range(-n_modes, n_modes + 1):
                # Recompute Jacobian for visualization
                k_squared = n1**2 + n2**2
                grid_size = ANALYSIS_PARAMS['grid_size']
                scale_factor = 1.0 / grid_size
                
                # Build Jacobian using full matrix size
                J = np.zeros((total_pops, total_pops))
                for i in range(total_pops):
                    for j in range(total_pops):
                        w_tilde = network_params['A'][i, j] * np.exp(
                            -2 * np.pi**2 * k_squared * network_params['sigma'][i, j]**2 * scale_factor**2
                        )
                        if i == j:
                            J[i, j] = (-1.0 / network_params['tau'][i] + 
                                      (network_params['gain'][i] * w_tilde) / network_params['tau'][i])
                        else:
                            J[i, j] = (network_params['gain'][i] * w_tilde) / network_params['tau'][i]
                
                eigenvalues = np.linalg.eigvals(J)
                all_eigenvalues.extend(eigenvalues)
        
        all_eigenvalues = np.array(all_eigenvalues)
        ax2.scatter(all_eigenvalues.real, all_eigenvalues.imag, alpha=0.3, s=10, 
                   c='gray', label='All modes')
        
        # Highlight critical eigenvalue
        n1_crit, n2_crit = critical_mode
        k_squared_crit = n1_crit**2 + n2_crit**2
        grid_size = ANALYSIS_PARAMS['grid_size']
        scale_factor = 1.0 / grid_size
        J_crit = np.zeros((total_pops, total_pops))
        for i in range(total_pops):
            for j in range(total_pops):
                w_tilde = network_params['A'][i, j] * np.exp(
                    -2 * np.pi**2 * k_squared_crit * network_params['sigma'][i, j]**2 * scale_factor**2
                )
                if i == j:
                    J_crit[i, j] = (-1.0 / network_params['tau'][i] + 
                                   (network_params['gain'][i] * w_tilde) / network_params['tau'][i])
                else:
                    J_crit[i, j] = (network_params['gain'][i] * w_tilde) / network_params['tau'][i]
        
        crit_eigenvalues = np.linalg.eigvals(J_crit)
        max_real_idx = np.argmax(crit_eigenvalues.real)
        crit_eigenvalue = crit_eigenvalues[max_real_idx]
        ax2.scatter(crit_eigenvalue.real, crit_eigenvalue.imag, s=200, 
                   c='red', marker='*', label='Critical mode', zorder=10)
        
        ax2.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Real(λ)')
        ax2.set_ylabel('Imag(λ)')
        ax2.set_title('Eigenvalue Spectrum', fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Remove text box (metrics now shown in CLI output only)
        # Keep the subplot space but don't display anything
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        layers_desc = f"Full Network ({n_layers} layers × {n_populations} populations)" if n_layers > 1 else f"{layers[0]} Network ({n_populations} populations)"
        plt.suptitle(f'{stage_name} Detailed Analysis ({layers_desc})',
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure as SVG
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(self.output_dir) / f'{stage_name}_stability_detail_{n_populations}pop.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def plot_eigenvalue_spectra(self, all_results: Dict[str, Dict], n_populations: int) -> None:
        """
        Plot eigenvalue spectra for all stages in the complex plane.
        
        Args:
            all_results: Dictionary mapping stage names to analysis results
            n_populations: Number of populations (2 or 3)
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_EIGENVALUE)
        axes = axes.flatten()
        
        stages = ['P4', 'P8', 'P12', 'P16']
        
        for idx, stage in enumerate(stages):
            if stage not in all_results:
                continue
                
            result = all_results[stage]
            ax = axes[idx]
            network_params = result['network_params']
            total_pops = len(network_params['tau'])
            
            # Compute eigenvalues for critical mode
            critical_mode = result['critical_mode']
            n1_crit, n2_crit = critical_mode
            k_squared_crit = n1_crit**2 + n2_crit**2
            grid_size = ANALYSIS_PARAMS['grid_size']
            scale_factor = 1.0 / grid_size
            
            J_crit = np.zeros((total_pops, total_pops))
            for i in range(total_pops):
                for j in range(total_pops):
                    w_tilde = network_params['A'][i, j] * np.exp(
                        -2 * np.pi**2 * k_squared_crit * network_params['sigma'][i, j]**2 * scale_factor**2
                    )
                    if i == j:
                        J_crit[i, j] = (-1.0 / network_params['tau'][i] + 
                                       (network_params['gain'][i] * w_tilde) / network_params['tau'][i])
                    else:
                        J_crit[i, j] = (network_params['gain'][i] * w_tilde) / network_params['tau'][i]
            
            eigenvalues = np.linalg.eigvals(J_crit)
            
            stage_color = STAGE_COLORS.get(stage, '#666666')
            ax.scatter(eigenvalues.real, eigenvalues.imag, s=50, 
                      c=stage_color, alpha=0.7, label=stage)
            ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Real(λ)')
            ax.set_ylabel('Imag(λ)')
            ax.set_title(f'{stage} (k={result["critical_k"]:.2f})', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Eigenvalue Spectra (Critical Modes)', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save figure as SVG
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(self.output_dir) / f'eigenvalue_spectrum_{n_populations}pop.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    def plot_layer_coupling_comparison(self, l5_results: Dict[str, Dict],
                                       l4_results: Dict[str, Dict], 
                                       l23_results: Dict[str, Dict],
                                       full_results: Dict[str, Dict],
                                       n_populations: int) -> None:
        """
        Compare stability across layer isolation conditions.
        
        Args:
            l5_results: Results for L5-only analysis
            l4_results: Results for L4-only analysis
            l23_results: Results for L23-only analysis
            full_results: Results for full network (all layers) analysis
            n_populations: Number of populations (2 or 3)
        """
        stages = ['P4', 'P8', 'P12', 'P16']
        
        # Convert distances to max real eigenvalues
        l5_max_real = [-l5_results[s]['distance'] for s in stages]
        l4_max_real = [-l4_results[s]['distance'] for s in stages]
        l23_max_real = [-l23_results[s]['distance'] for s in stages]
        full_max_real = [-full_results[s]['distance'] for s in stages]
        
        fig, ax = plt.subplots(figsize=FIGSIZE_COUPLING)
        
        x = np.arange(len(stages))
        width = 0.2  # Narrower bars to fit 4 per stage
        
        # Layer colors
        layer_colors = {
            'L5': '#999999',     # Light grey
            'L4': '#555555',     # Medium grey  
            'L23': '#222222',    # Dark grey
            'Full': '#4472c4'    # Blue for full network
        }
        
        bars1 = ax.bar(x - 1.5*width, l23_max_real, width, 
                      label='L23 only', color=layer_colors['L23'], alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, l4_max_real, width, 
                      label='L4 only', color=layer_colors['L4'], alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, l5_max_real, width, 
                      label='L5 only', color=layer_colors['L5'], alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, full_max_real, width, 
                      label='Full Network', color=layer_colors['Full'], alpha=0.8)
        
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Developmental Stage', fontweight='bold')
        ax.set_ylabel('Max Re(λ)', fontweight='bold')
        ax.set_title('Layer Isolation Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars (only for bars that are clearly visible)
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                # Only label if height is significant
                if abs(height) > 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', 
                           va='bottom' if height > 0 else 'top',
                           fontsize=7)
        
        plt.tight_layout()
        
        # Save figure as SVG
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(self.output_dir) / f'layer_isolation_comparison_{n_populations}pop.svg'
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
