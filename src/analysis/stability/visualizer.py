"""Visualization module for stability analysis results."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plots from showing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Dict, List, Tuple

from .config import (
    DEVELOPMENTAL_STAGES, LAYERS, ANALYSIS_PARAMS, 
    COLORMAP, COLORBAR_PARAMS, REGIMES
)

from src.model.config import GRID_SIZE, CELL_COLORS

# Constants for regime visualization
REGIME_COLORS = {
    'inhibition \n destabilised': '#151515',  # dark_grey
    'intrinsically \n unstable': '#AC1E12',   # dark_red  
    'intrinsically \n stable': '#CCCCCC',     # light_grey
    'inhibition \n stabilised': '#214F7F'     # dark_blue
}

# Labels in order from bottom to top of colorbar
REGIME_LABELS = [
    "inhibition \n destabilised",
    "intrinsically \n unstable", 
    "intrinsically \n stable",
    "inhibition \n stabilised"
]

# Publication-quality styling
FONT_CONFIG = {
    'font_family': 'Arial',
    'font_sizes': {
        'title': 18,
        'ylabel': 16,
        'colorbar': 14,
        'colorbar_ticks': 12,
        'tick_labels': 12,
        'condition_labels': 14
    },
    'figure_sizes': {
        'layer_wise': (16, 4),
        'column_wise': (8, 10),
        'effectiveness': (5, 5),          # Square single plot size
        'effectiveness_2x2': (8, 8),      # Square 2x2 subplots (celltype effectiveness) - smaller
        'phase_diagram': (16, 4),         # Wide format for phase diagrams
        'regime_percentages': (10, 10),   # Square 2x2 subplots
        'heatmap_single': (8, 6),         # Single heatmap
        'heatmap_dual': (10, 5)           # Dual heatmap side-by-side - smaller and more square
    },
    'dpi': 300,
    'colorbar_width': 0.008
}

# Font sizes for easy access
FONT_SIZES = FONT_CONFIG['font_sizes']

# Common plot settings
DPI = FONT_CONFIG['dpi']
OUTPUT_DIR = 'outputs/stability'
COLORBAR_WIDTH = FONT_CONFIG['colorbar_width']
LINE_COLOR = 'black'
REFERENCE_LINE_ALPHA = 0.5

# Standardized color schemes for consistency across plots
STANDARD_COLORS = {
    'cell_types': {
        'SST': CELL_COLORS['SST'],  # Use colors from model config
        'PV': CELL_COLORS['PV']
    },
    'layers': {
        'L5': '#999999',     # Light grey
        'L4': '#555555',     # Medium grey  
        'L23': '#222222',    # Dark grey
        'L2/3': '#222222'    # Alias for L23
    },
    'colormaps': {
        'heatmap': 'Reds',
        'diverging': 'RdBu_r',
        'sequential': 'viridis'
    }
}

# Configure matplotlib for publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = FONT_CONFIG['font_sizes']['tick_labels']
plt.rcParams['axes.labelsize'] = FONT_CONFIG['font_sizes']['tick_labels']
plt.rcParams['xtick.labelsize'] = FONT_CONFIG['font_sizes']['tick_labels']
plt.rcParams['ytick.labelsize'] = FONT_CONFIG['font_sizes']['tick_labels']
plt.rcParams['legend.fontsize'] = FONT_CONFIG['font_sizes']['colorbar']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# Inhibition effectiveness plot constants
EFFECTIVENESS_FIGSIZE = (8, 6)
LINE_STYLES = {'driven': '-', 'idle': '--'}
LINE_WIDTH = 3
SHADE_COLOR = 'lightgrey'
SHADE_ALPHA = 0.7

# Layer-specific effectiveness plot constants
LAYER_COLORS = {
    'L23': '#1f4e79',  # dark blue
    'L4': '#4472c4',   # mid blue  
    'L5': '#8db4e2'    # light blue
}
LAYER_ALPHA = 0.4

# Phase diagram plot constants
PHASE_DIAGRAM_FIGSIZE = (16, 4)
PHASE_DIAGRAM_COLOR = 'black'
PHASE_DIAGRAM_MARKERS = {
    'driven': 'o',   # circles
    'idle': '^'      # triangles
}
PHASE_DIAGRAM_ALPHA = 0.4
PHASE_DIAGRAM_MARKER_SIZE = 10
QUADRANT_ALPHA = 0.2
REFERENCE_LINE_WIDTH = 2.0
REFERENCE_LINE_ALPHA_THICK = 0.8


class StabilityVisualizer:
    """Creates publication-quality visualizations for stability analysis results."""
    
    def __init__(self):
        """Initialize visualizer with publication-quality parameters and color schemes."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Font and figure configuration
        self.font_sizes = FONT_CONFIG['font_sizes']
        self.figure_sizes = FONT_CONFIG['figure_sizes']
        self.dpi = FONT_CONFIG['dpi']
        self.colorbar_width = FONT_CONFIG['colorbar_width']
        
        # Create output directories
        self.summary_dir = os.path.join(OUTPUT_DIR, 'summary')
        self.snapshots_dir = os.path.join(OUTPUT_DIR, 'snapshots')
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Grid parameters from config
        self.grid_size = GRID_SIZE  # Use actual grid size from model config
        self.boundary = ANALYSIS_PARAMS['boundary_exclude']
        self.layer_patch_size = ANALYSIS_PARAMS['layer_patch_size']
        self.column_patch_size = ANALYSIS_PARAMS['column_patch_size']
        
        # Create regime colormap
        self._create_regime_colormap()
    
    def _create_regime_colormap(self):
        """Create custom colormap for regime classification."""
        # Extract colors in the same order as labels
        regime_colors = [REGIME_COLORS[label] for label in REGIME_LABELS]
        
        self.regime_cmap = ListedColormap(regime_colors)
        self.regime_norm = BoundaryNorm(
            boundaries=np.arange(len(REGIME_LABELS) + 1) - 0.5, 
            ncolors=len(REGIME_LABELS)
        )
    
    def _regime_to_index(self, regime_name: str) -> int:
        """Convert regime name to numeric index."""
        try:
            return REGIME_LABELS.index(regime_name)
        except ValueError:
            return -1
    
    def _create_grid(self, data: List, coords: List[Tuple], patch_size: int) -> np.ndarray:
        """Create spatial grid from coordinate-indexed data."""
        grid_dim = self.grid_size - 2 * self.boundary - patch_size + 1
        grid = np.full((grid_dim, grid_dim), np.nan)
        
        for i, (x, y) in enumerate(coords):
            grid_x, grid_y = x - self.boundary, y - self.boundary
            if isinstance(data[i], str):  # Regime data
                regime_index = self._regime_to_index(data[i])
                grid[grid_x, grid_y] = regime_index if regime_index >= 0 else np.nan
            else:  # Numeric data
                grid[grid_x, grid_y] = data[i]
        
        return grid
    
    def _create_layer_grids(self, layer_data: Dict, coords: List[Tuple]) -> Dict:
        """Convert layer-wise results to spatial grids."""
        grids = {}
        
        for layer in LAYERS:
            grids[layer] = {}
            
            # Lambda grids
            for condition in ['full', 'e_only']:
                grids[layer][condition] = self._create_grid(
                    layer_data[condition][layer], coords, self.layer_patch_size
                )
            
            # Regime grid
            regime_grid = self._create_grid(
                layer_data['regimes'][layer], coords, self.layer_patch_size
            )
            grids[layer]['regimes'] = {'regimes': regime_grid}
        
        return grids
    
    def _create_column_grids(self, column_data: Dict, coords: List[Tuple]) -> Dict:
        """Convert column-wise results to spatial grids."""
        grids = {}
        
        # Lambda grids
        for condition in ['full', 'e_only']:
            grids[condition] = self._create_grid(
                column_data[condition], coords, self.column_patch_size
            )
        
        # Regime grid
        grids['regimes'] = self._create_grid(
            column_data['regimes'], coords, self.column_patch_size
        )
        
        return grids
    
    def _configure_axis(self, ax):
        """Configure axis appearance (remove ticks)."""
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _plot_heatmap(self, ax, data: np.ndarray, center_zero: bool = False):
        """Plot heatmap with appropriate colormap and scaling."""
        if center_zero:
            vmax = COLORBAR_PARAMS['difference_max']
            vmin = -vmax
            cmap = 'RdBu'
        else:
            vmin = COLORBAR_PARAMS['lambda_max_min']
            vmax = COLORBAR_PARAMS['lambda_max_max']
            cmap = COLORMAP
        
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                      origin='lower', aspect='equal')
        self._configure_axis(ax)
        return im
    
    def _plot_regime_heatmap(self, ax, data):
        """Plot stability regime classification heatmap."""
        # Handle different data structures
        if isinstance(data, dict):
            regime_data = (data['regimes']['regimes'] 
                          if 'regimes' in data and isinstance(data['regimes'], dict) 
                          else data['regimes'])
        else:
            regime_data = data
        
        im = ax.imshow(regime_data, cmap=self.regime_cmap, norm=self.regime_norm,
                      origin='lower', aspect='equal')
        self._configure_axis(ax)
        return im
    
    def _plot_thalamic_input(self, ax, thalamic_input: np.ndarray):
        """Plot thalamic input heatmap."""
        im = ax.imshow(thalamic_input, cmap='gray', vmin=0, vmax=1, 
                      origin='lower', aspect='equal')
        self._configure_axis(ax)
        return im
    
    def _add_colorbar(self, fig, im, position: List[float], label: str, 
                     ticks: List = None, tick_labels: List = None):
        """Add colorbar with poster-optimized styling."""
        cbar_ax = fig.add_axes(position)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(label, rotation=270, labelpad=12, fontsize=self.font_sizes['colorbar'])
        
        if ticks:
            cbar.set_ticks(ticks)
        if tick_labels:
            cbar.set_ticklabels(tick_labels, fontsize=self.font_sizes['colorbar_ticks'])
        else:
            cbar.ax.tick_params(labelsize=self.font_sizes['tick_labels'])
        
        return cbar
    
    def _setup_figure(self, regime: str, snapshot_idx: int, figsize: Tuple, title_prefix: str):
        """Setup figure with poster-optimized titles (no grey subtitle)."""
        del snapshot_idx  # Unused in poster style
        fig = plt.figure(figsize=figsize)
        
        # Main title only (no grey parameter subtitle for poster style)
        fig.suptitle(f'{title_prefix} - {regime.capitalize()}', 
                    fontsize=self.font_sizes['title'], fontweight='bold', y=0.95)
        
        return fig
    
    def _save_figure(self, fig, regime: str, snapshot_idx: int, plot_type: str):
        """Save figure to appropriate directory."""
        del fig  # Unused - plt.savefig uses current figure
        filename = f'{plot_type}_{regime}_snapshot{snapshot_idx}.svg'
        filepath = os.path.join(self.snapshots_dir, filename)
        plt.savefig(filepath, format='svg', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved figure: {filepath}")
    
    def _add_layer_wise_colorbars(self, fig, axes, ims):
        """Add colorbars for layer-wise plots."""
        del axes  # Unused with manual positioning
        # Use manual positioning for colorbars to match manual subplot layout
        cbar_bottom = 0.15  # Bottom of lowest layer
        cbar_height = 0.55  # Total height spanning all layers
        
        if ims['lambda']:
            vmin, vmax = COLORBAR_PARAMS['lambda_max_min'], COLORBAR_PARAMS['lambda_max_max']
            self._add_colorbar(fig, ims['lambda'], 
                             [0.88, cbar_bottom, self.colorbar_width, cbar_height],
                             r'$\lambda_{\mathrm{max}}$', [vmin, 0, vmax])
        
        if ims['regime']:
            self._add_colorbar(fig, ims['regime'], 
                             [0.93, cbar_bottom, self.colorbar_width, cbar_height],
                             'Stability Regime', list(range(len(REGIME_LABELS))), REGIME_LABELS)
    
    def _add_column_wise_colorbars(self, fig, axes, ims):
        """Add colorbars for column-wise plots."""
        positions = {
            'lambda': [0.82, axes[1, 0].get_position().y0, COLORBAR_WIDTH, 
                      axes[0, 0].get_position().y1 - axes[1, 0].get_position().y0],
            'diff': [0.82, axes[2, 0].get_position().y0, COLORBAR_WIDTH, 
                    axes[2, 0].get_position().height],
            'regime': [0.82, axes[3, 0].get_position().y0, COLORBAR_WIDTH, 
                      axes[3, 0].get_position().height],
            'thalamic': [0.82, axes[4, 0].get_position().y0, COLORBAR_WIDTH, 
                        axes[4, 0].get_position().height]
        }
        
        if ims['lambda']:
            vmin, vmax = COLORBAR_PARAMS['lambda_max_min'], COLORBAR_PARAMS['lambda_max_max']
            self._add_colorbar(fig, ims['lambda'], positions['lambda'], 
                             r'$\lambda_{\mathrm{max}}$', [vmin, 0, vmax])
        
        if ims['diff']:
            vmax_diff = COLORBAR_PARAMS['difference_max']
            self._add_colorbar(fig, ims['diff'], positions['diff'], 
                             r'$\Delta\lambda_{\mathrm{max}}$', [-vmax_diff, 0, vmax_diff])
        
        if ims['regime']:
            self._add_colorbar(fig, ims['regime'], positions['regime'], 
                             'Stability Regime', list(range(len(REGIME_LABELS))), REGIME_LABELS)
        
        if ims['thalamic']:
            self._add_colorbar(fig, ims['thalamic'], positions['thalamic'], 
                             'Thalamic Input', [0, 0.5, 1])
        
    def create_layer_wise_figure(self, results: Dict, regime: str, snapshot_idx: int):
        """Create layer-wise figure."""
        print(f"Creating layer-wise figure for {regime} snapshot {snapshot_idx}")
        
        fig = self._setup_figure(regime, snapshot_idx, self.figure_sizes['layer_wise'], 'Layer-wise stability')
        
        # Manual subplot positioning to eliminate gaps between layers
        axes = np.empty((3, 16), dtype=object)
        
        # Define precise positions: [left, bottom, width, height]
        subplot_width = 0.04  # Width of each subplot
        subplot_height = 0.15  # Height of each subplot
        left_start = 0.08
        col_spacing = 0.046  # Small spacing between columns
        
        # Layer positions (no gap between layers)
        layer_positions = [0.55, 0.35, 0.15]  # Top to bottom: L23, L4, L5
        
        # Column positions (with age group separations)
        col_positions = []
        for age_group in range(4):  # P4, P8, P12, P16
            base_x = left_start + age_group * (4 * col_spacing + 0.02)  # Small gap between age groups
            for condition in range(4):  # Full, E-only, Diff, Regime
                col_positions.append(base_x + condition * col_spacing)
        
        # Create subplots with manual positioning
        for layer_idx in range(3):
            for col_idx in range(16):
                left = col_positions[col_idx]
                bottom = layer_positions[layer_idx]
                axes[layer_idx, col_idx] = fig.add_axes([left, bottom, subplot_width, subplot_height])
        
        # Plot data (same logic as parent class)
        ims = {'lambda': None, 'diff': None, 'regime': None, 'thalamic': None}
        
        for age_idx, stage in enumerate(DEVELOPMENTAL_STAGES):
            if stage not in results or regime not in results[stage] or snapshot_idx not in results[stage][regime]:
                continue
                
            stage_data = results[stage][regime][snapshot_idx]
            grids = self._create_layer_grids(stage_data['layer_wise'], stage_data['layer_coords'])
            
            base_col = age_idx * 4
            
            for layer_idx, layer in enumerate(LAYERS):
                ims['lambda'] = self._plot_heatmap(axes[layer_idx, base_col + 0], grids[layer]['full'])
                ims['lambda'] = self._plot_heatmap(axes[layer_idx, base_col + 1], grids[layer]['e_only'])
                ims['diff'] = self._plot_heatmap(axes[layer_idx, base_col + 2], 
                                               grids[layer]['full'] - grids[layer]['e_only'], center_zero=True)
                ims['regime'] = self._plot_regime_heatmap(axes[layer_idx, base_col + 3], grids[layer]['regimes'])
        
        # Add row labels (not bold, closer to axes, no italic)
        layer_labels = ['L2/3', 'L4', 'L5']
        for i, label in enumerate(layer_labels):
            axes[i, 0].set_ylabel(label, fontsize=self.font_sizes['ylabel'], 
                                 rotation=0, labelpad=12, ha='right', va='center')
        
        # Add age labels above first column of each group (lower to avoid overlap)
        for i, stage in enumerate(DEVELOPMENTAL_STAGES):
            x_pos = 0.17 + ([0, 5, 10, 15][i] / 18) * 0.74
            fig.text(x_pos, 0.82, f'$\\mathbf{{{stage}}}$', ha='center', va='center',
                    fontsize=self.font_sizes['ylabel'], fontweight='bold', transform=fig.transFigure)
        
        # Add condition labels for first age group (lower than age labels)
        condition_labels = ['Full', 'E-only', 'Diff', 'Regime']
        for cond_idx, label in enumerate(condition_labels):
            axes[0, cond_idx].text(0.5, 1.25, label, transform=axes[0, cond_idx].transAxes,
                                 ha='center', va='center', fontsize=self.font_sizes['condition_labels'], 
                                 color='black')
        
        # Clean up all visible axes
        for row in range(3):
            for col in range(16):
                if axes[row, col].get_visible():
                    self._configure_axis(axes[row, col])
        
        # Add colorbars
        self._add_layer_wise_colorbars(fig, axes, ims)
        
        self._save_figure(fig, regime, snapshot_idx, 'layer_wise')
    
    def create_column_wise_figure(self, results: Dict, regime: str, snapshot_idx: int):
        """Create column-wise figure."""
        print(f"Creating column-wise figure for {regime} snapshot {snapshot_idx}")
        
        fig = self._setup_figure(regime, snapshot_idx, self.figure_sizes['column_wise'], 'Column-wise stability')
        
        # Create 5×4 grid with equal row and column spacing
        gs = fig.add_gridspec(5, 4, hspace=0.08, wspace=0.08,
                             left=0.12, right=0.78, top=0.88, bottom=0.15)
        
        axes = np.array([[fig.add_subplot(gs[row, col]) for col in range(4)] for row in range(5)])
        
        # Plot data (same logic as parent class)
        ims = {'lambda': None, 'diff': None, 'regime': None, 'thalamic': None}
        
        for col_idx, stage in enumerate(DEVELOPMENTAL_STAGES):
            if stage not in results:
                continue
                
            stage_data = results[stage][regime][snapshot_idx]
            grids = self._create_column_grids(stage_data['column_wise'], stage_data['column_coords'])
            
            ims['lambda'] = self._plot_heatmap(axes[0, col_idx], grids['full'])
            ims['lambda'] = self._plot_heatmap(axes[1, col_idx], grids['e_only'])
            ims['diff'] = self._plot_heatmap(axes[2, col_idx], grids['full'] - grids['e_only'], center_zero=True)
            ims['regime'] = self._plot_regime_heatmap(axes[3, col_idx], grids)
            ims['thalamic'] = self._plot_thalamic_input(axes[4, col_idx], stage_data['thalamic_input'])
        
        # Add row labels with larger font and reduced padding
        row_labels = ['Full Jacobian', 'E-only', 'Difference', 'Regime', 'Input']
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(f'{label}', fontsize=self.font_sizes['ylabel'], 
                                 rotation=0, labelpad=20, ha='right', va='center')
        
        # Add column labels (developmental stages)
        for j, stage in enumerate(DEVELOPMENTAL_STAGES):
            if j < 4:
                axes[0, j].text(0.5, 1.12, f'$\\mathbf{{{stage}}}$', 
                               transform=axes[0, j].transAxes, ha='center', va='center', 
                               fontsize=self.font_sizes['ylabel'], fontweight='bold')
        
        # Clean up all axes
        for row in range(5):
            for col in range(4):
                self._configure_axis(axes[row, col])
        
        # Add colorbars
        self._add_column_wise_colorbars(fig, axes, ims)
        
        self._save_figure(fig, regime, snapshot_idx, 'column_wise')
    
    def _calculate_regime_percentages(self, results: Dict) -> Dict:
        """Calculate percentage of each stability regime across snapshots and developmental stages."""
        percentages = {}
        
        for condition in REGIMES:
            percentages[condition] = {}
            
            for analysis_type in ['layer_wise', 'column_wise']:
                percentages[condition][analysis_type] = {}
                
                # Initialize regime counters for each stage
                for stage in DEVELOPMENTAL_STAGES:
                    percentages[condition][analysis_type][stage] = {
                        regime: [] for regime in REGIME_LABELS
                    }
                
                # Collect regime data across all snapshots
                for stage in DEVELOPMENTAL_STAGES:
                    if stage not in results or condition not in results[stage]:
                        continue
                    
                    # Collect regime counts for each snapshot
                    snapshot_regime_counts = {regime: [] for regime in REGIME_LABELS}
                    
                    for snapshot_idx in results[stage][condition].keys():
                        stage_data = results[stage][condition][snapshot_idx]
                        
                        if analysis_type == 'layer_wise':
                            all_regimes = []
                            for layer in LAYERS:
                                layer_regimes = stage_data['layer_wise']['regimes'][layer]
                                all_regimes.extend(layer_regimes)
                        else:  # column_wise
                            all_regimes = stage_data['column_wise']['regimes']
                        
                        # Count occurrences of each regime in this snapshot
                        total_patches = len(all_regimes)
                        if total_patches == 0:
                            continue
                            
                        for regime in REGIME_LABELS:
                            count = all_regimes.count(regime)
                            percentage = (count / total_patches) * 100
                            snapshot_regime_counts[regime].append(percentage)
                    
                    # Average across snapshots for this stage
                    for regime in REGIME_LABELS:
                        if snapshot_regime_counts[regime]:
                            avg_percentage = np.mean(snapshot_regime_counts[regime])
                            percentages[condition][analysis_type][stage][regime] = avg_percentage
                        else:
                            percentages[condition][analysis_type][stage][regime] = 0.0
        
        return percentages
    
    def _plot_regime_percentages(self, ax, percentages: Dict, condition: str, analysis_type: str, title: str):
        """Plot regime percentages for a single condition and analysis type."""
        x_positions = range(len(DEVELOPMENTAL_STAGES))
        bottom = np.zeros(len(DEVELOPMENTAL_STAGES))
        
        # Plot stacked bars in reverse order (inhibition stabilized at bottom)
        for regime in reversed(REGIME_LABELS):
            values = [percentages[condition][analysis_type][stage].get(regime, 0) for stage in DEVELOPMENTAL_STAGES]
            ax.bar(x_positions, values, bottom=bottom, 
                  color=REGIME_COLORS[regime], label=regime.replace('\n', ' '))
            bottom += values
        
        # Configure axis
        ax.set_title(title, fontsize=self.font_sizes['ylabel'], fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
    
    def create_regime_percentage_plots(self, results: Dict):
        """Create regime percentage plots."""
        print("Creating regime percentage plots...")
        
        percentages = self._calculate_regime_percentages(results)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_sizes['regime_percentages'])
        fig.suptitle('Stability Regime Ratios', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        
        plot_configs = [
            (0, 0, 'driven', 'column_wise', 'Column-wise    |    Driven'),
            (0, 1, 'idle', 'column_wise', 'Column-wise    |    Idle'),
            (1, 0, 'driven', 'layer_wise', 'Layer-wise    |    Driven'),
            (1, 1, 'idle', 'layer_wise', 'Layer-wise    |    Idle')
        ]
        
        for row, col, condition, analysis_type, title in plot_configs:
            ax = axes[row, col]
            self._plot_regime_percentages(ax, percentages, condition, analysis_type, title)
            ax.set_title(title, fontsize=self.font_sizes['ylabel'], fontweight='bold')
            
            if col == 0:
                ax.set_ylabel('Cortical area (%)', fontsize=self.font_sizes['ylabel'])
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        filepath = os.path.join(self.summary_dir, 'regime_percentages.svg')
        plt.savefig(filepath, format='svg', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved figure: {filepath}")
    
    def _calculate_inhibition_effectiveness(self, results: Dict) -> Dict:
        """Calculate inhibition effectiveness index (lambda_e_only - lambda_full) over development."""
        effectiveness = {'driven': {}, 'idle': {}}
        
        for condition in REGIMES:
            for stage in DEVELOPMENTAL_STAGES:
                if stage not in results or condition not in results[stage]:
                    continue
                
                snapshot_values = []
                
                for snapshot_idx in results[stage][condition].keys():
                    stage_data = results[stage][condition][snapshot_idx]
                    
                    # Get column-wise lambda values
                    lambda_full = stage_data['column_wise']['full']
                    lambda_e_only = stage_data['column_wise']['e_only']
                    
                    # Calculate effectiveness for each patch (positive = effective inhibition)
                    patch_effectiveness = [e_only - full for full, e_only in zip(lambda_full, lambda_e_only)]
                    
                    # Mean effectiveness across patches for this snapshot
                    snapshot_mean = np.mean(patch_effectiveness)
                    snapshot_values.append(snapshot_mean)
                
                # Store mean, std, and sample size across snapshots
                if snapshot_values:
                    effectiveness[condition][stage] = {
                        'mean': np.mean(snapshot_values),
                        'std': np.std(snapshot_values),
                        'n': len(snapshot_values)
                    }
        
        return effectiveness
    
    def _extract_effectiveness_data(self, effectiveness: Dict, condition: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract means and stds for a given condition across developmental stages."""
        means, stds = [], []
        
        for stage in DEVELOPMENTAL_STAGES:
            if stage in effectiveness[condition]:
                means.append(effectiveness[condition][stage]['mean'])
                stds.append(effectiveness[condition][stage]['std'])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        return np.array(means), np.array(stds)
    
    def _set_custom_yticks(self, ax):
        """Set custom y-ticks to show only 0 and the top value with 1 decimal place."""
        ylim = ax.get_ylim()
        _, ymax = ylim
        
        # Round to 1 decimal place and format consistently
        top_tick = round(ymax, 1)
        top_label = f'{top_tick:.1f}'
        
        ax.set_yticks([0, top_tick])
        ax.set_yticklabels(['0.0', top_label])
    
    def _plot_effectiveness_timeseries(self, ax, effectiveness: Dict, title: str, ylabel: str):
        """Plot effectiveness timeseries with error bands."""
        x_positions = range(len(DEVELOPMENTAL_STAGES))
        
        for condition in REGIMES:
            if condition in effectiveness:
                means, stds = self._extract_effectiveness_data(effectiveness, condition)
                
                # Plot mean line with error bands
                ax.plot(x_positions, means, color=LINE_COLOR, linewidth=LINE_WIDTH,
                       linestyle=LINE_STYLES[condition], label=condition.capitalize())
                ax.fill_between(x_positions, means - stds, means + stds,
                               color=SHADE_COLOR, alpha=SHADE_ALPHA)
        
        # Configure axis
        ax.set_title(title, fontsize=self.font_sizes['title'], fontweight='bold')
        ax.set_xlabel('Developmental Age', fontsize=self.font_sizes['ylabel'])
        ax.set_ylabel(ylabel, fontsize=self.font_sizes['ylabel'])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
        ax.legend(fontsize=self.font_sizes['colorbar'], frameon=True)
        ax.axhline(y=0, color=LINE_COLOR, linestyle=':', alpha=REFERENCE_LINE_ALPHA, linewidth=1)
    
    def create_inhibition_effectiveness_plot(self, results: Dict):
        """Create inhibition effectiveness plot."""
        print("Creating inhibition effectiveness plot...")
        
        effectiveness = self._calculate_inhibition_effectiveness(results)
        
        _, ax = plt.subplots(1, 1, figsize=self.figure_sizes['effectiveness'])
        self._plot_effectiveness_timeseries(ax, effectiveness, 
                                           'Inhibition Effectiveness',
                                           r'$\lambda_{\mathrm{E\text{-}only}} - \lambda_{\mathrm{full}}$')
        
        plt.tight_layout()
        filepath = os.path.join(self.summary_dir, 'inhibition_effectiveness.svg')
        plt.savefig(filepath, format='svg', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved figure: {filepath}")
    
    def create_layer_effectiveness_plot(self, results: Dict):
        """Create layer-specific inhibition effectiveness plot."""
        print("Creating layer-specific inhibition effectiveness plot...")
        
        layer_effectiveness = self._calculate_layer_effectiveness(results)
        
        # Use standardized layer colors
        LAYER_COLORS_CLEAN = STANDARD_COLORS['layers']
        
        _, ax = plt.subplots(1, 1, figsize=self.figure_sizes['effectiveness'])
        x_positions = range(len(DEVELOPMENTAL_STAGES))
        
        # Plot each layer with updated colors and markers
        for layer in LAYERS:
            display_layer = 'L2/3' if layer == 'L23' else layer
            means, sems = [], []
            ns = []  # Sample sizes for SEM calculation
            
            for stage in DEVELOPMENTAL_STAGES:
                if stage in layer_effectiveness[layer]:
                    means.append(layer_effectiveness[layer][stage]['mean'])
                    # Convert std to SEM: SEM = std / sqrt(n)
                    std = layer_effectiveness[layer][stage]['std']
                    n = layer_effectiveness[layer][stage].get('n', 1)  # Get sample size
                    sem = std / np.sqrt(n) if n > 0 else 0
                    sems.append(sem)
                    ns.append(n)
                else:
                    means.append(np.nan)
                    sems.append(np.nan)
                    ns.append(0)
            
            means, sems = np.array(means), np.array(sems)
            color = LAYER_COLORS_CLEAN[display_layer]
            
            # Plot line with filled circular markers
            ax.plot(x_positions, means, color=color, linewidth=3, 
                   linestyle='-', label=display_layer, marker='o', 
                   markersize=8, markerfacecolor=color, markeredgecolor=color)
            
            # Add SEM shaded regions with no border
            ax.fill_between(x_positions, means - sems, means + sems, 
                           color=color, alpha=0.3, linewidth=0, edgecolor='none')
        
        # Configure axis
        ax.set_title('Inhibition Effectiveness', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        ax.set_xlabel('Developmental Age', fontsize=self.font_sizes['ylabel'])
        ax.set_ylabel(r'$\lambda_{\mathrm{E\text{-}only}} - \lambda_{\mathrm{full}}$', 
                     fontsize=self.font_sizes['ylabel'])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
        ax.legend(fontsize=self.font_sizes['colorbar'], frameon=True)
        ax.axhline(y=0, color='lightgrey', linestyle='-', alpha=0.8, linewidth=2, zorder=0)
        
        # Custom y-ticks: only 0 and top value
        self._set_custom_yticks(ax)
        
        plt.tight_layout()
        filepath = os.path.join(self.summary_dir, 'layer_specific_inhibition_effectiveness.svg')
        plt.savefig(filepath, format='svg', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved figure: {filepath}")
    
    def _calculate_layer_effectiveness(self, results: Dict) -> Dict:
        """Calculate layer-specific inhibition effectiveness for driven condition only."""
        layer_effectiveness = {layer: {} for layer in LAYERS}
        
        for layer in LAYERS:
            for stage in DEVELOPMENTAL_STAGES:
                if stage not in results or 'driven' not in results[stage]:
                    continue
                
                # Collect effectiveness values from driven condition only
                snapshot_values = []
                
                for snapshot_idx in results[stage]['driven'].keys():
                    stage_data = results[stage]['driven'][snapshot_idx]
                    
                    # Get layer-wise data for this layer
                    lambda_full = stage_data['layer_wise']['full'][layer]
                    lambda_e_only = stage_data['layer_wise']['e_only'][layer]
                    
                    # Calculate effectiveness for each patch in this layer
                    patch_effectiveness = [e_only - full for full, e_only in zip(lambda_full, lambda_e_only)]
                    
                    # Mean effectiveness across patches for this snapshot/layer
                    snapshot_mean = np.mean(patch_effectiveness)
                    snapshot_values.append(snapshot_mean)
                
                # Store statistics across snapshots
                if snapshot_values:
                    layer_effectiveness[layer][stage] = {
                        'mean': np.mean(snapshot_values),
                        'std': np.std(snapshot_values),
                        'n': len(snapshot_values)
                    }
        
        return layer_effectiveness
    
    def _collect_phase_diagram_data(self, results: Dict) -> Dict:
        """Collect lambda_full vs lambda_e_only data for phase diagrams."""
        phase_data = {}
        
        for stage in DEVELOPMENTAL_STAGES:
            phase_data[stage] = {'driven': {'full': [], 'e_only': []}, 
                               'idle': {'full': [], 'e_only': []}}
            
            for condition in REGIMES:
                if stage not in results or condition not in results[stage]:
                    continue
                
                for snapshot_idx in results[stage][condition].keys():
                    stage_data = results[stage][condition][snapshot_idx]
                    
                    # Get column-wise data for all patches in this snapshot
                    lambda_full = stage_data['column_wise']['full']
                    lambda_e_only = stage_data['column_wise']['e_only']
                    
                    # Add each patch as a separate point
                    phase_data[stage][condition]['full'].extend(lambda_full)
                    phase_data[stage][condition]['e_only'].extend(lambda_e_only)
        
        return phase_data
    
    def _determine_phase_diagram_limits(self, phase_data: Dict) -> Tuple[float, float]:
        """Determine consistent axis limits across all stages and conditions."""
        all_full_values = []
        all_e_only_values = []
        
        for stage in DEVELOPMENTAL_STAGES:
            for condition in REGIMES:
                if stage in phase_data:
                    all_full_values.extend(phase_data[stage][condition]['full'])
                    all_e_only_values.extend(phase_data[stage][condition]['e_only'])
        
        if all_full_values and all_e_only_values:
            full_range = max(all_full_values) - min(all_full_values)
            e_only_range = max(all_e_only_values) - min(all_e_only_values)
            
            # Use the larger range to make square plots centered at zero
            max_range = max(full_range, e_only_range)
            limit = max_range / 2 * 1.1  # Add 10% padding
            
            return -limit, limit
        else:
            return -1, 1  # fallback
    
    def _add_quadrant_shading(self, ax, lim_min: float, lim_max: float):
        """Add background shading for each stability regime quadrant with increased alpha for poster."""
        quadrants = [
            # (x_range, y_range, regime_key)
            ([lim_min, 0], [lim_min, 0], 'intrinsically \n stable'),      # Bottom-left
            ([lim_min, 0], [0, lim_max], 'inhibition \n stabilised'),     # Top-left  
            ([0, lim_max], [lim_min, 0], 'inhibition \n destabilised'),   # Bottom-right
            ([0, lim_max], [0, lim_max], 'intrinsically \n unstable')     # Top-right
        ]
        
        # Increased alpha for better visibility in poster
        poster_alpha = 0.35
        
        for x_range, y_range, regime_key in quadrants:
            ax.fill_between(x_range, y_range[0], y_range[1], 
                          color=REGIME_COLORS[regime_key], alpha=poster_alpha, zorder=0)
    
    def _add_reference_lines(self, ax, lim_min: float, lim_max: float):
        """Add reference lines (axes and diagonal)."""
        # Thick reference lines at zero
        ax.axhline(y=0, color=LINE_COLOR, linestyle='-', 
                  alpha=REFERENCE_LINE_ALPHA_THICK, linewidth=REFERENCE_LINE_WIDTH, zorder=1)
        ax.axvline(x=0, color=LINE_COLOR, linestyle='-', 
                  alpha=REFERENCE_LINE_ALPHA_THICK, linewidth=REFERENCE_LINE_WIDTH, zorder=1)
        # Diagonal line
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', 
               alpha=REFERENCE_LINE_ALPHA, linewidth=1, zorder=1)
    
    def _configure_phase_subplot(self, ax, stage: str, lim_min: float, lim_max: float, 
                               show_legend: bool = False):
        """Configure individual phase diagram subplot."""
        ax.set_title(stage, fontsize=self.font_sizes['ylabel'], fontweight='bold')
        ax.set_xlabel(r'$\lambda_{\mathrm{full}}$', fontsize=self.font_sizes['ylabel'])
        ax.set_ylabel(r'$\lambda_{\mathrm{E\text{-}only}}$', fontsize=self.font_sizes['ylabel'])
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal', adjustable='box')
        ax.locator_params(axis='both', nbins=4)
        
        # Add visual elements
        self._add_quadrant_shading(ax, lim_min, lim_max)
        self._add_reference_lines(ax, lim_min, lim_max)
        
        if show_legend:
            ax.legend(fontsize=self.font_sizes['colorbar'], frameon=True, loc='upper left')
    
    def create_stability_phase_diagrams(self, results: Dict):
        """Create stability phase diagrams."""
        print("Creating stability phase diagrams...")
        
        phase_data = self._collect_phase_diagram_data(results)
        lim_min, lim_max = self._determine_phase_diagram_limits(phase_data)
        
        fig, axes = plt.subplots(1, 4, figsize=self.figure_sizes['phase_diagram'])
        fig.suptitle('Stability Phase Diagrams', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        
        for i, stage in enumerate(DEVELOPMENTAL_STAGES):
            ax = axes[i]
            
            if stage in phase_data:
                for condition in REGIMES:
                    if phase_data[stage][condition]['full']:
                        ax.scatter(phase_data[stage][condition]['full'],
                                 phase_data[stage][condition]['e_only'],
                                 c='black', alpha=0.1, s=5, 
                                 marker='o' if condition == 'driven' else '^',
                                 edgecolors='none', label=condition.capitalize())
            
            self._configure_phase_subplot(ax, stage, lim_min, lim_max, show_legend=(i == 0))
            ax.set_xlabel(r'$\lambda_{\mathrm{full}}$', fontsize=self.font_sizes['ylabel'])
            
            # Only show y-axis label on leftmost plot
            if i == 0:
                ax.set_ylabel(r'$\lambda_{\mathrm{E\text{-}only}}$', fontsize=self.font_sizes['ylabel'])
            else:
                ax.set_ylabel('')  # Remove y-axis label for non-leftmost plots
            
            ax.set_title(stage, fontsize=self.font_sizes['ylabel'], fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.summary_dir, 'phase_diagrams.svg')
        plt.savefig(filepath, format='svg', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved figure: {filepath}")
    
    def generate_all_figures(self, results: Dict):
        """Generate all visualization figures."""
        print("Generating all stability analysis figures...")
        
        # Generate regime percentage plots first
        self.create_regime_percentage_plots(results)
        
        # Generate inhibition effectiveness plot
        self.create_inhibition_effectiveness_plot(results)
        
        # Generate layer-specific effectiveness plot
        self.create_layer_effectiveness_plot(results)
        
        # Generate cell-type specific effectiveness plots
        self.create_celltype_effectiveness_plot(results)
        
        # Generate layer-specific SST/PV heatmaps
        self.create_layer_specific_heatmaps(results)
        
        # Generate stability phase diagrams
        self.create_stability_phase_diagrams(results)
        
        # Collect available snapshots
        snapshots_by_regime = {}
        for regime in REGIMES:
            snapshots = set()
            for stage in DEVELOPMENTAL_STAGES:
                if stage in results and regime in results[stage]:
                    snapshots.update(results[stage][regime].keys())
            snapshots_by_regime[regime] = snapshots
        
        # Generate figures for each regime and snapshot
        for regime in REGIMES:
            for snapshot_idx in snapshots_by_regime[regime]:
                # Check if we have enough stages for this snapshot
                stages_with_snapshot = [
                    stage for stage in DEVELOPMENTAL_STAGES 
                    if (stage in results and regime in results[stage] and 
                        snapshot_idx in results[stage][regime])
                ]
                
                if len(stages_with_snapshot) >= 2:
                    self.create_layer_wise_figure(results, regime, snapshot_idx)
                    self.create_column_wise_figure(results, regime, snapshot_idx)
        
        print("All figures generated successfully!")
    
    def _calculate_celltype_effectiveness(self, results: Dict) -> Dict:
        """Calculate cell-type specific inhibition effectiveness."""
        effectiveness = {
            'SST_relative': {'driven': {}, 'idle': {}},       # lambda_e_pv_only - lambda_full (SST vs PV-only)
            'PV_relative': {'driven': {}, 'idle': {}},        # lambda_e_sst_only - lambda_full (PV vs SST-only)
            'SST_absolute': {'driven': {}, 'idle': {}},       # lambda_e_only - lambda_e_sst_only (SST vs no inhibition)
            'PV_absolute': {'driven': {}, 'idle': {}}         # lambda_e_only - lambda_e_pv_only (PV vs no inhibition)
        }
        
        for condition in REGIMES:
            for stage in DEVELOPMENTAL_STAGES:
                if stage not in results or condition not in results[stage]:
                    continue
                
                # Collect values across snapshots
                sst_relative_values = []
                pv_relative_values = []
                sst_absolute_values = []
                pv_absolute_values = []
                
                for snapshot_idx in results[stage][condition].keys():
                    stage_data = results[stage][condition][snapshot_idx]
                    
                    # Get column-wise data
                    lambda_full = stage_data['column_wise']['full']
                    lambda_e_only = stage_data['column_wise']['e_only']
                    lambda_e_pv_only = stage_data['column_wise']['e_pv_only']
                    lambda_e_sst_only = stage_data['column_wise']['e_sst_only']
                    
                    # Calculate relative effectiveness for each patch (how much each adds beyond the other)
                    sst_relative = [e_pv - full for full, e_pv in zip(lambda_full, lambda_e_pv_only)]
                    pv_relative = [e_sst - full for full, e_sst in zip(lambda_full, lambda_e_sst_only)]
                    
                    # Calculate absolute effectiveness for each patch (how much each adds from baseline)
                    sst_absolute = [e_only - e_sst for e_only, e_sst in zip(lambda_e_only, lambda_e_sst_only)]
                    pv_absolute = [e_only - e_pv for e_only, e_pv in zip(lambda_e_only, lambda_e_pv_only)]
                    
                    # Mean across patches for this snapshot
                    sst_relative_values.append(np.mean(sst_relative))
                    pv_relative_values.append(np.mean(pv_relative))
                    sst_absolute_values.append(np.mean(sst_absolute))
                    pv_absolute_values.append(np.mean(pv_absolute))
                
                # Store statistics across snapshots
                if sst_relative_values:
                    effectiveness['SST_relative'][condition][stage] = {
                        'mean': np.mean(sst_relative_values),
                        'std': np.std(sst_relative_values)
                    }
                    effectiveness['PV_relative'][condition][stage] = {
                        'mean': np.mean(pv_relative_values),
                        'std': np.std(pv_relative_values)
                    }
                    effectiveness['SST_absolute'][condition][stage] = {
                        'mean': np.mean(sst_absolute_values),
                        'std': np.std(sst_absolute_values)
                    }
                    effectiveness['PV_absolute'][condition][stage] = {
                        'mean': np.mean(pv_absolute_values),
                        'std': np.std(pv_absolute_values)
                    }
        
        return effectiveness
    
    def _calculate_layer_specific_effectiveness(self, results: Dict) -> Dict:
        """Calculate layer-specific cell-type effectiveness for SST and PV across layers."""
        effectiveness = {
            'SST_relative': {},       # lambda_e_pv_only - lambda_full (SST vs PV-only)
            'PV_relative': {},        # lambda_e_sst_only - lambda_full (PV vs SST-only)
            'SST_absolute': {},       # lambda_e_only - lambda_e_sst_only (SST vs no inhibition)
            'PV_absolute': {}         # lambda_e_only - lambda_e_pv_only (PV vs no inhibition)
        }
        
        # Initialize data structure for each layer and cell type
        for layer in LAYERS:
            for cell_type in ['SST', 'PV']:
                effectiveness[f'{cell_type}_relative'][layer] = {}
                effectiveness[f'{cell_type}_absolute'][layer] = {}
        
        # Only analyze driven condition as requested
        condition = 'driven'
        
        for stage in DEVELOPMENTAL_STAGES:
            if stage not in results or condition not in results[stage]:
                continue
            
            for layer in LAYERS:
                # Collect values across snapshots for this layer
                sst_relative_values = []
                pv_relative_values = []
                sst_absolute_values = []
                pv_absolute_values = []
                
                for snapshot_idx in results[stage][condition].keys():
                    stage_data = results[stage][condition][snapshot_idx]
                    
                    # Get layer-wise data for this specific layer
                    lambda_full = stage_data['layer_wise']['full'][layer]
                    lambda_e_only = stage_data['layer_wise']['e_only'][layer]
                    lambda_e_pv_only = stage_data['layer_wise']['e_pv_only'][layer]
                    lambda_e_sst_only = stage_data['layer_wise']['e_sst_only'][layer]
                    
                    # Calculate relative effectiveness for each patch in this layer (how much each adds beyond the other)
                    sst_relative = [e_pv - full for full, e_pv in zip(lambda_full, lambda_e_pv_only)]
                    pv_relative = [e_sst - full for full, e_sst in zip(lambda_full, lambda_e_sst_only)]
                    
                    # Calculate absolute effectiveness for each patch in this layer (how much each adds from baseline)
                    sst_absolute = [e_only - e_sst for e_only, e_sst in zip(lambda_e_only, lambda_e_sst_only)]
                    pv_absolute = [e_only - e_pv for e_only, e_pv in zip(lambda_e_only, lambda_e_pv_only)]
                    
                    # Mean across patches for this snapshot/layer
                    sst_relative_values.append(np.mean(sst_relative))
                    pv_relative_values.append(np.mean(pv_relative))
                    sst_absolute_values.append(np.mean(sst_absolute))
                    pv_absolute_values.append(np.mean(pv_absolute))
                
                # Store statistics across snapshots for this layer
                if sst_relative_values:
                    effectiveness['SST_relative'][layer][stage] = {
                        'mean': np.mean(sst_relative_values),
                        'std': np.std(sst_relative_values)
                    }
                    effectiveness['PV_relative'][layer][stage] = {
                        'mean': np.mean(pv_relative_values),
                        'std': np.std(pv_relative_values)
                    }
                    effectiveness['SST_absolute'][layer][stage] = {
                        'mean': np.mean(sst_absolute_values),
                        'std': np.std(sst_absolute_values)
                    }
                    effectiveness['PV_absolute'][layer][stage] = {
                        'mean': np.mean(pv_absolute_values),
                        'std': np.std(pv_absolute_values)
                    }
        
        return effectiveness
    
    def create_celltype_effectiveness_plot(self, results: Dict):
        """Create comprehensive cell-type specific inhibition effectiveness plot."""
        print("Creating cell-type specific inhibition effectiveness plot...")
        
        effectiveness = self._calculate_celltype_effectiveness(results)
        
        # Create 2x2 subplot layout
        _, axes = plt.subplots(2, 2, figsize=self.figure_sizes['effectiveness_2x2'])
        x_positions = range(len(DEVELOPMENTAL_STAGES))
        
        # Use standardized colors
        colors = STANDARD_COLORS['cell_types']
        
        # Plot 1: SST vs PV Relative Effectiveness (driven condition)
        ax = axes[0, 0]
        for cell_type in ['SST', 'PV']:
            key = f'{cell_type}_relative'
            means, stds = self._extract_effectiveness_data(effectiveness[key], 'driven')
            ax.plot(x_positions, means, color=colors[cell_type], linewidth=LINE_WIDTH,
                   linestyle='-', label=cell_type, marker='o', markersize=6)
            ax.fill_between(x_positions, means - stds, means + stds,
                           color=colors[cell_type], alpha=SHADE_ALPHA*0.5)
        
        # ax.set_title('Cell-Type Relative Effectiveness (Driven)', fontsize=self.font_sizes['title'], fontweight='bold')  # Title removed
        ax.set_ylabel(r'$\lambda_{\mathrm{partial}} - \lambda_{\mathrm{full}}$', 
                     fontsize=self.font_sizes['ylabel'])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
        ax.legend(fontsize=self.font_sizes['colorbar'])
        ax.axhline(y=0, color=LINE_COLOR, linestyle=':', alpha=REFERENCE_LINE_ALPHA, linewidth=1)
        # Set custom y-ticks: only 0 and max value
        self._set_custom_yticks(ax)
        
        # Plot 2: SST vs PV Absolute Effectiveness (driven condition)
        ax = axes[0, 1]
        for cell_type in ['SST', 'PV']:
            key = f'{cell_type}_absolute'
            means, stds = self._extract_effectiveness_data(effectiveness[key], 'driven')
            ax.plot(x_positions, means, color=colors[cell_type], linewidth=LINE_WIDTH,
                   linestyle='-', label=cell_type, marker='s', markersize=6)
            ax.fill_between(x_positions, means - stds, means + stds,
                           color=colors[cell_type], alpha=SHADE_ALPHA*0.5)
        
        # ax.set_title('Cell-Type Absolute Effectiveness (Driven)', fontsize=self.font_sizes['title'], fontweight='bold')  # Title removed
        ax.set_ylabel(r'$\lambda_{\mathrm{E\text{-}only}} - \lambda_{\mathrm{partial}}$', 
                     fontsize=self.font_sizes['ylabel'])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
        ax.legend(fontsize=self.font_sizes['colorbar'])
        ax.axhline(y=0, color=LINE_COLOR, linestyle=':', alpha=REFERENCE_LINE_ALPHA, linewidth=1)
        # Set custom y-ticks: only 0 and max value
        self._set_custom_yticks(ax)
        
        # Plot 3: SST vs PV Relative Effectiveness (idle condition)
        ax = axes[1, 0]
        for cell_type in ['SST', 'PV']:
            key = f'{cell_type}_relative'
            means, stds = self._extract_effectiveness_data(effectiveness[key], 'idle')
            ax.plot(x_positions, means, color=colors[cell_type], linewidth=LINE_WIDTH,
                   linestyle='-', label=cell_type, marker='o', markersize=6)
            ax.fill_between(x_positions, means - stds, means + stds,
                           color=colors[cell_type], alpha=SHADE_ALPHA*0.5)
        
        # ax.set_title('Cell-Type Relative Effectiveness (Idle)', fontsize=self.font_sizes['title'], fontweight='bold')  # Title removed
        ax.set_xlabel('Developmental Age', fontsize=self.font_sizes['ylabel'])
        ax.set_ylabel(r'$\lambda_{\mathrm{partial}} - \lambda_{\mathrm{full}}$', 
                     fontsize=self.font_sizes['ylabel'])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
        ax.legend(fontsize=self.font_sizes['colorbar'])
        ax.axhline(y=0, color=LINE_COLOR, linestyle=':', alpha=REFERENCE_LINE_ALPHA, linewidth=1)
        # Set custom y-ticks: only 0 and max value
        self._set_custom_yticks(ax)
        
        # Plot 4: SST vs PV Absolute Effectiveness (idle condition)
        ax = axes[1, 1]
        for cell_type in ['SST', 'PV']:
            key = f'{cell_type}_absolute'
            means, stds = self._extract_effectiveness_data(effectiveness[key], 'idle')
            ax.plot(x_positions, means, color=colors[cell_type], linewidth=LINE_WIDTH,
                   linestyle='-', label=cell_type, marker='s', markersize=6)
            ax.fill_between(x_positions, means - stds, means + stds,
                           color=colors[cell_type], alpha=SHADE_ALPHA*0.5)
        
        # ax.set_title('Cell-Type Absolute Effectiveness (Idle)', fontsize=self.font_sizes['title'], fontweight='bold')  # Title removed
        ax.set_xlabel('Developmental Age', fontsize=self.font_sizes['ylabel'])
        ax.set_ylabel(r'$\lambda_{\mathrm{E\text{-}only}} - \lambda_{\mathrm{partial}}$', 
                     fontsize=self.font_sizes['ylabel'])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
        ax.legend(fontsize=self.font_sizes['colorbar'])
        ax.axhline(y=0, color=LINE_COLOR, linestyle=':', alpha=REFERENCE_LINE_ALPHA, linewidth=1)
        # Set custom y-ticks: only 0 and max value
        self._set_custom_yticks(ax)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        filepath = os.path.join(self.summary_dir, 'celltype_specific_inhibition_effectiveness.svg')
        plt.savefig(filepath, format='svg', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved figure: {filepath}")
    
    def create_layer_specific_heatmaps(self, results: Dict):
        """Create layer-specific SST/PV effectiveness and contribution heatmaps for driven condition."""
        print("Creating layer-specific SST/PV heatmaps...")
        
        effectiveness = self._calculate_layer_specific_effectiveness(results)
        
        # Create data matrices for heatmaps
        # Y-axis: [SST L5, PV L5, SST L4, PV L4, SST L23, PV L23] (deep to superficial)
        # X-axis: developmental stages
        y_labels = ['SST L5', 'PV L5', 'SST L4', 'PV L4', 'SST L23', 'PV L23']
        x_labels = DEVELOPMENTAL_STAGES
        
        # Prepare data matrices
        effectiveness_matrix = np.zeros((len(y_labels), len(x_labels)))
        contribution_matrix = np.zeros((len(y_labels), len(x_labels)))
        
        # Fill matrices with data (deep to superficial order)
        layer_order = ['L5', 'L4', 'L23']  # Deep to superficial
        
        for i, layer in enumerate(layer_order):
            for j, cell_type in enumerate(['SST', 'PV']):
                row_idx = i * 2 + j  # SST then PV for each layer
                
                for k, stage in enumerate(x_labels):
                    # Relative effectiveness: lambda_e_pv_only - lambda_full (for SST) or lambda_e_sst_only - lambda_full (for PV)
                    if cell_type == 'SST':
                        if layer in effectiveness['SST_relative'] and stage in effectiveness['SST_relative'][layer]:
                            effectiveness_matrix[row_idx, k] = effectiveness['SST_relative'][layer][stage]['mean']
                    else:  # PV
                        if layer in effectiveness['PV_relative'] and stage in effectiveness['PV_relative'][layer]:
                            effectiveness_matrix[row_idx, k] = effectiveness['PV_relative'][layer][stage]['mean']
                    
                    # Absolute effectiveness: lambda_e_only - lambda_e_sst_only (for SST) or lambda_e_only - lambda_e_pv_only (for PV)
                    if cell_type == 'SST':
                        if layer in effectiveness['SST_absolute'] and stage in effectiveness['SST_absolute'][layer]:
                            contribution_matrix[row_idx, k] = effectiveness['SST_absolute'][layer][stage]['mean']
                    else:  # PV
                        if layer in effectiveness['PV_absolute'] and stage in effectiveness['PV_absolute'][layer]:
                            contribution_matrix[row_idx, k] = effectiveness['PV_absolute'][layer][stage]['mean']
        
        # Create figure with two subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_sizes['heatmap_dual'])
        
        # Make both subplots square
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        
        cmap = STANDARD_COLORS['colormaps']['heatmap']
        
        # Use separate color scales for each subplot
        # Effectiveness heatmap
        eff_values = effectiveness_matrix.flatten()
        eff_values = eff_values[~np.isnan(eff_values)]  # Remove NaN values
        eff_vmin = 0.0  # Start at zero
        eff_vmax = np.max(eff_values) if len(eff_values) > 0 else 1.0  # Use max value for effectiveness
        
        # Contribution heatmap  
        contrib_values = contribution_matrix.flatten()
        contrib_values = contrib_values[~np.isnan(contrib_values)]  # Remove NaN values
        contrib_vmin = 0.0  # Start at zero
        contrib_vmax = np.max(contrib_values) if len(contrib_values) > 0 else 1.0  # Use max value for contribution
        
        # Plot relative effectiveness heatmap
        im1 = ax1.imshow(effectiveness_matrix, cmap=cmap, aspect='auto', vmin=eff_vmin, vmax=eff_vmax)
        # ax1.set_title('Layer-Specific Relative Effectiveness (Driven)', fontsize=self.font_sizes['title'], fontweight='bold')  # Title removed
        # ax1.set_xlabel('Developmental Age', fontsize=self.font_sizes['ylabel'])  # X-axis label removed
        # ax1.set_ylabel('Cell Type & Layer', fontsize=self.font_sizes['ylabel'])  # Y-axis label removed
        ax1.set_xticks(range(len(x_labels)))
        ax1.set_xticklabels(x_labels)
        ax1.set_yticks(range(len(y_labels)))
        ax1.set_yticklabels(y_labels)
        ax1.invert_yaxis()  # Flip y-axis so SST L5 is at bottom
        
        # Add colorbar for relative effectiveness
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label(r'$\lambda_{\mathrm{partial}} - \lambda_{\mathrm{full}}$', 
                       fontsize=self.font_sizes['colorbar'])
        # Set custom ticks: only 0 and vmax
        cbar1.set_ticks([0, eff_vmax])
        cbar1.set_ticklabels([f'{0:.2f}', f'{eff_vmax:.2f}'])
        
        # Plot absolute effectiveness heatmap
        im2 = ax2.imshow(contribution_matrix, cmap=cmap, aspect='auto', vmin=contrib_vmin, vmax=contrib_vmax)
        # ax2.set_title('Layer-Specific Absolute Effectiveness (Driven)', fontsize=self.font_sizes['title'], fontweight='bold')  # Title removed
        # ax2.set_xlabel('Developmental Age', fontsize=self.font_sizes['ylabel'])  # X-axis label removed
        # ax2.set_ylabel('Cell Type & Layer', fontsize=self.font_sizes['ylabel'])  # Y-axis label removed
        ax2.set_xticks(range(len(x_labels)))
        ax2.set_xticklabels(x_labels)
        ax2.set_yticks(range(len(y_labels)))
        ax2.set_yticklabels(y_labels)
        ax2.invert_yaxis()  # Flip y-axis so SST L5 is at bottom
        
        # Add colorbar for absolute effectiveness
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label(r'$\lambda_{\mathrm{E\text{-}only}} - \lambda_{\mathrm{partial}}$', 
                       fontsize=self.font_sizes['colorbar'])
        # Set custom ticks: only 0 and vmax
        cbar2.set_ticks([0, contrib_vmax])
        cbar2.set_ticklabels([f'{0:.2f}', f'{contrib_vmax:.2f}'])
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4)
        
        # Save effectiveness heatmap
        filepath_eff = os.path.join(self.summary_dir, 'layer_specific_effectiveness_driven.svg')
        plt.savefig(filepath_eff, format='svg', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved effectiveness heatmap: {filepath_eff}")
    
