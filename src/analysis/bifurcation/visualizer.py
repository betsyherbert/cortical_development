"""Visualization module for bifurcation analysis.

This module handles all figure generation for stability maps, gain maps,
and gain spectra, ensuring consistent styling and layout across all analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple

from .core import get_nested_value
from .config import (
    OUTPUT_DIR,
    SCANNABLE_PARAMETERS,
    BIFURCATION_COLORMAP,
    GAIN_COLORMAP,
    SPECTRUM_COLORMAP,
    OPACITY_STABLE_FAR,
    OPACITY_STABLE_NEAR,
    OPACITY_UNSTABLE,
    STABILITY_THRESHOLD,
    GAIN_CLIP_MAX,
    GAIN_OPACITY_MIN,
    GAIN_OPACITY_MAX,
    SPECTRUM_LOG_SCALE,
)


class BifurcationVisualizer:
    """Handles all bifurcation visualization with consistent styling."""
    
    def __init__(self):
        """Initialize visualizer with default style settings."""
        # Figure dimensions
        self.fig_width = 13.7
        self.fig_height_per_row = 3.5
        
        # Spine widths
        self.default_spine_width = 0.8
        self.bold_spine_width = 1.6
        
        # Font sizes
        self.title_fontsize = 14
        self.subtitle_fontsize = 13
        self.label_fontsize = 11
        self.secondary_label_fontsize = 10
        self.tick_fontsize = 9
        self.secondary_tick_fontsize = 8
        
        # Layout parameters
        self.hspace = 0.45
        self.wspace = 0.18
        self.left_margin = 0.07
        self.right_margin = 0.83
        self.top_margin = 0.74
        self.bottom_margin = 0.08
    
    def create_stability_map_figure(
        self,
        results: Dict,
        param_pairs: List[Tuple[str, str]],
        stages: List[str],
        mode: str = 'fixed_absolute'
    ) -> plt.Figure:
        """Create multi-row, multi-stage 2D stability map figure.
        
        Args:
            results: Results dict organized as {param_pair: {stage: stage_results}}
            param_pairs: List of tuples (param_x_key, param_y_key)
            stages: List of stage names
            mode: Range mode for axis emphasis
            
        Returns:
            matplotlib Figure object
        """
        n_rows = len(param_pairs)
        n_stages = len(stages)
        
        # Determine spine widths based on mode
        emphasize_ratio = mode == 'fixed_ratio'
        emphasize_absolute = mode == 'fixed_absolute'
        primary_width = self.bold_spine_width if emphasize_ratio else self.default_spine_width
        secondary_width = self.bold_spine_width if emphasize_absolute else self.default_spine_width
        
        # Create figure with n_rows×n_stages grid
        fig_height = self.fig_height_per_row * n_rows
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(n_rows, n_stages, figure=fig,
                     hspace=self.hspace, wspace=self.wspace,
                     left=self.left_margin, right=0.88,
                     top=0.92, bottom=0.08)
        
        # Determine global k range for consistent colormap across all rows
        all_k_values = []
        for param_pair in param_pairs:
            if param_pair in results:
                for stage in stages:
                    if stage in results[param_pair]:
                        all_k_values.append(results[param_pair][stage]['k_matrix'])
        
        if all_k_values:
            k_min = 0
            k_max = min(np.max([np.max(k) for k in all_k_values]), 5.0)
        else:
            k_min, k_max = 0, 5.0
        
        norm = colors.Normalize(vmin=k_min, vmax=k_max)
        cmap = plt.colormaps[BIFURCATION_COLORMAP]
        
        # Plot each row (parameter pair) and column (stage)
        for row_idx, param_pair in enumerate(param_pairs):
            param_x_key, param_y_key = param_pair
            param_x_spec = SCANNABLE_PARAMETERS[param_x_key]
            param_y_spec = SCANNABLE_PARAMETERS[param_y_key]
            
            if param_pair not in results:
                continue
            
            pair_results = results[param_pair]
            
            # Determine consistent axis limits across all stages in this row
            x_limits_all = []
            y_limits_all = []
            for stage in stages:
                if stage in pair_results:
                    x_vals = pair_results[stage]['param_x_values']
                    y_vals = pair_results[stage]['param_y_values']
                    x_limits_all.extend([x_vals[0], x_vals[-1]])
                    y_limits_all.extend([y_vals[0], y_vals[-1]])
            
            if x_limits_all and y_limits_all:
                x_lim = (min(x_limits_all), max(x_limits_all))
                y_lim = (min(y_limits_all), max(y_limits_all))
            else:
                x_lim = (0, 1)
                y_lim = (0, 1)
            
            for stage_idx, stage_name in enumerate(stages):
                if stage_name not in pair_results:
                    continue
                
                stage_result = pair_results[stage_name]
                preset = stage_result['preset']
                x_values = stage_result['param_x_values']
                y_values = stage_result['param_y_values']
                k_matrix = stage_result['k_matrix']
                stability_matrix = stage_result['stability_matrix']
                flatness_matrix = stage_result.get('flatness_matrix', None)
                
                ax = fig.add_subplot(gs[row_idx, stage_idx])
            
            # Transpose matrices (swap axes: x=first param, y=second param)
            k_matrix_T = k_matrix.T
            stability_matrix_T = stability_matrix.T
            flatness_matrix_T = flatness_matrix.T if flatness_matrix is not None else None
            
            # Compute alpha values based on stability
            alpha_matrix = np.zeros_like(stability_matrix_T)
            alpha_matrix[stability_matrix_T < STABILITY_THRESHOLD] = OPACITY_STABLE_FAR
            alpha_matrix[(stability_matrix_T >= STABILITY_THRESHOLD) & (stability_matrix_T < 0)] = OPACITY_STABLE_NEAR
            alpha_matrix[stability_matrix_T >= 0] = OPACITY_UNSTABLE
            
            # Create RGBA image (grey for flat spectra, colormap otherwise)
            rgba_image = np.zeros((*k_matrix_T.shape, 4))
            grey_color = (0.5, 0.5, 0.5)
            for i in range(k_matrix_T.shape[0]):
                for j in range(k_matrix_T.shape[1]):
                    if flatness_matrix_T is not None and flatness_matrix_T[i, j]:
                        color_rgb = grey_color
                    else:
                        color_rgb = cmap(norm(k_matrix_T[i, j]))[:3]
                    rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])
            
            # Display image
            extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
            ax.imshow(rgba_image, origin='lower', extent=extent, interpolation='nearest')
            
            # Add stability boundary contour
            try:
                ax.contour(x_values, y_values, stability_matrix_T,
                          levels=[0], colors='white', linewidths=1.5, linestyles='--', alpha=0.8)
            except (ValueError, RuntimeError):
                pass
            
            # Mark preset point
            preset_x = stage_result['preset_x_value']
            preset_y = stage_result['preset_y_value']
            ax.scatter(preset_x, preset_y,
                      marker='o', s=120, edgecolor='black', linewidth=1.5,
                      facecolor='white', zorder=10)
            
            # Apply consistent axis limits for this row
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('auto')
            ax.locator_params(axis='x', nbins=4)
            ax.locator_params(axis='y', nbins=5)
            
            # Primary axes spines
            ax.spines['bottom'].set_linewidth(primary_width)
            ax.spines['left'].set_linewidth(primary_width)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Axis labels
            if stage_idx == 0:
                ax.set_ylabel(param_y_spec.get_axis_label(absolute=False), 
                            fontsize=self.label_fontsize, labelpad=8)
            else:
                ax.set_ylabel('')
            
            # Only show x-labels on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel(param_x_spec.get_axis_label(absolute=False),
                             fontsize=self.label_fontsize, labelpad=8)
            else:
                ax.set_xlabel('')
            
            ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.5)
            
            # Secondary axes (absolute values)
            # Get reference values for ratio conversion
            if param_y_spec.use_ratio and param_y_spec.reference_param:
                ref_spec_y = SCANNABLE_PARAMETERS[param_y_spec.reference_param]
                ref_value_y = get_nested_value(preset, ref_spec_y.path)
            else:
                ref_value_y = 1.0
            
            ax2 = ax.secondary_yaxis('right', functions=(
                lambda x, rv=ref_value_y: x * rv,
                lambda x, rv=ref_value_y: x / rv
            ))
            if stage_idx == n_stages - 1:
                ax2.set_ylabel(param_y_spec.get_axis_label(absolute=True),
                             fontsize=self.secondary_label_fontsize, labelpad=8)
                ax2.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
            else:
                ax2.set_ylabel('')
                ax2.tick_params(labelright=False, length=0)
            ax2.spines['right'].set_linewidth(secondary_width)
            for spine_name in ['left', 'top', 'bottom']:
                ax2.spines[spine_name].set_visible(False)
            
            # Top axis - only show on top row
            if param_x_spec.use_ratio and param_x_spec.reference_param:
                ref_spec_x = SCANNABLE_PARAMETERS[param_x_spec.reference_param]
                ref_value_x = get_nested_value(preset, ref_spec_x.path)
            else:
                ref_value_x = 1.0
            
            ax3 = ax.secondary_xaxis('top', functions=(
                lambda x, rv=ref_value_x: x * rv,
                lambda x, rv=ref_value_x: x / rv
            ))
            if row_idx == 0:
                ax3.set_xlabel(param_x_spec.get_axis_label(absolute=True),
                              fontsize=self.secondary_label_fontsize, labelpad=8)
                ax3.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
            else:
                ax3.set_xlabel('')
                ax3.tick_params(labeltop=False, length=0)
            ax3.spines['top'].set_linewidth(secondary_width)
            for spine_name in ['bottom', 'left', 'right']:
                ax3.spines[spine_name].set_visible(False)
            
            # Stage label - only on top row
            if row_idx == 0:
                ax.set_title(stage_name, fontsize=self.subtitle_fontsize, 
                            fontweight='bold', pad=21)
        
        # Add shared colorbar spanning all rows
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Position colorbar to span full height of plots
        cbar_bottom = 0.08 + 0.05
        cbar_top = 0.92 - 0.05
        cbar_height = cbar_top - cbar_bottom
        cbar_ax = fig.add_axes([0.905, cbar_bottom, 0.015, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Spatial freq. w/ max Re($\\lambda$)', 
                      fontsize=self.label_fontsize, labelpad=18)
        cbar.ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.6)
        
        # Add grey legend at bottom of colorbar
        grey_legend_bottom = 0.08 + 0.01
        stable_ax = fig.add_axes([0.905, grey_legend_bottom, 0.015, 0.03])
        stable_ax.imshow(np.full((10, 1), 0.5), cmap='Greys', vmin=0, vmax=1,
                        origin='lower', aspect='auto')
        stable_ax.set_xticks([])
        stable_ax.set_yticks([])
        for spine in stable_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
        stable_ax.text(0.5, -0.25, 'No dominant\nspatial mode',
                      transform=stable_ax.transAxes, fontsize=self.secondary_label_fontsize,
                      rotation=0, va='top', ha='center')
        
        # Overall title
        title = f'Stability Landscapes: {mode.replace("_", " ").title()} Mode'
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight='bold', y=0.96)
        
        return fig
    
    def create_gain_map_figure(
        self,
        results: Dict,
        param_pairs: List[Tuple[str, str]],
        stages: List[str],
        mode: str = 'fixed_absolute'
    ) -> plt.Figure:
        """Create multi-row, multi-stage 2D gain map figure.
        
        Args:
            results: Results dict organized as {param_pair: {stage: stage_results}}
            param_pairs: List of tuples (param_x_key, param_y_key)
            stages: List of stage names
            mode: Range mode for axis emphasis
            
        Returns:
            matplotlib Figure object
        """
        n_rows = len(param_pairs)
        n_stages = len(stages)
        
        # Determine spine widths
        emphasize_ratio = mode == 'fixed_ratio'
        emphasize_absolute = mode == 'fixed_absolute'
        primary_width = self.bold_spine_width if emphasize_ratio else self.default_spine_width
        secondary_width = self.bold_spine_width if emphasize_absolute else self.default_spine_width
        
        # Create figure with n_rows×n_stages grid
        fig_height = self.fig_height_per_row * n_rows
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(n_rows, n_stages, figure=fig,
                     hspace=self.hspace, wspace=self.wspace,
                     left=self.left_margin, right=0.88,
                     top=0.92, bottom=0.08)
        
        # Determine global k range across all rows
        all_k_values = []
        for param_pair in param_pairs:
            if param_pair in results:
                for stage in stages:
                    if stage in results[param_pair]:
                        all_k_values.append(results[param_pair][stage]['k_matrix'])
        
        if all_k_values:
            k_min = 0
            k_max = min(np.max([np.max(k) for k in all_k_values]), 5.0)
        else:
            k_min, k_max = 0, 5.0
        
        norm = colors.Normalize(vmin=k_min, vmax=k_max)
        cmap = plt.colormaps[GAIN_COLORMAP]
        
        # Plot each row (parameter pair) and column (stage)
        for row_idx, param_pair in enumerate(param_pairs):
            param_x_key, param_y_key = param_pair
            param_x_spec = SCANNABLE_PARAMETERS[param_x_key]
            param_y_spec = SCANNABLE_PARAMETERS[param_y_key]
            
            if param_pair not in results:
                continue
            
            pair_results = results[param_pair]
            
            # Determine consistent axis limits across all stages in this row
            x_limits_all = []
            y_limits_all = []
            for stage in stages:
                if stage in pair_results:
                    x_vals = pair_results[stage]['param_x_values']
                    y_vals = pair_results[stage]['param_y_values']
                    x_limits_all.extend([x_vals[0], x_vals[-1]])
                    y_limits_all.extend([y_vals[0], y_vals[-1]])
            
            if x_limits_all and y_limits_all:
                x_lim = (min(x_limits_all), max(x_limits_all))
                y_lim = (min(y_limits_all), max(y_limits_all))
            else:
                x_lim = (0, 1)
                y_lim = (0, 1)
            
            for stage_idx, stage_name in enumerate(stages):
                if stage_name not in pair_results:
                    continue
                
                stage_result = pair_results[stage_name]
                preset = stage_result['preset']
                x_values = stage_result['param_x_values']
                y_values = stage_result['param_y_values']
                k_matrix = stage_result['k_matrix']
                gain_matrix = stage_result['gain_matrix']
                flatness_matrix = stage_result.get('flatness_matrix', None)
                
                ax = fig.add_subplot(gs[row_idx, stage_idx])
            
            # Transpose matrices
            k_matrix_T = k_matrix.T
            gain_matrix_T = gain_matrix.T
            flatness_matrix_T = flatness_matrix.T if flatness_matrix is not None else None
            
            # Compute alpha values based on gain (log-scale)
            gain_valid = np.where(np.isnan(gain_matrix_T), 1.0, gain_matrix_T)
            gain_clipped = np.clip(gain_valid, 1.0, GAIN_CLIP_MAX)
            log_gain = np.log10(gain_clipped)
            
            log_min = np.log10(1.0)
            log_max = np.log10(GAIN_CLIP_MAX)
            normalized = (log_gain - log_min) / (log_max - log_min)
            normalized = np.clip(normalized, 0, 1)
            
            alpha_matrix = GAIN_OPACITY_MIN + normalized * (GAIN_OPACITY_MAX - GAIN_OPACITY_MIN)
            alpha_matrix = np.where(np.isnan(gain_matrix_T), 0.1, alpha_matrix)
            
            # Create RGBA image
            rgba_image = np.zeros((*k_matrix_T.shape, 4))
            grey_color = (0.5, 0.5, 0.5)
            for i in range(k_matrix_T.shape[0]):
                for j in range(k_matrix_T.shape[1]):
                    if flatness_matrix_T is not None and flatness_matrix_T[i, j]:
                        color_rgb = grey_color
                    else:
                        color_rgb = cmap(norm(k_matrix_T[i, j]))[:3]
                    rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])
            
                # Display image
                extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
                ax.imshow(rgba_image, origin='lower', extent=extent, interpolation='nearest')
                
                # Mark preset point
                preset_x = stage_result['preset_x_value']
                preset_y = stage_result['preset_y_value']
                ax.scatter(preset_x, preset_y,
                          marker='o', s=120, edgecolor='black', linewidth=1.5,
                          facecolor='white', zorder=10)
                
                # Apply consistent axis limits for this row
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_aspect('auto')
                ax.locator_params(axis='x', nbins=4)
                ax.locator_params(axis='y', nbins=5)
                
                # Styling (same as stability maps)
                ax.spines['bottom'].set_linewidth(primary_width)
                ax.spines['left'].set_linewidth(primary_width)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                if stage_idx == 0:
                    ax.set_ylabel(param_y_spec.get_axis_label(absolute=False),
                                fontsize=self.label_fontsize, labelpad=8)
                else:
                    ax.set_ylabel('')
                
                # Only show x-labels on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel(param_x_spec.get_axis_label(absolute=False),
                                 fontsize=self.label_fontsize, labelpad=8)
                else:
                    ax.set_xlabel('')
                
                ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.5)
                
                # Secondary axes
                if param_y_spec.use_ratio and param_y_spec.reference_param:
                    ref_spec_y = SCANNABLE_PARAMETERS[param_y_spec.reference_param]
                    ref_value_y = get_nested_value(preset, ref_spec_y.path)
                else:
                    ref_value_y = 1.0
                
                ax2 = ax.secondary_yaxis('right', functions=(
                    lambda x, rv=ref_value_y: x * rv,
                    lambda x, rv=ref_value_y: x / rv
                ))
                if stage_idx == n_stages - 1:
                    ax2.set_ylabel(param_y_spec.get_axis_label(absolute=True),
                                 fontsize=self.secondary_label_fontsize, labelpad=8)
                    ax2.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
                else:
                    ax2.set_ylabel('')
                    ax2.tick_params(labelright=False, length=0)
                ax2.spines['right'].set_linewidth(secondary_width)
                for spine_name in ['left', 'top', 'bottom']:
                    ax2.spines[spine_name].set_visible(False)
                
                # Top axis - only show on top row
                if param_x_spec.use_ratio and param_x_spec.reference_param:
                    ref_spec_x = SCANNABLE_PARAMETERS[param_x_spec.reference_param]
                    ref_value_x = get_nested_value(preset, ref_spec_x.path)
                else:
                    ref_value_x = 1.0
                
                ax3 = ax.secondary_xaxis('top', functions=(
                    lambda x, rv=ref_value_x: x * rv,
                    lambda x, rv=ref_value_x: x / rv
                ))
                if row_idx == 0:
                    ax3.set_xlabel(param_x_spec.get_axis_label(absolute=True),
                                  fontsize=self.secondary_label_fontsize, labelpad=8)
                    ax3.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
                else:
                    ax3.set_xlabel('')
                    ax3.tick_params(labeltop=False, length=0)
                ax3.spines['top'].set_linewidth(secondary_width)
                for spine_name in ['bottom', 'left', 'right']:
                    ax3.spines[spine_name].set_visible(False)
                
                # Stage label - only on top row
                if row_idx == 0:
                    ax.set_title(stage_name, fontsize=self.subtitle_fontsize,
                                fontweight='bold', pad=21)
        
        # Add shared colorbar spanning all rows
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Position colorbar to span full height of plots
        cbar_bottom = 0.08 + 0.05
        cbar_top = 0.92 - 0.05
        cbar_height = cbar_top - cbar_bottom
        cbar_ax = fig.add_axes([0.905, cbar_bottom, 0.015, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Spatial freq. w/ max gain',
                      fontsize=self.label_fontsize, labelpad=18)
        cbar.ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.6)
        
        # Add grey legend at bottom of colorbar
        grey_legend_bottom = 0.08 + 0.01
        stable_ax = fig.add_axes([0.905, grey_legend_bottom, 0.015, 0.03])
        stable_ax.imshow(np.full((10, 1), 0.5), cmap='Greys', vmin=0, vmax=1,
                        origin='lower', aspect='auto')
        stable_ax.set_xticks([])
        stable_ax.set_yticks([])
        for spine in stable_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
        stable_ax.text(0.5, -0.25, 'Flat gain\nspectrum',
                      transform=stable_ax.transAxes, fontsize=self.secondary_label_fontsize,
                      rotation=0, va='top', ha='center')
        
        # Overall title
        title = f'Gain Landscapes: {mode.replace("_", " ").title()} Mode'
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight='bold', y=0.96)
        
        return fig
    
    def create_gain_spectrum_figure(
        self,
        results: Dict,
        param_keys: List[str],
        stages: List[str]
    ) -> plt.Figure:
        """Create multi-row, multi-stage 1D gain spectrum heatmaps.
        
        Args:
            results: Results dict organized as {param_key: {stage: stage_results}}
            param_keys: List of parameter keys being swept
            stages: List of stage names
            
        Returns:
            matplotlib Figure object
        """
        n_rows = len(param_keys)
        n_stages = len(stages)
        
        # Create figure with n_rows×(n_stages+1) grid (extra column for colorbar)
        fig_width = 3.5 * n_stages
        fig_height = 3.5 * n_rows
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(n_rows, n_stages + 1, figure=fig,
                     height_ratios=[1] * n_rows,
                     width_ratios=[1] * n_stages + [0.05],
                     hspace=0.3, wspace=0.3)
        
        # Compute global colormap normalization across all rows
        all_gains = []
        for param_key in param_keys:
            if param_key in results:
                for stage_name in stages:
                    if stage_name in results[param_key]:
                        gain_matrix = results[param_key][stage_name]['gain_matrix']
                        valid_gains = gain_matrix[~np.isnan(gain_matrix)]
                        if len(valid_gains) > 0:
                            all_gains.extend(valid_gains)
        
        if len(all_gains) > 0:
            if SPECTRUM_LOG_SCALE:
                gains_array = np.array(all_gains)
                gains_positive = gains_array[gains_array > 0]
                if len(gains_positive) > 0:
                    vmin = np.log10(np.percentile(gains_positive, 5))
                    vmax = np.log10(np.percentile(gains_positive, 95))
                else:
                    vmin, vmax = 0, 1
            else:
                vmin = np.percentile(all_gains, 5)
                vmax = np.percentile(all_gains, 95)
        else:
            vmin, vmax = 0, 1
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps[SPECTRUM_COLORMAP]
        
        # Plot each row (parameter) and column (stage)
        for row_idx, param_key in enumerate(param_keys):
            param_spec = SCANNABLE_PARAMETERS[param_key]
            
            if param_key not in results:
                continue
            
            param_results = results[param_key]
            
            # Determine consistent axis limits across all stages in this row
            all_param_values = []
            all_k_values = []
            for stage in stages:
                if stage in param_results:
                    all_param_values.extend([
                        param_results[stage]['param_values'][0],
                        param_results[stage]['param_values'][-1]
                    ])
                    all_k_values.extend([
                        param_results[stage]['k_values'][0],
                        param_results[stage]['k_values'][-1]
                    ])
            
            if all_param_values and all_k_values:
                param_lim = (min(all_param_values), max(all_param_values))
                k_lim = (min(all_k_values), max(all_k_values))
            else:
                param_lim = (0, 1)
                k_lim = (0, 10)
            
            for stage_idx, stage_name in enumerate(stages):
                if stage_name not in param_results:
                    continue
                
                stage_result = param_results[stage_name]
                k_values = stage_result['k_values']
                param_values = stage_result['param_values']
                gain_matrix = stage_result['gain_matrix']
                preset_value = stage_result['preset_value']
                
                ax = fig.add_subplot(gs[row_idx, stage_idx])
                
                # Apply log scale if enabled
                if SPECTRUM_LOG_SCALE:
                    plot_data = np.log10(gain_matrix, where=(gain_matrix > 0))
                    plot_data[gain_matrix <= 0] = np.nan
                else:
                    plot_data = gain_matrix
                
                # Create heatmap
                extent = [k_values[0], k_values[-1], param_values[0], param_values[-1]]
                im = ax.imshow(plot_data, origin='lower', extent=extent,
                              aspect='auto', cmap=cmap, norm=norm,
                              interpolation='nearest')
                
                # Apply consistent axis limits for this row
                ax.set_xlim(k_lim)
                ax.set_ylim(param_lim)
                
                # Mark preset parameter value
                ax.axhline(preset_value, color='white', linestyle='--',
                          linewidth=1.5, alpha=0.8)
                
                # Labels
                # Only show x-labels on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Spatial Frequency (1/grid unit)',
                                 fontsize=self.secondary_label_fontsize)
                else:
                    ax.set_xlabel('')
                
                if stage_idx == 0:
                    ax.set_ylabel(param_spec.get_axis_label(absolute=True),
                                fontsize=self.label_fontsize)
                else:
                    ax.set_ylabel('')
                
                # Only show stage titles on top row
                if row_idx == 0:
                    ax.set_title(stage_name, fontsize=self.subtitle_fontsize, fontweight='bold')
                
                ax.tick_params(labelsize=self.tick_fontsize)
        
        # Add shared colorbar spanning all rows
        cbar_ax = fig.add_subplot(gs[:, n_stages])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        if SPECTRUM_LOG_SCALE:
            cbar.set_label('log₁₀(Gain)', fontsize=self.secondary_label_fontsize)
        else:
            cbar.set_label('Gain', fontsize=self.secondary_label_fontsize)
        cbar.ax.tick_params(labelsize=self.secondary_tick_fontsize)
        
        # Overall title
        title = 'Gain Spectra: Parameter Sweeps'
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight='bold', y=0.995)
        
        return fig
    
    def generate_all_figures(self, results: Dict, mode: str = 'fixed_absolute') -> None:
        """Generate all figures from results.
        
        Args:
            results: Dictionary with 'stability', 'gain_maps', and/or 'gain_spectra' keys
            mode: Range mode for determining axis emphasis
        """
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stages = ['P4', 'P8', 'P12', 'P16']
        
        # Generate stability map figures (one figure with all parameter pairs)
        if 'stability' in results:
            print("\nGenerating stability maps figure...")
            stability_results = results['stability']
            param_pairs = list(stability_results.keys())
            
            if param_pairs:
                fig = self.create_stability_map_figure(
                    stability_results, param_pairs, stages, mode
                )
                
                filename = f'stability_maps_{mode}.svg'
                filepath = output_dir / filename
                fig.savefig(filepath, format='svg', bbox_inches='tight')
                print(f"  Saved: {filepath}")
                plt.close(fig)
        
        # Generate gain map figures (one figure with all parameter pairs)
        if 'gain_maps' in results:
            print("\nGenerating gain maps figure...")
            gain_map_results = results['gain_maps']
            param_pairs = list(gain_map_results.keys())
            
            if param_pairs:
                fig = self.create_gain_map_figure(
                    gain_map_results, param_pairs, stages, mode
                )
                
                filename = f'gain_maps_{mode}.svg'
                filepath = output_dir / filename
                fig.savefig(filepath, format='svg', bbox_inches='tight')
                print(f"  Saved: {filepath}")
                plt.close(fig)
        
        # Generate gain spectrum figures (one figure with all parameters)
        if 'gain_spectra' in results:
            print("\nGenerating gain spectra figure...")
            spectrum_results = results['gain_spectra']
            param_keys = list(spectrum_results.keys())
            
            if param_keys:
                fig = self.create_gain_spectrum_figure(
                    spectrum_results, param_keys, stages
                )
                
                filename = 'gain_spectra.svg'
                filepath = output_dir / filename
                fig.savefig(filepath, format='svg', bbox_inches='tight', dpi=300)
                print(f"  Saved: {filepath}")
                plt.close(fig)
        
        print(f"\nAll figures saved to: {output_dir}")

