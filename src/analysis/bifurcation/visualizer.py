"""Visualization module for bifurcation analysis.

This module handles all figure generation for stability maps, gain maps,
and gain spectra, ensuring consistent styling and layout across all analyses.
"""

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from src.analysis.common import (
    DOUBLE_COLUMN_WIDTH_MM,
    FIGURE_FONT_SIZES_PT,
    DEVELOPMENTAL_STAGES,
    apply_matplotlib_style,
    compute_figsize_inches,
    mm_to_inches,
    save_figure,
)

from .config import (
    ANALYSIS_PARAMS,
    BIFURCATION_COLORMAP,
    GAIN_CLIP_MAX,
    GAIN_COLORMAP,
    GAIN_OPACITY_MAX,
    GAIN_OPACITY_MIN,
    OPACITY_STABLE_FAR,
    OPACITY_STABLE_NEAR,
    OPACITY_UNSTABLE,
    OUTPUT_DIR,
    SCANNABLE_PARAMETERS,
    SPECTRUM_COLORMAP,
    SPECTRUM_LOG_SCALE,
    SPECTRUM_Y_MARGIN,
    STABILITY_THRESHOLD,
)
from .core import get_nested_value


class BifurcationVisualizer:
    """Handles all bifurcation visualization with consistent styling."""

    def __init__(self):
        """Initialize visualizer with default style settings."""
        apply_matplotlib_style()

        # Figure dimensions (mm-based, Nature double-column standard)
        self.fig_width_mm = DOUBLE_COLUMN_WIDTH_MM  # 183 mm
        self.fig_height_per_row_mm = 45.0  # Height per row in mm

        # Convert to inches for Matplotlib
        self.fig_width = mm_to_inches(self.fig_width_mm)
        self.fig_height_per_row = mm_to_inches(self.fig_height_per_row_mm)

        # Spine widths (figure-specific, not in global rcParams)
        self.default_spine_width = 0.8
        # Slightly emphasized, but not visually heavy (used to indicate fixed axis)
        self.bold_spine_width = 1.3

        # Font sizes (from centralized spec)
        self.title_fontsize = FIGURE_FONT_SIZES_PT["figure_title"]
        self.subtitle_fontsize = FIGURE_FONT_SIZES_PT["axes_title"]
        self.label_fontsize = FIGURE_FONT_SIZES_PT["axis_label"]
        self.secondary_label_fontsize = FIGURE_FONT_SIZES_PT["axis_label"]
        self.tick_fontsize = FIGURE_FONT_SIZES_PT["tick_label"]
        self.secondary_tick_fontsize = FIGURE_FONT_SIZES_PT["tick_label"]

        # Layout parameters
        self.hspace = 0.45
        # Slightly more horizontal breathing room between stage panels
        self.wspace = 0.30
        self.left_margin = 0.07
        self.right_margin = 0.83
        self.top_margin = 0.74
        self.bottom_margin = 0.08

    def create_stability_map_figure(
        self,
        results: dict,
        param_pairs: list[tuple[str, str]],
        stages: list[str],
        mode: str = "fixed_absolute",
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

        # Determine spine widths and axis semantics
        # Note: Data is ALWAYS in absolute units (extent uses absolute values)
        # So primary axis ALWAYS shows absolute values
        # Secondary axis ALWAYS shows ratios (via division)
        # Emphasis (bold spine) indicates what's "fixed" across stages
        primary_absolute = True  # Always show absolute on primary
        secondary_absolute = False  # Always show ratios on secondary

        if mode == "fixed_absolute":
            # Emphasize primary (absolute values are what's fixed)
            primary_width = self.bold_spine_width
            secondary_width = self.default_spine_width
        else:  # fixed_ratio
            # Emphasize secondary (ratios are what's fixed)
            primary_width = self.default_spine_width
            secondary_width = self.bold_spine_width

        # Create figure with n_rows×n_stages grid
        # Slightly taller per row than gain maps because we now show one x-label per row
        fig_height = self.fig_height_per_row * n_rows * 1.18
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(
            n_rows,
            n_stages,
            figure=fig,
            # Extra vertical spacing so row-level x-labels don't collide with the next row
            hspace=self.hspace + 0.18,
            wspace=self.wspace,
            left=self.left_margin,
            right=0.83,
            top=0.80,
            bottom=0.08,
        )

        # Determine global wavelength range for consistent colormap across all rows
        # Convert k (cycles/μm) to wavelength λ (μm): λ = 1/k
        all_wavelength_values = []
        for param_pair in param_pairs:
            if param_pair in results:
                for stage in stages:
                    if stage in results[param_pair]:
                        k_matrix = results[param_pair][stage]["k_matrix"]
                        # Convert k to wavelength, avoiding division by zero
                        wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)
                        all_wavelength_values.append(wavelength_matrix)

        if all_wavelength_values:
            # Use finite values only
            finite_wavelengths = [w[np.isfinite(w)] for w in all_wavelength_values]
            if any(len(w) > 0 for w in finite_wavelengths):
                wavelength_min = max(
                    np.min([np.min(w) for w in finite_wavelengths if len(w) > 0]), 50.0
                )
                wavelength_max = min(
                    np.max([np.max(w) for w in finite_wavelengths if len(w) > 0]), 2000.0
                )
            else:
                wavelength_min, wavelength_max = 50.0, 2000.0
        else:
            wavelength_min, wavelength_max = 50.0, 2000.0

        norm = colors.Normalize(vmin=wavelength_min, vmax=wavelength_max)
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
            # In fixed_absolute mode: all stages have same ranges (use any stage)
            # In fixed_ratio mode: ranges differ by stage (use intersection to avoid clipping)
            x_mins, x_maxs = [], []
            y_mins, y_maxs = [], []
            for stage in stages:
                if stage in pair_results:
                    x_vals = pair_results[stage]["param_x_values"]
                    y_vals = pair_results[stage]["param_y_values"]
                    x_mins.append(x_vals[0])
                    x_maxs.append(x_vals[-1])
                    y_mins.append(y_vals[0])
                    y_maxs.append(y_vals[-1])

            if x_mins and y_mins:
                # Use intersection: range covered by ALL stages (max of mins, min of maxs)
                # This ensures all stages have data for the entire displayed range
                x_lim = (max(x_mins), min(x_maxs))
                y_lim = (max(y_mins), min(y_maxs))
            else:
                x_lim = (0, 1)
                y_lim = (0, 1)

            for stage_idx, stage_name in enumerate(stages):
                if stage_name not in pair_results:
                    continue

                stage_result = pair_results[stage_name]
                preset = stage_result["preset"]
                x_values = stage_result["param_x_values"]
                y_values = stage_result["param_y_values"]
                k_matrix = stage_result["k_matrix"]
                stability_matrix = stage_result["stability_matrix"]
                flatness_matrix = stage_result.get("flatness_matrix", None)

                ax = fig.add_subplot(gs[row_idx, stage_idx])

                # Convert k to wavelength: λ = 1/k (μm)
                wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)

                # Identify NaN values (failed steady-state convergence)
                nan_mask = np.isnan(k_matrix) | np.isnan(stability_matrix)

                # Compute alpha values based on stability
                alpha_matrix = np.zeros_like(stability_matrix)
                alpha_matrix[stability_matrix < STABILITY_THRESHOLD] = OPACITY_STABLE_FAR
                alpha_matrix[
                    (stability_matrix >= STABILITY_THRESHOLD) & (stability_matrix < 0)
                ] = OPACITY_STABLE_NEAR
                alpha_matrix[stability_matrix >= 0] = OPACITY_UNSTABLE

                # Create RGBA image (grey for flat spectra, colormap otherwise, white for NaN)
                rgba_image = np.zeros((*wavelength_matrix.shape, 4))
                grey_color = (0.5, 0.5, 0.5)
                white_color = (1.0, 1.0, 1.0)
                for i in range(wavelength_matrix.shape[0]):
                    for j in range(wavelength_matrix.shape[1]):
                        if nan_mask[i, j]:
                            # NaN values render as white (failed convergence)
                            rgba_image[i, j] = (*white_color, 1.0)
                        elif flatness_matrix is not None and flatness_matrix[i, j]:
                            color_rgb = grey_color
                            rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])
                        else:
                            color_rgb = cmap(norm(wavelength_matrix[i, j]))[:3]
                            rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])

                # Display image
                extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
                ax.imshow(rgba_image, origin="lower", extent=extent, interpolation="nearest")

                # Add stability boundary contour
                with contextlib.suppress(ValueError, RuntimeError):
                    ax.contour(
                        x_values,
                        y_values,
                        stability_matrix,
                        levels=[0],
                        colors="white",
                        linewidths=1.0,
                        linestyles="--",
                        alpha=0.8,
                    )

                # Mark preset point
                preset_x = stage_result["preset_x_value"]
                preset_y = stage_result["preset_y_value"]
                ax.scatter(
                    preset_x,
                    preset_y,
                    marker="o",
                    s=30,
                    edgecolor="black",
                    linewidth=1.2,
                    facecolor="white",
                    zorder=10,
                )

                # Apply consistent axis limits for this row
                if mode == "fixed_ratio":
                    # In fixed_ratio mode, each stage uses a different *absolute* scan range.
                    # Using a cross-stage intersection in absolute units can collapse the visible
                    # region to a sliver and make plots appear blank.
                    ax.set_xlim(x_values[0], x_values[-1])
                    ax.set_ylim(y_values[0], y_values[-1])
                else:
                    ax.set_xlim(x_lim)
                    ax.set_ylim(y_lim)
                ax.set_aspect("auto")
                ax.locator_params(axis="x", nbins=4)
                ax.locator_params(axis="y", nbins=5)

                # Primary axes spines
                ax.spines["bottom"].set_linewidth(primary_width)
                ax.spines["left"].set_linewidth(primary_width)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Axis labels (primary axes)
                # Make labels grey if not dominant, black if dominant
                primary_label_color = "black" if primary_width == self.bold_spine_width else "0.5"
                # Show y-labels on all leftmost axes
                if stage_idx == 0:
                    ax.set_ylabel(
                        param_y_spec.get_axis_label(absolute=primary_absolute),
                        fontsize=self.label_fontsize,
                        labelpad=6,
                        color=primary_label_color,
                    )
                else:
                    ax.set_ylabel("")

                # Show x-label once per row (leftmost panel) since each row can use a different x-param
                if stage_idx == 0:
                    ax.set_xlabel(
                        param_x_spec.get_axis_label(absolute=primary_absolute),
                        fontsize=self.label_fontsize,
                        labelpad=6,
                        color=primary_label_color,
                    )
                else:
                    ax.set_xlabel("")

                # Set tick styling - make dominant axis ticks/labels black
                if primary_width == self.bold_spine_width:
                    # Primary axis is dominant - make ticks and labels black
                    ax.tick_params(
                        labelsize=self.tick_fontsize,
                        length=3,
                        width=0.5,
                        color="black",
                        labelcolor="black",
                    )
                else:
                    # Primary axis is not dominant - use default grey
                    ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.5)

                # Secondary axes - always convert absolute to ratio
                # Get reference values for ratio conversion
                if param_y_spec.use_ratio and param_y_spec.reference_param:
                    ref_spec_y = SCANNABLE_PARAMETERS[param_y_spec.reference_param]
                    ref_value_y = get_nested_value(preset, ref_spec_y.path)
                else:
                    ref_value_y = 1.0

                # Always convert: absolute (primary) → ratio (secondary)
                # Forward: absolute → ratio (divide), Inverse: ratio → absolute (multiply)
                def conv_y_fwd(x, rv=ref_value_y):
                    return x / rv if rv != 0 else x

                def conv_y_inv(x, rv=ref_value_y):
                    return x * rv

                ax2 = ax.secondary_yaxis("right", functions=(conv_y_fwd, conv_y_inv))
                # Reduce number of ticks on secondary y-axis to allow 1 d.p. formatting
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
                # Make labels grey if not dominant, black if dominant
                secondary_label_color = "black" if secondary_width == self.bold_spine_width else "0.5"
                # Show y-labels on all rightmost axes
                if stage_idx == n_stages - 1:
                    ax2.set_ylabel(
                        param_y_spec.get_axis_label(absolute=secondary_absolute),
                        fontsize=self.secondary_label_fontsize,
                        labelpad=10,
                        color=secondary_label_color,
                    )
                    # Make ticks/labels black if secondary axis is dominant
                    if secondary_width == self.bold_spine_width:
                        ax2.tick_params(
                            labelsize=self.secondary_tick_fontsize,
                            length=2,
                            width=0.5,
                            color="black",
                            labelcolor="black",
                        )
                    else:
                        ax2.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
                else:
                    # Keep secondary axis minimal on interior panels: no tick labels without an axis label
                    ax2.set_ylabel("")
                    ax2.tick_params(
                        labelright=False,
                        right=False,
                        length=0,
                    )
                ax2.spines["right"].set_linewidth(secondary_width)
                for spine_name in ["left", "top", "bottom"]:
                    ax2.spines[spine_name].set_visible(False)

                # Top axis - show on all top row axes
                if param_x_spec.use_ratio and param_x_spec.reference_param:
                    ref_spec_x = SCANNABLE_PARAMETERS[param_x_spec.reference_param]
                    ref_value_x = get_nested_value(preset, ref_spec_x.path)
                else:
                    ref_value_x = 1.0

                # Always convert: absolute (primary) → ratio (secondary)
                def conv_x_fwd(x, rv=ref_value_x):
                    return x / rv if rv != 0 else x

                def conv_x_inv(x, rv=ref_value_x):
                    return x * rv

                ax3 = ax.secondary_xaxis("top", functions=(conv_x_fwd, conv_x_inv))
                if row_idx == 0:
                    ax3.set_xlabel(
                        param_x_spec.get_axis_label(absolute=secondary_absolute),
                        fontsize=self.secondary_label_fontsize,
                        labelpad=6,
                        color=secondary_label_color,
                    )
                    # Make ticks/labels black if secondary axis is dominant
                    if secondary_width == self.bold_spine_width:
                        ax3.tick_params(
                            labelsize=self.secondary_tick_fontsize,
                            length=2,
                            width=0.5,
                            color="black",
                            labelcolor="black",
                        )
                    else:
                        ax3.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
                else:
                    ax3.set_xlabel("")
                    ax3.tick_params(labeltop=False, length=0)
                ax3.spines["top"].set_linewidth(secondary_width)
                for spine_name in ["bottom", "left", "right"]:
                    ax3.spines[spine_name].set_visible(False)

                # Stage label - only on top row
                if row_idx == 0:
                    ax.set_title(
                        stage_name, fontsize=self.subtitle_fontsize, fontweight="bold", pad=10
                    )

        # Add 2D opacity legend (replacing colorbar)
        # Position to span full height of plots (moved right to avoid overlap)
        # Start above the grey legend box (which is at 0.09-0.12 + text below)
        cbar_bottom = 0.14
        cbar_top = 0.80
        cbar_height = cbar_top - cbar_bottom
        cbar_width = 0.05  # Wider to accommodate 3 columns and rotated labels
        cbar_ax = fig.add_axes([0.95, cbar_bottom, cbar_width, cbar_height])

        # Create 2D opacity legend: 3 columns for the three opacity levels
        # Each column shows the full viridis colormap, with only opacity varying
        n_ticks = 100  # High resolution for smooth colormap gradient
        opacity_legend = np.zeros((n_ticks, 3, 4))  # RGBA array

        # Fill each row with colors from the full colormap (varying by row)
        # Each column has the same color but different opacity
        wavelength_values_for_legend = np.linspace(wavelength_min, wavelength_max, n_ticks)
        for i, wavelength_val in enumerate(wavelength_values_for_legend):
            color_rgb = cmap(norm(wavelength_val))[:3]
            # Left column: stable far (0.3), middle: stable near (0.6), right: unstable (1.0)
            opacity_legend[i, 0, :3] = color_rgb
            opacity_legend[i, 0, 3] = OPACITY_STABLE_FAR
            opacity_legend[i, 1, :3] = color_rgb
            opacity_legend[i, 1, 3] = OPACITY_STABLE_NEAR
            opacity_legend[i, 2, :3] = color_rgb
            opacity_legend[i, 2, 3] = OPACITY_UNSTABLE

        # Display the 2D legend
        cbar_ax.imshow(opacity_legend, origin="lower", aspect="auto", interpolation="nearest")

        # Set up axes
        cbar_ax.set_xlim(-0.5, 2.5)
        cbar_ax.set_ylim(0, n_ticks - 1)

        # X-axis labels for the three lambda ranges (on TOP)
        cbar_ax.set_xticks([0, 1, 2])
        cbar_ax.set_xticklabels(
            [
                "Stable (far)",
                "Stable (near)",
                "Unstable",
            ],
            fontsize=self.secondary_tick_fontsize,
            rotation=45,
            ha="left",
        )
        cbar_ax.xaxis.set_label_position("top")
        cbar_ax.xaxis.tick_top()
        cbar_ax.tick_params(
            axis="x", bottom=False, top=True, labelsize=self.secondary_tick_fontsize
        )

        # Y-axis: show wavelength values (μm) with ticks and label on RIGHT
        wavelength_tick_positions = np.linspace(0, n_ticks - 1, 6)  # 6 ticks
        wavelength_tick_values = np.linspace(wavelength_min, wavelength_max, 6)
        cbar_ax.set_yticks(wavelength_tick_positions)
        cbar_ax.set_yticklabels(
            [f"{w:.0f}" for w in wavelength_tick_values], fontsize=self.tick_fontsize
        )
        cbar_ax.set_ylabel(
            r"Wavelength w/ max Re($\lambda$) ($\mu$m)", fontsize=self.label_fontsize, labelpad=15
        )
        cbar_ax.yaxis.set_label_position("right")
        cbar_ax.yaxis.tick_right()
        cbar_ax.tick_params(axis="y", left=False, right=True, labelsize=self.tick_fontsize)

        # Style the axes - black border all around
        cbar_ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.6)
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)

        # Add grey legend at bottom of colorbar (aligned with 2D legend, full width)
        grey_legend_bottom = 0.08 + 0.01
        stable_ax = fig.add_axes([0.95, grey_legend_bottom, cbar_width, 0.03])
        stable_ax.imshow(
            np.full((10, 1), 0.5), cmap="Greys", vmin=0, vmax=1, origin="lower", aspect="auto"
        )
        stable_ax.set_xticks([])
        stable_ax.set_yticks([])
        for spine in stable_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
        stable_ax.text(
            0.5,
            -0.25,
            "No dominant\nspatial mode",
            transform=stable_ax.transAxes,
            fontsize=self.secondary_label_fontsize,
            rotation=0,
            va="top",
            ha="center",
        )

        # Overall title (positioned above stage titles)
        title = f'Stability Landscapes: {mode.replace("_", " ").title()} Values'
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight="bold", y=1.03)

        return fig

    def create_gain_map_figure(
        self,
        results: dict,
        param_pairs: list[tuple[str, str]],
        stages: list[str],
        mode: str = "fixed_absolute",
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

        # Determine spine widths and axis semantics
        # Note: Data is ALWAYS in absolute units (extent uses absolute values)
        # So primary axis ALWAYS shows absolute values
        # Secondary axis ALWAYS shows ratios (via division)
        # Emphasis (bold spine) indicates what's "fixed" across stages
        primary_absolute = True  # Always show absolute on primary
        secondary_absolute = False  # Always show ratios on secondary

        if mode == "fixed_absolute":
            # Emphasize primary (absolute values are what's fixed)
            primary_width = self.bold_spine_width
            secondary_width = self.default_spine_width
        else:  # fixed_ratio
            # Emphasize secondary (ratios are what's fixed)
            primary_width = self.default_spine_width
            secondary_width = self.bold_spine_width

        # Create figure with n_rows×n_stages grid
        fig_height = self.fig_height_per_row * n_rows
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(
            n_rows,
            n_stages,
            figure=fig,
            hspace=self.hspace,
            wspace=self.wspace,
            left=self.left_margin,
            right=0.83,
            top=0.80,
            bottom=0.08,
        )

        # Determine global wavelength range across all rows
        # Convert k (cycles/μm) to wavelength λ (μm): λ = 1/k
        all_wavelength_values = []
        for param_pair in param_pairs:
            if param_pair in results:
                for stage in stages:
                    if stage in results[param_pair]:
                        k_matrix = results[param_pair][stage]["k_matrix"]
                        # Convert k to wavelength, avoiding division by zero
                        wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)
                        all_wavelength_values.append(wavelength_matrix)

        if all_wavelength_values:
            # Use finite values only
            finite_wavelengths = [w[np.isfinite(w)] for w in all_wavelength_values]
            if any(len(w) > 0 for w in finite_wavelengths):
                wavelength_min = max(
                    np.min([np.min(w) for w in finite_wavelengths if len(w) > 0]), 50.0
                )
                wavelength_max = min(
                    np.max([np.max(w) for w in finite_wavelengths if len(w) > 0]), 2000.0
                )
            else:
                wavelength_min, wavelength_max = 50.0, 2000.0
        else:
            wavelength_min, wavelength_max = 50.0, 2000.0

        norm = colors.Normalize(vmin=wavelength_min, vmax=wavelength_max)
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
            # In fixed_absolute mode: all stages have same ranges (use any stage)
            # In fixed_ratio mode: ranges differ by stage (use intersection to avoid clipping)
            x_mins, x_maxs = [], []
            y_mins, y_maxs = [], []
            for stage in stages:
                if stage in pair_results:
                    x_vals = pair_results[stage]["param_x_values"]
                    y_vals = pair_results[stage]["param_y_values"]
                    x_mins.append(x_vals[0])
                    x_maxs.append(x_vals[-1])
                    y_mins.append(y_vals[0])
                    y_maxs.append(y_vals[-1])

            if x_mins and y_mins:
                # Use intersection: range covered by ALL stages (max of mins, min of maxs)
                # This ensures all stages have data for the entire displayed range
                x_lim = (max(x_mins), min(x_maxs))
                y_lim = (max(y_mins), min(y_maxs))
            else:
                x_lim = (0, 1)
                y_lim = (0, 1)

            for stage_idx, stage_name in enumerate(stages):
                if stage_name not in pair_results:
                    continue

                stage_result = pair_results[stage_name]
                preset = stage_result["preset"]
                x_values = stage_result["param_x_values"]
                y_values = stage_result["param_y_values"]
                k_matrix = stage_result["k_matrix"]
                gain_matrix = stage_result["gain_matrix"]
                flatness_matrix = stage_result.get("flatness_matrix", None)

                ax = fig.add_subplot(gs[row_idx, stage_idx])

                # Convert k to wavelength: λ = 1/k (μm)
                wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)

                # Identify NaN values (failed steady-state convergence)
                nan_mask = np.isnan(k_matrix) | np.isnan(gain_matrix)

                # Compute alpha values based on gain (log-scale)
                gain_valid = np.where(np.isnan(gain_matrix), 1.0, gain_matrix)
                gain_clipped = np.clip(gain_valid, 1.0, GAIN_CLIP_MAX)
                log_gain = np.log10(gain_clipped)

                log_min = np.log10(1.0)
                log_max = np.log10(GAIN_CLIP_MAX)
                normalized = (log_gain - log_min) / (log_max - log_min)
                normalized = np.clip(normalized, 0, 1)

                alpha_matrix = GAIN_OPACITY_MIN + normalized * (GAIN_OPACITY_MAX - GAIN_OPACITY_MIN)

                # Create RGBA image (white for NaN values)
                rgba_image = np.zeros((*wavelength_matrix.shape, 4))
                grey_color = (0.5, 0.5, 0.5)
                white_color = (1.0, 1.0, 1.0)
                for i in range(wavelength_matrix.shape[0]):
                    for j in range(wavelength_matrix.shape[1]):
                        if nan_mask[i, j]:
                            # NaN values render as white (failed convergence)
                            rgba_image[i, j] = (*white_color, 1.0)
                        elif flatness_matrix is not None and flatness_matrix[i, j]:
                            color_rgb = grey_color
                            rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])
                        else:
                            color_rgb = cmap(norm(wavelength_matrix[i, j]))[:3]
                            rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])

                # Display image
                extent = [x_values[0], x_values[-1], y_values[0], y_values[-1]]
                ax.imshow(rgba_image, origin="lower", extent=extent, interpolation="nearest")

                # Mark preset point
                preset_x = stage_result["preset_x_value"]
                preset_y = stage_result["preset_y_value"]
                ax.scatter(
                    preset_x,
                    preset_y,
                    marker="o",
                    s=30,
                    edgecolor="black",
                    linewidth=1.2,
                    facecolor="white",
                    zorder=10,
                )

                # Apply consistent axis limits for this row
                if mode == "fixed_ratio":
                    ax.set_xlim(x_values[0], x_values[-1])
                    ax.set_ylim(y_values[0], y_values[-1])
                else:
                    ax.set_xlim(x_lim)
                    ax.set_ylim(y_lim)
                ax.set_aspect("auto")
                ax.locator_params(axis="x", nbins=4)
                ax.locator_params(axis="y", nbins=5)

                # Styling (same as stability maps)
                ax.spines["bottom"].set_linewidth(primary_width)
                ax.spines["left"].set_linewidth(primary_width)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Make labels grey if not dominant, black if dominant
                primary_label_color = "black" if primary_width == self.bold_spine_width else "0.5"
                if stage_idx == 0:
                    ax.set_ylabel(
                        param_y_spec.get_axis_label(absolute=primary_absolute),
                        fontsize=self.label_fontsize,
                        labelpad=6,
                        color=primary_label_color,
                    )
                else:
                    ax.set_ylabel("")

                # Show x-labels on bottom row and top row (tau_SST)
                if row_idx == n_rows - 1 or row_idx == 0:
                    ax.set_xlabel(
                        param_x_spec.get_axis_label(absolute=primary_absolute),
                        fontsize=self.label_fontsize,
                        labelpad=6,
                        color=primary_label_color,
                    )
                else:
                    ax.set_xlabel("")

                # Set tick styling - make dominant axis ticks/labels black
                if primary_width == self.bold_spine_width:
                    # Primary axis is dominant - make ticks and labels black
                    ax.tick_params(
                        labelsize=self.tick_fontsize,
                        length=3,
                        width=0.5,
                        color="black",
                        labelcolor="black",
                    )
                else:
                    # Primary axis is not dominant - use default grey
                    ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.5)

                # Secondary axes - always convert absolute to ratio
                if param_y_spec.use_ratio and param_y_spec.reference_param:
                    ref_spec_y = SCANNABLE_PARAMETERS[param_y_spec.reference_param]
                    ref_value_y = get_nested_value(preset, ref_spec_y.path)
                else:
                    ref_value_y = 1.0

                # Always convert: absolute (primary) → ratio (secondary)
                def conv_y_fwd(x, rv=ref_value_y):
                    return x / rv if rv != 0 else x

                def conv_y_inv(x, rv=ref_value_y):
                    return x * rv

                ax2 = ax.secondary_yaxis("right", functions=(conv_y_fwd, conv_y_inv))
                # Reduce number of ticks on secondary y-axis to allow 1 d.p. formatting
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
                # Make labels grey if not dominant, black if dominant
                secondary_label_color = "black" if secondary_width == self.bold_spine_width else "0.5"
                if stage_idx == n_stages - 1:
                    ax2.set_ylabel(
                        param_y_spec.get_axis_label(absolute=secondary_absolute),
                        fontsize=self.secondary_label_fontsize,
                        labelpad=10,
                        color=secondary_label_color,
                    )
                    # Make ticks/labels black if secondary axis is dominant
                    if secondary_width == self.bold_spine_width:
                        ax2.tick_params(
                            labelsize=self.secondary_tick_fontsize,
                            length=2,
                            width=0.5,
                            color="black",
                            labelcolor="black",
                        )
                    else:
                        ax2.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
                else:
                    ax2.set_ylabel("")
                    ax2.tick_params(labelright=False, length=0)
                ax2.spines["right"].set_linewidth(secondary_width)
                for spine_name in ["left", "top", "bottom"]:
                    ax2.spines[spine_name].set_visible(False)

                # Top axis - only show on top row
                if param_x_spec.use_ratio and param_x_spec.reference_param:
                    ref_spec_x = SCANNABLE_PARAMETERS[param_x_spec.reference_param]
                    ref_value_x = get_nested_value(preset, ref_spec_x.path)
                else:
                    ref_value_x = 1.0

                # Always convert: absolute (primary) → ratio (secondary)
                def conv_x_fwd(x, rv=ref_value_x):
                    return x / rv if rv != 0 else x

                def conv_x_inv(x, rv=ref_value_x):
                    return x * rv

                ax3 = ax.secondary_xaxis("top", functions=(conv_x_fwd, conv_x_inv))
                if row_idx == 0:
                    ax3.set_xlabel(
                        param_x_spec.get_axis_label(absolute=secondary_absolute),
                        fontsize=self.secondary_label_fontsize,
                        labelpad=6,
                        color=secondary_label_color,
                    )
                    # Make ticks/labels black if secondary axis is dominant
                    if secondary_width == self.bold_spine_width:
                        ax3.tick_params(
                            labelsize=self.secondary_tick_fontsize,
                            length=2,
                            width=0.5,
                            color="black",
                            labelcolor="black",
                        )
                    else:
                        ax3.tick_params(labelsize=self.secondary_tick_fontsize, length=2, width=0.5)
                else:
                    ax3.set_xlabel("")
                    ax3.tick_params(labeltop=False, length=0)
                ax3.spines["top"].set_linewidth(secondary_width)
                for spine_name in ["bottom", "left", "right"]:
                    ax3.spines[spine_name].set_visible(False)

                # Stage label - only on top row
                if row_idx == 0:
                    ax.set_title(
                        stage_name, fontsize=self.subtitle_fontsize, fontweight="bold", pad=10
                    )

        # Add shared colorbar spanning all rows
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Position colorbar to span full height of plots (moved right to avoid overlap)
        # Start above the grey legend box (which is at 0.09-0.12 + text below)
        cbar_bottom = 0.14
        cbar_top = 0.80
        cbar_height = cbar_top - cbar_bottom
        cbar_ax = fig.add_axes([0.93, cbar_bottom, 0.015, cbar_height])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar.set_label(
            r"Wavelength w/ max gain ($\mu$m)", fontsize=self.label_fontsize, labelpad=15
        )
        cbar.ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.6)

        # Add grey legend at bottom of colorbar
        grey_legend_bottom = 0.08 + 0.01
        stable_ax = fig.add_axes([0.93, grey_legend_bottom, 0.015, 0.03])
        stable_ax.imshow(
            np.full((10, 1), 0.5), cmap="Greys", vmin=0, vmax=1, origin="lower", aspect="auto"
        )
        stable_ax.set_xticks([])
        stable_ax.set_yticks([])
        for spine in stable_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
        stable_ax.text(
            0.5,
            -0.25,
            "Flat gain\nspectrum",
            transform=stable_ax.transAxes,
            fontsize=self.secondary_label_fontsize,
            rotation=0,
            va="top",
            ha="center",
        )

        # Overall title (positioned above stage titles)
        title = f'Gain Landscapes: {mode.replace("_", " ").title()} Mode'
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight="bold", y=1.03)

        return fig

    def create_gain_spectrum_figure(
        self, results: dict, param_keys: list[str], stages: list[str]
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
        # Use mm-based sizing: width per stage, height per row
        width_per_stage_mm = 40.0  # mm per stage column
        height_per_row_mm = 40.0  # mm per row
        fig_width_mm = width_per_stage_mm * n_stages
        fig_height_mm = height_per_row_mm * n_rows
        fig_width, fig_height = compute_figsize_inches(fig_width_mm, fig_height_mm)
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(
            n_rows,
            n_stages + 1,
            figure=fig,
            height_ratios=[1] * n_rows,
            width_ratios=[1] * n_stages + [0.05],
            hspace=0.3,
            wspace=0.3,
        )

        # Compute global colormap normalization across all rows
        all_gains = []
        for param_key in param_keys:
            if param_key in results:
                for stage_name in stages:
                    if stage_name in results[param_key]:
                        gain_matrix = results[param_key][stage_name]["gain_matrix"]
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

            anatomical_grid_size_um = ANALYSIS_PARAMS["anatomical_grid_size"]

            # Determine consistent wavelength-axis limits across all stages in this row
            # Here, k_values are mode numbers (dimensionless). Physical wavelength is:
            #   λ (μm) = L (μm) / k_mode
            all_wavelength_values = []
            for stage in stages:
                if stage in param_results:
                    k_values = param_results[stage]["k_values"]
                    # Convert k_mode to wavelength (excluding k=0)
                    wavelength_values = np.where(
                        k_values > 0, anatomical_grid_size_um / k_values, np.inf
                    )
                    finite_wavelengths = wavelength_values[np.isfinite(wavelength_values)]
                    if len(finite_wavelengths) > 0:
                        all_wavelength_values.extend(
                            [np.min(finite_wavelengths), np.max(finite_wavelengths)]
                        )

            if all_wavelength_values:
                wavelength_lim = (min(all_wavelength_values), max(all_wavelength_values))
            else:
                wavelength_lim = (50, 2000)

            for stage_idx, stage_name in enumerate(stages):
                if stage_name not in param_results:
                    continue

                stage_result = param_results[stage_name]
                k_values = stage_result["k_values"]
                param_values = stage_result["param_values"]
                gain_matrix = stage_result["gain_matrix"]
                preset_value = stage_result["preset_value"]

                ax = fig.add_subplot(gs[row_idx, stage_idx])

                # Convert k_mode to wavelength: λ (μm) = L (μm) / k_mode
                wavelength_values = np.where(
                    k_values > 0, anatomical_grid_size_um / k_values, np.inf
                )
                # Filter out infinite values for extent
                finite_mask = np.isfinite(wavelength_values)
                if np.sum(finite_mask) > 0:
                    wavelength_extent = [
                        np.min(wavelength_values[finite_mask]),
                        np.max(wavelength_values[finite_mask]),
                    ]
                else:
                    wavelength_extent = [50, 2000]

                # Apply log scale if enabled
                if SPECTRUM_LOG_SCALE:
                    plot_data = np.log10(gain_matrix, where=(gain_matrix > 0))
                    plot_data[gain_matrix <= 0] = np.nan
                else:
                    plot_data = gain_matrix

                # Create heatmap. We display wavelength on the x-axis; larger wavelength
                # (smaller k) should appear on the left, so we reverse the x-axis below.
                extent = [
                    wavelength_extent[0],
                    wavelength_extent[1],
                    param_values[0],
                    param_values[-1],
                ]
                ax.imshow(
                    plot_data,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                )

                # Apply axis limits: wavelength-axis consistent across row, y-axis adaptive per stage
                ax.set_xlim(wavelength_lim[1], wavelength_lim[0])

                # Set y-axis limits based on preset value with fixed margin
                param_lim_min = preset_value * SPECTRUM_Y_MARGIN[0]
                param_lim_max = preset_value * SPECTRUM_Y_MARGIN[1]
                ax.set_ylim(param_lim_min, param_lim_max)

                # Mark preset parameter value
                ax.axhline(preset_value, color="white", linestyle="--", linewidth=1.5, alpha=0.8)

                # Labels
                # Only show x-labels on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=self.secondary_label_fontsize)
                else:
                    ax.set_xlabel("")

                if stage_idx == 0:
                    ax.set_ylabel(
                        param_spec.get_axis_label(absolute=True), fontsize=self.label_fontsize
                    )
                else:
                    ax.set_ylabel("")

                # Only show stage titles on top row
                if row_idx == 0:
                    ax.set_title(stage_name, fontsize=self.subtitle_fontsize, fontweight="bold")

                ax.tick_params(labelsize=self.tick_fontsize)

            # Add colorbar for this row (same limits across all rows)
            cbar_ax = fig.add_subplot(gs[row_idx, n_stages])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            if SPECTRUM_LOG_SCALE:
                cbar.set_label("log₁₀(Gain)", fontsize=self.secondary_label_fontsize)
            else:
                cbar.set_label("Gain", fontsize=self.secondary_label_fontsize)
            cbar.ax.tick_params(labelsize=self.secondary_tick_fontsize)

        # Overall title
        title = "Gain Spectra: Parameter Sweeps"
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight="bold", y=0.995)

        return fig

    def create_compressed_stability_map_figure(
        self,
        results: dict,
        stages: list[str],
    ) -> plt.Figure:
        """Create compressed stability map figure showing SST-PV ratio analysis.

        Args:
            results: Results dict organized as {stage: stage_results}
            stages: List of stage names

        Returns:
            matplotlib Figure object
        """
        n_stages = len(stages)

        # Create figure with single row of n_stages panels
        fig_height = self.fig_height_per_row * 1.18  # Slightly taller for x-labels
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(
            1,
            n_stages,
            figure=fig,
            hspace=0.45,
            wspace=self.wspace,
            left=self.left_margin,
            right=0.83,
            top=0.80,
            bottom=0.08,
        )

        # Determine global wavelength range for consistent colormap across all stages
        all_wavelength_values = []
        for stage in stages:
            if stage in results:
                k_matrix = results[stage]["k_matrix"]
                # Convert k to wavelength, avoiding division by zero
                wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)
                all_wavelength_values.append(wavelength_matrix)

        if all_wavelength_values:
            # Use finite values only
            finite_wavelengths = [w[np.isfinite(w)] for w in all_wavelength_values]
            if any(len(w) > 0 for w in finite_wavelengths):
                wavelength_min = max(
                    np.min([np.min(w) for w in finite_wavelengths if len(w) > 0]), 50.0
                )
                wavelength_max = min(
                    np.max([np.max(w) for w in finite_wavelengths if len(w) > 0]), 2000.0
                )
            else:
                wavelength_min, wavelength_max = 50.0, 2000.0
        else:
            wavelength_min, wavelength_max = 50.0, 2000.0

        norm = colors.Normalize(vmin=wavelength_min, vmax=wavelength_max)
        cmap = plt.colormaps[BIFURCATION_COLORMAP]

        # Plot each stage
        for stage_idx, stage_name in enumerate(stages):
            if stage_name not in results:
                continue

            stage_result = results[stage_name]
            tau_ratio_values = stage_result["tau_ratio_values"]
            sigma_ratio_values = stage_result["sigma_ratio_values"]
            k_matrix = stage_result["k_matrix"]
            stability_matrix = stage_result["stability_matrix"]
            flatness_matrix = stage_result.get("flatness_matrix", None)

            ax = fig.add_subplot(gs[0, stage_idx])

            # Convert k to wavelength: λ = 1/k (μm)
            wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)

            # Identify NaN values (failed steady-state convergence)
            nan_mask = np.isnan(k_matrix) | np.isnan(stability_matrix)

            # Compute alpha values based on stability
            alpha_matrix = np.zeros_like(stability_matrix)
            alpha_matrix[stability_matrix < STABILITY_THRESHOLD] = OPACITY_STABLE_FAR
            alpha_matrix[
                (stability_matrix >= STABILITY_THRESHOLD) & (stability_matrix < 0)
            ] = OPACITY_STABLE_NEAR
            alpha_matrix[stability_matrix >= 0] = OPACITY_UNSTABLE

            # Create RGBA image (grey for flat spectra, colormap otherwise, white for NaN)
            rgba_image = np.zeros((*wavelength_matrix.shape, 4))
            grey_color = (0.5, 0.5, 0.5)
            white_color = (1.0, 1.0, 1.0)
            for i in range(wavelength_matrix.shape[0]):
                for j in range(wavelength_matrix.shape[1]):
                    if nan_mask[i, j]:
                        # NaN values render as white (failed convergence)
                        rgba_image[i, j] = (*white_color, 1.0)
                    elif flatness_matrix is not None and flatness_matrix[i, j]:
                        color_rgb = grey_color
                        rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])
                    else:
                        color_rgb = cmap(norm(wavelength_matrix[i, j]))[:3]
                        rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])

            # Display image
            extent = [
                tau_ratio_values[0],
                tau_ratio_values[-1],
                sigma_ratio_values[0],
                sigma_ratio_values[-1],
            ]
            ax.imshow(rgba_image, origin="lower", extent=extent, interpolation="nearest")

            # Add stability boundary contour
            with contextlib.suppress(ValueError, RuntimeError):
                ax.contour(
                    tau_ratio_values,
                    sigma_ratio_values,
                    stability_matrix,
                    levels=[0],
                    colors="white",
                    linewidths=1.0,
                    linestyles="--",
                    alpha=0.8,
                )

            # Mark preset point
            preset_tau_ratio = stage_result["preset_tau_ratio"]
            preset_sigma_ratio = stage_result["preset_sigma_ratio"]
            ax.scatter(
                preset_tau_ratio,
                preset_sigma_ratio,
                marker="o",
                s=30,
                edgecolor="black",
                linewidth=1.2,
                facecolor="white",
                zorder=10,
            )

            # Set axis limits
            ax.set_xlim(tau_ratio_values[0], tau_ratio_values[-1])
            ax.set_ylim(sigma_ratio_values[0], sigma_ratio_values[-1])
            
            # Calculate aspect ratio to make plots visually square
            # Aspect ratio = (x_range) / (y_range) to maintain square appearance
            x_range = tau_ratio_values[-1] - tau_ratio_values[0]
            y_range = sigma_ratio_values[-1] - sigma_ratio_values[0]
            aspect_ratio = x_range / y_range if y_range > 0 else 1.0
            ax.set_aspect(aspect_ratio, adjustable="box")
            
            ax.locator_params(axis="x", nbins=4)
            ax.locator_params(axis="y", nbins=5)

            # Axis styling - make all spines visible
            ax.spines["bottom"].set_linewidth(self.default_spine_width)
            ax.spines["left"].set_linewidth(self.default_spine_width)
            ax.spines["top"].set_linewidth(self.default_spine_width)
            ax.spines["right"].set_linewidth(self.default_spine_width)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            # Labels
            if stage_idx == 0:
                ax.set_ylabel(
                    r"$\sigma_{\mathrm{PV}} / \sigma_{\mathrm{SST}}$",
                    fontsize=self.label_fontsize,
                    labelpad=6,
                )
            else:
                ax.set_ylabel("")

            ax.set_xlabel(
                r"$\tau_{\mathrm{PV}} / \tau_{\mathrm{SST}}$",
                fontsize=self.label_fontsize,
                labelpad=6,
            )

            ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.5)

            # Stage label
            ax.set_title(
                stage_name, fontsize=self.subtitle_fontsize, fontweight="bold", pad=10
            )

        # Add 2D opacity legend (same as standard stability maps)
        cbar_bottom = 0.14
        cbar_top = 0.80
        cbar_height = cbar_top - cbar_bottom
        cbar_width = 0.05
        cbar_ax = fig.add_axes([0.95, cbar_bottom, cbar_width, cbar_height])

        # Create 2D opacity legend
        n_ticks = 100
        opacity_legend = np.zeros((n_ticks, 3, 4))  # RGBA array

        wavelength_values_for_legend = np.linspace(wavelength_min, wavelength_max, n_ticks)
        for i, wavelength_val in enumerate(wavelength_values_for_legend):
            color_rgb = cmap(norm(wavelength_val))[:3]
            opacity_legend[i, 0, :3] = color_rgb
            opacity_legend[i, 0, 3] = OPACITY_STABLE_FAR
            opacity_legend[i, 1, :3] = color_rgb
            opacity_legend[i, 1, 3] = OPACITY_STABLE_NEAR
            opacity_legend[i, 2, :3] = color_rgb
            opacity_legend[i, 2, 3] = OPACITY_UNSTABLE

        # Display the 2D legend
        cbar_ax.imshow(opacity_legend, origin="lower", aspect="auto", interpolation="nearest")

        # Set up axes
        cbar_ax.set_xlim(-0.5, 2.5)
        cbar_ax.set_ylim(0, n_ticks - 1)

        # X-axis labels for the three stability ranges
        cbar_ax.set_xticks([0, 1, 2])
        cbar_ax.set_xticklabels(
            ["Stable (far)", "Stable (near)", "Unstable"],
            fontsize=self.secondary_tick_fontsize,
            rotation=45,
            ha="left",
        )
        cbar_ax.xaxis.set_label_position("top")
        cbar_ax.xaxis.tick_top()
        cbar_ax.tick_params(
            axis="x", bottom=False, top=True, labelsize=self.secondary_tick_fontsize
        )

        # Y-axis: wavelength values
        wavelength_tick_positions = np.linspace(0, n_ticks - 1, 6)
        wavelength_tick_values = np.linspace(wavelength_min, wavelength_max, 6)
        cbar_ax.set_yticks(wavelength_tick_positions)
        cbar_ax.set_yticklabels(
            [f"{w:.0f}" for w in wavelength_tick_values], fontsize=self.tick_fontsize
        )
        cbar_ax.set_ylabel(
            r"Wavelength w/ max Re($\lambda$) ($\mu$m)", fontsize=self.label_fontsize, labelpad=15
        )
        cbar_ax.yaxis.set_label_position("right")
        cbar_ax.yaxis.tick_right()
        cbar_ax.tick_params(axis="y", left=False, right=True, labelsize=self.tick_fontsize)

        # Style the axes
        cbar_ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.6)
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)

        # Add grey legend at bottom of colorbar
        grey_legend_bottom = 0.08 + 0.01
        stable_ax = fig.add_axes([0.95, grey_legend_bottom, cbar_width, 0.03])
        stable_ax.imshow(
            np.full((10, 1), 0.5), cmap="Greys", vmin=0, vmax=1, origin="lower", aspect="auto"
        )
        stable_ax.set_xticks([])
        stable_ax.set_yticks([])
        for spine in stable_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
        stable_ax.text(
            0.5,
            -0.25,
            "No dominant\nspatial mode",
            transform=stable_ax.transAxes,
            fontsize=self.secondary_label_fontsize,
            rotation=0,
            va="top",
            ha="center",
        )

        # Overall title
        title = "Stability Landscape: SST-PV Maturation Ratios"
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight="bold", y=1.03)

        return fig

    def create_maturity_stability_map_figure(
        self,
        results: dict,
        stages: list[str],
    ) -> plt.Figure:
        """Create maturity stability map figure showing SST/PV maturation indices.

        Args:
            results: Results dict organized as {stage: stage_results}
            stages: List of stage names

        Returns:
            matplotlib Figure object
        """
        n_stages = len(stages)

        # Create figure with single row of n_stages panels
        fig_height = self.fig_height_per_row * 1.18  # Slightly taller for x-labels
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(
            1,
            n_stages,
            figure=fig,
            hspace=0.45,
            wspace=self.wspace,
            left=self.left_margin,
            right=0.83,
            top=0.80,
            bottom=0.08,
        )

        # Determine global wavelength range for consistent colormap across all stages
        all_wavelength_values = []
        for stage in stages:
            if stage in results:
                k_matrix = results[stage]["k_matrix"]
                # Convert k to wavelength, avoiding division by zero
                wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)
                all_wavelength_values.append(wavelength_matrix)

        if all_wavelength_values:
            # Use finite values only
            finite_wavelengths = [w[np.isfinite(w)] for w in all_wavelength_values]
            if any(len(w) > 0 for w in finite_wavelengths):
                wavelength_min = max(
                    np.min([np.min(w) for w in finite_wavelengths if len(w) > 0]), 50.0
                )
                wavelength_max = min(
                    np.max([np.max(w) for w in finite_wavelengths if len(w) > 0]), 2000.0
                )
            else:
                wavelength_min, wavelength_max = 50.0, 2000.0
        else:
            wavelength_min, wavelength_max = 50.0, 2000.0

        norm = colors.Normalize(vmin=wavelength_min, vmax=wavelength_max)
        cmap = plt.colormaps[BIFURCATION_COLORMAP]

        # Plot each stage
        for stage_idx, stage_name in enumerate(stages):
            if stage_name not in results:
                continue

            stage_result = results[stage_name]
            sst_maturity_values = stage_result["sst_maturity_values"]
            pv_maturity_values = stage_result["pv_maturity_values"]
            k_matrix = stage_result["k_matrix"]
            stability_matrix = stage_result["stability_matrix"]
            flatness_matrix = stage_result.get("flatness_matrix", None)

            ax = fig.add_subplot(gs[0, stage_idx])

            # Convert k to wavelength: λ = 1/k (μm)
            wavelength_matrix = np.where(k_matrix > 0, 1.0 / k_matrix, np.inf)

            # Identify NaN values (failed steady-state convergence)
            nan_mask = np.isnan(k_matrix) | np.isnan(stability_matrix)

            # Compute alpha values based on stability
            alpha_matrix = np.zeros_like(stability_matrix)
            alpha_matrix[stability_matrix < STABILITY_THRESHOLD] = OPACITY_STABLE_FAR
            alpha_matrix[
                (stability_matrix >= STABILITY_THRESHOLD) & (stability_matrix < 0)
            ] = OPACITY_STABLE_NEAR
            alpha_matrix[stability_matrix >= 0] = OPACITY_UNSTABLE

            # Create RGBA image (grey for flat spectra, colormap otherwise, white for NaN)
            rgba_image = np.zeros((*wavelength_matrix.shape, 4))
            grey_color = (0.5, 0.5, 0.5)
            white_color = (1.0, 1.0, 1.0)
            for i in range(wavelength_matrix.shape[0]):
                for j in range(wavelength_matrix.shape[1]):
                    if nan_mask[i, j]:
                        # NaN values render as white (failed convergence)
                        rgba_image[i, j] = (*white_color, 1.0)
                    elif flatness_matrix is not None and flatness_matrix[i, j]:
                        color_rgb = grey_color
                        rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])
                    else:
                        color_rgb = cmap(norm(wavelength_matrix[i, j]))[:3]
                        rgba_image[i, j] = (*color_rgb, alpha_matrix[i, j])

            # Display image
            extent = [
                sst_maturity_values[0],
                sst_maturity_values[-1],
                pv_maturity_values[0],
                pv_maturity_values[-1],
            ]
            ax.imshow(rgba_image, origin="lower", extent=extent, interpolation="nearest")

            # Add stability boundary contour
            with contextlib.suppress(ValueError, RuntimeError):
                ax.contour(
                    sst_maturity_values,
                    pv_maturity_values,
                    stability_matrix,
                    levels=[0],
                    colors="white",
                    linewidths=1.0,
                    linestyles="--",
                    alpha=0.8,
                )

            # Mark preset point
            preset_sst_maturity = stage_result["preset_sst_maturity"]
            preset_pv_maturity = stage_result["preset_pv_maturity"]
            ax.scatter(
                preset_sst_maturity,
                preset_pv_maturity,
                marker="o",
                s=30,
                edgecolor="black",
                linewidth=1.2,
                facecolor="white",
                zorder=10,
            )

            # Set axis limits
            ax.set_xlim(sst_maturity_values[0], sst_maturity_values[-1])
            ax.set_ylim(pv_maturity_values[0], pv_maturity_values[-1])

            # Calculate aspect ratio to make plots visually square
            x_range = sst_maturity_values[-1] - sst_maturity_values[0]
            y_range = pv_maturity_values[-1] - pv_maturity_values[0]
            aspect_ratio = x_range / y_range if y_range > 0 else 1.0
            ax.set_aspect(aspect_ratio, adjustable="box")

            ax.locator_params(axis="x", nbins=4)
            ax.locator_params(axis="y", nbins=5)

            # Axis styling - make all spines visible
            ax.spines["bottom"].set_linewidth(self.default_spine_width)
            ax.spines["left"].set_linewidth(self.default_spine_width)
            ax.spines["top"].set_linewidth(self.default_spine_width)
            ax.spines["right"].set_linewidth(self.default_spine_width)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            # Labels
            if stage_idx == 0:
                ax.set_ylabel(
                    "PV Maturity",
                    fontsize=self.label_fontsize,
                    labelpad=6,
                )
            else:
                ax.set_ylabel("")

            ax.set_xlabel(
                "SST Maturity",
                fontsize=self.label_fontsize,
                labelpad=6,
            )

            ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.5)

            # Stage label
            ax.set_title(
                stage_name, fontsize=self.subtitle_fontsize, fontweight="bold", pad=10
            )

        # Add 2D opacity legend (same as standard stability maps)
        cbar_bottom = 0.14
        cbar_top = 0.80
        cbar_height = cbar_top - cbar_bottom
        cbar_width = 0.05
        cbar_ax = fig.add_axes([0.95, cbar_bottom, cbar_width, cbar_height])

        # Create 2D opacity legend
        n_ticks = 100
        opacity_legend = np.zeros((n_ticks, 3, 4))  # RGBA array

        wavelength_values_for_legend = np.linspace(wavelength_min, wavelength_max, n_ticks)
        for i, wavelength_val in enumerate(wavelength_values_for_legend):
            color_rgb = cmap(norm(wavelength_val))[:3]
            opacity_legend[i, 0, :3] = color_rgb
            opacity_legend[i, 0, 3] = OPACITY_STABLE_FAR
            opacity_legend[i, 1, :3] = color_rgb
            opacity_legend[i, 1, 3] = OPACITY_STABLE_NEAR
            opacity_legend[i, 2, :3] = color_rgb
            opacity_legend[i, 2, 3] = OPACITY_UNSTABLE

        # Display the 2D legend
        cbar_ax.imshow(opacity_legend, origin="lower", aspect="auto", interpolation="nearest")

        # Set up axes
        cbar_ax.set_xlim(-0.5, 2.5)
        cbar_ax.set_ylim(0, n_ticks - 1)

        # X-axis labels for the three stability ranges
        cbar_ax.set_xticks([0, 1, 2])
        cbar_ax.set_xticklabels(
            ["Stable (far)", "Stable (near)", "Unstable"],
            fontsize=self.secondary_tick_fontsize,
            rotation=45,
            ha="left",
        )
        cbar_ax.xaxis.set_label_position("top")
        cbar_ax.xaxis.tick_top()
        cbar_ax.tick_params(
            axis="x", bottom=False, top=True, labelsize=self.secondary_tick_fontsize
        )

        # Y-axis: wavelength values
        wavelength_tick_positions = np.linspace(0, n_ticks - 1, 6)
        wavelength_tick_values = np.linspace(wavelength_min, wavelength_max, 6)
        cbar_ax.set_yticks(wavelength_tick_positions)
        cbar_ax.set_yticklabels(
            [f"{w:.0f}" for w in wavelength_tick_values], fontsize=self.tick_fontsize
        )
        cbar_ax.set_ylabel(
            r"Wavelength w/ max Re($\lambda$) ($\mu$m)", fontsize=self.label_fontsize, labelpad=15
        )
        cbar_ax.yaxis.set_label_position("right")
        cbar_ax.yaxis.tick_right()
        cbar_ax.tick_params(axis="y", left=False, right=True, labelsize=self.tick_fontsize)

        # Style the axes
        cbar_ax.tick_params(labelsize=self.tick_fontsize, length=3, width=0.6)
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)

        # Add grey legend at bottom of colorbar
        grey_legend_bottom = 0.08 + 0.01
        stable_ax = fig.add_axes([0.95, grey_legend_bottom, cbar_width, 0.03])
        stable_ax.imshow(
            np.full((10, 1), 0.5), cmap="Greys", vmin=0, vmax=1, origin="lower", aspect="auto"
        )
        stable_ax.set_xticks([])
        stable_ax.set_yticks([])
        for spine in stable_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
        stable_ax.text(
            0.5,
            -0.25,
            "No dominant\nspatial mode",
            transform=stable_ax.transAxes,
            fontsize=self.secondary_label_fontsize,
            rotation=0,
            va="top",
            ha="center",
        )

        # Overall title
        title = "Stability Landscape: Interneuron Maturation Indices"
        fig.suptitle(title, fontsize=self.title_fontsize, fontweight="bold", y=1.03)

        return fig

    def generate_all_figures(self, results: dict, mode: str = "fixed_absolute") -> None:
        """Generate all figures from results.

        Args:
            results: Dictionary with 'stability', 'gain_maps', and/or 'gain_spectra' keys
            mode: Range mode for determining axis emphasis
        """
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        stages = DEVELOPMENTAL_STAGES

        # Generate stability map figures (one figure with all parameter pairs)
        if "stability" in results:
            print("\nGenerating stability maps figure...")
            stability_results = results["stability"]
            param_pairs = list(stability_results.keys())

            if param_pairs:
                fig = self.create_stability_map_figure(stability_results, param_pairs, stages, mode)

                filename = f"stability_maps_{mode}.pdf"
                filepath = output_dir / filename
                save_figure(fig, filepath)
                print(f"  Saved: {filepath}")

        # Generate gain map figures (one figure with all parameter pairs)
        if "gain_maps" in results:
            print("\nGenerating gain maps figure...")
            gain_map_results = results["gain_maps"]
            param_pairs = list(gain_map_results.keys())

            if param_pairs:
                fig = self.create_gain_map_figure(gain_map_results, param_pairs, stages, mode)

                filename = f"gain_maps_{mode}.pdf"
                filepath = output_dir / filename
                save_figure(fig, filepath)
                print(f"  Saved: {filepath}")

        # Generate gain spectrum figures (one figure with all parameters)
        if "gain_spectra" in results:
            print("\nGenerating gain spectra figure...")
            spectrum_results = results["gain_spectra"]
            param_keys = list(spectrum_results.keys())

            if param_keys:
                fig = self.create_gain_spectrum_figure(spectrum_results, param_keys, stages)

                filename = "gain_spectra.pdf"
                filepath = output_dir / filename
                save_figure(fig, filepath)
                print(f"  Saved: {filepath}")

        # Generate compressed stability map figure (SST-PV ratio analysis)
        if "compressed_stability" in results:
            print("\nGenerating compressed stability map figure...")
            compressed_results = results["compressed_stability"]
            stage_names = list(compressed_results.keys())

            if stage_names:
                fig = self.create_compressed_stability_map_figure(compressed_results, stage_names)

                filename = "compressed_stability_maps.pdf"
                filepath = output_dir / filename
                save_figure(fig, filepath)
                print(f"  Saved: {filepath}")

        # Generate maturity stability map figure (SST/PV maturation indices)
        if "maturity_stability" in results:
            print("\nGenerating maturity stability map figure...")
            maturity_results = results["maturity_stability"]
            stage_names = list(maturity_results.keys())

            if stage_names:
                fig = self.create_maturity_stability_map_figure(maturity_results, stage_names)

                filename = "maturity_stability_maps.pdf"
                filepath = output_dir / filename
                save_figure(fig, filepath)
                print(f"  Saved: {filepath}")

        print(f"\nAll figures saved to: {output_dir}")
