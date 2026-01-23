"""Visualization for descriptive analysis results."""

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import os
from typing import Any

from tqdm import tqdm

from src.analysis.common import DEVELOPMENTAL_STAGES, apply_matplotlib_style, save_figure

from .config import (
    ANALYSIS_PARAMS,
    AVERAGE_FIRING_RATE_YLIM,
    CELL_COLORS,
    DPI,
    ERROR_BAR_ALPHA,
    FIGSIZE_TIMESERIES,
    FIGSIZE_TRENDS,
    FONT_SIZES,
    HEATMAP_VMAX,
    HEATMAP_VMIN,
    LAYER_COLORS,
    LAYERS,
    LINE_WIDTH,
    MARKER_SIZE,
    OUTPUT_DIR,
    POSTER_CELL_TYPES,
    SEM_FACTOR,
    SUBPLOT_PADDING,
    Y_MARGIN_FACTOR,
)


class ActivityVisualizer:
    """Creates visualization plots for descriptive activity analysis."""

    def __init__(self) -> None:
        """Initialize visualizer and ensure output directory exists."""
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_matplotlib_style()

    def _setup_matplotlib_style(self) -> None:
        """Configure matplotlib style for consistent poster-format plots."""
        apply_matplotlib_style(overrides={"axes.grid": False})

    def _convert_time_to_seconds(self, time_ms: list[float]) -> np.ndarray:
        """Convert time from milliseconds to seconds."""
        return np.array(time_ms) / 1000.0

    def _format_minimal_ticks(
        self, values: list[float], is_time: bool = False
    ) -> tuple[list[float], list[str]]:
        """Create minimal tick marks with just min and max values."""
        if len(values) == 0:
            return [], []

        vmin, vmax = min(values), max(values)

        if is_time:
            tick_values = self._get_time_ticks(vmax)
        else:
            tick_values = [vmin, vmax]

        tick_labels = self._format_tick_labels(tick_values, is_time)
        return tick_values, tick_labels

    def _get_time_ticks(self, vmax: float) -> list[int]:
        """Get appropriate time tick values."""
        if vmax <= 1:
            return [0, 1]
        elif vmax <= 5:
            return [0, 5]
        else:
            return [0, int(vmax)]

    def _format_tick_labels(self, tick_values: list[float], is_time: bool) -> list[str]:
        """Format tick labels with appropriate precision."""
        if is_time:
            return [f"{v}" for v in tick_values]

        # Determine precision based on value range
        range_val = max(tick_values) - min(tick_values)
        if range_val < 0.01:
            precision = 3
        elif range_val < 0.1:
            precision = 2
        elif range_val < 1:
            precision = 1
        else:
            precision = 0

        return [f"{v:.{precision}f}" for v in tick_values]

    def _format_zero_max_ticks(self, values: list[float]) -> tuple[list[float], list[str]]:
        """Create tick marks at zero and the actual maximum value."""
        if len(values) == 0:
            return [], []

        max_val = max(values)
        precision = self._get_precision(max_val)

        tick_values = [0, max_val]
        tick_labels = ["0", f"{max_val:.{precision}f}"]

        return tick_values, tick_labels

    def _get_precision(self, value: float) -> int:
        """Determine appropriate precision for a numeric value."""
        if value < 0.001:
            return 4
        elif value < 0.01:
            return 3
        elif value < 0.1:
            return 2
        elif value < 1:
            return 1
        else:
            return 0

    def _create_cell_colormaps(self) -> dict[str, Any]:
        """Create custom colormaps for each cell type (white to cell color)."""
        cell_cmaps = {}
        for cell_type in POSTER_CELL_TYPES:
            colors = ["white", CELL_COLORS[cell_type]]
            cell_cmaps[cell_type] = mcolors.LinearSegmentedColormap.from_list(
                f"{cell_type}_cmap", colors, N=256
            )
        return cell_cmaps

    def _prepare_heatmap_data(
        self, results: dict[str, Any], stage: str, cell_type: str
    ) -> np.ndarray:
        """Prepare spatial activity data for heatmap visualization."""
        stacked_data = []
        for layer in LAYERS:
            layer_data = results[stage]["spatial_activities"][layer][cell_type]
            # Dynamically determine grid size from data shape
            n_timepoints = layer_data.shape[0]
            n_cells = layer_data.shape[1]
            grid_size = int(np.sqrt(n_cells))

            # Reshape from (n_timepoints, grid_size^2) to (n_timepoints, grid_size, grid_size)
            reshaped = layer_data.reshape(n_timepoints, grid_size, grid_size)
            # Average across one spatial dimension to get (grid_size, n_timepoints)
            spatial_avg = np.mean(reshaped, axis=2)  # Average across columns
            stacked_data.append(spatial_avg.T)  # Transpose to (grid_size, n_timepoints)

        # Concatenate layers vertically
        return np.vstack(stacked_data)  # Shape: (3*grid_size, n_timepoints)

    def _add_layer_dividers(self, ax: plt.Axes, grid_size: int) -> None:
        """Add horizontal lines to separate layers in heatmap."""
        ax.axhline(y=grid_size - 0.5, color="lightgrey", linewidth=1)
        ax.axhline(y=2 * grid_size - 0.5, color="lightgrey", linewidth=1)

    def _setup_subplot_layout(self, fig: plt.Figure, axes: np.ndarray, num_rows: int) -> None:
        """Set up consistent subplot layout and spacing."""
        plt.tight_layout()
        plt.subplots_adjust(hspace=SUBPLOT_PADDING["hspace"], wspace=SUBPLOT_PADDING["wspace"])

    def _add_colorbar_to_heatmap(
        self, fig: plt.Figure, axes: np.ndarray, images: dict[str, Any]
    ) -> None:
        """Add vertical colorbars to the right of heatmap plots."""
        plt.subplots_adjust(right=SUBPLOT_PADDING["right_margin"])

        for row, cell_type in enumerate(POSTER_CELL_TYPES):
            ax = axes[row, -1]  # Rightmost column
            cbar_ax = fig.add_axes(
                [
                    ax.get_position().x1 + SUBPLOT_PADDING["colorbar_padding"],
                    ax.get_position().y0,
                    SUBPLOT_PADDING["colorbar_width"],
                    ax.get_position().height,
                ]
            )
            cbar = fig.colorbar(images[cell_type], cax=cbar_ax, orientation="vertical")
            cbar.ax.tick_params(labelsize=FONT_SIZES["tick_labels"])

            # Minimal colorbar ticks
            tick_values, tick_labels = self._format_minimal_ticks([HEATMAP_VMIN, HEATMAP_VMAX])
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels(tick_labels)

            if row == 1:  # Middle row gets the label
                cbar.set_label("Firing Rate", fontsize=FONT_SIZES["ylabel"], rotation=90)

    def _format_subplot_axes(
        self,
        ax: plt.Axes,
        row: int,
        col: int,
        stage: str,
        cell_type: str,
        is_heatmap: bool = False,
        ylim: list[float] = None,
    ) -> None:
        """Format individual subplot axes with consistent styling."""
        # Column headers (ages)
        if row == 0:
            ax.set_title(stage, fontweight="bold")

        # Row labels (cell types)
        if col == 0:
            ax.set_ylabel(
                f"{cell_type}",
                fontweight="bold",
                rotation=0,
                ha="right",
                va="center",
                color=CELL_COLORS[cell_type],
            )

            if is_heatmap:
                # Add layer labels on y-axis for heatmaps
                ax.set_yticks([10, 30, 50])
                ax.set_yticklabels(["L2/3", "L4", "L5"])
                ax.tick_params(left=False)
            elif ylim:
                # Y-axis formatting for timeseries
                tick_values, tick_labels = self._format_minimal_ticks(ylim)
                ax.set_yticks(tick_values)
                ax.set_yticklabels(tick_labels)
        else:
            if is_heatmap:
                ax.set_yticks([])
            elif ylim:
                tick_values, tick_labels = self._format_minimal_ticks(ylim)
                ax.set_yticks(tick_values)
                ax.set_yticklabels([])
                ax.tick_params(left=False)

    def _format_time_axis(self, ax: plt.Axes, time: np.ndarray, row: int, num_rows: int) -> None:
        """Format x-axis for time-based plots with consistent range."""
        # Use configured simulation duration for consistent x-axis across all plots
        max_time = ANALYSIS_PARAMS["simulation_duration"]
        ax.set_xlim(0, max_time)

        if row == num_rows - 1:  # Bottom row
            ax.set_xlabel("Time (s)")
            tick_values, tick_labels = self._format_minimal_ticks([0, max_time], is_time=True)
            ax.set_xticks(tick_values)
            ax.set_xticklabels(tick_labels)
        else:
            ax.set_xticks([])

    def create_activity_heatmaps(self, results: dict[str, Any]) -> None:
        """Create 3x4 activity heatmaps showing spatial activity over time."""
        fig, axes = plt.subplots(3, 4, figsize=FIGSIZE_TIMESERIES)
        fig.suptitle("Firing Rates", fontsize=FONT_SIZES["title"], fontweight="bold")

        cell_cmaps = self._create_cell_colormaps()
        images = {}

        for row, cell_type in enumerate(POSTER_CELL_TYPES):
            for col, stage in enumerate(DEVELOPMENTAL_STAGES):
                ax = axes[row, col]

                # Prepare and plot heatmap data
                heatmap_data = self._prepare_heatmap_data(results, stage, cell_type)
                im = ax.imshow(
                    heatmap_data,
                    aspect="auto",
                    cmap=cell_cmaps[cell_type],
                    vmin=HEATMAP_VMIN,
                    vmax=HEATMAP_VMAX,
                )

                # Store image for colorbar
                if col == len(DEVELOPMENTAL_STAGES) - 1:
                    images[cell_type] = im

                # Calculate grid_size from heatmap data (total height / 3 layers)
                grid_size = heatmap_data.shape[0] // 3
                self._add_layer_dividers(ax, grid_size)
                self._format_subplot_axes(ax, row, col, stage, cell_type, is_heatmap=True)

                # X-axis formatting for heatmaps
                if row == len(POSTER_CELL_TYPES) - 1:
                    ax.set_xlabel("Time (s)")
                    n_timepoints = heatmap_data.shape[1]
                    max_time = ANALYSIS_PARAMS["simulation_duration"]
                    ax.set_xlim(-0.5, n_timepoints - 0.5)  # Proper heatmap pixel boundaries
                    ax.set_xticks([0, n_timepoints - 1])
                    ax.set_xticklabels(["0", f"{int(max_time)}"])
                else:
                    ax.set_xticks([])

        self._add_colorbar_to_heatmap(fig, axes, images)
        self._save_plot("activity_heatmaps.pdf")

    def _calculate_global_ylim(self, results: dict[str, Any], data_key: str) -> list[float]:
        """Calculate global y-limits for consistent scaling across subplots."""
        all_values = []
        for stage in DEVELOPMENTAL_STAGES:
            for cell_type in POSTER_CELL_TYPES:
                for layer in LAYERS:
                    all_values.extend(results[stage][data_key][layer][cell_type])

        return [0, max(all_values) * (1 + Y_MARGIN_FACTOR)]

    def _plot_average_across_layers(
        self, ax: plt.Axes, results: dict[str, Any], stage: str, cell_type: str, data_key: str
    ) -> None:
        """Plot average timeseries across all layers for a given cell type."""
        time = self._convert_time_to_seconds(results[stage]["time"])

        # Collect data from all layers
        combined_data = []
        for layer in LAYERS:
            data = results[stage][data_key][layer][cell_type]
            combined_data.append(data)

        # Average across layers and plot
        avg_data = np.mean(combined_data, axis=0)
        ax.plot(time, avg_data, color=CELL_COLORS[cell_type], linewidth=1)

    def create_average_firing_rate_plots(self, results: dict[str, Any]) -> None:
        """Create 3x4 average firing rate timeseries plots."""
        fig, axes = plt.subplots(3, 4, figsize=FIGSIZE_TIMESERIES)
        fig.suptitle("Average Firing Rates", fontsize=FONT_SIZES["title"], fontweight="bold")

        # Use configurable ylim if provided, otherwise calculate automatically
        if AVERAGE_FIRING_RATE_YLIM is not None:
            ylim = AVERAGE_FIRING_RATE_YLIM
        else:
            ylim = self._calculate_global_ylim(results, "average_rates")

        for row, cell_type in enumerate(POSTER_CELL_TYPES):
            for col, stage in enumerate(DEVELOPMENTAL_STAGES):
                ax = axes[row, col]

                self._plot_average_across_layers(ax, results, stage, cell_type, "average_rates")
                ax.set_ylim(ylim)

                self._format_subplot_axes(ax, row, col, stage, cell_type, ylim=ylim)

                if col == 0:  # Only leftmost column gets y-tick labels
                    pass  # Already handled in _format_subplot_axes

                time = self._convert_time_to_seconds(results[stage]["time"])
                self._format_time_axis(ax, time, row, len(POSTER_CELL_TYPES))

        self._setup_subplot_layout(fig, axes, len(POSTER_CELL_TYPES))
        self._save_plot("average_firing_rates.pdf")

    def create_active_cell_fraction_plots(self, results: dict[str, Any]) -> None:
        """Create 3x4 active cell fraction timeseries plots."""
        fig, axes = plt.subplots(3, 4, figsize=FIGSIZE_TIMESERIES)
        fig.suptitle("Fraction of Active Cells", fontsize=FONT_SIZES["title"], fontweight="bold")

        ylim = self._calculate_global_ylim(results, "active_fractions")

        for row, cell_type in enumerate(POSTER_CELL_TYPES):
            for col, stage in enumerate(DEVELOPMENTAL_STAGES):
                ax = axes[row, col]

                self._plot_average_across_layers(ax, results, stage, cell_type, "active_fractions")
                ax.set_ylim(ylim)

                self._format_subplot_axes(ax, row, col, stage, cell_type, ylim=ylim)

                time = self._convert_time_to_seconds(results[stage]["time"])
                self._format_time_axis(ax, time, row, len(POSTER_CELL_TYPES))

        self._setup_subplot_layout(fig, axes, len(POSTER_CELL_TYPES))
        self._save_plot("active_cell_fractions.pdf")

    def _plot_trend_with_errorbars(
        self, ax: plt.Axes, x_pos: list[int], means: list[float], color: str, label: str = None
    ) -> None:
        """Plot trend line with error bars."""
        # Estimate SEM as fraction of mean for visualization
        sems = [m * SEM_FACTOR for m in means]

        ax.plot(
            x_pos,
            means,
            "o-",
            color=color,
            label=label,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )
        ax.fill_between(
            x_pos,
            [m - s for m, s in zip(means, sems, strict=False)],
            [m + s for m, s in zip(means, sems, strict=False)],
            color=color,
            alpha=ERROR_BAR_ALPHA,
            linewidth=0,
        )

    def _setup_trend_plot_axes(
        self,
        ax: plt.Axes,
        title: str,
        ylabel: str,
        x_pos: list[int],
        ylim: list[float],
        tick_values: list[float],
        tick_labels: list[str],
        show_ylabel: bool = True,
    ) -> None:
        """Set up axes for trend plots."""
        ax.set_title(title)
        if show_ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(DEVELOPMENTAL_STAGES)
        ax.set_ylim(ylim)
        ax.set_yticks(tick_values)
        if show_ylabel:
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_yticklabels([])
            ax.tick_params(left=False)

    def _add_trend_legends(self, fig: plt.Figure, axes: np.ndarray) -> None:
        """Add legends to the right of trend plots."""
        plt.subplots_adjust(right=0.8)

        fig.legend(
            axes[0].get_lines(),
            [line.get_label() for line in axes[0].get_lines()],
            loc="center left",
            bbox_to_anchor=(0.82, 0.75),
            title="Cell Types",
        )
        fig.legend(
            axes[1].get_lines(),
            [line.get_label() for line in axes[1].get_lines()],
            loc="center left",
            bbox_to_anchor=(0.82, 0.25),
            title="Layers",
        )

    def create_pairwise_correlation_trends(self, results: dict[str, Any]) -> None:
        """Create developmental trends in pairwise correlations."""
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRENDS)
        fig.suptitle("Pairwise Correlation", fontsize=FONT_SIZES["title"], fontweight="bold")

        x_pos = list(range(len(DEVELOPMENTAL_STAGES)))

        # Collect all correlation values for consistent y-axis
        all_correlation_values = self._collect_correlation_values(results)

        # Calculate consistent y-axis limits (handle case with no valid values)
        if all_correlation_values:
            max_val = max(all_correlation_values)
            y_buffer = max_val * 0.1
            correlation_ylim = [-y_buffer * 0.5, max_val + y_buffer]
            correlation_tick_values, correlation_tick_labels = self._format_zero_max_ticks(
                all_correlation_values
            )
        else:
            # Fallback if all values are NaN
            correlation_ylim = [0, 1]
            correlation_tick_values = [0, 1]
            correlation_tick_labels = ["0", "1"]

        # Plot 1: By cell type
        for cell_type in POSTER_CELL_TYPES:
            means = [
                results[stage]["correlations"]["by_celltype"].get(cell_type, np.nan)
                for stage in DEVELOPMENTAL_STAGES
            ]
            self._plot_trend_with_errorbars(
                axes[0], x_pos, means, CELL_COLORS[cell_type], cell_type
            )

        self._setup_trend_plot_axes(
            axes[0],
            "By Cell Type",
            "Pairwise Correlation",
            x_pos,
            correlation_ylim,
            correlation_tick_values,
            correlation_tick_labels,
        )

        # Plot 2: By layer
        for layer in LAYERS:
            means = [
                results[stage]["correlations"]["by_layer"].get(layer, np.nan)
                for stage in DEVELOPMENTAL_STAGES
            ]
            self._plot_trend_with_errorbars(axes[1], x_pos, means, LAYER_COLORS[layer], layer)

        self._setup_trend_plot_axes(
            axes[1],
            "By Layer",
            "",
            x_pos,
            correlation_ylim,
            correlation_tick_values,
            correlation_tick_labels,
            show_ylabel=False,
        )

        # Plot 3: Total network
        total_values = [results[stage]["correlations"]["total"] for stage in DEVELOPMENTAL_STAGES]
        self._plot_trend_with_errorbars(axes[2], x_pos, total_values, "black")

        self._setup_trend_plot_axes(
            axes[2],
            "Total Network",
            "",
            x_pos,
            correlation_ylim,
            correlation_tick_values,
            correlation_tick_labels,
            show_ylabel=False,
        )

        plt.tight_layout()
        self._add_trend_legends(fig, axes)
        self._save_plot("correlation_trends.pdf")

    def _collect_correlation_values(self, results: dict[str, Any]) -> list[float]:
        """Collect all correlation values for consistent scaling (excluding NaN)."""
        all_values = []

        for cell_type in POSTER_CELL_TYPES:
            values = [
                results[stage]["correlations"]["by_celltype"].get(cell_type, 0)
                for stage in DEVELOPMENTAL_STAGES
            ]
            all_values.extend(values)

        for layer in LAYERS:
            values = [
                results[stage]["correlations"]["by_layer"].get(layer, 0)
                for stage in DEVELOPMENTAL_STAGES
            ]
            all_values.extend(values)

        total_values = [results[stage]["correlations"]["total"] for stage in DEVELOPMENTAL_STAGES]
        all_values.extend(total_values)

        # Filter out NaN values
        return [v for v in all_values if not np.isnan(v)]

    def create_dimensionality_trends(self, results: dict[str, Any]) -> None:
        """Create developmental trends in network dimensionality (participation ratio)."""
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRENDS)
        fig.suptitle("Network Dimensionality (Participation Ratio)", fontsize=FONT_SIZES["title"], fontweight="bold")

        x_pos = list(range(len(DEVELOPMENTAL_STAGES)))

        # Collect all dimensionality values for consistent y-axis
        all_dim_values = self._collect_dimensionality_values(results)

        # Calculate consistent y-axis limits
        if all_dim_values:
            min_val = min(all_dim_values)
            max_val = max(all_dim_values)
            y_buffer = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            dim_ylim = [max(0, min_val - y_buffer), min(1, max_val + y_buffer)]
            dim_tick_values = [dim_ylim[0], dim_ylim[1]]
            dim_tick_labels = [f"{v:.2f}" for v in dim_tick_values]
        else:
            # Fallback if all values are NaN
            dim_ylim = [0, 1]
            dim_tick_values = [0, 1]
            dim_tick_labels = ["0", "1"]

        # Plot 1: By cell type
        for cell_type in POSTER_CELL_TYPES:
            means = [
                results[stage]["dimensionality"]["by_celltype"].get(cell_type, np.nan)
                for stage in DEVELOPMENTAL_STAGES
            ]
            self._plot_trend_with_errorbars(
                axes[0], x_pos, means, CELL_COLORS[cell_type], cell_type
            )

        self._setup_trend_plot_axes(
            axes[0],
            "By Cell Type",
            "Normalized PR",
            x_pos,
            dim_ylim,
            dim_tick_values,
            dim_tick_labels,
        )

        # Plot 2: By layer
        for layer in LAYERS:
            means = [
                results[stage]["dimensionality"]["by_layer"].get(layer, np.nan)
                for stage in DEVELOPMENTAL_STAGES
            ]
            self._plot_trend_with_errorbars(axes[1], x_pos, means, LAYER_COLORS[layer], layer)

        self._setup_trend_plot_axes(
            axes[1],
            "By Layer",
            "",
            x_pos,
            dim_ylim,
            dim_tick_values,
            dim_tick_labels,
            show_ylabel=False,
        )

        # Plot 3: Total network
        total_values = [results[stage]["dimensionality"]["total"] for stage in DEVELOPMENTAL_STAGES]
        self._plot_trend_with_errorbars(axes[2], x_pos, total_values, "black")

        self._setup_trend_plot_axes(
            axes[2],
            "Total Network",
            "",
            x_pos,
            dim_ylim,
            dim_tick_values,
            dim_tick_labels,
            show_ylabel=False,
        )

        plt.tight_layout()
        self._add_trend_legends(fig, axes)
        self._save_plot("dimensionality_trends.pdf")

    def _collect_dimensionality_values(self, results: dict[str, Any]) -> list[float]:
        """Collect all dimensionality values for consistent scaling (excluding NaN)."""
        all_values = []

        for cell_type in POSTER_CELL_TYPES:
            values = [
                results[stage]["dimensionality"]["by_celltype"].get(cell_type, 0)
                for stage in DEVELOPMENTAL_STAGES
            ]
            all_values.extend(values)

        for layer in LAYERS:
            values = [
                results[stage]["dimensionality"]["by_layer"].get(layer, 0)
                for stage in DEVELOPMENTAL_STAGES
            ]
            all_values.extend(values)

        total_values = [results[stage]["dimensionality"]["total"] for stage in DEVELOPMENTAL_STAGES]
        all_values.extend(total_values)

        # Filter out NaN values
        return [v for v in all_values if not np.isnan(v)]

    def create_synchronous_event_trends(self, results: dict[str, Any]) -> None:
        """Create developmental trends in synchronous events."""
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRENDS)
        fig.suptitle("Large Synchronous Events", fontsize=FONT_SIZES["title"], fontweight="bold")

        x_pos = list(range(len(DEVELOPMENTAL_STAGES)))
        sim_duration = ANALYSIS_PARAMS["simulation_duration"]

        # Collect all event values (converted to rates) for consistent y-axis
        all_event_values = self._collect_event_values(results, sim_duration)

        # Calculate consistent y-axis limits
        max_val = max(all_event_values)
        y_buffer = max_val * 0.1
        event_ylim = [-y_buffer * 0.5, max_val + y_buffer]
        event_tick_values, event_tick_labels = self._format_zero_max_ticks(all_event_values)

        # Plot 1: By cell type
        for cell_type in POSTER_CELL_TYPES:
            means = [
                results[stage]["synchronous_events"]["by_celltype"][cell_type] / sim_duration
                for stage in DEVELOPMENTAL_STAGES
            ]
            self._plot_trend_with_errorbars(
                axes[0], x_pos, means, CELL_COLORS[cell_type], cell_type
            )

        self._setup_trend_plot_axes(
            axes[0],
            "By Cell Type",
            "LSE / s",
            x_pos,
            event_ylim,
            event_tick_values,
            event_tick_labels,
        )

        # Plot 2: By layer
        for layer in LAYERS:
            means = [
                results[stage]["synchronous_events"]["by_layer"][layer] / sim_duration
                for stage in DEVELOPMENTAL_STAGES
            ]
            self._plot_trend_with_errorbars(axes[1], x_pos, means, LAYER_COLORS[layer], layer)

        self._setup_trend_plot_axes(
            axes[1],
            "By Layer",
            "",
            x_pos,
            event_ylim,
            event_tick_values,
            event_tick_labels,
            show_ylabel=False,
        )

        # Plot 3: Total network
        total_values = [
            results[stage]["synchronous_events"]["total"] / sim_duration
            for stage in DEVELOPMENTAL_STAGES
        ]
        self._plot_trend_with_errorbars(axes[2], x_pos, total_values, "black")

        self._setup_trend_plot_axes(
            axes[2],
            "Total Network",
            "",
            x_pos,
            event_ylim,
            event_tick_values,
            event_tick_labels,
            show_ylabel=False,
        )

        plt.tight_layout()
        self._add_trend_legends(fig, axes)
        self._save_plot("synchronous_event_trends.pdf")

    def _collect_event_values(self, results: dict[str, Any], sim_duration: float) -> list[float]:
        """Collect all event values (as rates) for consistent scaling."""
        all_values = []

        for cell_type in POSTER_CELL_TYPES:
            values = [
                results[stage]["synchronous_events"]["by_celltype"][cell_type] / sim_duration
                for stage in DEVELOPMENTAL_STAGES
            ]
            all_values.extend(values)

        for layer in LAYERS:
            values = [
                results[stage]["synchronous_events"]["by_layer"][layer] / sim_duration
                for stage in DEVELOPMENTAL_STAGES
            ]
            all_values.extend(values)

        total_values = [
            results[stage]["synchronous_events"]["total"] / sim_duration
            for stage in DEVELOPMENTAL_STAGES
        ]
        all_values.extend(total_values)

        return all_values

    def create_structural_ei_balance_trends(self, results: dict[str, Any]) -> None:
        """Create developmental trends in structural E-I balance.

        Shows how inhibition to E cells changes across developmental stages,
        based on connection strengths (not activity).

        Panel 1: Inhibition by cell type (SST vs PV magnitude)
        Panel 2: Total inhibition (sum of SST + PV)
        """
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TRENDS)
        fig.suptitle("Inhibition to E Cells", fontsize=FONT_SIZES["title"], fontweight="bold")

        x_pos = list(range(len(DEVELOPMENTAL_STAGES)))

        # --- Collect all inhibition values for consistent y-axis ---
        inhibitory_types = ["SST", "PV"]
        inhib_values = self._collect_inhibition_values(results)
        
        # Also include total inhibition values
        total_inhib_values = [
            results[stage]["structural_ei_balance"]["inhibition_total"]
            for stage in DEVELOPMENTAL_STAGES
        ]
        all_inhib_values = inhib_values + total_inhib_values

        # Calculate y-axis limits for both panels
        if all_inhib_values:
            max_inhib = max(all_inhib_values)
            y_buffer = max_inhib * 0.1
            inhib_ylim = [-y_buffer * 0.5, max_inhib + y_buffer]
            inhib_tick_values, inhib_tick_labels = self._format_zero_max_ticks(all_inhib_values)
        else:
            inhib_ylim = [0, 1]
            inhib_tick_values = [0, 1]
            inhib_tick_labels = ["0", "1"]

        # --- Panel 1: Inhibition by cell type (SST and PV) ---
        for cell_type in inhibitory_types:
            means = [
                results[stage]["structural_ei_balance"]["by_inhibitory_celltype"][cell_type]
                for stage in DEVELOPMENTAL_STAGES
            ]
            self._plot_trend_with_errorbars(
                axes[0], x_pos, means, CELL_COLORS[cell_type], cell_type
            )

        self._setup_trend_plot_axes(
            axes[0],
            "By Cell Type",
            "Scaled Strength",
            x_pos,
            inhib_ylim,
            inhib_tick_values,
            inhib_tick_labels,
        )

        # --- Panel 2: Total inhibition ---
        self._plot_trend_with_errorbars(axes[1], x_pos, total_inhib_values, "black")

        self._setup_trend_plot_axes(
            axes[1],
            "Total",
            "",
            x_pos,
            inhib_ylim,
            inhib_tick_values,
            inhib_tick_labels,
            show_ylabel=False,
        )

        plt.tight_layout()
        
        # Add legend only for first panel (cell types)
        plt.subplots_adjust(right=0.85)
        fig.legend(
            axes[0].get_lines(),
            [line.get_label() for line in axes[0].get_lines()],
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            title="Cell Types",
        )
        
        self._save_plot("structural_ei_balance_trends.pdf")

    def _collect_inhibition_values(self, results: dict[str, Any]) -> list[float]:
        """Collect all inhibition magnitude values for consistent scaling."""
        all_values = []
        for cell_type in ["SST", "PV"]:
            values = [
                results[stage]["structural_ei_balance"]["by_inhibitory_celltype"][cell_type]
                for stage in DEVELOPMENTAL_STAGES
            ]
            all_values.extend(values)
        return [v for v in all_values if not np.isnan(v) and v != float("inf")]

    def create_functional_ei_balance_trends(self, results: dict[str, Any]) -> None:
        """Create developmental trends in functional E-I balance.

        Shows how activity-weighted inhibition to E cells changes across
        developmental stages (activity × connection strength).

        Panel 1: Inhibition by cell type (SST vs PV magnitude)
        Panel 2: Total inhibition (sum of SST + PV)
        """
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TRENDS)
        fig.suptitle(
            "Functional Inhibition to E Cells", fontsize=FONT_SIZES["title"], fontweight="bold"
        )

        x_pos = list(range(len(DEVELOPMENTAL_STAGES)))

        # --- Collect all functional inhibition values for consistent y-axis ---
        inhibitory_types = ["SST", "PV"]
        functional_inhib_values = []

        for cell_type in inhibitory_types:
            values = [
                results[stage]["functional_ei_balance"]["by_inhibitory_celltype"][cell_type][
                    "mean"
                ]
                for stage in DEVELOPMENTAL_STAGES
            ]
            functional_inhib_values.extend(values)

        # Also include total inhibition values
        total_functional_inhib_values = [
            results[stage]["functional_ei_balance"]["total"]["mean"]
            for stage in DEVELOPMENTAL_STAGES
        ]
        all_functional_inhib_values = functional_inhib_values + total_functional_inhib_values

        # Calculate y-axis limits for both panels
        if all_functional_inhib_values:
            max_inhib = max(all_functional_inhib_values)
            y_buffer = max_inhib * 0.1
            inhib_ylim = [-y_buffer * 0.5, max_inhib + y_buffer]
            inhib_tick_values, inhib_tick_labels = self._format_zero_max_ticks(
                all_functional_inhib_values
            )
        else:
            inhib_ylim = [0, 1]
            inhib_tick_values = [0, 1]
            inhib_tick_labels = ["0", "1"]

        # --- Panel 1: Inhibition by cell type (SST and PV) ---
        for cell_type in inhibitory_types:
            means = [
                results[stage]["functional_ei_balance"]["by_inhibitory_celltype"][cell_type][
                    "mean"
                ]
                for stage in DEVELOPMENTAL_STAGES
            ]
            # Note: We have std available but using SEM_FACTOR for consistency with other plots
            self._plot_trend_with_errorbars(
                axes[0], x_pos, means, CELL_COLORS[cell_type], cell_type
            )

        self._setup_trend_plot_axes(
            axes[0],
            "By Cell Type",
            "Activity × Strength",
            x_pos,
            inhib_ylim,
            inhib_tick_values,
            inhib_tick_labels,
        )

        # --- Panel 2: Total inhibition ---
        self._plot_trend_with_errorbars(axes[1], x_pos, total_functional_inhib_values, "black")

        self._setup_trend_plot_axes(
            axes[1],
            "Total",
            "",
            x_pos,
            inhib_ylim,
            inhib_tick_values,
            inhib_tick_labels,
            show_ylabel=False,
        )

        plt.tight_layout()

        # Add legend only for first panel (cell types)
        plt.subplots_adjust(right=0.85)
        fig.legend(
            axes[0].get_lines(),
            [line.get_label() for line in axes[0].get_lines()],
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            title="Cell Types",
        )

        self._save_plot("functional_ei_balance_trends.pdf")

    def create_spatial_correlation_curves(self, results: dict[str, Any]) -> None:
        """Create C(r) spatial correlation curves for all developmental stages.

        Shows how correlation between spatial locations decays with distance,
        organized as a 4x4 grid (cell types + total × layers + total) with all 4 stages overlaid.
        Uses cell type colors with stage-specific opacity. Distance limited to 1000 μm.
        """
        # 4x4 grid: 3 cell types + total row, 3 layers + total column
        # Make wider than tall for better aspect ratio
        fig, axes = plt.subplots(4, 4, figsize=(12, 7))
        fig.suptitle(
            "Spatial Correlation C(r)", fontsize=FONT_SIZES["title"], fontweight="bold"
        )

        # Define stage opacity (alpha) for distinguishing curves
        # Increasing opacity with developmental stage
        stage_alphas = {
            "P0": 0.3,
            "P5": 0.5,
            "P10": 0.7,
            "P15": 1.0,
        }

        # Distance limit
        max_dist_um = 1000.0

        # Collect all correlation values for consistent y-axis (within distance limit)
        all_corr_values = []
        for stage in DEVELOPMENTAL_STAGES:
            for layer in LAYERS:
                for cell_type in POSTER_CELL_TYPES:
                    corr_data = results[stage]["spatial_correlations"][layer][cell_type]
                    # Use all correlations for y-axis scaling (xlim will handle display clipping)
                    valid_corr = corr_data["correlations"][~np.isnan(corr_data["correlations"])]
                    all_corr_values.extend(valid_corr)

        # Calculate y-axis limits
        if all_corr_values:
            min_corr = min(all_corr_values)
            max_corr = max(all_corr_values)
            y_buffer = (max_corr - min_corr) * 0.1 if max_corr > min_corr else 0.1
            corr_ylim = [min(0, min_corr - y_buffer), max_corr + y_buffer]
        else:
            corr_ylim = [-0.1, 1.0]

        dist_xlim = [0, max_dist_um]

        # Plot individual cell type × layer combinations
        for row, cell_type in enumerate(POSTER_CELL_TYPES):
            cell_color = CELL_COLORS[cell_type]
            
            for col, layer in enumerate(LAYERS):
                ax = axes[row, col]

                # Plot each stage with cell type color and stage-specific opacity
                # Plot full data - xlim will clip the display to max_dist_um
                for stage in DEVELOPMENTAL_STAGES:
                    corr_data = results[stage]["spatial_correlations"][layer][cell_type]
                    ax.plot(
                        corr_data["distances_um"],
                        corr_data["correlations"],
                        color=cell_color,
                        alpha=stage_alphas[stage],
                        linestyle="-",
                        linewidth=LINE_WIDTH,
                        label=stage if row == 0 and col == 0 else "",
                    )

                ax.set_xlim(dist_xlim)
                ax.set_ylim(corr_ylim)

                # Row labels (cell types) on left column
                if col == 0:
                    ax.set_ylabel(
                        f"{cell_type}",
                        fontweight="bold",
                        rotation=0,
                        ha="right",
                        va="center",
                        color=cell_color,
                    )

                # Column headers (layers) on top row
                if row == 0:
                    ax.set_title(layer, fontweight="bold")

                # X-axis labels only on bottom row
                if row == 3:
                    ax.set_xlabel("Distance (μm)")
                else:
                    ax.set_xticks([])

                # Y-axis ticks only on left column
                if col != 0:
                    ax.set_yticklabels([])
                    ax.tick_params(left=False)

        # Fourth column: Average across layers for each cell type
        for row, cell_type in enumerate(POSTER_CELL_TYPES):
            cell_color = CELL_COLORS[cell_type]
            ax = axes[row, 3]

            for stage in DEVELOPMENTAL_STAGES:
                # Average across layers - plot full data, xlim clips display
                distances = None
                corr_list = []
                for layer in LAYERS:
                    corr_data = results[stage]["spatial_correlations"][layer][cell_type]
                    if distances is None:
                        distances = corr_data["distances_um"]
                    corr_list.append(corr_data["correlations"])
                
                avg_corr = np.nanmean(corr_list, axis=0)
                ax.plot(
                    distances,
                    avg_corr,
                    color=cell_color,
                    alpha=stage_alphas[stage],
                    linestyle="-",
                    linewidth=LINE_WIDTH,
                )

            ax.set_xlim(dist_xlim)
            ax.set_ylim(corr_ylim)
            ax.set_yticklabels([])
            ax.tick_params(left=False)
            
            if row == 0:
                ax.set_title("Total", fontweight="bold")
            if row == 3:
                ax.set_xlabel("Distance (μm)")
            else:
                ax.set_xticks([])

        # Fourth row: Average across cell types for each layer
        for col, layer in enumerate(LAYERS):
            ax = axes[3, col]

            for stage in DEVELOPMENTAL_STAGES:
                # Average across cell types - plot full data, xlim clips display
                distances = None
                corr_list = []
                for cell_type in POSTER_CELL_TYPES:
                    corr_data = results[stage]["spatial_correlations"][layer][cell_type]
                    if distances is None:
                        distances = corr_data["distances_um"]
                    corr_list.append(corr_data["correlations"])
                
                avg_corr = np.nanmean(corr_list, axis=0)
                ax.plot(
                    distances,
                    avg_corr,
                    color="black",
                    alpha=stage_alphas[stage],
                    linestyle="-",
                    linewidth=LINE_WIDTH,
                )

            ax.set_xlim(dist_xlim)
            ax.set_ylim(corr_ylim)
            ax.set_xlabel("Distance (μm)")
            
            if col == 0:
                ax.set_ylabel(
                    "Total",
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    color="black",
                )
            else:
                ax.set_yticklabels([])
                ax.tick_params(left=False)

        # Bottom-right: Average across all cell types and layers
        ax = axes[3, 3]
        for stage in DEVELOPMENTAL_STAGES:
            # Plot full data, xlim clips display
            distances = None
            corr_list = []
            for layer in LAYERS:
                for cell_type in POSTER_CELL_TYPES:
                    corr_data = results[stage]["spatial_correlations"][layer][cell_type]
                    if distances is None:
                        distances = corr_data["distances_um"]
                    corr_list.append(corr_data["correlations"])
            
            avg_corr = np.nanmean(corr_list, axis=0)
            ax.plot(
                distances,
                avg_corr,
                color="black",
                alpha=stage_alphas[stage],
                linestyle="-",
                linewidth=LINE_WIDTH,
            )

        ax.set_xlim(dist_xlim)
        ax.set_ylim(corr_ylim)
        ax.set_xlabel("Distance (μm)")
        ax.set_yticklabels([])
        ax.tick_params(left=False)

        # Add legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.88)
        # Get handles from first subplot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.90, 0.5),
            title="Stage",
        )

        self._save_plot("spatial_correlation_curves.pdf")

    def create_correlation_length_trends(self, results: dict[str, Any]) -> None:
        """Create developmental trends in spatial correlation length ξ.

        Shows how the characteristic spatial scale of correlations changes
        across developmental stages, in the same format as other trend plots.
        """
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRENDS)
        fig.suptitle(
            "Spatial Correlation Length ξ", fontsize=FONT_SIZES["title"], fontweight="bold"
        )

        x_pos = list(range(len(DEVELOPMENTAL_STAGES)))

        # Collect all xi values for consistent y-axis
        all_xi_values = self._collect_correlation_length_values(results)

        # Calculate consistent y-axis limits
        if all_xi_values:
            max_val = max(all_xi_values)
            y_buffer = max_val * 0.1
            xi_ylim = [0, max_val + y_buffer]
            xi_tick_values, xi_tick_labels = self._format_zero_max_ticks(all_xi_values)
        else:
            xi_ylim = [0, 1000]
            xi_tick_values = [0, 1000]
            xi_tick_labels = ["0", "1000"]

        # Plot 1: By cell type (average across layers)
        for cell_type in POSTER_CELL_TYPES:
            means = []
            for stage in DEVELOPMENTAL_STAGES:
                # Average xi across layers for this cell type
                xi_values = [
                    results[stage]["spatial_correlations"][layer][cell_type]["xi_um"]
                    for layer in LAYERS
                ]
                xi_values = [v for v in xi_values if not np.isnan(v)]
                means.append(np.mean(xi_values) if xi_values else np.nan)

            self._plot_trend_with_errorbars(
                axes[0], x_pos, means, CELL_COLORS[cell_type], cell_type
            )

        self._setup_trend_plot_axes(
            axes[0],
            "By Cell Type",
            "ξ (μm)",
            x_pos,
            xi_ylim,
            xi_tick_values,
            xi_tick_labels,
        )

        # Plot 2: By layer (average across cell types)
        for layer in LAYERS:
            means = []
            for stage in DEVELOPMENTAL_STAGES:
                # Average xi across cell types for this layer
                xi_values = [
                    results[stage]["spatial_correlations"][layer][cell_type]["xi_um"]
                    for cell_type in POSTER_CELL_TYPES
                ]
                xi_values = [v for v in xi_values if not np.isnan(v)]
                means.append(np.mean(xi_values) if xi_values else np.nan)

            self._plot_trend_with_errorbars(axes[1], x_pos, means, LAYER_COLORS[layer], layer)

        self._setup_trend_plot_axes(
            axes[1],
            "By Layer",
            "",
            x_pos,
            xi_ylim,
            xi_tick_values,
            xi_tick_labels,
            show_ylabel=False,
        )

        # Plot 3: Total network (average across all)
        total_values = []
        for stage in DEVELOPMENTAL_STAGES:
            xi_values = [
                results[stage]["spatial_correlations"][layer][cell_type]["xi_um"]
                for layer in LAYERS
                for cell_type in POSTER_CELL_TYPES
            ]
            xi_values = [v for v in xi_values if not np.isnan(v)]
            total_values.append(np.mean(xi_values) if xi_values else np.nan)

        self._plot_trend_with_errorbars(axes[2], x_pos, total_values, "black")

        self._setup_trend_plot_axes(
            axes[2],
            "Total Network",
            "",
            x_pos,
            xi_ylim,
            xi_tick_values,
            xi_tick_labels,
            show_ylabel=False,
        )

        plt.tight_layout()
        self._add_trend_legends(fig, axes)
        self._save_plot("correlation_length_trends.pdf")

    def _collect_correlation_length_values(self, results: dict[str, Any]) -> list[float]:
        """Collect all correlation length values for consistent scaling (excluding NaN)."""
        all_values = []

        for stage in DEVELOPMENTAL_STAGES:
            for layer in LAYERS:
                for cell_type in POSTER_CELL_TYPES:
                    xi = results[stage]["spatial_correlations"][layer][cell_type]["xi_um"]
                    if not np.isnan(xi):
                        all_values.append(xi)

        return all_values

    def _save_plot(self, filename: str) -> None:
        """Save plot with consistent settings."""
        # Ensure .pdf extension
        if not filename.endswith(".pdf"):
            filename = filename.replace(".svg", ".pdf")
        filepath = os.path.join(self.output_dir, filename)
        fig = plt.gcf()  # Get current figure
        save_figure(fig, filepath, dpi=DPI)

    def generate_all_plots(self, results: dict[str, Any]) -> None:
        """Generate all visualization plots."""
        print("Generating all descriptive analysis plots...")

        # List of plot functions to execute with progress bar
        plot_functions = [
            ("Activity heatmaps", self.create_activity_heatmaps),
            ("Average firing rates", self.create_average_firing_rate_plots),
            ("Active cell fractions", self.create_active_cell_fraction_plots),
            ("Correlation trends", self.create_pairwise_correlation_trends),
            ("Dimensionality trends", self.create_dimensionality_trends),
            ("Synchronous events", self.create_synchronous_event_trends),
            ("Structural E-I balance", self.create_structural_ei_balance_trends),
            ("Functional E-I balance", self.create_functional_ei_balance_trends),
            ("Spatial correlation curves", self.create_spatial_correlation_curves),
            ("Correlation length trends", self.create_correlation_length_trends),
        ]

        for plot_name, plot_func in tqdm(plot_functions, desc="Generating plots", unit="plot"):
            plot_func(results)

        print(f"\nAll plots saved to: {self.output_dir}")
        print("Descriptive analysis visualization complete!")
