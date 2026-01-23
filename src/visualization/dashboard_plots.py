"""Plot creation helpers for the cortical simulation dashboard.

This module contains pure figure-creation functions that take data as input
and return Plotly figures. No simulation state dependencies.

The DashboardApp class calls these helpers from its callbacks.
"""

import numpy as np
import plotly.graph_objects as go

from src.model.config import COLORMAPS

from .dashboard_layout import AXIS_FONT_SIZE, GRAPH_LAYOUT, SUBTITLE_FONT_SIZE

# =============================================================================
# Heatmap Scaling Constants
# =============================================================================
# Import vmax from descriptive analysis for consistency across dashboard and analysis
from src.analysis.descriptive.config import HEATMAP_VMAX

HEATMAP_ZMIN = 0.0
HEATMAP_ZMAX = HEATMAP_VMAX  # Use same scale as descriptive analysis (0.5)


# =============================================================================
# Pure Figure Creation Functions
# =============================================================================


def create_heatmap_figure(
    data: np.ndarray,
    cell_type: str,
    zmin: float = HEATMAP_ZMIN,
    zmax: float = HEATMAP_ZMAX,
) -> go.Figure:
    """Create a heatmap figure for neural activity data.

    Args:
        data: 2D array of activity values
        cell_type: Cell type for colorscale selection ('E', 'SST', 'PV')
        zmin: Minimum value for color scale
        zmax: Maximum value for color scale

    Returns:
        Plotly Figure with heatmap
    """
    colorscale = COLORMAPS.get(cell_type, [[0, "black"], [1, "white"]])

    return go.Figure(
        data=[
            go.Heatmap(
                z=data,
                colorscale=colorscale,
                showscale=False,
                hoverinfo="skip",
                zmin=zmin,
                zmax=zmax,
            )
        ],
        layout=GRAPH_LAYOUT,
    )


def create_thalamus_heatmap_figure(data: np.ndarray) -> go.Figure:
    """Create a heatmap figure for thalamic input.

    Args:
        data: 2D array of thalamic input values

    Returns:
        Plotly Figure with grayscale heatmap
    """
    return go.Figure(
        data=[
            go.Heatmap(
                z=data,
                colorscale=[[0, "black"], [1, "white"]],
                showscale=False,
                hoverinfo="skip",
                zmin=0,
                zmax=1,
            )
        ],
        layout=GRAPH_LAYOUT,
    )


def create_stability_spectrum_figure(
    k_values: np.ndarray,
    max_real_values: np.ndarray,
    anatomical_grid_size: float,
    n_modes: int,
) -> go.Figure:
    """Create Plotly figure for stability spectrum (max Re(λ) vs wavelength).

    Args:
        k_values: Array of k values (mode numbers, dimensionless)
        max_real_values: Array of max real eigenvalues for each k
        anatomical_grid_size: Anatomical grid size in μm
        n_modes: Number of Fourier modes

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    highlight_data = None

    if len(k_values) > 0 and len(max_real_values) > 0:
        # Filter out k=0 to avoid division by zero
        nonzero_mask = k_values > 0
        k_values_nonzero = k_values[nonzero_mask]
        max_real_values_nonzero = max_real_values[nonzero_mask]

        if len(k_values_nonzero) > 0:
            wavelength_values = anatomical_grid_size / k_values_nonzero
            max_real_finite = max_real_values_nonzero

            # Main spectrum line
            fig.add_trace(
                go.Scatter(
                    x=wavelength_values,
                    y=max_real_finite,
                    mode="lines",
                    name="max Re(λ)",
                    line=dict(color="#2c3e50", width=2),
                    hovertemplate="L=%{x:.0f} μm<br>max Re(λ)=%{y:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

            # Stability boundary at y=0
            fig.add_trace(
                go.Scatter(
                    x=[wavelength_values.min(), wavelength_values.max()],
                    y=[0, 0],
                    mode="lines",
                    name="Stability boundary",
                    line=dict(color="gray", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Y-axis range
            y_min = min(max_real_finite.min(), -0.5)
            y_max = max(max_real_finite.max(), 0.5)
            y_range = y_max - y_min
            y_padding = y_range * 0.1

            # Dominant mode highlight
            max_idx = int(np.argmax(max_real_finite))
            max_value = max_real_finite[max_idx]
            if np.sum(np.isclose(max_real_finite, max_value)) == 1:
                highlight_wavelength = wavelength_values[max_idx]
                highlight_color = "#e74c3c" if max_value > 0 else "#7f8c8d"
                highlight_data = (highlight_wavelength, max_value, highlight_color)

            wavelength_min = wavelength_values.min() * 0.9
            wavelength_max = wavelength_values.max() * 1.1
        else:
            y_min, y_max, y_padding = -0.5, 0.5, 0
            wavelength_min = anatomical_grid_size / n_modes
            wavelength_max = anatomical_grid_size
            _add_no_data_annotation(fig, "Network not yet active")
    else:
        y_min, y_max, y_padding = -0.5, 0.5, 0
        wavelength_min = anatomical_grid_size / n_modes
        wavelength_max = anatomical_grid_size
        _add_no_data_annotation(fig, "Network not yet active")

    # Add highlight marker
    if highlight_data is not None:
        highlight_wavelength, highlight_val, highlight_color = highlight_data
        fig.add_trace(
            go.Scatter(
                x=[highlight_wavelength],
                y=[highlight_val],
                mode="markers",
                marker=dict(
                    size=11,
                    color=highlight_color,
                    symbol="star",
                    line=dict(color="#ffffff", width=1),
                ),
                hovertemplate="Dominant L=%{x:.0f} μm<br>max Re(λ)=%{y:.3f}<extra></extra>",
                showlegend=False,
                cliponaxis=False,
            )
        )

    fig.update_layout(
        title=dict(text="Stability Spectrum", x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(text="Wavelength (μm)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=False,
            range=[wavelength_min, wavelength_max],
        ),
        yaxis=dict(
            title=dict(text="max Re(λ)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            range=[y_min - y_padding, y_max + y_padding] if len(k_values) > 0 else [-0.5, 0.5],
        ),
        margin=dict(l=50, r=25, t=35, b=40),
        height=280,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=False,
    )

    return fig


def create_eigenvalue_spectrum_figure(
    eigenvalues: np.ndarray,
    k_max: float,
) -> go.Figure:
    """Create Plotly figure for eigenvalue spectrum in the complex plane.

    Args:
        eigenvalues: Complex array of eigenvalues to plot
        k_max: The k value (mode number) at which these eigenvalues were computed

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if len(eigenvalues) > 0:
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag

        fig.add_trace(
            go.Scatter(
                x=real_parts,
                y=imag_parts,
                mode="markers",
                name="Eigenvalues",
                marker=dict(
                    size=8,
                    color=real_parts,
                    colorscale="balance",
                    cmin=-1,
                    cmax=1,
                    line=dict(color="#ffffff", width=0.5),
                ),
                hovertemplate="Re(λ)=%{x:.3f}<br>Im(λ)=%{y:.3f}<extra></extra>",
                showlegend=False,
            )
        )

        # Stability boundary (Re(λ) = 0)
        y_min_abs = min(abs(imag_parts.min()), abs(imag_parts.max()))
        y_range = max(y_min_abs, 1) * 1.2
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[-y_range, y_range],
                mode="lines",
                name="Stability boundary",
                line=dict(color="gray", width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        x_min = min(real_parts.min(), -1)
        x_max = max(real_parts.max(), 0.5)
        x_range = x_max - x_min
        x_padding = x_range * 0.1
    else:
        x_min, x_max, x_padding = -1, 0.5, 0.1
        y_range = 1
        _add_no_data_annotation(fig, "No eigenvalues to display")

    title_text = f"Eigenvalues at k={k_max:.1f}" if k_max > 0 else "Eigenvalue Spectrum"

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(text="Re(λ)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            range=[x_min - x_padding, x_max + x_padding] if len(eigenvalues) > 0 else [-1.5, 0.5],
        ),
        yaxis=dict(
            title=dict(text="Im(λ)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=True,
            zerolinecolor="#e0e0e0",
            zerolinewidth=1,
            range=[-y_range, y_range],
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=50, r=25, t=35, b=40),
        height=280,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=False,
    )

    return fig


def create_static_gain_figure(
    k_values: np.ndarray,
    gain_values: np.ndarray,
    anatomical_grid_size: float,
) -> go.Figure:
    """Create Plotly figure for static gain spectrum G(k) vs wavelength.

    Args:
        k_values: Array of k values (mode numbers, dimensionless)
        gain_values: Array of gain values for each k
        anatomical_grid_size: Anatomical grid size in μm

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if len(k_values) > 0 and len(gain_values) > 0:
        # Filter out k=0 to avoid division by zero
        nonzero_mask = k_values > 0
        k_values_nonzero = k_values[nonzero_mask]
        gain_values_nonzero = gain_values[nonzero_mask]

        if len(k_values_nonzero) > 0:
            wavelength_values = anatomical_grid_size / k_values_nonzero

            fig.add_trace(
                go.Scatter(
                    x=wavelength_values,
                    y=gain_values_nonzero,
                    mode="lines",
                    name="G(k)",
                    line=dict(color="#2c3e50", width=2),
                    hovertemplate="L=%{x:.0f} μm<br>G=%{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )

            # Highlight peak gain
            max_idx = int(np.argmax(gain_values_nonzero))
            max_gain = gain_values_nonzero[max_idx]
            max_wavelength = wavelength_values[max_idx]

            fig.add_trace(
                go.Scatter(
                    x=[max_wavelength],
                    y=[max_gain],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color="#e74c3c",
                        symbol="star",
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="Peak: L=%{x:.0f} μm<br>G=%{y:.2f}<extra></extra>",
                    showlegend=False,
                    cliponaxis=False,
                )
            )

            y_max = max(gain_values_nonzero.max() * 1.1, 1)
            wavelength_min = wavelength_values.min() * 0.9
            wavelength_max = wavelength_values.max() * 1.1
        else:
            y_max = 1
            wavelength_min, wavelength_max = 100, 1000
            _add_no_data_annotation(fig, "Network not yet active")
    else:
        y_max = 1
        wavelength_min, wavelength_max = 100, 1000
        _add_no_data_annotation(fig, "Network not yet active")

    fig.update_layout(
        title=dict(text="Gain Spectrum", x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(text="Wavelength (μm)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=False,
            range=[wavelength_min, wavelength_max],
        ),
        yaxis=dict(
            title=dict(text="Gain G(k)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            zeroline=False,
            range=[0, y_max],
        ),
        margin=dict(l=50, r=25, t=35, b=40),
        height=280,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=False,
    )

    return fig


def create_correlation_time_series_figure(
    items: list[tuple[str, list[float]]],
    color_map: dict[str, str],
    title: str = "Correlation",
    x_label: str = "Time (s)",
    y_label: str = "Correlation",
) -> go.Figure:
    """Create a time series figure for correlation data.

    Args:
        items: List of (name, values) tuples for each series
        color_map: Dictionary mapping names to colors
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for name, values in items:
        if values:
            x_values = list(range(len(values)))
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=values,
                    mode="lines",
                    name=name,
                    line=dict(color=color_map.get(name, "#2c3e50"), width=2),
                    showlegend=True,
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            range=[-0.1, 1.1],
        ),
        margin=dict(l=50, r=25, t=35, b=40),
        height=200,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=AXIS_FONT_SIZE - 2),
        ),
    )

    return fig


def create_event_time_series_figure(
    items: list[tuple[str, list[float]]],
    color_map: dict[str, str],
    title: str = "Events",
    x_label: str = "Time (s)",
    y_label: str = "Events",
) -> go.Figure:
    """Create a time series figure for event count data.

    Args:
        items: List of (name, values) tuples for each series
        color_map: Dictionary mapping names to colors
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    max_y = 1

    for name, values in items:
        if values:
            x_values = list(range(len(values)))
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=values,
                    mode="lines",
                    name=name,
                    line=dict(color=color_map.get(name, "#2c3e50"), width=2),
                    showlegend=True,
                )
            )
            if values:
                max_y = max(max_y, max(values) * 1.1)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=SUBTITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=True,
            gridcolor="#e0e0e0",
            range=[0, max_y],
        ),
        margin=dict(l=50, r=25, t=35, b=40),
        height=200,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=AXIS_FONT_SIZE - 2),
        ),
    )

    return fig


def create_spatiotemporal_gain_figure(
    k_values: np.ndarray,
    omega_values: np.ndarray,
    gain_matrix: np.ndarray,
    anatomical_grid_size: float,
) -> go.Figure:
    """Create Plotly figure for spatiotemporal amplification map A(k,omega).

    Args:
        k_values: Array of spatial frequencies k (mode numbers, dimensionless)
        omega_values: Array of temporal frequencies omega (Hz)
        gain_matrix: 2D array of gain values [k_idx, omega_idx]
        anatomical_grid_size: Anatomical grid size in micrometers

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if len(k_values) > 0 and len(omega_values) > 0 and gain_matrix.size > 0:
        # Filter out k=0 to avoid division by zero
        nonzero_mask = k_values > 0
        k_values_nonzero = k_values[nonzero_mask]
        gain_matrix_nonzero = gain_matrix[nonzero_mask, :]

        if len(k_values_nonzero) > 0:
            wavelength_values = anatomical_grid_size / k_values_nonzero
            # Flip gain matrix since wavelength is inverse of k
            gain_matrix_flipped = np.flipud(gain_matrix_nonzero)
            wavelength_min = wavelength_values.min()
            wavelength_max = wavelength_values.max()

            fig.add_trace(
                go.Heatmap(
                    x=wavelength_values[::-1],  # Reverse so larger wavelengths on left
                    y=omega_values,
                    z=gain_matrix_flipped.T,  # Transpose so wavelength is on x-axis
                    colorscale="Hot",
                    colorbar=dict(
                        title=dict(
                            text="Amplification",
                            side="right",
                            font=dict(size=AXIS_FONT_SIZE),
                        ),
                        tickfont=dict(size=AXIS_FONT_SIZE),
                        len=1.0,
                        thickness=12,
                    ),
                    hovertemplate="L=%{x:.0f} μm<br>ω=%{y:.2f} Hz<br>Gain=%{z:.2f}<extra></extra>",
                )
            )

            wavelength_range = [wavelength_min * 0.9, wavelength_max * 1.1]
        else:
            wavelength_range = [100, 1000]
            _add_no_data_annotation(fig, "Network not yet active")
    else:
        wavelength_range = [100, 1000]
        _add_no_data_annotation(fig, "Network not yet active")

    fig.update_layout(
        title=dict(
            text="Spatiotemporal Gain",
            x=0.5,
            xanchor="center",
            font=dict(size=SUBTITLE_FONT_SIZE),
        ),
        xaxis=dict(
            title=dict(text="Wavelength (μm)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=False,
            range=wavelength_range,
        ),
        yaxis=dict(
            title=dict(text="Temporal freq ω (Hz)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            showgrid=False,
            range=[0, 1],
        ),
        margin=dict(l=50, r=25, t=35, b=40),
        height=280,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


# =============================================================================
# Empty/Initial Figure Creators
# =============================================================================


def create_empty_message_figure(message: str, height: int = 280) -> go.Figure:
    """Create an empty figure with a centered message.

    Args:
        message: Message to display
        height: Figure height in pixels

    Returns:
        Plotly Figure with annotation
    """
    fig = go.Figure()
    _add_no_data_annotation(fig, message)
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height,
        margin=dict(l=50, r=25, t=35, b=40),
    )
    return fig


def create_initial_correlation_figure(
    items: list[str],
    color_map: dict[str, str],
    display_seconds: float = 10.0,
) -> go.Figure:
    """Create an initial correlation line plot figure with placeholder traces.

    Args:
        items: List of item names (e.g., layer names or cell types)
        color_map: Dictionary mapping names to colors
        display_seconds: X-axis range in seconds

    Returns:
        Plotly Figure with empty traces for each item
    """
    fig = go.Figure()
    for item in items:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="lines",
                name=item,
                line=dict(color=color_map.get(item, "#2c3e50"), width=2),
            )
        )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            range=[0, display_seconds],
            showgrid=True,
            gridcolor="rgba(220, 220, 220, 0.5)",
        ),
        yaxis=dict(
            title=dict(text="Correlation", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(220, 220, 220, 0.5)",
        ),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            font=dict(size=AXIS_FONT_SIZE),
        ),
        margin=dict(l=60, r=15, t=15, b=40),
        height=220,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
    )
    return fig


def create_initial_event_figure(
    items: list[str],
    color_map: dict[str, str],
    display_seconds: float = 10.0,
) -> go.Figure:
    """Create an initial event rate line plot figure with placeholder traces.

    Args:
        items: List of item names (e.g., layer names or cell types)
        color_map: Dictionary mapping names to colors
        display_seconds: X-axis range in seconds

    Returns:
        Plotly Figure with empty traces for each item
    """
    fig = go.Figure()
    for item in items:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="lines",
                name=item,
                line=dict(color=color_map.get(item, "#2c3e50"), width=2),
            )
        )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            range=[0, display_seconds],
            showgrid=True,
            gridcolor="rgba(220, 220, 220, 0.5)",
        ),
        yaxis=dict(
            title=dict(text="Events/s", font=dict(size=AXIS_FONT_SIZE)),
            tickfont=dict(size=AXIS_FONT_SIZE),
            range=[0, 5],
            showgrid=True,
            gridcolor="rgba(220, 220, 220, 0.5)",
        ),
        margin=dict(l=60, r=15, t=15, b=40),
        height=220,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# =============================================================================
# Helper Functions
# =============================================================================


def _add_no_data_annotation(fig: go.Figure, message: str) -> None:
    """Add a centered annotation for missing data.

    Args:
        fig: Plotly figure to add annotation to
        message: Message to display
    """
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=SUBTITLE_FONT_SIZE, color="gray"),
    )

