"""Smoke tests for the dashboard module.

These tests verify the dashboard can be imported and layout components
can be constructed without running a server or browser.
"""

import pytest


class TestDashboardImports:
    """Test that dashboard modules can be imported without errors."""

    def test_import_dashboard_layout(self):
        """Test that dashboard_layout module imports successfully."""
        from src.visualization import dashboard_layout

        assert dashboard_layout is not None

    def test_import_dashboard_plots(self):
        """Test that dashboard_plots module imports successfully."""
        from src.visualization import dashboard_plots

        assert dashboard_plots is not None

    def test_import_dashboard_compute(self):
        """Test that dashboard_compute module imports successfully."""
        from src.visualization import dashboard_compute

        assert dashboard_compute is not None

    def test_import_dashboard_utils(self):
        """Test that dashboard_utils module imports successfully."""
        from src.visualization import dashboard_utils

        assert dashboard_utils is not None


class TestLayoutConstruction:
    """Test that layout components can be constructed without a server."""

    def test_create_control_panel(self):
        """Test that control panel can be created."""
        from src.visualization.dashboard_layout import create_control_panel

        panel = create_control_panel()
        assert panel is not None
        # Verify it's a Dash Div component
        assert hasattr(panel, "children")

    def test_create_grid_info_boxes(self):
        """Test that grid info boxes can be created."""
        from src.visualization.dashboard_layout import create_grid_info_boxes

        boxes = create_grid_info_boxes()
        assert boxes is not None
        assert hasattr(boxes, "className")
        assert boxes.className == "mb-3"

    def test_create_preset_buttons(self):
        """Test that preset buttons can be created."""
        from src.visualization.dashboard_layout import create_preset_buttons

        buttons = create_preset_buttons()
        assert buttons is not None
        assert hasattr(buttons, "children")

    def test_create_slider(self):
        """Test that individual sliders can be created."""
        from src.visualization.dashboard_layout import create_slider

        slider = create_slider(
            id_prefix="test",
            cell_type="E",
            min_val=0,
            max_val=100,
            step=1,
            initial_value=50,
            marks={0: "0", 50: "50", 100: "100"},
        )
        assert slider is not None
        assert slider.id == "test-e-slider"
        assert slider.min == 0
        assert slider.max == 100
        assert slider.value == 50


class TestPlotConstruction:
    """Test that plot helpers can create figures."""

    def test_create_heatmap_figure(self):
        """Test that heatmap figures can be created."""
        import numpy as np

        from src.visualization.dashboard_plots import create_heatmap_figure

        data = np.random.rand(10, 10)
        fig = create_heatmap_figure(data, cell_type="E")
        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_create_thalamus_heatmap_figure(self):
        """Test that thalamus heatmap figures can be created."""
        import numpy as np

        from src.visualization.dashboard_plots import create_thalamus_heatmap_figure

        data = np.random.rand(10, 10)
        fig = create_thalamus_heatmap_figure(data)
        assert fig is not None
        assert len(fig.data) == 1

    def test_create_stability_spectrum_empty(self):
        """Test stability spectrum figure with empty data."""
        import numpy as np

        from src.visualization.dashboard_plots import create_stability_spectrum_figure

        fig = create_stability_spectrum_figure(
            k_values=np.array([]),
            max_real_values=np.array([]),
            anatomical_grid_size=1000.0,
            n_modes=32,
        )
        assert fig is not None
        # Should have annotation for no data
        assert len(fig.layout.annotations) > 0

    def test_create_eigenvalue_spectrum_empty(self):
        """Test eigenvalue spectrum figure with empty data."""
        import numpy as np

        from src.visualization.dashboard_plots import create_eigenvalue_spectrum_figure

        fig = create_eigenvalue_spectrum_figure(
            eigenvalues=np.array([]),
            k_max=0.0,
        )
        assert fig is not None


class TestComputeHelpers:
    """Test that compute helpers work correctly."""

    def test_compute_group_correlation(self):
        """Test group correlation computation."""
        import numpy as np

        from src.visualization.dashboard_compute import compute_group_correlation

        # Create a simple correlation matrix
        corr_matrix = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )

        # Test with all indices
        result = compute_group_correlation(corr_matrix, np.array([0, 1, 2]))
        assert result > 0
        assert result < 1

        # Test with single index (should return 0)
        result = compute_group_correlation(corr_matrix, np.array([0]))
        assert result == 0.0

    def test_get_unique_k_squared_values(self):
        """Test k-squared value generation."""
        from src.visualization.dashboard_compute import get_unique_k_squared_values

        k_squared = get_unique_k_squared_values(n_modes=4)
        assert 0 in k_squared
        assert 1 in k_squared
        assert 2 in k_squared
        assert 4 in k_squared


class TestUtilityHelpers:
    """Test utility helpers."""

    def test_format_population_title_full_network(self):
        """Test title formatting for full network."""
        from src.visualization.dashboard_utils import format_population_title

        # 9 populations = full network
        result = format_population_title(["L23_E", "L23_SST", "L23_PV", "L4_E", "L4_SST", "L4_PV", "L5_E", "L5_SST", "L5_PV"])
        assert result == " full network"

        # None also means full network
        result = format_population_title(None)
        assert result == " full network"

    def test_format_population_title_subset(self):
        """Test title formatting for population subset."""
        from src.visualization.dashboard_utils import format_population_title

        result = format_population_title(["L4_E", "L4_PV"])
        assert "L4_E" in result
        assert "L4_PV" in result

    def test_format_population_title_empty(self):
        """Test title formatting for empty selection."""
        from src.visualization.dashboard_utils import format_population_title

        result = format_population_title([])
        assert result == ""

    def test_parse_connection_cell_id(self):
        """Test connection cell ID parsing."""
        from src.visualization.dashboard_utils import parse_connection_cell_id

        # Normal connection
        result = parse_connection_cell_id("L4-E-L5-SST")
        assert result == ("L4", "E", "L5", "SST")

        # Thalamic connection
        result = parse_connection_cell_id("Th-None-L4-E")
        assert result == ("Th", None, "L4", "E")

        # Invalid format
        result = parse_connection_cell_id("invalid")
        assert result is None

    def test_is_valid_click(self):
        """Test click validation."""
        from src.visualization.dashboard_utils import is_valid_click

        assert is_valid_click(1) is True
        assert is_valid_click(5) is True
        assert is_valid_click(0) is False
        assert is_valid_click(None) is False

    def test_no_update_tuple(self):
        """Test no_update tuple generation."""
        import dash

        from src.visualization.dashboard_utils import no_update_tuple

        result = no_update_tuple(3)
        assert len(result) == 3
        assert all(r == dash.no_update for r in result)

