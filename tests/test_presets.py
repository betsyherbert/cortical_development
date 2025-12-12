"""Tests for developmental presets.

Note: All spatial parameters (thalamic_widths, outgoing_widths) are in μm.
"""

import pytest
from src.model.presets import P0_PRESET, P5_PRESET, P10_PRESET, P15_PRESET
from src.model.config import CELL_TYPES, LAYERS, ANATOMICAL_GRID_SIZE


def test_all_presets_exist():
    """Test all expected presets are defined."""
    assert P0_PRESET is not None
    assert P5_PRESET is not None
    assert P10_PRESET is not None
    assert P15_PRESET is not None


def test_preset_structure(preset_name='P0'):
    """Test preset has required structure."""
    presets = {
        'P0': P0_PRESET,
        'P5': P5_PRESET,
        'P10': P10_PRESET,
        'P15': P15_PRESET
    }
    
    preset = presets[preset_name]
    
    # Required top-level keys
    required_keys = [
        'time_constants',
        'thalamic_widths', 'outgoing_widths',
        'strength_scaling', 'thalamic_alpha',
        'connection_strengths', 'background_input'
    ]
    
    for key in required_keys:
        assert key in preset, f"Missing key: {key} in {preset_name}"


def test_preset_time_constants():
    """Test preset time constants are valid."""
    preset = P0_PRESET
    
    assert 'time_constants' in preset
    
    for cell_type in CELL_TYPES:
        assert cell_type in preset['time_constants']
        tau = preset['time_constants'][cell_type]
        assert isinstance(tau, (int, float))
        assert tau > 0  # Time constants should be positive


def test_preset_connection_strengths():
    """Test preset connection strengths structure."""
    preset = P0_PRESET
    
    assert 'connection_strengths' in preset
    
    # Check for at least some connection strengths
    # (exact structure depends on implementation)
    assert len(preset['connection_strengths']) > 0


def test_all_presets_have_same_structure():
    """Test all presets have the same structure."""
    presets = [P0_PRESET, P5_PRESET, P10_PRESET, P15_PRESET]
    
    # Get keys from first preset
    base_keys = set(P0_PRESET.keys())
    
    for preset in presets[1:]:
        preset_keys = set(preset.keys())
        assert preset_keys == base_keys, "Presets should have same structure"


def test_preset_thalamic_alpha():
    """Test preset thalamic alpha is in valid range."""
    presets = {
        'P0': P0_PRESET,
        'P5': P5_PRESET,
        'P10': P10_PRESET,
        'P15': P15_PRESET
    }
    
    for name, preset in presets.items():
        assert 'thalamic_alpha' in preset
        alpha = preset['thalamic_alpha']
        assert isinstance(alpha, (int, float))
        assert 0 <= alpha <= 1  # Alpha should be in [0, 1]


@pytest.mark.parametrize("preset_name", ['P0', 'P5', 'P10', 'P15'])
def test_preset_valid_structure(preset_name):
    """Parametrized test for all presets."""
    presets = {
        'P0': P0_PRESET,
        'P5': P5_PRESET,
        'P10': P10_PRESET,
        'P15': P15_PRESET
    }
    
    preset = presets[preset_name]
    
    # Verify required structure exists
    assert 'time_constants' in preset
    assert 'thalamic_alpha' in preset
    assert 'connection_strengths' in preset
    assert 'strength_scaling' in preset


@pytest.mark.parametrize("preset_name", ['P0', 'P5', 'P10', 'P15'])
def test_preset_spatial_parameters_in_um(preset_name):
    """Test that all spatial parameters are in μm (anatomical units)."""
    presets = {
        'P0': P0_PRESET,
        'P5': P5_PRESET,
        'P10': P10_PRESET,
        'P15': P15_PRESET
    }
    
    preset = presets[preset_name]
    
    # Check thalamic_widths are in reasonable μm range (10-500 μm)
    for cell_type in CELL_TYPES:
        width = preset['thalamic_widths'][cell_type]
        assert 10.0 <= width <= 500.0, f"thalamic_width {cell_type} = {width} should be in μm range [10, 500]"
    
    # Check outgoing_widths are in reasonable μm range (10-500 μm)
    for cell_type in CELL_TYPES:
        width = preset['outgoing_widths'][cell_type]
        assert 10.0 <= width <= 500.0, f"outgoing_width {cell_type} = {width} should be in μm range [10, 500]"


def test_spatial_parameters_scale_correctly():
    """Test that spatial parameters scale correctly with anatomical grid size."""
    # P0 preset should have:
    # - thalamic_widths['E'] = 200.0 μm (2.0 grid units × 100 μm/grid)
    # - outgoing_widths['E'] = 300.0 μm (3.0 grid units × 100 μm/grid)
    
    assert P0_PRESET['thalamic_widths']['E'] == 200.0  # μm
    assert P0_PRESET['outgoing_widths']['E'] == 300.0  # μm
    
    # With default 2000 μm grid and 20 grid points, 1 grid unit = 100 μm
    # So 200 μm = 2 grid units, 300 μm = 3 grid units

