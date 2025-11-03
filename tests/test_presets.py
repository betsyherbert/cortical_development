"""Tests for developmental presets."""

import pytest
from src.model.presets import P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET
from src.model.config import CELL_TYPES, LAYERS


def test_all_presets_exist():
    """Test all expected presets are defined."""
    assert P4_PRESET is not None
    assert P8_PRESET is not None
    assert P12_PRESET is not None
    assert P16_PRESET is not None


def test_preset_structure(preset_name='P4'):
    """Test preset has required structure."""
    presets = {
        'P4': P4_PRESET,
        'P8': P8_PRESET,
        'P12': P12_PRESET,
        'P16': P16_PRESET
    }
    
    preset = presets[preset_name]
    
    # Required top-level keys
    required_keys = [
        'time_constants', 'gains', 'noise_params',
        'thalamic_widths', 'outgoing_widths',
        'strength_scaling', 'thalamic_alpha',
        'connection_strengths'
    ]
    
    for key in required_keys:
        assert key in preset, f"Missing key: {key} in {preset_name}"


def test_preset_time_constants():
    """Test preset time constants are valid."""
    preset = P4_PRESET
    
    assert 'time_constants' in preset
    
    for cell_type in CELL_TYPES:
        assert cell_type in preset['time_constants']
        tau = preset['time_constants'][cell_type]
        assert isinstance(tau, (int, float))
        assert tau > 0  # Time constants should be positive


def test_preset_gains():
    """Test preset gains are valid."""
    preset = P4_PRESET
    
    assert 'gains' in preset
    
    for cell_type in CELL_TYPES:
        assert cell_type in preset['gains']
        gain = preset['gains'][cell_type]
        assert isinstance(gain, (int, float))
        assert gain >= 0  # Gains should be non-negative


def test_preset_noise_params():
    """Test preset noise parameters are valid."""
    preset = P4_PRESET
    
    assert 'noise_params' in preset
    
    for cell_type in CELL_TYPES:
        assert cell_type in preset['noise_params']
        noise_params = preset['noise_params'][cell_type]
        
        assert 'mean' in noise_params
        assert 'std' in noise_params
        assert 'c' in noise_params
        
        # Check std is non-negative
        assert noise_params['std'] >= 0
        
        # Check correlation is in valid range
        assert 0 <= noise_params['c'] <= 1


def test_preset_connection_strengths():
    """Test preset connection strengths structure."""
    preset = P4_PRESET
    
    assert 'connection_strengths' in preset
    
    # Check for at least some connection strengths
    # (exact structure depends on implementation)
    assert len(preset['connection_strengths']) > 0


def test_all_presets_have_same_structure():
    """Test all presets have the same structure."""
    presets = [P4_PRESET, P8_PRESET, P12_PRESET, P16_PRESET]
    
    # Get keys from first preset
    base_keys = set(P4_PRESET.keys())
    
    for preset in presets[1:]:
        preset_keys = set(preset.keys())
        assert preset_keys == base_keys, "Presets should have same structure"


def test_preset_thalamic_alpha():
    """Test preset thalamic alpha is in valid range."""
    presets = {
        'P4': P4_PRESET,
        'P8': P8_PRESET,
        'P12': P12_PRESET,
        'P16': P16_PRESET
    }
    
    for name, preset in presets.items():
        assert 'thalamic_alpha' in preset
        alpha = preset['thalamic_alpha']
        assert isinstance(alpha, (int, float))
        assert 0 <= alpha <= 1  # Alpha should be in [0, 1]


@pytest.mark.parametrize("preset_name", ['P4', 'P8', 'P12', 'P16'])
def test_preset_valid_structure(preset_name):
    """Parametrized test for all presets."""
    presets = {
        'P4': P4_PRESET,
        'P8': P8_PRESET,
        'P12': P12_PRESET,
        'P16': P16_PRESET
    }
    
    preset = presets[preset_name]
    
    # Verify required structure exists
    assert 'time_constants' in preset
    assert 'gains' in preset
    assert 'noise_params' in preset
    assert 'thalamic_alpha' in preset

