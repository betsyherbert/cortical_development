"""Tests for pickle versioning utilities."""

import pytest
import tempfile
import os
import pickle
from pathlib import Path
from src.analysis.utils import save_with_version, load_with_version
from src import PICKLE_FORMAT_VERSION


def test_save_with_version():
    """Test saving with version metadata."""
    test_data = {"key": "value", "number": 42}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    
    try:
        save_with_version(test_data, filepath)
        
        # Load and check structure
        versioned_data = load_with_version(filepath)
        
        assert 'version' in versioned_data
        assert 'package_version' in versioned_data
        assert 'timestamp' in versioned_data
        assert 'data' in versioned_data
        
        assert versioned_data['version'] == PICKLE_FORMAT_VERSION
        assert versioned_data['data'] == test_data
        assert versioned_data['timestamp'] is not None
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_load_with_version():
    """Test loading with version checking."""
    test_data = {"key": "value", "number": 42}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    
    try:
        save_with_version(test_data, filepath)
        versioned_data = load_with_version(filepath)
        
        assert versioned_data['data'] == test_data
        assert versioned_data['version'] == PICKLE_FORMAT_VERSION
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_load_with_version_wrong_major_version():
    """Test loading with wrong major version raises error."""
    test_data = {"key": "value", "number": 42}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    
    try:
        # Save with wrong major version (2.0 instead of 1.0)
        versioned_data = {
            'version': '2.0',  # Wrong major version
            'package_version': '0.1.0',
            'timestamp': '2024-01-01 00:00:00',
            'data': test_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(versioned_data, f)
        
        # Should raise ValueError for major version mismatch
        with pytest.raises(ValueError, match="incompatible"):
            load_with_version(filepath)
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_load_with_version_same_major_different_minor():
    """Test loading with same major but different minor version works."""
    test_data = {"key": "value", "number": 42}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    
    try:
        # Save with same major but different minor version (1.1 instead of 1.0)
        versioned_data = {
            'version': '1.1',  # Same major, different minor
            'package_version': '0.1.0',
            'timestamp': '2024-01-01 00:00:00',
            'data': test_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(versioned_data, f)
        
        # Should work (same major version)
        loaded_data = load_with_version(filepath)
        assert loaded_data['data'] == test_data
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_load_with_version_old_format():
    """Test loading old format (no version metadata) handles gracefully."""
    test_data = {"key": "value", "number": 42}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    
    try:
        # Save in old format (direct pickle, no version wrapper)
        with open(filepath, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Should load but return wrapped structure with None for version fields
        versioned_data = load_with_version(filepath)
        
        assert versioned_data['data'] == test_data
        assert versioned_data['version'] is None
        assert versioned_data['package_version'] is None
        assert versioned_data['timestamp'] is None
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_load_with_version_min_version_check():
    """Test minimum version check."""
    test_data = {"key": "value", "number": 42}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    
    try:
        # Save with old version
        versioned_data = {
            'version': '0.9',  # Below minimum
            'package_version': '0.1.0',
            'timestamp': '2024-01-01 00:00:00',
            'data': test_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(versioned_data, f)
        
        # Should raise ValueError if min_version is too high
        with pytest.raises(ValueError, match="below minimum"):
            load_with_version(filepath, min_version='1.0')
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_save_load_roundtrip():
    """Test save and load roundtrip preserves data."""
    test_data = {
        "nested": {"structure": "test"},
        "numbers": [1, 2, 3],
        "string": "test"
    }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        filepath = f.name
    
    try:
        save_with_version(test_data, filepath)
        versioned_data = load_with_version(filepath)
        
        assert versioned_data['data'] == test_data
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)

