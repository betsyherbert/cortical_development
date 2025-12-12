"""Smoke tests for toy model package structure.

These tests verify that the toy model package is properly structured
and importable. They do NOT test any model behavior (not implemented yet).
"""

import pytest


def test_toy_package_imports():
    """Test that the toy package can be imported."""
    from src import toy

    assert hasattr(toy, "__version__")


def test_toy_version_exists():
    """Test that toy package has a version string."""
    from src.toy import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
