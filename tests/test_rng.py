"""Tests for random number generator management."""

import numpy as np
import pytest

from src.model.config import RANDOM_SEED, get_default_seed, get_rng, seed_random


def test_seed_random_default():
    """Test seed_random with default seed."""
    seed = seed_random()
    assert seed == RANDOM_SEED


def test_seed_random_custom():
    """Test seed_random with custom seed."""
    custom_seed = 42
    seed = seed_random(custom_seed)
    assert seed == custom_seed


def test_seed_random_reproducibility():
    """Test that seed_random produces reproducible results."""
    # First sequence
    seed_random(123)
    values1 = np.random.randn(10)

    # Second sequence with same seed
    seed_random(123)
    values2 = np.random.randn(10)

    # Should be identical
    np.testing.assert_array_equal(values1, values2)


def test_seed_random_different_seeds():
    """Test that different seeds produce different results."""
    seed_random(111)
    values1 = np.random.randn(10)

    seed_random(222)
    values2 = np.random.randn(10)

    # Should be different
    assert not np.allclose(values1, values2)


def test_get_default_seed():
    """Test get_default_seed returns correct value."""
    default_seed = get_default_seed()
    assert default_seed == RANDOM_SEED
    assert isinstance(default_seed, int)


def test_seed_random_returns_seed():
    """Test that seed_random returns the seed that was set."""
    # With default
    seed1 = seed_random()
    assert seed1 == RANDOM_SEED

    # With custom
    seed2 = seed_random(999)
    assert seed2 == 999


def test_get_rng_returns_generator():
    """Test that get_rng returns a numpy Generator."""
    rng = get_rng()
    assert isinstance(rng, np.random.Generator)


def test_get_rng_reproducibility():
    """Test that get_rng with same seed produces reproducible results."""
    rng1 = get_rng(42)
    values1 = rng1.standard_normal(10)

    rng2 = get_rng(42)
    values2 = rng2.standard_normal(10)

    np.testing.assert_array_equal(values1, values2)


def test_get_rng_independent_of_global():
    """Test that get_rng is independent of global np.random state."""
    # Set global seed
    seed_random(100)
    global_val1 = np.random.rand()

    # Get explicit rng with different seed
    rng = get_rng(200)
    rng_val = rng.random()

    # Reset global and get same value
    seed_random(100)
    global_val2 = np.random.rand()

    # Global should reproduce
    assert global_val1 == global_val2

    # rng value should be different from global (different seed)
    assert rng_val != global_val1
