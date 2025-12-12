"""Tests for the timing utilities."""

import time

import pytest

from src.analysis.timing import Timer, timed, timed_section


def test_timer_basic():
    """Test basic timer start/stop functionality."""
    timer = Timer()
    timer.start("test")
    time.sleep(0.01)  # 10ms
    duration = timer.stop("test")

    assert duration >= 0.01
    assert timer.get("test") >= 0.01
    assert timer.get_count("test") == 1


def test_timer_multiple_calls():
    """Test timer accumulates multiple calls."""
    timer = Timer()

    for _ in range(3):
        timer.start("test")
        time.sleep(0.01)
        timer.stop("test")

    assert timer.get_count("test") == 3
    assert timer.get("test") >= 0.03
    assert timer.get_average("test") >= 0.01


def test_timer_reset():
    """Test timer reset functionality."""
    timer = Timer()
    timer.start("test")
    timer.stop("test")

    timer.reset("test")
    assert timer.get("test") == 0.0
    assert timer.get_count("test") == 0

    timer.start("test2")
    timer.stop("test2")
    timer.reset()  # Reset all
    assert timer.get("test2") == 0.0


def test_timed_section_context_manager():
    """Test timed_section context manager."""
    timer = Timer()

    with timed_section("block", timer):
        time.sleep(0.01)

    assert timer.get("block") >= 0.01


def test_timed_decorator():
    """Test timed decorator."""
    timer = Timer()

    @timed("my_func", timer=timer)
    def my_function():
        time.sleep(0.01)
        return 42

    result = my_function()
    assert result == 42
    assert timer.get("my_func") >= 0.01


def test_timer_summary():
    """Test timer summary output."""
    timer = Timer()
    timer.start("fast")
    timer.stop("fast")
    timer.start("slow")
    time.sleep(0.01)
    timer.stop("slow")

    summary = timer.summary()
    assert "fast" in summary
    assert "slow" in summary
    assert "Total" in summary
