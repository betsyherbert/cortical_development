"""Lightweight timing utilities for profiling analyses.

Use these to measure execution time of analysis steps without heavy dependencies.
No external profiling libraries required.
"""

import time
from contextlib import contextmanager
from functools import wraps


class Timer:
    """Simple timer for measuring execution time.

    Example:
        >>> timer = Timer()
        >>> timer.start('loading')
        >>> # ... do loading ...
        >>> timer.stop('loading')
        >>> timer.start('processing')
        >>> # ... do processing ...
        >>> timer.stop('processing')
        >>> timer.summary()
        loading: 1.23s
        processing: 4.56s
        Total: 5.79s
    """

    def __init__(self):
        self._start_times: dict[str, float] = {}
        self._durations: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def start(self, name: str) -> None:
        """Start timing a named section."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a named section and return duration in seconds."""
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was never started")

        duration = time.perf_counter() - self._start_times[name]

        if name in self._durations:
            self._durations[name] += duration
            self._counts[name] += 1
        else:
            self._durations[name] = duration
            self._counts[name] = 1

        del self._start_times[name]
        return duration

    def get(self, name: str) -> float:
        """Get total duration for a named section in seconds."""
        return self._durations.get(name, 0.0)

    def get_count(self, name: str) -> int:
        """Get number of times a section was timed."""
        return self._counts.get(name, 0)

    def get_average(self, name: str) -> float:
        """Get average duration for a named section in seconds."""
        count = self._counts.get(name, 0)
        if count == 0:
            return 0.0
        return self._durations[name] / count

    def reset(self, name: str | None = None) -> None:
        """Reset timer(s). If name is None, reset all."""
        if name is None:
            self._start_times.clear()
            self._durations.clear()
            self._counts.clear()
        else:
            self._start_times.pop(name, None)
            self._durations.pop(name, None)
            self._counts.pop(name, None)

    def summary(self, sort_by_time: bool = True) -> str:
        """Generate a summary string of all timed sections.

        Args:
            sort_by_time: If True, sort by duration (slowest first)

        Returns:
            Formatted summary string
        """
        if not self._durations:
            return "No timing data recorded."

        items = list(self._durations.items())
        if sort_by_time:
            items.sort(key=lambda x: x[1], reverse=True)

        lines = []
        for name, duration in items:
            count = self._counts[name]
            if count > 1:
                avg = duration / count
                lines.append(f"  {name}: {duration:.3f}s total ({count}x, avg {avg:.3f}s)")
            else:
                lines.append(f"  {name}: {duration:.3f}s")

        total = sum(self._durations.values())
        lines.append(f"  {'─' * 30}")
        lines.append(f"  Total: {total:.3f}s")

        return "\n".join(lines)

    def print_summary(self, sort_by_time: bool = True) -> None:
        """Print timing summary to stdout."""
        print("\nTiming Summary:")
        print(self.summary(sort_by_time))


@contextmanager
def timed_section(name: str, timer: Timer | None = None, print_time: bool = False):
    """Context manager for timing a code section.

    Args:
        name: Name of the section being timed
        timer: Optional Timer instance to record to (creates local if None)
        print_time: If True, print duration when section completes

    Example:
        >>> timer = Timer()
        >>> with timed_section('data_loading', timer):
        ...     data = load_large_dataset()
        >>> with timed_section('processing', timer):
        ...     results = process(data)
        >>> timer.print_summary()
    """
    local_timer = timer or Timer()
    local_timer.start(name)
    try:
        yield local_timer
    finally:
        duration = local_timer.stop(name)
        if print_time:
            print(f"  [{name}] {duration:.3f}s")


def timed(name: str | None = None, timer: Timer | None = None, print_time: bool = False):
    """Decorator for timing function execution.

    Args:
        name: Timer name (defaults to function name)
        timer: Timer instance to record to (must be provided for aggregation)
        print_time: If True, print duration after each call

    Example:
        >>> timer = Timer()
        >>> @timed('expensive_op', timer=timer, print_time=True)
        ... def expensive_operation(x):
        ...     return x ** 2
        >>> expensive_operation(10)
        [expensive_op] 0.001s
        100
    """

    def decorator(func):
        timer_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            local_timer = timer or Timer()
            local_timer.start(timer_name)
            try:
                return func(*args, **kwargs)
            finally:
                duration = local_timer.stop(timer_name)
                if print_time:
                    print(f"  [{timer_name}] {duration:.3f}s")

        return wrapper

    return decorator
