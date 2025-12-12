"""Utility functions for the cortical simulation dashboard.

This module contains small shared helpers used across dashboard callbacks.
Keep utilities minimal - only add here if code is duplicated 2+ times.
"""

import json

import dash


def get_triggered_id() -> str | None:
    """Get the ID of the component that triggered the callback.

    Returns:
        The triggered component ID string, or None if no trigger
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    return ctx.triggered[0]["prop_id"]


def get_triggered_value():
    """Get the value that triggered the callback.

    Returns:
        The triggered value, or None if no trigger
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    return ctx.triggered[0]["value"]


def parse_pattern_match_id(prop_id: str) -> dict | None:
    """Parse a pattern-matching callback ID from the prop_id string.

    Args:
        prop_id: The prop_id string from callback context (e.g., '{"type":"cell","id":"L4_E"}.n_clicks')

    Returns:
        Dictionary with the ID components, or None if parse fails
    """
    try:
        # Extract the JSON part before the property name
        json_str = prop_id.split(".")[0]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError, AttributeError):
        return None


def is_valid_click(triggered_value) -> bool:
    """Check if a click event is valid (not None or 0).

    Args:
        triggered_value: The value from callback trigger

    Returns:
        True if this is a real click event
    """
    return triggered_value is not None and triggered_value != 0


def no_update_tuple(n: int):
    """Return a tuple of n dash.no_update values.

    Args:
        n: Number of no_update values to return

    Returns:
        Tuple of dash.no_update values
    """
    return tuple(dash.no_update for _ in range(n))


def parse_connection_cell_id(cell_id_str: str) -> tuple[str, str | None, str, str] | None:
    """Parse a connection cell ID into components.

    Args:
        cell_id_str: Cell ID string like "L4-E-L5-SST" or "Th-None-L4-E"

    Returns:
        Tuple of (source_layer, source_cell, target_layer, target_cell) or None if invalid
    """
    parts = cell_id_str.split("-")
    if len(parts) < 4:
        return None

    source_layer = parts[0]
    source_cell = parts[1] if parts[1] != "None" else None
    target_layer = parts[2]
    target_cell = parts[3]

    return source_layer, source_cell, target_layer, target_cell


def format_population_title(selected_pops: list[str] | None) -> str:
    """Format a title suffix showing selected populations.

    Args:
        selected_pops: List of population IDs or None for full network

    Returns:
        Title string (e.g., " full network" or ": L23_E + L4_SST")
    """
    if selected_pops is None or len(selected_pops) == 9:
        return " full network"
    elif len(selected_pops) == 0:
        return ""
    elif len(selected_pops) <= 4:
        return ": " + " + ".join(selected_pops)
    else:
        return f" {len(selected_pops)} populations"

