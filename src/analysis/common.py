"""Shared configuration constants and utilities for all analysis modules."""

import pickle
import time
from pathlib import Path
from typing import Any, Optional, Dict

from src.model.presets import P0_PRESET, P5_PRESET, P10_PRESET, P15_PRESET
from src import PICKLE_FORMAT_VERSION, __version__

# Developmental stages - used by all analysis modules
DEVELOPMENTAL_STAGES = ['P0', 'P5', 'P10', 'P15']
PRESETS = {
    'P0': P0_PRESET,
    'P5': P5_PRESET, 
    'P10': P10_PRESET,
    'P15': P15_PRESET
}

# Statistical visualization constants
ERROR_BAR_ALPHA = 0.2
LINE_WIDTH = 2
MARKER_SIZE = 6
SEM_FACTOR = 0.1

# Figure settings
FIGSIZE_TRENDS = (7, 2.5)
DPI = 300


def save_with_version(obj: Any, filepath: str, format_version: Optional[str] = None) -> None:
    """Save object with version metadata wrapped in a dict.
    
    Args:
        obj: Object to save
        filepath: Path to save file
        format_version: Format version (defaults to PICKLE_FORMAT_VERSION)
    
    The saved structure is:
    {
        'version': str,      # Format version
        'package_version': str,  # Package version
        'timestamp': str,    # ISO timestamp
        'data': Any         # The actual object
    }
    """
    if format_version is None:
        format_version = PICKLE_FORMAT_VERSION
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    versioned_data = {
        'version': format_version,
        'package_version': __version__,
        'timestamp': timestamp,
        'data': obj
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(versioned_data, f)


def load_with_version(filepath: str, min_version: Optional[str] = None) -> Dict[str, Any]:
    """Load object with version checking.
    
    Args:
        filepath: Path to load file from
        min_version: Minimum required format version (optional)
    
    Returns:
        Dictionary with keys:
        - 'version': Format version of loaded file
        - 'package_version': Package version that created the file
        - 'timestamp': Timestamp when file was created
        - 'data': The actual loaded object
    
    Raises:
        ValueError: If file format is incompatible or version check fails
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        try:
            loaded = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load pickle file {filepath}: {e}")
    
    # Check if it's a versioned file (new format)
    if isinstance(loaded, dict) and 'version' in loaded and 'data' in loaded:
        version = loaded['version']
        current_version = PICKLE_FORMAT_VERSION
        
        # Version check: compare major version numbers
        if version != current_version:
            try:
                version_major = int(version.split('.')[0])
                current_major = int(current_version.split('.')[0])
                
                if version_major != current_major:
                    raise ValueError(
                        f"File format version {version} is incompatible with current version {current_version}. "
                        f"Major version mismatch indicates breaking changes. "
                        f"Please re-run the analysis to generate new results."
                    )
                elif version_major == current_major:
                    # Same major version, different minor - should be compatible
                    # but warn about potential differences
                    pass
            except (ValueError, IndexError):
                # Can't parse version, but continue anyway
                pass
        
        # Check minimum version if specified
        if min_version is not None:
            try:
                min_major = int(min_version.split('.')[0])
                file_major = int(version.split('.')[0])
                if file_major < min_major:
                    raise ValueError(
                        f"File format version {version} is below minimum required version {min_version}"
                    )
            except (ValueError, IndexError):
                pass
        
        return loaded
    
    # Old format (no version metadata)
    # Return a wrapped structure for compatibility
    return {
        'version': None,
        'package_version': None,
        'timestamp': None,
        'data': loaded
    }

