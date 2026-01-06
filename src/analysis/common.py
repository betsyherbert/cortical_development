"""Shared configuration constants and utilities for all analysis modules."""

import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src import PICKLE_FORMAT_VERSION, __version__
from src.model.presets import P0_PRESET, P5_PRESET, P10_PRESET, P15_PRESET

# Developmental stages - used by all analysis modules
DEVELOPMENTAL_STAGES = ["P0", "P5", "P10", "P15"]
PRESETS = {"P0": P0_PRESET, "P5": P5_PRESET, "P10": P10_PRESET, "P15": P15_PRESET}

# Statistical visualization constants
ERROR_BAR_ALPHA = 0.2
LINE_WIDTH = 2
MARKER_SIZE = 6
SEM_FACTOR = 0.1

# Figure settings
FIGSIZE_TRENDS = (7, 2.5)  # Deprecated: use mm-based sizing instead
DPI = 300

# =============================================================================
# Figure Saving Utilities
# =============================================================================


def save_figure(fig: plt.Figure, filepath: str | Path, dpi: int = DPI) -> None:
    """Save figure to PDF with publication-quality settings.
    
    Single source of truth for all figure saving. Ensures consistent format,
    resolution, and styling across all analysis modules.
    
    Args:
        fig: Matplotlib figure object to save
        filepath: Path where figure should be saved (will use .pdf extension)
        dpi: Resolution in dots per inch (default: 300)
    
    Side effects:
        Closes the figure after saving
    """
    filepath = Path(filepath)
    # Ensure .pdf extension
    if filepath.suffix != ".pdf":
        filepath = filepath.with_suffix(".pdf")
    
    fig.savefig(
        filepath,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)

# =============================================================================
# Figure Styling (Fonts + Sizes)
# =============================================================================

# Nature-style figure dimensions (double-column width)
DOUBLE_COLUMN_WIDTH_MM = 183.0
"""Standard double-column figure width in millimeters (Nature standard)."""


def mm_to_inches(mm: float) -> float:
    """Convert millimeters to inches for Matplotlib figure sizing.

    Args:
        mm: Size in millimeters

    Returns:
        Size in inches
    """
    return mm / 25.4


def compute_figsize_inches(width_mm: float, height_mm: float | None = None, aspect_ratio: float | None = None) -> tuple[float, float]:
    """Compute figure size in inches from millimeter dimensions.

    Args:
        width_mm: Figure width in millimeters
        height_mm: Figure height in millimeters (if provided)
        aspect_ratio: Height/width ratio (used if height_mm is None)

    Returns:
        Tuple of (width_inches, height_inches)

    Raises:
        ValueError: If neither height_mm nor aspect_ratio is provided
    """
    width_inches = mm_to_inches(width_mm)
    if height_mm is not None:
        height_inches = mm_to_inches(height_mm)
    elif aspect_ratio is not None:
        height_inches = width_inches * aspect_ratio
    else:
        raise ValueError("Must provide either height_mm or aspect_ratio")
    return (width_inches, height_inches)


# Font sizes in points (appropriate for final print size at 183 mm width)
# Nature recommends 5-7 pt for figure text, with panel labels at 8 pt
FIGURE_FONT_SIZES_PT: dict[str, int] = {
    "figure_title": 7,  # Maximum recommended by Nature
    "axes_title": 7,
    "axis_label": 7,
    "tick_label": 6,  # Between 5-7 pt range
    "legend": 6,
    "annotation": 6,
    "colorbar_label": 7,
    "colorbar_tick": 6,
    "panel_label": 8,  # Bold panel labels (a, b, c) per Nature
}
"""Standard font sizes (in points) used across all analysis figures.

These sizes are appropriate for final print size at 183 mm width (double-column).
Nature recommends 5-7 pt for figure text, with panel labels at 8 pt bold.
"""

PREFERRED_SANS_FONTS: list[str] = ["Helvetica", "Arial", "DejaVu Sans"]
"""Preferred sans-serif fonts (Helvetica first for thinner, cleaner appearance).

Note:
    Helvetica provides a thinner weight than DejaVu Sans and includes Greek letters.
    Falls back to Arial, then DejaVu Sans for portability across systems.
"""


def resolve_available_sans_font(preferred_fonts: list[str] | None = None) -> str:
    """Select the first available sans-serif font from a preference list.

    Args:
        preferred_fonts: Ordered list of preferred font family names.
            If None, uses PREFERRED_SANS_FONTS.

    Returns:
        Name of the selected font family (guaranteed to exist).

    Side effects:
        Imports Matplotlib font manager (no global rcParams mutation).
    """
    import matplotlib.font_manager as font_manager

    fonts = preferred_fonts or PREFERRED_SANS_FONTS
    for font_name in fonts:
        try:
            resolved_path = font_manager.findfont(font_name, fallback_to_default=False)
            if resolved_path:
                return font_name
        except Exception:
            continue

    # Guaranteed fallback bundled with Matplotlib
    return "DejaVu Sans"


def apply_matplotlib_style(overrides: dict[str, Any] | None = None) -> None:
    """Apply consistent Matplotlib style (fonts + sizes + lines) for analysis figures.

    This centralizes figure styling so all analysis modules render consistently.
    Uses Helvetica for both text and math (with fallback for missing symbols) to
    ensure consistent rendering of Greek letters and subscripts with a thinner,
    cleaner appearance than DejaVu Sans.

    Font sizes are appropriate for final print size at 183 mm width (double-column).
    Line widths, tick sizes, and marker sizes are standardized for publication quality.

    Args:
        overrides: Optional rcParams overrides applied after defaults.

    Side effects:
        Mutates Matplotlib global rcParams.
    """
    import matplotlib as mpl

    selected_font = resolve_available_sans_font()

    rcparams: dict[str, Any] = {
        # Font family (text) - Helvetica first for thinner appearance
        "font.family": "sans-serif",
        "font.sans-serif": PREFERRED_SANS_FONTS,
        "font.size": FIGURE_FONT_SIZES_PT["tick_label"],
        # Font sizes (key elements) - appropriate for 183 mm final width
        "figure.titlesize": FIGURE_FONT_SIZES_PT["figure_title"],
        "axes.titlesize": FIGURE_FONT_SIZES_PT["axes_title"],
        "axes.labelsize": FIGURE_FONT_SIZES_PT["axis_label"],
        "xtick.labelsize": FIGURE_FONT_SIZES_PT["tick_label"],
        "ytick.labelsize": FIGURE_FONT_SIZES_PT["tick_label"],
        "legend.fontsize": FIGURE_FONT_SIZES_PT["legend"],
        # Math text: use Helvetica (same as text) with fallback only for missing symbols
        "mathtext.fontset": "custom",
        "mathtext.rm": selected_font,  # Regular math = Helvetica
        "mathtext.sf": selected_font,  # Sans-serif math = Helvetica
        "mathtext.it": f"{selected_font}:italic",  # Italic math = Helvetica italic
        "mathtext.bf": f"{selected_font}:bold",  # Bold math = Helvetica bold
        "mathtext.default": "regular",
        "mathtext.fallback": "stixsans",  # Fallback only for symbols Helvetica lacks
        # Line styling (publication quality)
        "lines.linewidth": 1.0,  # Standard line width
        "lines.markersize": 4.0,  # Standard marker size
        # Axes styling
        "axes.linewidth": 0.5,  # Axes spine width
        # Tick styling
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3.0,  # Tick length
        "ytick.major.size": 3.0,
        "xtick.major.pad": 2.0,  # Gap between tick and label (reduced from default ~3.5)
        "ytick.major.pad": 2.0,
        "xtick.color": "0.5",  # Grey tick marks (0.5 = 50% grey)
        "ytick.color": "0.5",
        "xtick.labelcolor": "0.5",  # Grey tick labels (0.5 = 50% grey)
        "ytick.labelcolor": "0.5",
        "xtick.minor.width": 0.25,
        "ytick.minor.width": 0.25,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        # Legend styling (clean, minimal)
        "legend.frameon": False,  # No frame
        "legend.handlelength": 1.5,  # Handle length
        "legend.handletextpad": 0.5,  # Padding between handle and text
        # DPI (consistent output)
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        # Glyph correctness
        "axes.unicode_minus": False,
        # Vector-export friendliness
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    if overrides:
        rcparams.update(overrides)

    mpl.rcParams.update(rcparams)

# =============================================================================
# Output Directory Management
# =============================================================================

# Base output directory for all generated artifacts (untracked by git)
OUTPUTS_BASE = Path("outputs")


def get_output_dir(analysis_name: str, create: bool = True) -> Path:
    """Get the output directory for a specific analysis.

    Args:
        analysis_name: Name of the analysis (e.g., 'bifurcation', 'stability', 'descriptive')
        create: If True, create the directory if it doesn't exist

    Returns:
        Path to the output directory (e.g., outputs/bifurcation/)
    """
    output_dir = OUTPUTS_BASE / analysis_name
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_subdir(analysis_name: str, subdir: str, create: bool = True) -> Path:
    """Get a subdirectory within an analysis output directory.

    Args:
        analysis_name: Name of the analysis (e.g., 'stability')
        subdir: Subdirectory name (e.g., 'snapshots', 'summary')
        create: If True, create the directory if it doesn't exist

    Returns:
        Path to the subdirectory (e.g., outputs/stability/snapshots/)
    """
    subdir_path = get_output_dir(analysis_name, create=create) / subdir
    if create:
        subdir_path.mkdir(parents=True, exist_ok=True)
    return subdir_path


def make_run_metadata(
    seed: int | None = None,
    stages: list | None = None,
    params: dict[str, Any] | None = None,
    **extra,
) -> dict[str, Any]:
    """Create standardized run metadata for saving with results.

    Args:
        seed: Random seed used for the run (None if not applicable)
        stages: Developmental stages analyzed (e.g., ['P0', 'P5', 'P10', 'P15'])
        params: Key analysis parameters (e.g., grid_size, duration)
        **extra: Additional metadata fields

    Returns:
        Dictionary with standardized metadata fields

    Example:
        >>> metadata = make_run_metadata(seed=4, stages=['P0', 'P5'], params={'grid_size': 20})
        >>> save_with_version(results, 'output.pkl', metadata=metadata)
    """
    from src.model.config import RANDOM_SEED

    meta = {
        "seed": seed if seed is not None else RANDOM_SEED,
        "stages": stages,
        "params": params or {},
    }
    meta.update(extra)
    return meta


def save_with_version(
    obj: Any,
    filepath: str,
    format_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save object with version metadata wrapped in a dict.

    Args:
        obj: Object to save
        filepath: Path to save file
        format_version: Format version (defaults to PICKLE_FORMAT_VERSION)
        metadata: Optional run metadata (seed, stages, params, etc.)
                  Use make_run_metadata() to create standardized metadata.

    The saved structure is:
    {
        'version': str,           # Format version
        'package_version': str,   # Package version
        'timestamp': str,         # ISO timestamp
        'metadata': dict | None,  # Run metadata (seed, stages, params, etc.)
        'data': Any               # The actual object
    }
    """
    if format_version is None:
        format_version = PICKLE_FORMAT_VERSION

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    versioned_data = {
        "version": format_version,
        "package_version": __version__,
        "timestamp": timestamp,
        "metadata": metadata,
        "data": obj,
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(versioned_data, f)


def load_with_version(filepath: str, min_version: str | None = None) -> dict[str, Any]:
    """Load object with version checking.

    Args:
        filepath: Path to load file from
        min_version: Minimum required format version (optional)

    Returns:
        Dictionary with keys:
        - 'version': Format version of loaded file
        - 'package_version': Package version that created the file
        - 'timestamp': Timestamp when file was created
        - 'metadata': Run metadata dict (seed, stages, params) or None
        - 'data': The actual loaded object

    Raises:
        ValueError: If file format is incompatible or version check fails
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        try:
            loaded = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load pickle file {filepath}: {e}")

    # Check if it's a versioned file (new format)
    if isinstance(loaded, dict) and "version" in loaded and "data" in loaded:
        version = loaded["version"]
        current_version = PICKLE_FORMAT_VERSION

        # Version check: compare major version numbers
        if version != current_version:
            try:
                version_major = int(version.split(".")[0])
                current_major = int(current_version.split(".")[0])

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
                min_major = int(min_version.split(".")[0])
                file_major = int(version.split(".")[0])
                if file_major < min_major:
                    raise ValueError(
                        f"File format version {version} is below minimum required version {min_version}"
                    )
            except (ValueError, IndexError):
                pass

        # Ensure metadata key exists (backward compatibility)
        if "metadata" not in loaded:
            loaded["metadata"] = None

        return loaded

    # Old format (no version metadata)
    # Return a wrapped structure for compatibility
    return {
        "version": None,
        "package_version": None,
        "timestamp": None,
        "metadata": None,
        "data": loaded,
    }
