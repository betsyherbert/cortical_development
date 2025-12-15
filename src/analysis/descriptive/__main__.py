"""Command-line interface for descriptive analysis.

Usage:
    python -m src.analysis.descriptive
"""

from .run_analysis import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
