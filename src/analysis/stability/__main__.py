"""Allow running stability analysis as a module with python -m src.analysis.stability"""

import sys

from src.analysis.stability.run_analysis import main

if __name__ == "__main__":
    sys.exit(main()) 