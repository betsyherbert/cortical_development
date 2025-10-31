"""Allow running stability analysis as a module with python -m src.analysis.stability_analysis"""

import sys

from src.analysis.stability_analysis.run_analysis import main

if __name__ == "__main__":
    sys.exit(main()) 