#!/usr/bin/env python
"""
Wrapper script to ensure se_st_upgrade is in Python path before importing
"""
import sys
from pathlib import Path

# Add se_st_upgrade to Python path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Now import and run the training script
from cli.train import main

if __name__ == "__main__":
    sys.exit(main())
