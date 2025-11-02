#!/usr/bin/env python
"""
Wrapper script for inference
"""
import sys
from pathlib import Path

# Add se_st_upgrade to Python path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Now import and run the inference script
from cli.infer import main

if __name__ == "__main__":
    sys.exit(main())
