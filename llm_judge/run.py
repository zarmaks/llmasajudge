#!/usr/bin/env python3
"""Run script for LLM-as-a-Judge."""

import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli import main

if __name__ == "__main__":
    main()
