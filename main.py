#!/usr/bin/env python3
"""Repository entry point for LLM-as-a-Judge.

This thin wrapper lets users run the command-line interface from the
repository root using ``python main.py``.  It simply forwards arguments to
``src.cli.main`` while supporting ``--csv`` as an alias for ``--in``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project package importable
sys.path.insert(0, str(Path(__file__).parent / "llm_judge"))

from src.cli import main as cli_main  # type: ignore


def main() -> None:
    """Entry point that normalizes CLI arguments and delegates to ``src.cli``."""
    # Replace ``--csv`` with the ``src.cli`` expected ``--in`` flag
    args = ["--in" if arg == "--csv" else arg for arg in sys.argv[1:]]
    sys.argv = [sys.argv[0]] + args
    cli_main()


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    main()
