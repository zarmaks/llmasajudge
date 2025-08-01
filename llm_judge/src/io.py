from __future__ import annotations

from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# Simple CSV helpers
# -----------------------------------------------------------------------------

def load_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame, preserving column order."""
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
