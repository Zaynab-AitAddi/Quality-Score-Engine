#!/usr/bin/env python3
"""Show the generated feature table (parquet) in a human-friendly way."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quality_engine.config import PROJECT_ROOT


def main(path: Path | None = None) -> None:
    p = path or (PROJECT_ROOT / "output" / "features.parquet")
    if not p.exists():
        print(f"No feature table found at {p}")
        return
    df = pd.read_parquet(p)
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Show generated features.parquet")
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()
    main(args.path)
