#!/usr/bin/env python3
"""Validate a generated feature table against the Project A contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from quality_engine.validation import validate_feature_table_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a parquet feature table")
    parser.add_argument("feature_table", type=Path)
    args = parser.parse_args()

    result = validate_feature_table_file(args.feature_table)
    print(result)
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
