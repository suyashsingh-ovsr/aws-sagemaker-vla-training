#!/usr/bin/env python3
"""
Export all merged parquet files to CSV for local inspection.

Usage:
    python3 scripts/dataset_build/export_merged_to_csv.py
    python3 scripts/dataset_build/export_merged_to_csv.py --merged-dir merged_dataset/ --out-dir merged_csv/
    python3 scripts/dataset_build/export_merged_to_csv.py --cols episode_index frame_index timestamp language_persistent
"""

import argparse
import glob
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, default=Path("merged_dataset"),
                        help="Directory with merged parquet files (default: merged_dataset/)")
    parser.add_argument("--out-dir", type=Path, default=Path("merged_csv"),
                        help="Output directory for CSV files (default: merged_csv/)")
    # Default columns exported to CSV — the 4 columns used for training/review.
    # Override with --cols if you need different columns.
    DEFAULT_COLS = ["episode_index", "frame_index", "timestamp", "subtask"]
    parser.add_argument("--cols", nargs="+", default=DEFAULT_COLS,
                        help=f"Columns to export (default: {DEFAULT_COLS})")
    args = parser.parse_args()

    import pandas as pd

    parquets = sorted(args.merged_dir.glob("data/**/*.parquet"))
    if not parquets:
        print(f"No parquet files found in {args.merged_dir}/data/")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for p in parquets:
        df = pd.read_parquet(p)
        if args.cols:
            missing = [c for c in args.cols if c not in df.columns]
            if missing:
                print(f"  WARNING: columns not found: {missing}")
            df = df[[c for c in args.cols if c in df.columns]]

        rel = p.relative_to(args.merged_dir)
        out_path = args.out_dir / rel.with_suffix(".csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  {rel} -> {out_path}  ({len(df)} rows)")

    print(f"\nDone. CSV files in {args.out_dir}/")


if __name__ == "__main__":
    main()
