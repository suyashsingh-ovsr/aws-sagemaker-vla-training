#!/usr/bin/env python3
"""View a parquet file as CSV or interactive table."""

# to see the parquet file
'''  python3 scripts/view_parquet.py merged_dataset/data/chunk-000/file-000.parquet
     python3 scripts/view_parquet.py merged_dataset/data/chunk-000/file-001.parquet   

export to CSV and open in spreadsheet:   

   python3 scripts/view_parquet.py merged_dataset/data/chunk-000/file-000.parquet \
  --csv /tmp/check.csv --rows 0   

  python3 scripts/view_parquet.py merged_dataset/data/chunk-000/file-001.parquet \
  --csv /tmp/check_1.csv --rows 0  '''

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="View parquet file")
    parser.add_argument("path", help="Path to .parquet file")
    parser.add_argument("--csv", help="Save to CSV file instead of printing")
    parser.add_argument("--rows", type=int, default=50, help="Max rows to print (default 50, 0=all)")
    parser.add_argument("--cols", nargs="+", help="Show only these columns")
    args = parser.parse_args()

    df = pd.read_parquet(args.path)

    if args.cols:
        df = df[args.cols]

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved {len(df)} rows to {args.csv}")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 80)

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")

    if args.rows == 0:
        print(df.to_string())
    else:
        print(df.head(args.rows).to_string())
        if len(df) > args.rows:
            print(f"\n... {len(df) - args.rows} more rows (use --rows 0 for all)")


if __name__ == "__main__":
    main()
