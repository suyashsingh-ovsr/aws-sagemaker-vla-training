#!/usr/bin/env python3
"""
make_annotation_review.py

Reads annotation export parquets and produces a compact, human-readable CSV —
one row per subtask annotation, deduplicated across all episodes.

The annotation export parquets store the full subtask list on EVERY frame
(repeated), which is correct for training but hard to read. This script
collapses that into one row per annotation for easy review.

Accepts either:
  - A single parquet file
  - An annotation export folder (loads all parquet files inside automatically)

Usage:
    # Pass the export folder (recommended — loads all episodes):
    python3 scripts/dataset_build/make_annotation_review.py annotations/exports/andrejarden__air_bag_middle_mixed_box_annotated_20260921_164820 --out annotation_review.csv

    # Or pass a single parquet file:
    python3 scripts/dataset_build/make_annotation_review.py \\
        annotations/exports/.../data/chunk-000/file-000.parquet \\
        --out annotation_review.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def build_review(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-frame, repeated-list columns into one row per
    distinct annotation atom, deduplicated by (episode, style, content,
    timestamp). Adds 'order' (1, 2, 3... within each episode) and
    'end_timestamp' for persistent/subtask-style atoms (when the NEXT
    persistent atom starts, or the episode's last frame if this is the
    last one). Event-style atoms (speech/vqa) fire on a single frame and
    have no end_timestamp."""
    rows = []
    seen = set()

    for col_name in ("language_persistent", "language_events"):
        if col_name not in df.columns:
            continue
        for _, row in df.iterrows():
            atoms = row[col_name]
            if atoms is None or len(atoms) == 0:
                continue
            for atom in atoms:
                key = (
                    int(row["episode_index"]),
                    atom.get("style"),
                    atom.get("content"),
                    round(float(atom.get("timestamp", 0.0)), 6),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "episode_index": int(row["episode_index"]),
                    "timestamp": atom.get("timestamp"),
                    "style": atom.get("style"),
                    "content": atom.get("content"),
                    "_is_persistent": col_name == "language_persistent",
                })

    review = pd.DataFrame(rows)
    if review.empty:
        return review

    review = review.sort_values(["episode_index", "timestamp"]).reset_index(drop=True)

    # order: 1, 2, 3... within each episode, in timestamp order.
    review["order"] = review.groupby("episode_index").cumcount() + 1

    # end_timestamp: only meaningful for persistent/broadcast atoms.
    # For each episode, it's the next persistent atom's start time, or
    # this episode's last recorded frame timestamp if this is the last one.
    episode_end = df.groupby("episode_index")["timestamp"].max().to_dict()
    end_values: list[float | None] = [None] * len(review)
    for ep, group in review[review["_is_persistent"]].groupby("episode_index"):
        idxs = group.index.tolist()
        ts_sorted = group["timestamp"].tolist()
        for i, row_idx in enumerate(idxs):
            if i + 1 < len(idxs):
                end_values[row_idx] = ts_sorted[i + 1]
            else:
                end_values[row_idx] = episode_end.get(ep)
    review["end_timestamp"] = end_values

    review = review.drop(columns=["_is_persistent"])
    review = review[["episode_index", "order", "timestamp", "end_timestamp", "style", "content"]]
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to a parquet file OR an annotation export folder (loads all parquet files)")
    parser.add_argument("--out", default="annotation_review.csv", help="Output CSV path (default: annotation_review.csv)")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Accept either a single parquet file or an export folder (loads all parquets inside)
    if p.is_dir():
        parquets = sorted(p.glob("data/**/*.parquet"))
        if not parquets:
            print(f"No parquet files found in {p}/data/", file=sys.stderr)
            sys.exit(1)
        df = pd.concat([pd.read_parquet(f) for f in parquets], ignore_index=True)
        print(f"Loaded {len(parquets)} parquet file(s) from {p}")
    else:
        df = pd.read_parquet(p)

    review = build_review(df)

    if review.empty:
        print("No annotations found in this file -- nothing to write.")
        sys.exit(0)

    review.to_csv(args.out, index=False)
    print(f"Wrote {len(review)} annotation row(s) to {args.out}")
    print()
    print(review.to_string(index=False))


if __name__ == "__main__":
    main()
