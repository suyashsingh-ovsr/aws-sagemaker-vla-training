#!/usr/bin/env python3
"""
Merges subtask annotations into the original robot dataset parquet files.
Adds 3 columns to each frame: subtask, language_persistent, language_events.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Max allowed joint angle difference (degrees) between annotation export and original dataset.
# Catches wrong-frame merges (real misalignment = 10–100+ degrees). Allows float rounding.
JOINT_TOLERANCE = 1e-4


def find_latest_export(repo_root: Path, hf_repo: str) -> Path:
    """Find the latest annotation export folder by timestamp in annotations/exports/."""
    slug = hf_repo.replace("/", "__")
    candidates = sorted((repo_root / "annotations" / "exports").glob(f"{slug}_annotated_*"))
    if not candidates:
        raise FileNotFoundError(f"No annotation exports found for {hf_repo}")
    return candidates[-1]


def load_annotation_export(export_dir: Path) -> pd.DataFrame:
    """Load all parquet files from the annotation export folder into one DataFrame."""
    parquets = sorted(export_dir.glob("data/**/*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {export_dir}/data/")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    required = {"episode_index", "frame_index", "observation.state", "language_persistent", "language_events"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation export is missing columns: {missing}")
    print(f"  Loaded annotation export: {len(df)} frames, episodes {sorted(df['episode_index'].unique())}")
    return df


def download_hf_dataset(hf_repo: str, output_dir: Path) -> Path:
    """Download the original dataset from HuggingFace (parquet only, videos skipped)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: run 'pip install huggingface_hub' first", file=sys.stderr)
        sys.exit(1)
    return Path(snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=str(output_dir),
    ))


def load_original_parquets(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    """Load original parquet files as {relative_path: DataFrame} to preserve folder structure."""
    parquet_files = sorted(dataset_dir.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}/data/")
    print(f"  Found {len(parquet_files)} parquet file(s) in original dataset")
    return {str(p.relative_to(dataset_dir)): pd.read_parquet(p) for p in parquet_files}


def verify_joints(orig_df: pd.DataFrame, annot_df: pd.DataFrame, tolerance: float) -> list[dict]:
    """
    Safety check: confirm annotation export references the exact same frames as the original dataset.
    Matches on (episode_index, frame_index), then compares all 6 joint values within tolerance.
    Returns list of mismatched frames (empty = all verified, safe to merge).
    """
    annot_subset = annot_df[["episode_index", "frame_index", "observation.state"]].rename(
        columns={"observation.state": "_annot_state"}
    )
    matched = orig_df[["episode_index", "frame_index", "observation.state"]].merge(
        annot_subset, on=["episode_index", "frame_index"], how="inner"
    )
    mismatches = []
    for _, row in matched.iterrows():
        max_diff = float(np.max(np.abs(
            np.array(row["observation.state"], dtype=np.float64) -
            np.array(row["_annot_state"], dtype=np.float64)
        )))
        if max_diff > tolerance:
            mismatches.append({
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "max_joint_diff": max_diff,
            })
    return mismatches


def active_subtask(language_persistent, frame_timestamp: float) -> str:
    """Return the single active subtask label for a frame based on its timestamp.
    Picks the subtask with the highest start time that is <= frame timestamp."""
    items = list(language_persistent) if language_persistent is not None else []
    subtasks = [a for a in items if a.get("style") == "subtask"]
    candidates = [a for a in subtasks if a.get("timestamp", float("inf")) <= frame_timestamp]
    if not candidates:
        return ""
    return str(max(candidates, key=lambda a: a["timestamp"]).get("content", ""))


def merge_annotations(orig_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join annotation columns onto original dataset by (episode_index, frame_index).
    Unannotated frames get empty lists and empty subtask label."""
    # Drop existing annotation columns if dataset was already annotated
    orig_df = orig_df.drop(columns=[c for c in ["language_persistent", "language_events", "subtask"] if c in orig_df.columns])
    annot_cols = annot_df[["episode_index", "frame_index", "language_persistent", "language_events"]]
    merged = orig_df.merge(annot_cols, on=["episode_index", "frame_index"], how="left")

    def fill_empty(val):
        return [] if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    merged["language_persistent"] = merged["language_persistent"].apply(fill_empty)
    merged["language_events"] = merged["language_events"].apply(fill_empty)
    merged["subtask"] = merged.apply(
        lambda row: active_subtask(row["language_persistent"], float(row["timestamp"])),
        axis=1,
    )
    return merged


def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, default=None,
                        help="Annotation export folder. Default: latest in annotations/exports/")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace dataset repo id (e.g. your_username/your_dataset_repo)")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Path to already-downloaded dataset (skips HF download)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the merged parquet files")
    parser.add_argument("--tolerance", type=float, default=JOINT_TOLERANCE,
                        help=f"Max allowed joint difference for verification (default: {JOINT_TOLERANCE})")
    parser.add_argument("--push", action="store_true",
                        help="Push merged dataset to HuggingFace after writing")
    parser.add_argument("--push-repo", type=str, default=None,
                        help="HF repo to push to (e.g. your_username/your_dataset_repo_annotated). Defaults to --hf-repo")
    args = parser.parse_args()

    if args.hf_repo is None:
        for cfg_path in [repo_root / "scripts" / "dataset_config.yaml", repo_root / "dataset_config.yaml"]:
            if cfg_path.exists():
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text())
                args.hf_repo = f"{cfg['hf_user']}/{cfg['repo']}"
                print(f"Using HF repo from dataset_config.yaml: {args.hf_repo}")
                break
    if args.hf_repo is None and args.dataset_dir is None:
        print("ERROR: provide --hf-repo or --dataset-dir", file=sys.stderr)
        sys.exit(1)

    if args.export_dir is None:
        args.export_dir = find_latest_export(repo_root, args.hf_repo)
    print(f"\nAnnotation export: {args.export_dir}")

    print("\n[1/4] Loading annotation export...")
    annot_df = load_annotation_export(args.export_dir)

    print("\n[2/4] Loading original dataset...")
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        download_dir = args.output_dir.parent / f"_hf_download_{args.hf_repo.replace('/', '__')}"
        download_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {args.hf_repo} to {download_dir}...")
        dataset_dir = download_hf_dataset(args.hf_repo, download_dir)
    orig_parquets = load_original_parquets(dataset_dir)
    all_orig = pd.concat(orig_parquets.values(), ignore_index=True)
    print(f"  {len(all_orig)} total frames across {all_orig['episode_index'].nunique()} episodes")

    print(f"\n[3/4] Verifying joint match (tolerance={args.tolerance})...")
    mismatches = verify_joints(all_orig, annot_df, args.tolerance)
    n_matched = len(all_orig.merge(annot_df[["episode_index", "frame_index"]], on=["episode_index", "frame_index"], how="inner"))
    print(f"  Matched frames: {n_matched} / {len(all_orig)}")
    if mismatches:
        print(f"  ERROR: {len(mismatches)} joint mismatches found — annotations do not match the original data.")
        for m in mismatches[:10]:
            print(f"    episode={m['episode_index']} frame={m['frame_index']} max_diff={m['max_joint_diff']:.6f}")
        print("  Aborting. Fix the mismatches before merging.")
        sys.exit(1)
    print(f"  OK — all {n_matched} frames verified. Joint values are identical.")

    print(f"\n[4/4] Merging and writing to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_src = dataset_dir / "meta"
    if meta_src.exists():
        meta_dst = args.output_dir / "meta"
        if meta_dst.exists():
            shutil.rmtree(meta_dst)
        shutil.copytree(meta_src, meta_dst)
        print("  Copied meta/ from original dataset.")

    videos_src = dataset_dir / "videos"
    if videos_src.exists():
        videos_dst = args.output_dir / "videos"
        if videos_dst.exists():
            shutil.rmtree(videos_dst)
        shutil.copytree(videos_src, videos_dst)
        print("  Copied videos/ from original dataset.")

    total_written = 0
    for rel_path, df in orig_parquets.items():
        merged = merge_annotations(df, annot_df)
        out_path = args.output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)
        n_annotated = (merged["language_persistent"].apply(len) > 0).sum()
        print(f"  {rel_path}: {len(merged)} rows, {n_annotated} annotated frames")
        total_written += len(merged)

    print(f"\nDone. {total_written} rows written to {args.output_dir}/")
    print(f"Annotated episodes: {sorted(annot_df['episode_index'].unique())}")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        push_repo = args.push_repo or args.hf_repo
        api.create_repo(repo_id=push_repo, repo_type="dataset", exist_ok=True)
        print(f"\nPushing to https://huggingface.co/datasets/{push_repo} ...")
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=push_repo,
            repo_type="dataset",
            allow_patterns=["data/**/*.parquet", "meta/**", "videos/**"],
        )
        print(f"Done. https://huggingface.co/datasets/{push_repo}")


if __name__ == "__main__":
    main()
