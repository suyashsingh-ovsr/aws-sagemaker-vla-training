#!/usr/bin/env python3
"""
Export rollout annotation analysis to Excel.

Reads the latest annotation export + merged dataset parquet.
Writes one sheet: stats summary at top, per-episode detail below.

Usage:
    python3 scripts/dataset_build/export_rollout_analysis.py
    python3 scripts/dataset_build/export_rollout_analysis.py --output my_analysis.xlsx
    python3 scripts/dataset_build/export_rollout_analysis.py --repo-id Suyash44/rollout_smolvla_...
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Valid outcome labels — anything else is treated as "unlabeled"
OUTCOME_LABELS = {"success", "failure"}


def parse_dataset_name(repo_id: str) -> dict:
    """Pull training params out of a repo_id like rollout_smolvla_..._20ksteps_30fps_20eps_..."""
    p = {"repo_id": repo_id, "model": "", "training_steps": "", "fps": "", "rollout_episodes": ""}
    m = re.search(r"(\d+)ksteps", repo_id)
    if m:
        p["training_steps"] = int(m.group(1)) * 1000
    m = re.search(r"(\d+)fps", repo_id)
    if m:
        p["fps"] = int(m.group(1))
    m = re.search(r"(\d+)eps", repo_id)
    if m:
        p["rollout_episodes"] = int(m.group(1))
    m = re.search(r"rollout_([a-z]+)_", repo_id)
    if m:
        p["model"] = m.group(1)
    return p


def load_config(repo_root: Path) -> dict:
    """Read dataset_config.yaml — returns full config dict."""
    for cfg_path in [repo_root / "scripts" / "dataset_config.yaml", repo_root / "dataset_config.yaml"]:
        if cfg_path.exists():
            import yaml
            return yaml.safe_load(cfg_path.read_text())
    raise FileNotFoundError("No dataset_config.yaml found")


def find_latest_export(repo_root: Path, repo_id: str) -> Path:
    """Find the most recent annotation export folder for this repo."""
    slug = repo_id.replace("/", "__")
    candidates = sorted((repo_root / "annotations" / "exports").glob(f"{slug}_annotated_*"))
    if not candidates:
        raise FileNotFoundError(f"No annotation exports found for {repo_id}")
    return candidates[-1]


def _extract_speech_text(atom: dict) -> str:
    """
    Speech events can be stored two ways:
      - Simple: {style: 'speech', content: 'some text'}
      - Tool call: {style: None, tool_calls: [{function: {name: 'say', arguments: {text: '...'}}}]}
    Returns the text string, or '' if not a speech event.
    """
    # Simple style
    if atom.get("style") == "speech":
        return atom.get("content") or ""
    # Tool call style (visualizer stores speech as 'say' tool calls)
    for tc in _iter_atoms(atom.get("tool_calls")):
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            if isinstance(fn, dict) and fn.get("name") == "say":
                args = fn.get("arguments", {})
                if isinstance(args, dict):
                    return args.get("text") or ""
    return ""


def _iter_atoms(val):
    """Safely iterate annotation atoms — handles list, numpy array, None, or NaN."""
    if val is None:
        return []
    if isinstance(val, float):  # NaN from pandas
        return []
    try:
        return list(val)
    except TypeError:
        return []


def extract_episode_annotations(annot_df: pd.DataFrame) -> dict:
    """
    Per episode: extract outcome label, retry count, failure notes.

    outcome   = last subtask annotation chronologically (success/success_retry/failure)
    retries   = count of speech events containing the word 'retry'
    notes     = all speech event contents joined with '; '
    """
    results = {}

    for episode in sorted(annot_df["episode_index"].unique()):
        ep = annot_df[annot_df["episode_index"] == episode]

        # Use sets to deduplicate — persistent annotations repeat on every frame
        subtasks = set()
        speech_events = set()

        for _, row in ep.iterrows():
            for atom in _iter_atoms(row.get("language_persistent")):
                if isinstance(atom, dict) and atom.get("style") == "subtask":
                    subtasks.add((atom.get("timestamp", 0), atom.get("content", "")))
            for atom in _iter_atoms(row.get("language_events")):
                if not isinstance(atom, dict):
                    continue
                # Speech events are stored as tool_calls with style=None
                text = _extract_speech_text(atom)
                if text:
                    speech_events.add((atom.get("timestamp", 0), text))

        # Last subtask = final outcome label
        outcome = "unlabeled"
        if subtasks:
            outcome = max(subtasks, key=lambda x: x[0])[1]

        retry_count = sum(1 for _, c in speech_events if "retry" in c.lower())
        notes = "; ".join(c for _, c in sorted(speech_events) if c)

        results[episode] = {"outcome": outcome, "retry_count": retry_count, "failure_notes": notes}

    return results


def compute_jerkiness(merged_dir: Path) -> dict:
    """
    Matches the visualizer's 'Most Jerky Episodes' metric exactly:
      mean |Δa| normalized by each joint's range, skipping inactive joints.

    motorRange[d] = max(action_d) - min(action_d) across all episodes.
    A joint is inactive in an episode if its p95(|Δa|) < 0.1% of its range.
    mean_delta = mean of (|a[t+1,d] - a[t,d]| / motorRange[d]) over active joints and frames.
    """
    parquets = sorted(merged_dir.glob("data/**/*.parquet"))
    if not parquets:
        return {}

    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    col = "action" if "action" in df.columns else "observation.state"

    # Stack all actions: list of (episode_index, np.array shape (T, D))
    episodes = []
    for ep_idx in sorted(df["episode_index"].unique()):
        ep = df[df["episode_index"] == ep_idx]
        actions = np.array(ep[col].tolist(), dtype=np.float64)
        episodes.append((int(ep_idx), actions))

    if not episodes:
        return {}

    action_dim = episodes[0][1].shape[1]

    # Per-joint range across all episodes.
    # Match JS: `hi - lo || 1` — use actual range unless exactly 0, only then fall back to 1.
    all_actions = np.concatenate([a for _, a in episodes], axis=0)
    raw_ranges = all_actions.max(axis=0) - all_actions.min(axis=0)
    motor_ranges = np.where(raw_ranges == 0, 1.0, raw_ranges)

    ACTIVITY_THRESHOLD = 0.001  # joint inactive if p95(|Δa|) < 0.1% of its range
    results = {}

    for ep_idx, actions in episodes:
        if len(actions) < 2:
            results[ep_idx] = 0.0
            continue

        deltas = np.abs(np.diff(actions, axis=0))  # shape (T-1, D)
        n = len(deltas)

        # Match JS percentile: sorted array at floor(n * 0.95) index
        p95 = np.sort(deltas, axis=0)[int(np.floor(n * 0.95))]
        active = p95 >= ACTIVITY_THRESHOLD * motor_ranges

        total, count = 0.0, 0
        for d in range(action_dim):
            if not active[d]:
                continue
            total += float(np.sum(deltas[:, d] / motor_ranges[d]))
            count += n

        results[ep_idx] = total / count if count > 0 else 0.0

    # Per-episode label: relative rank within this dataset (p33/p67)
    deltas_list = list(results.values())
    p33, p67 = np.percentile(deltas_list, 33), np.percentile(deltas_list, 67)
    episode_results = {
        ep: (delta, "smooth" if delta <= p33 else "moderate" if delta <= p67 else "jerky")
        for ep, delta in results.items()
    }

    # Overall jerkiness: σ of normalized |Δa| per joint across all frames.
    # Fixed thresholds on normalized σ (σ/motorRange) match the visualizer's classification.
    # σ < 0.03 → smooth, 0.03-0.06 → moderate, ≥ 0.06 → jerky per dimension.
    # Overall = "jerky" if more jerky dims than smooth, "smooth" if reverse, else "moderate".
    all_deltas = np.concatenate([
        np.abs(np.diff(a, axis=0)) for _, a in episodes if len(a) >= 2
    ], axis=0)  # shape (total_frames, D)
    sigma_per_dim = all_deltas.std(axis=0)  # raw σ per joint (matches visualizer's σ display)
    SIGMA_SMOOTH = 0.03   # below this = smooth joint
    SIGMA_JERKY  = 0.06   # above this = jerky joint
    dim_labels = ["smooth" if s < SIGMA_SMOOTH else "jerky" if s >= SIGMA_JERKY else "moderate"
                  for s in sigma_per_dim]
    n_jerky = dim_labels.count("jerky")
    n_smooth = dim_labels.count("smooth")
    # "jerky" if jerky dims >= smooth dims (tie goes to jerky — matches visualizer behavior)
    if n_jerky > 0 and n_jerky >= n_smooth:
        overall = "jerky"
    elif n_smooth >= n_jerky + dim_labels.count("moderate"):
        overall = "smooth"
    else:
        overall = "moderate"

    return episode_results, overall


def write_excel(output_path: Path, params: dict, episode_rows: list, overall_jerkiness: str = "unknown"):
    """Write Summary sheet + Episodes sheet to Excel file."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("ERROR: pip install openpyxl", file=sys.stderr)
        sys.exit(1)

    wb = openpyxl.Workbook()

    # ── Summary sheet ──────────────────────────────────────────────────────────
    ws_sum = wb.active
    ws_sum.title = "Summary"

    total = len(episode_rows)
    n_success = sum(1 for r in episode_rows if r["outcome"] == "success")
    n_failure = sum(1 for r in episode_rows if r["outcome"] == "failure")
    n_unlabeled = sum(1 for r in episode_rows if r["outcome"] == "unlabeled")
    success_rate = f"{n_success / total * 100:.1f}%" if total > 0 else "N/A"

    retry_total = sum(r["retry_count"] for r in episode_rows)
    episodes_with_retries = sum(1 for r in episode_rows if r["retry_count"] > 0)
    all_notes = [r["failure_notes"] for r in episode_rows if r["failure_notes"]]

    summary_rows = [
        ("── Dataset ──", ""),
        ("repo_id", params["repo_id"]),
        ("model", params["model"]),
        ("training_steps", params["training_steps"]),
        ("fps", params["fps"]),
        ("rollout_episodes", params["rollout_episodes"]),
        ("setup", params.get("setup", "")),
        ("object", params.get("object", "")),
        ("", ""),
        ("overall_jerkiness", overall_jerkiness),
        ("", ""),
        ("── Results ──", ""),
        ("total_episodes", total),
        ("success", n_success),
        ("failure", n_failure),
        ("unlabeled", n_unlabeled),
        ("success_rate", success_rate),
        ("", ""),
        ("── Retries ──", ""),
        ("total_retries", retry_total),
        ("episodes_with_retries", episodes_with_retries),
        ("", ""),
        ("── Failure Notes ──", ""),
        ("notes", "; ".join(all_notes) if all_notes else "—"),
    ]

    bold = Font(bold=True)
    ws_sum.append(["Metric", "Value"])
    ws_sum["A1"].font = bold
    ws_sum["B1"].font = bold

    jerk_fill = {
        "smooth":   PatternFill("solid", fgColor="C6EFCE"),
        "moderate": PatternFill("solid", fgColor="FFEB9C"),
        "jerky":    PatternFill("solid", fgColor="FFC7CE"),
    }

    for metric, value in summary_rows:
        ws_sum.append([metric, value])
        if metric.startswith("──"):
            ws_sum.cell(ws_sum.max_row, 1).font = bold
        # Color the overall_jerkiness value cell
        if metric == "overall_jerkiness" and isinstance(value, str):
            f = jerk_fill.get(value)
            if f:
                ws_sum.cell(ws_sum.max_row, 2).fill = f
                ws_sum.cell(ws_sum.max_row, 2).font = bold
        # Highlight the failure notes row in light orange
        if metric == "notes" and value and value != "—":
            notes_fill = PatternFill("solid", fgColor="FCE4D6")
            ws_sum.cell(ws_sum.max_row, 1).fill = notes_fill
            ws_sum.cell(ws_sum.max_row, 2).fill = notes_fill

    ws_sum.column_dimensions["A"].width = 30
    ws_sum.column_dimensions["B"].width = 70

    # ── Episode table — appended below summary in same sheet ───────────────────
    ws_sum.append([])  # blank separator row

    ep_headers = ["model", "episode", "outcome", "retry_count", "failure_notes", "mean_delta_action", "setup", "object"]
    ws_sum.append(ep_headers)
    header_row = ws_sum.max_row
    for col in range(1, len(ep_headers) + 1):
        ws_sum.cell(header_row, col).font = bold

    fill = {
        "success":       PatternFill("solid", fgColor="C6EFCE"),
        "success_retry": PatternFill("solid", fgColor="FFEB9C"),
        "failure":       PatternFill("solid", fgColor="FFC7CE"),
        "unlabeled":     PatternFill("solid", fgColor="D9D9D9"),
    }

    for row in episode_rows:
        ws_sum.append([
            row["model_name"],
            row["episode"],
            row["outcome"],
            row["retry_count"],
            row["failure_notes"],
            round(row["mean_delta"], 4),
            row["setup"],
            row["object"],
        ])
        outcome_fill = fill.get(row["outcome"])
        if outcome_fill:
            for col in range(1, len(ep_headers) + 1):
                ws_sum.cell(ws_sum.max_row, col).fill = outcome_fill

    # Widen columns for the episode table (cols A-F)
    col_widths = [12, 18, 14, 45, 20, 14]
    for i, w in enumerate(col_widths, 1):
        ws_sum.column_dimensions[get_column_letter(i)].width = max(
            ws_sum.column_dimensions[get_column_letter(i)].width or 0, w
        )

    wb.save(output_path)
    print(f"Saved: {output_path.resolve()}")


def main():
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, default=None, help="Annotation export folder (default: latest)")
    parser.add_argument("--merged-dir", type=Path, default=None, help="Merged dataset dir (default: merged_dataset/)")
    parser.add_argument("--repo-id", type=str, default=None, help="HF repo_id (default: from dataset_config.yaml)")
    parser.add_argument("--output", type=Path, default=Path("rollout_analysis.xlsx"), help="Output Excel path")
    args = parser.parse_args()

    if args.merged_dir is None:
        args.merged_dir = repo_root / "merged_dataset"

    cfg = load_config(repo_root)
    if args.repo_id is None:
        args.repo_id = f"{cfg['hf_user']}/{cfg['repo']}"
        print(f"repo_id: {args.repo_id}")

    params = parse_dataset_name(args.repo_id)
    if cfg.get("model_name"):
        params["model"] = cfg["model_name"]
    params["setup"] = cfg.get("setup", "")
    params["object"] = cfg.get("object", "")

    if args.export_dir is None:
        args.export_dir = find_latest_export(repo_root, args.repo_id)
    print(f"annotation export: {args.export_dir}")

    print("loading annotations...")
    parquets = sorted(args.export_dir.glob("data/**/*.parquet"))
    if not parquets:
        print(f"ERROR: no parquet files in {args.export_dir}/data/", file=sys.stderr)
        sys.exit(1)
    annot_df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    print("extracting episode outcomes...")
    episode_annots = extract_episode_annotations(annot_df)

    print("computing jerkiness from merged dataset...")
    jerkiness, overall_jerkiness = compute_jerkiness(args.merged_dir)

    # Union of annotated episodes + episodes in merged parquet
    all_episodes = sorted(set(episode_annots) | set(jerkiness))

    episode_rows = []
    for ep in all_episodes:
        annot = episode_annots.get(ep, {"outcome": "unlabeled", "retry_count": 0, "failure_notes": ""})
        mean_delta, jerk_label = jerkiness.get(ep, (0.0, "unknown"))
        episode_rows.append({
            "model_name": params["model"],
            "episode": ep,
            "outcome": annot["outcome"],
            "retry_count": annot["retry_count"],
            "failure_notes": annot["failure_notes"],
            "mean_delta": mean_delta,
            "jerk_label": jerk_label,
            "setup": params.get("setup", ""),
            "object": params.get("object", ""),
        })

    write_excel(args.output, params, episode_rows, overall_jerkiness)

    # Quick terminal summary
    total = len(episode_rows)
    success = sum(1 for r in episode_rows if r["outcome"] == "success")
    print(f"\nsuccess: {success}/{total} ({success / total * 100:.1f}%)" if total else "no episodes")


if __name__ == "__main__":
    main()
