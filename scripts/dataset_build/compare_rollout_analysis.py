#!/usr/bin/env python3
"""
Compare multiple rollout analysis Excel files into a single comparison Excel.

Reads all .xlsx files from a folder and produces:
  - Comparison sheet: one row per model with key metrics side-by-side
  - All Episodes sheet: combined episode table from all files

Usage:
    python3 scripts/dataset_build/compare_rollout_analysis.py
    python3 scripts/dataset_build/compare_rollout_analysis.py --input-dir ~/Desktop/rollout_analysis_all
    python3 scripts/dataset_build/compare_rollout_analysis.py --output comparison.xlsx
"""

import argparse
import sys
from pathlib import Path

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print("ERROR: pip install openpyxl", file=sys.stderr)
    sys.exit(1)


JERK_FILL = {
    "smooth":   "C6EFCE",
    "moderate": "FFEB9C",
    "jerky":    "FFC7CE",
}

OUTCOME_FILL = {
    "success":       "C6EFCE",
    "success_retry": "FFEB9C",
    "failure":       "FFC7CE",
    "unlabeled":     "D9D9D9",
}


def parse_summary_sheet(ws) -> dict:
    """Extract key-value pairs from Summary sheet until the episode table header row."""
    data = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        key = str(row[0]).strip()
        if key.startswith("──"):
            continue
        val = row[1] if len(row) > 1 else None
        # Episode table header: model | episode | outcome | ...
        if key == "model" and len(row) > 1 and row[1] == "episode":
            break
        if key:
            data[key] = val
    return data


def parse_episode_rows(ws) -> list:
    """Extract episode table rows from Summary sheet."""
    rows = []
    header_row_idx = None
    headers = []

    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if row and row[0] == "model" and len(row) > 1 and row[1] == "episode":
            header_row_idx = i
            headers = [str(c) if c is not None else "" for c in row]
            continue
        if header_row_idx and row and row[0] is not None:
            entry = dict(zip(headers, row))
            rows.append(entry)

    return rows


def load_excel(path: Path) -> dict:
    """Load one analysis Excel; return dict with summary params + episode rows."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active  # Summary sheet

    summary = parse_summary_sheet(ws)
    episodes = parse_episode_rows(ws)

    return {"summary": summary, "episodes": episodes, "filename": path.name}


def _pct(n, total):
    if not total:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def _pct_float(n, total):
    if not total:
        return 0.0
    return round(n / total * 100, 1)


def _cell_fill(ws, row, col, hex_color):
    ws.cell(row, col).fill = PatternFill("solid", fgColor=hex_color)


def write_comparison(output_path: Path, datasets: list):
    wb = openpyxl.Workbook()

    # ── Comparison sheet ────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Comparison"

    bold = Font(bold=True)
    center = Alignment(horizontal="center")

    headers = [
        "model", "training_steps", "fps", "rollout_episodes",
        "setup", "object",
        "total_episodes", "success", "failure", "unlabeled",
        "success_rate_%", "total_retries", "episodes_with_retries",
        "overall_jerkiness",
    ]

    ws.append(headers)
    for col in range(1, len(headers) + 1):
        ws.cell(1, col).font = bold
        ws.cell(1, col).alignment = center

    for ds in datasets:
        s = ds["summary"]
        eps = ds["episodes"]
        total = int(s.get("total_episodes") or len(eps))
        n_success = int(s.get("success") or sum(1 for r in eps if r.get("outcome") == "success"))
        n_failure = int(s.get("failure") or sum(1 for r in eps if r.get("outcome") == "failure"))
        n_unlabeled = int(s.get("unlabeled") or sum(1 for r in eps if r.get("outcome") == "unlabeled"))
        success_pct = _pct_float(n_success, total)

        row_data = [
            s.get("model", ""),
            s.get("training_steps", ""),
            s.get("fps", ""),
            s.get("rollout_episodes", ""),
            s.get("setup", ""),
            s.get("object", ""),
            total,
            n_success,
            n_failure,
            n_unlabeled,
            success_pct,
            s.get("total_retries", ""),
            s.get("episodes_with_retries", ""),
            s.get("overall_jerkiness", ""),
        ]
        ws.append(row_data)
        r = ws.max_row

        # Color success_rate cell by value
        pct_col = headers.index("success_rate_%") + 1
        pct_val = success_pct if isinstance(success_pct, float) else 0
        if pct_val >= 70:
            _cell_fill(ws, r, pct_col, "C6EFCE")
        elif pct_val >= 40:
            _cell_fill(ws, r, pct_col, "FFEB9C")
        else:
            _cell_fill(ws, r, pct_col, "FFC7CE")

        # Color jerkiness cell
        jerk_col = headers.index("overall_jerkiness") + 1
        jerk_val = s.get("overall_jerkiness", "")
        if jerk_val in JERK_FILL:
            _cell_fill(ws, r, jerk_col, JERK_FILL[jerk_val])

    # Auto-width
    col_widths = [18, 16, 6, 18, 14, 14, 16, 10, 10, 12, 16, 16, 22, 18]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    # ── All Episodes sheet ──────────────────────────────────────────────────────
    ws_ep = wb.create_sheet("All Episodes")

    ep_headers = ["model", "episode", "outcome", "retry_count", "failure_notes", "mean_delta_action", "setup", "object"]
    ws_ep.append(ep_headers)
    for col in range(1, len(ep_headers) + 1):
        ws_ep.cell(1, col).font = bold

    for ds in datasets:
        for row in ds["episodes"]:
            ws_ep.append([
                row.get("model", ""),
                row.get("episode", ""),
                row.get("outcome", ""),
                row.get("retry_count", ""),
                row.get("failure_notes", ""),
                row.get("mean_delta_action", ""),
                row.get("setup", ""),
                row.get("object", ""),
            ])
            outcome = row.get("outcome", "")
            if outcome in OUTCOME_FILL:
                for col in range(1, len(ep_headers) + 1):
                    _cell_fill(ws_ep, ws_ep.max_row, col, OUTCOME_FILL[outcome])

    ep_widths = [18, 10, 14, 14, 50, 18, 14, 14]
    for i, w in enumerate(ep_widths, 1):
        ws_ep.column_dimensions[get_column_letter(i)].width = w

    wb.save(output_path)
    print(f"Saved: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path,
                        default=Path.home() / "Desktop" / "rollout_analysis_all",
                        help="Folder containing .xlsx analysis files")
    parser.add_argument("--output", type=Path,
                        default=Path("rollout_comparison.xlsx"),
                        help="Output Excel path")
    args = parser.parse_args()

    xlsx_files = sorted(args.input_dir.glob("*.xlsx"))
    if not xlsx_files:
        print(f"ERROR: no .xlsx files in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(xlsx_files)} files:")
    datasets = []
    for f in xlsx_files:
        print(f"  {f.name}")
        datasets.append(load_excel(f))

    # Sort by training_steps for consistent ordering
    def sort_key(ds):
        val = ds["summary"].get("training_steps", 0)
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    datasets.sort(key=sort_key)

    write_comparison(args.output, datasets)

    # Quick terminal summary
    print("\n── Comparison ──")
    print(f"{'model':<20} {'steps':>8} {'success%':>10} {'jerkiness':<12}")
    for ds in datasets:
        s = ds["summary"]
        eps = ds["episodes"]
        total = int(s.get("total_episodes") or len(eps))
        n_success = int(s.get("success") or sum(1 for r in eps if r.get("outcome") == "success"))
        pct = _pct(n_success, total)
        print(f"  {s.get('model', ''):<18} {str(s.get('training_steps', '')):>8} {pct:>10}  {s.get('overall_jerkiness', ''):<12}")


if __name__ == "__main__":
    main()
