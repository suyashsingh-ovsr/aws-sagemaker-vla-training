# Annotation Pipeline Demo

## 0. Start visualizer (internet required)

> Run from `~/projects/lerobot-dataset-visualizer`

**Terminal 1 — start backend:**
```bash
cd ~/projects/lerobot-dataset-visualizer/backend
source .venv/bin/activate
uvicorn app:app --port 7861 --reload
```

**Terminal 2 — start frontend:**
```bash
cd ~/projects/lerobot-dataset-visualizer
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun dev
```

Open browser at http://localhost:3000 — enter your dataset repo (e.g. `your_hf_username/your_repo_name`), click Go.
Annotate subtasks in the Annotations tab, click Save episode for each episode.

---

---
> Steps 1–7 must be run from the `S0101-bimanual` repo root.

## 1. View original dataset (no annotations)
```bash
python3 scripts/dataset_build/view_parquet.py \
  $(ls -d _hf_download_*/ | head -1)data/chunk-000/file-000.parquet \
  --cols episode_index frame_index timestamp observation.state
```

## 2. Export subtask annotations from visualizer
```bash
./scripts/dataset_build/export_subtask_annotations.sh
```

## 3. Review exported annotations
```bash
python3 scripts/dataset_build/make_annotation_review.py \
  $(ls -dt annotations/exports/*/ | head -1)
# Output: annotation_review.csv
```

## 4. View annotation export (what was annotated)
```bash
# Print in terminal
python3 scripts/dataset_build/view_parquet.py \
  "$(ls -dt annotations/exports/*/ | head -1)data/chunk-000/file-000.parquet"

# Export to CSV for full inspection
python3 scripts/dataset_build/view_parquet.py \
  "$(ls -dt annotations/exports/*/ | head -1)data/chunk-000/file-000.parquet" \
  --csv /tmp/annotation_export.csv --rows 0
```

## 5. Merge annotations into dataset + push to HuggingFace
```bash
python3 scripts/dataset_build/merge_annotations_to_parquet.py \
  --output-dir merged_dataset/ \
  --push \
  --push-repo your_hf_username/your_repo_name_annotated
```

## 6. View merged dataset (with annotations)
```bash
# Key columns only
python3 scripts/dataset_build/view_parquet.py \
  merged_dataset/data/chunk-000/file-000.parquet \
  --cols episode_index frame_index timestamp subtask

# Full parquet — export to CSV for complete inspection
python3 scripts/dataset_build/view_parquet.py \
  merged_dataset/data/chunk-000/file-000.parquet \
  --csv /tmp/merged_full.csv --rows 0
```

## 7. Export complete merged dataset to CSV (all parquet files)
```bash
python3 scripts/dataset_build/export_merged_to_csv.py
# Output: merged_csv/data/chunk-000/file-000.csv and file-001.csv
```

## 8. Export rollout analysis to Excel
```bash
python3 scripts/dataset_build/export_rollout_analysis.py
# Output: rollout_analysis.xlsx (Summary sheet + Episodes sheet)
```

Annotate each episode in the visualizer with one of these subtask labels before running:
- `success` — grasped + lifted
- `failure` — never grasped/lifted

Add `speech` events labeled `retry` for each retry attempt.
Add `speech` events with free text for failure notes.

---

## Rollback to older annotation export
**Use this if latest annotations have mistakes and you want to restore a previous version.**
```bash
python3 scripts/dataset_build/merge_annotations_to_parquet.py \
  --output-dir merged_dataset/ \
  --export-dir annotations/exports/<old_timestamp_folder> \
  --push \
  --push-repo your_hf_username/your_repo_name_annotated
```

**List available exports:**
```bash
ls annotations/exports/
```
