#!/usr/bin/env bash
#
# export_subtask_annotations.sh
#
# Exports whatever subtask annotations are currently saved on the running
# lerobot-dataset-visualizer backend (the same one you use at localhost:3002)
# into a NEW timestamped folder inside THIS repo (S0101-bimanual), instead of
# the default /tmp location.
#
# Requirements before running this:
#   1. The visualizer backend must already be running (uvicorn app:app --port 7861)
#   2. You must have already clicked "Save episode" in the browser for every
#      episode you annotated -- this script only exports what's already saved
#      on the backend, it does not add annotations itself.
#
# Usage:
#   ./export_subtask_annotations.sh <hf_username>/<repo_name>
#
#   EXAMPLE (real values, no labels, no colons -- just username/repo joined
#   by a single slash, exactly like the Hugging Face dataset URL path):
#     ./export_subtask_annotations.sh andrejarden/air_bag_middle_mixed_box
#
#   RECOMMENDED: leave the argument off entirely. The script will read
#   hf_user and repo automatically from dataset_config.yaml (root or
#   scripts/), so most of the time you just run:
#     ./export_subtask_annotations.sh
#
#   Only pass the argument explicitly when you want to export a DIFFERENT
#   dataset than the one in dataset_config.yaml, just for this one run.
#
# If you omit the dataset argument, it defaults to the repo declared in
# dataset_config.yaml (if present).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:7861}"
OUT_BASE="$REPO_ROOT/annotations/exports"

# Resolve dataset repo_id: explicit arg > dataset_config.yaml > error
# dataset_config.yaml may live at the repo root OR inside scripts/ -- check both.
CONFIG_FILE=""
if [ -f "$REPO_ROOT/dataset_config.yaml" ]; then
    CONFIG_FILE="$REPO_ROOT/dataset_config.yaml"
elif [ -f "$REPO_ROOT/scripts/dataset_config.yaml" ]; then
    CONFIG_FILE="$REPO_ROOT/scripts/dataset_config.yaml"
fi

if [ "${1:-}" != "" ]; then
    DATASET="$1"
elif [ -n "$CONFIG_FILE" ]; then
    HF_USER=$(grep -E '^hf_user:' "$CONFIG_FILE" | awk '{print $2}')
    REPO=$(grep -E '^repo:' "$CONFIG_FILE" | awk '{print $2}')
    if [ -z "$HF_USER" ] || [ -z "$REPO" ]; then
        echo "Could not parse hf_user/repo from dataset_config.yaml. Pass the dataset explicitly:" >&2
        echo "  $0 <hf_user>/<repo>" >&2
        exit 1
    fi
    DATASET="${HF_USER}/${REPO}"
else
    echo "No dataset specified and no dataset_config.yaml found." >&2
    echo "Usage: $0 <hf_user>/<repo>" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATASET_SLUG="${DATASET/\//__}"
OUT_DIR="$OUT_BASE/${DATASET_SLUG}_annotated_${TIMESTAMP}"

mkdir -p "$OUT_BASE"

echo "Dataset:    $DATASET"
echo "Backend:    $BACKEND_URL"
echo "Output to:  $OUT_DIR"
echo ""

# Confirm the backend actually has this dataset loaded / reachable first.
if ! curl -sf "$BACKEND_URL/api/episodes/0/atoms?repo_id=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$DATASET")" > /dev/null 2>&1; then
    echo "WARNING: could not confirm the backend has '$DATASET' loaded." >&2
    echo "Make sure you've opened it in the browser at least once this session." >&2
fi

RESPONSE_AND_STATUS=$(curl -s -w "\n__HTTP_STATUS__:%{http_code}" -X POST "$BACKEND_URL/api/export" \
    -H "Content-Type: application/json" \
    -d "{\"repo_id\": \"$DATASET\", \"output_dir\": \"$OUT_DIR\", \"copy_videos\": true}")

HTTP_STATUS=$(echo "$RESPONSE_AND_STATUS" | grep '__HTTP_STATUS__:' | sed 's/__HTTP_STATUS__://')
RESPONSE=$(echo "$RESPONSE_AND_STATUS" | sed '/__HTTP_STATUS__:/d')

echo "HTTP status: $HTTP_STATUS"
echo "Response body:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

if [ "$HTTP_STATUS" != "200" ]; then
    echo "Export FAILED (HTTP $HTTP_STATUS). See response body above for the reason." >&2
    echo "Nothing was written to $OUT_DIR." >&2
    exit 1
fi

echo "Done. This folder is new and independent -- nothing else in $OUT_BASE was touched."
echo "Consider committing $OUT_DIR/meta/ (small) to git, but NOT videos/ or data/ unless you use git-lfs."
