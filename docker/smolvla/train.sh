#!/usr/bin/env bash
set -euo pipefail

# Configurable SmolVLA training parameters
DATASET_REPO_ID="${DATASET_REPO_ID:?DATASET_REPO_ID is required}"
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/ml/checkpoints/output}"
HF_POLICY_REPO="${HF_POLICY_REPO:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

export PATH="/usr/local/lib/python3.12/dist-packages/bin:/usr/local/bin:${PATH}"

ARGS=(
    "--policy.path=${BASE_MODEL}"
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--output_dir=${OUTPUT_DIR}"
    "--steps=${STEPS}"
    "--batch_size=${BATCH_SIZE}"
    "--save_freq=${SAVE_FREQ}"
    "--save_checkpoint_to_hub=${PUSH_TO_HUB}"
)

if [[ -n "${HF_POLICY_REPO}" ]]; then
    ARGS+=("--policy.repo_id=${HF_POLICY_REPO}")
fi

echo "Starting SmolVLA fine-tuning"
echo "Base model: ${BASE_MODEL}"
echo "Dataset: ${DATASET_REPO_ID}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Checkpoint frequency: ${SAVE_FREQ}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" /opt/ml/model

lerobot-train "${ARGS[@]}"

# Copy completed training artifacts to SageMaker's model directory.
cp -a "${OUTPUT_DIR}/." /opt/ml/model/

echo "SmolVLA training completed."
