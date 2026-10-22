#!/usr/bin/env bash
set -euo pipefail

# Configurable GR00T training parameters
DATASET_REPO_ID="${DATASET_REPO_ID:-Suyash44/blue_cylinder_grasp_2}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/ml/checkpoints/output}"
HF_POLICY_REPO="${HF_POLICY_REPO:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -n "${HF_TOKEN}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export PATH="/usr/local/lib/python3.12/dist-packages/bin:/usr/local/bin:$PATH"

# Training arguments
ARGS=(
  --policy.type=groot
  "--policy.base_model_path=${BASE_MODEL}"
  "--policy.embodiment_tag=${EMBODIMENT_TAG}"
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

echo "Starting GR00T fine-tuning"
echo "Model: ${BASE_MODEL}"
echo "Dataset: ${DATASET_REPO_ID}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Checkpoint frequency: ${SAVE_FREQ}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" /opt/ml/model

lerobot-train "${ARGS[@]}"

# Package training results for SageMaker's final model artifact.
cp -a "${OUTPUT_DIR}/." /opt/ml/model/
