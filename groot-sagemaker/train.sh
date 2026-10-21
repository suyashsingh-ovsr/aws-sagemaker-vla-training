#!/bin/bash
set -e

# SageMaker mounts S3 channel "training" at /opt/ml/input/data/training/
# S3 layout: s3://oversonic-groot-training/datasets/ → contains blue_cylinder_grasp_2/
DATASET_REPO_ID=${DATASET_REPO_ID:-Suyash44/blue_cylinder_grasp_2}
OUTPUT_DIR=/opt/ml/model/output

BASE_MODEL=${BASE_MODEL:-nvidia/GR00T-N1.7-3B}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-new_embodiment}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_FREQ=${SAVE_FREQ:-5000}
HF_POLICY_REPO=${HF_POLICY_REPO:-Suyash44/groot-blue-cylinder-finetune}

export PATH="/usr/local/lib/python3.12/dist-packages/bin:/usr/local/bin:$PATH"

# Set HuggingFace token if provided (enables push during training)
if [ -n "${HF_TOKEN}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

lerobot-train \
    --policy.type=groot \
    --policy.base_model_path="${BASE_MODEL}" \
    --policy.embodiment_tag="${EMBODIMENT_TAG}" \
    --policy.repo_id="${HF_POLICY_REPO}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --output_dir="${OUTPUT_DIR}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --batch_size="${BATCH_SIZE}" \
    --save_checkpoint_to_hub=true
