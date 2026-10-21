#!/usr/bin/env bash
# Generic training script for any policy type.
# Usage: ./scripts/train.sh --policy <act|diffusion|groot|smolvla> [--config <path>]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="$(cd "${REPO_ROOT}/../lerobot" && pwd)"
for venv in "${LEROBOT_ROOT}/.venv" "${REPO_ROOT}/.venv"; do
  if [[ -f "${venv}/bin/activate" ]]; then
    source "${venv}/bin/activate"
    break
  fi
done

DATASET_FILE="scripts/dataset_config.yaml"
TRAINING_CONFIG=""
policy_type=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      policy_type="$2"
      shift 2
      ;;
    --config)
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$policy_type" ]]; then
  echo "Usage: $0 --policy <act|diffusion|groot|smolvla> [--config <path>]" >&2
  exit 1
fi

# Default config paths per policy type
if [[ -z "$TRAINING_CONFIG" ]]; then
  case "$policy_type" in
    act)
      TRAINING_CONFIG="scripts/training_configs/act_config.yaml"
      ;;
    diffusion)
      TRAINING_CONFIG="scripts/training_configs/diffusion_config.yaml"
      ;;
    groot)
      TRAINING_CONFIG="scripts/training_configs/groot_config.yaml"
      ;;
    smolvla)
      TRAINING_CONFIG="scripts/training_configs/smolvla_config.yaml"
      ;;
    *)
      echo "Unknown policy type: $policy_type" >&2
      exit 1
      ;;
  esac
fi

for file in "$DATASET_FILE" "$TRAINING_CONFIG"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing $file" >&2
    exit 1
  fi
done

get_dataset_value() {
  local key="$1"
  local value

  value=$(sed -n "s/^${key}:[[:space:]]*//p" "$DATASET_FILE" | tail -n 1)
  if [[ -z "$value" ]]; then
    echo "Missing ${key} in ${DATASET_FILE}" >&2
    exit 1
  fi
  value=${value#\"}
  value=${value%\"}
  printf "%s" "$value"
}

get_train_value() {
  local key="$1"
  local value

  value=$(sed -n "s/^${key}:[[:space:]]*//p" "$TRAINING_CONFIG" | tail -n 1)
  if [[ -z "$value" ]]; then
    echo "Missing ${key} in ${TRAINING_CONFIG}" >&2
    exit 1
  fi
  value=${value#\"}
  value=${value%\"}
  printf "%s" "$value"
}

# Check for HF token
if [[ -z "${HF_API_TOKEN:-}" ]]; then
  read -rsp "HF_API_TOKEN not set. Enter HuggingFace token: " HF_API_TOKEN
  echo
fi
export HF_TOKEN="${HF_API_TOKEN}"

# Get dataset config
hf_user=$(get_dataset_value "hf_user")
repo=$(get_dataset_value "repo")

if [[ "$repo" != */* ]]; then
  repo_id="${hf_user}/${repo}"
else
  repo_id="$repo"
fi

# Get training config
output_dir=$(get_train_value "output_dir")
steps=$(get_train_value "steps")
batch_size=$(get_train_value "batch_size")
save_freq=$(get_train_value "save_freq")

# Build policy-specific args array
policy_args=()
case "$policy_type" in
  groot)
    base_model_path=$(get_train_value "base_model_path")
    embodiment_tag=$(get_train_value "embodiment_tag")
    lora_rank=$(get_train_value "lora_rank")
    tune_diffusion_model=$(get_train_value "tune_diffusion_model")
    tune_projector=$(get_train_value "tune_projector")
    tune_llm=$(get_train_value "tune_llm")
    tune_visual=$(get_train_value "tune_visual")

    policy_args+=(
      "--policy.base_model_path=${base_model_path}"
      "--policy.embodiment_tag=${embodiment_tag}"
      "--policy.tune_diffusion_model=${tune_diffusion_model}"
      "--policy.tune_projector=${tune_projector}"
      "--policy.tune_llm=${tune_llm}"
      "--policy.tune_visual=${tune_visual}"
      "--policy.lora_rank=${lora_rank}"
    )
    ;;
  act|diffusion|smolvla)
    # Add policy-specific params as needed
    ;;
esac

# Build model repo_id (dataset repo + policy suffix)
model_repo="${repo}_${policy_type}"
if [[ "$hf_user" != "" ]]; then
  model_repo_id="${hf_user}/${model_repo}"
else
  model_repo_id="${model_repo}"
fi

echo "Training ${policy_type} on ${repo_id}"
echo "Model repo: ${model_repo_id}"
echo "Output: ${output_dir}"

lerobot-train \
  --policy.type="${policy_type}" \
  "${policy_args[@]}" \
  --dataset.repo_id="${repo_id}" \
  --policy.repo_id="${model_repo_id}" \
  --policy.push_to_hub=false \
  --batch_size="${batch_size}" \
  --steps="${steps}" \
  --save_freq="${save_freq}" \
  --output_dir="${output_dir}" \
  --wandb.enable=false
