#!/usr/bin/env bash
# Run a SmolVLA policy on both follower arms (bimanual).
# Usage: run_smolvla_bimanual.sh --model <hf_repo_id_or_local_path> [--record [--rollout-repo <repo_id>] [--resume]]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEROBOT_ROLLOUT=(lerobot-rollout)

PORT_FILE="scripts/port.yaml"
CAMERA_FILE="scripts/camera.yaml"
DATASET_FILE="scripts/dataset_config.yaml"

model=""
record=false
resume=false
rollout_repo=""
pretrained_revision=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    --pretrained_revision)
      pretrained_revision="$2"
      shift 2
      ;;
    --record)
      record=true
      shift
      ;;
    --resume)
      resume=true
      shift
      ;;
    --rollout-repo)
      rollout_repo="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$model" ]]; then
  echo "Usage: $0 --model <hf_repo_id_or_local_path> [--record [--rollout-repo <repo_id>] [--resume]]" >&2
  exit 1
fi

if [[ -n "$rollout_repo" && "$record" == false ]]; then
  echo "--rollout-repo requires --record" >&2
  exit 1
fi

for file in "$PORT_FILE" "$CAMERA_FILE" "$DATASET_FILE"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing $file" >&2
    exit 1
  fi
done

get_port() {
  local key="$1"
  local value
  value=$(sed -n "s/^${key}:[[:space:]]*//p" "$PORT_FILE" | tail -n 1)
  if [[ -z "$value" ]]; then
    echo "Missing port for ${key} in ${PORT_FILE}" >&2
    exit 1
  fi
  printf "%s" "$value"
}

get_cam_value() {
  local section="$1"
  local key="$2"
  local value
  value=$(awk -v section="$section" -v key="$key" '
    BEGIN { current = "" }
    /^[^[:space:]]/ {
      split($1, a, ":");
      current = a[1];
    }
    current == section && $1 == key ":" { print $2 }
  ' "$CAMERA_FILE" | tail -n 1)
  if [[ -z "$value" ]]; then
    echo "Missing ${section}.${key} in ${CAMERA_FILE}" >&2
    exit 1
  fi
  printf "%s" "$value"
}

get_dataset_value() {
  local key="$1"
  local value
  value=$(sed -n "s/^${key}:[[:space:]]*//p" "$DATASET_FILE" | tail -n 1 | sed 's/[[:space:]]*#.*//')
  if [[ -z "$value" ]]; then
    echo "Missing ${key} in ${DATASET_FILE}" >&2
    exit 1
  fi
  value=${value#\"}
  value=${value%\"}
  printf "%s" "$value"
}

left_follower=$(get_port "left_follower")
right_follower=$(get_port "right_follower")

front_type=$(get_cam_value "front" "type")
front_serial=$(get_cam_value "front" "serial_number_or_name")
front_width=$(get_cam_value "front" "width")
front_height=$(get_cam_value "front" "height")
front_fps=$(get_cam_value "front" "fps")

wrist_type=$(get_cam_value "wrist" "type")
wrist_serial=$(get_cam_value "wrist" "serial_number_or_name")
wrist_width=$(get_cam_value "wrist" "width")
wrist_height=$(get_cam_value "wrist" "height")
wrist_fps=$(get_cam_value "wrist" "fps")

wrist2_type=$(get_cam_value "wrist2" "type")
wrist2_serial=$(get_cam_value "wrist2" "serial_number_or_name")
wrist2_width=$(get_cam_value "wrist2" "width")
wrist2_height=$(get_cam_value "wrist2" "height")
wrist2_fps=$(get_cam_value "wrist2" "fps")

cameras="{ front: { type: ${front_type}, serial_number_or_name: ${front_serial}, width: ${front_width}, height: ${front_height}, fps: ${front_fps} }, left_wrist: { type: ${wrist_type}, serial_number_or_name: ${wrist_serial}, width: ${wrist_width}, height: ${wrist_height}, fps: ${wrist_fps} }, right_wrist: { type: ${wrist2_type}, serial_number_or_name: ${wrist2_serial}, width: ${wrist2_width}, height: ${wrist2_height}, fps: ${wrist2_fps} } }"

rename_map='{}'

if [[ -z "${HF_API_TOKEN:-}" ]]; then
  read -rsp "HF_API_TOKEN not set. Enter HuggingFace token: " HF_API_TOKEN
  echo
fi
export HF_TOKEN="${HF_API_TOKEN}"

single_task=$(get_dataset_value "single_task")
hf_user=$(get_dataset_value "hf_user")
inference_fps=$(get_dataset_value "inference_fps")
num_episodes=$(get_dataset_value "num_episodes")
episode_time_s=$(get_dataset_value "episode_time_s")
reset_time_s=$(get_dataset_value "reset_time_s")
model_slug="${model##*/}"

if [[ -n "$rollout_repo" ]]; then
  if [[ "$rollout_repo" == */* ]]; then
    repo_id="$rollout_repo"
  else
    repo_id="${hf_user}/${rollout_repo}"
  fi
else
  repo_id="${hf_user}/rollout_${model_slug}_bimanual"
fi

if $record; then
  "${LEROBOT_ROLLOUT[@]}" \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port="${left_follower}" \
    --robot.right_arm_config.port="${right_follower}" \
    --robot.id=bimanual_follower \
    --robot.calibration_dir="${REPO_ROOT}/scripts/calibrations/ssingh_3/robots" \
    --robot.cameras="${cameras}" \
    --policy.path="${model}" \
    ${pretrained_revision:+--policy.pretrained_revision="${pretrained_revision}"} \
    --rename_map="${rename_map}" \
    --device cuda \
    --strategy.type=episodic \
    --dataset.repo_id="${repo_id}" \
    --dataset.fps="${inference_fps}" \
    --dataset.num_episodes="${num_episodes}" \
    --dataset.single_task="${single_task}" \
    --dataset.episode_time_s="${episode_time_s}" \
    --dataset.reset_time_s="${reset_time_s}" \
    --dataset.push_to_hub=true \
    --resume="${resume}" \
    --display_data=false
else
  "${LEROBOT_ROLLOUT[@]}" \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port="${left_follower}" \
    --robot.right_arm_config.port="${right_follower}" \
    --robot.id=bimanual_follower \
    --robot.calibration_dir="${REPO_ROOT}/scripts/calibrations/ssingh_3/robots" \
    --robot.cameras="${cameras}" \
    --policy.path="${model}" \
    ${pretrained_revision:+--policy.pretrained_revision="${pretrained_revision}"} \
    --rename_map="${rename_map}" \
    --device cuda \
    --strategy.type=base \
    --fps="${inference_fps}" \
    --display_data=false
fi
