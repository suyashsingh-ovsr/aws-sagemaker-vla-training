#!/usr/bin/env bash
# Base script: run a SmolVLA policy on a single arm (no teleoperation).
# Usage: run_smolvla.sh --arm left|right --model <hf_repo_id> [--record]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEROBOT_ROLLOUT=(uv run --project "${REPO_ROOT}" lerobot-rollout)

PORT_FILE="scripts/port.yaml"
CAMERA_FILE="scripts/camera.yaml"
DATASET_FILE="scripts/dataset_config.yaml"

arm=""
model=""
record=false
resume=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)
      arm="$2"
      shift 2
      ;;
    --model)
      model="$2"
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
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$arm" ]]; then
  echo "Usage: $0 --arm left|right --model <hf_repo_id> [--record]" >&2
  exit 1
fi
if [[ "$arm" != "left" && "$arm" != "right" ]]; then
  echo "--arm must be 'left' or 'right'" >&2
  exit 1
fi
if [[ -z "$model" ]]; then
  echo "--model is required" >&2
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
  value=$(sed -n "s/^${key}:[[:space:]]*//p" "$DATASET_FILE" | tail -n 1)
  if [[ -z "$value" ]]; then
    echo "Missing ${key} in ${DATASET_FILE}" >&2
    exit 1
  fi
  value=${value#\"}
  value=${value%\"}
  printf "%s" "$value"
}

follower_port=$(get_port "${arm}_follower")

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

cameras="{ front: { type: ${front_type}, serial_number_or_name: ${front_serial}, width: ${front_width}, height: ${front_height}, fps: ${front_fps} }, wrist: { type: ${wrist_type}, serial_number_or_name: ${wrist_serial}, width: ${wrist_width}, height: ${wrist_height}, fps: ${wrist_fps} }, wrist2: { type: ${wrist2_type}, serial_number_or_name: ${wrist2_serial}, width: ${wrist2_width}, height: ${wrist2_height}, fps: ${wrist2_fps} } }"

rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2","observation.images.wrist2":"observation.images.camera3"}'

if [[ -z "${HF_API_TOKEN:-}" ]]; then
  read -rsp "HF_API_TOKEN not set. Enter HuggingFace token: " HF_API_TOKEN
  echo
fi
export HF_TOKEN="${HF_API_TOKEN}"

hf_user=$(get_dataset_value "hf_user")
single_task=$(get_dataset_value "single_task")
model_slug="${model##*/}"
repo_id="${hf_user}/eval_${model_slug}_${arm}"

if $record; then
  "${LEROBOT_ROLLOUT[@]}" \
    --robot.type=so101_follower \
    --robot.port="${follower_port}" \
    --robot.id="bimanual_follower_${arm}" \
    --robot.cameras="${cameras}" \
    --policy.path="${model}" \
    --rename_map="${rename_map}" \
    --strategy.type=sentry \
    --dataset.repo_id="${repo_id}" \
    --dataset.fps=15 \
    --dataset.num_episodes=10 \
    --dataset.single_task="${single_task}" \
    --dataset.push_to_hub=false \
    --resume="${resume}" \
    --display_data=true
else
  "${LEROBOT_ROLLOUT[@]}" \
    --robot.type=so101_follower \
    --robot.port="${follower_port}" \
    --robot.id="bimanual_follower_${arm}" \
    --robot.cameras="${cameras}" \
    --policy.path="${model}" \
    --rename_map="${rename_map}" \
    --strategy.type=base \
    --fps=15 \
    --display_data=true
fi
