#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEROBOT_ROOT="$(cd "${REPO_ROOT}/../lerobot" && pwd)"
for venv in "${LEROBOT_ROOT}/.venv" "${REPO_ROOT}/.venv"; do
  if [[ -f "${venv}/bin/activate" ]]; then
    source "${venv}/bin/activate"
    break
  fi
done

PORT_FILE="scripts/port.yaml"
CAMERA_FILE="scripts/camera.yaml"
TRAIN_FILE="scripts/jobs/local/groot_training_config.yaml"

policy_path=""
task="Grasp the blue cylinder"
duration=60
calibration="ssingh_3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path)
      policy_path="$2"
      shift 2
      ;;
    --task)
      task="$2"
      shift 2
      ;;
    --duration)
      duration="$2"
      shift 2
      ;;
    --calibration)
      calibration="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

for file in "$PORT_FILE" "$CAMERA_FILE" "$TRAIN_FILE"; do
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

get_train_value() {
  local key="$1"
  local value

  value=$(sed -n "s/^${key}:[[:space:]]*//p" "$TRAIN_FILE" | tail -n 1)
  if [[ -z "$value" ]]; then
    echo "Missing ${key} in ${TRAIN_FILE}" >&2
    exit 1
  fi
  value=${value#\"}
  value=${value%\"}
  printf "%s" "$value"
}

if [[ -z "$policy_path" ]]; then
  output_dir=$(get_train_value "output_dir")
  policy_path="${output_dir}/checkpoints/last/pretrained_model"
  echo "No --policy-path given. Using last checkpoint: ${policy_path}" >&2
fi

if [[ ! -d "$policy_path" ]]; then
  echo "Policy path not found: ${policy_path}" >&2
  exit 1
fi

left_follower=$(get_port "left_follower")
right_follower=$(get_port "right_follower")
left_leader=$(get_port "left_leader")
right_leader=$(get_port "right_leader")

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

lerobot-rollout \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port="${left_follower}" \
  --robot.right_arm_config.port="${right_follower}" \
  --robot.id=bimanual_follower \
  --robot.calibration_dir="${REPO_ROOT}/scripts/calibrations/${calibration}/robots" \
  --robot.cameras="${cameras}" \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port="${left_leader}" \
  --teleop.right_arm_config.port="${right_leader}" \
  --teleop.id=bimanual_leader \
  --teleop.calibration_dir="${REPO_ROOT}/scripts/calibrations/${calibration}/teleoperators" \
  --policy.path="${policy_path}" \
  --task="${task}" \
  --duration="${duration}" \
  --display_data=false
