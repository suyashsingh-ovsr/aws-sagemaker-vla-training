#!/usr/bin/env bash
set -euo pipefail

PORT_FILE="scripts/port.yaml"
CAMERA_FILE="scripts/camera.yaml"

if [[ ! -f "$PORT_FILE" ]]; then
  echo "Missing $PORT_FILE" >&2
  exit 1
fi

if [[ ! -f "$CAMERA_FILE" ]]; then
  echo "Missing $CAMERA_FILE" >&2
  exit 1
fi

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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CALIB_DIR="${REPO_ROOT}/scripts/calibrations/ssingh_4"

uv run lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port="${left_follower}" \
  --robot.right_arm_config.port="${right_follower}" \
  --robot.id=bimanual_follower \
  --robot.calibration_dir="${CALIB_DIR}/robots" \
  --robot.cameras="${cameras}" \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port="${left_leader}" \
  --teleop.right_arm_config.port="${right_leader}" \
  --teleop.id=bimanual_leader \
  --teleop.calibration_dir="${CALIB_DIR}/teleoperators" \
  --display_data=true \
  "$@"
