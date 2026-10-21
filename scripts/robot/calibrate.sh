#!/usr/bin/env bash
set -euo pipefail

PORT_FILE="scripts/port.yaml"

if [[ ! -f "$PORT_FILE" ]]; then
  echo "Missing $PORT_FILE" >&2
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

left_follower=$(get_port "left_follower")
right_follower=$(get_port "right_follower")
left_leader=$(get_port "left_leader")
right_leader=$(get_port "right_leader")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CALIB_DIR="${REPO_ROOT}/scripts/calibrations/ssingh_4"

echo "Calibrating follower arms..."
lerobot-calibrate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port="${left_follower}" \
  --robot.right_arm_config.port="${right_follower}" \
  --robot.id=bimanual_follower \
  --robot.calibration_dir="${CALIB_DIR}/robots"

echo "Calibrating leader arms..."
lerobot-calibrate \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port="${left_leader}" \
  --teleop.right_arm_config.port="${right_leader}" \
  --teleop.id=bimanual_leader \
  --teleop.calibration_dir="${CALIB_DIR}/teleoperators"
