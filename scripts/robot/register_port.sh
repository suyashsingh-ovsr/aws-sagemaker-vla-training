#!/usr/bin/env bash
set -euo pipefail

find_port() {
  local label="$1"
  local port

  echo "$label" > /dev/tty

  local before after diff
  before=$(ls /dev/serial/by-id/ 2>/dev/null || true)

  echo "Unplug USB cable from this arm, then press Enter." > /dev/tty
  read -r < /dev/tty

  sleep 0.5
  after=$(ls /dev/serial/by-id/ 2>/dev/null || true)

  diff=$(comm -23 <(echo "$before" | sort) <(echo "$after" | sort))

  if [[ -z "$diff" ]]; then
    echo "Error: no port disappeared. Try again." >&2
    exit 1
  fi

  local count
  count=$(echo "$diff" | wc -l)
  if [[ "$count" -gt 1 ]]; then
    echo "Error: multiple ports disappeared: $diff" >&2
    exit 1
  fi

  port="/dev/serial/by-id/${diff}"
  echo "Detected: $port" > /dev/tty
  echo "Reconnect USB cable and press Enter." > /dev/tty
  read -r < /dev/tty
  sleep 0.5

  printf "%s\n" "$port"
}

left_follower_port=$(find_port "[1/4] Left follower arm")
left_leader_port=$(find_port "[2/4] Left leader arm")
right_follower_port=$(find_port "[3/4] Right follower arm")
right_leader_port=$(find_port "[4/4] Right leader arm")

cat <<EOF > port.yaml
left_follower: ${left_follower_port}
left_leader: ${left_leader_port}
right_follower: ${right_follower_port}
right_leader: ${right_leader_port}
EOF

echo "Saved ports to port.yaml"
