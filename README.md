# Bimanual SO-101 Robot Workflow

Complete workflow from setup to deployment for new users.

---

## Prerequisites

```bash
# Install lerobot with hardware dependencies
cd ../lerobot
uv sync --extra feetech --extra intelrealsense

# For GR00T training, install flash-attn first
uv pip install flash-attn --no-build-isolation
uv sync --extra groot

# Activate environment
source .venv/bin/activate
```

---

## Workflow Steps

### 1. Register USB Ports

**Script:** `scripts/robot/register_port.sh`

**Purpose:** Detect and map USB ports for all 4 arms (left/right follower/leader).

**Config files used:** None (interactive)

**Config files written:** `scripts/port.yaml`

**Usage:**
```bash
cd STOVSN-RobotArmSetup
./scripts/robot/register_port.sh
```

**Output:** `scripts/port.yaml`
```yaml
left_follower: /dev/serial/by-id/usb-...
left_leader: /dev/serial/by-id/usb-...
right_follower: /dev/serial/by-id/usb-...
right_leader: /dev/serial/by-id/usb-...
```

---

### 1b. Set USB Device Permissions

**Purpose:** Allow non-root access to USB serial devices.

**Required:** Before calibration, teleoperation, or recording.

**Usage:**
```bash
# Grant read/write permissions to all USB serial devices
sudo chmod 666 /dev/serial/by-id/*

# Or grant permissions to specific devices from port.yaml
sudo chmod 666 /dev/serial/by-id/usb-1a86_USB_Single_Serial_*
```

**Note:** Permissions reset after reboot. Add udev rule for persistent permissions:
```bash
# Create udev rule
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/99-usb-serial.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

### 2. Calibrate Arms

**Script:** `scripts/robot/calibrate.sh`

**Purpose:** Calibrate joint ranges for follower and leader arms.

**Config files used:**
- `scripts/port.yaml` (read)

**Config files written:**
- Calibration JSONs stored in lerobot's calibration directory
- Location: `~/.cache/lerobot/calibration/robots/` and `~/.cache/lerobot/calibration/teleoperators/`

**Usage:**
```bash
./scripts/robot/calibrate.sh
```

**Interactive process:**
- Follow prompts to move each joint to min/max positions
- Calibration saved with robot ID: `bimanual_follower`, `bimanual_leader`

---

### 3. Configure Cameras

**Config file:** `scripts/camera.yaml`

**Purpose:** Map camera serial numbers before teleoperation/recording.

**Setup:**
```bash
# Find connected RealSense cameras
lerobot-find-cameras

# Edit camera.yaml with detected serial numbers
nano scripts/camera.yaml
```

**Config format:**
```yaml
front:
  type: intelrealsense
  serial_number_or_name: "105322251816"  # Replace with your serial
  width: 640
  height: 480
  fps: 15
wrist:  # left arm camera
  type: intelrealsense
  serial_number_or_name: "114122250584"  # Replace with your serial
  width: 640
  height: 480
  fps: 15
wrist2:  # right arm camera
  type: intelrealsense
  serial_number_or_name: "311322303786"  # Replace with your serial
  width: 640
  height: 480
  fps: 15
```

---

### 4. Teleoperation (Test)

**Script:** `scripts/robot/teleop.sh`

**Purpose:** Test teleoperation before recording. Leader arms control follower arms in real-time.

**Config files used:**
- `scripts/port.yaml` (read)
- `scripts/camera.yaml` (read)

**Config files written:** None

**Usage:**
```bash
./scripts/robot/teleop.sh
```

**Notes:**
- Press Ctrl+C to stop
- Verify smooth control before recording

---

### 5. Configure Dataset

**Config file:** `scripts/dataset_config.yaml`

**Purpose:** Define dataset metadata before recording.

**REQUIRED:** Edit this file before running record.sh.

**Setup:**
```bash
nano scripts/dataset_config.yaml
```

**Config format:**
```yaml
num_eposides: 40                # Total episodes to record
single_task: "Task description"  # What robot should do
hf_user: your_username          # HuggingFace username
repo: dataset_name              # Dataset repo name
push_to_hub: True               # Auto-upload to HF
episode_time_s: 30              # Max episode duration
reset_time_s: 10                # Time between episodes
```

**Notes:**
- `hf_user`: Your HuggingFace username
- `repo`: Dataset will be created at `hf_user/repo`
- `single_task`: Clear task description for what demonstrations show
- Ensure HF account has write permissions

---

### 6. Record Demonstrations

**Script:** `scripts/robot/record.sh` (bimanual) or `scripts/robot/record_single.sh` (single arm)

**Purpose:** Record demonstration episodes and upload to HuggingFace dataset.

**Config files used:**
- `scripts/port.yaml` (read)
- `scripts/camera.yaml` (read)
- `scripts/dataset_config.yaml` (read)

**Config files written:**
- Dataset uploaded to HuggingFace: `{hf_user}/{repo}`

**Usage:**
```bash
# Bimanual recording
./scripts/robot/record.sh [--resume | --reset]

# Single arm recording
./scripts/robot/record_single.sh --arm left [--resume | --reset]
```

**Flags:**
- `--resume`: Continue from last episode
- `--reset`: Start from episode 0

**Environment:**
- Requires `HF_TOKEN` or `HF_API_TOKEN` environment variable for dataset upload

---

### 7. Train Policy

**Script:** `scripts/train.sh`

**Purpose:** Train a policy model on recorded dataset.

**Config files used:**
- `scripts/dataset_config.yaml` (read: hf_user, repo)
- `scripts/training_configs/{policy}_config.yaml` (read: training hyperparams)

**Config files written:**
- Model checkpoints saved to `output_dir` (defined in training config)

**Usage:**
```bash
# Train with specific policy
./scripts/train.sh --policy groot

# Use custom config
./scripts/train.sh --policy act --config path/to/custom_config.yaml
```

**Supported policies:**
- `act` — Action Chunking Transformer
- `diffusion` — Diffusion Policy
- `groot` — GR00T (NVIDIA)
- `smolvla` — SmolVLA

**Training config example** (`scripts/training_configs/groot_config.yaml`):
```yaml
base_model_path: nvidia/GR00T-N1.5-3B
embodiment_tag: new_embodiment
output_dir: ./outputs/groot
steps: 10000
batch_size: 32
save_freq: 1000
lora_rank: 0
tune_diffusion_model: true
tune_projector: true
tune_llm: false
tune_visual: false
```

**Notes:**
- Requires `HF_API_TOKEN` environment variable or will prompt interactively
- Training writes checkpoints to `output_dir`
- Monitor with tensorboard: `tensorboard --logdir outputs/groot`

---

### 8. Rollout / Deploy

**Option A: Replay episode** (test dataset quality)

**Script:** `scripts/robot/replay.sh`

**Purpose:** Replay a recorded episode on follower arms.

**Config files used:**
- `scripts/port.yaml` (read)
- `scripts/dataset_config.yaml` (read: hf_user, repo)

**Usage:**
```bash
./scripts/robot/replay.sh --episode 5
```

---

**Option B: Run trained policy** (real deployment)

**Script:** `scripts/robot/run_policy/run_{policy}.sh` or demo scenarios

**Purpose:** Run trained model on real robot.

**Config files used:**
- `scripts/port.yaml` (read)
- `scripts/camera.yaml` (read)
- `scripts/dataset_config.yaml` (optional: for recording rollout)

**Usage:**
```bash
# Direct policy runner
./scripts/robot/run_policy/run_smolvla.sh \
  --arm left \
  --model your_username/trained_model_repo

# Or use demo scenario wrapper
./scripts/demo/01_close_empty.sh --arm left
```

**Flags:**
- `--arm left|right`: Which arm to control
- `--model <hf_repo_id>`: HuggingFace model repo
- `--record`: Record rollout as new dataset episodes
- `--resume`: Resume recording from last episode

---

## Configuration Files Summary

| File | Purpose | Used by steps |
|------|---------|---------------|
| `scripts/port.yaml` | USB port mappings | 2, 4, 6, 8 |
| `scripts/camera.yaml` | Camera serial numbers | 3, 4, 6, 8 |
| `scripts/dataset_config.yaml` | Dataset metadata | 5, 6, 7, 8 |
| `scripts/training_configs/*.yaml` | Training hyperparameters | 7 |

---

## Quick Reference

```bash
# Full workflow from scratch
./scripts/robot/register_port.sh                    # 1. Setup ports
sudo chmod 666 /dev/serial/by-id/*                  # 1b. Set permissions
./scripts/robot/calibrate.sh                        # 2. Calibrate arms

# BEFORE recording: configure cameras and dataset
lerobot-find-cameras                                # 3. Find camera serials
nano scripts/camera.yaml                            # 3. Edit camera config
nano scripts/dataset_config.yaml                    # 5. Edit dataset config

./scripts/robot/teleop.sh                           # 4. Test teleoperation
./scripts/robot/record.sh                           # 6. Record demos
./scripts/train.sh --policy groot                   # 7. Train model
./scripts/robot/replay.sh --episode 0               # 8a. Test replay
./scripts/robot/run_policy/run_groot.sh \
  --arm left \
  --model your_username/model                       # 8b. Deploy model
```

---


## Troubleshooting

### Port detection fails
```bash
# List available USB devices
ls /dev/serial/by-id/

# Manually edit scripts/port.yaml if needed
```

### Calibration lost after reboot
Calibration persists in `~/.cache/lerobot/`. To recalibrate:
```bash
./scripts/robot/calibrate.sh
```

### Training fails with CUDA OOM
Reduce `batch_size` in training config:
```yaml
batch_size: 16  # Was 32
```

### Model repo not found
Push trained model to HuggingFace:
```bash
cd outputs/groot/final  # Your output_dir
huggingface-cli upload your_username/model_name .
```

---

## Annotation Pipeline

After recording rollouts, annotate episodes with outcome labels and export analysis to Excel.

See **[README_demo.md](README_demo.md)** for the full step-by-step annotation pipeline:
- Visualize and annotate episodes (success/failure/retry) in the browser
- Merge annotations into dataset parquet files
- Push annotated dataset to HuggingFace
- Export per-episode analysis (jerkiness, retry count, failure notes) to Excel
