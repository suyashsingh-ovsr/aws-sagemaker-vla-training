# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Fork of Hugging Face LeRobot adding bimanual SO-101 robotic arm support. Two new modules added:
- `lerobot/src/lerobot/robots/bi_so_follower/` — controls left+right follower arms as single robot
- `lerobot/src/lerobot/teleoperators/bi_so_leader/` — leader arm controller for bimanual teleoperation

Helper scripts in `scripts/` wrap the `lerobot-*` CLI tools with config from `scripts/*.yaml`.

- `scripts/robot/` — robot operations: teleop, record, replay, calibrate, register_port
- `scripts/robot/run_policy/` — base runners: `run_smolvla.sh`, `run_act.sh`, `run_groot.sh`
- `scripts/demo/` — demo scenario scripts (call run_policy runners with specific model)
- `scripts/train.sh` — generic training script for all policy types
- `scripts/training_configs/` — training hyperparameters per policy type
- `scripts/jobs/huggingface/` — HuggingFace training job launchers (`launch_*.py`, `schedule_*.py`)

## Installation

`lerobot` is a path dependency pointing to `../lerobot` (the huggingface/lerobot clone at `/home/andrea/PycharmProjects/lerobot`).

```bash
# Create venv and install base + hardware deps
cd /home/andrea/PycharmProjects/lerobot
uv sync --extra feetech --extra intelrealsense
source .venv/bin/activate

# GR00T (flash-attn needs NVCC — must pre-install before syncing groot)
uv pip install flash-attn --no-build-isolation
uv sync --extra groot
```

Root `pyproject.toml` has `no-build-isolation-package = ["flash-attn"]`. To update lerobot, pull in `../lerobot`.

## Robot Workflow

Scripts in `scripts/` must be run from specific directories:

| Script | Run from | Purpose |
|--------|----------|---------|
| `scripts/robot/register_port.sh` | repo root | Find USB ports → writes `port.yaml` |
| `scripts/robot/calibrate.sh` | repo root | Calibrate all 4 arms |
| `scripts/robot/teleop.sh` | repo root | Live teleoperation (bimanual) |
| `scripts/robot/record.sh [--resume\|--reset]` | repo root | Record demos → dataset (bimanual) |
| `scripts/robot/record_single.sh --arm left\|right [--resume\|--reset]` | repo root | Record single arm |
| `scripts/robot/replay.sh --episode <N>` | repo root | Replay episode on follower |
| `scripts/train.sh --policy <type>` | repo root | Train policy (act\|diffusion\|groot\|smolvla) |
| `scripts/demo/0N_<name>.sh --arm left\|right` | repo root | Run demo model on robot |

Config read by scripts:
- `scripts/port.yaml` — 4 USB ports: `left_follower`, `left_leader`, `right_follower`, `right_leader`
- `scripts/camera.yaml` — 3 cameras: `front`, `wrist` (left), `wrist2` (right)
- `scripts/dataset_config.yaml` — HF username, repo name, episode count, task description
- `scripts/training_configs/{policy}_config.yaml` — training hyperparameters per policy type
- `scripts/calibration/robots/` — follower arm calibration JSONs (stored in lerobot's cache)
- `scripts/calibration/teleoperators/` — leader arm calibration JSONs (stored in lerobot's cache)

**Full workflow guide:** See `README.md` for complete step-by-step instructions.

## Testing

```bash
cd lerobot
make test-end-to-end DEVICE=cpu

# Single policy
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-train DEVICE=cpu
```

## Code Architecture

**Configuration system:** Draccus (dataclass-based). Each robot/teleop/policy has a paired `config_*.py` with a dataclass, and an implementation `*.py`. CLI entry points defined in `lerobot/pyproject.toml` under `[project.scripts]`.

**Adding a new robot type:** Create `lerobot/src/lerobot/robots/<name>/` with `config_<name>.py` (dataclass) and `<name>.py` (implementation). Follow `bi_so_follower/` as reference for bimanual patterns.

**Motor control:** Feetech motors via `lerobot/src/lerobot/motors/feetech/`. Requires `--extra feetech` when syncing.

**Policies:** ACT, Diffusion, TDMPC, SmolVLA, Pi0, VQBet, SAC — all in `lerobot/src/lerobot/policies/`. Train with `lerobot-train --policy.type=<name>`.

## Key CLI Commands

```bash
lerobot-find-port          # discover USB serial ports
lerobot-find-cameras       # discover cameras
lerobot-calibrate          # interactive joint range calibration
lerobot-teleoperate        # real-time teleoperation
lerobot-record             # record demonstration dataset
lerobot-replay             # replay recorded episodes
lerobot-train              # train a policy
lerobot-eval               # evaluate a policy
```

## Code Quality

Pre-commit hooks configured in `lerobot/.pre-commit-config.yaml`: Ruff, Black, mypy, bandit, gitleaks, typos. Run `pre-commit install` inside `lerobot/` before contributing to the library code.

## Updating lerobot

`lerobot` is a path dependency on `../lerobot` (clone of huggingface/lerobot). To update:

```bash
cd /home/andrea/PycharmProjects/lerobot
git pull
```

No subtree management needed.
