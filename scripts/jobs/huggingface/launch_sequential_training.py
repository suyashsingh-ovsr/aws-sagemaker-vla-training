#!/usr/bin/env python3
"""
Sequential training jobs for wood/blue → BG injection:

  Job A: smolvla_base → so101_airbag_all_positions_30fps_h264 (15k) → smolvla_wood_blue
  Job B: smolvla_wood_blue → air_bag_bg_p30_only (6k)  → smolvla_wood_blue_plus_bg30
  Job C: smolvla_wood_blue → air_bag_bg_p50_only (8k)  → smolvla_wood_blue_plus_bg50

Jobs B/C scheduled +3h after A (a10g-small, 15k steps ~ 2h).
Run this script AFTER air_bag_bg_p30_only and air_bag_bg_p50_only are built.
"""
import os
import sys
from datetime import datetime, timezone, timedelta
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

RENAME_MAP = '{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# Exclude ep 6 (length=2) and eps 75-79 (timestamp drift past video duration)
WB_EPISODES = list(range(0, 6)) + list(range(7, 75))
WB_EPISODES_STR = "[" + ",".join(str(e) for e in WB_EPISODES) + "]"


def make_cmd(dataset, model, steps, policy_path, warmup, decay):
    return (
        f'pip install -U pip && '
        f'pip install "lerobot[smolvla]" && '
        f'pip uninstall -y torchcodec && '
        f'lerobot-train '
        f'--policy.path={policy_path} '
        f'--dataset.repo_id=andrejarden/{dataset} '
        f'--dataset.video_backend=pyav '
        f'--output_dir=outputs/train/{model} '
        f'--job_name={model} '
        f'--policy.repo_id=andrejarden/{model} '
        f'--policy.push_to_hub=true '
        f'--policy.device=cuda '
        f'--policy.empty_cameras=1 '
        f'--batch_size=32 '
        f'--steps={steps} '
        f'--save_freq=2000 '
        f'--log_freq=100 '
        f'--num_workers=4 '
        f'--wandb.enable=false '
        f'--policy.freeze_vision_encoder=true '
        f'--policy.train_expert_only=true '
        f'--policy.train_state_proj=true '
        f'--policy.scheduler_warmup_steps={warmup} '
        f'--policy.scheduler_decay_steps={decay} '
        f"--rename_map='{RENAME_MAP}'"
    )


# Job A — wood/blue base (run immediately)
cmd_a = (
    f'pip install -U pip && '
    f'pip install "lerobot[smolvla]" && '
    f'pip uninstall -y torchcodec && '
    f'lerobot-train '
    f'--policy.path=lerobot/smolvla_base '
    f'--dataset.repo_id=andrejarden/so101_airbag_all_positions_30fps_h264 '
    f'--dataset.video_backend=pyav '
    f'--dataset.episodes={WB_EPISODES_STR} '
    f'--output_dir=outputs/train/smolvla_wood_blue '
    f'--job_name=smolvla_wood_blue '
    f'--policy.repo_id=andrejarden/smolvla_wood_blue '
    f'--policy.push_to_hub=true '
    f'--policy.device=cuda '
    f'--policy.empty_cameras=1 '
    f'--batch_size=32 '
    f'--steps=15000 '
    f'--save_freq=5000 '
    f'--log_freq=100 '
    f'--num_workers=4 '
    f'--wandb.enable=false '
    f'--policy.freeze_vision_encoder=true '
    f'--policy.train_expert_only=true '
    f'--policy.train_state_proj=true '
    f'--policy.scheduler_warmup_steps=500 '
    f'--policy.scheduler_decay_steps=15000 '
    f"--rename_map='{RENAME_MAP}'"
)

# Schedule B/C 3h from now
now_utc = datetime.now(timezone.utc)
inject_time = now_utc + timedelta(hours=3)
inject_cron = f"{inject_time.minute} {inject_time.hour} {inject_time.day} {inject_time.month} *"

cmd_b = make_cmd(
    dataset="air_bag_bg_p30_only",
    model="smolvla_wood_blue_plus_bg30",
    steps=6000,
    policy_path="andrejarden/smolvla_wood_blue",
    warmup=200,
    decay=6000,
)

cmd_c = make_cmd(
    dataset="air_bag_bg_p50_only",
    model="smolvla_wood_blue_plus_bg50",
    steps=8000,
    policy_path="andrejarden/smolvla_wood_blue",
    warmup=200,
    decay=8000,
)

JOBS = [
    {"label": "A — wood/blue base", "cmd": cmd_a, "scheduled": False},
    {"label": "B — BG p30 injection", "cmd": cmd_b, "scheduled": True},
    {"label": "C — BG p50 injection", "cmd": cmd_c, "scheduled": True},
]

for job_def in JOBS:
    print(f"\n--- {job_def['label']} ---")
    if job_def["scheduled"]:
        print(f"Schedule: {inject_cron} UTC (+3h from now)")
        job = api.create_scheduled_job(
            image="python:3.11",
            command=["bash", "-lc", job_def["cmd"]],
            schedule=inject_cron,
            flavor="a10g-small",
            timeout="4h",
            secrets={"HF_TOKEN": HF_TOKEN},
            namespace="andrejarden",
        )
    else:
        job = api.run_job(
            image="python:3.11",
            command=["bash", "-lc", job_def["cmd"]],
            flavor="a10g-small",
            timeout="4h",
            secrets={"HF_TOKEN": HF_TOKEN},
            namespace="andrejarden",
        )
    print(f"Job ID : {job.id}")
    print(f"Status : {job.status}")

print("\nDone.")
print(f"Jobs B/C scheduled at {inject_cron} UTC.")
print("NOTE: run this script only after air_bag_bg_p30_only and air_bag_bg_p50_only are built.")
