#!/usr/bin/env python3
"""
Sequential ACT training for WB base + BG injection:

  Job A: act_wood_blue         — so101_airbag_all_positions_30fps_h264 (30k)
  Job B: act_wood_blue_plus_bg30 — act_wood_blue → air_bag_bg_p30_only (6k fine-tune)
  Job C: act_wood_blue_plus_bg50 — act_wood_blue → air_bag_bg_p50_only (8k fine-tune)

Jobs B/C scheduled +5h after A (a10g-small, 30k steps ~ 4h).
"""
import os
import sys
from datetime import datetime, timezone, timedelta
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Unified camera name standard: front→camera1, wrist→camera2 (matches SmolVLA and all future policies)
RENAME_MAP = '{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# Exclude ep 6 (length=2) and eps 75-79 (timestamp drift)
WB_EPISODES = list(range(0, 6)) + list(range(7, 75))
WB_EPISODES_STR = "[" + ",".join(str(e) for e in WB_EPISODES) + "]"

# Job A — WB base from scratch
cmd_a = (
    f"pip install -U pip && "
    f'pip install "lerobot[feetech]" && '
    f"pip uninstall -y torchcodec && "
    f"lerobot-train "
    f"--policy.type=act "
    f"--dataset.repo_id=andrejarden/so101_airbag_all_positions_30fps_h264 "
    f"--dataset.video_backend=pyav "
    f"--dataset.episodes={WB_EPISODES_STR} "
    f"--output_dir=outputs/train/act_wood_blue "
    f"--job_name=act_wood_blue "
    f"--policy.repo_id=andrejarden/act_wood_blue "
    f"--policy.push_to_hub=true "
    f"--policy.device=cuda "
    f"--batch_size=32 "
    f"--steps=30000 "
    f"--save_freq=5000 "
    f"--log_freq=100 "
    f"--num_workers=4 "
    f"--wandb.enable=false "
    f"--policy.optimizer_lr=1e-5 "
    f"--rename_map='{RENAME_MAP}'"
)


def make_inject_cmd(dataset, model, steps, warmup):
    return (
        f"pip install -U pip && "
        f'pip install "lerobot[feetech]" && '
        f"pip uninstall -y torchcodec && "
        f"lerobot-train "
        f"--policy.path=andrejarden/act_wood_blue "
        f"--dataset.repo_id=andrejarden/{dataset} "
        f"--dataset.video_backend=pyav "
        f"--output_dir=outputs/train/{model} "
        f"--job_name={model} "
        f"--policy.repo_id=andrejarden/{model} "
        f"--policy.push_to_hub=true "
        f"--policy.device=cuda "
        f"--batch_size=32 "
        f"--steps={steps} "
        f"--save_freq=2000 "
        f"--log_freq=100 "
        f"--num_workers=4 "
        f"--wandb.enable=false "
        f"--policy.optimizer_lr=1e-5 "
        f"--rename_map='{RENAME_MAP}'"
    )


cmd_b = make_inject_cmd("air_bag_bg_p30_only", "act_wood_blue_plus_bg30", steps=6000, warmup=200)
cmd_c = make_inject_cmd("air_bag_bg_p50_only", "act_wood_blue_plus_bg50", steps=8000, warmup=200)

now_utc = datetime.now(timezone.utc)
inject_time = now_utc + timedelta(hours=5)
inject_cron = f"{inject_time.minute} {inject_time.hour} {inject_time.day} {inject_time.month} *"

JOBS = [
    {"label": "A — WB base", "cmd": cmd_a, "scheduled": False},
    {"label": "B — BG p30 injection", "cmd": cmd_b, "scheduled": True},
    {"label": "C — BG p50 injection", "cmd": cmd_c, "scheduled": True},
]

for job_def in JOBS:
    print(f"\n--- {job_def['label']} ---")
    if job_def["scheduled"]:
        print(f"Schedule: {inject_cron} UTC (+5h from now)")
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
            timeout="10h",
            secrets={"HF_TOKEN": HF_TOKEN},
            namespace="andrejarden",
        )
    print(f"Job ID : {job.id}")
    print(f"Status : {job.status}")

print(f"\nDone. Jobs B/C scheduled at {inject_cron} UTC (+5h).")
