#!/usr/bin/env python3
"""Schedule 4 HF training jobs for airbag models 2-5, staggered 5h apart."""

import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

RENAME_MAP = '{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# Jobs 2-5, staggered 5h apart starting 10:35 UTC today (2026-06-26)
# Cron: "minute hour day month *"
JOBS = [
    {
        "model": "smolvla_air_bag_close_empty_full",
        "dataset": "air_bag_close_empty_full_box",
        "schedule": "10 11 26 6 *",   # 11:10 UTC today
    },
    {
        "model": "smolvla_air_bag_black_green_all",
        "dataset": "air_bag_black_green_all_variants",
        "schedule": "10 16 26 6 *",   # 16:10 UTC today (+5h)
    },
    {
        "model": "smolvla_air_bag_wood_blue_p30",
        "dataset": "air_bag_wood_blue_plus30_black_green",
        "schedule": "10 21 26 6 *",   # 21:10 UTC today (+10h)
    },
    {
        "model": "smolvla_air_bag_wood_blue_p50",
        "dataset": "air_bag_wood_blue_plus50_black_green",
        "schedule": "10 2 27 6 *",    # 02:10 UTC tomorrow (+15h)
    },
]

for job in JOBS:
    model = job["model"]
    dataset = job["dataset"]
    schedule = job["schedule"]

    cmd_str = (
        f'pip install -U pip && '
        f'pip install "lerobot[smolvla]" && '
        f'pip uninstall -y torchcodec && '
        f'lerobot-train '
        f'--policy.path=lerobot/smolvla_base '
        f'--dataset.repo_id=andrejarden/{dataset} '
        f'--dataset.video_backend=pyav '
        f'--output_dir=outputs/train/{model} '
        f'--job_name={model} '
        f'--policy.repo_id=andrejarden/{model} '
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

    print(f"\n--- Scheduling: {model} ---")
    print(f"Dataset : andrejarden/{dataset}")
    print(f"Schedule: {schedule} (UTC)")

    job_info = api.create_scheduled_job(
        image="python:3.11",
        command=["bash", "-lc", cmd_str],
        schedule=schedule,
        flavor="a10g-small",
        timeout="6h",
        secrets={"HF_TOKEN": HF_TOKEN},
        namespace="andrejarden",
    )
    print(f"Job ID  : {job_info.id}")
    print(f"Status  : {job_info.status}")

print("\nDone. All 4 jobs scheduled.")
