#!/usr/bin/env python3
"""
Launch 3 ACT training jobs for models 1-3 (concurrent).

  Job 1: act_air_bag_close_empty        — air_bag_close_empty_box (30k)
  Job 2: act_air_bag_close_empty_full   — air_bag_close_empty_full_box (30k)
  Job 3: act_air_bag_black_green_all    — air_bag_black_green_all_variants (30k)

Run launch_act_sequential.py for models 4-6 (WB base + BG injection).
"""
import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Unified camera name standard: front→camera1, wrist→camera2 (matches SmolVLA and all future policies)
RENAME_MAP = '{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

JOBS = [
    {
        "model": "act_air_bag_close_empty",
        "dataset": "air_bag_close_empty_box",
    },
    {
        "model": "act_air_bag_close_empty_full",
        "dataset": "air_bag_close_empty_full_box",
    },
    {
        "model": "act_air_bag_black_green_all",
        "dataset": "air_bag_black_green_all_variants",
    },
]

STEPS = 30000

for job in JOBS:
    model = job["model"]
    dataset = job["dataset"]

    cmd_str = (
        f"apt-get update && apt-get install -y gcc && "
        f"pip install -U pip && "
        f'pip install "lerobot" && '
        f"pip uninstall -y torchcodec && "
        f"lerobot-train "
        f"--policy.type=act "
        f"--dataset.repo_id=andrejarden/{dataset} "
        f"--dataset.video_backend=pyav "
        f"--output_dir=outputs/train/{model} "
        f"--job_name={model} "
        f"--policy.repo_id=andrejarden/{model} "
        f"--policy.push_to_hub=true "
        f"--policy.device=cuda "
        f"--batch_size=32 "
        f"--steps={STEPS} "
        f"--save_freq=5000 "
        f"--log_freq=100 "
        f"--num_workers=4 "
        f"--wandb.enable=false "
        f"--policy.optimizer_lr=1e-5 "
        f"--rename_map='{RENAME_MAP}'"
    )

    print(f"\n--- Launching: {model} ---")
    print(f"Dataset: andrejarden/{dataset}")

    job_info = api.run_job(
        image="python:3.11",
        command=["bash", "-lc", cmd_str],
        flavor="a10g-large",
        timeout="8h",
        secrets={"HF_TOKEN": HF_TOKEN},
        namespace="andrejarden",
    )
    print(f"Job ID : {job_info.id}")
    print(f"Status : {job_info.status}")

print("\nDone. 3 jobs running concurrently.")
print("Run launch_act_sequential.py for models 4-6.")
