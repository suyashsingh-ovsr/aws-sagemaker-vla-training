#!/usr/bin/env python3
"""Launch p30 and p50 injection jobs on a100-large.

  Job B: smolvla_wood_blue → air_bag_bg_p30_only (6k) → smolvla_air_bag_wood_blue_p30
  Job C: smolvla_wood_blue → air_bag_bg_p50_only (8k) → smolvla_air_bag_wood_blue_p50
"""
import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_API_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

RENAME_MAP = '{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

JOBS = [
    {
        "label": "B — p30 injection",
        "dataset": "air_bag_bg_p30_only",
        "model": "smolvla_air_bag_wood_blue_p30",
        "steps": 6000,
    },
    {
        "label": "C — p50 injection",
        "dataset": "air_bag_bg_p50_only",
        "model": "smolvla_air_bag_wood_blue_p50",
        "steps": 8000,
    },
]

for j in JOBS:
    steps = j["steps"]
    cmd = (
        f'pip install -U pip && '
        f'pip install "lerobot[smolvla]" && '
        f'pip uninstall -y torchcodec && '
        f'lerobot-train '
        f'--policy.path=andrejarden/smolvla_wood_blue '
        f'--dataset.repo_id=andrejarden/{j["dataset"]} '
        f'--dataset.video_backend=pyav '
        f'--output_dir=outputs/train/{j["model"]} '
        f'--job_name={j["model"]} '
        f'--policy.repo_id=andrejarden/{j["model"]} '
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
        f'--policy.scheduler_warmup_steps=200 '
        f'--policy.scheduler_decay_steps={steps} '
        f"--rename_map='{RENAME_MAP}'"
    )

    print(f"\n--- {j['label']} ---")
    job = api.run_job(
        image="python:3.11",
        command=["bash", "-lc", cmd],
        flavor="a100-large",
        timeout="4h",
        secrets={"HF_TOKEN": HF_TOKEN},
        namespace="andrejarden",
    )
    print(f"Job ID : {job.id}")
    print(f"Status : {job.status}")

print("\nDone.")
