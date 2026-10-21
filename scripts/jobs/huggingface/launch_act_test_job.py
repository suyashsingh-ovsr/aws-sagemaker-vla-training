#!/usr/bin/env python3
"""One-shot test: act_air_bag_close_empty with fixed image + workers."""
import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

RENAME_MAP = '{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

cmd_str = (
    "apt-get update && apt-get install -y gcc && "
    "pip install -U pip && "
    'pip install "lerobot" && '
    "pip uninstall -y torchcodec && "
    "lerobot-train "
    "--policy.type=act "
    "--dataset.repo_id=andrejarden/air_bag_close_empty_box "
    "--dataset.video_backend=pyav "
    "--output_dir=outputs/train/act_air_bag_close_empty "
    "--job_name=act_air_bag_close_empty "
    "--policy.repo_id=andrejarden/act_air_bag_close_empty "
    "--policy.push_to_hub=true "
    "--policy.device=cuda "
    "--batch_size=32 "
    "--steps=30000 "
    "--save_freq=5000 "
    "--log_freq=100 "
    "--num_workers=16 "
    "--wandb.enable=false "
    "--policy.optimizer_lr=1e-5 "
    f"--rename_map='{RENAME_MAP}'"
)

job = api.run_job(
    image="pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime",
    command=["bash", "-lc", cmd_str],
    flavor="a10g-large",
    timeout="8h",
    secrets={"HF_TOKEN": HF_TOKEN},
    namespace="andrejarden",
)
print(f"Job ID : {job.id}")
print(f"Status : {job.status}")
