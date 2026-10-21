#!/usr/bin/env python3
"""
Cancel all scheduled HF jobs for andrejarden, then schedule the p50 injection job at 23:45 CEST (21:45 UTC).

Injection chain:
  andrejarden/smolvla_wood_blue  (currently training)
    → air_bag_bg_p50_only  (8k steps)
    → andrejarden/smolvla_air_bag_wood_blue_p50  (matches deploy/05_wood_blue_p50.sh)
"""
import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_API_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
NAMESPACE = "andrejarden"

RENAME_MAP = '{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# --- Cancel all scheduled jobs ---
print("Fetching scheduled jobs...")
scheduled = list(api.list_jobs(namespace=NAMESPACE, status="scheduled"))
print(f"Found {len(scheduled)} scheduled job(s).")
for job in scheduled:
    print(f"  Cancelling {job.id} ...")
    api.cancel_job(job_id=job.id, namespace=NAMESPACE)
    print(f"  Cancelled.")

# --- Schedule p50 injection at 23:45 CEST = 21:45 UTC, June 30 2026 ---
SCHEDULE = "45 21 30 6 *"

cmd = (
    'pip install -U pip && '
    'pip install "lerobot[smolvla]" && '
    'pip uninstall -y torchcodec && '
    'lerobot-train '
    '--policy.path=andrejarden/smolvla_wood_blue '
    '--dataset.repo_id=andrejarden/air_bag_bg_p50_only '
    '--dataset.video_backend=pyav '
    '--output_dir=outputs/train/smolvla_air_bag_wood_blue_p50 '
    '--job_name=smolvla_air_bag_wood_blue_p50 '
    '--policy.repo_id=andrejarden/smolvla_air_bag_wood_blue_p50 '
    '--policy.push_to_hub=true '
    '--policy.device=cuda '
    '--policy.empty_cameras=1 '
    '--batch_size=32 '
    '--steps=8000 '
    '--save_freq=2000 '
    '--log_freq=100 '
    '--num_workers=4 '
    '--wandb.enable=false '
    '--policy.freeze_vision_encoder=true '
    '--policy.train_expert_only=true '
    '--policy.train_state_proj=true '
    '--policy.scheduler_warmup_steps=200 '
    '--policy.scheduler_decay_steps=8000 '
    f"--rename_map='{RENAME_MAP}'"
)

print(f"\nScheduling smolvla_air_bag_wood_blue_p50 at {SCHEDULE} UTC (23:45 CEST)...")
job = api.create_scheduled_job(
    image="python:3.11",
    command=["bash", "-lc", cmd],
    schedule=SCHEDULE,
    flavor="a10g-small",
    timeout="4h",
    secrets={"HF_TOKEN": HF_TOKEN},
    namespace=NAMESPACE,
)
print(f"Job ID : {job.id}")
print(f"Status : {job.status}")
print("\nDone.")
