#!/usr/bin/env python3
"""Launch HF job to rebuild air_bag_wood_blue_plus30/50 datasets (v4 fix: wood/blue was av1)."""
import os
import sys
from huggingface_hub import HfApi, hf_hub_download, upload_file

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Upload script to HF so the job can fetch it
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "../airbag_dataset_build/hf_job_rebuild_p30_p50_v4.py")
SCRIPT_REPO = "andrejarden/air_bag_wood_blue_plus30_black_green"

print("Uploading v4 script to HF dataset repo...")
upload_file(
    path_or_fileobj=SCRIPT_PATH,
    path_in_repo="hf_job_rebuild_p30_p50_v4.py",
    repo_id=SCRIPT_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("Uploaded.")

cmd = (
    "apt-get update -q && apt-get install -y -q ffmpeg python3 python3-pip && "
    "pip install -U pip -q && "
    "pip install 'lerobot[smolvla]' -q && "
    "pip uninstall -y torchcodec && "
    "python3 - <<'PYEOF'\n"
    "from huggingface_hub import hf_hub_download\n"
    "p = hf_hub_download(\n"
    "    repo_id='andrejarden/air_bag_wood_blue_plus30_black_green',\n"
    "    filename='hf_job_rebuild_p30_p50_v4.py',\n"
    "    repo_type='dataset',\n"
    "    force_download=True\n"
    ")\n"
    "exec(open(p).read())\n"
    "PYEOF"
)

print("Launching job...")
job = api.run_job(
    image="ubuntu:22.04",
    command=["bash", "-lc", cmd],
    flavor="a10g-small",
    timeout="6h",
    secrets={"HF_TOKEN": HF_TOKEN},
    namespace="andrejarden",
)
print(f"Job ID : {job.id}")
print(f"Status : {job.status}")
print(f"URL    : {job.url}")
