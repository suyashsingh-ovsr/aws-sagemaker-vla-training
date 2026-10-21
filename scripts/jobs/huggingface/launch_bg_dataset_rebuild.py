#!/usr/bin/env python3
"""Launch HF job to build air_bag_bg_p30_only and air_bag_bg_p50_only."""
import os
import sys
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "../airbag_dataset_build/rebuild_bg_p30_p50_only.py")
SCRIPT_REPO = "andrejarden/air_bag_bg_p30_only"

print("Creating repos if needed...")
for repo in ["andrejarden/air_bag_bg_p30_only", "andrejarden/air_bag_bg_p50_only"]:
    try:
        api.create_repo(repo_id=repo, repo_type="dataset", private=False, token=HF_TOKEN)
        print(f"  Created {repo}")
    except Exception:
        print(f"  {repo} already exists")

print("Uploading script to HF...")
upload_file(
    path_or_fileobj=SCRIPT_PATH,
    path_in_repo="rebuild_bg_p30_p50_only.py",
    repo_id=SCRIPT_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("Uploaded.")

cmd = (
    "pip install -U pip -q && "
    "pip install 'lerobot[smolvla]' -q && "
    "pip uninstall -y torchcodec && "
    "python3 - <<'PYEOF'\n"
    "from huggingface_hub import hf_hub_download\n"
    "p = hf_hub_download(\n"
    "    repo_id='andrejarden/air_bag_bg_p30_only',\n"
    "    filename='rebuild_bg_p30_p50_only.py',\n"
    "    repo_type='dataset',\n"
    "    force_download=True\n"
    ")\n"
    "exec(open(p).read())\n"
    "PYEOF"
)

print("Launching dataset rebuild job...")
job = api.run_job(
    image="python:3.11",
    command=["bash", "-lc", cmd],
    flavor="a10g-small",
    timeout="4h",
    secrets={"HF_API_TOKEN": HF_TOKEN},
    namespace="andrejarden",
)
print(f"Job ID : {job.id}")
print(f"Status : {job.status}")
print(f"URL    : {job.url}")
