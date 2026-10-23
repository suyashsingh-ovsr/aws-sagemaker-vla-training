# ECR Image Reference

Pre-built training images hosted in AWS ECR (account `339879234866`, region `eu-north-1`).
No Docker or local build required — pass the image URI directly to the launcher.

---

## **GR00T**

```
339879234866.dkr.ecr.eu-north-1.amazonaws.com/robotics-groot:v1
```

```bash
python scripts/jobs/sagemaker/launch_training.py \
  --model-type groot \
  --ecr-image 339879234866.dkr.ecr.eu-north-1.amazonaws.com/robotics-groot:v1 \
  --dataset-repo-id ORG/DATASET \
  --hf-token hf_YOUR_TOKEN \
  --dry-run
```

---

## **SmolVLA**

```
339879234866.dkr.ecr.eu-north-1.amazonaws.com/robotics-smolvla:v1
```

```bash
python scripts/jobs/sagemaker/launch_training.py \
  --model-type smolvla \
  --ecr-image 339879234866.dkr.ecr.eu-north-1.amazonaws.com/robotics-smolvla:v1 \
  --dataset-repo-id ORG/DATASET \
  --hf-token hf_YOUR_TOKEN \
  --dry-run
```

---

Remove `--dry-run` once the configuration is verified. See `docs/SAGEMAKER_TRAINING.md` for the full parameter reference and preflight checklist.
