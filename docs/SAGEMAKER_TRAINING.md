# Multi-model VLA training on AWS SageMaker

**Models:** NVIDIA GR00T N1.7 and Hugging Face SmolVLA
**Workflow:** Select a model and dataset → build its Docker image → publish to Amazon ECR → submit a SageMaker training job → inspect checkpoints and results in Amazon S3.

> **Validation status:** This is an implementation and operator guide, **not a claim that either model has successfully trained in SageMaker**. Python/Bash syntax and some GR00T launcher dry-run checks were performed during development. The Docker image builds, pretrained-model compatibility, dataset schemas, GPU memory requirements, and end-to-end SageMaker execution still need validation by the first operator. Do not skip the dry run and preflight review.

## 1. Repository layout

```text
.
├── docker/
│   ├── groot/
│   │   ├── Dockerfile
│   │   ├── train.sh
│   │   └── nvrtc_workaround.py
│   └── smolvla/
│       ├── Dockerfile
│       └── train.sh
├── configs/
│   ├── groot.yaml
│   └── smolvla.yaml
├── scripts/jobs/sagemaker/
│   ├── launch_training.py
│   └── requirements.txt
├── docs/
│   └── SAGEMAKER_TRAINING.md
└── vendor/lerobot/
```

GR00T and SmolVLA use **separate images** to isolate dependencies. Both are launched with the **same** `scripts/jobs/sagemaker/launch_training.py` script. The legacy `launch_groot_training.py`, if present, is not used by this guide.

## 2. Prerequisites and access

On the **colleague's machine**, install:

- Git
- Docker Engine with enough free disk space for a CUDA/PyTorch image
- AWS CLI v2, configured for AWS IAM Identity Center (SSO) or other approved credentials
- Python 3 and `venv`/`pip` for the **launcher only**

The AWS identity used by the operator needs permission to use **SageMaker Training Jobs**, **Amazon ECR**, **Amazon S3**, and to pass the approved **SageMaker execution role** (`iam:PassRole`). The execution role itself needs ECR image pull, S3 write permissions, and any other resources used by the training job. The selected GPU instance must be offered in the AWS Region and the account must have adequate SageMaker quota.

You also need access to the chosen Hugging Face dataset and pretrained model. Public datasets/models are easiest for the first test. Private or gated resources require a separate, approved secret-delivery design; **this repository does not yet implement that**.

All example account settings below reflect the project's current environment. Colleagues in different AWS accounts must replace them with their own approved values.

## 3. Clone and install launcher dependencies

```bash
# Use the approved GitHub repository and access method.
git clone git@github.com:suyashsingh-ovsr/groot-sagemaker-training.git
cd groot-sagemaker-training

# The launcher runs on your computer; the model runs in the Docker container on SageMaker.
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/jobs/sagemaker/requirements.txt
```

The launcher requirements contain `boto3` and `PyYAML`. **Do not install GR00T or SmolVLA training dependencies locally** just to submit jobs: those are installed by the corresponding Dockerfile.

## 4. Authenticate and confirm the AWS account

The example configuration uses the `datascience` AWS CLI profile and region `eu-north-1`.

```bash
aws sso login --profile datascience
aws sts get-caller-identity --profile datascience
aws configure get region --profile datascience
```

Check the reported AWS account ID against the account you're authorized to use. The current launcher defaults to account `339879234866`, region `eu-north-1`, a role named `SageMakerExecutionRole`, and S3 bucket `oversonic-groot-training`. **These are defaults, not universally valid settings.** Supply overrides as appropriate.

## 5. Choose your model and dataset

The model-specific defaults live in:

- `configs/groot.yaml` — GR00T base model, Docker path, ECR repository, instance, steps, batch size, checkpoint frequency, embodiment tag.
- `configs/smolvla.yaml` — SmolVLA base model, Docker path, ECR repository, instance, steps, batch size, checkpoint frequency.

You can edit these defaults in your own branch, **or override them per job** with command-line options. Prefer command-line overrides for experiments so that the shared defaults stay consistent.

| Option | Meaning | Example |
|---|---|---|
| `--model-type` | Which container/policy to run | `groot`, `smolvla` |
| `--dataset-repo-id` | Hugging Face dataset repository | `ORG/ROBOT_DATASET` |
| `--base-model` | Starting pretrained policy | `nvidia/GR00T-N1.7-3B` |
| `--steps` | Training update steps | `10000` |
| `--batch-size` | Training batch size | `2` |
| `--save-freq` | Checkpoint interval in steps | `2000` |
| `--instance-type` | SageMaker training GPU instance | `ml.g6e.xlarge` |
| `--embodiment-tag` | GR00T embodiment tag | `new_embodiment` |
| `--hf-policy-repo` | Policy repository identifier (not an upload token) | `ORG/MODEL_NAME` |
| `--ecr-image` | Full ECR image URI override | `ACCOUNT.dkr.ecr.REGION.amazonaws.com/robotics-groot:v1` |
| `--aws-profile`, `--aws-region`, `--aws-account-id` | AWS identity and destination | `datascience`, `eu-north-1`, `ACCOUNT_ID` |
| `--role-arn`, `--s3-bucket` | Training role and storage | Approved role ARN, bucket name |
| `--image-tag` | Image tag, if using default image URI construction | `v1` |
| `--job-prefix` | SageMaker job prefix | `vla-groot-test` |
| `--volume-size` | SageMaker storage allocation in GB | `100` |
| `--max-runtime-hours` | Job runtime limit | `8` |
| `--dry-run` | Print configuration; do **not** create a job | Flag |

**Dataset compatibility matters:** the dataset must use a schema, features, cameras, actions, and annotations that the selected policy can ingest. A dataset that works with GR00T is not automatically valid for SmolVLA. The chosen GPU and batch size also need to fit the model's memory requirements.

## 6. Review the container and training entrypoints

| Model | Dockerfile | Entrypoint |
|---|---|---|
| GR00T | `docker/groot/Dockerfile` | `docker/groot/train.sh` |
| SmolVLA | `docker/smolvla/Dockerfile` | `docker/smolvla/train.sh` |

Docker builds **must be run from the repository root**: both Dockerfiles copy `vendor/lerobot/` into the image. The GR00T Dockerfile also installs `nvrtc_workaround.py`. The image's entrypoint reads the variables produced by the SageMaker launcher.

The SmolVLA entrypoint is intended to load a pretrained policy with `--policy.path=<repo>`. The GR00T entrypoint is intended to use `--policy.type=groot` with a base model and embodiment tag. **Model-specific argument compatibility still needs to be confirmed against the exact vendored LeRobot version before the first paid training run.**

## 7. Build the selected Docker image (only when ready to train)

**GR00T:**

```bash
docker build -f docker/groot/Dockerfile -t robotics-groot:v1 .
```

**SmolVLA:**

```bash
docker build -f docker/smolvla/Dockerfile -t robotics-smolvla:v1 .
```

Only build the model you intend to use. Building may download several GB and take a long time. If a build fails, diagnose the specific failed layer before uploading or launching anything. **Neither Dockerfile is currently certified as successfully built.**

## 8. Create the ECR repositories, authenticate, and push

The launcher's default ECR repository names are `robotics-groot` and `robotics-smolvla`. ECR repositories must exist in the **same AWS account and region** selected for SageMaker.

```bash
export PROFILE=datascience
export REGION=eu-north-1
export ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
export REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

# Create the selected repository if it does not already exist.
# For GR00T:
aws ecr describe-repositories --repository-names robotics-groot --profile "$PROFILE" --region "$REGION" \
  || aws ecr create-repository --repository-name robotics-groot --profile "$PROFILE" --region "$REGION"

aws ecr get-login-password --profile "$PROFILE" --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

docker tag robotics-groot:v1 "$REGISTRY/robotics-groot:v1"
docker push "$REGISTRY/robotics-groot:v1"
```

For **SmolVLA**, perform the corresponding operation with `robotics-smolvla`, and tag/push `robotics-smolvla:v1`. If your organization already maintains approved ECR repositories, use those and pass their full image URI through `--ecr-image` instead of creating new repositories.

Verify that the tag exists:

```bash
aws ecr describe-images --repository-name robotics-groot --image-ids imageTag=v1 \
  --profile "$PROFILE" --region "$REGION"
```

Use `robotics-smolvla` in the verification command for SmolVLA.

## 9. Dry-run the SageMaker job configuration

A dry run displays the exact `CreateTrainingJob` request without calling SageMaker. **Always inspect** its `TrainingImage`, `RoleArn`, `InstanceType`, `Environment`, `CheckpointConfig`, and S3 output path.

**GR00T:**

```bash
.venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type groot \
  --dataset-repo-id ORG/COMPATIBLE_GROOT_DATASET \
  --steps 10000 \
  --batch-size 2 \
  --save-freq 5000 \
  --dry-run
```

**SmolVLA:**

```bash
.venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type smolvla \
  --dataset-repo-id ORG/COMPATIBLE_SMOLVLA_DATASET \
  --steps 10000 \
  --batch-size 2 \
  --save-freq 2000 \
  --dry-run
```

If using a different approved ECR image, add `--ecr-image FULL_ECR_URI`. For a different AWS account or role, add `--aws-account-id`, `--role-arn`, `--aws-profile`, `--aws-region` and `--s3-bucket` as required. Do not assume the configuration's default GPU is sufficient for every experiment.

## 10. Launch training on SageMaker

Once the following have been verified, **remove `--dry-run`** from the relevant command:

1. The Docker build succeeded and its exact tag exists in ECR.
2. The account, region, role, bucket and ECR image URI are correct.
3. The operator has permission to create jobs and pass the training role.
4. The chosen instance type has sufficient quota, memory and expected budget.
5. The base model and dataset are accessible and compatible.
6. The model-specific training arguments were checked against vendored LeRobot.

For example, after substituting a compatible dataset:

```bash
.venv/bin/python scripts/jobs/sagemaker/launch_training.py \
  --model-type groot \
  --dataset-repo-id ORG/COMPATIBLE_GROOT_DATASET \
  --steps 10000 \
  --batch-size 2
```

The launcher prints the job name and ARN after AWS accepts the request. **Submitting a job does not mean training has succeeded.** Inspect SageMaker job status and CloudWatch logs, and review errors if the container exits.

## 11. Checkpoints and final artifacts

Both training scripts currently use these paths:

| Purpose | Container location | AWS destination |
|---|---|---|
| Intermediate checkpoints | `/opt/ml/checkpoints/output` | `s3://BUCKET/checkpoints/JOB_NAME/` |
| Final packaged model data | `/opt/ml/model` | `s3://BUCKET/output/JOB_NAME/output/model.tar.gz` |

SageMaker's `CheckpointConfig` synchronizes files from `/opt/ml/checkpoints` to the specified S3 location while the job runs. On successful completion, the training entrypoint copies the run output into `/opt/ml/model` for SageMaker to package. **This duplicates files on local storage**, so set `--volume-size` with sufficient headroom. The final archive may contain multiple checkpoints, not merely the smallest inference-only weights.

**Automatic training resume from a prior SageMaker job is not implemented in the shared launcher.** An S3 checkpoint backup is not the same as verified resume support. Do not promise automatic recovery from interruptions until this has been added and tested.

## 12. Hugging Face access and secrets

The training scripts fetch the dataset and pretrained model using Hugging Face repository IDs. This is practical for **public and accessible** repositories. **Private and gated access is not yet implemented in this template.** Do not commit API keys, Hugging Face tokens, AWS access keys or `.env` files to Git, and do not pass secrets through the launcher or `CreateTrainingJob.Environment` without an approved secure mechanism.

`PUSH_TO_HUB` is disabled in the shared launcher. A policy repository ID does **not** authorize a Hub upload by itself; final outputs are saved to S3.

## 13. Common problems

| Symptom | Check |
|---|---|
| `No module named yaml` or `boto3` | Install `scripts/jobs/sagemaker/requirements.txt` inside the launcher's virtualenv |
| ECR image not found | Verify account, region, repository and tag; confirm the image was pushed |
| `AccessDenied` / role errors | AWS SSO session, SageMaker permissions, ECR/S3 role policies, `iam:PassRole` |
| SageMaker instance unavailable | Region availability and SageMaker training instance quota |
| Hugging Face 401/403 | Dataset/model visibility and authentication; private models are not yet supported |
| LeRobot argument/configuration error | Check the vendored LeRobot API and model-specific `train.sh` |
| CUDA OOM | Reduce batch size, choose a larger GPU or adjust model-specific settings |
| Missing checkpoints | Check actual model output path, checkpoint creation and SageMaker sync logs |
| No final `model.tar.gz` | Inspect job status, errors, runtime and `/opt/ml/model` contents |
| Job costs unexpectedly high | Confirm instance type, runtime limit and stop failed/unneeded jobs |

## 14. Development and handoff standards

- Keep model-specific dependencies in separate Docker images.
- Keep AWS-launching logic shared in `launch_training.py`.
- Do not commit secrets, training datasets, videos, virtual environments or checkpoints.
- Keep the source and dependency versions pinned or reviewed when updating images.
- Review changes on a feature branch and merge through your team's normal process.
- For new models, add a Dockerfile, entrypoint, model config and an explicit compatibility test.
- Before declaring the workflow production-ready, perform **both** image builds, a compatible-data smoke test, and at least one end-to-end SageMaker run for each model.

**Security and ownership:** Access to this repository, company source code, dataset repositories and AWS resources must be authorized by the organization, even when the GitHub repository is private.
