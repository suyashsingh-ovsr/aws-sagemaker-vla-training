"""
Launch GR00T fine-tuning job on AWS SageMaker.

Usage:
    python scripts/jobs/sagemaker/launch_groot_training.py
    python scripts/jobs/sagemaker/launch_groot_training.py --steps 10000
    python scripts/jobs/sagemaker/launch_groot_training.py --hf-token hf_xxx
"""

import argparse
import datetime
import boto3

AWS_REGION = "eu-north-1"
AWS_PROFILE = "datascience"
ACCOUNT_ID = "339879234866"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole"
ECR_IMAGE = f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/oversonic-groot:latest"
S3_BUCKET = "oversonic-groot-training"
INSTANCE_TYPE = "ml.g6e.xlarge"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--hf-token", default="", help="HuggingFace token for pushing checkpoints")
    parser.add_argument("--hf-repo-prefix", default="Suyash44/groot-blue-cylinder")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"groot-blue-cylinder-{timestamp}"

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    sm = session.client("sagemaker")

    hyperparameters = {
        "STEPS": str(args.steps),
        "BATCH_SIZE": str(args.batch_size),
        "SAVE_FREQ": str(args.save_freq),
        "EMBODIMENT_TAG": args.embodiment_tag,
        "DATASET_REPO_ID": "Suyash44/blue_cylinder_grasp_2",
        "HF_REPO_PREFIX": args.hf_repo_prefix,
    }
    if args.hf_token:
        hyperparameters["HF_TOKEN"] = args.hf_token

    # SageMaker passes hyperparameters as CLI args to the container entrypoint.
    # Our train.sh reads them as env vars, so we pass via Environment instead.
    response = sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": ECR_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{S3_BUCKET}/datasets/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{S3_BUCKET}/output/",
        },
        CheckpointConfig={
            "S3Uri": f"s3://{S3_BUCKET}/checkpoints/{job_name}/",
            "LocalPath": "/opt/ml/checkpoints",
        },
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 8 * 3600,  # 8 hour max
        },
        Environment=hyperparameters,
    )

    print(f"Launched: {job_name}")
    print(f"ARN: {response['TrainingJobArn']}")
    print(f"Monitor: https://eu-north-1.console.aws.amazon.com/sagemaker/home?region=eu-north-1#/jobs/{job_name}")
    print(f"Checkpoints: s3://{S3_BUCKET}/checkpoints/{job_name}/")
    print(f"Model output: s3://{S3_BUCKET}/output/{job_name}/output/model.tar.gz")


if __name__ == "__main__":
    main()
