#!/usr/bin/env bash
# Run FaceMesh training on one AWS GPU machine, for both validation splits.
#
# One time setup (from your laptop):
#   1. Make an S3 bucket and upload the dataset:  aws s3 cp data/coma.npz s3://YOUR_BUCKET/data/coma.npz
#   2. Launch an EC2 g5.xlarge (one NVIDIA A10G GPU, 24 GB) with the
#      "AWS Deep Learning Base GPU AMI (Ubuntu)". It already has the NVIDIA driver,
#      Docker and the NVIDIA container toolkit. Give it an IAM role that can read and write the bucket.
#   3. SSH in, git clone this repo, cd into it, then run:  BUCKET=YOUR_BUCKET ./scripts/train_on_aws.sh
#   4. STOP THE INSTANCE when it finishes. You pay per hour while it is running.

set -euo pipefail   # stop on the first error, on unset variables, and on failures inside pipes

BUCKET="${BUCKET:?set BUCKET to your S3 bucket name}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-60}"

mkdir -p data runs
aws s3 cp "s3://${BUCKET}/data/coma.npz" data/coma.npz

docker build -t facemesh .

for SPLIT in subject interpolation; do
    # --gpus all gives the container the GPU; --shm-size gives DataLoader workers room to share batches;
    # -v mounts host folders so the checkpoint and metrics survive after the container is removed
    docker run --rm --gpus all --shm-size=8g \
        -v "$(pwd)/data:/data" -v "$(pwd)/runs:/runs" \
        facemesh --data /data/coma.npz --out_dir "/runs/${RUN_NAME}/${SPLIT}" \
        --split "${SPLIT}" --epochs "${EPOCHS}" --batch_size 64 --amp
done

aws s3 sync "runs/${RUN_NAME}" "s3://${BUCKET}/runs/${RUN_NAME}"
echo "saved checkpoints and metrics to s3://${BUCKET}/runs/${RUN_NAME}"
echo "REMINDER: stop this instance now (EC2 console, or: sudo shutdown -h now) so you stop paying for it."
