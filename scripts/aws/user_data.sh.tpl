#!/bin/bash
# EC2 user-data for Amazon Linux 2023: install Docker, pull the image from ECR, run it hardened.
# Values in @AT_SIGNS@ are filled in by deploy.sh. Progress is written to /var/log/user-data.log and to the serial
# console, so it can be read with `aws ec2 get-console-output` without SSH or SSM command access.
set -euo pipefail
exec > >(tee -a /var/log/user-data.log | tee /dev/console) 2>&1

echo "BANKCHURN: user-data started"
dnf install -y docker
systemctl enable --now docker
echo "BANKCHURN: docker installed"

aws ecr get-login-password --region @REGION@ | docker login --username AWS --password-stdin @REGISTRY@
docker pull @REGISTRY@/@REPO@:@TAG@
echo "BANKCHURN: image pulled (tag @TAG@)"

# XGB_NTHREADS: Python inside a container sees the host's CPUs, so pass the instance's vCPU count explicitly
docker run -d --name bank-churn-api --restart unless-stopped -p 8000:8000 \
  --memory @MEMORY@ --memory-swap @MEMORY@ \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  -e XGB_NTHREADS="$(nproc)" \
  @REGISTRY@/@REPO@:@TAG@
echo "BANKCHURN: container started (XGB_NTHREADS=$(nproc))"
