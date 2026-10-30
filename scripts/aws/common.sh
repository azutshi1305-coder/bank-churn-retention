#!/usr/bin/env bash
# Shared settings and helpers for the AWS deployment scripts. Source this file; do not run it.
set -euo pipefail

export MSYS_NO_PATHCONV=1   # Git Bash on Windows would otherwise rewrite arguments that start with "/"
command -v aws >/dev/null 2>&1 || export PATH="$PATH:/c/Program Files/Amazon/AWSCLIV2"
command -v docker >/dev/null 2>&1 || export PATH="$PATH:/c/Program Files/Docker/Docker/resources/bin"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGION="${AWS_REGION:-ap-southeast-2}"   # the account's organization only allows this region; the CLI default may differ
APP="bank-churn-api"
PROJECT_TAG="bank-churn-retention"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.micro}"
MEMORY_LIMIT="${MEMORY_LIMIT:-512m}"
PORT=8000
ROLE_NAME="$APP-ec2-role"
PROFILE_NAME="$APP-ec2-profile"
SG_NAME="$APP-sg"
AMI_PARAM="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"

ERR_LOG="$(mktemp)"
trap 'rm -f "$ERR_LOG"' EXIT

# Account ids, ARNs and key-like strings never reach the terminal or the logs.
mask() { sed -E 's/[0-9]{12}/************/g; s#arn:aws[a-z-]*:[^ ")]*#<arn>#g; s/(AKIA|ASIA)[A-Z0-9]{16}/<key>/g'; }

# aws in the configured region; strips Windows CR; stderr is masked, shown, and kept so a denial can be recognised.
aws_() { aws --region "$REGION" "$@" 2> >(mask | tee -a "$ERR_LOG" >&2) | tr -d '\r'; }
# same, but quiet: for existence checks where failure is an expected answer
aws_q() { aws --region "$REGION" "$@" 2>/dev/null | tr -d '\r'; }

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

on_error() {
  echo "FAILED at line $1 (step: ${STEP:-unknown})" >&2
  if grep -q "explicit deny in a service control policy" "$ERR_LOG" 2>/dev/null; then
    echo "-> Blocked by the organization's service control policy (SCP). Stop here and report it; do not work around it." >&2
  fi
  echo "Resources created so far are tagged project=$PROJECT_TAG; run scripts/aws/teardown.sh to remove them." >&2
}

current_ip() {
  local ip; ip="$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '\r\n')"
  [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "could not determine your public IPv4 address"
  echo "$ip"
}

find_instances() {  # ids of this project's instances that are not terminated
  aws_q ec2 describe-instances --filters "Name=tag:project,Values=$PROJECT_TAG" "Name=tag:Name,Values=$APP" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text
}
find_sg() { aws_q ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text | grep -v '^None$' || true; }
