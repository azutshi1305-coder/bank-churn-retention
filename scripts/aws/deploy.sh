#!/usr/bin/env bash
# Deploy the churn API to AWS: an ECR repository plus one EC2 instance running the container.
#
#   scripts/aws/deploy.sh                       build the image here, push it to ECR (with a network preflight), then deploy
#   scripts/aws/deploy.sh --use-ecr-image [TAG] skip the local build and push; deploy an image already in ECR
#                                               (TAG defaults to `git rev-parse --short=7 HEAD`, the tag CI pushes)
#
# Creates (all tagged project=bank-churn-retention): ECR repo (immutable tags, scan on push, keep last 3 images),
# IAM role + instance profile (ECR read-only, SSM core), a security group (TCP 8000 from your current IP only),
# and one EC2 instance. Safe to re-run: existing resources are reused. Stops at the first error, including an
# organization SCP denial. Tear everything down with scripts/aws/teardown.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
trap 'on_error $LINENO' ERR
cd "$REPO_ROOT"

USE_ECR_IMAGE=""
IMAGE_TAG_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --use-ecr-image)
      USE_ECR_IMAGE=1
      if [ -n "${2:-}" ] && [[ "$2" != -* ]]; then IMAGE_TAG_ARG="$2"; shift; fi ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done

STEP="identity"
principal="$(aws sts get-caller-identity --query Arn --output text | tr -d '\r')"
case "$principal" in
  *:root) die "refusing to deploy as the root user; use an IAM user or role" ;;
  *:user/*) kind="IAM user" ;;
  *) kind="role/other" ;;
esac
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text | tr -d '\r')"
REGISTRY="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
log "authenticated as ${kind} '${principal##*/}'; region $REGION"

STEP="image tag"
if [ -n "$USE_ECR_IMAGE" ]; then
  SHA="${IMAGE_TAG_ARG:-$(git rev-parse --short=7 HEAD)}"
else
  SHA="$(git rev-parse --short=7 HEAD)"
  STEP="check image sources are committed"
  if [ -n "$(git status --porcelain -- src artifacts Dockerfile requirements-serve.txt .dockerignore)" ]; then
    die "uncommitted changes in files that go into the image; commit first so the image tag $SHA matches the code"
  fi
fi
IMAGE="$REGISTRY/$APP:$SHA"
log "image tag: $SHA"

if [ -n "$USE_ECR_IMAGE" ]; then
  # ---------------------------------------------------------------- image already in ECR (pushed by CI or an earlier run)
  STEP="verify image in ECR"
  [ -n "$(aws_q ecr describe-repositories --repository-names "$APP" --query 'repositories[0].repositoryName' --output text)" ] \
    || die "ECR repository $APP does not exist in $REGION; run without --use-ecr-image once, or let CI create the first push"
  if [ -z "$(aws_q ecr describe-images --repository-name "$APP" --image-ids "imageTag=$SHA" --query 'imageDetails[0].imageTags[0]' --output text | grep -v '^None$' || true)" ]; then
    tags="$(aws_q ecr list-images --repository-name "$APP" --filter tagStatus=TAGGED --query 'imageIds[].imageTag' --output text | tr '\t' ' ')"
    die "image tag '$SHA' is not in ECR repository $APP. Tags present: ${tags:-none}. Pass one explicitly: --use-ecr-image <TAG>"
  fi
  log "using image tag $SHA already in ECR; skipping local build and push"
  log "compressed image size in ECR: $(aws_q ecr describe-images --repository-name "$APP" --image-ids "imageTag=$SHA" --query 'imageDetails[0].imageSizeInBytes' --output text | awk '{printf "%.0f MB", $1/1048576}')"
  log "scan findings by severity: $(aws_q ecr describe-image-scan-findings --repository-name "$APP" --image-id "imageTag=$SHA" --query 'imageScanFindings.findingSeverityCounts' --output json | tr -d ' \n')"
else
# ---------------------------------------------------------------- ECR
STEP="ecr repository"
if [ -z "$(aws_q ecr describe-repositories --repository-names "$APP" --query 'repositories[0].repositoryName' --output text)" ]; then
  aws_ ecr create-repository --repository-name "$APP" --image-tag-mutability IMMUTABLE \
    --image-scanning-configuration scanOnPush=true --tags "Key=project,Value=$PROJECT_TAG" >/dev/null
  log "created ECR repository $APP (immutable tags, scan on push)"
else
  log "ECR repository $APP already exists"
fi
STEP="ecr lifecycle policy"
aws_ ecr put-lifecycle-policy --repository-name "$APP" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep the last 3 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null
log "lifecycle policy: keep the last 3 images"

STEP="network preflight"
# A slow or lossy path to the region makes a ~300 MB push hopeless (each attempt then fails after 5-15 minutes).
# Time a 1 MB layer-part upload to ECR first and stop early with a clear message. SKIP_PREFLIGHT=1 disables this.
if [ -z "${SKIP_PREFLIGHT:-}" ]; then
  PF_FILE="$(mktemp)"; head -c 1048576 /dev/urandom > "$PF_FILE"
  PF_PATH="$(cygpath -m "$PF_FILE" 2>/dev/null || echo "$PF_FILE")"   # the Windows aws.exe needs a Windows path
  UPLOAD_ID="$(aws_ ecr initiate-layer-upload --repository-name "$APP" --query uploadId --output text)"
  pf_start="$(date +%s.%N)"
  if aws --region "$REGION" ecr upload-layer-part --repository-name "$APP" --upload-id "$UPLOAD_ID" --part-first-byte 0 \
       --part-last-byte 1048575 --layer-part-blob "fileb://$PF_PATH" --cli-read-timeout 45 --cli-connect-timeout 20 >/dev/null 2>&1; then
    pf_secs="$(awk -v a="$pf_start" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')"
  else
    pf_secs="failed"
  fi
  rm -f "$PF_FILE"
  if [ "$pf_secs" = "failed" ] || awk -v s="$pf_secs" 'BEGIN{exit !(s > 10)}'; then
    die "network preflight failed: a 1 MB upload to ECR in $REGION took '${pf_secs}' s (need <= 10 s, i.e. >= 100 KB/s). The ~300 MB image push cannot succeed on this connection; switch networks and re-run, or set SKIP_PREFLIGHT=1 to try anyway."
  fi
  log "network preflight ok: 1 MB uploaded to ECR in ${pf_secs}s"
fi

STEP="docker build and push"
docker build -q -t "$APP:$SHA" . >/dev/null
log "built $APP:$SHA"
if [ -z "$(aws_q ecr describe-images --repository-name "$APP" --image-ids "imageTag=$SHA" --query 'imageDetails[0].imageTags[0]' --output text | grep -v '^None$' || true)" ]; then
  docker tag "$APP:$SHA" "$IMAGE"
  pushed=""
  for attempt in 1 2 3 4 5; do   # uploads of the large layer can time out on a slow link; layers already sent are skipped on retry
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null 2>&1
    log "pushing image (attempt $attempt of 5)"
    if docker push "$IMAGE" > >(mask | tail -2) 2>&1; then pushed=yes; break; fi
    sleep 5
  done
  [ -n "$pushed" ] || die "docker push failed 5 times (network timeouts?); re-run the script to resume"
  log "pushed image tag $SHA"
else
  log "image tag $SHA is already in ECR (tags are immutable); skipping push"
fi

STEP="image scan"
aws_ ecr wait image-scan-complete --repository-name "$APP" --image-id "imageTag=$SHA"
log "compressed image size in ECR: $(aws_ ecr describe-images --repository-name "$APP" --image-ids "imageTag=$SHA" --query 'imageDetails[0].imageSizeInBytes' --output text | awk '{printf "%.0f MB", $1/1048576}')"
log "scan findings by severity: $(aws_ ecr describe-image-scan-findings --repository-name "$APP" --image-id "imageTag=$SHA" --query 'imageScanFindings.findingSeverityCounts' --output json | tr -d ' \n')"

fi   # end: build/push here vs. use the image already in ECR

# ---------------------------------------------------------------- IAM
STEP="iam role"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --description "bank-churn-api EC2: pull from ECR, Session Manager" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --tags "Key=project,Value=$PROJECT_TAG" >/dev/null
  log "created IAM role $ROLE_NAME"
fi
for policy in AmazonEC2ContainerRegistryReadOnly AmazonSSMManagedInstanceCore; do
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "arn:aws:iam::aws:policy/$policy"
done
STEP="iam instance profile"
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" --tags "Key=project,Value=$PROJECT_TAG" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
  log "created instance profile $PROFILE_NAME; waiting for IAM propagation"
  sleep 15
fi

# ---------------------------------------------------------------- network
STEP="security group"
VPC_ID="$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(find_sg)"
if [ -z "$SG_ID" ]; then
  SG_ID="$(aws_ ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
    --description "bank-churn-api: TCP $PORT from one IP only" \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$SG_NAME},{Key=project,Value=$PROJECT_TAG}]" \
    --query GroupId --output text)"
  log "created security group $SG_NAME"
fi
MY_IP="$(current_ip)"
if ! aws_q ec2 describe-security-groups --group-ids "$SG_ID" --query 'SecurityGroups[0].IpPermissions[].IpRanges[].CidrIp' --output text | tr '\t' '\n' | grep -qx "$MY_IP/32"; then
  aws_ ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port "$PORT" --cidr "$MY_IP/32" >/dev/null
fi
log "security group allows TCP $PORT from your current IP only (no port 22)"

# ---------------------------------------------------------------- EC2
STEP="ami lookup"
AMI_ID="$(aws_ ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)"
SUBNET_ID="$(aws_ ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)"

STEP="ec2 instance"
INSTANCE_ID="$(find_instances | awk '{print $1}')"
if [ -n "$INSTANCE_ID" ]; then
  log "an instance for this project already exists; reusing it"
else
  USER_DATA="$(sed -e "s|@REGION@|$REGION|g" -e "s|@REGISTRY@|$REGISTRY|g" -e "s|@REPO@|$APP|g" -e "s|@TAG@|$SHA|g" \
                   -e "s|@MEMORY@|$MEMORY_LIMIT|g" "$REPO_ROOT/scripts/aws/user_data.sh.tpl")"
  INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" --subnet-id "$SUBNET_ID" \
    --security-group-ids "$SG_ID" --iam-instance-profile "Name=$PROFILE_NAME" --associate-public-ip-address \
    --metadata-options HttpTokens=required,HttpPutResponseHopLimit=1,HttpEndpoint=enabled \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=8,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}' \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$APP},{Key=project,Value=$PROJECT_TAG}]" \
                         "ResourceType=volume,Tags=[{Key=project,Value=$PROJECT_TAG}]" \
    --query 'Instances[0].InstanceId' --output text)"
  log "launched $INSTANCE_TYPE instance (IMDSv2 required, encrypted 8 GiB gp3 disk)"
fi
STEP="wait for instance"
aws_ ec2 wait instance-running --instance-ids "$INSTANCE_ID"
PUBLIC_IP="$(aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
log "instance running"

STEP="wait for the API"
log "waiting for user-data (Docker install + image pull) and the API on port $PORT; this takes a few minutes"
for attempt in $(seq 1 60); do
  if curl -fsS --max-time 5 "http://$PUBLIC_IP:$PORT/health" >/dev/null 2>&1; then
    log "API is up after about $((attempt * 15)) s"
    echo
    echo "  health : http://$PUBLIC_IP:$PORT/health"
    echo "  docs   : http://$PUBLIC_IP:$PORT/docs"
    echo "  verify : python scripts/smoke_test.py --base-url http://$PUBLIC_IP:$PORT"
    echo "  status : scripts/aws/status.sh"
    exit 0
  fi
  sleep 15
done
die "the API did not answer within 15 minutes; run scripts/aws/status.sh to read the instance's console output"
