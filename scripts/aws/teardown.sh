#!/usr/bin/env bash
# Delete everything deploy.sh created: the EC2 instance (and its disk), security group, IAM role and instance
# profile, and the ECR repository with all its images. Asks for confirmation unless called with --yes.
#
#   scripts/aws/teardown.sh [--yes]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ "${1:-}" != "--yes" ]; then
  echo "This will DELETE the instance, security group, IAM role/profile and ECR repository '$APP' in $REGION."
  read -r -p "Type 'delete' to continue: " answer
  [ "$answer" = "delete" ] || { echo "aborted"; exit 1; }
fi

ids="$(find_instances | tr '\t' ' ')"
if [ -n "$ids" ]; then
  log "terminating instance(s)"
  aws_ ec2 terminate-instances --instance-ids $ids >/dev/null
  aws_ ec2 wait instance-terminated --instance-ids $ids
fi

SG_ID="$(find_sg)"
if [ -n "$SG_ID" ]; then
  for attempt in 1 2 3 4 5 6; do
    if aws_q ec2 delete-security-group --group-id "$SG_ID" >/dev/null; then log "deleted security group"; break; fi
    sleep 10   # the network interface can take a moment to release
  done
fi

if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" 2>/dev/null || true
  aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME"
  log "deleted instance profile"
fi
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  for policy in $(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[].PolicyArn' --output text | tr -d '\r'); do
    aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$policy"
  done
  aws iam delete-role --role-name "$ROLE_NAME"
  log "deleted IAM role"
fi

if [ -n "$(aws_q ecr describe-repositories --repository-names "$APP" --query 'repositories[0].repositoryName' --output text)" ]; then
  aws_ ecr delete-repository --repository-name "$APP" --force >/dev/null
  log "deleted ECR repository"
fi

echo "--- leftover check (all should be empty) ---"
echo "instances        : '$(find_instances)'"
echo "security group   : '$(find_sg)'"
echo "ecr repository   : '$(aws_q ecr describe-repositories --repository-names "$APP" --query 'repositories[0].repositoryName' --output text)'"
echo "iam role/profile : '$(aws iam get-role --role-name "$ROLE_NAME" --query Role.RoleName --output text 2>/dev/null || true)' '$(aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" --query InstanceProfile.InstanceProfileName --output text 2>/dev/null || true)'"
