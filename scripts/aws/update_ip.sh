#!/usr/bin/env bash
# Point the security group at your CURRENT public IP (use after your IP changes). Replaces any existing port-8000 rule.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SG_ID="$(find_sg)"
[ -n "$SG_ID" ] || die "security group $SG_NAME not found in $REGION"
MY_IP="$(current_ip)"
for cidr in $(aws_ ec2 describe-security-groups --group-ids "$SG_ID" --query "SecurityGroups[0].IpPermissions[?FromPort==\`$PORT\`].IpRanges[].CidrIp" --output text); do
  aws_ ec2 revoke-security-group-ingress --group-id "$SG_ID" --protocol tcp --port "$PORT" --cidr "$cidr" >/dev/null
done
aws_ ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port "$PORT" --cidr "$MY_IP/32" >/dev/null
log "port $PORT is now open to your current IP only"
