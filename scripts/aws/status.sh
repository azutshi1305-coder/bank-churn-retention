#!/usr/bin/env bash
# Show the deployed instance's state, the public URLs, a health probe, and the tail of its console output
# (user-data progress lines start with "BANKCHURN:"). Needs no SSH or SSM command access.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

INSTANCE_ID="$(find_instances | awk '{print $1}')"
[ -n "$INSTANCE_ID" ] || { echo "no instance for project $PROJECT_TAG in $REGION"; exit 0; }
read -r STATE IP TYPE <<<"$(aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress,InstanceType]' --output text)"
echo "instance : $TYPE, state=$STATE"
if [ "$IP" != "None" ] && [ -n "$IP" ]; then
  echo "health   : http://$IP:$PORT/health -> $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$IP:$PORT/health" || true)"
  echo "docs     : http://$IP:$PORT/docs"
fi
echo "--- console output (BANKCHURN lines, then the last 15 lines) ---"
OUT="$(aws_ ec2 get-console-output --instance-id "$INSTANCE_ID" --latest --output text 2>/dev/null | mask || true)"
echo "$OUT" | grep "BANKCHURN" || echo "(no BANKCHURN lines yet; console output lags the instance by a few minutes)"
echo "$OUT" | tail -15
