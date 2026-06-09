#!/bin/bash
# Start all services with per-service log files using uv run.
# Each service logs to logs/<name>.log and its PID to logs/<name>.pid.
# Usage: ./start_logged.sh [services...]   (default: all)
set -u
cd "$(dirname "$0")"
mkdir -p logs

start() {
  local name="$1"; local mod="$2"
  echo "Starting $name ($mod)..."
  uv run python -m "$mod" >"logs/$name.log" 2>&1 &
  echo $! >"logs/$name.pid"
}

WHICH="${*:-registry tax compliance law customer}"

for s in $WHICH; do
  case "$s" in
    registry)   start registry registry; sleep 2 ;;
    tax)        start tax tax_agent ;;
    compliance) start compliance compliance_agent ;;
    law)        start law law_agent ;;
    customer)   start customer customer_agent ;;
  esac
done

echo "Done launching: $WHICH"
