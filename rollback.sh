#!/usr/bin/env bash
# Emergency / Manual Rollback Script for TensorFold Stage
set -euo pipefail

STAGE_CONTAINER="qwen38-flash-next-tf-stage"
LIVE_CONTAINER="qwen38-flash-next-tf"
PORT="${PORT:-8888}"

echo "Performing rollback to live container..."
docker rm -f "$STAGE_CONTAINER" >/dev/null 2>&1 || true

if ! docker ps --format '{{.Names}}' | grep -qx "$LIVE_CONTAINER"; then
  echo "Starting $LIVE_CONTAINER..."
  docker start "$LIVE_CONTAINER"
fi

until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok":[ ]*true'; do
  sleep 2
  printf "."
done

"$HOME/scripts/tf-watch.sh" --pause off
echo " Live container is up and tf-watch watcher resumed."
