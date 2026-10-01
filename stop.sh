#!/usr/bin/env bash
# Stop the TensorFold container
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"

echo "Stopping $CONTAINER_NAME..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
echo "Stopped."
