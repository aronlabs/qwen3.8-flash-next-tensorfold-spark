#!/usr/bin/env bash
# Staged Sandbox Test Runner for TensorFold Concurrency Optimizations
# DOES NOT execute automatically; run when ready to evaluate against the live model.
set -euo pipefail

LABEL="${1:-stage-test}"
IMAGE="${IMAGE:-tensorfold-qwen38:v0.6.0-concurrent-stage}"
STAGE_CONTAINER="qwen38-flash-next-tf-stage"
LIVE_CONTAINER="qwen38-flash-next-tf"
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
PARALLEL="${PARALLEL:-5}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-int8}"
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"
CHECKPOINT_SLOTS="${CHECKPOINT_SLOTS:-6}"
DECODE_SHARE="${DECODE_SHARE:-0.25}"
MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-kernels}"

echo "================================================================="
echo "TensorFold Staged Concurrency Evaluation Harness"
echo "Label:            $LABEL"
echo "Image:            $IMAGE"
echo "KV Dtype:         $KV_DTYPE"
echo "Parallel Streams: $PARALLEL"
echo "Decode Share:     $DECODE_SHARE"
echo "Checkpoint Slots: $CHECKPOINT_SLOTS"
echo "================================================================="

# 1. Safety check: ensure tf-watch is paused
echo "[1/6] Pausing tf-watch health watcher..."
"$HOME/scripts/tf-watch.sh" --pause on

# 2. Stop live container cleanly (do not remove it)
if docker ps --format '{{.Names}}' | grep -qx "$LIVE_CONTAINER"; then
  echo "[2/6] Stopping live container '$LIVE_CONTAINER' (kept for instant rollback)..."
  docker stop "$LIVE_CONTAINER"
else
  echo "[2/6] Live container '$LIVE_CONTAINER' already stopped."
fi

# 3. Clean up any previous stage container
docker rm -f "$STAGE_CONTAINER" >/dev/null 2>&1 || true

# 4. Launch staged container
echo "[3/6] Starting staged container '$STAGE_CONTAINER'..."
mkdir -p "$KERNEL_CACHE/torch_extensions_v060" "$KERNEL_CACHE/triton_v060"

docker run -d --name "$STAGE_CONTAINER" \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions_v060 \
  -e TRITON_CACHE_DIR=/cache/triton_v060 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$KERNEL_CACHE":/cache \
  "$IMAGE" \
  tensorfold serve "$MODEL_ID" \
    --name "Qwen3.8-Flash-Next" \
    --parallel "$PARALLEL" \
    --context "$CONTEXT" \
    --kv-dtype "$KV_DTYPE" \
    --mtp-drafts "$MTP_DRAFTS" \
    --mtp-confidence "$MTP_CONFIDENCE" \
    --checkpoint-slots "$CHECKPOINT_SLOTS" \
    --decode-share "$DECODE_SHARE" \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --ple-on-ssd \
    --thinking \
    --host "$HOST" \
    --port "$PORT"

# 5. Wait for endpoint ready
echo "[4/6] Waiting for staged endpoint http://127.0.0.1:$PORT/health to become ready..."
ATTEMPTS=0
MAX_ATTEMPTS=60
until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok":[ ]*true'; do
  sleep 5
  ATTEMPTS=$((ATTEMPTS + 1))
  if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
    echo "ERROR: Timed out waiting for staged container to start."
    echo "Logs from stage container:"
    docker logs --tail 50 "$STAGE_CONTAINER"
    echo "Executing emergency rollback..."
    docker rm -f "$STAGE_CONTAINER" >/dev/null 2>&1 || true
    docker start "$LIVE_CONTAINER"
    "$HOME/scripts/tf-watch.sh" --pause off
    exit 1
  fi
  printf "."
done
echo " READY!"

# 6. Run Parity & Benchmark
echo "[5/6] Executing Parity and Concurrency Benchmarks..."
python3 "$(dirname "$0")/tfab-concurrent.py" all "$LABEL"

# 7. Tear down stage container and restore live container
echo "[6/6] Benchmark complete. Restoring live container '$LIVE_CONTAINER'..."
docker rm -f "$STAGE_CONTAINER" >/dev/null 2>&1 || true
docker start "$LIVE_CONTAINER"

# Wait for live container to be healthy
echo "Waiting for live container health..."
until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok":[ ]*true'; do
  sleep 3
  printf "."
done
echo " Live container healthy."

"$HOME/scripts/tf-watch.sh" --pause off
echo "tf-watch watcher resumed."
echo "Evaluation complete! Results stored under $(dirname "$0")/results/$LABEL-bench.json"
