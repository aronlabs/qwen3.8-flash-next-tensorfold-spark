#!/usr/bin/env bash
# Staged NVFP4 Test Runner for DGX Spark
# Runs Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 in sandbox without --ple-on-ssd
set -euo pipefail

MODEL_ID="Mia-AiLab/Qwen3.8-Flash-Next-NVFP4"
IMAGE="tensorfold-qwen38:v0.6.0-concurrent-stage"
STAGE_CONTAINER="qwen38-flash-next-tf-stage"
LIVE_CONTAINER="qwen38-flash-next-tf"
PORT=8888
HOST="0.0.0.0"
PARALLEL=5
CONTEXT=262144
KV_DTYPE="int8"
MTP_DRAFTS=6
MTP_CONFIDENCE=0.65
CHECKPOINT_SLOTS=6
HF_CACHE="$HOME/.cache/huggingface"
KERNEL_CACHE="$HOME/.cache/tensorfold-kernels"

echo "================================================================="
echo "TensorFold ModelOpt NVFP4 Staged Evaluation"
echo "Model:            $MODEL_ID"
echo "Image:            $IMAGE"
echo "RAM Mode:         100% RAM Resident (NO --ple-on-ssd)"
echo "Parallel Streams: $PARALLEL"
echo "Context:          $CONTEXT"
echo "================================================================="

# 1. Pause tf-watch
echo "[1/4] Pausing tf-watch..."
"$HOME/scripts/tf-watch.sh" --pause on

# 2. Stop live container cleanly
if docker ps --format '{{.Names}}' | grep -qx "$LIVE_CONTAINER"; then
  echo "[2/4] Stopping live container '$LIVE_CONTAINER'..."
  docker stop "$LIVE_CONTAINER"
fi

docker rm -f "$STAGE_CONTAINER" >/dev/null 2>&1 || true

# 3. Launch NVFP4 container
echo "[3/4] Starting NVFP4 staged container..."
docker run -d --name "$STAGE_CONTAINER" \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e TENSORFOLD_PREFILL_ROWS=8192 \
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
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --thinking \
    --host "$HOST" \
    --port "$PORT"

# 4. Wait for health
echo "[4/4] Waiting for NVFP4 endpoint to become ready..."
until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok":[ ]*true'; do
  sleep 4
  printf "."
done
echo " READY!"
echo ""
echo "NVFP4 is now serving on http://10.10.2.196:8888/v1!"
echo "You can now run your sparkDash benchmarks against it."
echo "When done, restore the live setup by running: ./rollback.sh"
