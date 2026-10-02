#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on NVIDIA DGX Spark with TensorFold 0.6.1 + Concurrency & Speed Patches
set -euo pipefail

IMAGE="${IMAGE:-tensorfold-qwen38:v0.6.1-concurrent}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"
MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-kernels}"
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
PARALLEL="${PARALLEL:-5}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-int8}"
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.65}"
CHECKPOINT_SLOTS="${CHECKPOINT_SLOTS:-6}"
DECODE_SHARE="${DECODE_SHARE:-0.25}"
PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

echo "Starting $CONTAINER_NAME ($IMAGE)..."
mkdir -p "$KERNEL_CACHE/torch_extensions_v061" "$KERNEL_CACHE/triton_v061"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Stopping previous container..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

docker run -d --name "$CONTAINER_NAME" \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e TENSORFOLD_PREFILL_ROWS="$PREFILL_ROWS" \
  -e TENSORFOLD_SSD_NATIVE=1 \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions_v061 \
  -e TRITON_CACHE_DIR=/cache/triton_v061 \
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
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --ple-on-ssd \
    --thinking \
    --host "$HOST" \
    --port "$PORT"

echo "Server container started. Logs: docker logs -f $CONTAINER_NAME"
echo "Endpoint: http://$HOST:$PORT/v1"
