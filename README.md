# Qwen3.8 Flash-Next on NVIDIA DGX Spark with TensorFold 0.6.0

High-throughput, exact local inference for **Qwen3.8-Flash-Next** on a single **NVIDIA DGX Spark** (GB10 ARM64, 121 GB unified memory).

This repository ports the high-performance patches from **@MiaAI_Lab's** Spark recipe onto **@ashxhart's** newly released **TensorFold 0.6.0**, unlocking:
- **Up to 205.6 aggregate tok/s** decode (Structured) and **192.4 aggregate tok/s** (Code) across 5 concurrent streams.
- **Single-stream code decode at 96.9 tok/s** (+42.5% over the v0.3.6.3 baseline).
- **Sub-60 ms TTFT** for solo prose decoding.
- **A flat ~2,575 tok/s sustained prefill** across 4k, 8k, 16k, and 32k contexts.
- **~18 GB of host memory freed** (startup reservation dropped from 102.5 GiB to 84.7 GiB) through dynamic lane KV cache allocation.

---

## Credits & Acknowledgments

This project builds directly upon the work of two creators:

* **[Ash Hart (@ashxhart)](https://github.com/ashhart/TensorFold):** Creator of **TensorFold**, an open-source, local-first inference engine designed for exact speculative decoding. Version 0.6.0 introduces groundbreaking architectural features including in-flight chunked prefilling, dynamic stream memory allocation, CUDA lane prompt caching, and native Prometheus metrics.
* **[Mia's AI Lab (@MiaAI_Lab)](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold):** Created the original DGX Spark deployment recipe and wrote the critical C++ persistent multithreaded SSD n-gram reader (`ssd_read.cpp`), asynchronous read-ahead, and prefill chunk geometry optimizations that make Flash-Next fly on unified memory.
* **[Vontra](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP):** For the MLX affine 4-bit group-32 checkpoint with the MTP head.

---

## Architecture & What Was Changed

Upstream TensorFold 0.6.0 added major concurrent scheduling enhancements, but Mia's custom DGX Spark optimizations were originally written against v0.3.6.3. We extracted the core speed logic from Mia's patch stack and cleanly ported it to 0.6.0 (`v060-speed.patch`):

1. **Persistent Multithreaded SSD Reader (`ssd_read.cpp`):**
   Flash-Next utilizes ~29.8 GB of hashed n-gram tables (`--ple-on-ssd`). Rather than relying on Python GIL-bound reads or standard mmap page faults, Mia's native C++ extension uses a persistent thread pool of `pread` calls to stream n-gram rows with minimal latency.
2. **Asynchronous Prefill Read-Ahead:**
   Prefill chunks read upcoming n-gram rows asynchronously while the GPU executes layer compute.
3. **Configurable Prefill Chunk Rows:**
   Allows scaling the prefill chunk window (`TENSORFOLD_PREFILL_ROWS=4096`) for text-only workloads, yielding 10–15% faster prefill over the default 2,048-row chunks.
4. **Dynamic Stream Allocation (TensorFold 0.6.0 Native):**
   Replaces v0.3.6.3's static pre-allocation of the full 5 × 262,144 KV cache with dynamic per-stream growth (`first: 256` entry buffers).
5. **In-Flight Chunked Prefill (TensorFold 0.6.0 Native):**
   Incoming prompts interleave into the decode rounds rather than stalling active generating streams.

---

## Benchmark Results (sparkDash)

Hardware: Single NVIDIA DGX Spark (GB10 ARM64, 121 GB unified memory).  
Configuration: 5 streams × 262,144 tokens, int8 KV cache, SSD n-gram tables (`--ple-on-ssd`), MTP confidence 0.60, max 6 drafts.

### 1. Prefill Throughput

Prefill throughput remains completely flat past 32k tokens:

* **4k:** 2,473.4 tok/s · TTFT 1.67 s (4,133 tokens)
* **8k:** 2,576.2 tok/s · TTFT 3.19 s (8,227 tokens)
* **16k:** 2,583.8 tok/s · TTFT 6.36 s (16,424 tokens)
* **32k:** 2,574.1 tok/s · TTFT 12.75 s (32,807 tokens)

### 2. Code Decode Throughput (Agent & Developer Workloads)

Solo code generation jumps from 68 tok/s to **96.9 tok/s** (+42.5%), and 5 concurrent streams deliver **192.4 aggregate tok/s** with TTFT flat across all 5 streams:

* **×1:** 96.9 agg · 96.9/stream · TTFT 187 ms
* **×2:** 143.2 agg · 73.4/stream · TTFT 191 ms
* **×3:** 168.2 agg · 57.1/stream · TTFT 213 ms
* **×4:** 172.0 agg · 46.3/stream · TTFT 218 ms
* **×5:** 192.4 agg · 41.1/stream · TTFT 223 ms

### 3. Structured Decode Throughput

* **×1:** 123.6 agg · 123.6/stream · TTFT 79 ms
* **×2:** 184.6 agg · 92.8/stream · TTFT 177 ms
* **×3:** 187.8 agg · 69.7/stream · TTFT 333 ms
* **×4:** 150.6 agg · 51.1/stream · TTFT 233 ms
* **×5:** 205.6 agg · 56.7/stream · TTFT 416 ms

### 4. Prose Decode Throughput

Solo TTFT cut by 58% (down to 60 ms) while scaling to 134 agg tok/s:

* **×1:** 68.7 agg · 68.7/stream · TTFT 60 ms
* **×2:** 94.1 agg · 47.6/stream · TTFT 188 ms
* **×3:** 113.2 agg · 38.6/stream · TTFT 235 ms
* **×4:** 123.2 agg · 31.6/stream · TTFT 246 ms
* **×5:** 134.3 agg · 29.4/stream · TTFT 445 ms

---

## Installation & Setup

### Prerequisites
* NVIDIA DGX Spark (or a GH200 / ARM64 CUDA environment with Compute Capability ≥ 9.0).
* Docker with NVIDIA Container Toolkit installed (`--gpus all`).
* ~106 GB of disk space for the model weights under `~/.cache/huggingface`.

### Step 1: Clone Repository
```bash
git clone https://github.com/aronlabs/qwen3.8-flash-next-tensorfold-spark.git
cd qwen3.8-flash-next-tensorfold-spark
```

### Step 2: Build the Container Image
The build installs upstream TensorFold 0.6.0 on NVIDIA's PyTorch 26.07 ARM64 container and applies `v060-speed.patch`:

```bash
docker build -t tensorfold-qwen38:v0.6.0-speed .
```

### Step 3: Download Checkpoint (if not already cached)
Download `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` using `huggingface-cli` or TensorFold pull:
```bash
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
  tensorfold-qwen38:v0.6.0-speed \
  tensorfold pull Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
```

### Step 4: Start Serving
Launch the server with the included script:
```bash
./start.sh
```

Or manually via Docker:
```bash
docker run -d --name qwen38-flash-next-tf \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions_v060 \
  -e TRITON_CACHE_DIR=/cache/triton_v060 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/tensorfold-kernels:/cache \
  tensorfold-qwen38:v0.6.0-speed \
  tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
    --name "Qwen3.8-Flash-Next" \
    --parallel 5 \
    --context 262144 \
    --kv-dtype int8 \
    --mtp-drafts 6 \
    --mtp-confidence 0.60 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --ple-on-ssd \
    --thinking \
    --host 0.0.0.0 \
    --port 8888
```

The server initializes weights and compiles JIT extensions in ~2.5–3 minutes. Once ready, it advertises an OpenAI-compatible API on `http://0.0.0.0:8888/v1`.

### Step 5: Verification & Health Check
Check status:
```bash
curl -s http://127.0.0.1:8888/health
```
Expected response:
```json
{
  "ok": true,
  "backend": "tensorfold",
  "busy": false,
  "streams": { "decoding": 0, "prefilling": 0, "max": 5 },
  "context_length": 262144
}
```

Check Prometheus metrics:
```bash
curl -s http://127.0.0.1:8888/metrics
```

Run a test completion:
```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Say hello in one word."}]
  }'
```

---

## Benchmarking & Parity Validation

This repository includes `tfab.py` to run exactness and speed checks against reference outputs:

```bash
# Run full benchmark and parity suite
python3 tfab.py
```

The benchmark results and parity verification logs are archived under `results/`.

---

## License

* TensorFold is licensed under Apache-2.0 by Ash Hart.
* Ported modifications and scripts are provided under Apache-2.0.
