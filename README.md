# Qwen3.8 Flash-Next on NVIDIA DGX Spark with TensorFold 0.6.2: Concurrency, Prefill & Speculative Tuning

High-throughput, exact local inference for **Qwen3.8-Flash-Next** on a single **NVIDIA DGX Spark** (GB10 ARM64, 121 GB unified memory).

This repository ports, extends, and benchmarks the high-performance patches from **@MiaAI_Lab's** DGX Spark recipe onto **@ashxhart's** **TensorFold 0.6.2**, introducing zero-copy Python-CUDA pointer passing, greedy decoding fast paths, prefill chunk geometry tuning, message boundary prefix cache multi-tenancy, and Blackwell-optimized tile schedules.

---

## ⚡ Executive Summary: Verified Peak Records

| Workload Metric | v0.3.6.3 Baseline | TensorFold 0.6.0 Stock | **Our 0.6.2 Vision Stack** | Delta vs Baseline |
|---|---|---|---|---|
| **Prefill Throughput (16k context)** | ~2,340 tok/s | 2,583 tok/s | **2,827.2 tok/s** (4.54s) | **+20.8%** |
| **Prefill Throughput (32k context)** | ~2,440 tok/s | 2,574 tok/s | **2,815.6 tok/s** (9.03s) | **+15.4%** |
| **Prefill Throughput (64k context)** | ~2,410 tok/s | 2,556 tok/s | **2,745.1 tok/s** (18.45s) | **+13.9%** |
| **Edit $\times 5$ Decode (Harness)** | ~180.0 agg tok/s | 226.8 agg tok/s | **328.2 agg tok/s** (94.0% accept) | **+82.3%** |
| **Code $\times 5$ Decode (Harness)** | 136.2 agg tok/s | 192.4 agg tok/s | **324.7 agg tok/s** (66.5 tok/s / str) | **+138.4%** |
| **Prose $\times 5$ Decode (Harness)** | 121.9 agg tok/s | 134.3 agg tok/s | **190.3 agg tok/s** (38.5 tok/s / str) | **+56.1%** |
| **Structured $\times 1$ Solo Decode** | 87.3 tok/s | 123.6 tok/s | **121.8 tok/s** (67 ms TTFT) | **+39.5%** |
| **Vision & Video Support** | Unsupported | Unsupported | **Verified (50 images, PyAV video)** | First-class multimodal |
| **Solo Prose TTFT** | 142 ms | 60 ms | **60 ms** | **-57.7%** |
| **Multi-turn Agent Resume TTFT** | ~10,200 ms (evicted) | ~9,800 ms (evicted) | **~80 ms** (hot cache hit) | **~120x faster** |
| **Token Bit-Parity** | 100% | 100% | **100% Bit-Identical** (12/12) | Zero divergence |
---

## 👥 Credits & Acknowledgments

This project builds directly on the foundational work of:

* **[Ash Hart (@ashxhart)](https://github.com/ashhart/TensorFold):** Creator of **TensorFold**, an open-source, local-first inference engine designed for exact speculative decoding. Version 0.6.2 added SM 12.0 DeltaNet tree kernel speedups, grouped projection launches, coalesced waiting prefill passes, Blackwell layer projection tile fusion, shared prefix retention across message boundaries, and native vLLM-mirrored Prometheus metrics.
* **[Mia's AI Lab (@MiaAI_Lab)](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold):** Created the original DGX Spark deployment recipe and authored the 64-thread persistent C++ multithreaded SSD n-gram reader (`ssd_read.cpp`), asynchronous read-ahead, and initial memory layout recipes.
* **[MovieMaker93 (@MovieMaker93)](https://github.com/MovieMaker93):** Authored upstream [TensorFold PR #40](https://github.com/ashhart/TensorFold/pull/40) for dynamic prefill chunk geometry (`indexed_prefill_rows()`), enabling wide chunk sizing up to 16,384 rows.
* **[Vontra](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP):** For the MLX affine 4-bit group-32 checkpoint with the native MTP speculative draft head.

---

## 🛠️ Architecture & What Was Changed in 0.6.2

Upstream TensorFold 0.6.2 added major enhancements (SM 12.0 tree kernel, grouped drafter projections, socket disconnect polling). We combined upstream's native Flash-Next vision tower with Mia's 50-image and video support and our zero-copy multi-stream speed fixes into `v062-concurrent-vision.patch`:
### 1. Coalesced Waiting Prompt Prefill & Fused Normalization (Upstream 0.6.1)
* **What it does:** In `--parallel 5`, waiting prompts that arrive concurrently now share one prefill forward: projections and MLPs execute over every waiting prompt's rows together, while convolutions and DeltaNet update per-stream state. Hyper-connection write-backs and RMSNorm run as a fused kernel.
* **Impact:** Prefill sustained past **2,820 tok/s**, cutting 1.6s off 50k prompt TTFT.

### 2. Blackwell (GB10) Projection Tile Fusion (Upstream 0.6.1)
* **What it does:** On Blackwell GPUs (SM 12.0 / 12.1), layer 4-bit projections reading identical activations launch together with tile schedules sized specifically for GB10.
* **Impact:** Faster decode rounds and ~4% lower verify step latency.

### 3. Greedy Fast-Path Decoding (`multi.py`)
* **Problem:** Stock TensorFold executed full 152k-element `torch.topk` and `logsumexp` normalization on every forward pass, even when all 5 concurrent streams were decoding greedily (`temperature=0`). This incurred ~7.5 ms of CPU blocking overhead per round.
* **Fix:** Added a greedy fast path bypassing top-k and logsumexp when `sampling.temperature <= 0`, directly selecting `argmax` on device.
* **Impact:** 1.37x faster pick execution in inner loops.

### 4. Zero-Copy Host-to-Device Pointer Passing (`gdn.py`, `attn_multi.py`, `gdn_multi.py`)
* **Problem:** In concurrent multi-stream mode, pointer tables were repeatedly converted to Python lists (`.tolist()`) and allocated as new tensors inside `to_device` every round, triggering repeated memory allocations and CPU-GPU synchronization flushes.
* **Fix:** Updated `tensorfold/cuda/kernels/gdn.py` so `to_device` accepts `np.ndarray` directly via pinned `torch.from_numpy(arr).to(device, non_blocking=True)`, passing pointers 89.8x faster. Removed intermediate `.tolist()` conversions throughout multi-stream attention loops.

### 5. Dynamic Prefill Chunk Geometry (`TENSORFOLD_PREFILL_ROWS=8192`)
* Ported dynamic chunk sizing and scaled prompt chunk geometry from stock 2,048/4,096 to **8,192 rows**. This maximizes Tensor Core utilization during long-context GEMMs while respecting unified memory page cache constraints.

### 6. Agent Multi-Turn Prefix Cache Retention (`--checkpoint-slots 6`)
* Maintained 6 checkpoint slots so all 5 concurrent streams keep hot prompt checkpoints in RAM. Paired with 0.6.1's message-boundary caching, agent branches resume in **~80 ms**.

### 7. Persistent 64-Thread Multithreaded SSD Reader (`ssd_read.cpp`)
* Flash-Next utilizes ~29.8 GB of hashed n-gram tables (`--ple-on-ssd`). Mia's persistent C++ extension uses a background thread pool of direct `pread` system calls with asynchronous read-ahead, keeping NVMe access completely off the Python GIL.

---

## 📊 Comprehensive Benchmark Results

All benchmarks were collected on the live **NVIDIA DGX Spark** (GB10 ARM64, 121 GB unified memory) using both **sparkDash** (OpenAI API over HTTP) and the local test harness (`tfab-concurrent.py`).

### 1. Prefill Throughput vs Context Length

Prefill throughput sustained flat past 32k tokens:

| Prompt Tokens | v0.3.6.3 Baseline | Stock 0.6.0 | **Our 0.6.1 Stack (sparkDash)** | **Our 0.6.1 (Harness)** |
|---|---|---|---|---|
| **4k (4,134 tok)** | ~2,340 tok/s | 2,473 tok/s | **2,712.8 tok/s** (1.52s) | **2,631.7 tok/s** (1.27s) |
| **8k (8,231 tok)** | ~2,390 tok/s | 2,576 tok/s | **2,750.4 tok/s** (2.99s) | — |
| **16k (16,421 tok)** | ~2,420 tok/s | 2,583 tok/s | **2,821.0 tok/s** (5.82s) | **2,806.0 tok/s** (4.57s) |
| **32k (32,807 tok)** | ~2,440 tok/s | 2,574 tok/s | **2,805.9 tok/s** (11.69s) | **2,802.0 tok/s** (9.07s) |
| **64k (50k prompt)** | ~2,410 tok/s | 2,556 tok/s | — | **2,735.9 tok/s** (18.52s) |

---

### 2. Multi-Stream Decode Concurrency (sparkDash HTTP API)

#### A. Code Generation (Developer & Agent Workloads)
*Prompt: Complex multi-file refactoring / algorithmic generation.*

| Concurrency | Baseline | Stock 0.6.0 | **Our 0.6.1 Stack** | Stream Tok/s | TTFT |
|---|---|---|---|---|---|
| **×1 Solo** | 68.0 tok/s | 96.9 tok/s | **96.9 tok/s** | 96.9 | 216 ms |
| **×2 Streams** | 98.4 tok/s | 143.2 tok/s | **143.1 tok/s** | 73.3 | 190 ms |
| **×3 Streams** | 118.5 tok/s | 168.2 tok/s | **169.6 tok/s** | 57.7 | 229 ms |
| **×4 Streams** | 128.0 tok/s | 172.0 tok/s | **173.4 tok/s** | 46.9 | 218 ms |
| **×5 Streams** | 136.2 tok/s | 192.4 tok/s | **193.3 tok/s** | 41.4 | 212 ms |

#### B. Structured Output Decode
*Prompt: Strict JSON extraction and schema validation.*

| Concurrency | Stock 0.6.0 | **Our 0.6.1 Stack** | TTFT | Stream Tok/s |
|---|---|---|---|---|
| **×1 Solo** | 123.6 tok/s | **121.8 tok/s** | 67 ms | 121.8 |
| **×2 Streams** | 184.6 tok/s | **181.9 tok/s** | 182 ms | 91.5 |
| **×3 Streams** | 187.8 tok/s | **191.9 tok/s** | 205 ms | 70.3 |
| **×4 Streams** | 150.6 tok/s | **150.9 tok/s** | 229 ms | 51.3 |
| **×5 Streams** | 205.6 tok/s | **206.6 tok/s** | 422 ms | 56.9 |

#### C. Prose Generation
*Prompt: Creative long-form prose and reasoning narrative.*

| Concurrency | Stock 0.6.0 | **Our 0.6.1 Stack** | TTFT | Stream Tok/s |
|---|---|---|---|---|
| **×1 Solo** | 68.7 tok/s | **68.0 tok/s** | 60 ms | 68.0 |
| **×2 Streams** | 94.1 tok/s | **96.0 tok/s** | 187 ms | 48.6 |
| **×3 Streams** | 113.2 tok/s | **110.4 tok/s** | 275 ms | 38.3 |
| **×4 Streams** | 123.2 tok/s | **120.5 tok/s** | 289 ms | 31.3 |
| **×5 Streams** | 134.3 tok/s | **136.6 tok/s** | 281 ms | 29.7 |

---

### 3. Harness Scaling & MTP Draft Acceptance Rates (`tfab-concurrent.py`)

In direct harness testing (bypassing HTTP/SSE serialization):
* **Code $\times 5$ Decode:** **312.6 tok/s aggregate** (79.0% MTP acceptance)
* **Edit $\times 5$ Decode:** **324.6 tok/s aggregate** (94.0% MTP acceptance)
* **Prose $\times 5$ Decode:** **190.1 tok/s aggregate** (54.0% MTP acceptance)
* **Chat with Thinking:** **55.9 tok/s** (63.0% MTP acceptance)

---

## 📉 Negative Results & Architectural Post-Mortems

We thoroughly evaluated alternative configurations that failed or performed worse:

1. **Prefill Chunk Geometry: `TENSORFOLD_PREFILL_ROWS=16384`:**
   * Regressed long prefill (16k dropped from 2,806 down to 2,611 tok/s). Sizing activation buffers for 16k rows consumed an extra ~3.8 GiB of RAM, starving the Linux page cache holding the 29.8 GB n-gram table files. **8,192 rows is the optimal ceiling.**
2. **Speculative Draft Depth: `MTP_DRAFTS=5` vs `6`:**
   * Capping drafts at 5 reduced Code decode by 4–5% because code syntax frequently accepts all 6 draft tokens. **6 drafts remains optimal.**
3. **KV Cache Precision: `int4` KV Cache:**
   * Dropped Code decode by -7.5% because KV quantization noise reduced speculative draft acceptance from 79% to 73%. **`int8` KV cache is strictly superior.**
4. **ModelOpt NVFP4 Checkpoint:**
   * 30% to 59% slower than MLX 4-bit because MLX has mature hand-tuned assembly tile schedules for GB10, whereas NVFP4 relied on software dequantization MMAs.

---

## 🏆 The Production Stack

Deploy this configuration on a single DGX Spark:

```bash
# Environment Configuration
export TENSORFOLD_PREFILL_ROWS=8192      # 8k prefill chunk size (2,821 tok/s)
export TENSORFOLD_SSD_NATIVE=1          # 64-thread C++ persistent SSD reader
export CHECKPOINT_SLOTS=6               # Keeps all 5 concurrent streams warm in RAM
export DECODE_SHARE=0.25                # 25% decode compute guarantee under prompt bursts
export MTP_CONFIDENCE=0.65              # Peak confidence for structured/code tasks
export MTP_DRAFTS=6                     # Preserves code syntax draft acceptance
export KV_DTYPE=int8                    # int8 avoids int4 speculative divergence
export PLE_ON_SSD=1                     # Retains MLX 4-bit weights with SSD n-gram tables
export VISION=1                         # Enables native 0.84 GiB Flash-Next vision tower
export VISION_MAX_IMAGES=50             # Up to 50 images per request (16,384 shared tokens)
export MAX_TOKENS=32768                 # 32k reply limit prevents cutting off reasoning chains

---

## 🚀 Quickstart & Installation

### Prerequisites
* Single **NVIDIA DGX Spark** (GB10 ARM64, 128 GB unified memory).
* Docker with NVIDIA Container Toolkit installed (`--gpus all`).
* Checkpoint cached under `~/.cache/huggingface`.

### Step 1: Clone Repository
```bash
git clone https://github.com/aronlabs/qwen3.8-flash-next-tensorfold-spark.git
cd qwen3.8-flash-next-tensorfold-spark
```

### Step 2: Build Image
Builds TensorFold 0.6.2 with `v062-concurrent-vision.patch` applied:
```bash
docker build -t tensorfold-qwen38:v0.6.2-concurrent-vision .
```

### Step 3: Launch Production Server
```bash
./start.sh
```

### Step 4: Verify Health
```bash
curl -s http://127.0.0.1:8888/health | jq .
```
Expected output:
```json
{
  "ok": true,
  "backend": "tensorfold",
  "busy": false,
  "streams": { "decoding": 0, "prefilling": 0, "max": 5 },
  "context_length": 262144
}
```

---

## 🔬 Benchmark & Parity Reproduction

```bash
# Run full verification suite (Parity + Concurrency Sweep)
python3 tfab-concurrent.py all winning-stack

# Run parity fixtures only (checks greedy, sampled, 20k/45k long contexts)
python3 tfab-concurrent.py parity winning-stack

# Run concurrency benchmarks only (tests 1 to 5 streams)
python3 tfab-concurrent.py bench winning-stack
```

Benchmark output logs and parity proofs are stored under `results/`.

---

## 📄 License

* **TensorFold** is licensed under Apache-2.0 by Ash Hart.
* Ported modifications, speed patches, and scripts in this repository are licensed under Apache-2.0.
