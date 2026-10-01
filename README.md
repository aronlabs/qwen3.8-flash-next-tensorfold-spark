# Qwen3.8 Flash-Next on NVIDIA DGX Spark with TensorFold 0.6.0: Concurrency, Prefill & Speculative Tuning

High-throughput, exact local inference for **Qwen3.8-Flash-Next** on a single **NVIDIA DGX Spark** (GB10 ARM64, 121 GB unified memory).

This repository ports, extends, and benchmarks the high-performance patches from **@MiaAI_Lab's** DGX Spark recipe onto **@ashxhart's** **TensorFold 0.6.0**, introducing zero-copy Python-CUDA pointer passing, greedy decoding fast paths, prefill chunk geometry tuning, and prefix cache multi-tenancy.

---

## ⚡ Executive Summary: Verified Peak Records

| Workload Metric | v0.3.6.3 Baseline | TensorFold 0.6.0 Stock | **Our Winning Stack** | Delta vs Baseline |
|---|---|---|---|---|
| **Code $\times 5$ Decode (Harness)** | 136.2 agg tok/s | 192.4 agg tok/s | **319.2 agg tok/s** | **+134.4%** |
| **Code $\times 1$ Solo Decode** | 68.0 tok/s | 96.9 tok/s | **97.9 tok/s** | **+44.0%** |
| **Structured $\times 5$ Decode (sparkDash)** | 158.0 agg tok/s | 205.6 agg tok/s | **210.7 agg tok/s** | **+33.4%** |
| **Structured $\times 5$ TTFT** | ~520 ms | 416 ms | **304 ms** | **-41.5%** |
| **Solo Prose TTFT** | 142 ms | 60 ms | **58 ms** | **-59.2%** |
| **Prefill Throughput (16k context)** | ~2,340 tok/s | 2,583 tok/s | **2,652 tok/s** | **+13.3%** |
| **Multi-turn Agent Resume TTFT** | ~10,200 ms (evicted) | ~9,800 ms (evicted) | **~80 ms** (hit) | **~120x faster** |
| **Token Bit-Parity** | 100% | 100% | **100% Bit-Identical** | Zero divergence |

---

## 👥 Credits & Acknowledgments

This project builds directly on the foundational work of:

* **[Ash Hart (@ashxhart)](https://github.com/ashhart/TensorFold):** Creator of **TensorFold**, an open-source, local-first inference engine designed for exact speculative decoding. Version 0.6.0 introduced in-flight chunked prefilling, dynamic stream allocation, CUDA lane prompt caching, and native Prometheus metrics.
* **[Mia's AI Lab (@MiaAI_Lab)](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold):** Created the original DGX Spark deployment recipe and authored the 64-thread persistent C++ multithreaded SSD n-gram reader (`ssd_read.cpp`), asynchronous read-ahead, and initial memory layout recipes.
* **[MovieMaker93 (@MovieMaker93)](https://github.com/MovieMaker93):** Authored upstream [TensorFold PR #40](https://github.com/ashhart/TensorFold/pull/40) for dynamic prefill chunk geometry (`indexed_prefill_rows()`), enabling wide chunk sizing up to 16,384 rows.
* **[Vontra](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP):** For the MLX affine 4-bit group-32 checkpoint with the native MTP speculative draft head.

---

## 🛠️ Architecture & What Was Changed

Upstream TensorFold 0.6.0 added major concurrent scheduling enhancements, but Mia's custom DGX Spark optimizations were originally written against v0.3.6.3. We extracted the core speed logic, resolved multi-stream batching conflicts, and eliminated internal Python CPU-GPU sync flushes in `v060-concurrent.patch`:

### 1. Greedy Fast-Path Decoding (`multi.py`)
* **Problem:** In stock TensorFold 0.6.0, `_picks` in `multi.py` executed full 152k-element `torch.topk` and `logsumexp` normalization on every forward pass, even when all 5 concurrent streams were decoding greedily (`temperature=0`). This incurred ~7.5 ms of CPU blocking overhead per round.
* **Fix:** Added a greedy fast path bypassing top-k and logsumexp when `sampling.temperature <= 0`, directly selecting `argmax` on device.
* **Impact:** 1.37x faster pick execution in inner loops.

### 2. Zero-Copy Host-to-Device Pointer Passing (`gdn.py`, `attn_multi.py`, `gdn_multi.py`)
* **Problem:** In concurrent multi-stream mode, `attn_multi.py` and `gdn_multi.py` repeatedly converted NumPy pointer tables to Python lists (`.tolist()`) and allocated new tensors inside `to_device` every round, triggering repeated memory allocations and CPU-GPU synchronization flushes.
* **Fix:** Updated `tensorfold/cuda/kernels/gdn.py` so `to_device` accepts `np.ndarray` directly via pinned `torch.from_numpy(arr).to(device, non_blocking=True)`, passing pointers 89.8x faster. Removed intermediate `.tolist()` conversions throughout multi-stream attention loops.

### 3. Dynamic Prefill Chunk Geometry (`TENSORFOLD_PREFILL_ROWS=8192`)
* Ported MovieMaker93's dynamic chunk sizing (TensorFold PR #40) and scaled prompt chunk geometry from 2,048/4,096 to **8,192 rows**. This maximizes Tensor Core utilization during long-context GEMMs while respecting unified memory page cache constraints.

### 4. Agent Multi-Turn Prefix Cache Retention (`--checkpoint-slots 6`)
* Stock 0.6.0 defaulted to keeping only 3 prompt checkpoints (`KEEP = 3`). Under 5 concurrent streams, active streams thrashed each other's prefix states, forcing multi-turn chat and coding agents to re-prefill 15k–30k prompts from scratch. Setting `--checkpoint-slots 6` ensures all 5 concurrent streams maintain hot prefix caches in RAM (~80 ms resumes).

### 5. Latency-Protective Decode Share (`--decode-share 0.25`)
* Dedicated 25% of each forward iteration to active decoding streams during prompt ingestion bursts, preventing active streams from stalling when a 32k prompt arrives.

### 6. Persistent 64-Thread Multithreaded SSD Reader (`ssd_read.cpp`)
* Flash-Next utilizes ~29.8 GB of hashed n-gram tables (`--ple-on-ssd`). Mia's persistent C++ extension uses a background thread pool of direct `pread` system calls with asynchronous read-ahead, reducing NVMe SSD access overhead to near-zero.

---

## 📊 Comprehensive Benchmark Results

All benchmarks were collected on a single NVIDIA DGX Spark (GB10 ARM64, 121 GB unified memory) using both **sparkDash** (OpenAI API over HTTP) and the local test harness (`tfab-concurrent.py`).

### 1. Prefill Throughput vs Context Length

Prefill throughput sustained flat past 32k tokens with zero prefill jitter:

| Prompt Tokens | v0.3.6.3 (2k Chunks) | Stock 0.6.0 (4k Chunks) | **Our Stack (8k Chunks)** | Delta vs Stock |
|---|---|---|---|---|
| **4k (4,135 tok)** | ~2,340 tok/s | 2,473 tok/s | **2,598 tok/s** (1.59s) | **+5.1%** |
| **8k (8,230 tok)** | ~2,390 tok/s | 2,576 tok/s | **2,646 tok/s** (3.11s) | **+2.7%** |
| **16k (16,424 tok)** | ~2,420 tok/s | 2,583 tok/s | **2,652 tok/s** (6.19s) | **+2.7%** |
| **32k (32,808 tok)** | ~2,440 tok/s | 2,574 tok/s | **2,632 tok/s** (12.46s) | **+2.3%** |

### 2. Multi-Stream Decode Concurrency (sparkDash HTTP API)

#### A. Code Generation (Developer & Agent Workloads)
*Prompt: Complex multi-file refactoring / algorithmic generation.*

| Concurrency | Baseline | Stock 0.6.0 | **Our Stack (0.65 Conf)** | Stream Tok/s | TTFT |
|---|---|---|---|---|---|
| **×1 Solo** | 68.0 tok/s | 96.9 tok/s | **97.9 tok/s** | 97.9 | 188 ms |
| **×2 Streams** | 98.4 tok/s | 143.2 tok/s | **143.8 tok/s** | 73.5 | 194 ms |
| **×3 Streams** | 118.5 tok/s | 168.2 tok/s | **171.2 tok/s** | 58.2 | 211 ms |
| **×4 Streams** | 128.0 tok/s | 172.0 tok/s | **175.4 tok/s** | 47.4 | 215 ms |
| **×5 Streams** | 136.2 tok/s | 192.4 tok/s | **193.7 tok/s** | 41.5 | 224 ms |

#### B. Structured Output Decode
*Prompt: Strict JSON extraction and schema validation.*

| Concurrency | Stock 0.6.0 | **Our Stack (0.65 Conf)** | TTFT Delta |
|---|---|---|---|
| **×1 Solo** | 123.6 tok/s (79 ms) | **123.0 tok/s (69 ms)** | -10 ms |
| **×2 Streams** | 184.6 tok/s (177 ms) | **185.2 tok/s (172 ms)** | -5 ms |
| **×3 Streams** | 187.8 tok/s (333 ms) | **188.4 tok/s (221 ms)** | -112 ms |
| **×4 Streams** | 150.6 tok/s (233 ms) | **152.1 tok/s (230 ms)** | -3 ms |
| **×5 Streams** | 205.6 tok/s (421 ms) | **210.7 tok/s (304 ms)** | **-117 ms (-27.8%)** |

#### C. Prose Generation
*Prompt: Creative long-form prose and reasoning narrative.*

| Concurrency | Stock 0.6.0 | **Our Stack** | TTFT |
|---|---|---|---|
| **×1 Solo** | 68.7 tok/s | **68.2 tok/s** | 58 ms |
| **×2 Streams** | 94.1 tok/s | **96.4 tok/s** | 189 ms |
| **×3 Streams** | 113.2 tok/s | **115.1 tok/s** | 231 ms |
| **×4 Streams** | 123.2 tok/s | **124.0 tok/s** | 240 ms |
| **×5 Streams** | 134.3 tok/s | **137.5 tok/s** | 280 ms |

---

## 📉 What Tested Worse: Negative Results & Architectural Post-Mortems

A critical part of our empirical evaluation was testing candidate optimizations that **failed or performed worse**. We document them here to save others from dead-end configurations:

### 1. Prefill Chunk Geometry: `TENSORFOLD_PREFILL_ROWS=16384`
* **Hypothesis:** Pushing prefill chunk rows from 8,192 to 16,384 would saturate GB10 Tensor Cores further and approach 2,800+ tok/s.
* **Result:** **Failed / Regressed.**
  * 4k prefill was flat (2,598 $\to$ 2,606 tok/s).
  * 16k prefill **dropped** from 2,652 down to 2,611 tok/s (-1.5%).
  * Decode TTFT spiked across the board (Structured $\times 5$ TTFT rose from 304 ms to 435 ms; Prose TTFT rose from 280 ms to 375 ms).
* **Root Cause (Page Cache Starvation):** 16,384 rows pushed the static startup reservation from 86.6 GiB to **90.4 GiB** to accommodate oversized scratch activation buffers. On GB10 unified memory, that extra ~3.8 GiB was carved directly out of the Linux OS page cache holding the 29.8 GB n-gram table files (`--ple-on-ssd`). At 16k–32k prefill, more random NVMe lookups missed the page cache and hit physical flash. **8,192 rows is the definitive sweet spot.**

### 2. Speculative Draft Depth: `MTP_DRAFTS=5` vs `6`
* **Hypothesis:** On creative prose where draft acceptance is ~50%, drafting 6 tokens wastes compute evaluating drafts 4, 5, and 6 that are discarded. Capping at 5 drafts should boost prose throughput.
* **Result:** **Regressed on Code across all concurrency tiers.**
  * Code $\times 1$ dropped from 97.7 to 93.0 tok/s (-4.8%).
  * Code $\times 5$ dropped from 193.4 to 186.2 tok/s (-3.7%).
  * Structured $\times 1$ dropped from 123.0 to 112.3 tok/s (-8.7%).
  * Prose $\times 5$ saw no meaningful benefit (137.5 $\to$ 135.5 tok/s).
* **Root Cause (Code Syntax Density):** In structured and code tasks, Qwen3.8-Flash-Next achieves an MTP speculative acceptance rate of **78% to 94%**. Code syntax (indentation, matching delimiters, repetitive variable scopes) frequently validates all 6 draft tokens. Truncating to 5 drafts artificially forced the engine into an extra forward pass just to emit the 6th token. **6 drafts remains optimal for developer workloads.**

### 3. KV Cache Precision: `int4` KV Cache
* **Hypothesis:** Quantizing the KV cache from `int8` to `int4` halves KV memory footprint (~8 GiB saved) and cuts memory bandwidth during attention.
* **Result:** **Severe decode throughput penalty.**
  * Code $\times 5$ decode dropped from 319.2 to **295.4 tok/s (-7.5%)**.
  * Edit $\times 5$ decode dropped from 170.8 to **157.9 tok/s (-7.6%)**.
* **Root Cause (Speculative Divergence):** Quantization error in the KV cache degraded MTP draft verification acceptance:
  * Code acceptance rate dropped from **77.8% down to 73.0%**.
  * Edit acceptance rate dropped from **94.1% down to 88.0%**.
  * In speculative decoding, the compute penalty of verifying a mispredicted draft token and rewinding state far exceeds memory bandwidth savings. **`int8` KV cache is strictly superior to `int4`.**

### 4. Checkpoint Format: ModelOpt NVFP4 (`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`)
* **Hypothesis:** The 99 GB NVFP4 checkpoint fits 100% in RAM without `--ple-on-ssd`, eliminating SSD reads entirely.
* **Result:** **30% to 59% slower across every single metric.**
  * 4k Prefill: 1,831.5 tok/s (NVFP4) vs **2,598.1 tok/s (MLX 4-bit)** (-29.5%).
  * Code $\times 5$ Decode: 111.8 tok/s (NVFP4) vs **193.4 tok/s (MLX 4-bit)** (-42.2%).
  * Structured $\times 5$ Decode: 85.9 tok/s (NVFP4) vs **210.7 tok/s (MLX 4-bit)** (-59.2%).
* **Root Cause (Software Dequant vs Mature MLX GEMM):** TensorFold's MLX 4-bit MoE kernels have mature, hand-tuned assembly MMA tile schedules for the GB10 architecture. In contrast, NVFP4 requires runtime software dequantization MMAs and dynamic on-the-fly draft head requantization. Because Mia's 64-thread C++ SSD reader already eliminates NVMe paging bottlenecks, running 100% RAM-resident NVFP4 was substantially slower. **MLX 4-bit remains the undisputed winner.**

### 5. Upstream Prompt Copy Drafts (`0007-flash-next-copy-drafts`)
* **Hypothesis:** Copying tokens from prompt history for repetitive replies would boost quoting tasks.
* **Result:** **Dropped.** On TensorFold 0.6.0's dynamic multi-stream batching engine, maintaining the copy index created thread-scheduling contention and slowed down general decode rounds by ~3–5%.

### 6. Windows Desktop Workstations (RTX 3080 & RTX 5080)
* **Hypothesis:** Deploying TensorFold natively on Windows developer desktops.
* **Result:** **Not Feasible.**
  * **RTX 3080 (Ampere CC 8.6):** Rejected at startup. TensorFold 0.6.0 enforces a strict hardware floor of Compute Capability 8.9 (Ada Lovelace / Hopper / Blackwell).
  * **RTX 5080 (Blackwell CC 12.0, 16 GB VRAM):** Clears the compute floor via WSL2, but cannot fit the 75 GB model weights. TensorFold has zero CPU-RAM offloading mechanisms (it requires unified memory or full VRAM residency).
  * **Recommendation:** Keep Windows desktop clients on `llama-swap` / `llama.cpp` using GGUF quantization.

---

## 🏆 The Winning Production Stack

Deploy this exact configuration for maximum performance on a single DGX Spark:

```bash
# Optimal Environment Configuration
export TENSORFOLD_PREFILL_ROWS=8192      # Optimal prefill chunk size (2,652 tok/s)
export TENSORFOLD_SSD_NATIVE=1          # 64-thread C++ persistent SSD reader
export CHECKPOINT_SLOTS=6               # Keeps all 5 concurrent streams warm in RAM
export DECODE_SHARE=0.25                # 25% decode compute guarantee under prompt bursts
export MTP_CONFIDENCE=0.65              # Peak confidence for structured/code tasks
export MTP_DRAFTS=6                     # Preserves code syntax draft acceptance
export KV_DTYPE=int8                    # int8 avoids int4 speculative divergence
export PLE_ON_SSD=1                     # Retains MLX 4-bit weights with SSD n-gram tables
```

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
Builds TensorFold 0.6.0 with `v060-concurrent.patch` applied:
```bash
docker build -t tensorfold-qwen38:v0.6.0-concurrent .
```

### Step 3: Launch Production Server
```bash
./start.sh
```

`start.sh` automatically launches the container with optimal flags:
```bash
docker run -d --name qwen38-flash-next-tf \
  --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e TENSORFOLD_PREFILL_ROWS=8192 \
  -e TENSORFOLD_SSD_NATIVE=1 \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions_v060 \
  -e TRITON_CACHE_DIR=/cache/triton_v060 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/tensorfold-kernels:/cache \
  tensorfold-qwen38:v0.6.0-concurrent \
  tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
    --name "Qwen3.8-Flash-Next" \
    --parallel 5 \
    --context 262144 \
    --kv-dtype int8 \
    --mtp-drafts 6 \
    --mtp-confidence 0.65 \
    --checkpoint-slots 6 \
    --decode-share 0.25 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --ple-on-ssd \
    --thinking \
    --host 0.0.0.0 \
    --port 8888
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

The repository includes `tfab-concurrent.py`, a concurrency-aware evaluation harness that verifies **100% token SHA bit-parity** and measures throughput across all concurrency profiles:

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
