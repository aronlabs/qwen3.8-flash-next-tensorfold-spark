# Qwen3.8 Flash-Next on NVIDIA DGX Spark with TensorFold 0.6 (python-0.6 @ed78d6f): Concurrency, Prefill, Speculative & Copy-Draft Tuning

High-throughput, exact local inference for **Qwen3.8-Flash-Next** on a single **NVIDIA DGX Spark** (GB10 ARM64, 121 GB unified memory).

This repository ports, extends, and benchmarks the high-performance patches from **@MiaAI_Lab's** DGX Spark recipe onto **@ashxhart's** **TensorFold** (the `python-0.6` branch at [`ed78d6f`](https://github.com/ashhart/TensorFold/commit/ed78d6fc204d89d90b045bf033d6551e7714f3a1): v0.6.6 plus 12 commits), introducing Cortex-X925 core affinity pinning (`--cpuset-cpus "5-9,15-19"`), zero-copy Python-CUDA pointer passing, greedy decoding fast paths, prefill chunk geometry tuning, message boundary prefix cache multi-tenancy, and Blackwell-optimized tile schedules. The current build adds **copy drafts**, authored by **[BobClawblaw (@BobClawblaw)](https://github.com/BobClawblaw)** in [TensorFold PR #468](https://github.com/ashhart/TensorFold/pull/468) (commit [`6bf4caf`](https://github.com/ashhart/TensorFold/commit/6bf4caf)), with a small local patch so they work at this recipe's 6-draft depth.

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

## 🆕 Copy drafts on python-0.6 (2026-10-07)

**Why not TensorFold 1.0?** 1.0.0/1.0.1 (released 2026-10-07) is the native Zig engine. On the GB10 it serves only Nemotron 3.5 Lightning; Flash Next on 1.0 is Apple-only and answers one request at a time. The release's Linux binary run on this Spark says so directly: `no registered Zig family serves model_type qwen4_exp; the 0.6 line may`. The Python engine continues on the `python-0.6` branch, which this build now tracks.

**What copy drafts do.** BobClawblaw's [PR #468](https://github.com/ashhart/TensorFold/pull/468): when a stream's last 8 tokens already appeared earlier in its prompt or reply, the round drafts the continuation of that earlier text instead of asking the MTP head. The verify step is unchanged, so every reply equals one-token decoding. It helps whenever a reply repeats its input: quoting, editing a passage, rewriting a file.

**Why they needed a patch here.** Upstream's `CopyIndex.chain()` only copies when the draft depth is at least the 8-token match length. This recipe drafts 6 a round (`--mtp-drafts 6`, also TensorFold's own CUDA default for Flash Next), so stock copy drafts never fired: draft counts matched v0.6.6 exactly. Raising the depth to 8 or 12 turns them on but costs single-stream fresh prose and code 11-25%, because MTP chains also run deeper. `local-copy-match.patch` (3 lines) keeps the 8-token match and trims the copied chain to the depth, the same approach Mia's AI Lab uses in its [`nvfp4-v066`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/nvfp4-v066) recipe patch. Nothing else changes: same concurrent-vision patchset (it applies to the branch with one 22-line offset), same flags, same `-bg` background id.

**Repeat-heavy replies** (`copy-bench.py`: a ~300-word passage and a 160-line Python file; median decode tok/s; every reply token-identical to v0.6.6):

| Task | v0.6.6 greedy | This build, greedy | v0.6.6 sampled | This build, sampled |
|---|---|---|---|---|
| Quote the passage | 102.8 | **135.1 (+31%)** | 92.8 | **113.5 (+22%)** |
| Fix typos in the passage | 100.7 | **114.1 (+13%)** | 67.6 | **72.5 (+7%)** |
| Rename a function across the file | 98.7 | **134.9 (+37%)** | 64.0 | **66.7 (+4%)** |
| Fresh prose | 57.0 | 56.7 (-0.5%) | 65.0 | 64.7 (-0.4%) |
| Fresh code | 88.7 | 86.3 (-2.8%) | 59.5 | 59.3 (-0.3%) |

Greedy is temperature 0 with thinking off. Sampled is the server's own sampling (1.0 / 0.95 / 20) with thinking on, so most tokens are new reasoning that copies cannot help with; that is why the gain shrinks there.

**Standard sweep against v0.6.6** ([`ab-bench`](https://bench.gummie.dev) protocol: temperature 0, thinking off, 400 tokens, one cold round discarded then 3 timed rounds; aggregate decode tok/s, v0.6.6 -> this build):

| Prompt | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| structured | 129 -> 127 | 233 -> 232 | 322 -> 322 | 408 -> 406 | 462 -> 441 |
| prose | 71 -> 71 | 130 -> 130 | 169 -> 182 | 228 -> 228 | 269 -> 269 |
| code | 104 -> 101 | 149 -> 142 | 179 -> 170 | 185 -> 182 | 209 -> 203 |
| json | 92 -> 88 | 168 -> 145* | 235 -> 225 | 295 -> 283 | 346 -> 331 |

| Prefill | v0.6.6 | This build |
|---|---|---|
| 1K | 1762 tok/s, 0.58 s | 1781 tok/s, 0.57 s |
| 2K | 2351 tok/s, 0.86 s | 2390 tok/s, 0.85 s |
| 4K | 2706 tok/s, 1.49 s | 2701 tok/s, 1.49 s |
| 8K | 2882 tok/s, 2.80 s | 2878 tok/s, 2.81 s |
| 16K | 2872 tok/s, 5.62 s | 2856 tok/s, 5.65 s |
| 32K | 2795 tok/s, 11.54 s | 2783 tok/s, 11.59 s |

\* One of the three rounds read 112 tok/s; the other two match the rest of the row (about -4%).

**The trade-off, stated plainly.** Prefill and prose are unchanged. Short greedy code and JSON replies lose 2-5%: these prompts repeat their own patterns, so some proposed copies get rejected and take the place of MTP drafts. A control run of the same branch without `local-copy-match.patch` (copies never fire) matched v0.6.6 on code (104 / 149 / 179 / 190 / 209 tok/s), so the cost comes from copy drafts, not from the rest of the branch. For agent work that quotes, edits and rewrites its input, the gain is larger than that cost. To turn copies off without rebuilding, add `-e TENSORFOLD_COPY_DRAFTS=0` to the `docker run` in `start.sh`.

Raw results: `results/v0.6.6-py06/` (`copy-bench-a-066*` = v0.6.6, `copy-bench-b-py06*` = branch without the patch at depth 6, 8 and 12, `copy-bench-c-copy8-d6*` = this build). The full sweep, with per-round data and CSV exports, is on [bench.gummie.dev](https://bench.gummie.dev).

---

## 🆕 TensorFold 0.6.6 (2026-10-06): background priority by model id

TensorFold 0.6.6 changes one thing over 0.6.5: the opt-in `--name-priority ID=background` flag on the CUDA server. A request that names that served id (`--name` or an `--alias`) and sends no `priority` of its own is served as `priority: "background"`, so it yields to foreground requests; a request's own `priority` always wins. It suits clients that can pick a model id but cannot add a field to the request body, such as batch extractors or sub-agents.

`start.sh` now serves two ids: `Qwen3.8-Flash-Next` (normal priority) and `Qwen3.8-Flash-Next-bg` (`--alias Qwen3.8-Flash-Next-bg --name-priority Qwen3.8-Flash-Next-bg=background`). Delete those two flags to serve one id. `v065-concurrent-vision.patch` applies unchanged to 0.6.6 (the file keeps its name), so the speed patchset is the same.

**Foreground latency behind background load.** A short foreground request is sent 4 s into a load of long generations (median of 3 repetitions, 5 streams, `bgprio-bench.py`):

| Scenario | Foreground time to first token |
|---|---|
| 4 of 5 streams busy, no flag | 0.28 s |
| 4 of 5 streams busy, load on the `-bg` id | 0.20 s |
| All 5 streams busy, no flag | **33.0 s** |
| All 5 streams busy, load on the `-bg` id | **0.28 s** |
| All 5 busy on the `-bg` id, foreground sends its own `"priority":"normal"` | 0.27 s |
| Foreground also on the `-bg` id, no priority | 32.1 s (it is background too) |

Background streams decode at 11.6 chunks/s per stream with the flag against 11.7 without, and a sweep sent entirely to the `-bg` id matches the foreground sweep (prefill within 1%, decode the same).

**Speed against 0.6.5** (idle server, [`ab-bench`](https://bench.gummie.dev) protocol: temperature 0, thinking off, 400 tokens, one cold round discarded then 3 timed rounds; aggregate decode tok/s, 0.6.5 -> 0.6.6):

| Prompt | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| structured | 129 -> 129 | 233 -> 233 | 324 -> 322 | 409 -> 408 | 464 -> 462 |
| prose | 71 -> 71 | 130 -> 130 | 169 -> 169 | 230 -> 228 | 268 -> 269 |
| code | 104 -> 104 | 147 -> 149 | 179 -> 179 | 186 -> 185 | 209 -> 209 |
| json | 92 -> 92 | 168 -> 168 | 236 -> 235 | 296 -> 295 | 342 -> 346 |

| Prefill | 0.6.5 | 0.6.6 |
|---|---|---|
| 1K | 1756 tok/s, 0.58 s | 1762 tok/s, 0.58 s |
| 2K | 2353 tok/s, 0.86 s | 2351 tok/s, 0.86 s |
| 4K | 2711 tok/s, 1.49 s | 2706 tok/s, 1.49 s |
| 8K | 2873 tok/s, 2.81 s | 2882 tok/s, 2.80 s |
| 16K | 2874 tok/s, 5.62 s | 2872 tok/s, 5.62 s |
| 32K | 2795 tok/s, 11.54 s | 2795 tok/s, 11.54 s |

Every decode cell is within 1.1% of 0.6.5 and prefill within 0.4%; time to first token is unchanged. These `ab-bench` figures use a different harness from the sparkDash tables further down, so compare like with like. The full archive, with per-round data and CSV exports, is at [bench.gummie.dev](https://bench.gummie.dev). `bgprio-bench.py` in this repo reproduces the latency table.

---

## 👥 Credits & Acknowledgments

This project builds directly on the foundational work of:

* **[Ash Hart (@ashxhart)](https://github.com/ashhart/TensorFold):** Creator of **TensorFold**, an open-source, local-first inference engine designed for exact speculative decoding. Version 0.6.2 added SM 12.0 DeltaNet tree kernel speedups, grouped projection launches, coalesced waiting prefill passes, Blackwell layer projection tile fusion, shared prefix retention across message boundaries, and native vLLM-mirrored Prometheus metrics.
* **[Mia's AI Lab (@MiaAI_Lab)](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold):** Created the original DGX Spark deployment recipe and authored the 64-thread persistent C++ multithreaded SSD n-gram reader (`ssd_read.cpp`), asynchronous read-ahead, and initial memory layout recipes.
* **[MovieMaker93 (@MovieMaker93)](https://github.com/MovieMaker93):** Authored upstream [TensorFold PR #40](https://github.com/ashhart/TensorFold/pull/40) for dynamic prefill chunk geometry (`indexed_prefill_rows()`), enabling wide chunk sizing up to 16,384 rows.
* **[Vontra](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP):** For the MLX affine 4-bit group-32 checkpoint with the native MTP speculative draft head.
* **[philip-pentatonic (@philip-pentatonic)](https://github.com/philip-pentatonic):** Authored the `--name-priority` flag in [TensorFold PR #445](https://github.com/ashhart/TensorFold/pull/445), released in 0.6.6.
* **[BobClawblaw (@BobClawblaw)](https://github.com/BobClawblaw):** Authored Flash Next CUDA copy drafts in [TensorFold PR #468](https://github.com/ashhart/TensorFold/pull/468) (commit [`6bf4caf`](https://github.com/ashhart/TensorFold/commit/6bf4caf)) and image input on the serial engine and two ranks in [TensorFold PR #473](https://github.com/ashhart/TensorFold/pull/473), both on the `python-0.6` branch this build uses. `local-copy-match.patch` only changes when his copy drafts fire.

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

All benchmarks were collected on the live **NVIDIA DGX Spark** (GB10 ARM64, 121 GB unified memory) using **sparkDash** (OpenAI API over HTTP) on 2026-10-03 against TensorFold v0.6.5 with Cortex-X925 core pinning.

### 1. Prefill Throughput vs Context Length

| Prompt Context | Exact Tokens | Prefill Speed | Time to First Token |
|---|---|---|---|
| **4k** | 4,133 tok | **2,672.8 tok/s** | 1.55 s |
| **8k** | 8,229 tok | **2,789.6 tok/s** | 2.95 s |
| **16k** | 16,421 tok | **2,748.1 tok/s** | 5.98 s |
| **32k** | 32,805 tok | **2,696.4 tok/s** | 12.17 s |

### 2. Multi-Stream Decode Scaling (1 to 5 Streams)

#### Structured Output (`Count 1 to 200`)
| Concurrent Streams | Aggregate Throughput | Per-Stream Throughput | Time to First Token |
|---|---|---|---|
| **$\times 1$ Solo** | 117.1 tok/s | 117.1 tok/s | 67 ms |
| **$\times 2$** | 121.0 tok/s | 61.5 tok/s | 184 ms |
| **$\times 3$** | 131.8 tok/s | 49.5 tok/s | 215 ms |
| **$\times 4$** | 128.3 tok/s | 49.5 tok/s | 324 ms |
| **$\times 5$** | **175.3 tok/s** | 50.0 tok/s | **267 ms** *(37% faster TTFT vs 0.6.1)* |

#### Code Generation (Python Quicksort)
| Concurrent Streams | Aggregate Throughput | Per-Stream Throughput | Time to First Token |
|---|---|---|---|
| **$\times 1$ Solo** | 67.0 tok/s | 67.0 tok/s | 245 ms |
| **$\times 2$** | 119.8 tok/s | 62.5 tok/s | 189 ms |
| **$\times 3$** | 172.0 tok/s | 58.5 tok/s | 211 ms |
| **$\times 4$** | 148.0 tok/s | 46.0 tok/s | 211 ms |
| **$\times 5$** | **186.7 tok/s** | 41.8 tok/s | **216 ms** |

#### Prose Generation (Hash Map Explanation)
| Concurrent Streams | Aggregate Throughput | Per-Stream Throughput | Time to First Token |
|---|---|---|---|
| **$\times 1$ Solo** | 60.5 tok/s | 60.5 tok/s | 62 ms |
| **$\times 2$** | 87.3 tok/s | 46.6 tok/s | 201 ms |
| **$\times 3$** | 108.7 tok/s | 38.7 tok/s | 221 ms |
| **$\times 4$** | 109.7 tok/s | 31.2 tok/s | 642 ms |
| **$\times 5$** | **124.1 tok/s** | 29.4 tok/s | **292 ms** |
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
Builds TensorFold's `python-0.6` branch at `ed78d6f` with `v065-concurrent-vision.patch` and `local-copy-match.patch` applied:
```bash
docker build -t tensorfold-qwen38:v0.6.6-py06-ed78d6f-copy8-concurrent-vision .
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

# Copy-draft bench: repeat-heavy vs fresh replies (bring your own ~300-word text and a Python file)
python3 copy-bench.py mylabel --passage passage.txt --code some_module.py            # greedy, thinking off
python3 copy-bench.py mylabel-sampled --passage passage.txt --code some_module.py --sampled --reps 4 --max-tokens 2500
```

Benchmark output logs and parity proofs are stored under `results/`.

---

## 📄 License

* **TensorFold** is licensed under Apache-2.0 by Ash Hart.
* Ported modifications, speed patches, and scripts in this repository are licensed under Apache-2.0.
