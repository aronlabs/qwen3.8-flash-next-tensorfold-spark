# Results: MiaAI-Lab recipe 1.0.0 (TensorFold Zig engine, INT4-AutoRound)

Raw data for the "Current build" section of the top-level README. Hardware: one DGX Spark (GB10, 121 GiB unified memory).
Recipe: [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `364f21a` (release 1.0.0), image `tensorfold-qwen38:zig-db28187` (TensorFold `db281878` plus the recipe's patches, label
`tf.patches=4144696f21e3`), checkpoint `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` @ `14642741`. Stock recipe, no local edits.

## Benchmarks, 2026-10-08 (idle server, stock recipe: no CPU pinning, 2,048-row prompt chunks)

| File | What it is |
|---|---|
| `ab-bench-protocol3.log` | ab-bench run 31: protocol 3, greedy, thinking off, 400 tokens, 1-8 streams, long prefill. Medians per cell, spread flags. Published at [bench.gummie.dev](https://bench.gummie.dev) as model `qwen3.8-flash-next-int4ar` |
| `zig100-{structured,prose,code}.json`, `zig100-r2-*.json` | sparkDash decode sweeps, x1-x5, 400 max tokens, two runs |
| `zig100-prefill.json`, `zig100-r2-prefill.json` | sparkDash prefill sweeps (4k / 16k / 32k / 64k) |
| `checks.log` | the recipe's own checks: `toolcheck.py` 3/3, `visioncheck.py` OK, `needle.py` (199,730-token prompt, found), `context_boundary.py --context 262144` 16/16 |

## NoRoom retest, 2026-10-11 (`retest-2026-10-11/`)

Goal: reproduce upstream issue [#39](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/issues/39)
(requests refused with `NoRoom` on an idle server after a mixed burst). Server as above, `--parallel 8 --context 262144`, FP8 KV, vision on.
Pool sizes are in `pool-lines.txt` (engine log line `sequence memory`).

| File | Run |
|---|---|
| `noroom-run1.log` | Issue #39's phases against the 33 GiB pool. **Flawed:** the 200k-token prompts were built with a 2.2 tokens/word guess and came out over 262,144 tokens, so phases 2, 3 and 6 returned HTTP 400 (context length), not NoRoom. Phases 1, 4 and 5 are valid (all OK). The script in this folder is the corrected version |
| `noroom-run2.log` | Same phases, prompts recalibrated against the server's own token count (about 4.15 tokens per generated word; 50,000 requested tokens had come back as 94,015). 8 short, 4x200k, 6x200k, mixed 3x100k + 4 large images, 15-token probe, 200k probes on the idle server: all OK, 0 NoRoom, `MemAvailable` 35-39 GiB throughout |
| `fill8.log` | 8 concurrent ~200k prompts at the 33 GiB pool. Not a valid result: eight 200k prefills take longer than the 1,800 s client timeout, so every request timed out client-side; the aborted requests stayed queued server-side (8 running, 17 waiting) until the container was restarted |
| `fill8b.log` | Pool shrunk to 18.3 GiB (`TENSORFOLD_MEMORY_RESERVE_GIB=25`, about the reporter's 18.9): two rounds of 8 concurrent ~140k prompts, `max_tokens` 4000, then idle probes. All 16 OK (576-1,559 s each), idle probes OK (130 s, 14 s), `MemAvailable` 37 GiB at start, 43-45 GiB at the end. 0 NoRoom |
| `noroom_repro.py`, `fill8.py`, `fill8b.py` | The drivers. Run them on the Spark; they read `/proc/meminfo` and talk to `:8888` |

Reading: not reproduced with synthetic load. 8 streams all near full context can still be refused by design: one full 262,144 window is
4.37 GiB of FP8 KV (CHANGELOG), the 33 GiB pool holds about 7.5 of them, and admission counts prompt plus `max_tokens`.
