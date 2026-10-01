#!/usr/bin/env python3
"""Enhanced concurrency and scaling benchmark harness for TensorFold on DGX Spark.

Measures:
  1. Parity (Token SHA verification across greedy, sampled, long-context, and 5-stream concurrent).
  2. Prefill throughput & TTFT (4k, 16k, 32k, 64k tokens).
  3. Concurrent decode scaling across 1, 2, 3, 4, and 5 streams for Code, Prose, and Structured Edit.
  4. MTP draft acceptance rates per concurrency tier.

Usage:
  python3 tfab-concurrent.py parity <LABEL>
  python3 tfab-concurrent.py bench <LABEL>
  python3 tfab-concurrent.py all <LABEL>
"""

import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("TF_URL", "http://127.0.0.1:8888/v1/chat/completions")
HERE = Path(__file__).resolve().parent

WORDS = (
    "time year people way day man thing woman life child world school state family student group country "
    "problem hand part place case week company system program question work government number night point "
    "home water room mother area money story fact month lot right study book eye job word business issue "
    "side kind head house service friend father power hour game line end member law car city community name "
    "river mountain signal engine garden theory market winter method bridge letter window voice paper field"
).split()


def prose(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < tokens * 0.72:
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += sentence[:-1] + [sentence[-1] + "."]
    return " ".join(out)


def code_block(seed: int) -> str:
    rng = random.Random(seed)
    lines = []
    for i in range(40):
        a, b = rng.choice(WORDS), rng.choice(WORDS)
        lines += [
            f"def {a}_{b}_{i}(x, items):",
            f'    """Return the {a} of the {b} items above x."""',
            f"    total = 0",
            f"    for item in items:",
            f"        if item.{a} > x:",
            f"            total += item.{b} * {rng.randint(2, 9)}",
            f"    return total",
            "",
        ]
    return "\n".join(lines)


EDIT = (
    "Here is a Python module:\n\n```python\n"
    + code_block(5)
    + "```\n\nReturn the whole module unchanged except: "
    "rename the parameter `x` to `threshold` everywhere. Output only the code."
)

CASES = {
    "short_greedy": (
        [{"role": "user", "content": "Write a haiku about rivers, then explain it in two sentences."}],
        160,
        {"temperature": 0},
    ),
    "think_sampled": (
        [{"role": "user", "content": "Is 391 prime? Reason briefly."}],
        384,
        {"thinking": True, "seed": 1234},
    ),
    "long20k_greedy": (
        [{"role": "user", "content": prose(20_000, 11) + "\n\nQuote the first sentence above, then the last one."}],
        96,
        {"temperature": 0},
    ),
    "long45k_sampled": (
        [{"role": "user", "content": prose(45_000, 12) + "\n\nWhich word appears most often above? Answer, then explain."}],
        96,
        {"seed": 7},
    ),
    "edit_greedy": (
        [{"role": "user", "content": EDIT}],
        1200,
        {"temperature": 0},
    ),
    "edit_sampled": (
        [{"role": "user", "content": EDIT}],
        1200,
        {"seed": 99},
    ),
}


def call(messages, max_tokens, *, thinking=False, temperature=None, seed=None, timeout=900) -> dict:
    body = {
        "model": "x",
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    t = time.perf_counter()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    tf, u = r["tensorfold"], r["usage"]
    return {
        "sha": tf.get("token_sha", ""),
        "wall": time.perf_counter() - t,
        "prefill_s": tf.get("prefill_s", 0.0),
        "decode_s": tf.get("decode_s", 0.0),
        "prompt": u.get("prompt_tokens", 0),
        "out": u.get("completion_tokens", 0),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "drafted": tf.get("drafted", 0),
        "accepted": tf.get("accepted", 0),
    }


def together(jobs):
    out, threads = {}, []
    for name, (args, kw) in jobs:
        def go(name=name, args=args, kw=kw):
            out[name] = call(*args, **kw)
        threads.append(threading.Thread(target=go))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def parity(label: str) -> None:
    print(f"\n=== Running Parity Verification: {label} ===", flush=True)
    res = {}
    for name, (msgs, mt, kw) in CASES.items():
        res[name] = call(msgs, mt, **kw)
        print(f"  {name:18s} sha={res[name]['sha']} prompt={res[name]['prompt']:6d} out={res[name]['out']:5d}", flush=True)

    res["resend_long20k"] = call(CASES["long20k_greedy"][0], CASES["long20k_greedy"][1], **CASES["long20k_greedy"][2])
    print(f"  resend_long20k     sha={res['resend_long20k']['sha']} cached={res['resend_long20k']['cached']}")

    names = ["short_greedy", "think_sampled", "long45k_sampled", "edit_greedy", "edit_sampled"]
    conc = together([(n, (CASES[n][:2], CASES[n][2])) for n in names])
    for n in names:
        res["concurrent_" + n] = conc[n]
        print(f"  concurrent {n:18s} sha={conc[n]['sha']}")

    out_file = HERE / "results" / f"{label}-parity.json"
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(res, indent=1))
    print(f"Parity written to {out_file}\n")


def bench(label: str) -> None:
    print(f"\n=== Running Concurrency & Prefill Benchmark: {label} ===", flush=True)
    base = 2500
    res = {"prefill": {}, "decode": {}}

    print("\n--- 1. Prefill Scaling ---", flush=True)
    for size, runs in ((4_000, 3), (16_000, 2), (32_000, 2), (64_000, 1)):
        rates = []
        for i in range(runs):
            r = call([{"role": "user", "content": prose(size, base + size + i) + "\n\nSay OK."}], 1)
            rates.append((r["prompt"], r["prefill_s"], r["wall"]))
        tok = statistics.median(p / s for p, s, _ in rates)
        ttft = statistics.median(w for _, _, w in rates)
        res["prefill"][str(size)] = {"tok_s": tok, "wall_s": ttft, "prompt": rates[0][0]}
        print(f"  prefill {size:6d}: {tok:7.1f} tok/s  wall {ttft:6.2f} s  ({rates[0][0]} tokens)", flush=True)

    kinds = {
        "code": ([{"role": "user", "content": "Write a Python red-black tree with insert, delete and search, with docstrings."}], 512, {"temperature": 0}),
        "edit": ([{"role": "user", "content": EDIT}], 1200, {"temperature": 0}),
        "prose": ([{"role": "user", "content": "Write a 500-word story about a lighthouse keeper and a storm."}], 512, {"temperature": 0}),
        "chat_think": ([{"role": "user", "content": "Explain how a refrigerator works to a ten-year-old."}], 512, {"thinking": True, "seed": 42}),
    }

    print("\n--- 2. Multi-Stream Decode Scaling (1 to 5 Streams) ---", flush=True)
    for kind, (msgs, mt, kw) in kinds.items():
        stream_counts = (1, 2, 3, 4, 5) if kind != "chat_think" else (1,)
        for streams in stream_counts:
            reps = []
            num_reps = 3 if streams == 1 else 2
            for rep in range(num_reps):
                t = time.perf_counter()
                out = together([(f"{kind}{j}", ((msgs, mt), kw)) for j in range(streams)])
                wall = time.perf_counter() - t
                per = statistics.mean(r["out"] / max(0.001, r["decode_s"]) for r in out.values())
                agg = sum(r["out"] for r in out.values()) / max(0.001, wall)
                tot_drafted = sum(r["drafted"] or 0 for r in out.values())
                tot_accepted = sum(r["accepted"] or 0 for r in out.values())
                acc = tot_accepted / max(1, tot_drafted)
                reps.append((per, agg, acc))
            per, agg, acc = (statistics.median(x) for x in zip(*reps))
            key = f"{kind}x{streams}"
            res["decode"][key] = {
                "streams": streams,
                "per_stream": per,
                "aggregate_wall": agg,
                "acceptance": acc,
            }
            print(f"  decode {kind:10s} x{streams}: per-stream {per:6.1f} tok/s | aggregate {agg:6.1f} tok/s | accept {acc:.2f}", flush=True)

    out_file = HERE / "results" / f"{label}-bench.json"
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(res, indent=1))
    print(f"\nBenchmark written to {out_file}\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 tfab-concurrent.py parity|bench|all <LABEL>")
        sys.exit(1)
    mode, label = sys.argv[1], sys.argv[2]
    if mode in ("parity", "all"):
        parity(label)
    if mode in ("bench", "all"):
        bench(label)


if __name__ == "__main__":
    main()
