#!/usr/bin/env python3
"""Foreground TTFT behind background load, for TensorFold --name-priority (v0.6.6).

Usage: bgprio-bench.py BASE_URL MAIN_ID BG_ID OUT.json [reps] [streams=5]
Scenarios (each: N long background-ish requests, then a short foreground request 4 s later):
  ctl4 / ctl5   : load uses MAIN_ID, no priority (control: server treats everything as foreground)
  flag4 / flag5 : load uses BG_ID (--name-priority BG_ID=background); fg uses MAIN_ID
  own5          : load uses BG_ID; fg uses BG_ID + its own "priority":"normal" (own priority must win)
  bgself5       : load AND fg both use BG_ID, no priority (fg is itself background; must wait like the load)
With S = streams: S-1 loads = one stream free (names ctl4/flag4 for S=5); S loads = every stream busy, so fg must wait for a slot or preempt (ctl5/flag5/own5/bgself5 for S=5).
"""
import json, sys, threading, time, urllib.request

base, MAIN, BG, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
S = int(sys.argv[6]) if len(sys.argv) > 6 else 5
TOPICS = ["a lighthouse keeper", "the history of salt", "a failing space station", "tide pools",
          "a chess tournament", "medieval bridges", "a lost violin", "volcano monitoring"]


def call(model, prompt, max_tokens, extra=None, hold=None):
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    body.update(extra or {})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = (d.get("choices") or [{}])[0].get("delta", {})
            if ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return {"ttft": ttft, "chunks": n, "total": time.time() - t0}


def scenario(name, n_load, load_model, fg_model, fg_extra, rep):
    res = {}
    def load(i):
        res[f"bg{i}"] = call(load_model, f"Write a very long, detailed story about {TOPICS[(i + rep) % 8]}. "
                             f"Do not stop early. Variant {rep}-{i}.", 900)
    ths = [threading.Thread(target=load, args=(i,)) for i in range(n_load)]
    for t in ths: t.start()
    time.sleep(4)
    fg = call(fg_model, f"In one short sentence, what is {7 + rep} times 13?", 64, fg_extra)
    fg_done_at = time.time()
    for t in ths: t.join()
    bg_rate = [v["chunks"] / v["total"] for k, v in res.items()]
    return {"scenario": name, "rep": rep, "fg_ttft": fg["ttft"], "fg_total": fg["total"],
            "bg_chunks_per_s_mean": sum(bg_rate) / len(bg_rate), "bg_total_s_max": max(v["total"] for v in res.values())}


plan = [(f"ctl{S-1}", S - 1, MAIN, MAIN, None), (f"flag{S-1}", S - 1, BG, MAIN, None),
        (f"ctl{S}", S, MAIN, MAIN, None), (f"flag{S}", S, BG, MAIN, None),
        (f"own{S}", S, BG, BG, {"priority": "normal"}), (f"bgself{S}", S, BG, BG, None)]
rows = []
for rep in range(reps):
    for name, n, lm, fm, fx in plan:
        r = scenario(name, n, lm, fm, fx, rep)
        rows.append(r)
        print(f"{name:8s} rep{rep} fg_ttft={r['fg_ttft']:.2f}s fg_total={r['fg_total']:.2f}s "
              f"bg={r['bg_chunks_per_s_mean']:.1f} chunk/s", flush=True)
        time.sleep(3)
json.dump(rows, open(out, "w"), indent=1)
