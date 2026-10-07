#!/usr/bin/env python3
"""Copy-heavy vs fresh-text decode bench for a TensorFold server.

Usage: copy-bench.py LABEL --passage TEXT_FILE --code PY_FILE [--reps N] [--url URL] [--sampled]
Writes results/copy-bench-LABEL.json. Default: temperature 0, thinking off; --sampled uses the
server's own sampling with thinking on. Streamed; decode tok/s = (completion_tokens - 1) /
(last chunk - first token). Tasks: quote the passage, fix seeded typos in it, rename the code's
first function in the whole file, then a fresh story and fresh code as controls.
"""
import argparse, hashlib, json, os, pathlib, random, re, statistics, sys, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent


def typos(text, seed=7):
    rnd = random.Random(seed)
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and w.isalpha() and rnd.random() < 0.12:
            j = rnd.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def first_def(code):
    m = re.search(r"^def (\w+)\(", code, flags=re.M)
    return m.group(1)


def tasks(passage, code):
    fn = first_def(code)
    return fn, {
        "quote": f"Repeat the following text exactly, word for word, with no commentary:\n\n{passage}",
        "typofix": "Fix every spelling mistake in the following text. Output the full corrected text only, "
                   f"changing nothing else:\n\n{typos(passage)}",
        "rename": f"Rename the function `{fn}` to `{fn}_renamed` everywhere in this file. Output the complete "
                  f"modified file only, in one code block:\n\n```python\n{code}\n```",
        "fresh_prose": "Write an original 700-word short story about a lighthouse keeper who discovers a "
                       "message in a bottle. Plain prose, no headings.",
        "fresh_code": "Write a complete, well-commented Python module implementing an LRU cache with TTL "
                      "expiry, thread safety, and a small CLI demo. Output only the code.",
    }


def health(url):
    with urllib.request.urlopen(url.replace("/v1", "") + "/health", timeout=10) as r:
        return json.load(r)


def run(url, prompt, max_tokens, sampled=False):
    body = {
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not sampled:  # sampled: the server's production sampling (1.0/0.95/20) with thinking on
        body.update(temperature=0, chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = t_last = None
    text, usage, extra = [], None, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("tensorfold"):
                extra = ev["tensorfold"]
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if piece or delta.get("reasoning_content"):
                    now = time.monotonic()
                    t_first = t_first or now
                    t_last = now
                if piece:
                    text.append(piece)
    out = "".join(text)
    n = usage["completion_tokens"] if usage else None
    tps = (n - 1) / (t_last - t_first) if n and t_last and t_last > t_first else None
    return {"ttft_s": round(t_first - t0, 3) if t_first else None, "completion_tokens": n,
            "decode_tps": round(tps, 2) if tps else None,
            "sha": hashlib.sha256(out.encode()).hexdigest()[:16], "tensorfold": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--url", default=os.environ.get("TF_URL", "http://127.0.0.1:8888/v1"))
    ap.add_argument("--passage", type=pathlib.Path, required=True, help="~300-word prose text to quote")
    ap.add_argument("--code", type=pathlib.Path, required=True, help="Python file; its first 160 lines are used")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--sampled", action="store_true", help="production sampling, thinking on")
    a = ap.parse_args()
    code = "\n".join(a.code.read_text().splitlines()[:160])
    fn, task_map = tasks(a.passage.read_text().strip(), code)
    h = health(a.url)
    if h.get("requests_running"):
        sys.exit(f"server not idle: {h.get('requests_running')} running")
    res = {"label": a.label, "fn_renamed": fn, "tasks": {}}
    for name, prompt in task_map.items():
        reps = [run(a.url, prompt, a.max_tokens, a.sampled) for _ in range(a.reps)]
        tps = [r["decode_tps"] for r in reps if r["decode_tps"]]
        res["tasks"][name] = {"median_decode_tps": statistics.median(tps) if tps else None,
                              "shas": sorted({r["sha"] for r in reps}), "reps": reps}
        print(f"{name:12s} median {res['tasks'][name]['median_decode_tps']} tok/s "
              f"tokens {[r['completion_tokens'] for r in reps]} shas {res['tasks'][name]['shas']}", flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"copy-bench-{a.label}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
