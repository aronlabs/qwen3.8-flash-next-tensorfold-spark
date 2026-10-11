#!/usr/bin/env python3
"""Reproducer for MiaAI-Lab TensorFold #39 (NoRoom on idle server after mixed burst). Runs on the Spark.
Phases (each fired all at once, finished before the next): 1) 8 short  2) 4x~200k  3) 6x~200k  4) mixed 3x~100k + 4 images  5) 15-token probe
then probes a ~200k request on the idle server. Prints MemAvailable and /metrics idle state after each phase."""
import base64, io, json, os, random, struct, sys, threading, time, urllib.request, urllib.error, zlib

API = os.environ.get("API", "http://127.0.0.1:8888")
random.seed(1)
WORDS = open("/usr/share/dict/words").read().split() if os.path.exists("/usr/share/dict/words") else \
    [f"w{i}x{random.randint(0, 9999)}" for i in range(5000)]


def memavail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return round(int(l.split()[1]) / 1048576, 1)


def prompt_of(tokens):  # ~1.3 tokens/word is a rough guess; random words defeat the prompt cache
    return " ".join(random.choice(WORDS) + str(random.randint(0, 99)) for _ in range(int(tokens / 4.15)))


def png(w=1600, h=2200):
    raw = b"".join(b"\x00" + bytes(random.randrange(256) for _ in range(w * 3 // 8)) * 8 for _ in range(h))
    def ch(t, d): c = struct.pack(">I", len(d)) + t + d; return c + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw, 1)) + ch(b"IEND", b"")


def req(content, max_tokens, results, i):
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(API + "/v1/chat/completions", json.dumps(body).encode(),
                                                         {"Content-Type": "application/json"}), timeout=3500)
        d = json.load(r)
        results[i] = ("OK", round(time.time() - t, 1), d["usage"]["prompt_tokens"])
    except urllib.error.HTTPError as e:
        results[i] = (f"HTTP{e.code}", round(time.time() - t, 1), e.read()[:120].decode(errors="replace"))
    except Exception as e:
        results[i] = ("ERR", round(time.time() - t, 1), repr(e)[:100])


def phase(name, jobs):
    res = [None] * len(jobs)
    ts = [threading.Thread(target=req, args=(c, m, res, i)) for i, (c, m) in enumerate(jobs)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(f"== {name}: " + "; ".join(f"{r[0]}({r[1]}s)" for r in res), flush=True)
    for r in res:
        if r[0] != "OK": print("   ", r)
    idle()
    return res


def idle():
    time.sleep(5)
    try:
        h = json.load(urllib.request.urlopen(API + "/health", timeout=10))
    except Exception as e:
        h = {"err": repr(e)}
    print(f"   MemAvailable={memavail()} GiB health={ {k: h.get(k) for k in ('requests_running','streams')} }", flush=True)


def img_msg():
    b = base64.b64encode(png()).decode()
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}}, {"type": "text", "text": "Describe this image briefly."}]


if __name__ == "__main__":
    print("start MemAvailable", memavail()); idle()
    phase("1 8xshort", [("Write a short story about a fox.", 1500)] * 8)
    phase("2 4x200k", [(prompt_of(200000) + "\nSay OK.", 32) for _ in range(4)])
    phase("3 6x200k", [(prompt_of(200000) + "\nSay OK.", 32) for _ in range(6)])
    phase("4 mixed", [(prompt_of(100000) + "\nSay OK.", 32) for _ in range(3)] + [(img_msg(), 200) for _ in range(4)])
    phase("5 short", [("Say hi.", 15)])
    for n in (1, 2):
        phase(f"6 idle probe x{n} 200k", [(prompt_of(200000) + "\nSay OK.", 32) for _ in range(n)])
