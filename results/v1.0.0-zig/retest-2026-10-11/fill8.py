import sys; sys.path.insert(0, "/home/aron/nemotron-test")
import noroom_repro as r
print("start", r.memavail(), flush=True)
for rnd, (n, mt) in enumerate([(200000, 2000), (200000, 32768), (230000, 8000)], 1):
    r.phase(f"round{rnd} 8x{n//1000}k max_tokens={mt}", [(r.prompt_of(n) + "\nSummarize in one line.", mt) for _ in range(8)])
r.phase("idle probe 1x200k", [(r.prompt_of(200000) + "\nSay OK.", 32)])
