import sys; sys.path.insert(0, "/home/aron/nemotron-test")
import noroom_repro as r
print("start", r.memavail(), flush=True)
r.phase("A 8x140k mt=4000", [(r.prompt_of(140000) + "\nSummarize in one line.", 4000) for _ in range(8)])
r.phase("B 8x140k mt=4000", [(r.prompt_of(140000) + "\nSummarize in one line.", 4000) for _ in range(8)])
r.phase("C idle probe 1x140k", [(r.prompt_of(140000) + "\nSay OK.", 32)])
r.phase("D idle probe 1x20k", [(r.prompt_of(20000) + "\nSay OK.", 32)])
print("DONE")
