import re
from pathlib import Path

tf_root = Path("/home/aron/projects/tensorfold-v0.6.1-stage/TensorFold/src/tensorfold")

# 1. host_table.py
p_host = tf_root / "families/qwen4_exp/host_table.py"
t = p_host.read_text()
old_fn = "def open_table(model_dir: Path, shards: list[tuple[str, str]], scale, *, ssd: bool = False):"
new_fn = "def open_table(model_dir: Path, shards: list[tuple[str, str]], scale, *, ssd: bool = False, native=None):"
assert old_fn in t, "open_table signature not found in host_table.py"
t = t.replace(old_fn, new_fn)
old_tbl = 'table = BF16Table(files) if used[0] == "bf16" else SSDTable(files) if ssd else HostTable(files)'
new_tbl = 'table = BF16Table(files) if used[0] == "bf16" else SSDTable(files, native=native) if ssd else HostTable(files)'
assert old_tbl in t, "table instantiate not found in host_table.py"
t = t.replace(old_tbl, new_tbl)
p_host.write_text(t)
print("Updated host_table.py")

# 2. ssd_table.py
p_ssd = tf_root / "families/qwen4_exp/ssd_table.py"
t = p_ssd.read_text()
old_w = "WORKERS = 16                # reads in flight at once: os.pread releases the GIL"
new_w = "WORKERS = 16                # reads in flight at once: os.pread releases the GIL\nNATIVE_WORKERS = 64         # the native reader's threads hold no GIL: the NVMe's deep queue, no cost to the caller"
assert old_w in t, "WORKERS not found in ssd_table.py"
t = t.replace(old_w, new_w)

old_init = "    def __init__(self, files: list[tuple[Path, dict, dict, dict]], *, nocache: bool = True) -> None:"
new_init = "    def __init__(self, files: list[tuple[Path, dict, dict, dict]], *, nocache: bool = True, native=None) -> None:\n        self._native = native"
assert old_init in t, "__init__ not found in ssd_table.py"
t = t.replace(old_init, new_init)

old_gather = "        local, where = unique - self.starts[shard], self.fidx[shard]"
new_gather = "        local, where = unique - self.starts[shard], self.fidx[shard]\n        if self._native is not None:\n            return self._gather_native(unique.size, inverse, shard, local, where)"
assert old_gather in t, "gather point not found in ssd_table.py"
t = t.replace(old_gather, new_gather)

helper_methods = '''
    def _gather_native(self, n: int, inverse, shard, local, where) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The same runs as ``_reads``, listed as one array and read by ``native`` into one buffer."""

        widths = (self.wrow, self.grow, self.grow)
        buf = np.empty(n * sum(widths), dtype=np.uint8)
        runs, at = [], 0
        for part, width in enumerate(widths):
            runs.append(self._runs(where, self.bases[shard, part] + local * width, width, at))
            at += n * width
        self._native(np.concatenate(runs), buf, NATIVE_WORKERS)
        words = buf[:n * self.wrow].reshape(n, self.wrow)
        scales = buf[n * self.wrow:n * (self.wrow + self.grow)].reshape(n, self.grow)
        biases = buf[n * (self.wrow + self.grow):].reshape(n, self.grow)
        return (words[inverse].view(np.uint32), scales[inverse].view(np.uint16), biases[inverse].view(np.uint16))

    def _runs(self, where: np.ndarray, offsets: np.ndarray, width: int, at: int) -> np.ndarray:
        """``_reads`` as rows (fd, offset, size, at) of an int64 array, row i of the part landing at ``at + i * width``."""

        n = offsets.size
        if not n:
            return np.zeros((0, 4), dtype=np.int64)
        cut = np.ones(n, dtype=bool)
        cut[1:] = (where[1:] != where[:-1]) | (offsets[1:] != offsets[:-1] + width)
        first = np.flatnonzero(cut)
        cut |= (np.arange(n) - first[np.cumsum(cut) - 1]) % max(1, MAX_READ // width) == 0
        first = np.flatnonzero(cut)
        rows = np.diff(first, append=n)
        return np.stack([self._fd_of[where[first]], offsets[first], rows * width, at + first * width], axis=1)
'''

old_reads = "    def _reads(self, where: np.ndarray, offsets: np.ndarray, width: int, out: np.ndarray) -> list[tuple]:"
assert old_reads in t, "_reads not found in ssd_table.py"
t = t.replace(old_reads, helper_methods + "\n" + old_reads)
p_ssd.write_text(t)
print("Updated ssd_table.py")

# 3. weights.py
p_w = tf_root / "families/qwen4_exp/cuda/weights.py"
t = p_w.read_text()
old_imp = "from ..host_table import open_table, shard_keys"
new_imp = "from ..host_table import ReadAhead, open_table, shard_keys\nfrom .ssd_read import native_reader"
assert old_imp in t, "import not found in weights.py"
t = t.replace(old_imp, new_imp)

old_open = """        table = open_table(model_dir, [(rd.where[k + ".weight"], k) for k in keys],
                           lambda n: table_scale(base, n), ssd=ple_on_ssd)"""
new_open = """        table = open_table(model_dir, [(rd.where[k + ".weight"], k) for k in keys],
                           lambda n: table_scale(base, n), ssd=ple_on_ssd, native=native_reader() if ple_on_ssd else None)
        if ple_on_ssd:              # from SSD, a prompt pass's rows are read on a thread while the GPU runs the one before
            table = ReadAhead(table)"""
assert old_open in t, "open_table call not found in weights.py"
t = t.replace(old_open, new_open)
p_w.write_text(t)
print("Updated weights.py")

# 4. gdn.py
p_gdn = tf_root / "cuda/kernels/gdn.py"
t = p_gdn.read_text()
old_to_dev = """def to_device(values: Sequence[int], dtype: torch.dtype, device) -> torch.Tensor:
    \"\"\"One pinned, non-blocking host-to-device copy (the caching host allocator keeps the buffer until it lands).\"\"\"

    return torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)"""
new_to_dev = """def to_device(values: Sequence[int], dtype: torch.dtype, device) -> torch.Tensor:
    \"\"\"One pinned, non-blocking host-to-device copy (the caching host allocator keeps the buffer until it lands).\"\"\"

    import numpy as np

    if isinstance(values, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(values))
        if t.dtype != dtype:
            t = t.to(dtype)
        return t.pin_memory().to(device, non_blocking=True)
    return torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)"""
assert old_to_dev in t, "to_device not found in gdn.py"
t = t.replace(old_to_dev, new_to_dev)
p_gdn.write_text(t)
print("Updated gdn.py")

# 5. attn_multi.py
p_attn = tf_root / "families/qwen4_exp/cuda/attn_multi.py"
t = p_attn.read_text()
old_attn = """        dev = w.device
        ints = shared.to_device(np.concatenate([posr, sid, first, counts]).tolist(), torch.int32, dev)
        self.posr, self.sid = ints[:rows], ints[rows:2 * rows]
        self.first, self.counts = ints[2 * rows:2 * rows + n], ints[2 * rows + n:]
        self.ptrs = shared.to_device(ptrs.ravel().tolist(), torch.int64, dev).view(len(layers), PTRS * n)"""
new_attn = """        dev = w.device
        first_arr, counts_arr = np.asarray(first, dtype=np.int32), np.asarray(counts, dtype=np.int32)
        ints = shared.to_device(np.concatenate([posr, sid, first_arr, counts_arr]), torch.int32, dev)
        self.posr, self.sid = ints[:rows], ints[rows:2 * rows]
        self.first, self.counts = ints[2 * rows:2 * rows + n], ints[2 * rows + n:]
        self.ptrs = shared.to_device(ptrs.ravel(), torch.int64, dev).view(len(layers), PTRS * n)"""
assert old_attn in t, "attn_multi conversion not found"
t = t.replace(old_attn, new_attn)
p_attn.write_text(t)
print("Updated attn_multi.py")

# 6. gdn_multi.py
p_gdn_m = tf_root / "families/qwen4_exp/cuda/gdn_multi.py"
t = p_gdn_m.read_text()
old_gdn_m = """        dev = w.device
        i32 = shared.to_device(ints.tolist(), torch.int32, dev)
        i64 = shared.to_device(ptrs.ravel().tolist(), torch.int64, dev)"""
new_gdn_m = """        dev = w.device
        i32 = shared.to_device(ints, torch.int32, dev)
        i64 = shared.to_device(ptrs.ravel(), torch.int64, dev)"""
assert old_gdn_m in t, "gdn_multi conversion not found"
t = t.replace(old_gdn_m, new_gdn_m)
p_gdn_m.write_text(t)
print("Updated gdn_multi.py")

# 7. forward.py
p_fwd = tf_root / "families/qwen4_exp/cuda/forward.py"
t = p_fwd.read_text()
fwd_helpers = '''
def read_ahead(w: Weights, windows: Sequence[tuple[np.ndarray | None, Sequence[int]]]) -> None:
    """Start reading the n-gram rows a later ``stage`` of each (history, tokens) looks up, where a table reads ahead
    (``--ple-on-ssd``): the same bytes, read on a thread while the GPU runs the current forward."""

    if w.x3 is not None:
        return
    for layer in w.layers:
        p = layer.ple
        start = getattr(p.table, "read_ahead", None) if p is not None else None
        if start is None:
            continue
        for history, tokens in windows:
            if len(tokens):
                start(p.ngram.ids(history, np.asarray(tokens, dtype=np.int64)))


def history_after(w: Weights, history: np.ndarray | None, tokens: Sequence[int]) -> np.ndarray | None:
    """The n-gram history once ``tokens`` follow ``history`` (as ``commit`` leaves it after keeping them all)."""

    if history is None:
        return None
    return np.concatenate([history, np.asarray(tokens, dtype=np.int64)])[-(w.cfg.ngram_size - 1):]
'''
old_compute = "def compute(w: Weights, segs: Sequence[Seg], b: Buffers, *, logits: bool = True, context: int | None = None,"
assert old_compute in t, "compute definition not found in forward.py"
t = t.replace(old_compute, fwd_helpers + "\n\n" + old_compute)
p_fwd.write_text(t)
print("Updated forward.py")

# 8. decode.py
p_dec = tf_root / "families/qwen4_exp/cuda/decode.py"
t = p_dec.read_text()
old_fwd_imp = "from .forward import Cut, commit, cut_snapshot, forward"
new_fwd_imp = "from .forward import Cut, commit, compute, cut_snapshot, forward, history_after, read_ahead, stage"
assert old_fwd_imp in t, "forward import not found in decode.py"
t = t.replace(old_fwd_imp, new_fwd_imp)

old_prefill = "    logits = forward(w, st, pb, chunk, logits=final, cut=cut)"
new_prefill = """    segs = stage(w, pb, [(st, chunk)])
    if not final:                # the next chunk's n-gram rows are read while the GPU runs this one
        read_ahead(w, [(history_after(w, st.ple_history, chunk), prompt[end:end + e.prefill_rows])])
    logits = compute(w, segs, pb, logits=final, cuts=() if cut is None else (cut,))"""
assert old_prefill in t, "prefill_chunk forward call not found in decode.py"
t = t.replace(old_prefill, new_prefill)
p_dec.write_text(t)
print("Updated decode.py")

# 9. multi.py
p_multi = tf_root / "families/qwen4_exp/cuda/multi.py"
t = p_multi.read_text()
old_multi_imp = "from .forward import Cut, commit, compute, compute_mixed, converges, cut_snapshot, stage"
new_multi_imp = "from .forward import Cut, commit, compute, compute_mixed, converges, cut_snapshot, history_after, read_ahead, stage"
assert old_multi_imp in t, "forward import not found in multi.py"
t = t.replace(old_multi_imp, new_multi_imp)

old_pass_min = "PASS_MIN = 128                   # the fewest prompt rows a round's pass takes"
new_pass_min = "PASS_MIN = 128                   # the fewest prompt rows a round's pass takes\nREAD_AHEAD = 2                   # prompt pieces of the next pass whose n-gram rows are read ahead (ReadAhead.depth)"
assert old_pass_min in t, "PASS_MIN not found in multi.py"
t = t.replace(old_pass_min, new_pass_min)

read_ahead_method = '''
    def _read_ahead(self, pieces, rows: int) -> None:
        """Start reading the n-gram rows of the pass likely to follow this one (tables that read ahead, from SSD)
        while the GPU runs this one; a wrong guess only costs the reads."""

        after = {s.sid: a + n for s, a, n in pieces}
        windows, room = [], rows
        for s in sorted(self.filling, key=lambda x: x.background):
            start = after.get(s.sid, self.fills[s.sid][2])
            n = min(len(s.prompt) - start, room)
            if n <= 0:
                continue
            staged = s.st.ple_last if s.sid in after else None        # this pass's piece, not committed yet
            history = history_after(self.w, *staged) if staged is not None else s.st.ple_history
            windows.append((history, s.prompt[start:start + n]))
            room -= n
            if room <= 0 or len(windows) == READ_AHEAD:
                break
        if windows:
            read_ahead(self.w, windows)
'''
old_pass = "    def _pass(self) -> list[Stream]:"
assert old_pass in t, "_pass definition not found in multi.py"
t = t.replace(old_pass, read_ahead_method + "\n" + old_pass)

# Inside _pass:
old_pass_stage = "            segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])"
new_pass_stage = "            segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])\n            self._read_ahead(pieces, self.prefill_rows)"
assert old_pass_stage in t, "pass stage not found in multi.py"
t = t.replace(old_pass_stage, new_pass_stage)

# Inside _round:
old_round_stage = """            try:
                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                cuts = self._cuts(pieces, psegs)"""
new_round_stage = """            try:
                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                cuts = self._cuts(pieces, psegs)
                self._read_ahead(pieces, self._pass_rows())"""
assert old_round_stage in t, "round stage not found in multi.py"
t = t.replace(old_round_stage, new_round_stage)

# Greedy fast path in _picks:
old_picks = """        row = logits.float()
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])"""
new_picks = """        row = logits.float()
        if all(s is None or s.temperature <= 0 for s in samplings):
            top, col = row.max(dim=-1)
            lse = torch.logsumexp(row, dim=-1)
            col_cpu = col.cpu().numpy()
            prob_cpu = torch.exp(top - lse).cpu().numpy()
            out = []
            for i in range(len(positions)):
                c = int(col_cpu[i])
                tok = int(self.draft_host[c]) if self.draft_host is not None else c
                out.append((tok, float(prob_cpu[i])))
            return out
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])"""
assert old_picks in t, "_picks start not found in multi.py"
t = t.replace(old_picks, new_picks)
p_multi.write_text(t)
print("Updated multi.py")

# 10. geometry.py
p_geom = tf_root / "cuda/geometry.py"
t = p_geom.read_text()
indexed_fn = '''
def indexed_prefill_rows() -> int:
    """Flash Next's prompt chunk rows on MLX and NVFP4 checkpoints: PREFILL_ROWS, or TENSORFOLD_PREFILL_ROWS (256 to
    16,384; EXL3 packs keep PREFILL_ROWS). A row's bits never depend on its chunk: wider chunks prefill long prompts
    faster and take more memory, which on a unified-memory GPU the n-gram tables' page cache may need."""

    import os
    rows = int(os.environ.get("TENSORFOLD_PREFILL_ROWS") or PREFILL_ROWS)
    if not 256 <= rows <= 16384:
        raise ValueError(f"TENSORFOLD_PREFILL_ROWS: 256 to 16,384 rows, not {rows}")
    return rows
'''
old_size = 'def size(info: dict, name: str = "tensor") -> int:'
assert old_size in t, "size not found in geometry.py"
t = t.replace(old_size, indexed_fn + "\n\n" + old_size)
p_geom.write_text(t)
print("Updated geometry.py")

# 11. prompt_plan.py
p_plan = tf_root / "families/qwen4_exp/cuda/prompt_plan.py"
t = p_plan.read_text()
old_idle = "IDLE_ROWS = 4096"
new_idle = 'import os\nIDLE_ROWS = int(os.environ.get("TENSORFOLD_PREFILL_ROWS") or 4096)'
assert old_idle in t, "IDLE_ROWS not found in prompt_plan.py"
t = t.replace(old_idle, new_idle)

old_limit = "    limit = min(rows, PREFILL_ROWS) if live else rows"
new_limit = '    env_rows = int(os.environ.get("TENSORFOLD_PREFILL_ROWS") or 0)\n    limit = (min(rows, env_rows or PREFILL_ROWS) if live else rows)'
assert old_limit in t, "limit not found in prompt_plan.py"
t = t.replace(old_limit, new_limit)
p_plan.write_text(t)
print("Updated prompt_plan.py")

# 12. engine.py
p_eng = tf_root / "families/qwen4_exp/cuda/engine.py"
t = p_eng.read_text()
m = re.search(r"from tensorfold\.cuda\.geometry import [^\n]+", t)
assert m, "geometry import not found in engine.py"
t = t.replace(m.group(0), m.group(0) + ", indexed_prefill_rows")

old_plan_call = """        self.prefill_rows, prompt_workspace = (PREFILL_ROWS, 0) if exl3 else prompt_plan(
            self.capacity_plan, config(model_dir), torch.cuda.get_device_capability(), world=tp, vision=vision,
            fp8=prompt_precision.fp8())"""
new_plan_call = """        self.prefill_rows, prompt_workspace = (PREFILL_ROWS, 0) if exl3 else prompt_plan(
            self.capacity_plan, config(model_dir), torch.cuda.get_device_capability(), world=tp, vision=vision,
            fp8=prompt_precision.fp8())
        env_prefill = indexed_prefill_rows() if not exl3 else PREFILL_ROWS
        if env_prefill != PREFILL_ROWS:
            self.prefill_rows = env_prefill"""
assert old_plan_call in t, "prompt_plan call not found in engine.py"
t = t.replace(old_plan_call, new_plan_call)

old_geom_def = """        geometry = ((lambda text: indexed_stream_geometry(text, streams, each, KEEP, mtp=mtp, kv_bits=bits))
                    if streams > 1 else
                    (lambda text: gdn_geometry(text, tp, each, indexed=True, mtp=mtp, kv_bits=bits,
                                               kept=KEEP_SERIAL + 1)))"""
new_geom_def = """        rows_geom = PREFILL_ROWS if exl3 else indexed_prefill_rows()
        geometry = ((lambda text: indexed_stream_geometry(text, streams, each, KEEP, mtp=mtp, kv_bits=bits, prefill_rows=rows_geom))
                    if streams > 1 else
                    (lambda text: gdn_geometry(text, tp, each, indexed=True, mtp=mtp, kv_bits=bits,
                                               kept=KEEP_SERIAL + 1, prefill_rows=rows_geom)))"""
assert old_geom_def in t, "geometry def not found in engine.py"
t = t.replace(old_geom_def, new_geom_def)

old_how = '        if ple_on_ssd:\n            how = "read from SSD at each lookup"'
new_how = '''        if ple_on_ssd:
            from .ssd_read import native_reader
            how = (f"read from SSD at each lookup ({'native' if native_reader() is not None else 'Python'} reader, "
                   f"prompt rows read ahead)")'''
assert old_how in t, "how string not found in engine.py"
t = t.replace(old_how, new_how)

p_eng.write_text(t)
print("Updated engine.py")
print("All optimizations applied successfully!")
