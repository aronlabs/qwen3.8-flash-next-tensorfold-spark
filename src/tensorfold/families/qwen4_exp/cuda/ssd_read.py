"""The SSD n-gram reader's native path: a batch of preads on C++ threads, the GIL released (``ssd_read.cpp``)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    return load(name="tensorfold_qwen4_exp_ssd_read", sources=[str(Path(__file__).parent / "ssd_read.cpp")],
                extra_cflags=["-O2"], verbose=False)


@lru_cache(maxsize=1)
def native_reader():
    """``SSDTable``'s ``native`` reader, or None where the extension does not build (TENSORFOLD_SSD_NATIVE=0: None)."""

    if os.environ.get("TENSORFOLD_SSD_NATIVE", "1") == "0":
        return None
    try:
        ext = _ext()
    except Exception as exc:                                  # no compiler: the Python reader
        print(f"[tensorfold] n-gram SSD reads stay on Python threads (the native reader did not build: {exc})",
              flush=True)
        return None
    import torch

    def read(reads: np.ndarray, out: np.ndarray, threads: int) -> None:
        ext.read_rows(torch.from_numpy(np.ascontiguousarray(reads, dtype=np.int64)), torch.from_numpy(out), threads)

    return read
