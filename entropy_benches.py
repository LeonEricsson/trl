#!/usr/bin/env python
"""
Compare two entropy implementations on GPU while recording full‐timeline
CUDA-memory snapshots that can be explored with https://pytorch.org/memory_viz .

Requirements:
  • PyTorch ≥ 2.1 (built w/ CUDA)          • pynvml  (pip install pynvml)
"""

from __future__ import annotations

import datetime as dt
import pathlib
import threading
import time

import pynvml
import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────────
# 0. Helper – live-sampling of GPU utilisation via NVML
# ────────────────────────────────────────────────────────────────────────────────
def sample_gpu_util(interval_s: float = 0.05, device_idx: int = 0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    util: list[int] = []
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            util.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            time.sleep(interval_s)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()  # ← start the sampling thread
    return util, stop, th


# ────────────────────────────────────────────────────────────────────────────────
# 1.  The entropy functions
# ────────────────────────────────────────────────────────────────────────────────
def entropy_from_logits_nocast(logits: torch.Tensor, chunk_size: int | None = 1):
    outs = []
    for chunk in logits.split(chunk_size, dim=0):
        logZ = chunk.logsumexp(-1, keepdim=True)
        p = (chunk - logZ).exp()  # in-place
        outs.append(logZ.squeeze(-1) - (p * chunk).sum(-1))
    return torch.cat(outs, dim=0)


def rowise_entropy(logits: torch.Tensor, chunk_size: int = 1) -> torch.Tensor:
    outs = []
    for chunk in logits.split(chunk_size, dim=0):
        logp = F.log_softmax(chunk, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        outs.append(entropy)
    return torch.cat(outs, dim=0)


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 1) -> torch.Tensor:
    outs = []
    for chunk in logits.split(chunk_size, dim=0):
        c32 = chunk.float()
        logZ = torch.logsumexp(c32, -1, keepdim=True)
        p = (c32 - logZ).exp()
        entropy = logZ.squeeze(-1) - (p * c32).sum(-1)
        outs.append(entropy.to(chunk.dtype))
    return torch.cat(outs, 0)


def orig(logits):
    """
    Computes the entropy of a tensor of shape [B, S, V] row-wise to reduce the memory
    consumed by the softmax operation.
    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
    Returns:
        `torch.Tensor`:
            Entropy of each position in a sequence
    """
    per_token_entropies = []
    for (row_logits,) in zip(logits):  # loop to reduce peak mem consumption
        row_logps = F.log_softmax(row_logits, dim=-1)
        row_entropy = -torch.exp(row_logps) * row_logps
        per_token_entropies.append(row_entropy)
    per_token_entropies = torch.stack(per_token_entropies)
    return per_token_entropies.sum(-1)


def entropy_from_logits_nochunk(logits: torch.Tensor, chunk_size=None):
    return torch.stack(
        [
            (lz := lg.logsumexp(-1, keepdim=True)).squeeze(-1)  # log Z
            - ((lg - lz).exp() * lg).sum(-1)  #   −∑ p x
            for lg in logits
        ],
        dim=0,
    )


# ────────────────────────────────────────────────────────────────────────────────
# 2.  Benchmark runner with memory snapshots
# ────────────────────────────────────────────────────────────────────────────────
def run_benchmark(fn, logits, *, runs: int, label: str, snapshot_dir: pathlib.Path):
    # Prepare snapshot filename (e.g. snapshots/2025-06-22T14-52-38_rowise.pkl)
    stamp = dt.datetime.now().strftime("%Y-%m-dT%H-%M-%S")
    snap_path = snapshot_dir / f"{stamp}_{label}.pkl"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- WARM-UP ---------------------------------------------------------------
    for _ in range(5):
        fn(logits)

    # --- PERFORMANCE + MEMORY PROFILE -----------------------------------------
    torch.cuda.reset_peak_memory_stats()
    util, stop_evt, th = sample_gpu_util()

    torch.cuda.memory._record_memory_history(max_entries=200_000)
    start_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()

    for _ in range(runs):
        fn(logits)

    end_evt = torch.cuda.Event(enable_timing=True)
    end_evt.record()
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(str(snap_path))
    torch.cuda.memory._record_memory_history(enabled=None)

    # --- COLLECT METRICS -------------------------------------------------------
    stop_evt.set()
    th.join()
    elapsed_ms = start_evt.elapsed_time(end_evt) / runs
    peak_mem = torch.cuda.max_memory_allocated() / 2**20  # → MiB
    avg_util = sum(util) / len(util) if util else 0.0

    print(f"{label:<25} | {elapsed_ms:8.2f} ms | {peak_mem:8.1f} MiB | {avg_util:6.1f} % | {snap_path.name}")  # noqa: T201

    return snap_path


def run_benchmark_time(fn, logits, *, runs: int, label: str, chunk_size: int = 1) -> float:
    """Return average kernel latency in **milliseconds** (no memory snapshots)."""

    # Warm‑up to stabilise GPU clocks / cache
    for _ in range(5):
        fn(logits, chunk_size=chunk_size)

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()

    for _ in range(runs):
        fn(logits, chunk_size=chunk_size)

    end_evt = torch.cuda.Event(enable_timing=True)
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms = start_evt.elapsed_time(end_evt) / runs
    print(f"{label:<25} | {elapsed_ms:8.4f} ms")  # noqa: T201


# ────────────────────────────────────────────────────────────────────────────────
# 3.  Main entry –  synthetic workload and invocation
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")  # noqa: T201

    B, S, V = 64, 1024, 50_257  # GPT-like per-token logits
    logits = torch.randn(B, S, V, device=device, dtype=torch.float16)

    # ------------------------------------------------------------------
    #  correctness assertion (ref vs. optimised, both un-chunked & chunked)
    # ------------------------------------------------------------------
    o = orig(logits)
    ref = rowise_entropy(logits)
    fast = entropy_from_logits(logits)  # vectorised

    assert torch.allclose(o, ref, rtol=1e-03, atol=1e-06), "Does not match"

    print("✔ Entropy implementations match within tolerance.\n")  # noqa: T201

    chunk_sizes = [1]
    runs = 20

    # for cs in chunk_sizes:
    #     run_benchmark_time(rowise_entropy,      logits, runs=runs,
    #                        label=f"rowise_entropy({cs})",      chunk_size=cs)
    #     run_benchmark_time(entropy_from_logits,  logits, runs=runs,
    #                        label=f"entropy_from_logits({cs})",  chunk_size=cs)
    #     print("-" * 60)

    # ----------------------------------------------------------------——
    #   FULL PROFILE (uncomment if needed)
    # ----------------------------------------------------------------——
    snaps = []
    for cs in chunk_sizes:
        snaps.append(
            run_benchmark(orig, logits, runs=runs, label=f"orig({cs})", snapshot_dir=pathlib.Path("snapshots"))
        )
        snaps.append(
            run_benchmark(
                rowise_entropy,
                logits,
                runs=runs,
                label=f"rowise_entropy({cs})",
                snapshot_dir=pathlib.Path("snapshots"),
            )
        )
        snaps.append(
            run_benchmark(
                entropy_from_logits_nocast,
                logits,
                runs=runs,
                label=f"entropy_from_logits_nocast({cs})",
                snapshot_dir=pathlib.Path("snapshots"),
            )
        )
        snaps.append(
            run_benchmark(
                entropy_from_logits_nochunk,
                logits,
                runs=runs,
                label=f"entropy_from_logits_nochunk({cs})",
                snapshot_dir=pathlib.Path("snapshots"),
            )
        )
        snaps.append(
            run_benchmark(
                entropy_from_logits,
                logits,
                runs=runs,
                label=f"entropy_from_logits({cs})",
                snapshot_dir=pathlib.Path("snapshots"),
            )
        )

    print(  # noqa: T201
        "\nDone!  You can now drag any snapshot from ./snapshots/ onto "
        "https://pytorch.org/memory_viz for an interactive timeline."
    )
