#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import threading
import tracemalloc
from typing import Optional, Tuple, Dict, Any

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def get_current_rss_mb() -> float:
    """Return current process RSS in MB."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return 0.0


def get_gpu_memory_mb() -> float:
    """Return max GPU memory allocated in MB if CUDA is active, else 0."""
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


class PhaseTracker:
    """
    Context manager to track execution time and memory for a single phase
    (e.g., training or generation).
    """

    def __init__(self, phase_name: str = "phase"):
        self.phase_name = phase_name
        self.elapsed_time_s: float = 0.0
        self.peak_py_mem_mb: float = 0.0
        self.rss_before_mb: float = 0.0
        self.rss_after_mb: float = 0.0
        self.rss_delta_mb: float = 0.0
        self.gpu_peak_mb: float = 0.0
        self._t0: float = 0.0

    def __enter__(self):
        tracemalloc.start()
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.rss_before_mb = get_current_rss_mb()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time_s = time.perf_counter() - self._t0
        self.rss_after_mb = get_current_rss_mb()
        self.rss_delta_mb = max(0.0, self.rss_after_mb - self.rss_before_mb)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_py_mem_mb = peak_bytes / (1024 ** 2)
        if _HAS_TORCH and torch.cuda.is_available():
            self.gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    def to_dict(self) -> Dict[str, float]:
        return {
            f"{self.phase_name}_time_s": round(self.elapsed_time_s, 4),
            f"{self.phase_name}_peak_mem_mb": round(self.peak_py_mem_mb, 2),
            f"{self.phase_name}_rss_delta_mb": round(self.rss_delta_mb, 2),
            f"{self.phase_name}_gpu_peak_mb": round(self.gpu_peak_mb, 2),
        }


class SubprocessMonitor:
    """
    Monitors memory usage of a subprocess and its child tree every 0.1s.
    """

    def __init__(self, proc):
        self.proc = proc
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread = None
        if _HAS_PSUTIL:
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()

    def _monitor(self):
        try:
            p = psutil.Process(self.proc.pid)
            while not self._stop_event.is_set() and self.proc.poll() is None:
                try:
                    procs = [p] + p.children(recursive=True)
                    tot_rss = sum(pr.memory_info().rss for pr in procs if pr.is_running())
                    if tot_rss > self.peak_bytes:
                        self.peak_bytes = tot_rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                time.sleep(0.1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        return self.peak_bytes / (1024 ** 2)
