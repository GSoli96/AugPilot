"""Funzioni comuni ai worker: argomenti standard, misura di tempo/memoria, riga JSON finale."""
import argparse
import json
import os
import time
import traceback


def std_parser(desc: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return 0.0


def run_worker(fn, args) -> None:
    """fn(args) -> n_generated (int). Stampa come ultima riga il JSON del contratto."""
    t0 = time.time()
    rec = {"status": "SUCCESS", "n_generated": 0, "error_message": ""}
    try:
        rec["n_generated"] = int(fn(args))
    except Exception as e:  # noqa: BLE001
        rec = {"status": "FAILED", "n_generated": 0,
               "error_message": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"}
    rec["time_s"] = round(time.time() - t0, 3)
    rec["peak_mem_mb"] = round(_rss_mb(), 1)
    try:
        import torch
        if torch.cuda.is_available():
            rec["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    print(json.dumps(rec), flush=True)
