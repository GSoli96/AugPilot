#!/usr/bin/env python3
"""Worker TabularARGN reale: mostlyai-engine (split -> analyze -> encode -> train -> generate).

Gira nell'ambiente envs/argn (mostlyai-engine richiede torch recente). Ultima riga di stdout: JSON con
status (SUCCESS | FAILED), method_used, tempi. Nessun ripiego: se mostlyai-engine non funziona e' FAILED."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    try:
        _main()
    except Exception as e:
        print(json.dumps({"status": "FAILED", "method_used": "mostlyai-engine (TabularARGN)",
                          "error_message": f"{type(e).__name__}: {e}"[:1000]}))
        sys.exit(1)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_samples", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()

    from mostlyai.engine import split, analyze, encode, train, generate
    import torch

    dev = a.device
    dev_note = ""
    if dev != "cpu" and not torch.cuda.is_available():
        dev, dev_note = "cpu", f"GPU richiesta ma torch {torch.__version__} non vede CUDA: eseguito su CPU"

    df = pd.read_csv(a.csv)
    n_gen = len(df) if a.n_samples <= 0 else a.n_samples
    ws = BASE_DIR / "tmp" / f"mostly_ws_{Path(a.csv).stem}_{int(time.time() * 1000)}"
    ws.mkdir(parents=True, exist_ok=True)
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        split(tgt_data=df, workspace_dir=ws)
        analyze(workspace_dir=ws)
        encode(workspace_dir=ws)
        train(workspace_dir=ws, max_epochs=max(1, a.epochs), device=dev)
        t_train = time.perf_counter() - t0
        _, pk_tr = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        t1 = time.perf_counter()
        generate(workspace_dir=ws, sample_size=n_gen, device=dev)
        syn = ws / "SyntheticData"
        files = list(syn.glob("*.parquet")) + list(syn.glob("*.csv"))
        if not files:
            raise RuntimeError(f"nessun dato sintetico in {syn}")
        out = pd.read_parquet(files[0]) if files[0].suffix == ".parquet" else pd.read_csv(files[0])
        t_gen = time.perf_counter() - t1
        _, pk_gn = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        shutil.rmtree(ws, ignore_errors=True)

    if len(out) == 0:
        raise RuntimeError("mostlyai-engine ha generato 0 righe")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(json.dumps({"status": "SUCCESS", "method_used": "mostlyai-engine (TabularARGN)", "error_message": dev_note,
                      "train_time_s": round(t_train, 4), "gen_time_s": round(t_gen, 4),
                      "train_peak_mem_mb": round(pk_tr / 1024 ** 2, 2), "gen_peak_mem_mb": round(pk_gn / 1024 ** 2, 2),
                      "rss_delta_mb": 0.0}))


if __name__ == "__main__":
    import seed_util; seed_util.apply_env_seed()
    main()
