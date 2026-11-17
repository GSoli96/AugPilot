#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import tracemalloc
from pathlib import Path

import numpy as np
for _attr in ("long", "ulong"):
    if not hasattr(np, _attr):
        setattr(np, _attr, int)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
ALGORITHMS_DIR = BASE_DIR / "Algorithms"
goggle_src = (ALGORITHMS_DIR / "GOGGLE-main" / "src") if (ALGORITHMS_DIR / "GOGGLE-main").exists() else (BASE_DIR / "GOGGLE-main" / "src")
if str(goggle_src) not in sys.path:
    sys.path.insert(0, str(goggle_src))

from goggle.GoggleModel import GoggleModel


def main():
    try:
        _main()
    except Exception as e:  # mai SUCCESS dopo un errore
        print(json.dumps({"status": "FAILED", "error_message": f"{type(e).__name__}: {e}"[:1000]}))
        sys.exit(1)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    ct = ColumnTransformer(
        [("scaler", StandardScaler(), feature_cols)],
        remainder="passthrough"
    )
    X = ct.fit_transform(df)
    df_preprocessed = pd.DataFrame(X, columns=feature_cols + [target_col])

    valid_classes = sorted(df[target_col].unique())
    n_synth = len(df) if args.n_samples <= 0 else args.n_samples

    # Training
    tracemalloc.start()
    t0_train = time.perf_counter()
    model = GoggleModel(
        ds_name=csv_path.stem,
        input_dim=df_preprocessed.shape[1],
        encoder_dim=64,
        encoder_l=2,
        het_encoding=True,
        decoder_dim=16,
        decoder_l=2,
        threshold=0.1,
        decoder_arch="gcn",
        device=args.device,
        beta=1,
        learning_rate=0.01,
        seed=42,
        epochs=args.epochs,
    )
    model.fit(df_preprocessed)
    train_time = time.perf_counter() - t0_train
    _, peak_train = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Generation
    tracemalloc.start()
    t0_gen = time.perf_counter()
    X_ref = df_preprocessed.sample(n=n_synth, replace=True, random_state=42).reset_index(drop=True)
    X_synth = model.sample(X_ref)

    # Snap class to nearest valid label
    X_synth[target_col] = (
        X_synth[target_col]
        .apply(lambda x: min(valid_classes, key=lambda v: abs(v - x)))
        .astype(df[target_col].dtype)
    )

    # Inverse transform continuous features back to original scale
    X_synth_features = X_synth[feature_cols].values
    scaler = ct.named_transformers_["scaler"]
    inv_features = scaler.inverse_transform(X_synth_features)
    final_df = pd.DataFrame(inv_features, columns=feature_cols)
    final_df[target_col] = X_synth[target_col].values

    final_df.to_csv(out_path, index=False)
    gen_time = time.perf_counter() - t0_gen
    _, peak_gen = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = {
        "status": "SUCCESS",
        "method_used": "goggle",
        "error_message": "",
        "train_time_s": round(train_time, 4),
        "gen_time_s": round(gen_time, 4),
        "train_peak_mem_mb": round(peak_train / (1024 ** 2), 2),
        "gen_peak_mem_mb": round(peak_gen / (1024 ** 2), 2),
        "rss_delta_mb": round(max(peak_train, peak_gen) / (1024 ** 2), 2),
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    import seed_util; seed_util.apply_env_seed()
    main()
