#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import subprocess
import tracemalloc
from pathlib import Path

import numpy as np
for _attr in ("long", "ulong"):
    if not hasattr(np, _attr):
        setattr(np, _attr, int)

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
ALGORITHMS_DIR = BASE_DIR / "Algorithms"
TAB_DDPM_DIR = (ALGORITHMS_DIR / "tab-ddpm-main") if (ALGORITHMS_DIR / "tab-ddpm-main").exists() else (BASE_DIR / "tab-ddpm-main")

if str(TAB_DDPM_DIR) not in sys.path:
    sys.path.insert(0, str(TAB_DDPM_DIR))
if str(TAB_DDPM_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(TAB_DDPM_DIR / "scripts"))


def prepare_dataset_for_ddpm(df: pd.DataFrame, output_dir: Path, drop_constant: bool = False) -> dict:
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y_raw = df[target_col].values
    unique_labels = sorted(np.unique(y_raw))
    label_map = {v: i for i, v in enumerate(unique_labels)}
    y = np.array([label_map[v] for v in y_raw], dtype=np.int64)

    n_classes = max(2, len(unique_labels))
    n_features = X.shape[1]

    # Stratified or simple split
    try:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    except Exception:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    # CTAB-GAN(+) scarta per campionamento con rifiuto ogni riga con un valore fuori da [min, max] del train
    # (transformer.inverse_transform). Su una colonna costante nel train min == max e il valore ricostruito
    # u*4*std + mean non e' mai esattamente uguale: tutte le righe vengono scartate e, dopo il limite di
    # 600 s del ciclo di ricampionamento, l'output e' vuoto (es. yeast1: Attr5 = 0.0 in tutta la minoritaria).
    # Una colonna costante nel train non porta informazione: si toglie prima dell'addestramento e si
    # reinserisce col suo valore dopo la generazione.
    constant_cols: dict = {}
    if drop_constant:
        keep = []
        for j, c in enumerate(feature_cols):
            col = X_tr[:, j]
            col = col[~np.isnan(col)]
            if len(np.unique(col)) <= 1:
                constant_cols[c] = float(col[0]) if len(col) else 0.0
            else:
                keep.append(j)
        if not keep:
            raise RuntimeError("tutte le colonne sono costanti nel train: niente da apprendere")
        if constant_cols:
            feature_cols = [feature_cols[j] for j in keep]
            X_tr, X_val, X_test = X_tr[:, keep], X_val[:, keep], X_test[:, keep]
            n_features = len(keep)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, Xs, ys in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_test, y_test)]:
        np.save(output_dir / f"X_num_{split}.npy", Xs)
        np.save(output_dir / f"y_{split}.npy", ys)

    info = {
        "task_type": "binclass" if n_classes == 2 else "multiclass",
        "n_classes": n_classes,
        "num_num_features": n_features,
        "num_cat_features": 0,
        "feature_names": feature_cols,
        "target_name": target_col,
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=4))
    return {
        "n_classes": n_classes,
        "n_features": n_features,
        "train_size": len(y_tr),
        "feature_names": feature_cols,
        "target_name": target_col,
        "unique_labels": unique_labels,
        "constant_cols": constant_cols,
        "all_feature_names": [c for c in df.columns if c != target_col],
    }


def ensure_columns_entry(json_path: Path, dataset_name: str, include_general: bool):
    if not json_path.exists():
        cols = {}
    else:
        try:
            cols = json.loads(json_path.read_text())
        except Exception:
            cols = {}
    if dataset_name not in cols:
        entry = {
            "categorical_columns": ["y"],
            "mixed_columns": {},
            "integer_columns": [],
            "problem_type": {"Classification": "y"},
        }
        if include_general:
            entry["general_columns"] = []
        cols[dataset_name] = entry
        json_path.write_text(json.dumps(cols, indent=4))


def write_toml(path: Path, config: dict):
    try:
        import tomli_w
        with open(path, "wb") as f:
            tomli_w.dump(config, f)
    except ImportError:
        # Fallback to simple TOML serializer if tomli_w missing
        lines = []
        for k, v in config.items():
            if isinstance(v, dict):
                lines.append(f"[{k}]")
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        lines.append(f"[{k}.{sub_k}]")
                        for s_k, s_v in sub_v.items():
                            lines.append(f"{s_k} = {json.dumps(s_v)}")
                    else:
                        lines.append(f"{sub_k} = {json.dumps(sub_v)}")
            else:
                lines.append(f"{k} = {json.dumps(v)}")
        path.write_text("\n".join(lines))


def main():
    try:
        _main()
    except Exception as e:  # mai SUCCESS dopo un errore: JSON FAILED sull'ultima riga e codice di uscita 1
        print(json.dumps({"status": "FAILED", "error_message": f"{type(e).__name__}: {e}"[:1000]}))
        sys.exit(1)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["ddpm", "ctabgan", "ctabgan-plus", "tvae"])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_name = csv_path.stem

    df = pd.read_csv(csv_path)
    n_synth = len(df) if args.n_samples <= 0 else args.n_samples

    data_dir = TAB_DDPM_DIR / "data_imbalanced" / ds_name
    exp_dir = TAB_DDPM_DIR / "exp_imbalanced" / ds_name / args.algo
    exp_dir.mkdir(parents=True, exist_ok=True)

    meta = prepare_dataset_for_ddpm(df, data_dir, drop_constant=args.algo in ("ctabgan", "ctabgan-plus"))
    n_feat = meta["n_features"]
    n_cls = meta["n_classes"]
    n_train = meta["train_size"]
    # CTAB-GAN(+): il GMM per colonna ha n_clusters=10 componenti e richiede almeno altrettante righe di train
    # (60% del CSV in ingresso). Il DataTransformer riduce n_clusters a n_train; qui si annota l'adattamento.
    cluster_note = ""
    if args.algo in ("ctabgan", "ctabgan-plus") and n_train < 10:
        cluster_note = f"n_clusters ridotto da 10 a {n_train} (righe di train insufficienti per il GMM)"

    # Register in columns.json for CTAB-GAN
    ensure_columns_entry(TAB_DDPM_DIR / "CTAB-GAN" / "columns.json", ds_name, include_general=False)
    ensure_columns_entry(TAB_DDPM_DIR / "CTAB-GAN-Plus" / "columns.json", ds_name, include_general=True)

    # Config construction
    if args.algo == "ddpm":
        cfg = {
            "seed": 0,
            "parent_dir": str(exp_dir),
            "real_data_path": str(data_dir),
            "model_type": "mlp",
            "num_numerical_features": n_feat,
            "device": args.device,
            "model_params": {
                "num_classes": n_cls,
                "is_y_cond": True,
                "rtdl_params": {"d_layers": [128, 128], "dropout": 0.0},
            },
            "diffusion_params": {
                "num_timesteps": 100,
                "gaussian_loss_type": "mse",
                "scheduler": "cosine",
            },
            "train": {
                "main": {
                    "steps": args.steps,
                    "lr": 0.001,
                    "weight_decay": 1e-5,
                    "batch_size": min(n_train, 256),
                },
                "T": {
                    "seed": 0,
                    "normalization": "quantile",
                    "num_nan_policy": "__none__",
                    "cat_nan_policy": "__none__",
                    "cat_min_frequency": "__none__",
                    "cat_encoding": "__none__",
                    "y_policy": "default",
                },
            },
            "sample": {
                "num_samples": n_synth,
                "batch_size": 1000,
                "seed": 0,
            },
        }
        cfg_path = exp_dir / "config.toml"
        write_toml(cfg_path, cfg)
        cmd = [sys.executable, str(TAB_DDPM_DIR / "scripts" / "pipeline.py"), "--config", str(cfg_path), "--train", "--sample"]
        extra_paths = [str(TAB_DDPM_DIR), str(TAB_DDPM_DIR / "scripts")]
    else:
        cfg = {
            "parent_dir": str(exp_dir),
            "real_data_path": str(data_dir),
            "device": args.device,
            "seed": 0,
            "train_params": {"batch_size": min(n_train, 256)},
            "sample": {"num_samples": n_synth, "seed": 0},
        }
        cfg_path = exp_dir / "config.toml"
        write_toml(cfg_path, cfg)
        if args.algo == "ctabgan":
            script = TAB_DDPM_DIR / "CTAB-GAN" / "pipeline_ctabgan.py"
            extra_paths = [str(TAB_DDPM_DIR), str(TAB_DDPM_DIR / "CTAB-GAN")]
        elif args.algo == "ctabgan-plus":
            script = TAB_DDPM_DIR / "CTAB-GAN-Plus" / "pipeline_ctabganp.py"
            extra_paths = [str(TAB_DDPM_DIR), str(TAB_DDPM_DIR / "CTAB-GAN-Plus")]
        else: # tvae
            script = TAB_DDPM_DIR / "CTGAN" / "pipeline_tvae.py"
            extra_paths = [str(TAB_DDPM_DIR), str(TAB_DDPM_DIR / "CTGAN")]
        cmd = [sys.executable, str(script), "--config", str(cfg_path), "--train", "--sample"]

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing_pp] if existing_pp else []))

    t_start_wall = time.time()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, cwd=str(TAB_DDPM_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    tot_time = time.perf_counter() - t0

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"tab-ddpm {args.algo} failed: {err_msg}")

    # Export generated data to target CSV
    y_path = exp_dir / "y_train.npy"
    xn_path = exp_dir / "X_num_train.npy"

    # file piu' vecchio dell'avvio = residuo di una esecuzione precedente: non e' un risultato di questa
    status, method, note = "SUCCESS", args.algo, ""
    if xn_path.exists() and xn_path.stat().st_mtime < t_start_wall - 1:
        raise RuntimeError(f"{xn_path} non e' stato aggiornato da questa esecuzione (file obsoleto)")
    if xn_path.exists():
        X_num = np.load(xn_path, allow_pickle=True).astype(float)
        synth_df = pd.DataFrame(X_num, columns=meta["feature_names"])
        if meta.get("constant_cols"):  # colonne costanti tolte prima dell'addestramento: valore originale
            for c, v in meta["constant_cols"].items():
                synth_df[c] = v
            synth_df = synth_df[meta["all_feature_names"]]
            note = "colonne costanti nel train reinserite col loro valore: " + ", ".join(meta["constant_cols"])
        if len(synth_df) == 0:
            raise RuntimeError(f"{args.algo}: nessuna riga generata (campionamento con rifiuto: tutte le righe "
                               f"fuori dall'intervallo [min, max] del train)")
        if y_path.exists():
            y_arr = np.load(y_path, allow_pickle=True)
            # Map back to original label
            unique_labels = meta["unique_labels"]
            mapped_y = [unique_labels[min(int(v), len(unique_labels) - 1)] for v in y_arr]
            synth_df[meta["target_name"]] = mapped_y
        else:
            synth_df[meta["target_name"]] = df[meta["target_name"]].iloc[0]
            status = "FALLBACK"
            method = f"{args.algo}+label-copied-from-input"
            note = "y_train.npy assente: etichetta copiata dal dataset di input (non generata)"
        synth_df.to_csv(out_path, index=False)
    else:
        # If output file was not produced, raise error
        raise FileNotFoundError(f"Expected generated numpy file at {xn_path}")

    if cluster_note:
        note = (note + "; " if note else "") + cluster_note

    # Il sottoprocesso non separa train e sampling: non si inventa una ripartizione.
    metrics = {
        "status": status,
        "method_used": method,
        "error_message": note,
        "train_time_s": None,
        "gen_time_s": None,
        "total_time_s": round(tot_time, 4),
        "train_peak_mem_mb": 0.0,
        "gen_peak_mem_mb": 0.0,
        "rss_delta_mb": 0.0,
    }
    print(json.dumps(metrics))


if __name__ == "__main__":
    import seed_util; seed_util.apply_env_seed()
    main()
