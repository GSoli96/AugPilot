#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
for _attr in ("long", "ulong"):
    if not hasattr(np, _attr):
        setattr(np, _attr, int)

import pandas as pd
from sklearn.neighbors import NearestNeighbors
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from config import (
    BASE_DIR,
    ALGORITHMS_DIR,
    ENV_PYTHONS,
    PREFAIR_SENSITIVE,
)
from metrics_tracker import PhaseTracker, SubprocessMonitor, get_current_rss_mb

# ---------------------------------------------------------------------------
# Stato reale di ogni esecuzione (nessun fallback silenzioso)
#   status        SUCCESS  = eseguito il metodo richiesto
#                 FALLBACK = prodotto un risultato, ma con un metodo diverso/semplificato o completato
#                            con dati duplicati (vedi method_used ed error_message)
#                 FAILED   = nessun risultato valido (error_message spiega perche')
#   method_used   cosa e' stato davvero eseguito
# I ripieghi "il metodo reale e' fallito, uso un altro metodo" avvengono SOLO con allow_fallback=True
# (flag --allow-fallback di main.py); altrimenti l'esecuzione e' FAILED.
# ---------------------------------------------------------------------------
NAN = float("nan")


def _num(v, nd=4):
    """round() tollerante: None/non numerico -> NaN (mai un valore inventato)."""
    try:
        return round(float(v), nd) if v is not None else NAN
    except (TypeError, ValueError):
        return NAN


def _stderr_summary(stderr: bytes, tail: int = 1500) -> str:
    """Riga d'errore più significativa dello stderr (l'ultima con Error/Exception/Traceback, non un warning)
    più la coda dello stderr: prima si teneva solo la coda di 500 caratteri, spesso un warning."""
    txt = stderr.decode(errors="replace")
    lines = [l for l in txt.strip().splitlines() if l.strip()]
    key = next((l for l in reversed(lines) if any(w in l for w in ("Error", "Exception", "Traceback", "error:"))
                and "Warning" not in l), lines[-1] if lines else "")
    return f"{key.strip()[:400]}\n--- stderr (fine) ---\n{txt[-tail:]}" if txt.strip() else ""


def _run_worker(cmd, csv_path: Path, output_path: Path, label: str, default_method: str) -> Dict[str, Any]:
    """Lancia un worker in sottoprocesso e ne restituisce il record, con lo stato che il worker dichiara."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(BASE_DIR))
    monitor = SubprocessMonitor(proc)
    stdout, stderr = proc.communicate()
    t_tot = time.perf_counter() - t0
    peak_mem = monitor.stop()

    data: Dict[str, Any] = {}
    out_lines = [l for l in stdout.decode(errors="replace").strip().splitlines() if l.strip()]
    if out_lines:
        try:
            data = json.loads(out_lines[-1])
        except Exception:
            data = {}

    if proc.returncode != 0:
        msg = data.get("error_message") or ""
        raise RuntimeError(f"{label} failed with code {proc.returncode}: {msg}\n{_stderr_summary(stderr)}".strip())
    if not data:
        raise RuntimeError(f"{label}: l'ultima riga di stdout del worker non e' JSON, stato non verificabile: "
                           f"{(out_lines[-1] if out_lines else '')[:200]!r}")

    status = data.get("status", "SUCCESS")
    if status == "FAILED":
        raise RuntimeError(f"{label}: {data.get('error_message', 'worker FAILED')}\n{_stderr_summary(stderr)}".strip())
    if not output_path.exists():
        raise RuntimeError(f"{label}: il worker ha dichiarato {status} ma il file {output_path} non esiste")

    df_out = pd.read_csv(output_path)
    df_in = pd.read_csv(csv_path)
    if len(df_out) == 0:
        raise RuntimeError(f"{label}: output vuoto")

    return {
        "status": status,
        "method_used": data.get("method_used", default_method),
        "total_time_s": round(t_tot, 4),
        # NaN se il worker non misura le fasi (nessuna ripartizione inventata)
        "train_time_s": _num(data.get("train_time_s")),
        "gen_time_s": _num(data.get("gen_time_s")),
        "train_peak_mem_mb": _num(data.get("train_peak_mem_mb", peak_mem), 2),
        "gen_peak_mem_mb": _num(data.get("gen_peak_mem_mb", peak_mem), 2),
        "total_mem_peak_mb": round(max(peak_mem, data.get("train_peak_mem_mb") or 0.0), 2),
        "rss_delta_mb": _num(data.get("rss_delta_mb"), 2),
        "train_gpu_peak_mb": NAN,   # non misurato per i sottoprocessi
        "gen_gpu_peak_mb": NAN,
        "n_orig": len(df_in),
        "n_augmented": len(df_out),
        "error_message": data.get("error_message", ""),
    }


# ---------------------------------------------------------------------------
# Device resolver helper
# ---------------------------------------------------------------------------
def resolve_device(use_gpu: bool, requested_device: str = "auto") -> str:
    """Resolve compute device based on GPU flag and system capability."""
    if not use_gpu:
        return "cpu"
    if requested_device in ("cuda", "mps", "cpu"):
        return requested_device

    # auto detection
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# 1. SMOTE Runner
# ---------------------------------------------------------------------------
def run_smote(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
              n_samples: Optional[int] = None, epochs: int = 100, **kwargs) -> Dict[str, Any]:
    """Execute SMOTE data augmentation."""
    status_, method_, note_ = "SUCCESS", "imblearn.SMOTE", ""
    df = pd.read_csv(csv_path)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    n_gen = len(df) if n_samples is None else n_samples

    unique_classes = np.unique(y)
    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    # Phase 1: Training / Fitting
    with PhaseTracker("train") as pt_train:
        k = max(1, min(5, len(df) - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, indices = nn.kneighbors(X)

    # Phase 2: Generation / Sampling
    with PhaseTracker("gen") as pt_gen:
        rng = np.random.default_rng(42)
        if len(unique_classes) > 1:
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=42, k_neighbors=min(5, min(pd.Series(y).value_counts()) - 1))
                X_res, y_res = sm.fit_resample(X, y)
                synth_df = pd.DataFrame(X_res, columns=df.columns[:-1])
                synth_df[target_col] = y_res
            except Exception as e:
                # imblearn SMOTE e' fallito: l'interpolazione KNN NON e' il metodo richiesto
                if not kwargs.get("allow_fallback"):
                    raise RuntimeError(f"imblearn SMOTE fallito (nessun fallback senza --allow-fallback): "
                                       f"{type(e).__name__}: {e}")
                status_, method_ = "FALLBACK", "knn-interpolation (NON imblearn.SMOTE)"
                note_ = f"imblearn SMOTE fallito: {type(e).__name__}: {e}"[:300]
                synth_X, synth_y = [], []
                for _ in range(n_gen):
                    i = rng.integers(0, len(X))
                    neighbor = rng.choice(indices[i, 1:])
                    gap = rng.uniform(0, 1)
                    synth_X.append(X[i] + gap * (X[neighbor] - X[i]))
                    synth_y.append(y[i])
                synth_df = pd.DataFrame(synth_X, columns=df.columns[:-1])
                synth_df[target_col] = synth_y
        else:
            # Single class (e.g. _min minority datasets): imblearn non e' applicabile; si esegue
            # l'interpolazione SMOTE nativa tra vicini della stessa classe (definizione di SMOTE)
            method_ = "smote-knn-interpolation (single-class, nativo)"
            synth_X, synth_y = [], []
            for _ in range(n_gen):
                i = rng.integers(0, len(X))
                neighbor = rng.choice(indices[i, 1:])
                gap = rng.uniform(0, 1)
                synth_X.append(X[i] + gap * (X[neighbor] - X[i]))
                synth_y.append(y[i])
            synth_df = pd.DataFrame(synth_X, columns=df.columns[:-1])
            synth_df[target_col] = synth_y

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(synth_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# 2. Fair-SMOTE Runner
# ---------------------------------------------------------------------------
def run_fair_smote(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
                   n_samples: Optional[int] = None, epochs: int = 100, **kwargs) -> Dict[str, Any]:
    """Execute Fair-SMOTE data augmentation."""
    status_, method_, note_ = "SUCCESS", "fair-smote", ""
    df = pd.read_csv(csv_path)
    target_col = df.columns[-1]
    n_gen = len(df) if n_samples is None else n_samples

    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    fair_smote_dir = (ALGORITHMS_DIR / "Fair-SMOTE-master") if (ALGORITHMS_DIR / "Fair-SMOTE-master").exists() else (BASE_DIR / "Fair-SMOTE-master")
    if str(fair_smote_dir) not in sys.path:
        sys.path.insert(0, str(fair_smote_dir))

    from SMOTE import smote as FairSmoteClass

    # Training phase
    with PhaseTracker("train") as pt_train:
        fs_model = FairSmoteClass(df, auto=True)

    # Generation phase
    with PhaseTracker("gen") as pt_gen:
        # Check if single class or multi class
        if len(df[target_col].unique()) > 1:
            augmented_df = fs_model.run()
        else:
            # Dataset a classe singola: Fair-SMOTE (bilanciamento di classe e attributo sensibile) non
            # e' applicabile; si genera per interpolazione KNN SENZA logica di fairness.
            status_, method_ = "FALLBACK", "knn-interpolation senza fairness (NON Fair-SMOTE; classe singola)"
            note_ = "dataset con una sola classe: Fair-SMOTE non applicabile, usata interpolazione KNN"
            k = max(1, min(5, len(df) - 1))
            nn = NearestNeighbors(n_neighbors=k + 1).fit(df.iloc[:, :-1].values)
            distances, indices = nn.kneighbors(df.iloc[:, :-1].values)
            rng = np.random.default_rng(42)
            X = df.iloc[:, :-1].values
            y = df[target_col].values
            rows = []
            for _ in range(n_gen):
                idx = rng.integers(0, len(X))
                ngbr = rng.choice(indices[idx, 1:])
                gap = rng.uniform(0.1, 0.9)
                new_x = list(X[idx] + gap * (X[ngbr] - X[idx]))
                new_x.append(y[idx])
                rows.append(new_x)
            augmented_df = pd.DataFrame(rows, columns=df.columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(augmented_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# 3. SMOTE-CDNN Runner
# ---------------------------------------------------------------------------
def run_smote_cdnn(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
                   n_samples: Optional[int] = None, epochs: int = 100, **kwargs) -> Dict[str, Any]:
    """Execute SMOTE-CDNN data augmentation."""
    status_, method_, note_ = "SUCCESS", "smote-cdnn", ""
    df = pd.read_csv(csv_path)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    n_gen = len(df) if n_samples is None else n_samples

    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    cdnn_dir = (ALGORITHMS_DIR / "SMOTE-CDNN-main") if (ALGORITHMS_DIR / "SMOTE-CDNN-main").exists() else (BASE_DIR / "SMOTE-CDNN-main")
    if str(cdnn_dir) not in sys.path:
        sys.path.insert(0, str(cdnn_dir))

    from SMOTECDNN import SMOTECDNN, EditedCDNN

    with PhaseTracker("train") as pt_train:
        sc = SMOTECDNN(sampling_strategy="auto", random_state=42)

    with PhaseTracker("gen") as pt_gen:
        if len(np.unique(y)) > 1:
            # Con imbalanced-learn >= 0.12 BaseSampler.fit_resample() pretende _parameter_constraints, che il
            # repository (scritto per imblearn 0.8) non ha: si chiamano i due passi dell'algoritmo originale
            # (SMOTE, poi EditedCDNN._fit_resample) senza modificare il codice di terzi.
            classes_, y_int = np.unique(y, return_inverse=True)
            from imblearn.over_sampling import SMOTE as _ImbSMOTE
            # (_validate_estimator() passa n_jobs a SMOTE, argomento rimosso dalle versioni recenti)
            _k = max(1, min(5, int(np.bincount(y_int).min()) - 1))
            X_sm, y_sm = _ImbSMOTE(sampling_strategy="auto", random_state=42, k_neighbors=_k).fit_resample(X, y_int)
            X_res, y_res = EditedCDNN(sampling_strategy="all")._fit_resample(np.asarray(X_sm), np.asarray(y_sm).astype(int))
            augmented_df = pd.DataFrame(X_res, columns=df.columns[:-1])
            augmented_df[target_col] = classes_[np.asarray(y_res).astype(int)]
        else:
            # Dataset a classe singola: SMOTE-CDNN non e' applicabile. Si interpola come SMOTE e il
            # filtro EditedCDNN NON viene applicato (il commento originale lo prometteva ma non c'era).
            status_, method_ = "FALLBACK", "smote-interpolation senza filtro CDNN (NON SMOTE-CDNN; classe singola)"
            note_ = "dataset con una sola classe: usata interpolazione SMOTE senza EditedCDNN"
            k = max(1, min(5, len(df) - 1))
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
            _, indices = nn.kneighbors(X)
            rng = np.random.default_rng(42)
            synth_X = []
            for _ in range(n_gen):
                i = rng.integers(0, len(X))
                neighbor = rng.choice(indices[i, 1:])
                synth_X.append(X[i] + rng.uniform(0, 1) * (X[neighbor] - X[i]))
            augmented_df = pd.DataFrame(synth_X, columns=df.columns[:-1])
            augmented_df[target_col] = y[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(augmented_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# 4. CasTGAN Runner
# ---------------------------------------------------------------------------
def run_castgan(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
                n_samples: Optional[int] = None, epochs: int = 50, **kwargs) -> Dict[str, Any]:
    """Execute CasTGAN data augmentation."""
    status_, method_, note_ = "SUCCESS", "castgan", ""
    df = pd.read_csv(csv_path)
    dataset_name = csv_path.stem
    df.name = dataset_name
    n_gen = len(df) if n_samples is None else n_samples

    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    castgan_dir = (ALGORITHMS_DIR / "CasTGAN-main") if (ALGORITHMS_DIR / "CasTGAN-main").exists() else (BASE_DIR / "CasTGAN-main")
    if str(castgan_dir) not in sys.path:
        sys.path.insert(0, str(castgan_dir))

    from Model.CasTGAN import CasTGAN

    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    batch_size = min(64, max(8, len(df) // 2))
    batch_size += batch_size % 2  # CasTGAN vuole un batch pari (AssertionError altrimenti)

    with PhaseTracker("train") as pt_train:
        model = CasTGAN(
            batch_size=batch_size,
            cuda=use_gpu and torch.cuda.is_available() if "torch" in globals() or hasattr(__import__("torch"), "cuda") else False,
            vgm_mode=False,
        )
        model.fit(df, categorical_columns=cat_cols, epochs=epochs)

    with PhaseTracker("gen") as pt_gen:
        fake_df = model.sample(num_samples=n_gen)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fake_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(fake_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# 5. TabFairGAN Runner
# ---------------------------------------------------------------------------
def run_tabfairgan(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
                   n_samples: Optional[int] = None, epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Execute TabFairGAN data augmentation."""
    status_, method_, note_ = "SUCCESS", "tabfairgan", ""
    df = pd.read_csv(csv_path)
    n_gen = len(df) if n_samples is None else n_samples

    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    tfg_dir = (ALGORITHMS_DIR / "TabFairGAN-main" / "src") if (ALGORITHMS_DIR / "TabFairGAN-main").exists() else (BASE_DIR / "TabFairGAN-main" / "src")
    if str(tfg_dir) not in sys.path:
        sys.path.insert(0, str(tfg_dir))

    from tabfairgan.main import TFG

    dev = resolve_device(use_gpu, device)
    batch_size = min(32, max(4, len(df) // 2))

    with PhaseTracker("train") as pt_train:
        model = TFG(
            df=df,
            epochs=epochs,
            batch_size=batch_size,
            device=dev,
            fairness_config=None,
        )
        model.train()

    with PhaseTracker("gen") as pt_gen:
        fake_df = model.generate_fake_df(num_rows=n_gen)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fake_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(fake_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# 6. GOGGLE Runner (using isolated worker if needed)
# ---------------------------------------------------------------------------
def run_goggle(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
               n_samples: Optional[int] = None, epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Execute GOGGLE data augmentation via worker script."""
    worker_script = BASE_DIR / "workers" / "run_goggle_worker.py"
    py_exec = ENV_PYTHONS.get("goggle_env", sys.executable)
    n_gen = n_samples if n_samples is not None else -1
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    dev = "cuda" if (use_gpu and has_cuda) else "cpu"

    cmd = [
        py_exec,
        str(worker_script),
        "--csv", str(csv_path),
        "--out", str(output_path),
        "--device", dev,
        "--n_samples", str(n_gen),
        "--epochs", str(epochs),
    ]

    return _run_worker(cmd, csv_path, output_path, "GOGGLE", "goggle")


# ---------------------------------------------------------------------------
# 7. tab-ddpm family Runner (DDPM, CTAB-GAN, CTAB-GAN-Plus, TVAE)
# ---------------------------------------------------------------------------
def run_tab_ddpm_family(algo: str, csv_path: Path, output_path: Path, use_gpu: bool = False,
                        device: str = "cpu", n_samples: Optional[int] = None,
                        epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Execute one of the algorithms from tab-ddpm-main (ddpm, ctabgan, ctabgan-plus, tvae)."""
    worker_script = BASE_DIR / "workers" / "run_tab_ddpm_worker.py"
    py_exec = ENV_PYTHONS.get("clean_env", sys.executable)
    n_gen = n_samples if n_samples is not None else -1
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    dev = "cuda" if (use_gpu and has_cuda) else "cpu"

    cmd = [
        py_exec,
        str(worker_script),
        "--algo", algo,
        "--csv", str(csv_path),
        "--out", str(output_path),
        "--device", dev,
        "--n_samples", str(n_gen),
        "--steps", str(epochs),
    ]

    return _run_worker(cmd, csv_path, output_path, algo, algo)


# ---------------------------------------------------------------------------
# 8. Extra algorithms from NonUsati/
def run_algo_worker(algo: str, csv_path: Path, output_path: Path, use_gpu: bool = False,
                    device: str = "cpu", n_samples: Optional[int] = None,
                    epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Execute an algorithm worker from the Algorithms directory."""
    worker_script = BASE_DIR / "workers" / "run_algo_worker.py"
    # Choose best python interpreter for algorithms with dedicated conda environments
    if algo == "octree":
        py_exec = ENV_PYTHONS.get("octree", sys.executable)
    elif algo == "harmonic":
        py_exec = ENV_PYTHONS.get("harmonic", sys.executable)
    else:
        py_exec = ENV_PYTHONS.get("tabular", sys.executable)
    n_gen = n_samples if n_samples is not None else -1
    dev = "cuda" if use_gpu else "cpu"

    cmd = [
        py_exec,
        str(worker_script),
        "--algo", algo,
        "--csv", str(csv_path),
        "--out", str(output_path),
        "--device", dev,
        "--n_samples", str(n_gen),
        "--epochs", str(epochs),
    ]

    if kwargs.get("allow_fallback"):
        cmd.append("--allow_fallback")
    if kwargs.get("dc_file"):
        cmd.extend(["--dc_file", str(kwargs["dc_file"])])
    if kwargs.get("kamino_weights"):
        cmd.extend(["--kamino_weights", str(kwargs["kamino_weights"])])

    return _run_worker(cmd, csv_path, output_path, algo, algo)


run_nonusati_algo = run_algo_worker


KAMINO_METHOD_USED = "kamino-reimplementation (Python, non il repository originale)"


def run_kamino(*args, **kwargs) -> Dict[str, Any]:
    """Kamino tramite la reimplementazione Python (workers/kamino_engine.py), NON il codice di
    Algorithms/kamino-master. Anche se il worker termina senza errori, il risultato non e' quello del
    metodo originale: lo stato e' sempre FALLBACK e method_used lo dichiara."""
    res = run_algo_worker("kamino", *args, **kwargs)
    if res.get("status") == "SUCCESS":
        res["status"] = "FALLBACK"
    res["method_used"] = KAMINO_METHOD_USED
    return res


# ---------------------------------------------------------------------------
# 8. CuTS Runner (Customizable Tabular Synthesizer)
# ---------------------------------------------------------------------------
def run_cuts(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
             n_samples: Optional[int] = None, epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Execute CuTS (Customizable Tabular Synthesizer) data augmentation."""
    status_, method_, note_ = "SUCCESS", "cuts", ""
    df = pd.read_csv(csv_path)
    n_gen = len(df) if n_samples is None else n_samples

    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    cuts_dir = (ALGORITHMS_DIR / "cuts-main") if (ALGORITHMS_DIR / "cuts-main").exists() else (BASE_DIR / "cuts-main")
    if str(cuts_dir) not in sys.path:
        sys.path.insert(0, str(cuts_dir))

    from denoiser import Denoiser
    dev = resolve_device(use_gpu, device)
    if dev == "mps":
        dev = "cpu"

    # Preprocess dataframe into one-hot discrete format
    col_cats = {}
    total_dim = 0
    one_hot_index_map = {}
    encoded_cols = []

    for c in df.columns:
        s = df[c]
        if s.dtype == "object" or len(s.unique()) <= 20 or not np.issubdtype(s.dtype, np.number):
            cats = sorted(s.unique())
            col_cats[c] = ("cat", cats)
            mapping = {v: i for i, v in enumerate(cats)}
            idx = s.map(mapping).fillna(0).astype(int).values
            n_c = len(cats)
        else:
            n_bins = min(20, max(5, len(s.unique()) // 10))
            bins = np.quantile(s.dropna(), np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)
            if len(bins) <= 1:
                bins = np.array([float(s.min()) - 1.0, float(s.max()) + 1.0])
            col_cats[c] = ("num", bins)
            idx = np.digitize(s.values, bins[1:-1])
            n_c = len(bins) - 1

        n_c = max(1, n_c)
        one_hot = np.zeros((len(df), n_c), dtype=np.float32)
        one_hot[np.arange(len(df)), np.clip(idx, 0, n_c - 1)] = 1.0
        one_hot_index_map[c] = list(range(total_dim, total_dim + n_c))
        total_dim += n_c
        encoded_cols.append(one_hot)

    X_one_hot = np.concatenate(encoded_cols, axis=1)
    X_torch = torch.tensor(X_one_hot, dtype=torch.float32, device=dev)

    with PhaseTracker("train") as pt_train:
        denoiser = Denoiser(
            num_features=total_dim,
            one_hot_index_map=one_hot_index_map,
            architecture_type="residual",
            head="softmax",
            device=dev,
        )
        batch_size = min(512, max(32, len(df)))
        denoiser.fit(
            algorithm="input",
            full_one_hot_dataset=X_torch,
            workload=1 if len(df.columns) > 30 else 2,
            n_epochs=epochs,
            batch_size=batch_size,
            verbose=False,
        )

    with PhaseTracker("gen") as pt_gen:
        synth_torch = denoiser.generate_data(size=n_gen, sample=True)
        synth_data = synth_torch.detach().cpu().numpy()

        synth_dict = {}
        for c in df.columns:
            kind, info = col_cats[c]
            col_indices = one_hot_index_map[c]
            col_one_hot = synth_data[:, col_indices]
            argmax_idx = np.argmax(col_one_hot, axis=1)
            if kind == "cat":
                cats = info
                synth_dict[c] = [cats[min(i, len(cats) - 1)] for i in argmax_idx]
            else:
                bins = info
                bin_mins = bins[np.clip(argmax_idx, 0, len(bins) - 2)]
                bin_maxs = bins[np.clip(argmax_idx + 1, 1, len(bins) - 1)]
                u = np.random.uniform(0, 1, size=len(argmax_idx))
                synth_dict[c] = bin_mins + u * (bin_maxs - bin_mins)

        augmented_df = pd.DataFrame(synth_dict, columns=df.columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(augmented_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# 9. TabularARGN Runner (Auto-Regressive Generative Network)
# ---------------------------------------------------------------------------
def run_tabular_argn(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
                     n_samples: Optional[int] = None, epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Execute TabularARGN data augmentation."""
    status_, method_, note_ = "SUCCESS", "mostlyai-engine (TabularARGN)", ""
    df = pd.read_csv(csv_path)
    n_gen = len(df) if n_samples is None else n_samples

    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()
    dev = resolve_device(use_gpu, device)

    mostlyai_available = False
    try:
        from mostlyai.engine import split, encode, analyze, train as mostly_train, generate as mostly_generate
        mostlyai_available = True
    except Exception:
        mostlyai_available = False

    # mostlyai-engine non e' in questo interprete: se esiste envs/argn lo si esegue li' (worker dedicato)
    argn_py = ENV_PYTHONS.get("argn", sys.executable)
    if not mostlyai_available and Path(argn_py).resolve() != Path(sys.executable).resolve():
        cmd = [argn_py, str(BASE_DIR / "workers" / "run_argn_worker.py"), "--csv", str(csv_path),
               "--out", str(output_path), "--device", ("cuda" if dev == "cuda" else "cpu"),
               "--n_samples", str(n_gen), "--epochs", str(epochs)]
        try:
            return _run_worker(cmd, csv_path, output_path, "TabularARGN", "mostlyai-engine (TabularARGN)")
        except Exception as e:
            if not kwargs.get("allow_fallback"):
                raise
            note_ = f"worker mostlyai-engine fallito ({type(e).__name__}: {e})"[:400]

    if mostlyai_available:
        import shutil
        ws_dir = BASE_DIR / "tmp" / f"mostly_ws_{csv_path.stem}_{int(time.time() * 1000)}"
        ws_dir.mkdir(parents=True, exist_ok=True)
        try:
            with PhaseTracker("train") as pt_train:
                split(tgt_data=df, workspace_dir=ws_dir)
                analyze(workspace_dir=ws_dir)
                encode(workspace_dir=ws_dir)
                mostly_train(workspace_dir=ws_dir, max_epochs=epochs)

            with PhaseTracker("gen") as pt_gen:
                mostly_generate(workspace_dir=ws_dir, sample_size=n_gen)
                syn_p = ws_dir / "synthetic-data.parquet"
                if not syn_p.exists():
                    syn_p = ws_dir / "synthetic-data.csv"
                augmented_df = pd.read_parquet(syn_p) if syn_p.suffix == ".parquet" else pd.read_csv(syn_p)
        finally:
            shutil.rmtree(ws_dir, ignore_errors=True)
    else:
        # mostlyai-engine non installato: si usa una rete autoregressiva scritta qui, NON TabularARGN
        status_ = "FALLBACK"
        method_ = "torch-autoregressive-reimplementation (NON mostlyai TabularARGN)"
        note_ = (note_ + " | " if note_ else "") + "mostlyai-engine non importabile: usata una reimplementazione semplificata"
        import torch.nn as nn
        import torch.optim as optim

        n_features = len(df.columns)
        vocab_sizes = []
        token_data = []
        col_mappings = []

        for c in df.columns:
            s = df[c]
            if s.dtype == "object" or len(s.unique()) <= 20 or not np.issubdtype(s.dtype, np.number):
                cats = sorted(s.unique())
                m = {v: i for i, v in enumerate(cats)}
                col_mappings.append(("cat", cats, m))
                idx = s.map(m).fillna(0).astype(int).values
                vocab_sizes.append(max(2, len(cats)))
            else:
                n_b = min(20, max(5, len(s.unique()) // 10))
                bins = np.quantile(s.dropna(), np.linspace(0, 1, n_b + 1))
                bins = np.unique(bins)
                if len(bins) <= 1:
                    bins = np.array([float(s.min()) - 1.0, float(s.max()) + 1.0])
                idx = np.digitize(s.values, bins[1:-1])
                col_mappings.append(("num", bins, None))
                vocab_sizes.append(max(2, len(bins) - 1))
            token_data.append(idx)

        tokens = np.stack(token_data, axis=1)
        tokens_torch = torch.tensor(tokens, dtype=torch.long, device=dev)

        embed_dim = 32
        embeddings = nn.ModuleList([nn.Embedding(v, embed_dim) for v in vocab_sizes]).to(dev)
        heads = nn.ModuleList([
            nn.Linear(embed_dim * i if i > 0 else embed_dim, vocab_sizes[i])
            for i in range(n_features)
        ]).to(dev)
        start_token = nn.Parameter(torch.randn(1, embed_dim, device=dev))

        optimizer = optim.Adam(list(embeddings.parameters()) + list(heads.parameters()) + [start_token], lr=0.005)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = min(256, max(16, len(df)))
        n_batches = int(np.ceil(len(df) / batch_size))

        with PhaseTracker("train") as pt_train:
            for ep in range(epochs):
                perm = torch.randperm(len(df), device=dev)
                for b in range(n_batches):
                    b_idx = perm[b * batch_size : (b + 1) * batch_size]
                    b_tok = tokens_torch[b_idx]

                    b_embeds = [embeddings[i](b_tok[:, i]) for i in range(n_features)]
                    total_loss = 0.0
                    for i in range(n_features):
                        if i == 0:
                            inp = start_token.expand(len(b_idx), -1)
                        else:
                            inp = torch.cat(b_embeds[:i], dim=-1)
                        logits = heads[i](inp)
                        total_loss = total_loss + loss_fn(logits, b_tok[:, i])

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

        with PhaseTracker("gen") as pt_gen:
            with torch.no_grad():
                gen_tokens = []
                gen_embeds = []
                for i in range(n_features):
                    if i == 0:
                        inp = start_token.expand(n_gen, -1)
                    else:
                        inp = torch.cat(gen_embeds, dim=-1)
                    logits = heads[i](inp)
                    probs = torch.softmax(logits, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    gen_tokens.append(sampled.cpu().numpy())
                    gen_embeds.append(embeddings[i](sampled))

            gen_tokens = np.stack(gen_tokens, axis=1)
            synth_dict = {}
            for i, c in enumerate(df.columns):
                kind, info, _ = col_mappings[i]
                if kind == "cat":
                    cats = info
                    synth_dict[c] = [cats[min(t, len(cats) - 1)] for t in gen_tokens[:, i]]
                else:
                    bins = info
                    t_idx = gen_tokens[:, i]
                    bin_mins = bins[np.clip(t_idx, 0, len(bins) - 2)]
                    bin_maxs = bins[np.clip(t_idx + 1, 1, len(bins) - 1)]
                    u = np.random.uniform(0, 1, size=len(t_idx))
                    synth_dict[c] = bin_mins + u * (bin_maxs - bin_mins)

            augmented_df = pd.DataFrame(synth_dict, columns=df.columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(augmented_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# 10. PreFair Runner (Fair Data Synthesis via Fair Bayesian Network)
# ---------------------------------------------------------------------------
def run_prefair_surrogate(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
                n_samples: Optional[int] = None, epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Execute PreFair data augmentation."""
    status_ = "FALLBACK"
    method_ = "surrogate:conditional-outcome-resampling (NON Algorithms/Prefair-main)"
    note_ = "Algorithms/Prefair-main non viene invocato: campionamento condizionato sugli attributi ammissibili"
    df = pd.read_csv(csv_path)
    n_gen = len(df) if n_samples is None else n_samples

    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    sensitive_col = None
    for c in feature_cols:
        if any(term in c.lower() for term in ["sex", "gender", "race", "age"]):
            sensitive_col = c
            break
    if sensitive_col is None:
        cat_cols = [c for c in feature_cols if df[c].dtype == "object" or len(df[c].unique()) <= 5]
        sensitive_col = cat_cols[0] if cat_cols else feature_cols[0]

    admissible_cols = [c for c in feature_cols if c != sensitive_col]

    with PhaseTracker("train") as pt_train:
        col_stats = {}
        discretized_df = pd.DataFrame(index=df.index)

        for c in df.columns:
            s = df[c]
            if s.dtype == "object" or len(s.unique()) <= 20 or not np.issubdtype(s.dtype, np.number):
                cats = sorted(s.unique())
                m = {v: i for i, v in enumerate(cats)}
                col_stats[c] = ("cat", cats, m)
                discretized_df[c] = s.map(m).fillna(0).astype(int)
            else:
                n_b = min(10, max(3, len(s.unique()) // 15))
                bins = np.quantile(s.dropna(), np.linspace(0, 1, n_b + 1))
                bins = np.unique(bins)
                if len(bins) <= 1:
                    bins = np.array([float(s.min()) - 1.0, float(s.max()) + 1.0])
                col_stats[c] = ("num", bins, None)
                discretized_df[c] = np.digitize(s.values, bins[1:-1])

        # PreFair: Outcome depends conditionally on admissible attributes only
        outcome_cond = discretized_df.groupby(admissible_cols)[target_col].value_counts(normalize=True).to_dict()
        outcome_marginal = discretized_df[target_col].value_counts(normalize=True).to_dict()

    with PhaseTracker("gen") as pt_gen:
        sample_indices = np.random.choice(len(df), size=n_gen, replace=True)
        sampled_rows = discretized_df.iloc[sample_indices].copy().reset_index(drop=True)

        new_outcomes = []
        for _, row in sampled_rows.iterrows():
            key = tuple(row[c] for c in admissible_cols) if len(admissible_cols) > 1 else row[admissible_cols[0]]
            matched = False
            if key in outcome_cond:
                sub_dist = outcome_cond[key]
                if isinstance(sub_dist, dict):
                    vals, probs = list(sub_dist.keys()), list(sub_dist.values())
                    new_outcomes.append(np.random.choice(vals, p=probs))
                    matched = True
            if not matched:
                vals, probs = list(outcome_marginal.keys()), list(outcome_marginal.values())
                new_outcomes.append(np.random.choice(vals, p=probs))

        sampled_rows[target_col] = new_outcomes

        synth_dict = {}
        for c in df.columns:
            kind, info, _ = col_stats[c]
            t_vals = sampled_rows[c].values
            if kind == "cat":
                cats = info
                synth_dict[c] = [cats[min(v, len(cats) - 1)] for v in t_vals]
            else:
                bins = info
                bin_mins = bins[np.clip(t_vals, 0, len(bins) - 2)]
                bin_maxs = bins[np.clip(t_vals + 1, 1, len(bins) - 1)]
                u = np.random.uniform(0, 1, size=len(t_vals))
                synth_dict[c] = bin_mins + u * (bin_maxs - bin_mins)

        augmented_df = pd.DataFrame(synth_dict, columns=df.columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(augmented_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


def _prefair_pick_columns(df: pd.DataFrame, dataset_name: Optional[str] = None):
    """Sceglie attributo sensibile / ammissibili / esito (l'esito e' l'ultima colonna).

    Prima consulta config.PREFAIR_SENSITIVE (chiave = nome dataset senza "_min" e senza estensione); in mancanza
    ricade sull'euristica sul nome. Restituisce (sensibile, ammissibili, esito, origine) con origine
    "config" oppure "euristica"."""
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    if dataset_name:
        base = re.sub(r"\.csv$", "", str(dataset_name), flags=re.IGNORECASE)
        base = re.sub(r"_min$", "", base, flags=re.IGNORECASE)
        configured = PREFAIR_SENSITIVE.get(base) or PREFAIR_SENSITIVE.get(base + "_min")
        if configured:
            if configured not in feature_cols:
                raise ValueError(f"PREFAIR_SENSITIVE['{base}']='{configured}' non e' una colonna feature del dataset "
                                 f"(colonne: {feature_cols})")
            return configured, [c for c in feature_cols if c != configured], target_col, "config"
    sensitive_col = None
    for c in feature_cols:
        if any(term in c.lower() for term in ["sex", "gender", "race", "age"]):
            sensitive_col = c
            break
    if sensitive_col is None:
        cat_cols = [c for c in feature_cols if df[c].dtype == "object" or len(df[c].unique()) <= 5]
        sensitive_col = cat_cols[0] if cat_cols else feature_cols[0]
    return sensitive_col, [c for c in feature_cols if c != sensitive_col], target_col, "euristica"


def _prefair_build_datasynthesizer() -> Path:
    """Crea (una volta) tmp/prefair_ds/DataSynthesizer = pacchetto pip DataSynthesizer + i due file
    modificati da PreFair (Prefair-main/additional_files/.../DataDescriber.py e lib/PrivBayes.py)."""
    import importlib.util
    import shutil
    work = BASE_DIR / "tmp" / "prefair_ds"
    marker = work / "DataSynthesizer" / "lib" / "PrivBayes.py"
    ov = ALGORITHMS_DIR / "Prefair-main" / "additional_files" / "DataSynthesizer_files" / "DataSynthesizer"
    if not ov.exists():
        raise RuntimeError(f"file PreFair non trovati: {ov}")
    if marker.exists() and "fair_greedy_bayes" in marker.read_text(encoding="utf-8", errors="replace"):
        _prefair_patch_pandas2(marker)
        return work
    spec = importlib.util.find_spec("DataSynthesizer")
    if spec is None or not spec.origin:
        raise RuntimeError("pacchetto pip 'DataSynthesizer' non installato (pip install DataSynthesizer)")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copytree(Path(spec.origin).parent, work / "DataSynthesizer")
    shutil.copy2(ov / "DataDescriber.py", work / "DataSynthesizer" / "DataDescriber.py")
    shutil.copy2(ov / "lib" / "PrivBayes.py", work / "DataSynthesizer" / "lib" / "PrivBayes.py")
    _prefair_patch_pandas2(marker)
    return work


def _prefair_patch_pandas2(privbayes: Path) -> None:
    """La PrivBayes di PreFair usa DataFrame.append, rimosso in pandas 2 ('DataFrame' object has no attribute
    'append'): nella copia di lavoro (non nell'originale in Algorithms/) lo sostituisco con pandas.concat.
    (Trovato dalla sessione di wsgrande.)"""
    s = privbayes.read_text(encoding="utf-8", errors="replace")
    old = "full_space = full_space.append(data_frame_append, ignore_index=True)"
    if old not in s:
        return
    s = s.replace(old, "full_space = concat([full_space, data_frame_append], ignore_index=True)")
    if "from pandas import DataFrame, merge, concat" not in s:
        s = s.replace("from pandas import DataFrame, merge", "from pandas import DataFrame, merge, concat", 1)
    privbayes.write_text(s, encoding="utf-8")


def _prefair_cap_pool() -> None:
    """PrivBayes di DataSynthesizer usa Pool() = os.cpu_count() processi: su Windows oltre 61 il pool
    va in errore (WaitForMultipleObjects accetta al massimo 63 handle) e il job resta bloccato.
    Limita i processi a OMP_NUM_THREADS (se impostato) e comunque a 32. (Trovato e provato su server5002.)"""
    import multiprocessing.pool as _mpp
    import DataSynthesizer.lib.PrivBayes as _pb
    if getattr(_pb.Pool, "_augpilot_capped", False):
        return
    try:
        cap = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except ValueError:
        cap = 0
    cap = max(1, min(cap or 32, 32, os.cpu_count() or 1))

    class _CappedPool(_mpp.Pool):
        _augpilot_capped = True

        def __init__(self, processes=None, *a, **k):
            super().__init__(min(processes or cap, cap), *a, **k)

    _pb.Pool = _CappedPool


def run_prefair(csv_path: Path, output_path: Path, use_gpu: bool = False, device: str = "cpu",
                n_samples: Optional[int] = None, epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """PreFair reale (FairPrivBayes di Algorithms/Prefair-main su DataSynthesizer, epsilon=10 come nel notebook).

    L'attributo sensibile viene da config.PREFAIR_SENSITIVE se il dataset e' elencato, altrimenti e' scelto per
    euristica sul nome della colonna (sex/gender/race/age) o, in mancanza, la prima colonna a pochi valori
    (la colonna usata e la sua origine sono registrate in method_used); l'esito e' l'ultima colonna. Se il metodo reale fallisce, il ripiego sul
    surrogato (campionamento condizionato) avviene SOLO con allow_fallback e lo stato e' FALLBACK."""
    try:
        # PrivBayes.calculate_k (versione PreFair) chiama warnings.filterwarnings("error") senza ripristinarlo:
        # senza catch_warnings il filtro resta attivo nel processo del job e ogni warning successivo
        # (sklearn, scipy, ...) diventa un'eccezione negli algoritmi e nelle valutazioni seguenti.
        import warnings
        with warnings.catch_warnings():
            return _run_prefair_real(csv_path, output_path, n_samples=n_samples, **kwargs)
    except Exception as e:
        if not kwargs.get("allow_fallback"):
            raise RuntimeError(f"PreFair fallito (nessun fallback senza --allow-fallback): {type(e).__name__}: {e}")
        res = run_prefair_surrogate(csv_path, output_path, use_gpu=use_gpu, device=device,
                                    n_samples=n_samples, epochs=epochs, **kwargs)
        res["error_message"] = f"PreFair reale fallito ({type(e).__name__}: {e})"[:300] + " | " + res.get("error_message", "")
        return res


def _run_prefair_real(csv_path: Path, output_path: Path, n_samples: Optional[int] = None,
                      epsilon: float = 10.0, **kwargs) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    n_gen = len(df) if n_samples is None else n_samples
    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    work = _prefair_build_datasynthesizer()
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    from DataSynthesizer.DataDescriber import DataDescriber
    from DataSynthesizer.DataGenerator import DataGenerator
    _prefair_cap_pool()

    sensitive_col, admissible, target_col, sens_source = _prefair_pick_columns(df, csv_path.stem)
    sens = {"admissible": admissible, "inadmissible": [sensitive_col], "outcome": [target_col]}
    desc_file = BASE_DIR / "tmp" / f"prefair_desc_{csv_path.stem}_{os.getpid()}.json"

    with PhaseTracker("train") as pt_train:
        describer = DataDescriber(category_threshold=30)
        describer.describe_dataset_in_fair_correlated_attribute_mode(
            dataset_file=str(csv_path), epsilon=epsilon, k=0,
            # Nessuna colonna e' una chiave: senza questo DataSynthesizer marca come candidate key ogni colonna
            # Integer a valori tutti distinti (frequente sulla minoritaria, poche righe) e la esclude dalla rete
            # bayesiana, mentre resta tra le "admissible" passate a fair_greedy_bayes -> KeyError "[...] not in index".
            attribute_to_is_categorical={}, attribute_to_is_candidate_key={c: False for c in df.columns},
            attribute_to_sensitive=sens, seed=42)
        describer.save_dataset_description_to_file(str(desc_file))

    with PhaseTracker("gen") as pt_gen:
        gen = DataGenerator()
        gen.generate_dataset_in_correlated_attribute_mode(n_gen, str(desc_file))
        out_df = gen.synthetic_dataset.copy()
    try:
        desc_file.unlink()
    except OSError:
        pass
    missing = [c for c in df.columns if c not in out_df.columns]
    if missing:
        raise RuntimeError(f"PreFair non ha generato le colonne {missing}")
    out_df = out_df[list(df.columns)]
    if len(out_df) == 0:
        raise RuntimeError("PreFair non ha generato righe")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    res = {
        "status": "SUCCESS",
        "method_used": f"prefair (FairPrivBayes, epsilon={epsilon:g}; sensibile={sensitive_col}, scelta {sens_source})",
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, get_current_rss_mb() - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(out_df),
        "error_message": "",
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


def run_syrfd(csv_path: Path, output_path: Path, use_gpu: bool = False,
              device: str = "cpu", n_samples: Optional[int] = None,
              epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Run SyRFDRes using syrfd_learning_v8_scalable.py."""
    status_, method_, note_ = "SUCCESS", "syrfd", ""
    t_tot_start = time.perf_counter()
    rss_start = get_current_rss_mb()

    df = pd.read_csv(csv_path)
    n_gen = n_samples if n_samples is not None else len(df)

    syrfd_dir = (ALGORITHMS_DIR / "SyRFDRes") if (ALGORITHMS_DIR / "SyRFDRes").exists() else (BASE_DIR / "SyRFDRes")
    if str(syrfd_dir) not in sys.path:
        sys.path.insert(0, str(syrfd_dir))

    from syrfd_learning_v8_scalable import SyRFDLearner

    # Locate RFD file in SyRFDRes/discovered_rfds_processed
    rfd_dir = syrfd_dir / "discovered_rfds_processed"
    rfd_file = None
    if rfd_dir.exists():
        stem = csv_path.stem
        matches = list(rfd_dir.glob(f"*{stem}*.txt"))
        if matches:
            rfd4 = [m for m in matches if "RFD4_" in m.name]
            rfd_file = rfd4[0] if rfd4 else matches[0]

    if rfd_file is None or not rfd_file.exists():
        # Nessun file di RFD scoperte per questo dataset: senza RFD il metodo non e' SyRFDRes.
        if not kwargs.get("allow_fallback"):
            raise RuntimeError(f"SyRFDRes: nessun file RFD per '{csv_path.stem}' in {rfd_dir} "
                               f"(nessun fallback senza --allow-fallback)")
        # file vuoto in tmp/ (non si scrive piu' dentro Algorithms/)
        empty_dir = BASE_DIR / "tmp" / "syrfd_empty_rfd"
        empty_dir.mkdir(parents=True, exist_ok=True)
        rfd_file = empty_dir / f"RFD4_E0.0_{csv_path.stem}.txt"
        rfd_file.write_text("****** DISCOVERED RFDs *******\n")
        status_ = "FALLBACK"
        method_ = "syrfd-senza-RFD (NON SyRFDRes con vincoli)"
        note_ = "nessun file RFD trovato: eseguito con elenco RFD vuoto"

    torch_dev = torch.device(device if use_gpu and torch.cuda.is_available() else "cpu")

    learner = SyRFDLearner(
        data_path=str(csv_path),
        rfd_path=str(rfd_file),
        threshold=4.0,
        latent_dim=64,
        rfd_mode="strict",
        device=torch_dev,
        enable_sklearn_metrics=False,
    )

    with PhaseTracker("train") as pt_train:
        warmup = max(1, int(epochs * 0.20))
        learner.train(
            epochs=epochs,
            warmup_epochs=warmup,
            log_every=max(1, epochs // 5),
            save_checkpoint=False,
        )

    with PhaseTracker("gen") as pt_gen:
        gen_df = learner.generate(num_samples=n_gen, rfd_max_attempts=15)
        if len(gen_df) == 0:
            # mai sostituire con righe del dataset reale e chiamarle sintetiche
            raise RuntimeError("SyRFDRes non ha generato alcuna riga valida")
        if len(gen_df) < n_gen:
            needed = n_gen - len(gen_df)
            extra = gen_df.sample(needed, replace=True, random_state=42)
            gen_df = pd.concat([gen_df, extra], ignore_index=True)
            status_ = "FALLBACK"
            method_ = f"{method_}+resample-padding({needed} righe duplicate)"
            note_ = (note_ + " | " if note_ else "") + f"generate() ha dato {n_gen - needed} righe su {n_gen}: {needed} duplicate"
        elif len(gen_df) > n_gen:
            gen_df = gen_df.iloc[:n_gen].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gen_df.to_csv(output_path, index=False)
    t_tot = time.perf_counter() - t_tot_start
    rss_end = get_current_rss_mb()

    res = {
        "status": status_,
        "method_used": method_,
        "total_time_s": round(t_tot, 4),
        "total_mem_peak_mb": round(max(pt_train.peak_py_mem_mb, pt_gen.peak_py_mem_mb), 2),
        "rss_delta_mb": round(max(0.0, rss_end - rss_start), 2),
        "n_orig": len(df),
        "n_augmented": len(gen_df),
        "error_message": note_,
    }
    res.update(pt_train.to_dict())
    res.update(pt_gen.to_dict())
    return res


# ---------------------------------------------------------------------------
# Dispatcher registry
# ---------------------------------------------------------------------------
ALGORITHM_HANDLERS = {
    "smote": run_smote,
    "fair-smote": run_fair_smote,
    "smote-cdnn": run_smote_cdnn,
    "castgan": run_castgan,
    "tabfairgan": run_tabfairgan,
    "goggle": run_goggle,
    "cuts": run_cuts,
    "tabular-argn": run_tabular_argn,
    "tabularargn": run_tabular_argn,
    "prefair": run_prefair,
    "syrfd": run_syrfd,
    "syrfdres": run_syrfd,
    "ddpm": lambda *a, **k: run_tab_ddpm_family("ddpm", *a, **k),
    "ctabgan": lambda *a, **k: run_tab_ddpm_family("ctabgan", *a, **k),
    "ctabgan-plus": lambda *a, **k: run_tab_ddpm_family("ctabgan-plus", *a, **k),
    "tvae": lambda *a, **k: run_tab_ddpm_family("tvae", *a, **k),
    # All algorithms in Algorithms/
    "tabsyn": lambda *a, **k: run_algo_worker("tabsyn", *a, **k),
    "tabdiff": lambda *a, **k: run_algo_worker("tabdiff", *a, **k),
    "octree": lambda *a, **k: run_algo_worker("octree", *a, **k),
    "harmonic": lambda *a, **k: run_algo_worker("harmonic", *a, **k),
    "synthetic-llm": lambda *a, **k: run_algo_worker("synthetic-llm", *a, **k),
    "synthetic_llm": lambda *a, **k: run_algo_worker("synthetic-llm", *a, **k),
    "kamino": run_kamino,
}


def execute_augmentation(algo: str, csv_path: Path, output_path: Path, use_gpu: bool = False,
                         device: str = "cpu", n_samples: Optional[int] = None,
                         epochs: int = 1000, **kwargs) -> Dict[str, Any]:
    """Safe dispatcher that catches any uncaught exception and returns a structured record."""
    algo_key = algo.lower().strip()
    handler = ALGORITHM_HANDLERS.get(algo_key)
    if not handler and algo_key.startswith("kamino"):
        handler = ALGORITHM_HANDLERS.get("kamino")
    if not handler:
        return {
            "status": "FAILED",
            "total_time_s": 0.0,
            "train_time_s": 0.0,
            "gen_time_s": 0.0,
            "train_peak_mem_mb": 0.0,
            "gen_peak_mem_mb": 0.0,
            "total_mem_peak_mb": 0.0,
            "rss_delta_mb": 0.0,
            "train_gpu_peak_mb": 0.0,
            "gen_gpu_peak_mb": 0.0,
            "n_orig": 0,
            "n_augmented": 0,
            "method_used": "",
            "error_message": f"Unknown algorithm: '{algo}'",
        }

    t0_all = time.perf_counter()
    try:
        return handler(
            csv_path=csv_path,
            output_path=output_path,
            use_gpu=use_gpu,
            device=device,
            n_samples=n_samples,
            epochs=epochs,
            **kwargs
        )
    except Exception as e:
        import traceback
        err_text = f"{type(e).__name__}: {str(e)}"
        return {
            "status": "FAILED",
            # tempo fino al fallimento (prima era 0.0: non si capiva se il metodo fosse mai partito)
            "total_time_s": round(time.perf_counter() - t0_all, 4),
            "error_traceback": traceback.format_exc()[-4000:],
            "train_time_s": 0.0,
            "gen_time_s": 0.0,
            "train_peak_mem_mb": 0.0,
            "gen_peak_mem_mb": 0.0,
            "total_mem_peak_mb": 0.0,
            "rss_delta_mb": 0.0,
            "train_gpu_peak_mb": 0.0,
            "gen_gpu_peak_mb": 0.0,
            "n_orig": 0,
            "n_augmented": 0,
            "method_used": "",
            "error_message": (err_text if len(err_text) <= 700 else err_text[:200] + " [...] " + err_text[-480:]),
        }
