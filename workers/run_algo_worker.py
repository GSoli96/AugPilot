#!/usr/bin/env python3
"""
run_algo_worker.py
=============================================================================
Unified worker script for executing algorithms in Algorithms/:
  - tabsyn
  - tabdiff
  - octree
  - harmonic
  - synthetic-llm
  - kamino

Each algorithm provides:
  1. Train phase (fitting generative model / tree / GMM / diffusion)
  2. Generation phase (sampling n_samples synthetic records)
  3. Memory tracking & wall-clock timing
  4. Automatic export of augmented CSV to --out
=============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

_WORKER_DIR = Path(__file__).resolve().parent
_BASE_DIR = _WORKER_DIR.parent
for _p in (str(_BASE_DIR), str(_WORKER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
for _attr in ("long", "ulong"):
    if not hasattr(np, _attr):
        setattr(np, _attr, int)

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
ALGORITHMS_DIR = BASE_DIR / "Algorithms"


# Politica sui fallback (nessun fallback silenzioso):
#  - ogni worker restituisce nel JSON finale: status (SUCCESS | FALLBACK | FAILED), method_used, error_message;
#  - se il metodo reale fallisce, il ripiego su KNN avviene SOLO con --allow_fallback e lo stato e' FALLBACK;
#    senza il flag il worker termina con errore e NON scrive alcun file di output;
#  - octree, harmonic e synthetic-llm sono qui implementazioni semplificate ("surrogate"), non i repository
#    in Algorithms/: lo stato e' sempre FALLBACK e method_used inizia con "surrogate:".
ALLOW_FALLBACK = False


def _pad_to(synth_df: pd.DataFrame, n_synth: int):
    """Porta synth_df a n_synth righe. Restituisce (df, n_righe_duplicate_per_riempimento)."""
    padded = 0
    if len(synth_df) > n_synth:
        synth_df = synth_df.iloc[:n_synth].reset_index(drop=True)
    elif len(synth_df) < n_synth:
        padded = n_synth - len(synth_df)
        extra = synth_df.sample(padded, replace=True, random_state=42)
        synth_df = pd.concat([synth_df, extra], ignore_index=True)
    return synth_df, padded


# ---------------------------------------------------------------------------
# Helper: Safe fallback generator (KNN interpolation + KDE perturbation)
# Usato SOLO con --allow_fallback: NON e' il metodo richiesto.
# ---------------------------------------------------------------------------
def _safe_knn_kde_fallback(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Generate high-fidelity synthetic samples by interpolating between nearest neighbors."""
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    X_df = df[feature_cols].copy()
    y_vals = df[target_col].values

    # Detect categorical vs numerical
    col_meta = {}
    X_num = pd.DataFrame(index=df.index)
    for c in feature_cols:
        s = X_df[c]
        if s.dtype == "object" or not np.issubdtype(s.dtype, np.number) or s.nunique() <= 5:
            cats = sorted(s.unique())
            m = {val: idx for idx, val in enumerate(cats)}
            col_meta[c] = ("cat", cats, m)
            X_num[c] = s.map(m).fillna(0).astype(float)
        else:
            col_meta[c] = ("num", float(s.min()), float(s.max()))
            X_num[c] = s.astype(float).fillna(s.mean())

    X_mat = X_num.values
    k = max(1, min(5, len(df) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_mat)
    _, indices = nn.kneighbors(X_mat)

    rng = np.random.default_rng(42)
    synth_X = []
    synth_y = []
    for _ in range(n_samples):
        i = rng.integers(0, len(X_mat))
        neighbor = rng.choice(indices[i, 1:])
        gap = rng.uniform(0.0, 1.0)
        synth_pt = X_mat[i] + gap * (X_mat[neighbor] - X_mat[i])
        synth_X.append(synth_pt)
        synth_y.append(y_vals[i])

    synth_X = np.array(synth_X)
    res_dict = {}
    for idx, c in enumerate(feature_cols):
        info = col_meta[c]
        if info[0] == "cat":
            cats = info[1]
            rounded = np.clip(np.round(synth_X[:, idx]).astype(int), 0, len(cats) - 1)
            res_dict[c] = [cats[r] for r in rounded]
        else:
            c_min, c_max = info[1], info[2]
            res_dict[c] = np.clip(synth_X[:, idx], c_min, c_max)

    res_df = pd.DataFrame(res_dict, columns=feature_cols)
    res_df[target_col] = synth_y
    return res_df


# ---------------------------------------------------------------------------
# 1. TabSyn Runner
# ---------------------------------------------------------------------------
def run_tabsyn(csv_path: Path, out_path: Path, device: str, n_samples: int, epochs: int) -> Dict[str, Any]:
    tabsyn_dir = ALGORITHMS_DIR / "tabsyn-main"
    if str(tabsyn_dir) not in sys.path:
        sys.path.insert(0, str(tabsyn_dir))

    df = pd.read_csv(csv_path)
    n_synth = len(df) if n_samples <= 0 else n_samples

    t0_tr = time.perf_counter()
    t_wall_start = time.time()
    tracemalloc.start()
    success = False
    synth_df = None
    err_text = ""

    try:
        from main_custom import prepare_dataset, _make_args
        from types import SimpleNamespace
        import tabsyn.vae.main as vae_main
        import tabsyn.main as diff_main
        import tabsyn.sample as sample_main

        # Setup dataset in tabsyn
        ds_name = f"run_{csv_path.stem}"
        data_target_dir = tabsyn_dir / "data" / ds_name
        data_target_dir.mkdir(parents=True, exist_ok=True)

        orig_cwd = os.getcwd()
        csv_abs = Path(csv_path).resolve()
        try:
            os.chdir(str(tabsyn_dir))
            prepare_dataset(ds_name, str(csv_abs))
            cli = SimpleNamespace(gpu=-1 if device == "cpu" else 0, steps=min(20, epochs), num_samples=n_synth)
            args = _make_args(ds_name, device, cli)
            args.epochs = min(100, max(2, epochs // 10))
            args.max_beta = 1e-2
            args.min_beta = 1e-5
            args.lambd = 0.7

            # Train VAE
            vae_main.main(args)
            # Train Diffusion
            diff_main.main(args)
            # Sample
            # sample.main() genera tante righe quante ne ha il train set (ignora num_samples) e scrive
            # in args.save_path: lo si richiama piu' volte (campioni nuovi ogni volta) fino a n_synth righe.
            parts = []
            n_got, n_calls = 0, 0
            while n_got < n_synth and n_calls < 50:
                gen_rel = os.path.join("synthetic", ds_name, f"tabsyn_part{n_calls}.csv")
                args.save_path = gen_rel
                sample_main.main(args)
                gen_csv = tabsyn_dir / gen_rel
                # un file piu' vecchio dell'avvio e' di una esecuzione precedente: non va riusato
                if not (gen_csv.exists() and gen_csv.stat().st_mtime >= t_wall_start - 1):
                    break
                part = pd.read_csv(gen_csv)
                if len(part) == 0:
                    break
                parts.append(part)
                n_got += len(part)
                n_calls += 1
            if parts:
                synth_df = pd.concat(parts, ignore_index=True).iloc[:n_synth].reset_index(drop=True)
                success = len(synth_df) > 0
        finally:
            os.chdir(orig_cwd)
    except Exception as e:
        success = False
        err_text = f"{type(e).__name__}: {e}"

    status, method = "SUCCESS", "tabsyn"
    if not success or synth_df is None or len(synth_df) == 0:
        err_text = err_text or "tabsyn non ha prodotto alcun campione"
        if not ALLOW_FALLBACK:
            tracemalloc.stop()
            raise RuntimeError(f"tabsyn fallito (nessun fallback senza --allow_fallback): {err_text}")
        synth_df = _safe_knn_kde_fallback(df, n_synth)
        status, method = "FALLBACK", "knn-interpolation (NON tabsyn)"

    tr_time = time.perf_counter() - t0_tr
    _, pk_tr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0_gn = time.perf_counter()
    tracemalloc.start()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df, padded = _pad_to(synth_df, n_synth)
    if padded:
        status, method = "FALLBACK", f"{method}+resample-padding({padded} righe duplicate)"
        err_text = (err_text + " | " if err_text else "") + f"{padded} righe riempite per ricampionamento"
    synth_df.to_csv(out_path, index=False)
    gn_time = time.perf_counter() - t0_gn
    _, pk_gn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "status": status,
        "method_used": method,
        "error_message": err_text,
        "train_time_s": round(tr_time, 4),
        "gen_time_s": round(gn_time, 4),
        "train_peak_mem_mb": round(pk_tr / (1024 ** 2), 2),
        "gen_peak_mem_mb": round(pk_gn / (1024 ** 2), 2),
        "rss_delta_mb": 0.0,
    }


# ---------------------------------------------------------------------------
# 2. TabDiff Runner
# ---------------------------------------------------------------------------
def run_tabdiff(csv_path: Path, out_path: Path, device: str, n_samples: int, epochs: int) -> Dict[str, Any]:
    tabdiff_dir = ALGORITHMS_DIR / "TabDiff"
    if str(tabdiff_dir) not in sys.path:
        sys.path.insert(0, str(tabdiff_dir))

    df = pd.read_csv(csv_path)
    n_synth = len(df) if n_samples <= 0 else n_samples

    t0_tr = time.perf_counter()
    t_wall_start = time.time()
    tracemalloc.start()
    synth_df = None
    err_text = ""

    try:
        from types import SimpleNamespace
        from tabdiff.main import main as tabdiff_main
        import json

        ds_name = f"run_{csv_path.stem}"
        ds_data_dir = tabdiff_dir / "data" / ds_name
        ds_data_dir.mkdir(parents=True, exist_ok=True)
        info_dir = tabdiff_dir / "data" / "Info"
        info_dir.mkdir(parents=True, exist_ok=True)

        # Build info and data
        dest_csv = ds_data_dir / f"{ds_name}.csv"
        df.to_csv(dest_csv, index=False)

        num_idx = [i for i, c in enumerate(df.columns[:-1]) if np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 5]
        cat_idx = [i for i, c in enumerate(df.columns[:-1]) if i not in num_idx]
        target_idx = [len(df.columns) - 1]

        info = {
            "name": ds_name,
            "task_type": "binclass",
            "header": "infer",
            "column_names": list(df.columns),
            "num_col_idx": num_idx,
            "cat_col_idx": cat_idx,
            "target_col_idx": target_idx,
            "file_type": "csv",
            "data_path": f"data/{ds_name}/{ds_name}.csv",
            "val_path": None,
            "test_path": None,
        }
        with open(info_dir / f"{ds_name}.json", "w") as f:
            json.dump(info, f, indent=4)
        with open(ds_data_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        # TabDiff legge i .npy prodotti da process_dataset.py (X_num/X_cat/y train+test, info.json esteso)
        import subprocess
        pd_proc = subprocess.run([sys.executable, "process_dataset.py", "--dataname", ds_name],
                                 cwd=str(tabdiff_dir), capture_output=True, text=True)
        if pd_proc.returncode != 0:
            raise RuntimeError(f"process_dataset.py fallito: {(pd_proc.stderr or pd_proc.stdout)[-400:]}")

        orig_cwd = os.getcwd()
        try:
            os.chdir(str(tabdiff_dir))
            # Train
            train_args = SimpleNamespace(
                dataname=ds_name,
                mode="train",
                method="tabdiff",
                gpu=-1 if device == "cpu" else 0,
                device=device,
                debug=False,
                no_wandb=True,
                exp_name="bench",
                deterministic=True,
                y_only=False,
                non_learnable_schedule=True,
                num_samples_to_generate=None,
                ckpt_path=None,
                report=False,
                num_runs=1,
                impute=False,
                trial_start=0,
                trial_size=50,
                resample_rounds=1,
                impute_condition="x_t",
                y_only_model_path=None,
                w_num=0.6,
                w_cat=0.6,
            )
            # Il grafico delle densita' (plotly + kaleido + Chrome) e' solo diagnostico e si blocca senza
            # Chrome: si disattiva. I campioni sono gia' salvati in samples.csv prima del grafico.
            import tabdiff.metrics as _tdm
            from PIL import Image as _PILImage
            for _obj in vars(_tdm).values():
                if isinstance(_obj, type) and hasattr(_obj, "plot_density"):
                    _obj.plot_density = lambda self, *a, **k: _PILImage.new("RGB", (1, 1))
            # Le metriche interne di TabDiff (MLE, density...) non servono al benchmark (ha le sue) e MLE
            # va in ZeroDivisionError con una sola classe: i campioni sono gia' scritti prima di evaluate().
            _tdm.TabMetrics.evaluate = lambda self, syn_df, *a, **k: ({}, {})
            # numero di passi = epochs (il default del repository e' 8000) e nessuna valutazione intermedia
            import src as _tdsrc
            _orig_load_config = _tdsrc.load_config
            def _load_config_epochs(path, _o=_orig_load_config):
                cfg = _o(path)
                cfg["train"]["main"]["steps"] = max(1, int(epochs))
                cfg["train"]["main"]["check_val_every"] = max(1, int(epochs))  # salva model_<epochs>.pt a fine corsa
                return cfg
            _tdsrc.load_config = _load_config_epochs
            # Con lo schedule non apprendibile alpha_t/alpha_s hanno forma (bs, 1): con >= 2 colonne categoriche
            # (es. iris0: Attr3 con <= 5 valori + classe) l'assegnazione q_xs[:, range(K), mask_index] = (bs, 1)
            # in _mdlm_update fa scattare un INTERNAL ASSERT di index_put su CUDA. Si espandono a (bs, K):
            # stesso valore per ogni colonna, risultato matematicamente identico.
            from tabdiff.models import unified_ctime_diffusion as _tducd
            _UCD = _tducd.UnifiedCtimeDiffusion
            if not getattr(_UCD._mdlm_update, "_augpilot_patched", False):
                _orig_mdlm = _UCD._mdlm_update
                def _mdlm_update_bcast(self, log_p_x0, x, alpha_t, alpha_s, _o=_orig_mdlm):
                    k = log_p_x0.shape[1]
                    if alpha_t.ndim == 2 and alpha_t.shape[1] == 1 and k > 1:
                        alpha_t = alpha_t.expand(-1, k).contiguous()
                    if alpha_s.ndim == 2 and alpha_s.shape[1] == 1 and k > 1:
                        alpha_s = alpha_s.expand(-1, k).contiguous()
                    return _o(self, log_p_x0, x, alpha_t, alpha_s)
                _mdlm_update_bcast._augpilot_patched = True
                _UCD._mdlm_update = _mdlm_update_bcast
            # checkpoint/risultati di una esecuzione precedente dello stesso dataset non vanno riusati
            import shutil as _sh
            for _d in (tabdiff_dir / "tabdiff" / "ckpt" / ds_name, tabdiff_dir / "tabdiff" / "result" / ds_name):
                _sh.rmtree(_d, ignore_errors=True)
            tabdiff_main(train_args)

            # Test/Sample
            test_args = SimpleNamespace(
                dataname=ds_name,
                mode="test",
                method="tabdiff",
                gpu=-1 if device == "cpu" else 0,
                device=device,
                debug=False,
                no_wandb=True,
                exp_name="bench",
                deterministic=True,
                y_only=False,
                non_learnable_schedule=True,
                num_samples_to_generate=n_synth,
                ckpt_path=str(tabdiff_dir / "tabdiff" / "ckpt" / ds_name / "bench" / f"model_{max(1, int(epochs))}.pt"),
                report=False,
                num_runs=1,
                impute=False,
                trial_start=0,
                trial_size=50,
                resample_rounds=1,
                impute_condition="x_t",
                y_only_model_path=None,
                w_num=0.6,
                w_cat=0.6,
            )
            tabdiff_main(test_args)

            # solo file scritti da QUESTA esecuzione (altrimenti si rischia il risultato di un altro dataset)
            res_cand = [f for f in (tabdiff_dir / "tabdiff" / "result" / ds_name).rglob("samples.csv")
                        if f.stat().st_mtime >= t_wall_start - 1]
            if res_cand:
                res_cand.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                synth_df = pd.read_csv(res_cand[0])
        finally:
            os.chdir(orig_cwd)
    except Exception as e:
        synth_df = None
        err_text = f"{type(e).__name__}: {e}"

    status, method = "SUCCESS", "tabdiff"
    if synth_df is None or len(synth_df) == 0:
        err_text = err_text or "tabdiff non ha prodotto alcun campione"
        if not ALLOW_FALLBACK:
            tracemalloc.stop()
            raise RuntimeError(f"tabdiff fallito (nessun fallback senza --allow_fallback): {err_text}")
        synth_df = _safe_knn_kde_fallback(df, n_synth)
        status, method = "FALLBACK", "knn-interpolation (NON tabdiff)"

    tr_time = time.perf_counter() - t0_tr
    _, pk_tr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0_gn = time.perf_counter()
    tracemalloc.start()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df, padded = _pad_to(synth_df, n_synth)
    if padded:
        status, method = "FALLBACK", f"{method}+resample-padding({padded} righe duplicate)"
        err_text = (err_text + " | " if err_text else "") + f"{padded} righe riempite per ricampionamento"
    synth_df.to_csv(out_path, index=False)
    gn_time = time.perf_counter() - t0_gn
    _, pk_gn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "status": status,
        "method_used": method,
        "error_message": err_text,
        "train_time_s": round(tr_time, 4),
        "gen_time_s": round(gn_time, 4),
        "train_peak_mem_mb": round(pk_tr / (1024 ** 2), 2),
        "gen_peak_mem_mb": round(pk_gn / (1024 ** 2), 2),
        "rss_delta_mb": 0.0,
    }


# ---------------------------------------------------------------------------
# 3. OCTree Runner (Orthogonal Cuts Decision Tree Synthesizer)
# ---------------------------------------------------------------------------
def run_octree(csv_path: Path, out_path: Path, device: str, n_samples: int, epochs: int) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    n_synth = len(df) if n_samples <= 0 else n_samples
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    t0_tr = time.perf_counter()
    tracemalloc.start()

    # Preprocessing
    col_encoders = {}
    X_num = pd.DataFrame(index=df.index)
    for c in feature_cols:
        s = df[c]
        if s.dtype == "object" or s.nunique() <= 5:
            cats = sorted(s.unique())
            m = {val: idx for idx, val in enumerate(cats)}
            col_encoders[c] = ("cat", cats, m)
            X_num[c] = s.map(m).fillna(0).astype(float)
        else:
            col_encoders[c] = ("num", float(s.min()), float(s.max()))
            X_num[c] = s.astype(float).fillna(s.mean())

    y_series = df[target_col]
    y_cats = sorted(y_series.unique())
    y_map = {v: i for i, v in enumerate(y_cats)}
    y_enc = y_series.map(y_map).values

    # Train tree ensemble to discover orthogonal cuts & split bounds
    from sklearn.tree import DecisionTreeClassifier
    trees = []
    max_d = max(2, min(8, int(np.log2(max(4, len(df))))))
    for seed in range(min(10, max(3, epochs // 50))):
        dt = DecisionTreeClassifier(max_depth=max_d, random_state=seed)
        dt.fit(X_num.values, y_enc)
        trees.append(dt)

    tr_time = time.perf_counter() - t0_tr
    _, pk_tr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0_gn = time.perf_counter()
    tracemalloc.start()

    # Sample points within the leaf hyperboxes discovered by OCTree
    rng = np.random.default_rng(42)
    leaf_nodes = [t.apply(X_num.values) for t in trees]
    synth_rows = []
    synth_y_list = []

    for _ in range(n_synth):
        tree_idx = rng.integers(0, len(trees))
        selected_tree = trees[tree_idx]
        sample_idx = rng.integers(0, len(df))
        leaf_id = leaf_nodes[tree_idx][sample_idx]

        in_leaf = np.where(leaf_nodes[tree_idx] == leaf_id)[0]
        if len(in_leaf) > 1:
            i1 = rng.choice(in_leaf)
            i2 = rng.choice(in_leaf)
            alpha = rng.uniform(0.1, 0.9)
            new_pt = alpha * X_num.values[i1] + (1 - alpha) * X_num.values[i2]
            chosen_y = y_enc[i1]
        else:
            new_pt = X_num.values[sample_idx] + rng.normal(0, 0.05, size=X_num.shape[1])
            chosen_y = y_enc[sample_idx]

        synth_rows.append(new_pt)
        synth_y_list.append(chosen_y)

    synth_rows = np.array(synth_rows)
    out_dict = {}
    for idx, c in enumerate(feature_cols):
        info = col_encoders[c]
        if info[0] == "cat":
            cats = info[1]
            rounded = np.clip(np.round(synth_rows[:, idx]).astype(int), 0, len(cats) - 1)
            out_dict[c] = [cats[r] for r in rounded]
        else:
            c_min, c_max = info[1], info[2]
            out_dict[c] = np.clip(synth_rows[:, idx], c_min, c_max)

    out_dict[target_col] = [y_cats[v] for v in synth_y_list]
    synth_df = pd.DataFrame(out_dict, columns=df.columns)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)
    gn_time = time.perf_counter() - t0_gn
    _, pk_gn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "status": "FALLBACK",
        "method_used": "surrogate:tree-ensemble-leaf-interpolation",
        "error_message": "Richiede LLM (OCTree: Llama-2-7B locale o API, feature generation): non eseguito. Surrogato semplificato (alberi + interpolazione nelle foglie), NON Algorithms/OCTree-main",
        "train_time_s": round(tr_time, 4),
        "gen_time_s": round(gn_time, 4),
        "train_peak_mem_mb": round(pk_tr / (1024 ** 2), 2),
        "gen_peak_mem_mb": round(pk_gn / (1024 ** 2), 2),
        "rss_delta_mb": 0.0,
    }


# ---------------------------------------------------------------------------
# 4. HARMONIC Runner (Tabular Language Model Tokenizer & Synthesizer)
# ---------------------------------------------------------------------------
def run_harmonic(csv_path: Path, out_path: Path, device: str, n_samples: int, epochs: int) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    n_synth = len(df) if n_samples <= 0 else n_samples

    t0_tr = time.perf_counter()
    tracemalloc.start()

    # Preprocessing & Tabular Serialization into tokens
    col_names = list(df.columns)
    col_tokens = {}
    for c in col_names:
        s = df[c]
        if np.issubdtype(s.dtype, np.number) and s.nunique() > 10:
            bins = np.quantile(s.dropna(), np.linspace(0, 1, 11))
            bins = np.unique(bins)
            if len(bins) <= 1:
                bins = np.array([float(s.min()) - 1.0, float(s.max()) + 1.0])
            col_tokens[c] = ("num", bins)
        else:
            cats = sorted(s.unique())
            col_tokens[c] = ("cat", cats)

    # Token transition distributions (HARMONIC autoregressive model)
    transitions = {}
    serialized_corpus = []
    for _, row in df.iterrows():
        tokens = []
        for c in col_names:
            val = row[c]
            info = col_tokens[c]
            if info[0] == "num":
                bins = info[1]
                b_idx = int(np.clip(np.digitize(val, bins[1:-1]), 0, len(bins) - 2))
                tokens.append(f"{c}_bin{b_idx}")
            else:
                tokens.append(f"{c}_{val}")
        serialized_corpus.append(tokens)

    # Frequency-based conditional n-gram distribution
    for tokens in serialized_corpus:
        for i in range(len(tokens) - 1):
            ctx = tokens[i]
            nxt = tokens[i + 1]
            if ctx not in transitions:
                transitions[ctx] = {}
            transitions[ctx][nxt] = transitions[ctx].get(nxt, 0) + 1

    tr_time = time.perf_counter() - t0_tr
    _, pk_tr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0_gn = time.perf_counter()
    tracemalloc.start()

    rng = np.random.default_rng(42)
    generated_rows = []
    first_col = col_names[0]

    for _ in range(n_synth):
        init_row = rng.choice(serialized_corpus)
        cur_token = init_row[0]
        gen_tokens = [cur_token]

        for step in range(1, len(col_names)):
            if cur_token in transitions and transitions[cur_token]:
                cand = list(transitions[cur_token].keys())
                weights = np.array(list(transitions[cur_token].values()), dtype=float)
                weights /= weights.sum()
                cur_token = rng.choice(cand, p=weights)
            else:
                cur_token = init_row[step]
            gen_tokens.append(cur_token)

        # Deserialize tokens into row
        row_dict = {}
        for c_idx, c in enumerate(col_names):
            t = gen_tokens[c_idx]
            info = col_tokens[c]
            if info[0] == "num":
                bins = info[1]
                try:
                    b_idx = int(t.split("_bin")[-1])
                except Exception:
                    b_idx = 0
                b_idx = min(b_idx, len(bins) - 2)
                b_min, b_max = bins[b_idx], bins[b_idx + 1]
                row_dict[c] = float(rng.uniform(b_min, b_max))
            else:
                val_str = t[len(c) + 1:]
                row_dict[c] = val_str
        generated_rows.append(row_dict)

    synth_df = pd.DataFrame(generated_rows, columns=col_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)

    gn_time = time.perf_counter() - t0_gn
    _, pk_gn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "status": "FALLBACK",
        "method_used": "surrogate:quantile-bin-ngram-sampler",
        "error_message": "Richiede LLM (HARMONIC: fine-tuning LoRA di un LLM 7B, HF): non eseguito. Surrogato semplificato (n-gram su colonne discretizzate), NON Algorithms/HARMONIC-main",
        "train_time_s": round(tr_time, 4),
        "gen_time_s": round(gn_time, 4),
        "train_peak_mem_mb": round(pk_tr / (1024 ** 2), 2),
        "gen_peak_mem_mb": round(pk_gn / (1024 ** 2), 2),
        "rss_delta_mb": 0.0,
    }


# ---------------------------------------------------------------------------
# 5. Synthetic-LLM Runner (Few-Shot Tabular LLM Synthesizer)
# ---------------------------------------------------------------------------
def run_synthetic_llm(csv_path: Path, out_path: Path, device: str, n_samples: int, epochs: int) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    n_synth = len(df) if n_samples <= 0 else n_samples

    t0_tr = time.perf_counter()
    tracemalloc.start()

    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    # Serialization template & prompt context builder
    templates = []
    for _, row in df.iterrows():
        items = [f"{col}: {row[col]}" for col in df.columns]
        templates.append(" | ".join(items))

    # Feature statistics for constrained few-shot prompting
    col_bounds = {}
    for c in feature_cols:
        s = df[c]
        if np.issubdtype(s.dtype, np.number):
            col_bounds[c] = (float(s.min()), float(s.max()), float(s.std() or 1.0))
        else:
            col_bounds[c] = list(s.unique())

    tr_time = time.perf_counter() - t0_tr
    _, pk_tr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0_gn = time.perf_counter()
    tracemalloc.start()

    rng = np.random.default_rng(42)
    synth_records = []

    # Generative sampling using few-shot exemplar interpolation
    for _ in range(n_synth):
        idx_anchor = rng.integers(0, len(df))
        anchor_row = df.iloc[idx_anchor]

        # Draw a second neighbor exemplar as prompt context
        idx_ctx = rng.integers(0, len(df))
        ctx_row = df.iloc[idx_ctx]

        new_row = {}
        for c in feature_cols:
            if c in col_bounds and isinstance(col_bounds[c], tuple):
                c_min, c_max, c_std = col_bounds[c]
                v1, v2 = float(anchor_row[c]), float(ctx_row[c])
                alpha = rng.uniform(0.0, 1.0)
                jitter = rng.normal(0, 0.05 * c_std)
                new_val = np.clip(alpha * v1 + (1 - alpha) * v2 + jitter, c_min, c_max)
                new_row[c] = new_val
            else:
                # Categorical attribute: pick with probability proportional to exemplar
                cats = col_bounds[c]
                p_anchor = 0.7
                if rng.uniform(0, 1) < p_anchor:
                    new_row[c] = anchor_row[c]
                else:
                    new_row[c] = rng.choice(cats)

        new_row[target_col] = anchor_row[target_col]
        synth_records.append(new_row)

    synth_df = pd.DataFrame(synth_records, columns=df.columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)

    gn_time = time.perf_counter() - t0_gn
    _, pk_gn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "status": "FALLBACK",
        "method_used": "surrogate:few-shot-exemplar-interpolation",
        "error_message": "Richiede LLM (EPIC: API OpenAI/GPT): non interrogato. Surrogato: interpolazione tra esemplari, NON Algorithms/synthetic-tabular-LLM-main",
        "train_time_s": round(tr_time, 4),
        "gen_time_s": round(gn_time, 4),
        "train_peak_mem_mb": round(pk_tr / (1024 ** 2), 2),
        "gen_peak_mem_mb": round(pk_gn / (1024 ** 2), 2),
        "rss_delta_mb": 0.0,
    }


# ---------------------------------------------------------------------------
# 6. Kamino Runner (Constraint-Aware Tabular Data Synthesizer)
# ---------------------------------------------------------------------------
from workers.kamino_engine import (
    parse_dc_file,
    build_compiled_dcs,
    learn_kamino_weights,
    run_kamino_synthesis,
)


def run_kamino(
    csv_path: Path,
    out_path: Path,
    device: str,
    n_samples: int,
    epochs: int,
    dc_file: Optional[str] = None,
    kamino_weights: str = "toggled",
) -> Dict[str, Any]:
    """
    Kamino: A Synthesizer for Relational Data with Integrity Constraints (VLDB 2021).
    Phase 1: Feature learning & Denial Constraint weight determination (ICweight SGD/toggle).
    Phase 2: Joint sampling from marginal models.
    Phase 3: Constraint evaluation with Gibbs energy acceptance-rejection & MCMC repair.
    """
    if not dc_file or not Path(dc_file).exists():
        raise ValueError(
            f"[KAMINO ERROR] Kamino cannot be executed without a specified Denial Constraints file! "
            f"No valid DC file was provided or found for dataset '{csv_path.stem}'. (dc_file={dc_file!r})"
        )

    df = pd.read_csv(csv_path)
    n_synth = len(df) if n_samples <= 0 else n_samples

    t0_tr = time.perf_counter()
    tracemalloc.start()

    # Parse DCs from either SyDC or Kamino .ic syntax
    dcs = parse_dc_file(Path(dc_file), list(df.columns))
    if not dcs:
        raise ValueError(
            f"[KAMINO ERROR] No valid Denial Constraints could be parsed from '{dc_file}' for dataset '{csv_path.stem}'."
        )

    compiled_dcs = build_compiled_dcs(dcs, list(df.columns))
    if not compiled_dcs:
        raise ValueError(
            f"[KAMINO ERROR] None of the {len(dcs)} Denial Constraints in '{dc_file}' match the columns of {csv_path.name}."
        )

    # Learn/determine Kamino DC weights based on requested strategy
    weights = learn_kamino_weights(
        df.values.astype(float),
        compiled_dcs,
        strategy=kamino_weights,
        seed=42,
    )

    tr_time = time.perf_counter() - t0_tr
    _, pk_tr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0_gn = time.perf_counter()
    tracemalloc.start()

    # Phase 2 & 3: Synthesis with constraint enforcement and repair
    synth_df = run_kamino_synthesis(
        df,
        n_synth=n_synth,
        compiled_dcs=compiled_dcs,
        weights=weights,
        epochs=epochs,
        seed=42,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)

    gn_time = time.perf_counter() - t0_gn
    _, pk_gn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "train_time_s": round(tr_time, 4),
        "gen_time_s": round(gn_time, 4),
        "train_peak_mem_mb": round(pk_tr / (1024 ** 2), 2),
        "gen_peak_mem_mb": round(pk_gn / (1024 ** 2), 2),
        "rss_delta_mb": 0.0,
        "status": "FALLBACK",
        "method_used": "kamino (reimplementazione Python workers/kamino_engine.py, NON il codice di Algorithms/kamino-master)",
        "error_message": "",
        "n_dcs": len(compiled_dcs),
        "dc_weight_strategy": kamino_weights,
    }


# ---------------------------------------------------------------------------
# CLI Dispatcher
# ---------------------------------------------------------------------------
ALGO_MAP = {
    "tabsyn": run_tabsyn,
    "tabdiff": run_tabdiff,
    "octree": run_octree,
    "harmonic": run_harmonic,
    "synthetic-llm": run_synthetic_llm,
    "synthetic_llm": run_synthetic_llm,
    "kamino": run_kamino,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dc_file", "--dc-file", type=str, default=None)
    parser.add_argument("--kamino_weights", "--kamino-weights", type=str, default="toggled")
    parser.add_argument("--allow_fallback", "--allow-fallback", action="store_true",
                        help="Se il metodo reale fallisce, ripiega su KNN (stato FALLBACK). Default: errore.")
    args = parser.parse_args()
    global ALLOW_FALLBACK
    ALLOW_FALLBACK = bool(args.allow_fallback)

    algo_key = args.algo.lower().strip()
    handler = ALGO_MAP.get(algo_key)
    if not handler:
        raise ValueError(f"Unknown algorithm for worker: '{args.algo}'")

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    try:
        metrics = _dispatch(handler, algo_key, csv_path, out_path, args)
    except Exception as e:  # il worker non deve mai riportare SUCCESS dopo un errore
        print(json.dumps({"status": "FAILED", "method_used": algo_key,
                          "error_message": f"{type(e).__name__}: {e}"[:1000]}))
        sys.exit(1)
    print(json.dumps(metrics))


def _dispatch(handler, algo_key, csv_path, out_path, args):
    if algo_key == "kamino":
        return handler(
            csv_path,
            out_path,
            args.device,
            args.n_samples,
            args.epochs,
            dc_file=args.dc_file,
            kamino_weights=args.kamino_weights,
        )
    return handler(csv_path, out_path, args.device, args.n_samples, args.epochs)


if __name__ == "__main__":
    import seed_util; seed_util.apply_env_seed()
    main()
