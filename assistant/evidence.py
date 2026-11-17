"""Lettura dei risultati misurati (CSV) con la fonte di ogni numero. Nessuna stima, nessun riempimento."""
from __future__ import annotations

import glob
import os
import re
from typing import Optional

import pandas as pd

from . import config

# Direzione nota delle metriche tabulari (da evaluate_metrics.py): solo quelle di cui la direzione e' certa.
FIDELITY_LOWER_BETTER = ["JS_mean", "KS_mean", "WD_mean", "pMSE", "MMD"]
DIVERSITY_HIGHER_BETTER = ["Precision", "Recall", "Coverage"]


def canon_algo(a: str) -> str:
    a = str(a).lower().strip()
    if a.startswith("kamino"):
        return "kamino"
    if a.startswith("sydc"):
        return "sydc"
    return a


def canon_ds(d: str) -> str:
    d = str(d).strip().lower()
    d = re.sub(r"_augmented$", "", d)
    return re.sub(r"_min$", "", d)


def _rel(p: str) -> str:
    return os.path.relpath(p, config.ROOT).replace("\\", "/")


def tabular_runtime() -> pd.DataFrame:
    """Tempi/stati delle esecuzioni tabulari: runtime_results/ (benchmark storico) e runtime_results_ui/ (lanciate
    dalla UI o dall'assistente, cartelle dedicate)."""
    fs = []
    for d in config.RESULTS_DIRS:
        fs += sorted(glob.glob(str(config.ROOT / d / "augmentation_benchmark_2*.csv")))
    parts = []
    for f in fs:
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        d["source"] = _rel(f)
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["algo"] = d["algorithm"].map(canon_algo)
    d["ds"] = d["dataset"].map(canon_ds)
    return d


def tabular_success_algos() -> set:
    d = tabular_runtime()
    return set() if d.empty else set(d.loc[d["status"] == "SUCCESS", "algo"])


def _merge_fd(f: pd.DataFrame, v: pd.DataFrame, sources: str) -> pd.DataFrame:
    skip = ("Dataset", "Algoritmo", "File", "algo", "ds")
    for d in (f, v):
        d["algo"] = d["Algoritmo"].map(canon_algo)
        d["ds"] = d["Dataset"].map(canon_ds)
    num_f = [c for c in f.columns if c not in skip]
    num_v = [c for c in v.columns if c not in skip and c != "N_Generated"]
    f = f.groupby(["ds", "algo"], as_index=False)[num_f].mean(numeric_only=True)
    v = v.groupby(["ds", "algo"], as_index=False)[num_v].mean(numeric_only=True)
    m = f.merge(v, on=["ds", "algo"], how="outer")
    m["sources"] = sources
    return m


def _ui_eval_tables():
    """Valutazioni delle esecuzioni lanciate da UI/assistente: file per dataset in evaluation_metrics_ui/{fidelity,diversity}/."""
    base = config.ROOT / config.UI_EVAL_DIR
    fs = sorted(glob.glob(str(base / "fidelity" / "fidelity_*.csv")))
    vs = sorted(glob.glob(str(base / "diversity" / "diversity_*.csv")))
    try:
        f = pd.concat([pd.read_csv(x) for x in fs if not x.endswith("all_datasets.csv")], ignore_index=True)
        v = pd.concat([pd.read_csv(x) for x in vs if not x.endswith("all_datasets.csv")], ignore_index=True)
    except Exception:  # noqa: BLE001  nessun file o file illeggibili
        return None
    return f, v


def tabular_metrics() -> pd.DataFrame:
    """Fidelity e diversity uniti per (dataset, algoritmo); le varianti (es. kamino-*) sono mediate.
    Fonti: evaluation_metrics/*_all_datasets.csv (storico) e evaluation_metrics_ui/ (UI/assistente); in caso di
    stessa coppia (dataset, algoritmo) vale la valutazione della UI, piu' recente."""
    base = config.ROOT / "evaluation_metrics"
    m = pd.DataFrame()
    try:
        m = _merge_fd(pd.read_csv(base / "fidelity_all_datasets.csv"), pd.read_csv(base / "diversity_all_datasets.csv"),
                      "evaluation_metrics/fidelity_all_datasets.csv + evaluation_metrics/diversity_all_datasets.csv")
    except Exception:  # noqa: BLE001
        pass
    ui = _ui_eval_tables()
    if ui is not None:
        mu = _merge_fd(ui[0], ui[1], f"{config.UI_EVAL_DIR}/fidelity/*.csv + {config.UI_EVAL_DIR}/diversity/*.csv")
        if m.empty:
            return mu
        keys = set(zip(mu["ds"], mu["algo"]))
        m = pd.concat([m[[(a, b) not in keys for a, b in zip(m["ds"], m["algo"])]], mu], ignore_index=True)
    return m


def modality_runtime(modality: str) -> pd.DataFrame:
    # CSV del benchmark/prove a mano (runtime_results_<m>/) e delle esecuzioni lanciate da UI e assistente (runtime_results_ui_<m>/)
    dirs = [config.BENCH_RES_DIRS[modality], config.UI_RES_DIRS[modality]]
    parts = []
    for f in sorted(f for d_ in dirs for f in glob.glob(str(config.ROOT / d_ / "*.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        d["source"] = _rel(f)
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    if "algorithm" in d:
        d["algo"] = d["algorithm"].astype(str).str.lower()
    if "dataset" in d:
        d["ds"] = d["dataset"].astype(str).str.lower()
    return d


def fnum(x) -> Optional[float]:
    try:
        v = float(x)
        return None if v != v else round(v, 6)
    except (TypeError, ValueError):
        return None
