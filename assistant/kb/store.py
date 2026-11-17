"""Lettura del KB (benchmark + meta-feature) con cache sulla data di modifica dei file."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parent
ENV_VAR = "AUGPILOT_KB_DIR"


@dataclass
class KB:
    dir: Path
    bench: pd.DataFrame
    meta: pd.DataFrame
    source: str          # file del benchmark letto (relativo alla cartella del KB)

    def rows(self, modality: str, level: Optional[str] = None) -> pd.DataFrame:
        d = self.bench[self.bench["modality"] == modality]
        if level and modality == "graph":
            d = d[d["level"] == level]
        return d


def canon(name: str) -> str:
    """Nome dataset canonico per il confronto (minuscolo, senza _min/_augmented)."""
    d = str(name).strip().lower()
    d = re.sub(r"_augmented$", "", d)
    return re.sub(r"_min$", "", d)


def canon_algo(name: str) -> str:
    a = str(name).lower().strip()
    if a.startswith("kamino"):
        return "kamino"
    if a.startswith("sydc"):
        return "sydc"
    return a


def kb_dir(path: Optional[os.PathLike] = None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get(ENV_VAR, "").strip()
    return Path(env) if env else DEFAULT_DIR


_cache: Dict[Tuple[str, float, float], KB] = {}


def load_kb(path: Optional[os.PathLike] = None) -> Optional[KB]:
    """KB dalla cartella `path` (default assistant/kb o $AUGPILOT_KB_DIR); None se non c'e' o e' vuoto."""
    d = kb_dir(path)
    pq, csv, meta = d / "benchmark.parquet", d / "benchmark.csv", d / "dataset_meta.csv"
    src = pq if pq.is_file() else csv if csv.is_file() else None
    if src is None:
        return None
    key = (str(src.resolve()), src.stat().st_mtime, meta.stat().st_mtime if meta.is_file() else 0.0)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    try:
        bench = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    except Exception:  # noqa: BLE001
        return None
    if bench.empty or not {"modality", "dataset", "algorithm", "status"} <= set(bench.columns):
        return None
    bench["level"] = bench["level"].fillna("").astype(str) if "level" in bench else ""
    bench["status"] = bench["status"].astype(str).str.upper()
    bench["_ds"] = bench["dataset"].map(canon)
    bench["_algo"] = bench["algorithm"].map(canon_algo)
    try:
        m = pd.read_csv(meta) if meta.is_file() else pd.DataFrame()
    except Exception:  # noqa: BLE001
        m = pd.DataFrame()
    if not m.empty:
        m["level"] = m["level"].fillna("").astype(str)
        m["_ds"] = m["dataset"].map(canon)
    kb = KB(dir=d, bench=bench, meta=m, source=src.name)
    _cache.clear()
    _cache[key] = kb
    return kb


COMPLETED_STATUSES = ("SUCCESS", "FALLBACK")


def completed_algos(modality: str, path: Optional[os.PathLike] = None) -> Optional[set]:
    """Algoritmi (nome canonico) con almeno una esecuzione completata (SUCCESS o FALLBACK) nel KB per la modalita'.

    None se il KB non c'e' (chi chiama decide cosa fare); riusa `load_kb`, quindi e' in cache.
    """
    kb = load_kb(path)
    if kb is None:
        return None
    d = kb.bench[(kb.bench["modality"] == modality) & kb.bench["status"].isin(COMPLETED_STATUSES)]
    return set(d["_algo"])
