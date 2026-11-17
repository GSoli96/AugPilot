"""Riepilogo del KB per modalita' (serve all'API /api/kb/summary e alla pagina «Prove»)."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from . import scoring, store

MODALITIES = ("tabular", "ts", "graph")


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return round(v, 6) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def kb_summary(modality: str, level: Optional[str] = None, dataset: Optional[str] = None,
               kb_dir: Optional[str] = None) -> Dict[str, Any]:
    """Per modalita' (e livello, per i grafi): dataset, algoritmi, punteggi per asse medi, tempo mediano.

    `algorithms`: una riga per algoritmo, media (semplice) sui dataset dei punteggi d'asse; `matrix`: una riga per
    (dataset, algoritmo). I punteggi d'asse sono percentili 0..1 (1 = migliore) tra gli algoritmi dello stesso dataset
    (vedi assistant/kb/scoring.py); `speed` e' il percentile del tempo."""
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    kb = store.load_kb(kb_dir)
    if kb is None:
        return {"modality": modality, "available": False, "datasets": [], "algorithms": [], "matrix": [],
                "message": "KB not built: run python -m assistant.kb.build_kb"}
    rows = kb.rows(modality, level)
    if dataset:
        rows = rows[rows["_ds"] == store.canon(dataset)]
    if rows.empty:
        return {"modality": modality, "level": level, "available": False, "datasets": [], "algorithms": [], "matrix": [],
                "kb": {"file": f"assistant/kb/{kb.source}"}, "message": "no runs in the KB for this modality"}
    matrix, by_algo = [], {}
    datasets = []
    for (name, lvl), g in rows.groupby(["dataset", "level"], sort=True):
        sc = scoring.score_dataset(g, modality)
        meta = scoring.meta_vector(kb, modality, lvl, name) or {}
        datasets.append({"dataset": name, "level": lvl or None, "n_runs": int(len(g)), "n_algorithms": len(sc),
                         "meta": {k: _f(v) for k, v in meta.items()}})
        for algo, info in sc.items():
            matrix.append({"dataset": name, "level": lvl or None, "algorithm": algo, "n_runs": info["n_runs"],
                           "n_seeds": info["n_seeds"], "failure_rate": _f(info["failure_rate"]),
                           "time_s_median": _f(info["time_s_median"]),
                           "axis_scores": {ax: _f(v) for ax, v in info["axis_scores"].items()}})
            by_algo.setdefault(algo, []).append(info)
    algorithms = []
    for algo, infos in sorted(by_algo.items()):
        axes = {}
        for ax in scoring.ALL_AXES:
            v = [i["axis_scores"][ax] for i in infos if i["axis_scores"].get(ax) is not None]
            axes[ax] = _f(np.mean(v)) if v else None
        times = [i["time_s_median"] for i in infos if i["time_s_median"] is not None]
        n_runs = sum(i["n_runs"] for i in infos)
        algorithms.append({"algorithm": algo, "n_datasets": len(infos), "n_runs": n_runs,
                           "failure_rate": _f(sum(i["n_failed"] for i in infos) / n_runs) if n_runs else None,
                           "time_s_median": _f(np.median(times)) if times else None, "axis_scores": axes})
    return {"modality": modality, "level": level, "available": True, "kb": {"file": f"assistant/kb/{kb.source}",
            "runs": int(len(rows))}, "datasets": datasets, "algorithms": algorithms, "matrix": matrix,
            "axes": list(scoring.ALL_AXES),
            "note": "axis scores = percentiles 0..1 (1 = best) among the algorithms of the same dataset; simple mean over datasets"}
