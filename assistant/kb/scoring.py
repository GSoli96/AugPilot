"""Punteggi per (dataset, algoritmo) dal KB e stima per dataset nuovi (k-NN sulle meta-feature).

Punteggio di metrica: percentile 0..1 dell'algoritmo tra quelli misurati sullo stesso dataset, nella stessa
combinazione (seed, epoche): 1 = migliore, 0 = peggiore (rango medio per i pari merito); con un solo algoritmo
0.5 (nessun confronto). 'target1' = distanza da 1 (piu' vicino e' meglio). Una run FAILED vale 0 su tutte le metriche
e sulla velocita'. Le run SUCCESS e FALLBACK entrano nel confronto (FALLBACK e' segnalato).
Punteggio d'asse = media dei punteggi delle metriche dell'asse; velocita' = percentile del tempo (basso = meglio).
Media sulle combinazioni (seed, epoche).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics_spec import AXES, is_metric_column, metric_spec
from .store import KB, canon

OK_STATUS = ("SUCCESS", "FALLBACK")
ALL_AXES = AXES + ("speed",)
K_NEIGHBORS = 3

# meta-feature usate dal k-NN per (modalita, livello); le dimensioni passano da log1p
META_FEATURES = {
    ("tabular", ""): ["n_rows", "n_cols", "ir", "low_card_frac", "n_min"],
    ("ts", ""): ["n_series", "length_T", "dims_D", "has_labels", "ir"],
    ("graph", "graph"): ["n_graphs", "avg_nodes", "avg_edges", "n_features", "n_classes", "ir"],
    ("graph", "node"): ["avg_nodes", "avg_edges", "n_features", "n_classes", "ir"],
}
NO_LOG = {"low_card_frac", "has_labels"}


def percentile_scores(values: pd.Series, direction: str) -> pd.Series:
    """Percentile 0..1 (1 = migliore) per i valori non nulli; NaN altrove."""
    v = values.astype(float)
    if direction == "target1":
        v = (v - 1.0).abs()
        direction = "lower"
    ok = v.notna()
    n = int(ok.sum())
    out = pd.Series(np.nan, index=values.index)
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.5
        return out
    r = v[ok].rank(method="average", ascending=(direction == "lower"))
    out[ok] = 1.0 - (r - 1.0) / (n - 1.0)
    return out


def metric_columns(df: pd.DataFrame, modality: str) -> Dict[str, Tuple[str, str]]:
    """{colonna: (direzione, asse)} per le metriche ordinabili presenti in df con almeno un valore."""
    out = {}
    for c in df.columns:
        if not is_metric_column(c):
            continue
        d, ax = metric_spec(modality, c)
        if d != "neutral" and ax and df[c].notna().any():
            out[c] = (d, ax)
    return out


def _merge_duplicates(d: pd.DataFrame, mcols: Sequence[str]) -> pd.DataFrame:
    """Un'unica riga per (algoritmo, seed, epoche): se ce ne sono piu' (varianti), media delle metriche, stato peggiore."""
    if not d.duplicated(["_algo", "seed", "epochs"]).any():
        return d
    rank = {"FAILED": 0, "FALLBACK": 1, "SUCCESS": 2}
    rows = []
    for (_, _, _), g in d.groupby(["_algo", "seed", "epochs"], sort=False):
        r = g.iloc[0].copy()
        r["status"] = min(g["status"], key=lambda s: rank.get(s, 0))
        for c in list(mcols) + ["time_s"]:
            r[c] = g[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def score_dataset(df: pd.DataFrame, modality: str) -> Dict[str, Dict[str, Any]]:
    """Punteggi per algoritmo sulle run `df` di UN dataset. Restituisce {algo: info}."""
    if df.empty:
        return {}
    cols = metric_columns(df, modality)
    d = _merge_duplicates(df.copy(), list(cols))
    algos = sorted(d["_algo"].unique())
    acc: Dict[str, Dict[str, Any]] = {
        a: {"axis": {ax: [] for ax in ALL_AXES}, "metric": {c: [] for c in cols}, "values": {c: [] for c in cols},
            "n_runs": 0, "n_failed": 0, "n_fallback": 0, "n_ok": 0, "seeds": set(), "times": []} for a in algos}
    for (_seed, _ep), g in d.groupby(["seed", "epochs"], sort=True):
        ok = g["status"].isin(OK_STATUS)
        pct: Dict[str, pd.Series] = {c: percentile_scores(g.loc[ok, c], dr) for c, (dr, _ax) in cols.items()}
        speed = percentile_scores(g.loc[ok, "time_s"], "lower") if "time_s" in g else pd.Series(dtype=float)
        for idx, row in g.iterrows():
            a = acc[row["_algo"]]
            a["n_runs"] += 1
            a["seeds"].add(int(row["seed"]))
            if row["status"] == "FAILED" or not ok[idx]:
                a["n_failed"] += 1
                for ax in ALL_AXES:
                    a["axis"][ax].append(0.0)
                for c in cols:
                    a["metric"][c].append(0.0)
                continue
            a["n_ok"] += 1
            a["n_fallback"] += int(row["status"] == "FALLBACK")
            if pd.notna(row.get("time_s")):
                a["times"].append(float(row["time_s"]))
            per_axis: Dict[str, List[float]] = {ax: [] for ax in AXES}
            for c, (_dr, ax) in cols.items():
                p = pct[c].get(idx, np.nan)
                if pd.notna(p):
                    per_axis[ax].append(float(p))
                    a["metric"][c].append(float(p))
                    a["values"][c].append(float(row[c]))
            for ax in AXES:
                a["axis"][ax].append(float(np.mean(per_axis[ax])) if per_axis[ax] else None)
            sp = speed.get(idx, np.nan)
            a["axis"]["speed"].append(float(sp) if pd.notna(sp) else None)
    out: Dict[str, Dict[str, Any]] = {}
    for algo, a in acc.items():
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return float(np.mean(xs)) if xs else None
        out[algo] = {
            "axis_scores": {ax: mean(a["axis"][ax]) for ax in ALL_AXES},
            "metric_scores": {c: mean(v) for c, v in a["metric"].items() if v},
            "metric_values": {c: mean(v) for c, v in a["values"].items() if v},
            "n_runs": a["n_runs"], "n_ok": a["n_ok"], "n_failed": a["n_failed"], "n_fallback": a["n_fallback"],
            "n_seeds": len(a["seeds"]),
            "failure_rate": a["n_failed"] / a["n_runs"] if a["n_runs"] else None,
            "time_s_median": float(np.median(a["times"])) if a["times"] else None,
        }
    return out


# ---------------------------------------------------------------- meta-feature e vicinanza
def _transform(feat: str, v: float) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    return float(v) if feat in NO_LOG else math.log1p(max(float(v), 0.0))


def meta_features(modality: str, level: str) -> List[str]:
    return META_FEATURES.get((modality, level if modality == "graph" else ""), [])


def meta_vector(kb: KB, modality: str, level: str, dataset: str) -> Optional[Dict[str, float]]:
    """Meta-feature grezze del dataset dal KB; None se non presenti."""
    if kb.meta.empty:
        return None
    m = kb.meta[(kb.meta["modality"] == modality) & (kb.meta["_ds"] == canon(dataset))]
    if modality == "graph" and level:
        m = m[m["level"] == level]
    if m.empty:
        return None
    r = m.iloc[0]
    return {f: (float(r[f]) if f in r and pd.notna(r[f]) else float("nan")) for f in meta_features(modality, str(r["level"]))}


def knn_neighbors(kb: KB, modality: str, level: str, query: Dict[str, float], candidates: Sequence[str],
                  k: int = K_NEIGHBORS) -> List[Dict[str, Any]]:
    """I k dataset candidati piu' vicini alla query (meta-feature standardizzate, log sulle dimensioni).

    La standardizzazione usa media e deviazione standard dei candidati; la distanza euclidea e' calcolata sulle
    sole meta-feature disponibili per la query (i valori mancanti nei candidati valgono la media, cioe' 0)."""
    feats = [f for f in meta_features(modality, level) if f in query and not math.isnan(_transform(f, query[f]))]
    if not feats:
        return []
    rows = []
    for c in candidates:
        v = meta_vector(kb, modality, level, c)
        if v is not None:
            rows.append((c, v))
    if not rows:
        return []
    X = np.array([[_transform(f, v.get(f, float("nan"))) for f in feats] for _, v in rows], dtype=float)
    mu = np.nanmean(np.where(np.isnan(X), np.nan, X), axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.nanstd(X, axis=0)
    sd = np.where((sd < 1e-9) | np.isnan(sd), 1.0, sd)
    Z = (np.where(np.isnan(X), mu, X) - mu) / sd
    q = (np.array([_transform(f, query[f]) for f in feats]) - mu) / sd
    dist = np.sqrt(((Z - q) ** 2).sum(axis=1))
    order = np.argsort(dist, kind="stable")[: max(1, k)]
    res = []
    for i in order:
        d = float(dist[i])
        res.append({"dataset": rows[i][0], "distance": d, "weight": 1.0 / max(d, 1e-3)})
    tot = sum(r["weight"] for r in res)
    for r in res:
        r["weight"] /= tot
    return res


def combine(per_dataset: List[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]) -> Dict[str, Dict[str, Any]]:
    """Media pesata dei punteggi di piu' dataset. per_dataset = [({dataset, weight, distance}, {algo: info}), ...].
    Per ogni algoritmo la media e' sui soli dataset dove e' misurato (pesi rinormalizzati)."""
    algos = sorted({a for _, s in per_dataset for a in s})
    out: Dict[str, Dict[str, Any]] = {}
    for a in algos:
        parts = [(m, s[a]) for m, s in per_dataset if a in s]

        def wmean(getter):
            num = den = 0.0
            for m, info in parts:
                v = getter(info)
                if v is not None:
                    num += m["weight"] * v
                    den += m["weight"]
            return num / den if den else None
        axes = {ax: wmean(lambda i, ax=ax: i["axis_scores"].get(ax)) for ax in ALL_AXES}
        mkeys = sorted({c for _, i in parts for c in i["metric_scores"]})
        n_runs = sum(i["n_runs"] for _, i in parts)
        out[a] = {
            "axis_scores": axes,
            "metric_scores": {c: wmean(lambda i, c=c: i["metric_scores"].get(c)) for c in mkeys},
            "metric_values": ({c: parts[0][1]["metric_values"].get(c) for c in mkeys} if len(parts) == 1 else {}),
            "n_runs": n_runs, "n_ok": sum(i["n_ok"] for _, i in parts), "n_failed": sum(i["n_failed"] for _, i in parts),
            "n_fallback": sum(i["n_fallback"] for _, i in parts),
            "n_seeds": max(i["n_seeds"] for _, i in parts),
            "failure_rate": (sum(i["n_failed"] for _, i in parts) / n_runs) if n_runs else None,
            "time_s_median": wmean(lambda i: i["time_s_median"]),
            "datasets": [{"dataset": m["dataset"], "distance": m.get("distance"), "weight": m["weight"],
                          "n_runs": i["n_runs"], "n_seeds": i["n_seeds"], "failure_rate": i["failure_rate"],
                          "axis_scores": i["axis_scores"]} for m, i in parts],
            "coverage": f"{len(parts)}/{len(per_dataset)}",
        }
    return out
