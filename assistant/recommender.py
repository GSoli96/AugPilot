"""Deterministic recommender: ranks algorithms ONLY from the catalog and measured results.

Rules (also printed in the result, field `rules`):
R1 An algorithm is ranked only if measured results exist (CSV). Without evidence it goes to `unranked`
   with the reason: no invented numbers, no "gut feeling" ordering.
R2 Status: "executable" (at least one successful real run), "wired" (connected, never tried), "described"
   (catalog only). "described" algorithms are never recommended as executable; with
   executable_only=true the "wired" ones are excluded too.
R3 Tabular, score = mean of the rank scores (1 = best, 0 = worst among the algorithms measured on the same
   dataset) on fidelity (JS, KS, WD, pMSE, MMD: lower is better) and diversity (Precision, Recall,
   Coverage: higher is better). Weights from the priority: fidelity 0.7/0.3, diversity 0.3/0.7, balanced 0.5/0.5.
R4 If the given dataset has measurements only those are used; otherwise all measured datasets, with a warning.
R5 Hard filters: fairness/privacy/constraints require the matching tag (derived from the description and paper
   title); time_budget_s excludes algorithms with a higher measured median time (unmeasured time = warning, not
   exclusion); gpu=false excludes only those declaring needs_gpu=true in the catalog (if unknown: warning).
R6 ts/graph: ordering by success rate then measured median time; quality metrics are shown without ordering
   them because their direction has not been verified. The `task` filter uses AlgoSpec.task.
R7 If n_rows exceeds the maximum n_orig measured for the algorithm, warning: scalability is not measured.
R8 If the KB exists (assistant/kb/benchmark.*, built from bench/results/**/*.jsonl) and covers the modality, the score
   comes from the KB and not from R3/R6: for each metric with a direction (assistant/kb/metrics_spec.py) a 0..1
   percentile among the algorithms of the same dataset (1 = best), mean over seeds, mean per axis (fidelity,
   diversity, privacy, utility) plus speed (time percentile); a FAILED run counts 0. Score = sum of axis weight x axis,
   with the profile weights (PROFILES) renormalized over the axes that have measurements; contributions and
   axis_scores are in the result.
R9 Dataset present in the KB: its runs are used. New dataset: 1/distance weighted mean of the k=3 nearest datasets
   of the same modality (and level, for graphs) by standardized meta-features; without meta-features, mean over all.
R10 The task filter is tolerant: generation accepts generation and oversampling; view only if requested.
R11 Without a KB (or without rows for the modality) fall back to R1-R7 and the result declares evidence="nessuna".
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from . import catalog, config, evidence
from .kb import scoring, store
from .kb.metrics_spec import AXES

RULES = [l.strip() for l in __doc__.split("\n") if re.match(r"R\d+ ", l)]
# Pesi (fidelity, diversity) del consigliere precedente (R3): restano per il ripiego senza KB.
WEIGHTS = {"fidelity": (0.7, 0.3), "diversity": (0.3, 0.7), "balanced": (0.5, 0.5)}
# Profili di priorita' del consigliere basato sul KB: peso per asse (somma 1). Gli assi senza misure nel risultato
# vengono tolti e i pesi rinormalizzati. I nomi fidelity/diversity/balanced restano compatibili con WEIGHTS.
PROFILES: Dict[str, Dict[str, float]] = {
    "balanced":  {"utility": 0.30, "fidelity": 0.25, "diversity": 0.20, "privacy": 0.15, "speed": 0.10},
    "utility":   {"utility": 0.55, "fidelity": 0.15, "diversity": 0.10, "privacy": 0.10, "speed": 0.10},
    "fidelity":  {"utility": 0.15, "fidelity": 0.55, "diversity": 0.15, "privacy": 0.05, "speed": 0.10},
    "diversity": {"utility": 0.15, "fidelity": 0.15, "diversity": 0.55, "privacy": 0.05, "speed": 0.10},
    "privacy":   {"utility": 0.15, "fidelity": 0.15, "diversity": 0.05, "privacy": 0.55, "speed": 0.10},
    "speed":     {"utility": 0.15, "fidelity": 0.15, "diversity": 0.10, "privacy": 0.05, "speed": 0.55},
}
PRIORITIES = tuple(PROFILES)
PERMISSIVE_LICENSE = r"^(-|n/a\b|mit\b|apache|bsd|isc\b)"   # licenze del catalogo che non generano avviso
TIME_WARN_S = 300.0   # sopra questo tempo mediano misurato si segnala "tempo alto"


def _rank_scores(df: pd.DataFrame, cols: List[str], lower_better: bool) -> pd.Series:
    """Per riga: media sui metrici del punteggio di rango in [0,1] (1 = migliore). NaN se nessun metrico."""
    parts = []
    for c in cols:
        if c not in df or df[c].notna().sum() == 0:
            continue
        n = df[c].notna().sum()
        r = df[c].rank(method="average", ascending=lower_better)  # 1 = migliore
        parts.append(1.0 if n == 1 else 1 - (r - 1) / (n - 1))
        parts[-1] = parts[-1].where(df[c].notna())
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series([float("nan")] * len(df), index=df.index)


def _recommend_tabular(cat, p) -> Dict[str, Any]:
    m = evidence.tabular_metrics()
    rt = evidence.tabular_runtime()
    notes: List[str] = ["Times and memory in runtime_results/ come from earlier runs on a different machine "
                        "and on small datasets."]
    ds = evidence.canon_ds(p["dataset"]) if p.get("dataset") else None
    scope = "all measured datasets"
    mm = m
    if ds and not m.empty and (m["ds"] == ds).any():
        mm = m[m["ds"] == ds]
        scope = f"dataset {ds}"
    elif ds:
        notes.append(f"No measurements for dataset '{ds}': the score uses other datasets (R4).")
    scores: Dict[str, Dict[str, Any]] = {}
    if not mm.empty:
        wf, wd = WEIGHTS.get(p.get("priority") or "balanced", WEIGHTS["balanced"])
        for d, g in mm.groupby("ds"):
            g = g.copy()
            g["fid"] = _rank_scores(g, evidence.FIDELITY_LOWER_BETTER, True)
            g["div"] = _rank_scores(g, evidence.DIVERSITY_HIGHER_BETTER, False)
            for _, r in g.iterrows():
                s = scores.setdefault(r["algo"], {"fid": [], "div": [], "ds": [], "n_cmp": []})
                s["fid"].append(r["fid"]); s["div"].append(r["div"]); s["ds"].append(d); s["n_cmp"].append(len(g))
    ranking, unranked, excluded = [], [], []
    for a in cat["algorithms"]:
        k = a["key"]
        if a["status"] == "described":
            unranked.append({"algorithm": k, "status": "described", "reason": "described only: not executable"})
            continue
        if p.get("executable_only") and a["status"] != "executable":
            excluded.append({"algorithm": k, "status": a["status"], "reason": "executable_only: never run successfully"})
            continue
        bad = [t for t in ("fairness", "privacy", "constraints") if p.get(f"need_{t}") and t not in a["tags"]]
        if bad:
            excluded.append({"algorithm": k, "status": a["status"], "reason": f"missing requirement: {', '.join(bad)}"})
            continue
        if p.get("gpu") is False and a.get("needs_gpu") is True:
            excluded.append({"algorithm": k, "status": a["status"], "reason": "requires GPU (catalog) but gpu=false"})
            continue
        rows = rt[(rt["algo"] == k) & (rt["status"] == "SUCCESS")] if not rt.empty else rt
        warn: List[str] = []
        tmed = None
        if len(rows):
            tmed = float(rows["total_time_s"].median())
            if p.get("time_budget_s") and tmed > p["time_budget_s"]:
                excluded.append({"algorithm": k, "status": a["status"],
                                 "reason": f"measured median time {tmed:.1f}s > budget {p['time_budget_s']}s "
                                           f"(source: {rows['source'].iloc[0]} and others)"})
                continue
            if p.get("n_rows") and p["n_rows"] > rows["n_orig"].max():
                warn.append(f"measured only up to n_orig={int(rows['n_orig'].max())} rows; with {p['n_rows']} "
                            "scalability is not measured (R7)")
        else:
            warn.append("time not measured")
            if p.get("time_budget_s"):
                warn.append("the time budget cannot be verified")
        if p.get("gpu") is False and a.get("needs_gpu") is None:
            warn.append("GPU requirement unknown in the catalog")
        if k == "kamino" and ds:
            dc = config.ROOT / "Algorithms" / "kamino-master" / "testdata" / ds / f"{ds}_dcs.out"
            if not dc.exists() and not list((config.ROOT / "Algorithms" / "kamino-master" / "testdata").glob(f"{ds}*/*_dcs.out")):
                warn.append("no Denial Constraints file found for this dataset")
        s = scores.get(k)
        if not s:
            unranked.append({"algorithm": k, "status": a["status"], "warnings": warn,
                             "reason": "no fidelity/diversity result measured (R1)"})
            continue
        wf, wd = WEIGHTS.get(p.get("priority") or "balanced", WEIGHTS["balanced"])
        fid = pd.Series(s["fid"]).mean(); dv = pd.Series(s["div"]).mean()
        parts = [(w, v) for w, v in ((wf, fid), (wd, dv)) if v == v]
        score = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
        ev = [{"metric": "fidelity_rank_score", "value": evidence.fnum(fid),
               "source": "evaluation_metrics/fidelity_all_datasets.csv", "note": f"mean rank over {len(s['ds'])} datasets: {sorted(set(s['ds']))}"},
              {"metric": "diversity_rank_score", "value": evidence.fnum(dv),
               "source": "evaluation_metrics/diversity_all_datasets.csv", "note": "1 = best"}]
        if tmed is not None:
            ev.append({"metric": "total_time_s_median", "value": evidence.fnum(tmed),
                       "source": ", ".join(sorted(set(rows["source"]))[:3]), "note": f"{len(rows)} successful runs"})
        ranking.append({"algorithm": k, "name": a["name"], "status": a["status"], "score": evidence.fnum(score),
                        "reasons": [f"rank score (R3, weights {wf}/{wd}) on {scope}; compared with "
                                    f"at most {max(s['n_cmp'])} algorithms"] +
                                   [f"requirement met: {t}" for t in ("fairness", "privacy", "constraints") if p.get(f"need_{t}")],
                        "evidence": ev, "warnings": warn})
    ranking.sort(key=lambda r: -(r["score"] if r["score"] is not None else -1))
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
    return {"scope": scope, "ranking": ranking[:p["top_k"]], "n_ranked_total": len(ranking),
            "unranked": unranked, "excluded": excluded, "notes": notes}


def task_ok(requested: Optional[str], spec_task: Optional[str]) -> bool:
    """Filtro di task tollerante (R10): generation accetta generation e oversampling; oversampling solo oversampling;
    view solo view. Il task del catalogo puo' essere composto ("generation/oversampling ..."): basta un termine."""
    if not requested:
        return True
    have = set(re.findall(r"[a-z]+", str(spec_task or "").lower()))
    accepted = {"generation": {"generation", "oversampling"}}.get(requested.lower(), {requested.lower()})
    return bool(have & accepted)


def _dataset_level(modality, dataset):
    """Grafi: 'node' (un solo grafo grande) o 'graph' (molti grafi), letto dal dataset; None se ignoto."""
    if modality != "graph" or not dataset:
        return None
    try:
        from . import datasets_tools
        kind = datasets_tools.analyze_dataset(modality, dataset).get("kind")
        return kind if kind in ("node", "graph") else None
    except Exception:  # noqa: BLE001
        return None


def _recommend_modality(cat, p, modality) -> Dict[str, Any]:
    rt = evidence.modality_runtime(modality)
    ranking, unranked, excluded = [], [], []
    ds_level = _dataset_level(modality, p.get("dataset"))
    for a in cat["algorithms"]:
        k = a["key"]
        if ds_level and a.get("level") and a["level"] != ds_level and a["status"] != "described":
            excluded.append({"algorithm": k, "status": a["status"],
                             "reason": f"level '{a['level']}' not suitable for a dataset at level '{ds_level}'"})
            continue
        if a["status"] == "described":
            unranked.append({"algorithm": k, "status": "described", "reason": "described only: not executable"})
            continue
        if p.get("executable_only") and a["status"] != "executable":
            excluded.append({"algorithm": k, "status": a["status"], "reason": "executable_only: never run successfully"})
            continue
        if not task_ok(p.get("task"), a["task"]):
            excluded.append({"algorithm": k, "status": a["status"], "reason": f"catalog task '{a['task']}' != '{p['task']}'"})
            continue
        if p.get("gpu") is False and a.get("needs_gpu"):
            excluded.append({"algorithm": k, "status": a["status"], "reason": "requires GPU but gpu=false"})
            continue
        rows = rt[rt["algo"] == k] if (not rt.empty and "algo" in rt) else rt
        if p.get("dataset") and len(rows) and "ds" in rows and (rows["ds"] == p["dataset"].lower()).any():
            rows = rows[rows["ds"] == p["dataset"].lower()]
        if not len(rows):
            unranked.append({"algorithm": k, "status": a["status"], "reason": "no measured results (R1)"})
            continue
        ok = rows[rows["status"] == "SUCCESS"] if "status" in rows else rows
        rate = len(ok) / len(rows)
        tcol = "time_s" if "time_s" in rows else None
        tmed = float(ok[tcol].median()) if (tcol and len(ok)) else None
        if p.get("time_budget_s") and tmed is not None and tmed > p["time_budget_s"]:
            excluded.append({"algorithm": k, "status": a["status"], "reason": f"median time {tmed:.1f}s > budget"})
            continue
        src = ", ".join(sorted(set(rows["source"]))[:3])
        ev = [{"metric": "success_rate", "value": evidence.fnum(rate), "source": src, "note": f"{len(rows)} runs"}]
        if tmed is not None:
            ev.append({"metric": "time_s_median", "value": evidence.fnum(tmed), "source": src, "note": "successful runs"})
        for c in rows.columns:
            if c in ("source", "algo", "ds", "algorithm", "dataset", "status", "time_s", "error_message", "modality") or not pd.api.types.is_numeric_dtype(rows[c]):
                continue
            v = evidence.fnum(ok[c].mean()) if len(ok) else None
            if v is not None:
                ev.append({"metric": c, "value": v, "source": src, "note": "mean; direction not verified, not used for ordering"})
        ranking.append({"algorithm": k, "name": a["name"], "status": a["status"], "_k": (-rate, tmed if tmed is not None else 1e18),
                        "reasons": ["ordered by measured success rate and median time (R6)"], "evidence": ev,
                        "warnings": [] if tmed is not None else ["time not measured"]})
    ranking.sort(key=lambda r: r["_k"])
    for i, r in enumerate(ranking, 1):
        r["rank"] = i; r.pop("_k")
    return {"scope": p.get("dataset") or "all measured datasets", "ranking": ranking[:p["top_k"]],
            "n_ranked_total": len(ranking), "unranked": unranked, "excluded": excluded, "notes": []}


# ---------------------------------------------------------------- consigliere basato sul KB (R8-R10)
def _kb_query_vector(kb, modality, level, dataset, p):
    """Meta-feature grezze di un dataset senza run: KB (dataset_meta), poi analisi del file, poi parametri dell'utente."""
    v = scoring.meta_vector(kb, modality, level or "", dataset) if dataset else None
    if v and any(x == x for x in v.values()):
        return v, "dataset_meta.csv"
    q: Dict[str, Any] = {}
    src = "user parameters"
    if dataset:
        try:
            from . import datasets_tools
            a = datasets_tools.analyze_dataset(modality, dataset)
            src = "dataset file analysis"
            if modality == "tabular":
                q = {"n_rows": a.get("rows"), "n_cols": (a.get("columns") or 0) - (1 if a.get("label_column") else 0),
                     "ir": a.get("imbalance_ratio")}
                cc = a.get("class_counts") or {}
                if cc:
                    q["n_min"] = min(cc.values())
            elif modality == "ts":
                q = {"n_series": a.get("series"), "length_T": a.get("length"), "dims_D": a.get("channels"),
                     "has_labels": 0.0 if a.get("label_column", True) is None else 1.0, "ir": a.get("imbalance_ratio")}
            elif a.get("kind") == "node":
                q = {"avg_nodes": a.get("nodes"), "avg_edges": a.get("edges"), "n_features": a.get("features"),
                     "n_classes": a.get("n_classes"), "ir": a.get("imbalance_ratio")}
            else:
                q = {"n_graphs": a.get("graphs"), "avg_nodes": a.get("mean_nodes"), "avg_edges": a.get("mean_edges"),
                     "n_features": a.get("features"), "n_classes": a.get("n_classes"), "ir": a.get("imbalance_ratio")}
        except Exception:  # noqa: BLE001
            q = {}
    if modality == "tabular":
        for k, key in (("n_rows", "n_rows"), ("n_cols", "n_cols"), ("ir", "imbalance_ratio")):
            if p.get(key) is not None and q.get(k) is None:
                q[k] = p[key]
    q = {k: float(x) for k, x in q.items() if x is not None and x == x}
    return (q or None), src


def _weights_for(priority: str, available: List[str]) -> Dict[str, float]:
    w = {ax: PROFILES[priority][ax] for ax in available}
    tot = sum(w.values())
    return {ax: v / tot for ax, v in w.items()} if tot else {}


def _score_candidate(info: Dict[str, Any], weights: Dict[str, float]):
    contrib = {ax: round(w * (info["axis_scores"].get(ax) or 0.0), 6) for ax, w in weights.items()}
    return round(sum(contrib.values()), 6), contrib


def _recommend_kb(cat, p, kb, modality, detail: bool) -> Optional[Dict[str, Any]]:
    ds = p.get("dataset")
    level = None
    if modality == "graph" and ds:
        m = kb.meta[(kb.meta["modality"] == "graph") & (kb.meta["_ds"] == store.canon(ds))] if not kb.meta.empty else None
        level = (str(m.iloc[0]["level"]) or None) if m is not None and len(m) else _dataset_level(modality, ds)
    rows = kb.rows(modality, level)
    if len(rows) == 0 or not rows["status"].isin(scoring.OK_STATUS).any():
        return None
    notes: List[str] = []
    ds_names = sorted(rows["dataset"].unique(), key=str.lower)
    by_canon = {store.canon(d): d for d in ds_names}
    used: List[Dict[str, Any]] = []
    if ds and store.canon(ds) in by_canon:
        mode, scope = "dataset", f"dataset {by_canon[store.canon(ds)]} (KB)"
        used = [{"dataset": by_canon[store.canon(ds)], "weight": 1.0, "distance": 0.0}]
    elif ds:
        q, qsrc = _kb_query_vector(kb, modality, level, ds, p)
        nb = (scoring.knn_neighbors(kb, modality, level or "", q, ds_names, scoring.K_NEIGHBORS) if q else [])
        if nb:
            mode, used = "knn", nb
            scope = (f"dataset {ds} not in the KB: estimate from the {len(nb)} nearest datasets "
                     f"({', '.join(n['dataset'] for n in nb)})")
            notes.append(f"Meta-features of dataset '{ds}' from: {qsrc}.")
        else:
            mode = "all"
            scope = "all KB datasets for the modality (dataset meta-features not available)"
            notes.append(f"Dataset '{ds}' not in the KB and without meta-features: mean score over all KB datasets.")
    else:
        mode, scope = "all", "all KB datasets for the modality"
    if mode == "all":
        used = [{"dataset": d, "weight": 1.0 / len(ds_names), "distance": None} for d in ds_names]
    per = []
    for m in used:
        per.append((m, scoring.score_dataset(rows[rows["_ds"] == store.canon(m["dataset"])], modality)))
    comb = scoring.combine(per)

    ranking, unranked, excluded = [], [], []
    cand: List[Dict[str, Any]] = []
    for a in cat["algorithms"]:
        k = a["key"]
        if modality == "graph" and level and a.get("level") and a["level"] != level and a["status"] != "described":
            excluded.append({"algorithm": k, "status": a["status"],
                             "reason": f"level '{a['level']}' not suitable for a dataset at level '{level}'"})
            continue
        if a["status"] == "described":
            unranked.append({"algorithm": k, "status": "described", "reason": "described only: not executable"})
            continue
        if p.get("executable_only") and a["status"] != "executable":
            excluded.append({"algorithm": k, "status": a["status"], "reason": "executable_only: never run successfully"})
            continue
        if modality != "tabular" and not task_ok(p.get("task"), a.get("task")):
            excluded.append({"algorithm": k, "status": a["status"], "reason": f"catalog task '{a['task']}' != '{p['task']}'"})
            continue
        bad = [t for t in ("fairness", "privacy", "constraints") if p.get(f"need_{t}") and t not in a["tags"]]
        if bad:
            excluded.append({"algorithm": k, "status": a["status"], "reason": f"missing requirement: {', '.join(bad)}"})
            continue
        if p.get("gpu") is False and a.get("needs_gpu") is True:
            excluded.append({"algorithm": k, "status": a["status"], "reason": "requires GPU (catalog) but gpu=false"})
            continue
        info = comb.get(store.canon_algo(k))
        if info is None:
            unranked.append({"algorithm": k, "status": a["status"],
                             "reason": "no run in the KB for this dataset/modality (R1)"})
            continue
        tmed = info.get("time_s_median")
        if p.get("time_budget_s") and tmed is not None and tmed > p["time_budget_s"]:
            excluded.append({"algorithm": k, "status": a["status"],
                             "reason": f"measured median time {tmed:.1f}s > budget {p['time_budget_s']}s"})
            continue
        cand.append({"a": a, "info": info})
    available = [ax for ax in (*AXES, "speed") if any(c["info"]["axis_scores"].get(ax) is not None for c in cand)]
    weights = _weights_for(p["priority"], available)
    src_file = f"assistant/kb/{kb.source}"
    for c in cand:
        a, info = c["a"], c["info"]
        k = a["key"]
        score, contrib = _score_candidate(info, weights)
        warn: List[str] = []
        if a["status"] != "executable":
            warn.append(f"catalog status '{a['status']}': not yet proven executable in the catalog")
        if info["n_fallback"]:
            warn.append(f"FALLBACK in {info['n_fallback']} of {info['n_runs']} runs: the method used is not the one requested")
        if info["n_failed"]:
            warn.append(f"{info['n_failed']} of {info['n_runs']} runs FAILED (failure rate "
                        f"{info['failure_rate']:.2f}): penalized with score 0")
        missing = [ax for ax in weights if info["axis_scores"].get(ax) is None]
        if missing:
            warn.append(f"no measurements for axes {', '.join(missing)}: contribution 0")
        lic = (a.get("license") or "").strip()
        if lic and not re.search(PERMISSIVE_LICENSE, lic, re.I):
            warn.append(f"license from the catalog: {lic} (check the terms of use)")
        tm = info.get("time_s_median")
        if tm is not None and tm > TIME_WARN_S:
            warn.append(f"high measured median time: {tm:.1f}s")
        if mode == "knn" and len(info["datasets"]) < len(used):
            warn.append(f"measured only on {info['coverage']} of the nearby datasets")
        if k == "kamino" and ds:
            dc = config.ROOT / "Algorithms" / "kamino-master" / "testdata"
            if not list(dc.glob(f"{store.canon(ds)}*/*_dcs.out")):
                warn.append("no Denial Constraints file found for this dataset")
        ev = []
        for dd in info["datasets"]:
            dsc = sum(w * (dd["axis_scores"].get(ax) or 0.0) for ax, w in weights.items())
            ev.append({"metric": "score_dataset", "value": evidence.fnum(dsc), "source": src_file,
                       "note": ("dataset runs" if mode == "dataset" else "KB dataset used as evidence"),
                       "dataset": dd["dataset"], "distance": evidence.fnum(dd["distance"]),
                       "weight": evidence.fnum(dd["weight"]), "n_runs": dd["n_runs"], "n_seeds": dd["n_seeds"],
                       "failure_rate": evidence.fnum(dd["failure_rate"])})
        if tm is not None:
            ev.append({"metric": "time_s_median", "value": evidence.fnum(tm), "source": src_file,
                       "note": f"median of successful runs ({info['n_ok']})"})
        item = {"algorithm": k, "name": a["name"], "status": a["status"], "score": score,
                "axis_scores": {ax: evidence.fnum(info["axis_scores"].get(ax)) for ax in (*AXES, "speed")},
                "contributions": contrib, "evidence": ev, "warnings": warn, "method": "kb:" + mode,
                "time_s_median": evidence.fnum(tm), "n_runs": info["n_runs"], "n_seeds": info["n_seeds"],
                "failure_rate": evidence.fnum(info["failure_rate"]),
                "reasons": [f"score from the KB (profile {p['priority']}) on {scope}; contributions: "
                            + ", ".join(f"{ax} {contrib[ax]:.3f}" for ax in contrib)]
                + [f"requirement met: {t}" for t in ("fairness", "privacy", "constraints") if p.get(f"need_{t}")]}
        if detail:
            item["metric_scores"] = {c_: evidence.fnum(v) for c_, v in info["metric_scores"].items()}
            item["metric_values"] = {c_: evidence.fnum(v) for c_, v in info["metric_values"].items()}
        ranking.append(item)
    ranking.sort(key=lambda r: (-r["score"], r["time_s_median"] if r["time_s_median"] is not None else 1e18, r["algorithm"]))
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
    method = ("Benchmark KB: for each metric with a direction, 0..1 percentile of the algorithm among those measured on the "
              "same dataset (1 = best), mean over seeds; axis score = mean of the axis metrics "
              "(speed = time percentile); FAILED = 0; score = sum of axis_weight x axis_score with the weights of "
              f"profile '{p['priority']}' renormalized over the axes with measurements.")
    if mode == "knn":
        method += (f" New dataset: mean of the scores of the {len(used)} nearest KB datasets (Euclidean distance on "
                   "standardized meta-features, log on sizes), normalized 1/distance weights.")
    if mode == "all":
        method += " Simple mean over the KB datasets."
    return {"scope": scope, "ranking": ranking[:p["top_k"]], "n_ranked_total": len(ranking), "unranked": unranked,
            "excluded": excluded, "notes": notes, "evidence": "kb", "method": method, "profile": p["priority"],
            "weights": {ax: round(w, 6) for ax, w in weights.items()},
            "weights_declared": PROFILES[p["priority"]], "axes_available": available,
            "similar_datasets": ([{"dataset": n["dataset"], "distance": evidence.fnum(n["distance"]),
                                   "weight": evidence.fnum(n["weight"])} for n in used] if mode == "knn" else []),
            "kb": {"file": src_file, "runs_in_modality": int(len(rows)), "datasets_in_modality": len(ds_names),
                   "mode": mode, **({"graph_level": level} if level else {})}}


def recommend(modality: str, task: Optional[str] = None, dataset: Optional[str] = None, n_rows: Optional[int] = None,
              n_cols: Optional[int] = None, imbalance_ratio: Optional[float] = None, gpu: Optional[bool] = None,
              time_budget_s: Optional[float] = None, need_privacy: bool = False, need_fairness: bool = False,
              need_constraints: bool = False, priority: str = "balanced", executable_only: bool = False,
              top_k: int = 5, kb_dir: Optional[str] = None, detail: bool = False) -> Dict[str, Any]:
    """Classifica gli algoritmi. Con un KB che copre la modalita' (R8-R10) il punteggio viene dal KB, altrimenti dai CSV
    storici (R1-R7) e il risultato dichiara `evidence: "nessuna"`. kb_dir: cartella del KB (default assistant/kb o
    $AUGPILOT_KB_DIR); detail=True aggiunge i punteggi per singola metrica (serve alla spiegazione)."""
    cat = catalog.get_catalog(modality)
    p = dict(task=task, dataset=dataset, n_rows=n_rows, n_cols=n_cols, imbalance_ratio=imbalance_ratio, gpu=gpu,
             time_budget_s=time_budget_s, need_privacy=need_privacy, need_fairness=need_fairness,
             need_constraints=need_constraints, priority=priority if priority in PROFILES else "balanced",
             executable_only=executable_only, top_k=max(1, int(top_k or 5)))
    out = None
    kb = store.load_kb(kb_dir)
    if kb is not None:
        out = _recommend_kb(cat, p, kb, modality, detail)
    if out is None:
        out = (_recommend_tabular(cat, p) if modality == "tabular" else _recommend_modality(cat, p, modality))
        out["evidence"] = "nessuna"
        out["method"] = ("fallback without KB: ordering from R1-R7 on the historical CSVs (runtime_results*/, evaluation_metrics/); "
                         "no evidence from the benchmark KB")
        out["notes"].append("Benchmark KB missing or without rows for this modality: the recommendation does not use the KB.")
    out.update({"modality": modality, "inputs": {k: v for k, v in p.items() if v not in (None, False)},
                "rules": RULES, "catalog_warnings": cat["warnings"]})
    if not out["ranking"]:
        out["message"] = ("No measured evidence or no compatible candidate: I cannot rank. "
                          "See `unranked` (candidates without measurements) and `excluded`.")
    if imbalance_ratio and out["evidence"] == "nessuna":
        out["notes"].append("imbalance_ratio and n_cols are recorded but no rule uses them: there are no "
                            "measurements linking these variables to the algorithms.")
    return out
