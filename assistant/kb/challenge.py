"""Logica della pagina «Sfida» di AugPilot: il visitatore indovina l'algoritmo migliore per un dataset del KB e poi si
confronta con il consiglio che AugPilot avrebbe dato per un dataset nuovo.

Tutti i numeri vengono dal KB (assistant/kb): nessun valore e' inventato. Il calcolo e' lo stesso di
bench/experiments/e1_loo.py (valutazione leave-one-dataset-out del consigliere):
  - classifica "vera": punteggi del dataset D (`scoring.score_dataset`) combinati con i pesi del profilo
    (`recommender._weights_for`, `recommender._score_candidate`);
  - consiglio: D e' tolto dal KB; i k dataset piu' vicini per meta-feature (`scoring.knn_neighbors`) vengono combinati
    (`scoring.combine`), cioe' cio' che il consigliere fa per un dataset nuovo. Si confrontano solo gli algoritmi
    misurati su D (come in e1_loo.evaluate_ranking).
Regret normalizzato (0 = scelta ottima, 1 = la peggiore): (massimo vero - vero della scelta) / (massimo - minimo veri).

Casi in cui non si mostra un consiglio (lo dichiara il risultato, `status` e `message`):
  too_few_algorithms  meno di MIN_ALGORITHMS algoritmi misurati sul dataset: niente sfida;
  no_advice           dataset senza meta-feature, nessun altro dataset con meta-feature, o meno di MIN_ALGORITHMS
                      algoritmi in comune tra D e i vicini: si mostrano solo la scelta e la classifica misurata.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from . import scoring, store

MODALITIES = ("tabular", "ts", "graph")
LEVELS = ("graph", "node")
MIN_ALGORITHMS = 3
TOP_N = 10
TOL = 1e-9


class ChallengeError(ValueError):
    """Richiesta non valida o dati assenti. `status` e' il codice HTTP suggerito (400 o 404)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def get_kb(kb_dir: Optional[str] = None) -> Optional[store.KB]:
    return store.load_kb(kb_dir)


def _profiles() -> Dict[str, Dict[str, float]]:
    from assistant.recommender import PROFILES
    return {k: dict(v) for k, v in PROFILES.items()}


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return round(v, 6) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def totals(scores: Dict[str, Dict[str, Any]], profile: str):
    """Punteggio totale e contributi per asse di ogni algoritmo con i pesi del profilo (come e1_loo.totals).
    Restituisce ({algo: score}, {algo: {asse: contributo}}, pesi usati)."""
    from assistant.recommender import _score_candidate, _weights_for
    avail = [ax for ax in scoring.ALL_AXES
             if any((s["axis_scores"].get(ax) is not None) for s in scores.values())]
    w = _weights_for(profile, avail)
    tot, contrib = {}, {}
    for a, s in scores.items():
        tot[a], contrib[a] = _score_candidate(s, w)
    return tot, contrib, w


def _rank(values: Dict[str, float]) -> Dict[str, int]:
    """Posizione 1..n con pari merito (stessa posizione per punteggi uguali entro TOL)."""
    return {a: 1 + sum(1 for b in values.values() if b > v + TOL) for a, v in values.items()}


def _order(values: Dict[str, float]) -> List[str]:
    return sorted(values, key=lambda a: (-values[a], a))


def regret(values: Dict[str, float], algo: str) -> float:
    """Regret normalizzato di `algo` rispetto a `values` (punteggi veri)."""
    vmax, vmin = max(values.values()), min(values.values())
    span = vmax - vmin
    return float((vmax - values[algo]) / span) if span > 1e-12 else 0.0


def chance_regret(values: Dict[str, float]) -> float:
    """Regret atteso di una scelta a caso (come e1_loo.random_baseline)."""
    return float(np.mean([regret(values, a) for a in values]))


def _rows_for(kb: store.KB, modality: str, level: Optional[str], dataset: str):
    """(righe del dataset, righe della modalita'/livello, livello risolto)."""
    if modality not in MODALITIES:
        raise ChallengeError(f"modality must be one of {list(MODALITIES)}")
    lv = (level or "") if modality == "graph" else ""
    if lv not in ("",) + LEVELS:
        raise ChallengeError("level must be node or graph")
    cn = store.canon(dataset or "")
    if not cn:
        raise ChallengeError("missing dataset")
    if modality == "graph" and not lv:
        d = kb.rows("graph")
        found = sorted(d.loc[d["_ds"] == cn, "level"].unique())
        if len(found) > 1:
            raise ChallengeError("for graphs, give the level (graph or node)")
        lv = found[0] if found else ""
    rows = kb.rows(modality, lv or None)
    d = rows[rows["_ds"] == cn]
    if d.empty:
        raise ChallengeError(f"dataset '{dataset}' not in the KB for {modality}"
                             + (f" ({lv})" if lv else ""), 404)
    return d, rows, lv


def _no_kb() -> ChallengeError:
    return ChallengeError("KB not built: run python -m assistant.kb.build_kb", 404)


# ---------------------------------------------------------------- elenco dei dataset
def list_datasets(kb: Optional[store.KB]) -> Dict[str, Any]:
    """Dataset del KB per modalita' e livello, con il numero di algoritmi misurati e se la sfida e' possibile."""
    out: Dict[str, Any] = {"available": kb is not None, "min_algorithms": MIN_ALGORITHMS, "k": scoring.K_NEIGHBORS,
                           "profiles": _profiles(), "datasets": []}
    if kb is None:
        out["message"] = "KB not built: run python -m assistant.kb.build_kb"
        return out
    for mod in MODALITIES:
        rows = kb.rows(mod)
        if rows.empty:
            continue
        for lv, g_lv in rows.groupby("level", sort=True):
            lv = str(lv) if mod == "graph" else ""
            names = {c: grp for c, grp in g_lv.groupby("_ds", sort=True)}
            with_meta = [c for c in names if scoring.meta_vector(kb, mod, lv, c) is not None]
            for c, grp in names.items():
                n_algo = int(grp["_algo"].nunique())
                has_meta = c in with_meta
                others_meta = [o for o in with_meta if o != c]
                if n_algo < MIN_ALGORITHMS:
                    note = f"only {n_algo} algorithms measured (at least {MIN_ALGORITHMS} needed)"
                elif not has_meta:
                    note = "no meta-features in the KB: AugPilot cannot estimate by similarity"
                elif not others_meta:
                    note = "no other dataset with meta-features: there are no neighbors"
                else:
                    note = ""
                out["datasets"].append({
                    "modality": mod, "level": lv, "dataset": str(grp["dataset"].iloc[0]),
                    "n_algorithms": n_algo, "n_runs": int(len(grp)),
                    "playable": n_algo >= MIN_ALGORITHMS, "advice_possible": n_algo >= MIN_ALGORITHMS and has_meta and bool(others_meta),
                    "note": note})
    out["datasets"].sort(key=lambda r: (MODALITIES.index(r["modality"]), r["level"], r["dataset"].lower()))
    return out


def candidates(kb: Optional[store.KB], modality: str, dataset: str, level: Optional[str] = None) -> Dict[str, Any]:
    """Algoritmi misurati sul dataset (l'elenco da cui il visitatore sceglie). Nessun punteggio: non anticipa la risposta."""
    if kb is None:
        raise _no_kb()
    d, _rows, lv = _rows_for(kb, modality, level, dataset)
    algos = []
    for a, g in d.groupby("_algo", sort=True):
        algos.append({"algorithm": a, "n_runs": int(len(g)), "n_failed": int((g["status"] == "FAILED").sum())})
    return {"modality": modality, "level": lv, "dataset": str(d["dataset"].iloc[0]), "n_algorithms": len(algos),
            "playable": len(algos) >= MIN_ALGORITHMS, "min_algorithms": MIN_ALGORITHMS, "algorithms": algos}


# ---------------------------------------------------------------- la sfida
def _entry(algo: str, info: Dict[str, Any], score: float, contrib: Dict[str, float], rank: int) -> Dict[str, Any]:
    return {"algorithm": algo, "rank": rank, "score": _f(score), "contributions": {k: _f(v) for k, v in contrib.items()},
            "axis_scores": {ax: _f(v) for ax, v in info["axis_scores"].items()}, "time_s_median": _f(info["time_s_median"]),
            "n_runs": info["n_runs"], "n_failed": info["n_failed"], "n_fallback": info["n_fallback"],
            "failure_rate": _f(info["failure_rate"])}


def reveal(kb: Optional[store.KB], modality: str, dataset: str, profile: str, guess: str,
           level: Optional[str] = None, k: int = scoring.K_NEIGHBORS, top_n: int = TOP_N) -> Dict[str, Any]:
    """Risultato della sfida: scelta del visitatore, consiglio leave-one-out di AugPilot, classifica misurata, regret."""
    if kb is None:
        raise _no_kb()
    profiles = _profiles()
    if profile not in profiles:
        raise ChallengeError(f"unknown profile: {profile}. Valid: {list(profiles)}")
    d, rows, lv = _rows_for(kb, modality, level, dataset)
    ds_name = str(d["dataset"].iloc[0])
    cn = store.canon(ds_name)
    base: Dict[str, Any] = {"modality": modality, "level": lv, "dataset": ds_name, "profile": profile,
                            "weights_declared": profiles[profile], "k": k, "min_algorithms": MIN_ALGORITHMS}
    true_scores = scoring.score_dataset(d, modality)
    if len(true_scores) < MIN_ALGORITHMS:
        return {**base, "status": "too_few_algorithms", "n_algorithms": len(true_scores),
                "message": f"Only {len(true_scores)} algorithms are measured on dataset '{ds_name}': at least "
                           f"{MIN_ALGORITHMS} are needed for the challenge to make sense. Choose another dataset."}
    guess = store.canon_algo(guess or "")
    if guess not in true_scores:
        raise ChallengeError(f"'{guess}' is not among the algorithms measured on {ds_name}")

    true_tot, true_contrib, true_w = totals(true_scores, profile)
    true_rank = _rank(true_tot)
    order = _order(true_tot)
    n = len(order)
    best = order[0]

    def entry(a: str) -> Dict[str, Any]:
        return _entry(a, true_scores[a], true_tot[a], true_contrib[a], true_rank[a])

    ranking = [entry(a) for a in order[:top_n]]
    guess_e = {**entry(guess), "regret": _f(regret(true_tot, guess))}
    res: Dict[str, Any] = {**base, "status": "ok", "n_algorithms": n, "weights": {a: _f(v) for a, v in true_w.items()},
                           "best": best, "best_score": _f(true_tot[best]), "ranking": ranking, "guess": guess_e,
                           "chance_regret": _f(chance_regret(true_tot)), "guess_is_best": guess_e["rank"] == 1,
                           "n_ties_at_top": sum(1 for a in order if true_rank[a] == 1)}

    # ---- consiglio per un dataset nuovo: D tolto dal KB, k vicini per meta-feature
    other_names = sorted({c for c in rows["_ds"].unique() if c != cn})
    q = scoring.meta_vector(kb, modality, lv, cn)
    reason = None
    if q is None:
        reason = "The dataset has no meta-features in the KB: AugPilot cannot estimate by similarity, so no advice is shown."
    elif not other_names:
        reason = "The KB has no other dataset for this modality: there are no neighbors, so no advice is shown."
    neigh = scoring.knn_neighbors(kb, modality, lv, q, other_names, k=k) if reason is None else []
    if reason is None and not neigh:
        reason = ("No other dataset with comparable meta-features: there are no neighbors, "
                  "so no advice is shown.")
    pred_ranked: List[str] = []
    if reason is None:
        per_ds = {nb["dataset"]: scoring.score_dataset(rows[rows["_ds"] == nb["dataset"]], modality) for nb in neigh}
        pred_scores = scoring.combine([(nb, per_ds[nb["dataset"]]) for nb in neigh])
        pred_tot, _pc, pred_w = totals(pred_scores, profile)
        common = sorted(set(pred_tot) & set(true_tot))
        if len(common) < MIN_ALGORITHMS:
            reason = (f"The neighboring datasets have only {len(common)} measured algorithms in common with '{ds_name}' "
                      f"(at least {MIN_ALGORITHMS} needed): no advice is shown.")
        else:
            pred_c = {a: pred_tot[a] for a in common}
            pred_ranked = _order(pred_c)
    if reason is not None:
        return {**res, "status": "no_advice", "message": reason, "advice": None, "outcome": None}

    pick = pred_ranked[0]
    pr = _rank(pred_c)
    shown_name = {c: str(rows.loc[rows["_ds"] == c, "dataset"].iloc[0]) for c in per_ds}
    advice = {**entry(pick), "regret": _f(regret(true_tot, pick)), "predicted_score": _f(pred_c[pick]),
              "predicted_rank": pr[pick], "n_compared": len(common),
              "top": [{"algorithm": a, "predicted_score": _f(pred_c[a]), "true_rank": true_rank[a]} for a in pred_ranked[:3]],
              "neighbors": [{"dataset": shown_name[nb["dataset"]], "distance": _f(nb["distance"]), "weight": _f(nb["weight"])}
                            for nb in neigh],
              "n_not_comparable": n - len(common),
              "weights": {a: _f(v) for a, v in pred_w.items()}}
    # il confronto usa il regret su tutti gli algoritmi misurati su D (la stessa scala del visitatore); `regret_common`
    # e' quello di e1_loo, calcolato sui soli algoritmi in comune con i vicini
    advice["regret_common"] = _f(regret({a: true_tot[a] for a in common}, pick))
    gv = regret(true_tot, guess)
    av = regret(true_tot, pick)
    outcome = "tie" if abs(gv - av) <= TOL else ("visitor" if gv < av else "augpilot")
    labels = {"tie": "Tie: same regret.", "visitor": "You did better than AugPilot.",
              "augpilot": "AugPilot did better than you."}
    if outcome == "tie" and guess == pick:
        labels["tie"] = "Tie: you picked the same algorithm."
    res.update({"status": "ok", "advice": advice, "advice_matches_best": true_rank[pick] == 1,
                "outcome": {"result": outcome, "label": labels[outcome], "visitor_regret": _f(gv),
                            "augpilot_regret": _f(av), "chance_regret": res["chance_regret"]}})
    return res
