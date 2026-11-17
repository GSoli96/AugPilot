"""E1 di WP12: valutazione leave-one-dataset-out del consigliere di AugPilot.

Per ogni modalità (e livello, per i grafi) e per ogni dataset D del KB:
  - "verità": i punteggi degli algoritmi misurati su D (stessa funzione del consigliere, `score_dataset`),
    combinati con i pesi del profilo (`_weights_for`, `_score_candidate`);
  - "consiglio": D è tolto dal KB; si prendono i k dataset più vicini per meta-feature (`knn_neighbors`) e si
    combinano i loro punteggi (`combine`), cioè esattamente ciò che il consigliere fa per un dataset nuovo.
Misure (sugli algoritmi misurati sia su D sia sui vicini):
  best_in_top3  1 se il migliore vero di D è tra i primi 3 consigliati
  overlap@3     |top3 consigliati ∩ top3 veri| / 3
  regret@1      (vero max - vero del primo consigliato) / (vero max - vero min), 0 = scelta ottima
  spearman      correlazione di rango tra consiglio e verità
Baseline: random (valore atteso analitico), global (media dei punteggi su tutti gli altri dataset, senza k-NN),
popular (un metodo fisso per modalità: smote / jitter / dropedge, se misurato).

Uso:  python -m bench.experiments.e1_loo [--profiles balanced utility] [--kb-dir assistant/kb]
Uscita: bench/experiments/out/e1_loo_rows.csv (una riga per dataset e profilo) e e1_loo_summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.kb import scoring  # noqa: E402
from assistant.kb.store import canon, load_kb  # noqa: E402
from assistant.recommender import _score_candidate, _weights_for  # noqa: E402

OUT = ROOT / "bench" / "experiments" / "out"
POPULAR = {"tabular": ["smote"], "ts": ["jitter"], "graph": ["dropedge", "nl_dropedge", "drop_edge"]}


def totals(scores: dict, profile: str) -> dict:
    avail = [ax for ax in scoring.ALL_AXES if any((s["axis_scores"].get(ax) is not None) for s in scores.values())]
    w = _weights_for(profile, avail)
    return {a: _score_candidate(s, w)[0] for a, s in scores.items()}


def evaluate_ranking(pred: dict, true: dict) -> dict:
    common = sorted(set(pred) & set(true))
    if len(common) < 3:
        return {}
    p = pd.Series({a: pred[a] for a in common}).sort_values(ascending=False, kind="stable")
    t = pd.Series({a: true[a] for a in common}).sort_values(ascending=False, kind="stable")
    tmax, tmin = t.iloc[0], t.iloc[-1]
    span = tmax - tmin
    top3p, top3t = list(p.index[:3]), list(t.index[:3])
    rho = spearmanr([pred[a] for a in common], [true[a] for a in common]).correlation
    return {
        "n_algos": len(common), "best_true": t.index[0], "top1_pred": p.index[0],
        "best_in_top3": float(t.index[0] in top3p),
        "overlap@3": len(set(top3p) & set(top3t)) / 3,
        "regret@1": float((tmax - true[p.index[0]]) / span) if span > 1e-12 else 0.0,
        "spearman": float(rho) if rho == rho else np.nan,
    }


def random_baseline(true: dict) -> dict:
    vals = np.array(sorted(true.values(), reverse=True))
    n = len(vals)
    span = vals[0] - vals[-1]
    return {"best_in_top3": min(1.0, 3 / n), "overlap@3": 3 / n if n >= 3 else 1.0,
            "regret@1": float(np.mean((vals[0] - vals) / span)) if span > 1e-12 else 0.0, "spearman": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", nargs="+", default=["balanced", "utility"])
    ap.add_argument("--kb-dir", default=None)
    ap.add_argument("--k", type=int, default=scoring.K_NEIGHBORS)
    a = ap.parse_args()
    kb = load_kb(a.kb_dir)
    if kb is None or kb.bench.empty:
        print("KB vuoto: costruirlo con python -m assistant.kb.build_kb")
        return 1
    rows = []
    groups = [("tabular", ""), ("ts", ""), ("graph", "graph"), ("graph", "node")]
    for mod, level in groups:
        d = kb.rows(mod, level or None)
        if d.empty:
            continue
        datasets = sorted(d["dataset"].unique())
        per_ds = {ds: scoring.score_dataset(d[d["dataset"] == ds], mod) for ds in datasets}
        for ds in datasets:
            others = [o for o in datasets if o != ds]
            q = scoring.meta_vector(kb, mod, level, ds)
            if not others or q is None:
                continue
            neigh = scoring.knn_neighbors(kb, mod, level, q, others, k=a.k)
            if not neigh:
                continue
            pred_s = scoring.combine([(n, per_ds[n["dataset"]]) for n in neigh])
            glob_s = scoring.combine([({"dataset": o, "weight": 1.0 / len(others)}, per_ds[o]) for o in others])
            for prof in a.profiles:
                true = totals(per_ds[ds], prof)
                pred = totals(pred_s, prof)
                glob = totals(glob_s, prof)
                base = {"modality": mod, "level": level or "-", "dataset": ds, "profile": prof,
                        "neighbors": ";".join(f"{n['dataset']}:{n['distance']:.2f}" for n in neigh)}
                r = evaluate_ranking(pred, true)
                if not r:
                    continue
                rows.append({**base, "method": "augpilot_knn", **r})
                g = evaluate_ranking(glob, true)
                if g:
                    rows.append({**base, "method": "global_mean", **g})
                rows.append({**base, "method": "random", "n_algos": r["n_algos"],
                             **random_baseline({k: true[k] for k in set(true) & set(pred)})})
                pop = next((p for p in POPULAR.get(mod, []) if p in true), None)
                if pop:
                    tv = pd.Series(true)
                    span = tv.max() - tv.min()
                    rows.append({**base, "method": f"popular:{pop}", "n_algos": len(tv),
                                 "best_in_top3": np.nan, "overlap@3": np.nan, "spearman": np.nan,
                                 "regret@1": float((tv.max() - tv[pop]) / span) if span > 1e-12 else 0.0})
    if not rows:
        print("Nessun dataset valutabile (servono almeno 2 dataset per modalità e 3 algoritmi in comune).")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1_loo_rows.csv", index=False)
    summ = (df.groupby(["modality", "level", "profile", "method"])
              [["best_in_top3", "overlap@3", "regret@1", "spearman"]].mean()
              .join(df.groupby(["modality", "level", "profile", "method"]).size().rename("n_datasets"))
              .reset_index())
    summ.to_csv(OUT / "e1_loo_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
