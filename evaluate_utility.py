"""Utilità ML dell'augmentation tabulare (WP2 di the project plan).

Protocollo (senza fughe dal test):
  TRTR  = classificatore addestrato sul train reale, valutato sul test reale;
  TR+S  = addestrato su train reale + righe sintetiche della minoritaria, valutato sullo stesso test.
Il test (30% stratificato del dataset completo) non è mai visto dai generatori.
Si riportano le metriche di entrambi e il delta (TR+S - TRTR).

Classificatori: LR, RF, XGB (se disponibile), MLP, con iperparametri di default fissati e seed;
niente grid search, per stare nei tempi del benchmark. Le definizioni delle metriche seguono
la pipeline di utilità del collega (`Algorithms/SyRFDRes/utility_src/metrics.py`): F1 e AUC-PR
della classe minoritaria (positiva), balanced accuracy.

Uso da riga di comando:
  python evaluate_utility.py --train T.csv --test S.csv --synth G.csv [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UTILITY_DIRECTION = {"f1": "higher", "bal_acc": "higher", "pr_auc": "higher"}


def _n_jobs() -> int:
    """Core per RF/XGB: OMP_NUM_THREADS se impostato (job con thread limitati), altrimenti tutti (-1)."""
    import os
    try:
        n = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else -1


def _models(seed: int) -> dict:
    models = {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=_n_jobs()),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                                             early_stopping=True, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                                      eval_metric="logloss", random_state=seed, n_jobs=_n_jobs(), verbosity=0)
    except Exception:
        pass
    return models


def _fit_eval(model, X_tr, y_tr, X_te, y_te) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        try:
            score = model.predict_proba(X_te)[:, 1]
        except Exception:
            score = None
    return {
        "f1": float(f1_score(y_te, pred, pos_label=1, zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y_te, pred)),
        "pr_auc": float(average_precision_score(y_te, score)) if score is not None else float("nan"),
    }


def utility_metrics(train: pd.DataFrame, test: pd.DataFrame, synth: pd.DataFrame, class_col: str,
                    min_label, seed: int = 0, models: list[str] | None = None) -> dict:
    """Restituisce {'<modello>_<metrica>_trtr', '..._trs', '..._delta', medie 'util_<metrica>_*'}."""
    feats = [c for c in train.columns if c != class_col]
    y_tr = (train[class_col] == min_label).astype(int).to_numpy()
    y_te = (test[class_col] == min_label).astype(int).to_numpy()
    X_tr = train[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    X_te = test[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    if len(synth):
        X_sy = synth[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        X_aug = np.vstack([X_tr, X_sy])
        y_aug = np.concatenate([y_tr, np.ones(len(X_sy), dtype=int)])
    else:
        X_aug, y_aug = X_tr, y_tr

    out: dict = {}
    t0 = time.perf_counter()
    for name, model in _models(seed).items():
        if models and name not in models:
            continue
        base = _fit_eval(model, X_tr, y_tr, X_te, y_te)
        aug = _fit_eval(_models(seed)[name], X_aug, y_aug, X_te, y_te)
        for k in UTILITY_DIRECTION:
            out[f"{name}_{k}_trtr"] = base[k]
            out[f"{name}_{k}_trs"] = aug[k]
            out[f"{name}_{k}_delta"] = aug[k] - base[k]
    for k in UTILITY_DIRECTION:
        for kind in ("trtr", "trs", "delta"):
            vals = [v for kk, v in out.items() if kk.endswith(f"_{k}_{kind}") and not np.isnan(v)]
            out[f"util_{k}_{kind}"] = float(np.mean(vals)) if vals else float("nan")
    out["utility_time_s"] = round(time.perf_counter() - t0, 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train, test, synth = pd.read_csv(a.train), pd.read_csv(a.test), pd.read_csv(a.synth)
    cls = "class" if "class" in train.columns else train.columns[-1]
    min_label = train[cls].value_counts().idxmin()
    synth = synth[[c for c in train.columns if c in synth.columns]]
    print(json.dumps(utility_metrics(train, test, synth, cls, min_label, a.seed), indent=2))


if __name__ == "__main__":
    main()
