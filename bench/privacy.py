"""Metriche di privacy per il tabulare (WP2): distanza dal record reale più vicino (DCR)
e rapporto tra primo e secondo vicino (NNDR), con il test come riferimento (holdout).

Tutte le distanze sono euclidee su feature normalizzate min-max, con il normalizzatore
stimato sulla minoritaria del train. Sottocampionamento a `max_rows` righe con seed.

Chiavi restituite (direzione in PRIVACY_DIRECTION):
  DCR_p5            5° percentile della distanza sintetico -> train_min (più alto = meno copie)
  DCR_median        mediana della stessa distanza
  DCR_holdout_p5    5° percentile della distanza test_min -> train_min (riferimento: dati reali mai visti)
  DCR_ratio         DCR_median / mediana(test_min -> train_min); ~1 = come dati nuovi, << 1 = memorizzazione
  exact_copy_rate   quota di sintetici a distanza < 1e-8 da un record del train
  closer_than_holdout_share  quota di sintetici più vicini al train che il record di test più vicino al train
                    (su un riferimento: mediana(test->train)); più basso è meglio
  NNDR_mean         media di d1/d2 verso il train (vicino a 1 = il sintetico non "punta" un singolo record)
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

PRIVACY_DIRECTION = {
    "DCR_p5": "higher", "DCR_median": "higher", "DCR_holdout_p5": "neutral",
    "DCR_ratio": "target1", "exact_copy_rate": "lower",
    "closer_than_holdout_share": "lower", "NNDR_mean": "higher",
}


def _minmax(train: np.ndarray, *others: np.ndarray):
    lo = train.min(axis=0)
    span = train.max(axis=0) - lo
    span[span < 1e-12] = 1.0
    return [(x - lo) / span for x in (train, *others)]


def _sub(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), size=n, replace=False)]


def privacy_metrics(train_min: np.ndarray, synth: np.ndarray, test_min: np.ndarray,
                    seed: int = 0, max_rows: int = 5000) -> dict:
    nan = {k: float("nan") for k in PRIVACY_DIRECTION}
    if len(train_min) < 2 or len(synth) == 0:
        return nan
    rng = np.random.default_rng(seed)
    tr, sy, te = _minmax(train_min.astype(float), synth.astype(float), test_min.astype(float))
    tr = _sub(tr, max_rows, rng)
    sy = _sub(sy, max_rows, rng)
    te = _sub(te, max_rows, rng)

    nn = NearestNeighbors(n_neighbors=2).fit(tr)
    d_sy, _ = nn.kneighbors(sy)
    dcr_sy = d_sy[:, 0]
    nndr = d_sy[:, 0] / np.maximum(d_sy[:, 1], 1e-12)

    out = {
        "DCR_p5": float(np.percentile(dcr_sy, 5)),
        "DCR_median": float(np.median(dcr_sy)),
        "exact_copy_rate": float(np.mean(dcr_sy < 1e-8)),
        "NNDR_mean": float(np.mean(nndr)),
    }
    if len(te) > 0:
        d_te, _ = nn.kneighbors(te, n_neighbors=1)
        dcr_te = d_te[:, 0]
        med_te = float(np.median(dcr_te))
        out["DCR_holdout_p5"] = float(np.percentile(dcr_te, 5))
        out["DCR_ratio"] = float(out["DCR_median"] / med_te) if med_te > 1e-12 else float("nan")
        out["closer_than_holdout_share"] = float(np.mean(dcr_sy < med_te))
    else:
        out.update({"DCR_holdout_p5": float("nan"), "DCR_ratio": float("nan"),
                    "closer_than_holdout_share": float("nan")})
    return out
