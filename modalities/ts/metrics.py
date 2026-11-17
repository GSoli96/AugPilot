"""Metriche per serie temporali: evaluate(original_path, generated_path) -> dict piatto di numeri.

Ispirate a TSGBench (Algorithms/TSGBench/src/evaluation.py, feature_based_measures.py, ds_ps.py),
riscritte in numpy/torch senza dipendenze extra (DTW propria, nessun pickle).
Tutte le metriche sono "piu' basso = meglio" tranne clf_acc_*.

Colonne restituite:
  n_orig, n_gen          numero di serie usate (sottocampione a max_n)
  MDD                    distanza media tra istogrammi marginali (50 bin) per (tempo, canale)
  ACD                    differenza di autocorrelazione (lag 1..min(T-1,64)); norma L2 sui lag, media sui canali
  SD, KD                 |differenza| di skewness / kurtosis (eccesso) per canale, media
  ED                     (NN) per ogni serie generata, distanza euclidea (su T*D valori) dalla serie originale piu'
                         vicina; media sulle generate. NON e' piu' l'accoppiamento per indice di TSGBench:
                         la chiave resta "ED" ma il significato e' "distanza dal vicino piu' prossimo"
  DTW                    (NN) come ED ma con DTW multivariata (cammino su costo euclideo quadrato, radice finale);
                         sottocampione di max 500 generate e 500 originali (rng seed 0)
  FD_feat                distanza di Frechet su feature semplici (media, std, ACF1, pendenza per canale);
                         NON e' il C-FID di TSGBench (che usa gli embedding di TS2Vec)
  DS                     Discriminative score: |acc - 0.5| di un GRU reale-vs-generato su test (25%)
  PS                     Predictive score (TSTR): MAE di un GRU addestrato sul generato che predice il passo
                         successivo, testato sull'originale
  PS_trtr                come PS ma addestrato sull'originale (riferimento)
  clf_acc_base, clf_acc_aug, clf_acc_delta   solo se l'originale ha `labels` e il generato ha `labels`:
                         accuratezza su hold-out (30%) dell'originale di una regressione logistica addestrata
                         sull'originale-train (base) e su originale-train + generato (aug)
Parametri: epochs (DS/PS), default 20 oppure variabile d'ambiente TS_METRIC_EPOCHS; device 'auto'|'cpu'|'cuda'.
"""
import os
import time

import numpy as np


def _load(path):
    z = np.load(path)
    return z["data"].astype("float64"), (z["labels"] if "labels" in z.files else None)


# ------------------------------------------------------------------ feature-based
def mdd(a, b, n_bins=50):
    lo = a.min(axis=0); hi = a.max(axis=0)  # (T,D)
    hi = np.where(hi - lo < 1e-9, lo + 1e-9, hi)
    T, D = lo.shape
    tot = 0.0
    for t in range(T):
        for d in range(D):
            e = np.linspace(lo[t, d], hi[t, d], n_bins + 1)
            ha, _ = np.histogram(np.clip(a[:, t, d], e[0], e[-1]), e, density=True)
            hb, _ = np.histogram(np.clip(b[:, t, d], e[0], e[-1]), e, density=True)
            tot += np.abs(ha - hb).mean()
    return tot / (T * D)


def _acf(x, max_lag):
    N, T, D = x.shape
    xc = x - x.mean(axis=(0, 1), keepdims=True)
    var = (xc ** 2).mean(axis=(0, 1)) + 1e-12
    return np.stack([(xc[:, :-l] * xc[:, l:]).mean(axis=(0, 1)) / var for l in range(1, max_lag + 1)])  # (L,D)


def acd(a, b):
    L = min(a.shape[1] - 1, 64)
    if L < 1:
        return float("nan")
    return float(np.sqrt(((_acf(a, L) - _acf(b, L)) ** 2).sum(0)).mean())


def _moments(x):
    f = x.reshape(-1, x.shape[2])
    m = f.mean(0); s = f.std(0) + 1e-12
    z = (f - m) / s
    return (z ** 3).mean(0), (z ** 4).mean(0) - 3.0


def sd_kd(a, b):
    sa, ka = _moments(a); sb, kb = _moments(b)
    return float(np.abs(sa - sb).mean()), float(np.abs(ka - kb).mean())


# ------------------------------------------------------------------ distanze
def ed(a, b):
    """Distanza euclidea media a COPPIE (serie i di a vs serie i di b). Non usata da evaluate (vedi ed_nn)."""
    return float(np.linalg.norm(a - b, axis=1).mean())  # norma sul tempo, media su coppie e canali


def _dtw_pairs(a, b):
    """DTW multivariata batched, una per coppia: costo = distanza euclidea quadrata tra vettori,
    risultato = sqrt(D[T,T]). a, b: (P,T,D). Restituisce un array (P,)."""
    P, T, _ = a.shape
    cost = ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)  # (P,T,T)
    D = np.full((P, T + 1, T + 1), np.inf); D[:, 0, 0] = 0
    for i in range(1, T + 1):
        for j in range(1, T + 1):
            D[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(D[:, i - 1, j - 1], D[:, i - 1, j]), D[:, i, j - 1])
    return np.sqrt(D[:, T, T])


def dtw(a, b):
    """DTW multivariata media a COPPIE (serie i di a vs serie i di b). Non usata da evaluate (vedi dtw_nn)."""
    return float(_dtw_pairs(a, b).mean())


def ed_nn(gen, real):
    """ED dal vicino piu' prossimo: per ogni serie generata, distanza euclidea (su T*D valori) minima verso le
    serie reali; media sulle generate."""
    from scipy.spatial.distance import cdist
    d = cdist(gen.reshape(len(gen), -1), real.reshape(len(real), -1), "euclidean")
    return float(d.min(axis=1).mean())


def dtw_nn(gen, real, pair_budget=2_000_000):
    """DTW dal vicino piu' prossimo: per ogni serie generata, DTW minima verso le serie reali; media sulle generate.
    Le coppie sono elaborate a blocchi per limitare la memoria (pair_budget = elementi float per blocco)."""
    G = len(gen)
    R, T, D = real.shape
    per_pair = T * T * max(D, 1)
    block = max(1, int(pair_budget // per_pair))  # coppie per blocco
    mins = np.empty(G)
    for g in range(G):
        best = np.inf
        for s in range(0, R, block):
            r = real[s:s + block]
            vals = _dtw_pairs(np.broadcast_to(gen[g], r.shape), r)
            best = min(best, float(vals.min()))
        mins[g] = best
    return float(mins.mean())


def _feat(x):
    N, T, D = x.shape
    xc = x - x.mean(1, keepdims=True)
    acf1 = (xc[:, :-1] * xc[:, 1:]).mean(1) / (xc.var(1) + 1e-12)
    tt = (np.arange(T) - (T - 1) / 2)[None, :, None]
    slope = (tt * xc).sum(1) / (tt ** 2).sum()
    return np.concatenate([x.mean(1), x.std(1), acf1, slope], axis=1)


def fd_feat(a, b):
    from scipy.linalg import sqrtm
    fa, fb = _feat(a), _feat(b)
    m1, m2 = fa.mean(0), fb.mean(0)
    c1 = np.atleast_2d(np.cov(fa, rowvar=False)); c2 = np.atleast_2d(np.cov(fb, rowvar=False))
    cm = sqrtm(c1 @ c2)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cm))


# ------------------------------------------------------------------ model-based (torch)
def _pick_device(device):
    import torch
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _gru_classifier_score(a, b, epochs, device, seed, hidden=32):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.concatenate([a, b]).astype("float32"); y = np.concatenate([np.ones(len(a)), np.zeros(len(b))]).astype("float32")
    idx = rng.permutation(len(X)); k = int(0.75 * len(X))
    tr, te = idx[:k], idx[k:]

    class Net(nn.Module):
        def __init__(s, d):
            super().__init__(); s.g = nn.GRU(d, hidden, batch_first=True); s.o = nn.Linear(hidden, 1)

        def forward(s, x):
            return s.o(s.g(x)[1][-1]).squeeze(-1)
    net = Net(X.shape[2]).to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3 * 5)
    Xt = torch.tensor(X, device=device); yt = torch.tensor(y, device=device)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        p = torch.tensor(rng.permutation(tr), device=device)
        for s in range(0, len(p), 128):
            bi = p[s:s + 128]
            opt.zero_grad(); lossf(net(Xt[bi]), yt[bi]).backward(); opt.step()
    with torch.no_grad():
        pred = (net(Xt[torch.tensor(te, device=device)]) > 0).float().cpu().numpy()
    return abs(float((pred == y[te]).mean()) - 0.5)


def _gru_predict_mae(train, test, epochs, device, seed, hidden=32):
    """Addestra su `train` a predire x[t+1] da x[:t+1]; MAE su `test`."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    D = train.shape[2]

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.g = nn.GRU(D, hidden, batch_first=True); s.o = nn.Linear(hidden, D)

        def forward(s, x):
            return s.o(s.g(x)[0])
    net = Net().to(device); opt = torch.optim.Adam(net.parameters(), 5e-3)
    Xtr = torch.tensor(train.astype("float32"), device=device)
    Xte = torch.tensor(test.astype("float32"), device=device)
    for _ in range(epochs):
        p = torch.tensor(rng.permutation(len(Xtr)), device=device)
        for s in range(0, len(p), 128):
            bi = p[s:s + 128]; x = Xtr[bi]
            opt.zero_grad(); nn.functional.l1_loss(net(x[:, :-1]), x[:, 1:]).backward(); opt.step()
    with torch.no_grad():
        return float((net(Xte[:, :-1]) - Xte[:, 1:]).abs().mean())


def _downstream(a, ya, b, yb, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(a)); k = int(0.7 * len(a))
    tr, te = idx[:k], idx[k:]
    fa = a.reshape(len(a), -1); fb = b.reshape(len(b), -1)

    def fit(X, y):
        sc = StandardScaler().fit(X)
        m = LogisticRegression(max_iter=300).fit(sc.transform(X), y)
        return float((m.predict(sc.transform(fa[te])) == ya[te]).mean())
    base = fit(fa[tr], ya[tr])
    aug = fit(np.concatenate([fa[tr], fb]), np.concatenate([ya[tr], yb]))
    return base, aug


def evaluate(original_path, generated_path, epochs=None, device="auto", seed=0, max_n=1000,
             model_based=True, max_dtw_series=500):
    t0 = time.time()
    a, ya = _load(original_path); b, yb = _load(generated_path)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"forma incompatibile: originale {a.shape}, generato {b.shape}")
    if not np.isfinite(b).all():
        raise ValueError("il generato contiene valori non finiti")
    epochs = int(epochs if epochs is not None else os.environ.get("TS_METRIC_EPOCHS", 20))
    rng = np.random.default_rng(seed)
    ia = np.sort(rng.choice(len(a), min(len(a), max_n), replace=False))
    ib = np.sort(rng.choice(len(b), min(len(b), max_n), replace=False))
    a0, ya0 = a, ya
    a, b = a[ia], b[ib]
    ya = ya[ia] if ya is not None else None
    yb = yb[ib] if yb is not None else None
    res = {"n_orig": len(a), "n_gen": len(b)}
    res["MDD"] = mdd(a, b)
    res["ACD"] = acd(a, b)
    res["SD"], res["KD"] = sd_kd(a, b)
    # ED e DTW sono distanze dal vicino piu' prossimo (generata -> reale), non per indice.
    res["ED"] = ed_nn(b, a)
    rng_nn = np.random.default_rng(0)
    ig = np.sort(rng_nn.choice(len(b), min(len(b), max_dtw_series), replace=False))
    ir = np.sort(rng_nn.choice(len(a), min(len(a), max_dtw_series), replace=False))
    res["DTW"] = dtw_nn(b[ig], a[ir])
    # FD_feat: Frechet su feature semplici, NON il C-FID di TSGBench (che usa embedding TS2Vec, non implementato).
    res["FD_feat"] = fd_feat(a, b)
    if model_based:
        dev = _pick_device(device)
        res["DS"] = _gru_classifier_score(a, b, epochs, dev, seed)
        res["PS"] = _gru_predict_mae(b, a, epochs, dev, seed)
        res["PS_trtr"] = _gru_predict_mae(a, a, epochs, dev, seed)
    if ya is not None and yb is not None and len(np.unique(ya)) > 1:
        base, aug = _downstream(a, ya, b, yb, seed)
        res.update({"clf_acc_base": base, "clf_acc_aug": aug, "clf_acc_delta": aug - base})
    res["metrics_time_s"] = round(time.time() - t0, 2)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in res.items()}


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("original"); p.add_argument("generated")
    p.add_argument("--epochs", type=int, default=None); p.add_argument("--device", default="auto")
    a = p.parse_args()
    print(json.dumps(evaluate(a.original, a.generated, a.epochs, a.device)))
