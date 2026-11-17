"""Aumentazioni classiche per serie temporali (numpy/scipy), un worker con --method.

Contratto: legge un .npz con `data` (N,T,D) e, se c'e', `labels` (N,); scrive un .npz con `data`
(SOLO i campioni generati, n_samples default = N) e `labels` (etichetta del campione di partenza,
se il dataset ne ha). `--epochs` e `--device` sono accettati ma ignorati: i metodi non si addestrano.
L'ultima riga di stdout e' il JSON del contratto (worker_utils.run_worker).

Metodi (chiave -> strategia):
  jitter            rumore gaussiano additivo (noise)
  scaling           scala per canale con fattore ~N(1, 0.1) (magnitude)
  magnitude_warp    moltiplica per una curva liscia (spline) ~1 (magnitude warping)
  time_warp         deforma l'asse del tempo con una curva liscia monotona (time warping)
  window_slice      ritaglia il 90% della finestra e la riallunga (cropping)
  window_warp       comprime/espande una finestra del 10% (window warping)
  permutation       rimescola segmenti della stessa serie (permutation)
  rotation          rotazione casuale dei canali (D>1) o ribaltamento attorno alla media (D=1)
  slice_shuffle     taglia e ricombina segmenti tra serie della stessa classe (cut-and-mix)
  dba               media baricentrica DTW pesata di serie della stessa classe (DBA, Petitjean 2011)
  smote             SMOTE: interpolazione verso un vicino (k-NN) della stessa classe
  mixup             mixup tra due serie della stessa classe, lambda ~ Beta(0.4, 0.4)
  block_bootstrap   moving block bootstrap dei blocchi della stessa serie
  decomp_bootstrap  trend + stagionalita' + ricampionamento a blocchi dei residui (STL-like, numpy)
  ssa               Singular Spectrum Analysis: ricostruzione a rango ridotto + residui ricampionati
  pca               PCA globale sulle finestre, punteggi perturbati, ricostruzione
  fourier_surrogate fasi casuali dello spettro (stesse fasi tra canali), spettro di ampiezza invariato
  iaaft             surrogato IAAFT (spettro + distribuzione dei valori), per canale
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from modalities.worker_utils import std_parser, run_worker  # noqa: E402

METHODS = ["jitter", "scaling", "magnitude_warp", "time_warp", "window_slice", "window_warp",
           "permutation", "rotation", "slice_shuffle", "dba", "smote", "mixup", "block_bootstrap",
           "decomp_bootstrap", "ssa", "pca", "fourier_surrogate", "iaaft"]


def _interp_rows(x, tt):
    """x (T,D); tt (T,) posizioni reali in [0, T-1] -> (T,D) con interpolazione lineare."""
    T = x.shape[0]
    base = np.arange(T)
    return np.stack([np.interp(tt, base, x[:, d]) for d in range(x.shape[1])], axis=1)


def _smooth_curve(rng, T, n_knots=4, sigma=0.2, D=1):
    from scipy.interpolate import CubicSpline
    xs = np.linspace(0, T - 1, n_knots + 2)
    out = np.empty((T, D))
    for d in range(D):
        ys = rng.normal(1.0, sigma, n_knots + 2)
        out[:, d] = CubicSpline(xs, ys)(np.arange(T))
    return out


# ---------------------------------------------------------------- metodi per singola serie
def m_jitter(x, i, c):
    return x[i] + c["rng"].normal(0, 0.05, x[i].shape) * c["std"]


def m_scaling(x, i, c):
    return x[i] * c["rng"].normal(1.0, 0.1, (1, x.shape[2]))


def m_magnitude_warp(x, i, c):
    return x[i] * _smooth_curve(c["rng"], x.shape[1], D=x.shape[2])


def m_time_warp(x, i, c):
    T = x.shape[1]
    speed = np.clip(_smooth_curve(c["rng"], T, sigma=0.3)[:, 0], 0.2, None)
    tt = np.cumsum(speed)
    tt = (tt - tt[0]) / (tt[-1] - tt[0]) * (T - 1)
    return _interp_rows(x[i], tt)


def m_window_slice(x, i, c):
    T = x.shape[1]
    w = max(2, int(round(0.9 * T)))
    s = c["rng"].integers(0, T - w + 1)
    return _interp_rows(x[i][s:s + w], np.linspace(0, w - 1, T))


def m_window_warp(x, i, c):
    rng = c["rng"]; T = x.shape[1]
    w = max(2, int(round(0.1 * T)))
    s = rng.integers(0, T - w)
    f = rng.choice([0.5, 2.0])
    w2 = max(1, int(round(w * f)))
    a = x[i][:s]; b = x[i][s:s + w]; z = x[i][s + w:]
    mid = _interp_rows(b, np.linspace(0, w - 1, w2)) if w2 > 1 else b[:1]
    y = np.concatenate([a, mid, z], axis=0)
    return _interp_rows(y, np.linspace(0, len(y) - 1, T))


def m_permutation(x, i, c):
    rng = c["rng"]; T = x.shape[1]
    n = min(4, T)
    cuts = np.sort(rng.choice(np.arange(1, T), n - 1, replace=False))
    segs = np.split(x[i], cuts)
    order = rng.permutation(len(segs))
    return np.concatenate([segs[k] for k in order], axis=0)


def m_rotation(x, i, c):
    rng = c["rng"]; D = x.shape[2]
    mu = x[i].mean(axis=0, keepdims=True)
    if D == 1:
        return (2 * mu - x[i]) if rng.random() < 0.5 else x[i][::-1].copy()
    from scipy.linalg import expm
    A = rng.normal(0, 0.3, (D, D)); A = A - A.T
    return (x[i] - mu) @ expm(A) + mu


def m_slice_shuffle(x, i, c):
    rng = c["rng"]; T = x.shape[1]
    n = min(4, T)
    cuts = np.concatenate([[0], np.sort(rng.choice(np.arange(1, T), n - 1, replace=False)), [T]])
    pool = c["pool"](i)
    out = np.empty_like(x[i])
    for a, b in zip(cuts[:-1], cuts[1:]):
        out[a:b] = x[rng.choice(pool)][a:b]
    return out


def _dtw_path(a, b, band):
    """DTW multivariata con banda di Sakoe-Chiba; restituisce il cammino come lista (i, j)."""
    T1, T2 = len(a), len(b)
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    D = np.full((T1 + 1, T2 + 1), np.inf); D[0, 0] = 0
    for p in range(1, T1 + 1):
        lo = max(1, p - band); hi = min(T2, p + band)
        for q in range(lo, hi + 1):
            D[p, q] = cost[p - 1, q - 1] + min(D[p - 1, q - 1], D[p - 1, q], D[p, q - 1])
    p, q, path = T1, T2, []
    while p > 0 and q > 0:
        path.append((p - 1, q - 1))
        k = np.argmin([D[p - 1, q - 1], D[p - 1, q], D[p, q - 1]])
        if k == 0: p, q = p - 1, q - 1
        elif k == 1: p -= 1
        else: q -= 1
    return path[::-1]


def m_dba(x, i, c):
    rng = c["rng"]; T = x.shape[1]
    pool = c["pool"](i)
    k = min(4, len(pool))
    sel = rng.choice(pool, k, replace=False) if len(pool) >= k else pool
    if i not in sel:
        sel[0] = i
    w = rng.dirichlet(np.ones(len(sel)))
    avg = x[sel[np.argmax(w)]].copy()
    band = max(2, int(0.2 * T))
    for _ in range(3):
        num = np.zeros_like(avg); den = np.zeros((T, 1))
        for s, ws in zip(sel, w):
            for (p, q) in _dtw_path(avg, x[s], band):
                num[p] += ws * x[s][q]; den[p] += ws
        avg = num / np.maximum(den, 1e-12)
    return avg


def m_smote(x, i, c):
    nn = c["knn"](i)
    j = c["rng"].choice(nn) if len(nn) else i
    return x[i] + c["rng"].random() * (x[j] - x[i])


def m_mixup(x, i, c):
    pool = c["pool"](i)
    j = c["rng"].choice(pool)
    lam = c["rng"].beta(0.4, 0.4)
    return lam * x[i] + (1 - lam) * x[j]


def _block_resample(rng, n, L):
    out = []
    while len(out) < n:
        s = rng.integers(0, max(1, n - L + 1))
        out.extend(range(s, min(s + L, n)))
    return np.array(out[:n])


def m_block_bootstrap(x, i, c):
    T = x.shape[1]
    return x[i][_block_resample(c["rng"], T, max(2, T // 4))]


def _moving_avg(y, w):
    pad = w // 2
    yp = np.pad(y, ((pad, w - 1 - pad), (0, 0)), mode="edge")
    k = np.ones(w) / w
    return np.stack([np.convolve(yp[:, d], k, mode="valid") for d in range(y.shape[1])], axis=1)


def m_decomp_bootstrap(x, i, c):
    T = x.shape[1]; p = c["period"]
    w = max(3, T // 4)
    trend = _moving_avg(x[i], w)
    det = x[i] - trend
    if p:
        seas = np.stack([det[ph::p].mean(axis=0) for ph in range(p)], axis=0)[np.arange(T) % p]
    else:
        seas = np.zeros_like(det)
    res = det - seas
    idx = _block_resample(c["rng"], T, max(2, T // 8))
    return trend + seas + res[idx]


def _ssa_recon(y, r, L):
    T = len(y); K = T - L + 1
    Xm = np.stack([y[k:k + L] for k in range(K)], axis=1)  # (L,K)
    U, s, Vt = np.linalg.svd(Xm, full_matrices=False)
    Xr = (U[:, :r] * s[:r]) @ Vt[:r]
    out = np.zeros(T); cnt = np.zeros(T)
    for k in range(K):
        out[k:k + L] += Xr[:, k]; cnt[k:k + L] += 1
    return out / cnt


def m_ssa(x, i, c):
    T = x.shape[1]; L = max(2, T // 2)
    rec = np.stack([_ssa_recon(x[i][:, d], c["ssa_r"], L) for d in range(x.shape[2])], axis=1)
    res = x[i] - rec
    return rec + res[_block_resample(c["rng"], T, max(2, T // 8))]


def m_pca(x, i, c):
    mu, comps, sd = c["pca"]
    z = (x[i].reshape(-1) - mu) @ comps.T
    z = z + c["rng"].normal(0, 0.3, z.shape) * sd
    return (z @ comps + mu).reshape(x[i].shape)


def m_fourier_surrogate(x, i, c):
    T = x.shape[1]
    F = np.fft.rfft(x[i], axis=0)
    ph = c["rng"].uniform(0, 2 * np.pi, (F.shape[0], 1))
    ph[0] = 0
    if T % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * (np.angle(F) + ph)), n=T, axis=0)


def m_iaaft(x, i, c, n_iter=30):
    T = x.shape[1]
    out = np.empty_like(x[i])
    for d in range(x.shape[2]):
        y = x[i][:, d]
        amp = np.abs(np.fft.rfft(y)); srt = np.sort(y)
        z = c["rng"].permutation(y)
        for _ in range(n_iter):
            Z = np.fft.rfft(z)
            z = np.fft.irfft(amp * np.exp(1j * np.angle(Z)), n=T)
            z = srt[np.argsort(np.argsort(z))]
        out[:, d] = z
    return out


FUNCS = {"jitter": m_jitter, "scaling": m_scaling, "magnitude_warp": m_magnitude_warp,
         "time_warp": m_time_warp, "window_slice": m_window_slice, "window_warp": m_window_warp,
         "permutation": m_permutation, "rotation": m_rotation, "slice_shuffle": m_slice_shuffle,
         "dba": m_dba, "smote": m_smote, "mixup": m_mixup, "block_bootstrap": m_block_bootstrap,
         "decomp_bootstrap": m_decomp_bootstrap, "ssa": m_ssa, "pca": m_pca,
         "fourier_surrogate": m_fourier_surrogate, "iaaft": m_iaaft}
assert set(FUNCS) == set(METHODS)


def _detect_period(x):
    """Periodo dominante (ACF media dei dati detrendati) o 0 se non c'e' stagionalita' evidente."""
    T = x.shape[1]
    if T < 8:
        return 0
    d = x - x.mean(1, keepdims=True)
    acf = []
    for lag in range(2, T // 2 + 1):
        a = (d[:, :-lag] * d[:, lag:]).mean() / (d.var() + 1e-12)
        acf.append(a)
    acf = np.array(acf)
    k = int(np.argmax(acf))
    return k + 2 if acf[k] > 0.3 else 0


def build_context(x, y, rng, method):
    N, T, D = x.shape
    lab = y if y is not None else np.zeros(N, dtype=int)
    members = {k: np.where(lab == k)[0] for k in np.unique(lab)}
    c = {"rng": rng, "std": x.reshape(-1, D).std(axis=0, keepdims=True) + 1e-8,
         "pool": lambda i: members[lab[i]]}
    if method == "smote":
        flat = x.reshape(N, -1)
        cache = {}

        def knn(i, k=5):
            if i not in cache:
                idx = members[lab[i]]
                dist = ((flat[idx] - flat[i]) ** 2).sum(1)
                order = idx[np.argsort(dist)]
                cache[i] = order[order != i][:k]
            return cache[i]
        c["knn"] = knn
    if method == "decomp_bootstrap":
        c["period"] = _detect_period(x)
    if method == "ssa":
        s = np.linalg.svd(np.stack([x[0, k:k + max(2, T // 2), 0] for k in range(T - max(2, T // 2) + 1)], 1),
                          compute_uv=False)
        c["ssa_r"] = int(max(1, np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.95) + 1))
    if method == "pca":
        flat = x.reshape(N, -1).astype("float64")
        mu = flat.mean(0)
        U, s, Vt = np.linalg.svd(flat - mu, full_matrices=False)
        var = s ** 2 / max(1, N - 1)
        r = int(np.searchsorted(np.cumsum(var) / var.sum(), 0.95) + 1)
        c["pca"] = (mu, Vt[:r], np.sqrt(var[:r]))
    return c


def run(args):
    if args.method not in FUNCS:
        raise ValueError(f"metodo sconosciuto: {args.method}; disponibili: {METHODS}")
    z = np.load(args.data)
    x = z["data"].astype("float64")
    y = z["labels"] if "labels" in z.files else None
    N = x.shape[0]
    n = args.n_samples or N
    rng = np.random.default_rng(args.seed)
    c = build_context(x, y, rng, args.method)
    base = np.arange(n) % N
    out = np.stack([FUNCS[args.method](x, int(i), c) for i in base]).astype("float32")
    if not np.isfinite(out).all():
        raise FloatingPointError("il metodo ha prodotto valori non finiti")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if y is not None:
        np.savez(args.out, data=out, labels=y[base])
    else:
        np.savez(args.out, data=out)
    return len(out)


if __name__ == "__main__":
    p = std_parser("aumentazioni classiche per serie temporali")
    p.add_argument("--method", required=True)
    p.add_argument("--label_col", default=None)
    run_worker(run, p.parse_args())
