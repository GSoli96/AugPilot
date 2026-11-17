"""Utilita' comuni ai worker ext_*.py: lettura .npz, MinMax interno, scrittura dell'output."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ALGO = os.path.join(ROOT, "Algorithms")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_npz(path):
    """Restituisce (data (N,T,D) float32, labels o None). Accetta un percorso o il nome di un dataset."""
    if not os.path.exists(path):
        from modalities.ts.datasets import resolve
        path = str(resolve(path))
    z = np.load(path, allow_pickle=False)
    x = z["data"].astype("float32")
    if x.ndim != 3:
        raise ValueError(f"{path}: forma {x.shape}, atteso (N,T,D)")
    y = z["labels"] if "labels" in z.files else None
    return x, y


def minmax_fit(x):
    """MinMax per canale su tutto il dataset; restituisce (x01, lo, hi)."""
    lo = x.min(axis=(0, 1), keepdims=True)
    hi = x.max(axis=(0, 1), keepdims=True)
    return (x - lo) / (hi - lo + 1e-7), lo, hi


def minmax_inv(z, lo, hi):
    return z * (hi - lo + 1e-7) + lo


def save_out(path, data, labels=None):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    kw = {"data": np.asarray(data, dtype="float32")}
    if labels is not None:
        kw["labels"] = np.asarray(labels)
    if not np.isfinite(kw["data"]).all():
        raise ValueError("l'output contiene NaN/inf")
    np.savez(path, **kw)


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
