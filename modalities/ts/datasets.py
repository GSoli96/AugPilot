"""Dataset di serie temporali. Formato: .npz con `data` (N, T, D) float32 e `labels` (N,) opzionale.
resolve(nome) -> percorso del file in datasets_ts/ (lo crea se e' un dataset noto); se non c'e', cerca tra i dataset
caricati dall'utente in datasets_user_ts/ (stesso formato).

Nomi noti (tutti piccoli, pensati per test rapidi):
  sine_small   sinusoidi sintetiche, N=256, T=24, D=5; labels = 2 classi (frequenza media)
  stock        Algorithms/TimeGAN/data/stock_data.csv, finestre T=24 (passo 1), D=6, MinMax globale
               per colonna; sottocampione deterministico di al massimo 1000 finestre
  energy       Algorithms/TimeGAN/data/energy_data.csv, come sopra, D=28
  stock_full, energy_full   come sopra ma con tutte le finestre (lunghi: non per i test piccoli)
  tv_sine, tv_stock, tv_energy, tv_air
               file *_subsampled_train_perc_100.npz di Algorithms/timeVAE/data/ (gia' (N,24,D),
               valori grezzi nel file; qui MinMax per canale; sottocampionati a 1000 finestre al massimo (seed 0)
  cbf          Cylinder-Bell-Funnel (Saito 1994), generato: N=300, T=64, D=1, 3 classi (con etichette)
  sine_cls     sinusoidi multicanale generate: N=300, T=48, D=3, 3 classi per frequenza (con etichette)
Nessun dataset UCR e' stato scaricato (non verificato).
"""
import re
from pathlib import Path
import numpy as np

DIR = Path(__file__).resolve().parents[2] / "datasets_ts"
ROOT = Path(__file__).resolve().parents[2]
USER_DIR = ROOT / "datasets_user_ts"   # dataset caricati dalla UI (assistant/config.py: USER_DIRS["ts"])
MAX_SMALL = 1000


def make_sine(n=256, t=24, d=5, seed=0):
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0, 0.1, (n, 1, d)); ph = rng.uniform(0, 0.1, (n, 1, d))
    x = np.sin(2 * np.pi * (np.arange(t)[None, :, None] * freq + ph))
    labels = (freq.mean(axis=(1, 2)) > 0.05).astype(int)
    return ((x + 1) / 2).astype("float32"), labels


def _windows_from_csv(csv, seq_len=24, max_n=MAX_SMALL, seed=0):
    import pandas as pd
    v = pd.read_csv(csv).values.astype("float64")
    v = (v - v.min(0)) / (v.max(0) - v.min(0) + 1e-7)
    w = np.stack([v[i:i + seq_len] for i in range(len(v) - seq_len + 1)]).astype("float32")
    if max_n and len(w) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(len(w), max_n, replace=False))
        w = w[idx]
    return w, None


def _timevae(name, max_n=MAX_SMALL, seed=0):
    f = ROOT / "Algorithms" / "timeVAE" / "data" / f"{name}_subsampled_train_perc_100.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    z = np.load(f)
    x = z["data"].astype("float32")
    if x.ndim != 3:
        raise ValueError(f"{f}: shape {x.shape}, expected (N,T,D)")
    if max_n and len(x) > max_n:
        x = x[np.sort(np.random.default_rng(seed).choice(len(x), max_n, replace=False))]
    lo = x.min(axis=(0, 1), keepdims=True); hi = x.max(axis=(0, 1), keepdims=True)
    return ((x - lo) / (hi - lo + 1e-7)).astype("float32"), None


def make_cbf(n=300, t=64, seed=0):
    """Cylinder-Bell-Funnel: classi 0 cylinder, 1 bell, 2 funnel."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    x = np.zeros((n, t, 1))
    tt = np.arange(t)
    for i in range(n):
        a = rng.integers(t // 8, t // 4); b = a + rng.integers(t // 4, t // 2 + 1)
        b = min(b, t - 1)
        eta = rng.normal(); amp = 6 + eta
        ind = ((tt >= a) & (tt <= b)).astype(float)
        if y[i] == 0:
            s = amp * ind
        elif y[i] == 1:
            s = amp * ind * (tt - a) / (b - a)
        else:
            s = amp * ind * (b - tt) / (b - a)
        x[i, :, 0] = s + rng.normal(size=t)
    x = (x - x.min()) / (x.max() - x.min())
    return x.astype("float32"), y


def make_sine_cls(n=300, t=48, d=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    f0 = np.array([0.03, 0.07, 0.12])[y][:, None, None]
    freq = f0 * rng.uniform(0.9, 1.1, (n, 1, d)); ph = rng.uniform(0, 1, (n, 1, d))
    x = np.sin(2 * np.pi * (np.arange(t)[None, :, None] * freq + ph)) + rng.normal(0, 0.05, (n, t, d))
    return ((x + 1.2) / 2.4).astype("float32"), y


_BUILDERS = {
    "sine_small": lambda: make_sine(),
    "stock": lambda: _windows_from_csv(ROOT / "Algorithms/TimeGAN/data/stock_data.csv"),
    "energy": lambda: _windows_from_csv(ROOT / "Algorithms/TimeGAN/data/energy_data.csv"),
    "stock_full": lambda: _windows_from_csv(ROOT / "Algorithms/TimeGAN/data/stock_data.csv", max_n=0),
    "energy_full": lambda: _windows_from_csv(ROOT / "Algorithms/TimeGAN/data/energy_data.csv", max_n=0),
    "tv_sine": lambda: _timevae("sine"),
    "tv_stock": lambda: _timevae("stockv"),
    "tv_energy": lambda: _timevae("energy"),
    "tv_air": lambda: _timevae("air"),
    "cbf": lambda: make_cbf(),
    "sine_cls": lambda: make_sine_cls(),
}


def list_datasets():
    return sorted(_BUILDERS)


def resolve(name: str) -> str:
    p = Path(name)
    if p.exists():
        return str(p)
    DIR.mkdir(exist_ok=True)
    f = DIR / f"{name}.npz"
    if not f.exists() and name in _BUILDERS:
        x, y = _BUILDERS[name]()
        if y is None:
            np.savez(f, data=x)
        else:
            np.savez(f, data=x, labels=y)
    if not f.exists() and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}", name):
        u = USER_DIR / f"{name}.npz"
        if u.is_file():
            return str(u)
    if not f.exists():
        raise FileNotFoundError(f"ts dataset '{name}' not found in {DIR} or {USER_DIR}; known: {list_datasets()}")
    return str(f)


def load(path):
    z = np.load(path)
    return z["data"].astype("float32"), (z["labels"] if "labels" in z.files else None)
