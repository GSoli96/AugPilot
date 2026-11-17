"""SigCWGAN (Ni et al., 2020; arXiv:2006.05421) - codice ORIGINALE di Algorithms/Conditional-Sig-Wasserstein-GANs
(lib.algos.sigcwgan.SigCWGAN, generatore SimpleGenerator, loss Sig-W1), con UN solo adattamento esterno:

- il repo importa `signatory` (estensione C++ per torch 1.4, non compilabile qui). Il worker inserisce in
  sys.modules un modulo `signatory` MINIMO scritto da noi (firma troncata via relazione di Chen, in puro torch,
  differenziabile; verificato su casi analitici). Non e' il codice originale di signatory.
- dati: il repo si addestra su finestre scorrevoli (p+q) di un unico cammino lungo. Qui le finestre sono prese
  da ogni serie del dataset (p=q=3, MinMax interno). Generazione: autoregressiva, condizionata sui primi p passi
  di una serie reale scelta a caso; l'output ha lunghezza T = p passi del prefisso reale + (T-p) generati.
  Il prefisso reale e' quindi copiato: il metodo e' condizionale per costruzione.
- iperparametri: profondita' della firma 2, mc_size 100, batch 200, hidden 3x50; --epochs = epoche da 50 passi.
Il repo ha `.cuda()` fisso in sample_indices: richiede GPU. Nessun file LICENSE nel repo.
"""
import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import ALGO, load_npz, minmax_fit, minmax_inv, save_out, seed_all  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402


def _install_signatory_shim():
    import torch

    def signature(path, depth, basepoint=False):
        """path (B,L,C) -> (B, C + C^2 + ... + C^depth); livelli concatenati, primo indice = piu' vecchio."""
        B, L, C = path.shape
        if basepoint is True:
            path = torch.cat([torch.zeros(B, 1, C, device=path.device, dtype=path.dtype), path], dim=1)
        elif basepoint is not False:
            path = torch.cat([basepoint.reshape(B, 1, C), path], dim=1)
        inc = path[:, 1:] - path[:, :-1]
        S = None
        for t in range(inc.shape[1]):
            d = inc[:, t]
            E = [d]  # E_m = d^{(x)m}/m!
            for m in range(2, depth + 1):
                E.append((E[-1].unsqueeze(2) * d.unsqueeze(1)).reshape(B, -1) / m)
            if S is None:
                S = E
                continue
            new = []
            for k in range(1, depth + 1):
                lvl = S[k - 1] + E[k - 1]
                for j in range(1, k):
                    lvl = lvl + (S[j - 1].unsqueeze(2) * E[k - j - 1].unsqueeze(1)).reshape(B, -1)
                new.append(lvl)
            S = new
        return torch.cat(S, dim=1)

    mod = types.ModuleType("signatory")
    mod.signature = signature
    mod.__doc__ = "shim minimo (puro torch) per SigCWGAN; non e' il pacchetto signatory"
    sys.modules["signatory"] = mod


def main(a):
    import torch
    if not torch.cuda.is_available() or a.device != "cuda":
        raise RuntimeError("SigCWGAN: il codice originale usa .cuda() (sample_indices): serve --device cuda")
    _install_signatory_shim()
    sys.path.insert(0, os.path.join(ALGO, "Conditional-Sig-Wasserstein-GANs"))
    from lib.algos.base import BaseConfig
    from lib.algos.sigcwgan import SigCWGAN, SigCWGANConfig
    from lib.augmentations import SignatureConfig, get_standard_augmentation

    seed_all(a.seed)
    x, _ = load_npz(a.data)
    N, T, D = x.shape
    p = q = 3
    if T < p + q:
        raise ValueError(f"serie troppo corte (T={T}) per p+q={p + q}")
    z, lo, hi = minmax_fit(x)
    # finestre scorrevoli (p+q) da ogni serie; la media 0.5 viene tolta come scala del repo (Scale(0.2) sul cumsum)
    wins = np.stack([z[:, i:i + p + q] for i in range(T - p - q + 1)], axis=1).reshape(-1, p + q, D)
    rng = np.random.default_rng(a.seed)
    if len(wins) > 20000:
        wins = wins[rng.choice(len(wins), 20000, replace=False)]
    mu = wins.mean(axis=(0, 1), keepdims=True)
    dev = "cuda:0"
    xr = torch.tensor(wins - mu, dtype=torch.float32, device=dev)
    sc = get_standard_augmentation(0.2)
    cfg = SigCWGANConfig(mc_size=100, sig_config_past=SignatureConfig(depth=2, augmentations=sc),
                         sig_config_future=SignatureConfig(depth=2, augmentations=sc))
    base = BaseConfig(seed=a.seed, batch_size=200, device=dev, p=p, q=q, hidden_dims=3 * (50,),
                      total_steps=max(1, a.epochs * 50), mc_samples=1000)
    algo = SigCWGAN(base_config=base, config=cfg, x_real=xr)
    algo.fit()
    n = a.n_samples or N
    idx = rng.choice(N, n, replace=True)
    prefix = torch.tensor(z[idx, :p] - mu, dtype=torch.float32, device=dev)
    algo.G.eval()
    with torch.no_grad():
        fut = algo.G.sample(T - p, prefix)
    g = torch.cat([prefix, fut], dim=1).cpu().numpy() + mu
    save_out(a.out, minmax_inv(g, lo, hi))
    return len(g)


if __name__ == "__main__":
    run_worker(main, std_parser(__doc__).parse_args())
