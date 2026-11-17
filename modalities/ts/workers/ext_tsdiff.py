"""TSDiff (Kollovieh et al., NeurIPS 2023; Apache-2.0) - codice ORIGINALE di
Algorithms/unconditional-time-series-diffusion (modello `TSDiff` con backbone S4, configurazione
`diffusion_small_config`: 100 passi di diffusione, campionamento DDPM del repo).

Adattamenti esterni (repo non modificato):
- il repo e' scritto per gluonts 0.12 / torch 1.13 / lightning 1.9; qui girano gluonts 0.17 e lightning 2.x.
  `gluonts.torch.modules.scaler` non esiste piu': il worker registra un modulo con un NOPScaler minimo
  (equivalente al vecchio: restituisce i dati e scala 1; la normalizzazione e' il MinMax interno).
  `uncond_ts_diff.model/__init__` (importa moduli gluonts rimossi) viene saltato: si importa solo
  `model.diffusion.tsdiff`.
- TSDiff e' UNIVARIATO: si addestra un modello per canale (D modelli, --epochs epoche ciascuno).
  Si usa use_lags=False, nessuna feature, context_length=T, prediction_length=0 (generazione incondizionata
  dell'intera finestra). Il ciclo di training e' nostro (Adam lr 1e-3, clip 0.5, batch 64 come nelle config
  del repo) e chiama `TSDiff.training_step` con un dict di dati (il ramo con tensore del repo ha un bug: `features` non assegnato). Le correlazioni tra canali NON sono modellate.
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import ALGO, load_npz, minmax_fit, minmax_inv, save_out, seed_all  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402


def _shims(repo_src):
    import torch
    from torch import nn

    class NOPScaler(nn.Module):
        def __init__(self, dim=-1, keepdim=False):
            super().__init__()
            self.dim, self.keepdim = dim, keepdim

        def forward(self, data, observed_indicator=None):
            return data, torch.ones_like(data).mean(dim=self.dim, keepdim=self.keepdim)

    m = types.ModuleType("gluonts.torch.modules.scaler")
    m.NOPScaler = NOPScaler
    m.MeanScaler = NOPScaler  # non usato: normalization="none"
    sys.modules["gluonts.torch.modules.scaler"] = m
    sys.path.insert(0, repo_src)
    pk = types.ModuleType("uncond_ts_diff.model")
    pk.__path__ = [os.path.join(repo_src, "uncond_ts_diff", "model")]
    sys.modules["uncond_ts_diff.model"] = pk


def main(a):
    import torch
    seed_all(a.seed)
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if a.device == "cuda" and dev.type != "cuda":
        raise RuntimeError("cuda richiesta ma non disponibile")
    repo_src = os.path.join(ALGO, "unconditional-time-series-diffusion", "src")
    _shims(repo_src)
    from uncond_ts_diff.model.diffusion.tsdiff import TSDiff
    from uncond_ts_diff.configs import diffusion_small_config

    x, _ = load_npz(a.data)
    N, T, D = x.shape
    z, lo, hi = minmax_fit(x)
    n = a.n_samples or N
    out = np.zeros((n, T, D), dtype="float32")
    bs = 64
    for d in range(D):
        model = TSDiff(**diffusion_small_config, context_length=T, prediction_length=0, freq=None,
                       normalization="none", use_features=False, use_lags=False, init_skip=True, lr=1e-3).to(dev)
        opt = model.configure_optimizers()[0][0]
        xd = torch.tensor(z[:, :, d:d + 1], dtype=torch.float32)
        model.train()
        for ep in range(a.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, bs):
                xb = xd[perm[i:i + bs]].to(dev)
                opt.zero_grad()
                B = xb.shape[0]
                batch = {"past_target": xb[:, :, 0], "past_observed_values": torch.ones_like(xb[:, :, 0]),
                         "future_target": xb.new_zeros(B, 0), "feat_static_cat": None, "feat_static_real": None,
                         "past_time_feat": xb.new_zeros(B, T, 0), "future_time_feat": None}
                loss = model.training_step(batch, 0)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        model.eval()
        chunks = []
        for i in range(0, n, 256):
            chunks.append(model.sample_n(num_samples=min(256, n - i)))
        out[:, :, d] = np.concatenate(chunks, 0)
    save_out(a.out, minmax_inv(out, lo, hi))
    return n


if __name__ == "__main__":
    run_worker(main, std_parser(__doc__).parse_args())
