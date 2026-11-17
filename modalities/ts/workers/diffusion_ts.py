"""Worker Diffusion-TS (Yuan & Qiao, ICLR 2024; repo Algorithms/diffusion-ts, MIT).
Usa il modello `Diffusion_TS` del repo con un ciclo di addestramento minimo (Adam + EMA), al posto di
`engine/solver.py` che richiede file CSV e config YAML. Dati scalati per canale in [-1, 1] e poi
de-normalizzati. Parametri del repo (stocks.yaml) tranne timesteps/epoche, ridotti per i test."""
import copy
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # il chiamante legge i pipe con la codifica locale: solo ASCII
    try:
        _s.reconfigure(encoding="ascii", errors="backslashreplace")
    except Exception:
        pass
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Algorithms" / "diffusion-ts"))

import numpy as np  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402


def run(a):
    import torch
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu"
    if a.device == "cuda" and dev == "cpu":
        raise RuntimeError("--device cuda richiesto ma CUDA non disponibile")
    z = np.load(a.data)
    x = z["data"].astype("float32")
    N, T, D = x.shape
    mn = x.min(axis=(0, 1), keepdims=True)
    mx = x.max(axis=(0, 1), keepdims=True)
    rng = np.where(mx - mn < 1e-8, 1.0, mx - mn)
    xs = (2 * (x - mn) / rng - 1).astype("float32")

    model = Diffusion_TS(seq_length=T, feature_size=D, n_layer_enc=2, n_layer_dec=2, d_model=64,
                         timesteps=a.timesteps, sampling_timesteps=a.timesteps, loss_type="l1",
                         beta_schedule="cosine", n_heads=4, mlp_hidden_times=4, attn_pd=0.0, resid_pd=0.0,
                         kernel_size=1, padding_size=0).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.96))
    data = torch.from_numpy(xs)
    bs = min(a.batch_size, N)
    step = 0
    for ep in range(a.epochs):
        perm = torch.randperm(N)
        model.train()
        for i in range(0, N, bs):
            b = data[perm[i:i + bs]].to(dev)
            loss = model(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            decay = min(a.ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
    n_out = a.n_samples or N
    outs = []
    with torch.no_grad():
        for i in range(0, n_out, 256):
            outs.append(ema.generate_mts(batch_size=min(256, n_out - i)).cpu().numpy())
    g = np.concatenate(outs, axis=0)
    gen = ((np.clip(g, -1, 1) + 1) / 2 * rng + mn).astype("float32")
    if gen.shape[1:] != (T, D) or not np.isfinite(gen).all():
        raise ValueError(f"output non valido: forma {gen.shape}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, data=gen)
    return gen.shape[0]


if __name__ == "__main__":
    p = std_parser("Diffusion-TS")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--ema_decay", type=float, default=0.995)
    args = p.parse_args()
    run_worker(run, args)
