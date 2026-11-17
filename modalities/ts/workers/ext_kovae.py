"""KoVAE (Naiman et al., ICLR 2024) - codice ORIGINALE di Algorithms/kovae (classe models.kovae.KoVAE).

VAE per serie temporali con prior lineare ispirato a Koopman. Il worker riproduce il ciclo di
training di run_regular.py (Adam lr 7e-4, batch 64, w_kl 0.007, w_pred_prior 0.005, z_dim 16, hidden 20,
3 livelli GRU) senza modificare il repo. Campiona dal prior appreso.
Il repo ha `.cuda()` scritto nel codice: richiede una GPU (nessun ripiego su CPU).
Licenza del repo: nessun file LICENSE (non verificata).
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import ALGO, load_npz, minmax_fit, minmax_inv, save_out, seed_all  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402

sys.path.insert(0, os.path.join(ALGO, "kovae"))


def main(a):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("KoVAE: il codice originale usa .cuda(): serve una GPU")
    if a.device != "cuda":
        raise RuntimeError("KoVAE gira solo con --device cuda (codice originale con .cuda() fisso)")
    from models.kovae import KoVAE

    seed_all(a.seed)
    x, y = load_npz(a.data)
    N, T, D = x.shape
    z, lo, hi = minmax_fit(x)
    args = argparse.Namespace(epochs=a.epochs, lr=7e-4, weight_decay=0., seed=a.seed, pinv_solver=False,
                              batch_size=64, seq_len=T, missing_value=0., batch_norm=True, num_layers=3,
                              z_dim=16, hidden_dim=20, num_steps=1, w_rec=1., w_kl=.007, w_pred_prior=.005,
                              inp_dim=D, device=torch.device("cuda:0"))
    model = KoVAE(args).to(args.device)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    xt = torch.tensor(z, dtype=torch.float32)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:  # BatchNorm non accetta un solo campione
                continue
            X = xt[idx].to(args.device)
            opt.zero_grad()
            x_rec, Z_enc, Z_prior = model(X)
            losses = model.loss(X, x_rec, Z_enc, Z_prior)
            losses[0].backward()
            opt.step()
    model.eval()
    n = a.n_samples or N
    outs = []
    with torch.no_grad():
        for i in range(0, n, 256):
            outs.append(model.sample_data(min(256, n - i)).cpu().numpy())
    g = np.concatenate(outs, 0)
    save_out(a.out, minmax_inv(g, lo, hi))
    return len(g)


if __name__ == "__main__":
    run_worker(main, std_parser(__doc__).parse_args())
