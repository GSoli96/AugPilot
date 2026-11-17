"""ImagenTime (Naiman et al., NeurIPS 2024) - codice ORIGINALE di Algorithms/ImagenTime
(models.model.ImagenTime, sampler EDM di Karras et al.), diffusione su immagini ottenute con delay embedding.

Il worker riproduce il ciclo di training di run_unconditional.py (AdamW lr 1e-4, wd 1e-5, EMA, U-Net 128 canali,
18 passi di campionamento) senza modificare il repo, con questi adattamenti esterni:
- nel repo mancano le cartelle `data/data_provider` e `data/long_range` (importate da utils/utils_data.py ma
  non usate dal percorso incondizionato): il worker registra due moduli vuoti con quei nomi.
- delay/embedding/img_resolution derivati da T (8x8 se T<=40, altrimenti 16x16); T=24 -> delay 3, embedding 8
  come nella config `stocks` del repo. canali immagine = D.
- --epochs = passaggi sul dataset (batch min(128, N)).
Il codice originale usa `.cuda()` fisso in img_to_ts: richiede GPU.
Licenza: il repo non ha LICENSE; models/networks.py deriva da NVIDIA EDM (CC BY-NC-SA 4.0, uso non commerciale).
"""
import argparse
import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import ALGO, load_npz, minmax_fit, minmax_inv, save_out, seed_all  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402


def _stub_missing_data_modules():
    for name, attr in (("data.data_provider", None), ("data.data_provider.data_factory", "data_provider"),
                       ("data.long_range", "parse_datasets")):
        m = types.ModuleType(name)
        if attr:
            setattr(m, attr, None)
        sys.modules[name] = m


def main(a):
    import torch
    if not torch.cuda.is_available() or a.device != "cuda":
        raise RuntimeError("ImagenTime: il codice originale usa .cuda() in img_to_ts: serve --device cuda")
    repo = os.path.join(ALGO, "ImagenTime")
    sys.path.insert(0, repo)
    _stub_missing_data_modules()
    from models.model import ImagenTime
    from models.sampler import DiffusionProcess

    seed_all(a.seed)
    x, _ = load_npz(a.data)
    N, T, D = x.shape
    if T < 8:
        raise ValueError(f"T={T} troppo corto per l'embedding 8x8")
    if T <= 40:
        res, emb = 8, 8
        delay = max(1, math.ceil((T - emb) / 6))
    else:
        res, emb = 16, 16
        delay = max(1, math.ceil((T - emb) / 14))
    z, lo, hi = minmax_fit(x)
    args = argparse.Namespace(
        seed=a.seed, beta1=1e-5, betaT=1e-2, deterministic=False, epochs=a.epochs, batch_size=min(128, N),
        learning_rate=1e-4, weight_decay=1e-5, seq_len=T, use_stft=False, delay=delay, embedding=emb,
        img_resolution=res, input_channels=D, unet_channels=128, ch_mult=[1, 2, 2, 2],
        attn_resolution=[8, 4, 2], diffusion_steps=18, ema=True, ema_warmup=100, device="cuda")
    model = ImagenTime(args=args, device="cuda").to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    xt = torch.tensor(z, dtype=torch.float32)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, args.batch_size):
            xb = xt[perm[i:i + args.batch_size]].to("cuda")
            x_img = model.ts_to_img(xb)
            opt.zero_grad()
            loss, _ = model.loss_fn(x_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            model.on_train_batch_end()
    model.eval()
    n = a.n_samples or N
    outs = []
    with torch.no_grad(), model.ema_scope():
        proc = DiffusionProcess(args, model.net, (D, res, res))
        for i in range(0, n, 128):
            b = min(128, n - i)
            model.ts_img.img_shape = None
            # img_shape serve a img_to_ts: lo fissa ts_to_img su un batch di dati reali della stessa forma
            model.ts_to_img(xt[:b].to("cuda"))
            img = proc.sampling(sampling_number=b)
            outs.append(model.img_to_ts(img).float().cpu().numpy())
    g = np.concatenate(outs, 0)
    save_out(a.out, minmax_inv(g, lo, hi))
    return len(g)


if __name__ == "__main__":
    run_worker(main, std_parser(__doc__).parse_args())
