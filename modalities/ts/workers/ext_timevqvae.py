"""TimeVQVAE (Lee, Malacarne, Aune, AISTATS 2023; MIT) - codice ORIGINALE di Algorithms/TimeVQVAE.

Stadio 1: VQ-VAE nel dominio STFT (rami LF/HF); stadio 2: MaskGIT bidirezionale (prior sui token),
condizionato all'etichetta se `labels` e' presente (classifier-free). Il worker usa le classi del repo
senza modificarle; i passi di training sono calcolati da --epochs (un'epoca = un passaggio sui dati,
batch 32): stadio 1 e stadio 2 usano ciascuno --epochs epoche. Iperparametri del repo, tranne la
dimensione dei codebook (--codebook, default 256 invece di 1024: dataset piccoli).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import ALGO, load_npz, minmax_fit, minmax_inv, save_out, seed_all  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402

sys.path.insert(0, os.path.join(ALGO, "TimeVQVAE", "src"))


def main(a):
    import torch
    import torch.nn.functional as F
    from timevqvae.vqvae import VQVAE
    from timevqvae.maskgit import MaskGIT, PriorModelConfig

    seed_all(a.seed)
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if a.device == "cuda" and dev.type != "cuda":
        raise RuntimeError("cuda richiesta ma non disponibile")
    x, y = load_npz(a.data)
    N, T, D = x.shape
    z, lo, hi = minmax_fit(x)
    xt = torch.tensor(z).permute(0, 2, 1).contiguous()          # (N, C, L)
    ncls = 0
    if y is not None:
        classes, yi = np.unique(y, return_inverse=True)
        ncls = len(classes)
        cond = torch.tensor(yi, dtype=torch.long).unsqueeze(1)
    else:
        cond = torch.zeros(N, 1, dtype=torch.long)
    bs = 32
    spe = max(1, int(np.ceil(N / bs)))
    steps = max(1, a.epochs * spe)
    cb = a.codebook

    vq = VQVAE(in_channels=D, input_length=T, n_fft=4, init_dim=4, hid_dim=128,
               downsampled_width_l=8, downsampled_width_h=32, encoder_n_resnet_blocks=2,
               decoder_n_resnet_blocks=2, codebook_size_l=cb, codebook_size_h=cb,
               kmeans_init=True, codebook_dim=8).to(dev)
    xt_d = xt.to(dev)
    # stadio 1 (la loss e' la somma delle ricostruzioni e dei termini VQ, come nel repo)
    opt = torch.optim.AdamW(vq.parameters(), lr=5e-3)
    vq.train()
    for s in range(steps):
        idx = torch.randint(0, N, (min(bs, N),), device=dev)
        out = vq(xt_d[idx])
        loss = sum(out.recons_loss.values()) + out.vq_losses["LF"]["loss"] + out.vq_losses["HF"]["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
    vq.eval()

    pc = lambda h, l, hd: PriorModelConfig(hidden_dim=h, n_layers=l, heads=hd, ff_mult=1, use_rmsnorm=True,
                                           p_unconditional=0.2, model_dropout=0.3, emb_dropout=0.3)
    mg = MaskGIT(vqvae=vq, lf_choice_temperature=10.0, hf_choice_temperature=0.0, lf_num_sampling_steps=10,
                 hf_num_sampling_steps=10, lf_codebook_size=cb, hf_codebook_size=cb, transformer_embedding_dim=128,
                 lf_prior_model_config=pc(128, 4, 2), hf_prior_model_config=pc(32, 1, 1),
                 classifier_free_guidance_scale=1.0, n_classes=ncls).to(dev)
    params = [p for p in mg.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(params, lr=5e-3)
    mg.train(); mg.vqvae.eval()
    cond_d = cond.to(dev)
    for s in range(steps):
        idx = torch.randint(0, N, (min(bs, N),), device=dev)
        l = mg(xt_d[idx], cond_d[idx] if ncls else None).total_mask_prediction_loss
        opt2.zero_grad(); l.backward(); opt2.step()
    mg.eval()

    n = a.n_samples or N
    outs, labs = [], []
    with torch.no_grad():
        if ncls:
            rng = np.random.default_rng(a.seed)
            lab_idx = rng.choice(len(classes), n, p=np.bincount(yi) / N)
            groups = [(k, int((lab_idx == k).sum())) for k in range(ncls)]
        else:
            groups = [(None, n)]
        for k, m in groups:
            done = 0
            while done < m:
                b = min(64, m - done)
                tl, th = mg.iterative_decoding(num_samples=b, mode="cosine", class_condition=k, device=dev)
                g = mg.decode_token_ind_to_timeseries(tl, "lf") + mg.decode_token_ind_to_timeseries(th, "hf")
                outs.append(g.permute(0, 2, 1).cpu().numpy())
                if ncls:
                    labs += [classes[k]] * b
                done += b
    g = np.concatenate(outs, 0)
    g = np.clip(g, -0.5, 1.5)  # i token possono produrre valori poco oltre [0,1]; taglio solo gli estremi
    save_out(a.out, minmax_inv(g, lo, hi), np.array(labs) if ncls else None)
    return len(g)


if __name__ == "__main__":
    p = std_parser(__doc__)
    p.add_argument("--codebook", type=int, default=256)
    run_worker(main, p.parse_args())
