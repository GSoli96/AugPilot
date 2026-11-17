"""Worker timeVAE (Desai et al. 2021; repo Algorithms/timeVAE, MIT), codice originale TensorFlow 2.16.
Gira nell'ambiente envs/ts_tf (TensorFlow CPU: da TF 2.11 non esiste TF con GPU su Windows nativo).
--method: vae_dense | vae_conv | timeVAE. Iperparametri del repo (config/hyperparameters.yaml),
batch 16. Dati scalati per canale in [0, 1] e poi riportati alla scala originale."""
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _s in (sys.stdout, sys.stderr):  # il chiamante legge i pipe con la codifica locale: solo ASCII
    try:
        _s.reconfigure(encoding="ascii", errors="backslashreplace")
    except Exception:
        pass
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Algorithms" / "timeVAE" / "src"))

import numpy as np  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402


def run(a):
    if a.device == "cuda":
        print("nota: timeVAE (TensorFlow) gira su CPU in questo ambiente", file=sys.stderr)
    from vae.vae_utils import instantiate_vae_model, set_seeds
    z = np.load(a.data)
    x = z["data"].astype("float32")
    N, T, D = x.shape
    mn = x.min(axis=(0, 1), keepdims=True)
    mx = x.max(axis=(0, 1), keepdims=True)
    rng = np.where(mx - mn < 1e-8, 1.0, mx - mn)
    xs = ((x - mn) / rng).astype("float32")
    kw = dict(latent_dim=a.latent_dim, hidden_layer_sizes=[50, 100, 200], reconstruction_wt=3.0)
    if a.method == "timeVAE":
        kw.update(trend_poly=0, custom_seas=None, use_residual_conn=True)
    vae = instantiate_vae_model(a.method, T, D, min(a.batch_size, N), **kw)
    set_seeds(a.seed)
    vae.fit_on_data(xs, max_epochs=a.epochs, verbose=0)
    n_out = a.n_samples or N
    g = np.asarray(vae.get_prior_samples(n_out))
    gen = (np.clip(g, 0, 1) * rng + mn).astype("float32")
    if gen.shape[1:] != (T, D) or not np.isfinite(gen).all():
        raise ValueError(f"output non valido: forma {gen.shape}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, data=gen)
    return gen.shape[0]


if __name__ == "__main__":
    p = std_parser("timeVAE")
    p.add_argument("--method", required=True, choices=["vae_dense", "vae_conv", "timeVAE"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--latent_dim", type=int, default=8)
    args = p.parse_args()
    run_worker(run, args)
