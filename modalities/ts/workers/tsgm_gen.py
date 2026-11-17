"""Worker per i generatori di tsgm (Keras 3, backend torch). Un solo file, --method sceglie il modello:
gan, wavegan, cgan, betavae, cbetavae, timegan, ddpm.
Dati scalati per canale in [-1, 1] (GAN) o [0, 1] (VAE, DDPM, TimeGAN) e poi riportati alla scala originale."""
import os
import sys
from pathlib import Path

# TimeGAN di tsgm non funziona con il backend torch (bug nella libreria: gradienti .numpy()): gira con
# TensorFlow CPU nell'ambiente envs/ts_tf. Gli altri metodi usano il backend torch (envs/ts).
_METHOD = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else ""
os.environ["KERAS_BACKEND"] = "tensorflow" if _METHOD == "timegan" else "torch"
for _s in (sys.stdout, sys.stderr):  # il chiamante legge i pipe con la codifica locale: solo ASCII
    try:
        _s.reconfigure(encoding="ascii", errors="backslashreplace")
    except Exception:
        pass
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Algorithms" / "tsgm"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402


class Scaler:
    def __init__(self, lo, hi, x):
        self.lo, self.hi = lo, hi
        self.mn = x.min(axis=(0, 1), keepdims=True)
        self.mx = x.max(axis=(0, 1), keepdims=True)
        self.rng = np.where(self.mx - self.mn < 1e-8, 1.0, self.mx - self.mn)

    def fwd(self, x):
        return (self.lo + (x - self.mn) / self.rng * (self.hi - self.lo)).astype("float32")

    def inv(self, z):
        return (np.clip(z, self.lo, self.hi) - self.lo) / (self.hi - self.lo) * self.rng + self.mn


def to_np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t)


def loader(*arrays, batch_size):
    if os.environ["KERAS_BACKEND"] == "tensorflow":
        import tensorflow as tf
        ds = tf.data.Dataset.from_tensor_slices(arrays if len(arrays) > 1 else arrays[0])
        return ds.shuffle(1024).batch(batch_size)
    import torch
    ds = torch.utils.data.TensorDataset(*[torch.from_numpy(a) for a in arrays])
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


def run(a):
    import keras
    import tsgm
    from tsgm.models.architectures import zoo
    np.random.seed(a.seed)
    keras.utils.set_random_seed(a.seed)
    z = np.load(a.data)
    x = z["data"].astype("float32")
    y = z["labels"] if "labels" in z.files else None
    N, T, D = x.shape
    n_out = a.n_samples or N
    bs = min(a.batch_size, N)
    lat = a.latent_dim
    m = a.method
    adam = lambda: keras.optimizers.Adam(learning_rate=a.lr)  # noqa: E731
    out_labels = None

    if m in ("gan", "cgan"):
        sc = Scaler(-1, 1, x)
        xs = sc.fwd(x)
        if m == "gan":
            arch = zoo["cgan_base_c4_l1"](seq_len=T, feat_dim=D, latent_dim=lat, output_dim=0)
            model = tsgm.models.cgan.GAN(discriminator=arch.discriminator, generator=arch.generator, latent_dim=lat)
            data = loader(xs, batch_size=bs)
        else:
            if y is None:
                raise ValueError("cgan richiede le etichette (`labels` nel .npz)")
            classes, yi = np.unique(y, return_inverse=True)
            oh = np.eye(len(classes), dtype="float32")[yi]
            arch = zoo["cgan_base_c4_l1"](seq_len=T, feat_dim=D, latent_dim=lat, output_dim=len(classes))
            model = tsgm.models.cgan.ConditionalGAN(discriminator=arch.discriminator, generator=arch.generator,
                                                    latent_dim=lat)
            data = loader(xs, oh, batch_size=bs)
        model.compile(d_optimizer=adam(), g_optimizer=adam(),
                      loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
        model.fit(data, epochs=a.epochs, verbose=0)
        if m == "gan":
            g = to_np(model.generate(n_out))
        else:
            pick = np.random.default_rng(a.seed).integers(0, N, n_out)  # etichette dalla distribuzione reale
            g = to_np(model.generate(oh[pick]))
            out_labels = y[pick]
        gen = sc.inv(g)

    elif m == "wavegan":
        # tsgm: WaveGAN e' univariato (generatore a 1 canale): un modello per canale.
        sc = Scaler(-1, 1, x)
        xs = sc.fwd(x)
        chans = []
        for d in range(D):
            arch = zoo["wavegan"](seq_len=T, feat_dim=1, latent_dim=lat, output_dim=1)
            model = tsgm.models.cgan.GAN(discriminator=arch.discriminator, generator=arch.generator, latent_dim=lat)
            model.compile(d_optimizer=adam(), g_optimizer=adam(),
                          loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
            model.fit(loader(xs[:, :, d:d + 1], batch_size=bs), epochs=a.epochs, verbose=0)
            g = to_np(model.generate(n_out))
            if g.shape[1:] != (T, 1):
                raise ValueError(f"WaveGAN ha prodotto forma {g.shape}, attesa (n,{T},1): seq_len non supportata")
            chans.append(g)
        gen = sc.inv(np.concatenate(chans, axis=2))

    elif m in ("betavae", "cbetavae"):
        sc = Scaler(0, 1, x)
        xs = sc.fwd(x)
        if m == "betavae":
            arch = zoo["vae_conv5"](seq_len=T, feat_dim=D, latent_dim=lat)
            model = tsgm.models.cvae.BetaVAE(arch.encoder, arch.decoder, beta=a.beta)
            model.compile(optimizer=adam())
            model.fit(xs, epochs=a.epochs, batch_size=bs, verbose=0)
            g = to_np(model.generate(n_out))
        else:
            if y is None:
                raise ValueError("cbetavae richiede le etichette (`labels` nel .npz)")
            classes, yi = np.unique(y, return_inverse=True)
            oh = np.eye(len(classes), dtype="float32")[yi]
            arch = zoo["cvae_conv5"](seq_len=T, feat_dim=D, latent_dim=lat, output_dim=len(classes))
            model = tsgm.models.cvae.cBetaVAE(arch.encoder, arch.decoder, latent_dim=lat, temporal=False,
                                              beta=a.beta)
            model.compile(optimizer=adam())
            model.fit(xs, oh, epochs=a.epochs, batch_size=bs, verbose=0)
            pick = np.random.default_rng(a.seed).integers(0, N, n_out)
            g = to_np(model.generate(oh[pick])[0])
            out_labels = y[pick]
        gen = sc.inv(g)

    elif m == "ddpm":
        sc = Scaler(0, 1, x)
        xs = sc.fwd(x)
        arch = zoo["ddpm_denoiser"](seq_len=T, feat_dim=D)
        ema = zoo["ddpm_denoiser"](seq_len=T, feat_dim=D).model
        model = tsgm.models.ddpm.DDPM(arch.model, ema, a.timesteps)
        model.compile(optimizer=adam())
        model.fit(xs, epochs=a.epochs, batch_size=bs, verbose=0)
        gen = sc.inv(to_np(model.generate(n_out)))

    elif m == "timegan":
        sc = Scaler(0, 1, x)
        xs = sc.fwd(x)
        model = tsgm.models.timeGAN.TimeGAN(seq_len=T, module="gru", hidden_dim=a.hidden_dim, n_features=D,
                                            n_layers=2, batch_size=bs)
        model.compile()
        model.fit(loader(xs, batch_size=bs), epochs=a.epochs)
        gen = sc.inv(to_np(model.generate(n_out)))
    else:
        raise ValueError(f"metodo sconosciuto: {m}")

    gen = np.asarray(gen, dtype="float32")
    if gen.shape[1:] != (T, D) or not np.isfinite(gen).all():
        raise ValueError(f"output non valido: forma {gen.shape}, finito={np.isfinite(gen).all()}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kw = {"data": gen}
    if out_labels is not None:
        kw["labels"] = out_labels
    np.savez(a.out, **kw)
    return gen.shape[0]


if __name__ == "__main__":
    p = std_parser("tsgm generators")
    p.add_argument("--method", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--timesteps", type=int, default=200)
    args = p.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    run_worker(run, args)
