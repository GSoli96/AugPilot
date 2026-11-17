"""Catalogo dei generatori di serie temporali (parte generativa).

Stato "executable" solo dopo l'esecuzione end-to-end di
`python run_modality.py --modality ts --algorithms <chiave> --datasets sine_small --epochs 2 --device cuda --no_eval`.
Ambienti: ts (torch 2.6 + Keras 3 backend torch), ts_tf (TensorFlow 2.16 CPU).
"""
from modalities.base import AlgoSpec, register

TSGM = "https://github.com/AlexanderVNikitin/tsgm"
TSGM_W = "modalities/ts/workers/tsgm_gen.py"
TVAE_W = "modalities/ts/workers/timevae.py"
DTS_W = "modalities/ts/workers/diffusion_ts.py"

# (chiave, method, nome, strategia, env, paper, note)
_TSGM = [
    ("tsgm-gan", "gan", "TSGM GAN (conv, unconditional)", "gan", "ts",
     "Goodfellow et al., NIPS 2014 (tsgm conv architecture)", ""),
    ("tsgm-wavegan", "wavegan", "TSGM WaveGAN", "gan", "ts",
     "Donahue et al., ICLR 2019 (arXiv:1802.04208)",
     "tsgm's WaveGAN is univariate: the worker trains one model per channel."),
    ("tsgm-cgan", "cgan", "TSGM Conditional GAN", "gan", "ts",
     "Mirza & Osindero 2014 (tsgm conv architecture)", "Requires `labels` in the .npz; the output has `labels`."),
    ("tsgm-betavae", "betavae", "TSGM BetaVAE (conv)", "vae", "ts", "Higgins et al., ICLR 2017", ""),
    ("tsgm-cbetavae", "cbetavae", "TSGM conditional BetaVAE", "vae", "ts",
     "Higgins et al., ICLR 2017 + label conditioning (tsgm)", "Requires `labels`; the output has `labels`."),
    ("tsgm-timegan", "timegan", "TSGM TimeGAN", "gan", "ts_tf",
     "Yoon et al., NeurIPS 2019",
     "With the torch backend tsgm.TimeGAN fails (library bug): it runs with TensorFlow CPU in ts_tf."),
    ("tsgm-ddpm", "ddpm", "TSGM DDPM (denoiser conv)", "diffusion", "ts", "Ho et al., NeurIPS 2020", ""),
]
_TSGM_GPU = {"timegan": False}

_TVAE = [
    ("timevae-dense", "vae_dense", "timeVAE: dense VAE (repo baseline)", "vae"),
    ("timevae-conv", "vae_conv", "timeVAE: convolutional VAE (repo baseline)", "vae"),
    ("timevae", "timeVAE", "timeVAE (decoder with trend/seasonality, residual)", "vae"),
]


def load():
    status = _STATUS
    for key, method, name, strat, env, paper, notes in _TSGM:
        register(AlgoSpec(modality="ts", key=key, name=name, strategy=strat, env=env, worker=TSGM_W,
                          status=status.get(key, "wired"), source=TSGM, license="Apache-2.0", paper=paper,
                          needs_gpu=_TSGM_GPU.get(method, True), notes=notes, extra={"method": method}))
    for key, method, name, strat in _TVAE:
        register(AlgoSpec(modality="ts", key=key, name=name, strategy=strat, env="ts_tf", worker=TVAE_W,
                          status=status.get(key, "wired"), source="Algorithms/timeVAE (local repo)",
                          license="MIT", paper="Desai et al., arXiv:2111.08095 (2021)", needs_gpu=False,
                          notes="Original TensorFlow 2.16 code: runs on CPU (no TF-GPU on native Windows).",
                          extra={"method": method}))
    register(AlgoSpec(modality="ts", key="diffusion-ts", name="Diffusion-TS", strategy="diffusion", env="ts",
                      worker=DTS_W, status=status.get("diffusion-ts", "wired"),
                      source="Algorithms/diffusion-ts (https://github.com/Y-debug-sys/Diffusion-TS)", license="MIT",
                      paper="Yuan & Qiao, ICLR 2024 (arXiv:2403.01742)", needs_gpu=True,
                      notes="Minimal training loop (Adam+EMA) on the repo's model; does not use solver.py."))
    # Solo descritti: richiedono TensorFlow 1.x (non installabile con Python 3.10).
    register(AlgoSpec(modality="ts", key="timegan-orig", name="TimeGAN (original code)", strategy="gan",
                      status="described", source="Algorithms/TimeGAN (https://github.com/jsyoon0823/TimeGAN)",
                      license="Apache-2.0", paper="Yoon et al., NeurIPS 2019",
                      notes="Requires TensorFlow 1.15 (tf.contrib), not runnable with Python 3.10; "
                            "TimeGAN is runnable via tsgm-timegan."))
    register(AlgoSpec(modality="ts", key="rgan", name="RGAN / RCGAN", strategy="gan", status="described",
                      source="Algorithms/RGAN (https://github.com/ratschlab/RGAN)", license="MIT",
                      paper="Esteban, Hyland, Raetsch, arXiv:1706.02633 (2017)",
                      notes="Requires TensorFlow 1.x (tf.contrib, mod_core_rnn_cell_impl): not runnable here."))


_STATUS = {k: "executable" for k in ['tsgm-gan', 'tsgm-wavegan', 'tsgm-cgan', 'tsgm-betavae', 'tsgm-cbetavae', 'tsgm-timegan', 'tsgm-ddpm', 'timevae-dense', 'timevae-conv', 'timevae', 'diffusion-ts']}  # verificati il 2026-11-09 (sine_small, 2 epoche)
