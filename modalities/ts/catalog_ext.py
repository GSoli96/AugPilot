"""Catalogo dei generatori/augmentation di serie temporali "estesi" (strategie diverse da classic e catalog_gen).

Stato "executable" solo dopo l'esecuzione end-to-end di
`python run_modality.py --modality ts --algorithms <chiave> --datasets sine_small --epochs 2 --device cuda`
(vedi EXECUTED). Tutti i worker girano nell'ambiente envs/ts. I campi `notes` dichiarano gli adattamenti:
"originale" = classi del repo importate senza modifiche; "riscrittura" = idea riscritta (non codice originale).
"""
from modalities.base import AlgoSpec, register

W = "modalities/ts/workers/"

# chiave: (nome, strategia, worker, extra, repo, licenza, paper, gpu, note)
_ALGOS = {
    "timevqvae": (
        "TimeVQVAE", "vq-vae", W + "ext_timevqvae.py", {},
        "Algorithms/TimeVQVAE (https://github.com/ML4ITS/TimeVQVAE)", "MIT",
        "Lee, Malacarne, Aune, AISTATS 2023 (arXiv:2303.04743)", True,
        "Original code (VQVAE + MaskGIT). Label-conditioned when labels are present. Codebook 256 (repo default 1024)."),
    "kovae": (
        "KoVAE (Koopman VAE)", "vae-koopman", W + "ext_kovae.py", {},
        "Algorithms/kovae (https://github.com/azencot-group/kovae)", "no license in the repo",
        "Naiman, Erichson, Ren, Mahoney, Azencot, ICLR 2024 (arXiv:2310.02619)", True,
        "Original code (models.kovae.KoVAE); training loop rewritten from run_regular.py. Requires a GPU (.cuda() in the repo)."),
    "sigcwgan": (
        "SigCWGAN (Conditional Sig-Wasserstein GAN)", "gan-signature", W + "ext_sigcwgan.py", {},
        "Algorithms/Conditional-Sig-Wasserstein-GANs (https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs)",
        "no license in the repo", "Ni, Szpruch, Wiese, Liao, Xiao, arXiv:2006.05421 (2020)", True,
        "Original code except `signatory` (cannot be compiled here): replaced by a minimal pure-torch module we wrote. "
        "Autoregressive generation conditioned on the prefix (3 steps) of a real series: the prefix is copied."),
    "imagentime": (
        "ImagenTime", "diffusion-image", W + "ext_imagentime.py", {},
        "Algorithms/ImagenTime (https://github.com/azencot-group/ImagenTime)",
        "no license in the repo; networks.py from NVIDIA EDM (CC BY-NC-SA 4.0)",
        "Naiman, Berman, Pemper, Arbiv, Fadlon, Azencot, NeurIPS 2024", True,
        "Original code (ImagenTime + EDM sampler); two `data.*` modules missing from the repo replaced by empty stubs. Requires a GPU (.cuda() in the repo)."),
    "tsdiff": (
        "TSDiff (unconditional S4 diffusion)", "diffusion-s4", W + "ext_tsdiff.py", {},
        "Algorithms/unconditional-time-series-diffusion (https://github.com/amazon-science/unconditional-time-series-diffusion)",
        "Apache-2.0", "Kollovieh, Ansari, Bohlke-Schneider, Zschiegner, Wang, Wang, NeurIPS 2023 (arXiv:2307.11494)", True,
        "Original code (TSDiff, S4 backbone) with gluonts 0.17 and a minimal scaler in place of the removed one. "
        "Univariate: one model per channel, cross-channel correlations not modeled. S4 without pykeops (slow kernel)."),
    "simpsi": (
        "SimPSI (spectrum-preserving augmentation)", "spectral-preserving", W + "ext_simpsi.py", {},
        "Algorithms/simpsi (https://github.com/hyun-ryu/simpsi)", "Apache-2.0",
        "Ryu, Kim et al., AAAI 2024 (arXiv:2312.05790)", False,
        "Base augmentation from the repo; spectral mix rule REWRITTEN with a magnitude-preservation map (unlearned equalizer, no classifier)."),
    "uchida-spawner": (
        "SPAWNER (uchidalab)", "dtw-averaging", W + "ext_uchida.py", {"method": "spawner"},
        "Algorithms/time_series_augmentation (https://github.com/uchidalab/time_series_augmentation)", "Apache-2.0",
        "Iwana & Uchida, PLOS ONE 2021; SPAWNER: Kamycki et al., 2019", False,
        "Original code. Uses labels (a single class if missing). The repo has an index error in the prototype selection (see the worker docstring)."),
    "uchida-wdba": (
        "Weighted DBA (uchidalab)", "dtw-averaging", W + "ext_uchida.py", {"method": "wdba"},
        "Algorithms/time_series_augmentation (https://github.com/uchidalab/time_series_augmentation)", "Apache-2.0",
        "Forestier, Petitjean, Dau, Webb, Keogh, ICDM 2017; Iwana & Uchida, PLOS ONE 2021", False,
        "Original code: medoid of 6 series of the class + exponential weights on the DTW distances. Different from the `dba` method in classic (Dirichlet weights)."),
    "uchida-rgw": (
        "Random guided warping (uchidalab)", "dtw-warping", W + "ext_uchida.py", {"method": "rgw"},
        "Algorithms/time_series_augmentation (https://github.com/uchidalab/time_series_augmentation)", "Apache-2.0",
        "Iwana & Uchida, PLOS ONE 2021 (arXiv:2004.08780)", False,
        "Original code. Same prototype index error as SPAWNER."),
    "uchida-dgw": (
        "Discriminative guided warping (uchidalab)", "dtw-warping", W + "ext_uchida.py", {"method": "dgw"},
        "Algorithms/time_series_augmentation (https://github.com/uchidalab/time_series_augmentation)", "Apache-2.0",
        "Iwana & Uchida, PLOS ONE 2021 (arXiv:2004.08780)", False,
        "Original code. Requires at least two classes to have negative prototypes (otherwise the series is left unchanged)."),
}

# Eseguiti con successo end-to-end il 2026-11-09 (sine_small, 2 epoche, cuda); aggiornato dopo i test.
EXECUTED = {"timevqvae", "kovae", "sigcwgan", "imagentime", "tsdiff", "simpsi",
            "uchida-spawner", "uchida-wdba", "uchida-rgw", "uchida-dgw"}


def load():
    for key, (name, strat, worker, extra, src, lic, paper, gpu, notes) in _ALGOS.items():
        register(AlgoSpec(modality="ts", key=key, name=name, strategy=strat, env="ts", worker=worker,
                          status="executable" if key in EXECUTED else "wired", source=src, license=lic,
                          paper=paper, needs_gpu=gpu, notes=notes, extra=extra))
