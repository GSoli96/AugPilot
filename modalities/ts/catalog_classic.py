"""Catalogo delle aumentazioni classiche per serie temporali (worker unico classic.py --method).
Lo stato "executable" si assegna solo dopo un'esecuzione reale (vedi EXECUTED)."""
from modalities import AlgoSpec, register

WORKER = "modalities/ts/workers/classic.py"
TSGM = "reimplemented in numpy/scipy; inspired by tsgm/augmentations.py (Apache-2.0)"
OWN = "own implementation in numpy/scipy"

# chiave: (nome, strategia, sorgente, licenza, paper, note)
_METHODS = {
    "jitter": ("Jittering", "noise", TSGM, "Apache-2.0 (inspiration)", "Um et al., ICMI 2017", "Gaussian noise at 5% of the channel std"),
    "scaling": ("Scaling", "magnitude", TSGM, "Apache-2.0 (inspiration)", "Um et al., ICMI 2017", "per-channel factor N(1,0.1)"),
    "magnitude_warp": ("Magnitude warping", "magnitude-warping", TSGM, "Apache-2.0 (inspiration)", "Um et al., ICMI 2017", "spline with 4 knots, sigma 0.2"),
    "time_warp": ("Time warping", "time-warping", TSGM, "Apache-2.0 (inspiration)", "Um et al., ICMI 2017", "smooth speed curve, sigma 0.3"),
    "window_slice": ("Window slicing", "cropping", OWN, "-", "Le Guennec et al., AALTD 2016", "90% window, stretched back"),
    "window_warp": ("Window warping", "window-warping", OWN, "-", "Le Guennec et al., AALTD 2016", "10% window x0.5 or x2"),
    "permutation": ("Permutation", "permutation", TSGM, "Apache-2.0 (inspiration)", "Um et al., ICMI 2017", "4 shuffled segments"),
    "rotation": ("Rotation / flipping", "rotation", OWN, "-", "Um et al., ICMI 2017", "orthogonal rotation of the channels (D>1); flipping (D=1)"),
    "slice_shuffle": ("Slice-and-shuffle (cut-and-mix)", "recombination", OWN, "-", "CutMix-style for time series", "segments taken from series of the same class"),
    "dba": ("DTW Barycenter Averaging", "dtw-averaging", OWN, "-", "Petitjean et al., Pattern Recognition 2011; Forestier et al., ICDM 2017", "4 series of the same class, Dirichlet weights, 3 iterations, 20% band"),
    "smote": ("SMOTE for time series", "oversampling", OWN, "-", "Chawla et al., JAIR 2002", "interpolation toward one of the 5 same-class neighbors"),
    "mixup": ("Intra-class mixup", "interpolation", OWN, "-", "Zhang et al., ICLR 2018", "lambda ~ Beta(0.4,0.4)"),
    "block_bootstrap": ("Moving block bootstrap", "bootstrap", OWN, "-", "Kunsch, Ann. Stat. 1989", "T/4 blocks of the same series"),
    "decomp_bootstrap": ("Decomposition + residual bootstrap", "decomposition", OWN, "-", "Bergmeir et al., IJF 2016 (idea); STL: Cleveland 1990", "moving-average trend + per-phase seasonality + block-resampled residuals (not statsmodels' STL)"),
    "ssa": ("SSA + resampled residuals", "subspace", OWN, "-", "Golyandina & Zhigljavsky, Singular Spectrum Analysis, Springer 2013", "rank r from 95% of the energy (computed on the first series)"),
    "pca": ("PCA with perturbed scores", "subspace", OWN, "-", "Jolliffe, Principal Component Analysis", "global PCA with 95% variance, noise 0.3*std per component"),
    "fourier_surrogate": ("Fourier surrogate (random phases)", "spectral", OWN, "-", "Theiler et al., Physica D 1992", "same random phases on all channels"),
    "iaaft": ("IAAFT", "spectral", OWN, "-", "Schreiber & Schmitz, PRL 1996", "30 iterations, per channel (cross-channel correlation not preserved)"),
}

# Tassonomia del campo `task` (corretta il 2026-11-16: prima tutto tranne smote era "view", e il consigliere con
# task="generation" escludeva bootstrap e DBA, che invece producono serie nuove).
#   oversampling: smote
#   generation:   metodi che ricombinano, ricampionano o sintetizzano serie nuove (non il solo ritocco di una serie)
#   view:         perturbazione di una singola serie (rumore, scala, deformazioni, ritaglio, permutazione, rotazione)
_GENERATION = {"dba", "slice_shuffle", "mixup", "block_bootstrap", "decomp_bootstrap", "ssa", "pca",
               "fourier_surrogate", "iaaft"}


def _task(key, strat):
    if strat == "oversampling":
        return "oversampling"
    return "generation" if key in _GENERATION else "view"


# Metodi eseguiti con successo end-to-end (run_modality.py / run_algorithm) il 2026-11-09 su sine_small e stock.
EXECUTED = set(_METHODS)


def load():
    for key, (name, strat, src, lic, paper, notes) in _METHODS.items():
        register(AlgoSpec(key=key, modality="ts", name=name, strategy=strat, task=_task(key, strat),
                          env="main", worker=WORKER, status="executable" if key in EXECUTED else "wired",
                          source=src, license=lic, paper=paper, needs_gpu=False, notes=notes,
                          extra={"method": key}))
