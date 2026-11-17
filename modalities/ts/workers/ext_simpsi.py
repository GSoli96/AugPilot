"""SimPSI (Ryu et al., AAAI 2024; Apache-2.0) - augmentation che preserva l'informazione spettrale.

USA il codice ORIGINALE solo per le trasformazioni di base (classi Jitter, Scale, Shift, Magwarp, Timewarp,
Permute di Algorithms/simpsi/har_sleepedf/dataloader/augmentations.py, caricate da percorso). La regola di
SimPSI (AugBoostDeep: x_f_psi = x_f*boost + x_f_aug*(1-boost), dove boost e' la mappa di preservazione per
frequenza) e' RISCRITTA qui, perche' nel repo e' accoppiata a un equalizzatore appreso con un classificatore.
Adattamento dichiarato: mappa "magnitudo" con equalizzatore NON appreso, boost = |X_f| / max|X_f| (mediato sui
canali, per serie), cioe' si preservano le frequenze con piu' energia. Non ci sono classificatore ne' mappe di
salienza o "spectrum-preservative". Non si addestra nulla: --epochs/--device ignorati.
Dati: z-score per canale internamente (le sigma 0.5 del repo presuppongono dati standardizzati), poi inverso.
Per ogni serie: sequenza di augmentation (--augs, default "jitter scale shift magwarp") -> mix spettrale (rfft).
L'output ha le etichette della serie di partenza se presenti. n_samples default N.
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import ALGO, load_npz, save_out, seed_all  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402


def _load_aug():
    f = os.path.join(ALGO, "simpsi", "har_sleepedf", "dataloader", "augmentations.py")
    spec = importlib.util.spec_from_file_location("simpsi_har_augmentations", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(a):
    import torch
    seed_all(a.seed)
    aug = _load_aug()
    x, y = load_npz(a.data)
    N, T, D = x.shape
    mu = x.mean(axis=(0, 1), keepdims=True)
    sd = x.std(axis=(0, 1), keepdims=True) + 1e-8
    z = (x - mu) / sd
    n = a.n_samples or N
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(N, n, replace=n > N)
    xt = torch.tensor(z[idx]).permute(0, 2, 1).contiguous()  # (b, c, T)
    names = a.augs.split()
    xa = xt.clone()
    for nm in names:
        xa = getattr(aug, nm.capitalize())()(xa)
    xf = torch.fft.rfft(xt, dim=-1)
    xf_aug = torch.fft.rfft(xa, dim=-1)
    mag = xf.abs().mean(dim=1, keepdim=True)
    boost = mag / (mag.amax(dim=-1, keepdim=True) + 1e-8)  # (b, 1, F) in [0, 1]
    xf_psi = xf * boost + xf_aug * (1 - boost)
    g = torch.fft.irfft(xf_psi, n=T, dim=-1).permute(0, 2, 1).numpy()
    save_out(a.out, g * sd + mu, None if y is None else y[idx])
    return len(g)


if __name__ == "__main__":
    p = std_parser(__doc__)
    p.add_argument("--augs", default="jitter scale shift magwarp")
    run_worker(main, p.parse_args())
