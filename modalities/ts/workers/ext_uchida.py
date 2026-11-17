"""uchidalab/time_series_augmentation (Iwana & Uchida, PLOS ONE 2021; Apache-2.0) - codice ORIGINALE di
Algorithms/time_series_augmentation/utils/augmentation.py, solo i metodi che NON sono gia' in classic.py:
  spawner          SPAWNER (Le Guennec / Kamycki et al.): media di due serie della stessa classe lungo un percorso DTW spezzato
  wdba             DBA pesato (Forestier et al. 2017): medoide + vicini pesati con peso esp(-d) nei percorsi DTW
  rgw              random guided warping (Iwana & Uchida): deforma la serie lungo il percorso DTW verso un prototipo casuale della classe
  dgw              discriminative guided warping: prototipo scelto con distanza media intra-classe vs extra-classe + slice variabile
Ogni metodo produce una variante per serie (output = serie di partenza aumentate, n_samples default N; se
n_samples != N si campionano/ripetono le serie di partenza). Servono le etichette (`labels`); se mancano tutte le
serie sono trattate come una sola classe. --epochs/--device ignorati (nessun training).
Segnalazione: nel repo spawner e random_guided_warp scelgono il prototipo con `x[np.where(l[choices]==l[i])[0]]`,
cioe' con indici relativi all'array `choices` (privo di i) e non a x: per gli indici >= i il prototipo e' il vicino
successivo, che puo' essere di un'altra classe. Il codice non e' modificato; vale come limite dei due metodi.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import ALGO, load_npz, minmax_fit, minmax_inv, save_out, seed_all  # noqa: E402
from modalities.worker_utils import std_parser, run_worker  # noqa: E402

METHODS = ("spawner", "wdba", "rgw", "dgw")


def main(a):
    seed_all(a.seed)
    sys.path.insert(0, os.path.join(ALGO, "time_series_augmentation"))
    import utils.augmentation as aug
    x, y = load_npz(a.data)
    N, T, D = x.shape
    if T < 4:
        raise ValueError("serie troppo corte")
    z, lo, hi = minmax_fit(x)
    z = z.astype("float64")
    lab = np.zeros(N, dtype=int) if y is None else np.unique(y, return_inverse=True)[1]
    n = a.n_samples or N
    # spawner: verbose=-1 e' "vero" nel repo e attiva un ramo con NameError, quindi si usa verbose=0
    fn = {"spawner": lambda xx, ll: aug.spawner(xx, ll, verbose=0),
          "wdba": lambda xx, ll: aug.wdba(xx, ll, verbose=-1),
          "rgw": lambda xx, ll: aug.random_guided_warp(xx, ll, verbose=-1),
          "dgw": lambda xx, ll: aug.discriminative_guided_warp(xx, ll, verbose=-1)}[a.method]
    rng = np.random.default_rng(a.seed)
    idx = np.concatenate([rng.permutation(N) for _ in range(int(np.ceil(n / N)))])[:n]
    outs = []
    for s in range(0, n, N):  # blocchi di al massimo N serie, ciascuna una volta (il metodo usa le altre come prototipi)
        sel = idx[s:s + N]
        outs.append(fn(z[sel], lab[sel]))
    g = np.concatenate(outs, 0)
    save_out(a.out, minmax_inv(g, lo, hi), None if y is None else y[idx])
    return len(g)


if __name__ == "__main__":
    p = std_parser(__doc__)
    p.add_argument("--method", required=True, choices=METHODS)
    run_worker(main, p.parse_args())
