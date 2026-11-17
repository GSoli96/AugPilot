"""G-Mixup (Han et al., ICML 2022; Algorithms/g-mixup): graphon per classe + interpolazione + campionamento.

Usa graphon_estimator.align_graphs (ordinamento dei nodi per grado) e graphon_estimator.universal_svd (USVT,
Chatterjee 2015) dal repository, senza modificarlo. Il campionamento del graphon e la gestione di dimensioni e
feature sono nel wrapper (il codice originale di utils.py usa CUDA fisso, soft label e dimensione fissa 400).
Per ogni grafo nuovo di classe bersaglio c: partner di classe o scelta a caso (anche c), lam = max(B,1-B) con
B ~ Beta(alpha,alpha); graphon = lam*W_c + (1-lam)*W_o (risoluzione R x R); numero di nodi n = quello di un grafo
di train di classe c; si ridimensiona il graphon a n x n, si campiona la matrice di adiacenza, si eliminano i
nodi isolati. Etichetta = c (classe dominante); il valore lam e in g["lam"]. Feature dei nodi: copiate da un grafo
di train di classe c, nodi ordinati per grado, per posizione relativa nell'ordinamento.
"""
import importlib.util
import sys

import numpy as np
import torch

import gl_common as C

R = 64          # risoluzione dei graphon
THRESH = 0.2    # soglia USVT usata dal codice di G-Mixup (utils.estimate_graphon)
NCAP = 200      # grafi con piu nodi: si usano i NCAP nodi di grado maggiore per la stima


def _load_ge():
    f = C.ROOT / "Algorithms" / "g-mixup" / "src" / "graphon_estimator.py"
    spec = importlib.util.spec_from_file_location("gmixup_graphon_estimator", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def degree_order(g):
    n = g["x"].shape[0]
    deg = np.bincount(g["edge_index"][0].numpy(), minlength=n)
    return np.argsort(-deg, kind="stable")


def transfer_features(donor, n_new, A_new):
    """x per i nodi nuovi: nodi del donatore e dei nuovi ordinati per grado, mappati per rango relativo."""
    od = degree_order(donor)
    deg_new = A_new.sum(1)
    on = np.argsort(-deg_new, kind="stable")
    pos = np.minimum((np.arange(n_new) * len(od) / n_new).astype(int), len(od) - 1)
    x = torch.zeros(n_new, donor["x"].shape[1])
    x[torch.from_numpy(on)] = donor["x"][torch.from_numpy(od[pos])]
    return x


def synth(args, d, train_idx, targets, rng):
    import cv2
    ge = _load_ge()
    graphs = d["graphs"]
    by_class = {}
    for i in train_idx:
        by_class.setdefault(graphs[i]["y"], []).append(i)
    W, n_trunc = {}, 0
    for c, idx in by_class.items():
        mats = []
        for i in idx:
            g = graphs[i]
            A = C.adj_from_graph(g)
            mats.append(A)
        aligned, _, max_num, _ = ge.align_graphs(mats, padding=True)
        aligned = [a[:NCAP, :NCAP] for a in aligned]
        n_trunc += sum(1 for m in mats if m.shape[0] > NCAP)
        est = ge.universal_svd(aligned, threshold=THRESH)
        W[c] = cv2.resize(est.astype(np.float32), (R, R), interpolation=cv2.INTER_LINEAR)
    C.EXTRA["n_graphs_truncated_for_graphon"] = n_trunc
    classes = sorted(by_class)
    out, retries = [], 0
    for c in targets:
        for _ in range(30):
            o = classes[int(rng.randint(len(classes)))]
            b = rng.beta(args.alpha, args.alpha)
            lam = max(b, 1 - b)
            Wm = lam * W[c] + (1 - lam) * W[o]
            donor = graphs[int(rng.choice(by_class[c]))]
            n = donor["x"].shape[0]
            Wn = cv2.resize(Wm, (n, n), interpolation=cv2.INTER_LINEAR)
            Wn = np.clip(Wn, 0, 1)
            S = (rng.rand(n, n) <= Wn).astype(np.int32)
            S = np.triu(S, 1)
            S = S + S.T
            keep = S.sum(1) > 0
            if keep.sum() < 2:
                retries += 1
                continue
            S = S[keep][:, keep]
            nn_ = S.shape[0]
            x = transfer_features(donor, nn_, S)
            g = C.graph_from_adj(S, x, c)
            g["lam"] = float(lam)
            out.append(g)
            break
        else:
            raise RuntimeError(f"gmixup: 30 tentativi falliti per la classe {c} (graphon troppo rado)")
    C.EXTRA["n_retries"] = retries
    return out


if __name__ == "__main__":
    p = C.parser(__doc__)
    p.add_argument("--alpha", type=float, default=1.0)
    a = p.parse_args()
    C.pick_device(a.device)
    C.run(lambda a: C.execute(a, synth, "gmixup", "oversampling"), a)
