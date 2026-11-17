"""Mixup e sovra-campionamento di interi grafi: ifmixup, smote_graph, ros.

Nessun repository esterno per questi tre: sono implementazioni nostre (numpy/torch) di idee note.
 - ifmixup: interpolazione di input di due grafi (ifMixup, Guo e Mao, 2021): si riempiono con zeri
   adiacenza e feature fino alla stessa dimensione, A = lam*A1 + (1-lam)*A2, X = lam*X1 + (1-lam)*X2 (ordine dei
   nodi quello del dataset, nessun allineamento). La matrice A (pesata) viene campionata con Bernoulli per
   ottenere archi binari. Grafo 1 di classe bersaglio c, grafo 2 a caso tra tutti i grafi di train;
   lam = max(B,1-B), B ~ Beta(alpha,alpha); etichetta c.
 - smote_graph: stile SMOTE (Chawla 2002) su interi grafi: due grafi di classe c (il secondo tra i k piu vicini
   del primo per un descrittore: sequenza dei gradi ordinata, n, media delle feature); i nodi sono ordinati per
   grado (allineamento), A e X sono interpolati con u ~ U(0,1) e l'adiacenza e campionata con Bernoulli; il numero
   di nodi e interpolato.
 - ros: random oversampling, copia esatta di un grafo di train della classe bersaglio (baseline).
"""
import numpy as np
import torch

import gl_common as C

METHODS = ["ifmixup", "smote_graph", "ros"]
KDEG = 30


def _sorted(g):
    n = g["x"].shape[0]
    deg = np.bincount(g["edge_index"][0].numpy(), minlength=n)
    o = np.argsort(-deg, kind="stable")
    A = C.adj_from_graph(g)[np.ix_(o, o)]
    return A, g["x"].numpy()[o]


def _pad(A, X, n):
    A2 = np.zeros((n, n), np.float32)
    A2[:A.shape[0], :A.shape[0]] = A
    X2 = np.zeros((n, X.shape[1]), np.float32)
    X2[:X.shape[0]] = X
    return A2, X2


def _bern(P, rng):
    S = (rng.rand(*P.shape) < P).astype(np.int32)
    S = np.triu(S, 1)
    return S + S.T


def _descr(g):
    n = g["x"].shape[0]
    deg = np.sort(np.bincount(g["edge_index"][0].numpy(), minlength=n))[::-1][:KDEG]
    v = np.zeros(KDEG)
    v[:len(deg)] = deg
    return np.concatenate([v, [n], g["x"].mean(0).numpy()])


def synth(args, d, train_idx, targets, rng):
    graphs = d["graphs"]
    by_class = {}
    for i in train_idx:
        by_class.setdefault(graphs[i]["y"], []).append(i)
    out = []
    if args.method == "ros":
        for c in targets:
            g = graphs[int(rng.choice(by_class[c]))]
            out.append({"x": g["x"].clone(), "edge_index": g["edge_index"].clone(), "y": c})
        return out
    if args.method == "ifmixup":
        for c in targets:
            g1 = graphs[int(rng.choice(by_class[c]))]
            g2 = graphs[int(rng.choice(train_idx))]
            b = rng.beta(args.alpha, args.alpha)
            lam = max(b, 1 - b)
            n = max(g1["x"].shape[0], g2["x"].shape[0])
            A1 = C.adj_from_graph(g1)
            A2 = C.adj_from_graph(g2)
            A1, X1 = _pad(A1, g1["x"].numpy(), n)
            A2, X2 = _pad(A2, g2["x"].numpy(), n)
            A = lam * A1 + (1 - lam) * A2
            X = lam * X1 + (1 - lam) * X2
            gg = C.graph_from_adj(_bern(A, rng), X, c)
            gg["lam"] = float(lam)
            out.append(gg)
        return out
    # smote_graph
    desc = {c: np.stack([_descr(graphs[i]) for i in idx]) for c, idx in by_class.items()}
    for c in targets:
        idx = by_class[c]
        a = int(rng.randint(len(idx)))
        if len(idx) == 1:
            b = a   # classe con un solo grafo di train: interpolazione con se stesso (copia perturbata dal Bernoulli)
        else:
            dist = np.linalg.norm(desc[c] - desc[c][a], axis=1)
            dist[a] = np.inf
            nn = np.argsort(dist)[:min(args.k, len(idx) - 1)]
            b = int(rng.choice(nn))
        g1, g2 = graphs[idx[a]], graphs[idx[b]]
        A1, X1 = _sorted(g1)
        A2, X2 = _sorted(g2)
        n = max(len(A1), len(A2))
        A1, X1 = _pad(A1, X1, n)
        A2, X2 = _pad(A2, X2, n)
        u = rng.rand()
        n_new = int(round(len(g1["x"]) + u * (len(g2["x"]) - len(g1["x"]))))
        n_new = max(2, min(n, n_new))
        A = (A1 + u * (A2 - A1))[:n_new, :n_new]
        X = (X1 + u * (X2 - X1))[:n_new]
        gg = C.graph_from_adj(_bern(A, rng), X, c)
        gg["lam"] = float(u)
        out.append(gg)
    return out


if __name__ == "__main__":
    p = C.parser(__doc__, METHODS, "ifmixup")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    C.pick_device(a.device)
    task = "oversampling"
    C.run(lambda a: C.execute(a, synth, a.method, task), a)
