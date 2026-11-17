"""S-Mixup (Ling et al., "Graph Mixup with Soft Alignments", ICML 2023; Algorithms/DIG, modulo auggraph/SMixup).

Usa dal repository DIG (licenza GPL-3.0, non modificato) GraphMatching/GMNet, triplet_loss e TripleSet. Il pacchetto
`dig` non viene importato come tale (il suo __init__ richiede dipendenze non necessarie): si creano pacchetti
segnaposto che puntano alle cartelle, cosi i sottomoduli si importano cosi come sono.
Passi: (1) si addestra la rete di matching GMNet con triplet loss (anchor, positivo = stessa classe, negativo =
altra classe) per --epochs epoche sui grafi di train; (2) per ogni grafo nuovo di classe c: grafo 1 = grafo di train
di classe c, grafo 2 = grafo di train a caso; allineamento morbido M = softmax_colonne(cos(h1, h2^T)) tra gli
embedding dei nodi; A = lam*A1 + (1-lam)*M A2 M^T, X = lam*X1 + (1-lam)*M X2, lam = max(B,1-B), B ~ Beta(alpha,alpha)
(codice di smixup.Mixup). Differenza dichiarata: l'adiacenza pesata (soglia 0.1 nel codice originale) e
campionata con Bernoulli per avere archi binari; etichetta = c invece del mixup di perdita (y1,y2,lam).
Numero di nodi = quello del grafo 1.
"""
import sys
import types

import numpy as np
import torch
import torch.nn.functional as Fn

import gl_common as C

DIG = C.ROOT / "Algorithms" / "DIG" / "dig"


def _stub(name, path):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m
    return m


def _import_dig():
    _stub("dig", DIG)
    _stub("dig.auggraph", DIG / "auggraph")
    _stub("dig.auggraph.method", DIG / "auggraph" / "method")
    _stub("dig.auggraph.method.SMixup", DIG / "auggraph" / "method" / "SMixup")
    _stub("dig.auggraph.dataset", DIG / "auggraph" / "dataset")
    import importlib
    gm = importlib.import_module("dig.auggraph.method.SMixup.model.GraphMatching")
    ut = importlib.import_module("dig.auggraph.method.SMixup.utils.utils")
    ds = importlib.import_module("dig.auggraph.dataset.aug_dataset")
    return gm.GraphMatching, ut.triplet_loss, ds.TripleSet


def _data(g):
    from torch_geometric.data import Data
    n = g["x"].shape[0]
    deg = torch.log1p(torch.bincount(g["edge_index"][0], minlength=n).float())[:, None]
    return Data(x=torch.cat([g["x"].float(), deg], 1), edge_index=g["edge_index"], y=torch.tensor([g["y"]]))


def synth(args, d, train_idx, targets, rng):
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
    GraphMatching, triplet_loss, TripleSet = _import_dig()
    dev = C.pick_device(args.device)
    graphs = d["graphs"]
    tr = [graphs[i] for i in train_idx]
    data = [_data(g) for g in tr]
    classes = {g["y"] for g in tr}
    if len(classes) < 2:
        raise ValueError("S-Mixup richiede almeno 2 classi nel train (triplet loss)")
    net = GraphMatching(in_dim=data[0].x.shape[1], num_layers=2, hidden=args.hidden, model_type="gmnet",
                        pool_type="sum", fuse_type="abs_diff").to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TripleSet(data), batch_size=32, shuffle=True)
    last = None
    for _ in range(args.epochs):
        net.train()
        tot = 0.0
        for a, p, n in loader:
            a, p, n = a.to(dev), p.to(dev), n.to(dev)
            opt.zero_grad()
            x1, y_ = net(a, p, pred_head=False)
            x2, z_ = net(a, n, pred_head=False)
            loss = torch.mean(triplet_loss(x1, y_, x2, z_))
            loss.backward()
            opt.step()
            tot += loss.item()
        last = tot / max(len(loader), 1)
    C.EXTRA["final_triplet_loss"] = None if last is None else round(last, 4)
    net.eval()
    by_class = {}
    for k, g in enumerate(tr):
        by_class.setdefault(g["y"], []).append(k)
    out = []
    with torch.no_grad():
        for c in targets:
            i = int(rng.choice(by_class[c]))
            j = int(rng.randint(len(tr)))
            b = rng.beta(args.alpha, args.alpha)
            lam = max(b, 1 - b)
            b1, b2 = Batch.from_data_list([data[i]]).to(dev), Batch.from_data_list([data[j]]).to(dev)
            h1, h2 = net.dis_encoder(b1, b2, node_emd=True)
            e1, e2 = Fn.normalize(h1, dim=1), Fn.normalize(h2, dim=1)
            M = torch.softmax(e1 @ e2.T, dim=0).double()
            n1, n2 = data[i].num_nodes, data[j].num_nodes
            A1 = torch.from_numpy(C.adj_from_graph(tr[i])).double().to(dev)
            A2 = torch.from_numpy(C.adj_from_graph(tr[j])).double().to(dev)
            A = lam * A1 + (1 - lam) * (M @ A2 @ M.T)
            X = lam * tr[i]["x"].float().to(dev) + (1 - lam) * (M.float() @ tr[j]["x"].float().to(dev))
            P = A.clamp(0, 1).cpu().numpy()
            S = (rng.rand(n1, n1) < P).astype(np.int32)
            S = np.triu(S, 1)
            S = S + S.T
            g = C.graph_from_adj(S, X.cpu(), c)
            g["lam"] = float(lam)
            out.append(g)
    return out


if __name__ == "__main__":
    p = C.parser(__doc__)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=32)
    a = p.parse_args()
    C.pick_device(a.device)
    C.run(lambda a: C.execute(a, synth, "smixup", "oversampling"), a)
