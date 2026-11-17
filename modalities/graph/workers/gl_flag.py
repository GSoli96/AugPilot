"""FLAG (Kong et al., CVPR 2022; Algorithms/FLAG): augmentation avversariale delle feature dei nodi.

FLAG e un metodo di TRAINING (perturbazioni delle feature trovate per salita del gradiente durante
l'addestramento). Qui lo si usa in due passi, per produrre un dataset statico:
 1. un GIN (modalities/graph/metrics.py) e addestrato sui grafi di train per --epochs epoche con la procedura FLAG
    (perturbazione iniziale uniforme in [-step,step], m passi di salita sign-gradiente, gradienti dei parametri
    accumulati; come in FLAG/ogb);
 2. per ogni grafo nuovo (grafo di train a caso della classe bersaglio) si cercano con PGD (10 passi di ampiezza
    eps/4, proiezione in [-eps,eps]) feature x + delta che massimizzano la perdita del modello; struttura ed
    etichetta restano quelle del grafo di partenza.
Il codice del repository FLAG (ogb/deep_gcns) e legato a OGB e non e importato: l'algoritmo e riscritto qui.
Differenza dichiarata rispetto al paper: l'output e un insieme statico di esempi avversariali, non una
perturbazione rigenerata a ogni passo di training.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

import gl_common as C


def _batch_inputs(graphs):
    from torch_geometric.data import Batch, Data
    ds = []
    for g in graphs:
        n = g["x"].shape[0]
        deg = torch.log1p(torch.bincount(g["edge_index"][0], minlength=n).float())[:, None]
        ds.append(Data(x=g["x"].float(), deg=deg, edge_index=g["edge_index"], y=torch.tensor([g["y"]])))
    return Batch.from_data_list(ds)


def synth(args, d, train_idx, targets, rng):
    from modalities.graph.metrics import GIN
    dev = C.pick_device(args.device)
    graphs = d["graphs"]
    tr = [graphs[i] for i in train_idx]
    ncls = max(g["y"] for g in graphs[:d["n_orig"]]) + 1
    F = tr[0]["x"].shape[1]
    model = GIN(F + 1, ncls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    m, step = args.m, args.step
    order = np.arange(len(tr))
    for _ in range(args.epochs):
        rng.shuffle(order)
        model.train()
        for s in range(0, len(order), 32):
            sub = [tr[i] for i in order[s:s + 32]]
            if len(sub) < 2:
                continue
            b = _batch_inputs(sub).to(dev)
            pert = torch.empty_like(b.x).uniform_(-step, step).requires_grad_()
            opt.zero_grad()
            for _k in range(m):
                out = model(torch.cat([b.x + pert, b.deg], 1), b.edge_index, b.batch)
                loss = Fn.cross_entropy(out, b.y) / m
                loss.backward()
                pert.data = pert.data + step * torch.sign(pert.grad.detach())
                pert.grad[:] = 0
            opt.step()
    model.eval()
    by_class = {}
    for g in tr:
        by_class.setdefault(g["y"], []).append(g)
    out = []
    for c in targets:
        g = by_class[c][int(rng.randint(len(by_class[c])))]
        b = _batch_inputs([g]).to(dev)
        delta = torch.empty_like(b.x).uniform_(-args.eps, args.eps)
        for _k in range(10):
            delta.requires_grad_(True)
            # BatchNorm in eval: un solo grafo e ammesso
            loss = Fn.cross_entropy(model(torch.cat([b.x + delta, b.deg], 1), b.edge_index, b.batch), b.y)
            grad, = torch.autograd.grad(loss, delta)
            delta = (delta.detach() + args.eps / 4 * grad.sign()).clamp(-args.eps, args.eps)
        out.append({"x": (b.x + delta).detach().cpu(), "edge_index": g["edge_index"].clone(), "y": c})
    return out


if __name__ == "__main__":
    p = C.parser(__doc__)
    p.add_argument("--m", type=int, default=3, help="passi di salita per batch (FLAG)")
    p.add_argument("--step", type=float, default=1e-2, help="ampiezza del passo di FLAG in training")
    p.add_argument("--eps", type=float, default=0.1, help="raggio L-inf della perturbazione esportata")
    a = p.parse_args()
    C.pick_device(a.device)
    C.run(lambda a: C.execute(a, synth, "flag", "view"), a)
