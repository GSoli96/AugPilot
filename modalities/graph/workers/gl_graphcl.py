"""Perturbazioni di grafi in stile GraphCL (Shen-Lab/GraphCL, NeurIPS 2020): dnodes, pedges, subgraph, mask_nodes.

Wrapper delle funzioni drop_nodes, permute_edges, subgraph, mask_nodes di Algorithms/GraphCL/unsupervised_TU/aug.py
(codice di terzi, non modificato). Per ogni grafo nuovo: si sceglie a caso un grafo di train della classe bersaglio e
lo si perturba (le funzioni di GraphCL usano rapporti fissi: 10 % di nodi/archi/feature, subgraph = 20 % dei nodi).
Nota: le funzioni di GraphCL lasciano invariato il numero di nodi (i nodi eliminati restano isolati) e
permute_edges, nel codice originale, rimuove il 10 % degli archi ma non ne aggiunge (idx_add non e usato).
"""
import importlib.util
from copy import deepcopy

import numpy as np
import torch

import gl_common as C

METHODS = ["dnodes", "pedges", "subgraph", "mask_nodes"]
FN = {"dnodes": "drop_nodes", "pedges": "permute_edges", "subgraph": "subgraph", "mask_nodes": "mask_nodes"}


def _load_aug():
    f = C.ROOT / "Algorithms" / "GraphCL" / "unsupervised_TU" / "aug.py"
    spec = importlib.util.spec_from_file_location("graphcl_aug", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synth(args, d, train_idx, targets, rng):
    aug = _load_aug()
    fn = getattr(aug, FN[args.method])
    by_class = {}
    for i in train_idx:
        by_class.setdefault(d["graphs"][i]["y"], []).append(i)
    out, retries = [], 0
    for c in targets:
        for _ in range(20):
            g = d["graphs"][int(rng.choice(by_class[c]))]
            if g["edge_index"].shape[1] == 0 or g["x"].shape[0] < 2:
                retries += 1
                continue
            data = C.to_pyg(g)
            data.edge_index = g["edge_index"].clone()
            try:
                r = fn(deepcopy(data))
            except Exception:  # noqa: BLE001
                retries += 1
                continue
            out.append(C.make_graph(r.x, r.edge_index, c))
            break
        else:
            raise RuntimeError(f"{args.method}: 20 tentativi falliti per la classe {c}")
    C.EXTRA["n_retries"] = retries
    return out


if __name__ == "__main__":
    p = C.parser(__doc__, METHODS, "dnodes")
    a = p.parse_args()
    C.pick_device(a.device)
    C.run(lambda a: C.execute(a, synth, "graphcl_" + a.method, "view"), a)
