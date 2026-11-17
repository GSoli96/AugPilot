"""Funzioni comuni ai worker del livello nodo (prefisso nl_). Eseguire con envs/graph."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from modalities.worker_utils import std_parser, run_worker  # noqa: E402,F401


def parser(desc=""):
    p = std_parser(desc)
    p.add_argument("--method", required=True)
    p.add_argument("--p", type=float, default=None, help="intensita' della perturbazione (se applicabile)")
    return p


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(a):
    return torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")


def load_node(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    if d.get("kind") != "node":
        raise ValueError(f"atteso kind='node', trovato {d.get('kind')!r}")
    return d


def save(out, d):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    d = dict(d)
    d["edge_index"] = d["edge_index"].long().contiguous().cpu()
    for k in ("x", "y", "train_mask", "val_mask", "test_mask"):
        d[k] = d[k].cpu()
    n = d["x"].shape[0]
    assert d["y"].shape[0] == n and d["train_mask"].shape[0] == n
    assert d["edge_index"].numel() == 0 or int(d["edge_index"].max()) < n
    torch.save(d, out)


def view_result(d, x=None, edge_index=None):
    """Vista perturbata: stessa dimensione, n_orig invariato."""
    r = dict(d)
    if x is not None:
        r["x"] = x
    if edge_index is not None:
        r["edge_index"] = edge_index
    return r


def topk_edges(edge_index, weight, k):
    """Tiene i k archi di peso maggiore (per sparsificare una diffusione)."""
    k = min(int(k), edge_index.shape[1])
    idx = torch.topk(weight, k).indices
    return edge_index[:, idx]


def append_nodes(d, new_x, new_y, new_edges):
    """Oversampling: nodi sintetici in coda; maschere False; n_orig invariato.
    new_edges: LongTensor[2,M] con indici globali (i nuovi nodi partono da N)."""
    import torch as T
    n = d["x"].shape[0]
    m = new_x.shape[0]
    r = dict(d)
    r["x"] = T.cat([d["x"], new_x.to(d["x"].dtype)], 0)
    r["y"] = T.cat([d["y"], new_y.to(d["y"].dtype)], 0)
    for k in ("train_mask", "val_mask", "test_mask"):
        r[k] = T.cat([d[k], T.zeros(m, dtype=T.bool)], 0)
    if new_edges.numel():
        ne = T.cat([new_edges, new_edges.flip(0)], 1)  # simmetrico
        r["edge_index"] = T.unique(T.cat([d["edge_index"], ne.cpu()], 1), dim=1)
    r["n_orig"] = d["n_orig"]
    return r


def minority_classes(d, k=3):
    """Le k classi con meno nodi nel train (a parita', meno nodi totali)."""
    y, tm = d["y"], d["train_mask"]
    c = int(y.max()) + 1
    tr = torch.bincount(y[tm], minlength=c)
    tot = torch.bincount(y, minlength=c)
    order = sorted(range(c), key=lambda i: (int(tr[i]), int(tot[i])))
    return order[:k]
