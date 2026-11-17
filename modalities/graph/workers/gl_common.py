"""Funzioni comuni dei worker graph-level (kind="graph"). Da eseguire in envs/graph.

Convenzioni (vedi anche modalities/graph/datasets.py):
 - ogni worker genera SOLO dai grafi del train split (datasets.graph_split, seed 0), mai da quelli di test;
 - output = grafi originali + grafi nuovi in coda, "n_orig" invariato (anche per i metodi di perturbazione "view":
   la coda contiene copie perturbate di grafi di train delle classi bersaglio);
 - --n_samples N = numero di grafi nuovi. Se manca: si portano le classi al livello della piu numerosa del train
   (se il train e gia bilanciato: meta' dei grafi di train, ripartiti in parti uguali tra le classi);
 - l'ultima riga di stdout e il JSON del contratto (come worker_utils.run_worker, piu campi extra).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modalities.graph.datasets import graph_split  # noqa: E402
from modalities.worker_utils import std_parser  # noqa: E402

EXTRA: dict = {}   # campi aggiuntivi per il JSON finale (es. numero di tentativi, grafi saltati)


def seed_all(seed: int) -> np.random.RandomState:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.RandomState(seed)


def clean_edge_index(ei: torch.Tensor, n: int) -> torch.Tensor:
    """Non orientato, senza self-loop, senza duplicati, ordinato per riga (richiesto da GraphAug)."""
    ei = ei.long()
    if ei.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    ei = ei[:, ei[0] != ei[1]]
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    key = ei[0] * max(n, 1) + ei[1]
    key = torch.unique(key)
    return torch.stack([key // max(n, 1), key % max(n, 1)]).long().contiguous()


def make_graph(x, ei, y) -> dict:
    x = torch.as_tensor(x, dtype=torch.float32)
    n = x.shape[0]
    return {"x": x.contiguous(), "edge_index": clean_edge_index(torch.as_tensor(ei), n), "y": int(y)}


def adj_from_graph(g: dict, n: int | None = None) -> np.ndarray:
    n = g["x"].shape[0] if n is None else n
    A = np.zeros((n, n), dtype=np.float32)
    ei = g["edge_index"].numpy()
    if ei.size:
        A[ei[0], ei[1]] = 1.0
    return A


def graph_from_adj(A: np.ndarray, x, y: int) -> dict:
    A = np.triu((A > 0).astype(np.int64), 1)
    r, c = np.nonzero(A)
    ei = torch.from_numpy(np.stack([r, c])) if len(r) else torch.zeros(2, 0, dtype=torch.long)
    return make_graph(x, ei, y)


def to_pyg(g: dict):
    from torch_geometric.data import Data
    return Data(x=g["x"].clone(), edge_index=g["edge_index"].clone(), y=torch.tensor([g["y"]]),
                num_nodes=g["x"].shape[0])


def plan_targets(ys_train, n_samples, rng) -> list:
    """Elenco delle classi bersaglio, una per grafo nuovo."""
    ys = np.asarray(ys_train)
    classes = sorted(set(ys.tolist()))
    counts = {c: int((ys == c).sum()) for c in classes}
    if n_samples is None:
        mx = max(counts.values())
        t = []
        for c in classes:
            t += [c] * (mx - counts[c])
        if not t:
            k = max(1, len(ys) // 2)
            t = [classes[i % len(classes)] for i in range(k)]
        return t
    cur = dict(counts)
    t = []
    for _ in range(int(n_samples)):
        c = min(classes, key=lambda k: (cur[k], k))
        t.append(c)
        cur[c] += 1
    return t


def load_graph_dataset(path: str) -> dict:
    d = torch.load(path, weights_only=False)
    if d.get("kind") != "graph":
        raise ValueError(f"{path}: atteso kind='graph', trovato {d.get('kind')!r}")
    return d


def execute(args, synth, method: str, task: str) -> int:
    """synth(args, d, train_idx, targets, rng) -> lista di grafi nuovi (dict x, edge_index, y)."""
    rng = seed_all(args.seed)
    d = load_graph_dataset(args.data)
    graphs = d["graphs"]
    train_idx, _ = graph_split([g["y"] for g in graphs])
    train_idx = [i for i in train_idx if i < d["n_orig"]]
    targets = plan_targets([graphs[i]["y"] for i in train_idx], args.n_samples, rng)
    new = synth(args, d, train_idx, targets, rng) if targets else []
    F = graphs[0]["x"].shape[1]
    for g in new:
        if g["x"].shape[1] != F:
            raise ValueError("feature dimension del grafo sintetico diversa dall'originale")
    out = {"kind": "graph", "graphs": list(graphs) + list(new), "n_orig": d["n_orig"],
           "meta": {"method": method, "task": task, "split": "graph_split(seed=0)", "n_new": len(new),
                    "targets": [int(t) for t in targets]}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    return len(new)


def run(fn, args) -> None:
    t0 = time.time()
    rec = {"status": "SUCCESS", "n_generated": 0, "error_message": ""}
    try:
        rec["n_generated"] = int(fn(args))
        rec.update(EXTRA)
    except Exception as e:  # noqa: BLE001
        rec = {"status": "FAILED", "n_generated": 0,
               "error_message": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"}
    rec["time_s"] = round(time.time() - t0, 3)
    try:
        import psutil
        rec["peak_mem_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        rec["peak_mem_mb"] = 0.0
    if torch.cuda.is_available():
        rec["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    print(json.dumps(rec), flush=True)


def parser(desc: str, methods=None, default_method=None):
    p = std_parser(desc)
    if methods:
        p.add_argument("--method", default=default_method, choices=methods)
    return p


def pick_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda richiesto ma CUDA non disponibile")
    return name
