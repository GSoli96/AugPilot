"""Dataset di grafi (vanno eseguiti nell'ambiente envs/graph: serve torch_geometric).

Formato del file .pt (torch.save di un dict), sia per input sia per output dei worker:
 - node-level:  {"kind": "node", "x": FloatTensor[N,F], "edge_index": LongTensor[2,E], "y": LongTensor[N],
                 "train_mask","val_mask","test_mask": BoolTensor[N], "n_orig": N_originale}
 - graph-level: {"kind": "graph", "graphs": [{"x","edge_index","y": int}, ...], "n_orig": numero originale}
Output di un algoritmo di oversampling: dataset originale + elementi sintetici in coda ("n_orig" invariato).
Output di un algoritmo "view": stessa dimensione, grafo perturbato ("n_orig" = N).
resolve(nome) -> percorso del .pt in datasets_graph/ (lo crea da PyG al primo uso: serve la rete).

Dataset di base
 - node (Planetoid): cora, citeseer, pubmed.
 - node (Amazon, split creato qui): amazon-photo (7650 nodi, 8 classi), amazon-computers (13752 nodi, 10 classi).
   Il pacchetto non ha maschere: 20 nodi per classe in train, 30 per classe in val, il resto in test (seed 0).
 - graph (TUDataset): mutag (188 grafi, 2 classi), proteins (1113, 2), enzymes (600, 6), imdb-binary (1000, 2),
   imdb-multi (1500, 3), nci1 (4110, 2), ptc_mr (344, 2), dd (1178, 2; grafi grandi, ~284 nodi in media),
   collab (5000, 3; grafi densi). I TU senza feature dei nodi (IMDB, COLLAB) ricevono x = 1 (una colonna di uno).

Varianti sbilanciate, per provare l'oversampling (suffisso "_imb", per ogni dataset di base, es. mutag_imb, cora_imb):
 - graph-level: le classi sono ordinate per numerosita decrescente (parita: indice di classe crescente); la classe
   di rango k (su C classi) conserva la frazione RHO**(k/(C-1)) dei propri grafi, con RHO = 0.2 (la piu numerosa
   il 100 %, la meno numerosa il 20 %). I grafi conservati sono scelti a caso con seed 0, ordine originale mantenuto.
   Esempio MUTAG: classe 1 (125 grafi) intera, classe 0 (63 grafi) ridotta a 13.
 - node-level: stesso schema applicato alle sole maschere di TRAIN: la classe di rango k conserva
   ceil(n_train_k * RHO**(k/(C-1))) nodi di train (scelti a caso, seed 0); gli altri nodi escono da train_mask
   (restano nel grafo, non nei val/test). Il grafo non cambia.

Split fisso dei grafi (graph-level), usato da worker e metriche: graph_split(y) -> (train_idx, test_idx), 70/30
stratificato, seed 0 indipendente dal seed dell'algoritmo. I worker generano solo dai grafi di train, cosi nessun
grafo sintetico deriva da un grafo di test.
"""
from pathlib import Path
import subprocess, sys

DIR = Path(__file__).resolve().parents[2] / "datasets_graph"
NODE = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
AMAZON = {"amazon-photo": "Photo", "amazon-computers": "Computers"}
GRAPH = {"mutag": "MUTAG", "proteins": "PROTEINS", "enzymes": "ENZYMES", "imdb-binary": "IMDB-BINARY",
         "imdb-multi": "IMDB-MULTI", "nci1": "NCI1", "ptc_mr": "PTC_MR", "dd": "DD", "collab": "COLLAB"}
RHO = 0.2
TEST_FRAC = 0.3


def graph_split(y, seed: int = 0, test_frac: float = TEST_FRAC):
    """Split stratificato dei grafi: restituisce (train_idx, test_idx) come liste ordinate di int."""
    import numpy as np
    y = np.asarray(list(y))
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = int(round(len(idx) * test_frac))
        if len(idx) >= 2:
            n_te = min(max(n_te, 1), len(idx) - 1)
        te += idx[:n_te].tolist()
        tr += idx[n_te:].tolist()
    return sorted(int(i) for i in tr), sorted(int(i) for i in te)


def _class_keep_fraction(counts: dict) -> dict:
    order = sorted(counts, key=lambda c: (-counts[c], c))
    C = len(order)
    return {c: (RHO ** (k / (C - 1)) if C > 1 else 1.0) for k, c in enumerate(order)}


def _imbalance(d: dict) -> dict:
    import numpy as np, torch, math
    rng = np.random.RandomState(0)
    if d["kind"] == "graph":
        ys = np.array([g["y"] for g in d["graphs"]])
        counts = {int(c): int((ys == c).sum()) for c in set(ys.tolist())}
        frac = _class_keep_fraction(counts)
        keep = []
        for c in sorted(counts):
            idx = np.where(ys == c)[0]
            k = max(2, int(round(len(idx) * frac[c])))
            keep += rng.choice(idx, size=min(k, len(idx)), replace=False).tolist()
        keep = sorted(keep)
        gs = [d["graphs"][i] for i in keep]
        return {"kind": "graph", "graphs": gs, "n_orig": len(gs)}
    y = d["y"].numpy()
    tm = d["train_mask"].numpy().copy()
    counts = {int(c): int(((y == c) & tm).sum()) for c in set(y.tolist())}
    frac = _class_keep_fraction(counts)
    new = np.zeros_like(tm)
    for c in counts:
        idx = np.where((y == c) & tm)[0]
        k = max(1, math.ceil(len(idx) * frac[c]))
        new[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
    out = dict(d)
    out["train_mask"] = torch.from_numpy(new)
    return out


def build(name: str, out: Path) -> None:
    import torch
    key = name.lower()
    if key.endswith("_imb"):
        base = torch.load(resolve(key[:-4]), weights_only=False)
        torch.save(_imbalance(base), out)
    elif key in NODE:
        from torch_geometric.datasets import Planetoid
        d = Planetoid(str(DIR / "raw"), NODE[key])[0]
        torch.save({"kind": "node", "x": d.x, "edge_index": d.edge_index, "y": d.y, "train_mask": d.train_mask,
                    "val_mask": d.val_mask, "test_mask": d.test_mask, "n_orig": int(d.num_nodes)}, out)
    elif key in AMAZON:
        from torch_geometric.datasets import Amazon
        d = Amazon(str(DIR / "raw"), AMAZON[key])[0]
        g = torch.Generator().manual_seed(0)
        N = d.num_nodes
        tr = torch.zeros(N, dtype=torch.bool)
        va = torch.zeros(N, dtype=torch.bool)
        for c in d.y.unique().tolist():
            idx = (d.y == c).nonzero().view(-1)
            idx = idx[torch.randperm(len(idx), generator=g)]
            tr[idx[:20]] = True
            va[idx[20:50]] = True
        te = ~(tr | va)
        torch.save({"kind": "node", "x": d.x, "edge_index": d.edge_index, "y": d.y, "train_mask": tr,
                    "val_mask": va, "test_mask": te, "n_orig": int(N)}, out)
    elif key in GRAPH:
        from torch_geometric.datasets import TUDataset
        ds = TUDataset(str(DIR / "raw"), GRAPH[key], use_node_attr=True)
        gs = []
        for g in ds:
            x = g.x if g.x is not None else torch.ones(g.num_nodes, 1)
            gs.append({"x": x, "edge_index": g.edge_index, "y": int(g.y)})
        torch.save({"kind": "graph", "graphs": gs, "n_orig": len(gs)}, out)
    else:
        raise FileNotFoundError(f"unknown graph dataset '{name}'")


def resolve(name: str) -> str:
    p = Path(name)
    if p.exists():
        return str(p)
    DIR.mkdir(exist_ok=True)
    f = DIR / f"{name.lower()}.pt"
    if not f.exists():
        root = Path(__file__).resolve().parents[2]
        py = root / "envs/graph/Scripts/python.exe"
        r = subprocess.run([str(py if py.exists() else sys.executable), "-c",
                            f"from modalities.graph.datasets import build; from pathlib import Path; build({name!r}, Path({str(f)!r}))"],
                           cwd=str(root), capture_output=True, text=True)
        if r.returncode != 0 or not f.exists():
            raise RuntimeError(f"failed to create dataset {name}: {r.stderr[-500:]}")
    return str(f)
