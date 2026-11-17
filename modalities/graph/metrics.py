"""Metriche della modalita grafi (node-level e graph-level). Da eseguire in envs/graph.

    evaluate(original_path, generated_path, clf_epochs=None, n_seeds=3, device=None) -> dict piatto di numeri

Entrambi i formati .pt (vedi datasets.py): kind "graph" e kind "node". Gli elementi nuovi sono quelli in coda
(indice >= n_orig); se il file generato ha lo stesso numero di elementi dell'originale (modalita "view": stessa
dimensione) le viste sostituiscono/affiancano gli originali come descritto sotto.

COLONNE (prefissi: struct_ = strutturali, div_ = diversita/novita, base_ = senza augmentation,
aug_ = con augmentation, delta_ = aug_ - base_). Tutte numeriche; NaN = non calcolabile (mai sostituito da 0).

Comuni
  n_syn                      elementi nuovi (grafi o nodi)
  gen_mode                   0 = elementi in coda, 1 = vista di pari dimensione

Graph-level (kind="graph"); confronto: grafi sintetici contro grafi di TRAIN originali (graph_split, seed 0)
  struct_orig_nodes_mean, struct_syn_nodes_mean, struct_orig_edges_mean, struct_syn_edges_mean (archi non orientati)
  struct_orig_density_mean, struct_syn_density_mean, struct_orig_clust_mean, struct_syn_clust_mean
  struct_deg_mmd             MMD^2 tra gli istogrammi (normalizzati, gradi 0..49+) per grafo; kernel gaussiano su
                             distanza L2, sigma = 1 (come GraphRNN, ma L2 invece di EMD); stimatore biased
  struct_deg_w1              Wasserstein-1 tra le distribuzioni dei gradi di tutti i nodi
  struct_clust_w1            Wasserstein-1 tra le distribuzioni del coefficiente di clustering medio per grafo
  struct_spec_mmd            MMD^2 tra gli istogrammi (50 bin in [0,2]) dello spettro del laplaciano normalizzato,
                             solo per grafi con <= 200 nodi; struct_spec_frac_used = quota di grafi usati
  div_syn_uniqueness         grafi sintetici distinti / sintetici (hash Weisfeiler-Lehman, 3 iterazioni, etichetta =
                             argmax delle feature del nodo)
  div_syn_novelty            quota di sintetici il cui hash WL non compare nel train originale
  div_syn_pairwise_dist, div_orig_pairwise_dist   distanza media a coppie tra descrittori (log n, densita,
                             clustering, istogramma dei gradi 0..9+) tra sintetici e tra grafi di train
  imb_ratio_before, imb_ratio_after   rapporto classe piu numerosa / meno numerosa nel train, prima e dopo
  base_macro_f1, base_accuracy, base_auc, aug_*, delta_*   GIN (3 strati, hidden 32, somma) addestrato su train
                             originale (base) o su train + sintetici (aug), valutato sui grafi di test originali.
                             Media su n_seeds. Feature = x del dataset + log(1+grado). AUC: binario sulla
                             classe 1, multiclasse one-vs-rest macro; NaN se manca una classe nel test.

Node-level (kind="node"); confronto: grafo originale contro grafo generato (completo)
  struct_orig_nodes, struct_gen_nodes, struct_orig_edges, struct_gen_edges, struct_orig_density, struct_gen_density
  struct_orig_clust, struct_gen_clust   coefficiente di clustering medio
  struct_deg_w1              Wasserstein-1 tra le distribuzioni dei gradi (originale vs generato)
  struct_spec_w1             Wasserstein-1 tra gli spettri del laplaciano normalizzato (solo se N <= 4000, altrimenti
                             NaN); struct_spec_computed = 1/0
  struct_homophily_orig, struct_homophily_gen   omofilia di arco (quota di archi tra nodi con stessa etichetta)
  struct_homophily_new       omofilia degli archi che toccano almeno un nodo sintetico (NaN se non ce ne sono)
  struct_new_edges           archi non orientati nuovi (nel generato meno originale)
  div_syn_feat_dup_frac      quota di nodi sintetici con feature identiche a un altro nodo (originale o sintetico)
  div_syn_feat_nn_dist, div_orig_feat_nn_dist   distanza L2 media al nodo piu vicino di stessa classe (sintetici
                             verso originali; originali tra loro, campione di 500, escluso se stesso)
  imb_ratio_before, imb_ratio_after   rapporto tra la classe piu e meno numerosa tra i nodi di train
  base_*, aug_*, delta_*     GCN a 2 strati (hidden 64) per clf_epochs epoche, epoca scelta sulla validazione
                             originale, valutato sui nodi di test originali sul grafo ORIGINALE. aug: si addestra sul
                             grafo generato con train = train originale + nodi sintetici; in modalita view si
                             sommano le perdite sul grafo originale e sulla vista.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modalities.graph.datasets import graph_split  # noqa: E402

NAN = float("nan")


# ---------------------------------------------------------------- modelli
def _gin_layer(i, o):
    from torch_geometric.nn import GINConv
    return GINConv(nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o)))


class GIN(nn.Module):
    """GIN piccolo per classificare interi grafi (3 strati, somma, un linear finale)."""

    def __init__(self, din, ncls, hid=32, layers=3):
        super().__init__()
        self.convs = nn.ModuleList([_gin_layer(din if k == 0 else hid, hid) for k in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid) for _ in range(layers)])
        self.out = nn.Linear(hid, ncls)

    def forward(self, x, edge_index, batch):
        from torch_geometric.nn import global_add_pool
        for c, b in zip(self.convs, self.bns):
            x = Fn.relu(b(c(x, edge_index)))
        return self.out(global_add_pool(x, batch))


class GCN(nn.Module):
    def __init__(self, din, ncls, hid=64, p=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.c1, self.c2, self.p = GCNConv(din, hid), GCNConv(hid, ncls), p

    def forward(self, x, edge_index):
        x = Fn.dropout(x, self.p, self.training)
        x = Fn.relu(self.c1(x, edge_index))
        x = Fn.dropout(x, self.p, self.training)
        return self.c2(x, edge_index)


# ---------------------------------------------------------------- utilita strutturali
def _adj_sparse(ei, n):
    import scipy.sparse as sp
    ei = np.asarray(ei)
    if ei.size == 0:
        return sp.csr_matrix((n, n))
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def _clustering(A):
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2.0
    denom = deg * (deg - 1) / 2.0
    c = np.zeros_like(deg)
    ok = denom > 0
    c[ok] = tri[ok] / denom[ok]
    return c


def _spectrum(A):
    n = A.shape[0]
    D = np.asarray(A.sum(1)).ravel()
    inv = np.zeros(n)
    inv[D > 0] = D[D > 0] ** -0.5
    M = A.toarray() * inv[:, None] * inv[None, :]
    L = np.diag((D > 0).astype(float)) - M
    return np.linalg.eigvalsh(L)


def _hist(v, bins, rng_):
    h, _ = np.histogram(v, bins=bins, range=rng_)
    s = h.sum()
    return h / s if s > 0 else h.astype(float)


def _mmd(X, Y, sigma=1.0):
    if len(X) == 0 or len(Y) == 0:
        return NAN
    X, Y = np.asarray(X), np.asarray(Y)

    def k(a, b):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d / (2 * sigma ** 2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def _w1(a, b):
    from scipy.stats import wasserstein_distance
    if len(a) == 0 or len(b) == 0:
        return NAN
    return float(wasserstein_distance(a, b))


def _gstat(g):
    n = int(g["x"].shape[0])
    A = _adj_sparse(g["edge_index"].numpy(), n)
    deg = np.asarray(A.sum(1)).ravel()
    m = A.nnz / 2.0
    clu = _clustering(A)
    return {"n": n, "m": m, "deg": deg, "clust": clu, "A": A,
            "density": (2 * m / (n * (n - 1))) if n > 1 else 0.0}


def _wl_hash(g):
    import networkx as nx
    x = g["x"].numpy()
    G = nx.Graph()
    for i in range(x.shape[0]):
        G.add_node(i, l=str(int(x[i].argmax()) if x[i].any() else -1))
    ei = g["edge_index"].numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="l", iterations=3)


def _desc(st):
    h = np.bincount(np.minimum(st["deg"].astype(int), 9), minlength=10) / max(st["n"], 1)
    return np.concatenate([[math.log(max(st["n"], 1)), st["density"], st["clust"].mean() if st["n"] else 0.0], h])


def _pairwise_mean(D):
    D = np.asarray(D)
    if len(D) < 2:
        return NAN
    d = np.sqrt(((D[:, None, :] - D[None, :, :]) ** 2).sum(-1))
    return float(d[np.triu_indices(len(D), 1)].mean())


# ---------------------------------------------------------------- classificazione
def _scores(y_true, prob):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = np.asarray(y_true)
    pred = prob.argmax(1)
    f1 = float(f1_score(y_true, pred, average="macro"))
    acc = float(accuracy_score(y_true, pred))
    try:
        if prob.shape[1] == 2:
            auc = float(roc_auc_score(y_true, prob[:, 1]))
        else:
            present = sorted(set(y_true.tolist()))
            if len(present) < prob.shape[1]:
                auc = NAN
            else:
                auc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = NAN
    return {"macro_f1": f1, "accuracy": acc, "auc": auc}


def _with_deg(g):
    n = g["x"].shape[0]
    deg = torch.bincount(g["edge_index"][0], minlength=n).float()
    return torch.cat([g["x"].float(), torch.log1p(deg)[:, None]], 1)


def _to_data(graphs):
    from torch_geometric.data import Data
    return [Data(x=_with_deg(g), edge_index=g["edge_index"], y=torch.tensor([g["y"]])) for g in graphs]


def _graph_clf(train, test, ncls, epochs, seed, dev):
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, te = _to_data(train), _to_data(test)
    model = GIN(tr[0].x.shape[1], ncls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    ld = DataLoader(tr, batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in ld:
            b = b.to(dev)
            if b.num_graphs < 2:
                continue   # BatchNorm non accetta un solo grafo
            opt.zero_grad()
            Fn.cross_entropy(model(b.x, b.edge_index, b.batch), b.y).backward()
            opt.step()
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for b in DataLoader(te, batch_size=128):
            b = b.to(dev)
            probs.append(Fn.softmax(model(b.x, b.edge_index, b.batch), 1).cpu())
            ys.append(b.y.cpu())
    return _scores(torch.cat(ys).numpy(), torch.cat(probs).numpy())


def _avg(runs):
    return {k: float(np.nanmean([r[k] for r in runs])) if not all(math.isnan(r[k]) for r in runs) else NAN
            for k in runs[0]}


def _imb(ys):
    c = np.bincount(np.asarray(ys, dtype=int))
    c = c[c > 0]
    return float(c.max() / c.min()) if len(c) else NAN


def _put_utility(res, base, aug):
    for k in base:
        res[f"base_{k}"] = base[k]
        res[f"aug_{k}"] = aug[k]
        res[f"delta_{k}"] = aug[k] - base[k]


# ---------------------------------------------------------------- graph-level
def _eval_graph(orig, gen, clf_epochs, n_seeds, dev):
    n_orig = orig["n_orig"]
    if gen["n_orig"] != n_orig or len(gen["graphs"]) < n_orig:
        raise ValueError("il file generato non e compatibile con l'originale (n_orig diverso)")
    og = orig["graphs"]
    train_idx, test_idx = graph_split([g["y"] for g in og[:n_orig]])
    view_mode = len(gen["graphs"]) == n_orig
    if view_mode:
        syn = [gen["graphs"][i] for i in train_idx]
    else:
        syn = gen["graphs"][n_orig:]
    tr = [og[i] for i in train_idx]
    te = [og[i] for i in test_idx]
    res = {"n_syn": len(syn), "gen_mode": 1 if view_mode else 0}

    st_o = [_gstat(g) for g in tr]
    st_s = [_gstat(g) for g in syn]
    if not st_s:
        raise ValueError("nessun grafo sintetico da valutare")
    for nm, st in (("orig", st_o), ("syn", st_s)):
        res[f"struct_{nm}_nodes_mean"] = float(np.mean([s["n"] for s in st]))
        res[f"struct_{nm}_edges_mean"] = float(np.mean([s["m"] for s in st]))
        res[f"struct_{nm}_density_mean"] = float(np.mean([s["density"] for s in st]))
        res[f"struct_{nm}_clust_mean"] = float(np.mean([s["clust"].mean() if s["n"] else 0 for s in st]))
    DB = 50
    res["struct_deg_mmd"] = _mmd([_hist(s["deg"], DB, (0, DB)) for s in st_o],
                                 [_hist(s["deg"], DB, (0, DB)) for s in st_s])
    res["struct_deg_w1"] = _w1(np.concatenate([s["deg"] for s in st_o]), np.concatenate([s["deg"] for s in st_s]))
    res["struct_clust_w1"] = _w1([s["clust"].mean() for s in st_o], [s["clust"].mean() for s in st_s])
    so = [_hist(_spectrum(s["A"]), 50, (0, 2)) for s in st_o if 0 < s["n"] <= 200]
    ss = [_hist(_spectrum(s["A"]), 50, (0, 2)) for s in st_s if 0 < s["n"] <= 200]
    res["struct_spec_mmd"] = _mmd(so, ss)
    res["struct_spec_frac_used"] = float((len(so) + len(ss)) / (len(st_o) + len(st_s)))

    ho = {_wl_hash(g) for g in tr}
    hs = [_wl_hash(g) for g in syn]
    res["div_syn_uniqueness"] = float(len(set(hs)) / len(hs))
    res["div_syn_novelty"] = float(np.mean([h not in ho for h in hs]))
    res["div_syn_pairwise_dist"] = _pairwise_mean([_desc(s) for s in st_s])
    res["div_orig_pairwise_dist"] = _pairwise_mean([_desc(s) for s in st_o])

    ys_tr = [g["y"] for g in tr]
    res["imb_ratio_before"] = _imb(ys_tr)
    res["imb_ratio_after"] = _imb(ys_tr + [g["y"] for g in syn])

    ncls = max(g["y"] for g in og[:n_orig]) + 1
    E = clf_epochs or 30
    base = _avg([_graph_clf(tr, te, ncls, E, s, dev) for s in range(n_seeds)])
    aug = _avg([_graph_clf(tr + syn, te, ncls, E, s, dev) for s in range(n_seeds)])
    _put_utility(res, base, aug)
    return res


# ---------------------------------------------------------------- node-level
def _homophily(ei, y):
    if ei.shape[1] == 0:
        return NAN
    return float((y[ei[0]] == y[ei[1]]).float().mean())


def _node_clf(views, orig, ncls, epochs, seed, dev):
    """views: lista di (x, ei, y, mask) su cui sommare la perdita; orig: dict per valutare."""
    torch.manual_seed(seed)
    model = GCN(views[0][0].shape[1], ncls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    xo, eo, yo = orig["x"].to(dev), orig["edge_index"].to(dev), orig["y"].to(dev)
    best, best_val = None, -1
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = 0
        for (vx, ve, vy, vm) in views:
            loss = loss + Fn.cross_entropy(model(vx, ve)[vm], vy[vm])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(xo, eo)
            val = float((out.argmax(1)[orig["val_mask"].to(dev)] == yo[orig["val_mask"].to(dev)]).float().mean())
            if val > best_val:
                best_val, best = val, Fn.softmax(out, 1).cpu()
    tm = orig["test_mask"]
    return _scores(orig["y"][tm].numpy(), best[tm].numpy())


def _eval_node(orig, gen, clf_epochs, n_seeds, dev):
    N0 = orig["n_orig"]
    if gen["n_orig"] != N0 or gen["x"].shape[0] < N0:
        raise ValueError("il file generato non e compatibile con l'originale (n_orig diverso)")
    Ng = gen["x"].shape[0]
    view_mode = Ng == N0
    res = {"n_syn": Ng - N0, "gen_mode": 1 if view_mode else 0}
    y_o, y_g = orig["y"], gen["y"]
    A_o, A_g = _adj_sparse(orig["edge_index"].numpy(), N0), _adj_sparse(gen["edge_index"].numpy(), Ng)
    for nm, A, n in (("orig", A_o, N0), ("gen", A_g, Ng)):
        res[f"struct_{nm}_nodes"] = float(n)
        res[f"struct_{nm}_edges"] = A.nnz / 2.0
        res[f"struct_{nm}_density"] = float(A.nnz / (n * (n - 1))) if n > 1 else 0.0
        res[f"struct_{nm}_clust"] = float(_clustering(A).mean())
    res["struct_deg_w1"] = _w1(np.asarray(A_o.sum(1)).ravel(), np.asarray(A_g.sum(1)).ravel())
    if max(N0, Ng) <= 4000:
        res["struct_spec_w1"] = _w1(_spectrum(A_o), _spectrum(A_g))
        res["struct_spec_computed"] = 1.0
    else:
        res["struct_spec_w1"] = NAN
        res["struct_spec_computed"] = 0.0
    res["struct_homophily_orig"] = _homophily(orig["edge_index"], y_o)
    res["struct_homophily_gen"] = _homophily(gen["edge_index"], y_g)
    eg = gen["edge_index"]
    touch = (eg[0] >= N0) | (eg[1] >= N0) if not view_mode else torch.zeros(eg.shape[1], dtype=torch.bool)
    res["struct_homophily_new"] = _homophily(eg[:, touch], y_g) if touch.any() else NAN
    res["struct_new_edges"] = res["struct_gen_edges"] - res["struct_orig_edges"]

    tm = orig["train_mask"]
    ys_tr = y_o[tm].tolist()
    syn_idx = torch.arange(N0, Ng)
    res["imb_ratio_before"] = _imb(ys_tr)
    res["imb_ratio_after"] = _imb(ys_tr + y_g[syn_idx].tolist()) if not view_mode else res["imb_ratio_before"]
    if not view_mode and len(syn_idx):
        xs, xo = gen["x"][syn_idx], orig["x"]
        allx = torch.cat([xo, xs])
        uniq = torch.unique(allx, dim=0).shape[0]
        # un nodo sintetico e duplicato se la sua riga compare piu di una volta in originale + sintetici
        _, inv, cnt = torch.unique(allx, dim=0, return_inverse=True, return_counts=True)
        res["div_syn_feat_dup_frac"] = float((cnt[inv[N0:]] > 1).float().mean())
        dists = []
        for c in y_g[syn_idx].unique().tolist():
            a = xs[y_g[syn_idx] == c]
            b = xo[y_o == c]
            dists.append(torch.cdist(a.to(dev), b.to(dev)).min(1).values.cpu())
        res["div_syn_feat_nn_dist"] = float(torch.cat(dists).mean())
        g_ = torch.Generator().manual_seed(0)
        samp = torch.randperm(N0, generator=g_)[:500]
        od = []
        for i in samp.tolist():
            same = (y_o == y_o[i]).nonzero().view(-1)
            same = same[same != i]
            if len(same):
                od.append(torch.cdist(xo[i:i + 1].to(dev), xo[same].to(dev)).min().cpu())
        res["div_orig_feat_nn_dist"] = float(torch.stack(od).mean()) if od else NAN
    else:
        res["div_syn_feat_dup_frac"] = res["div_syn_feat_nn_dist"] = res["div_orig_feat_nn_dist"] = NAN

    ncls = int(max(y_o.max(), y_g.max())) + 1
    E = clf_epochs or 100
    o = {k: orig[k] for k in ("x", "edge_index", "y", "val_mask", "test_mask")}
    xo_d, eo_d, yo_d, tm_d = orig["x"].to(dev), orig["edge_index"].to(dev), y_o.to(dev), tm.to(dev)
    base_views = [(xo_d, eo_d, yo_d, tm_d)]
    xg, eg_d, yg_d = gen["x"].to(dev), eg.to(dev), y_g.to(dev)
    if view_mode:
        aug_views = base_views + [(xg, eg_d, yg_d, tm_d)]
    else:
        m = torch.zeros(Ng, dtype=torch.bool)
        m[:N0] = tm
        m[N0:] = True
        aug_views = [(xg, eg_d, yg_d, m.to(dev))]
    base = _avg([_node_clf(base_views, o, ncls, E, s, dev) for s in range(n_seeds)])
    aug = _avg([_node_clf(aug_views, o, ncls, E, s, dev) for s in range(n_seeds)])
    _put_utility(res, base, aug)
    return res


def evaluate(original_path: str, generated_path: str, clf_epochs: int | None = None, n_seeds: int = 3,
             device: str | None = None) -> dict:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:   # processo lanciato fuori da envs/graph (es. run_modality.py): si delega all'ambiente
        return _evaluate_in_env(original_path, generated_path, clf_epochs, n_seeds, device)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    orig = torch.load(original_path, weights_only=False)
    gen = torch.load(generated_path, weights_only=False)
    if orig.get("kind") != gen.get("kind"):
        raise ValueError(f"formati diversi: {orig.get('kind')} vs {gen.get('kind')}")
    if orig["kind"] == "graph":
        return _eval_graph(orig, gen, clf_epochs, n_seeds, dev)
    if orig["kind"] == "node":
        return _eval_node(orig, gen, clf_epochs, n_seeds, dev)
    raise ValueError(f"kind sconosciuto: {orig['kind']}")


def _evaluate_in_env(original_path, generated_path, clf_epochs, n_seeds, device) -> dict:
    """Esegue evaluate() con il python di envs/graph e ne legge l'ultima riga JSON (errore esplicito se fallisce)."""
    import json
    import subprocess
    py = ROOT / "envs" / "graph" / "Scripts" / "python.exe"
    if not py.exists():
        py = ROOT / "envs" / "graph" / "bin" / "python"
    if not py.exists():
        raise RuntimeError("torch_geometric non disponibile e envs/graph non trovato")
    cmd = [str(py), str(Path(__file__).resolve()), original_path, generated_path, "--n_seeds", str(n_seeds)]
    if clf_epochs:
        cmd += ["--clf_epochs", str(clf_epochs)]
    if device:
        cmd += ["--device", device]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"metriche in envs/graph fallite: {(r.stderr or r.stdout)[-600:]}")
    return json.loads(lines[-1])


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("generated")
    ap.add_argument("--clf_epochs", type=int, default=None)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.original, a.generated, a.clf_epochs, a.n_seeds, a.device)))
