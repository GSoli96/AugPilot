"""Oversampling di classi minoritarie a livello nodo. Output: grafo originale + nodi sintetici in coda
(maschere False per i sintetici, n_orig invariato). Sorgente di training: solo i nodi con train_mask.

Metodi:
  gs_upsample   GraphSMOTE setting 'upsampling' (duplicazione di nodi con le loro adiacenze) - codice del repo
  gs_smote      GraphSMOTE setting 'smote' (SMOTE sulle feature, adiacenza del genitore) - codice del repo
  graphsmote    GraphSMOTE completo: encoder SAGE + SMOTE nello spazio di embedding + decoder degli archi
  node_mixup    mixup di coppie casuali di nodi della stessa classe (feature e archi)
  graphens      GraphENS: ego-network synthesis (neighbor sampling + saliency mixup), funzioni del repo
  cvae_nodes    VAE condizionale sulle feature dei nodi di train; archi verso i vicini piu' simili
"""
import sys
from nl_common import *  # noqa: F401,F403
from nl_common import (ROOT, parser, seed_all, load_node, save, append_nodes, minority_classes, run_worker,
                       device_of)

sys.path.insert(0, str(ROOT / "Algorithms" / "GraphSmote"))
K_MIN = 3


def _dense_adj(d):
    import torch
    n = d["x"].shape[0]
    if n > 10000:
        raise MemoryError(f"adiacenza densa O(N^2) con N={n}: troppo grande per questo metodo")
    a = torch.zeros(n, n)
    a[d["edge_index"][0], d["edge_index"][1]] = 1.0
    a.fill_diagonal_(0)
    return a


def _edges_new_rows(adj_full, n):
    import torch
    sub = adj_full[n:, :]
    r, c = torch.nonzero(sub, as_tuple=True)
    r = r + n
    keep = r != c
    return torch.stack([r[keep], c[keep]])


def _n_per_class(d, mins):
    tr = [int((d["y"][d["train_mask"]] == c).sum()) for c in mins]
    return tr, sum(tr)


def run_gs_repo(a, d, mode):
    """Chiama src_upsample/src_smote del repo GraphSmote; le classi minoritarie sono rimappate sulle ultime."""
    import torch
    import utils as gsu
    seed_all(a.seed)
    mins = minority_classes(d, K_MIN)
    c = int(d["y"].max()) + 1
    perm = torch.empty(c, dtype=torch.long)  # vecchia classe -> nuova
    rest = [i for i in range(c) if i not in mins]
    for i, m in enumerate(mins):
        perm[m] = c - 1 - i
    for j, r in enumerate(rest):
        perm[r] = j
    inv = torch.empty(c, dtype=torch.long)
    inv[perm] = torch.arange(c)
    y2 = perm[d["y"]]
    idx_train = torch.nonzero(d["train_mask"]).flatten()
    adj = _dense_adj(d).to_sparse()
    _, tot = _n_per_class(d, mins)
    portion = (a.n_samples / tot) if a.n_samples else 1.0
    if mode == "smote":
        # src_smote del repo fallisce (pdist su insieme vuoto) se la parte frazionaria di portion e' 0:
        # aggiungo una frazione che equivale a ~1 nodo in piu' per classe (documentato nel catalogo)
        min_n = min(_n_per_class(d, mins)[0])
        portion = int(portion) + max(portion - int(portion), 1.01 / max(min_n, 1))
    fn_ = gsu.src_smote if mode == "smote" else gsu.src_upsample
    adj2, x2, y2n, idx2 = fn_(adj, d["x"], y2, idx_train, portion=portion, im_class_num=K_MIN)
    n = d["x"].shape[0]
    new_x, new_y = x2[n:], inv[y2n[n:]]
    new_e = _edges_new_rows(adj2.to_dense(), n)
    return append_nodes(d, new_x, new_y, new_e), new_x.shape[0]


def run_graphsmote(a, d):
    import torch
    import torch.nn.functional as F
    import models as gsm
    import utils as gsu
    seed_all(a.seed)
    dev = device_of(a)
    x, y = d["x"].to(dev), d["y"].to(dev)
    n = x.shape[0]
    adj = _dense_adj(d).to(dev)
    idx_train = torch.nonzero(d["train_mask"]).flatten().to(dev)
    nclass = int(d["y"].max()) + 1
    mins = minority_classes(d, K_MIN)
    nhid = 64
    enc = gsm.Sage_En(x.shape[1], nhid, nhid, 0.1).to(dev)
    cls = gsm.Sage_Classifier(nhid, nhid, nclass, 0.1).to(dev)
    dec = gsm.Decoder(nhid, 0.1).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(cls.parameters()) + list(dec.parameters()),
                           lr=0.001, weight_decay=5e-4)

    def pairs(embed, portion):
        """SMOTE in embedding: per ogni nodo di train delle classi minoritarie, vicino piu' prossimo della classe."""
        ch_all, nb_all = [], []
        for m in mins:
            ch = idx_train[(y == m)[idx_train]]
            if ch.numel() < 2:
                raise ValueError(f"classe {m}: meno di 2 nodi di train, SMOTE impossibile")
            num = max(2, int(round(ch.numel() * portion)))
            ch = ch[torch.arange(num, device=dev) % ch.numel()]
            e = embed[ch].detach()
            dist = torch.cdist(e, e)
            dist.fill_diagonal_(float("inf"))
            nb = ch[dist.argmin(1)]
            ch_all.append(ch)
            nb_all.append(nb)
        return torch.cat(ch_all), torch.cat(nb_all)

    _, tot = _n_per_class(d, mins)
    portion = (a.n_samples / tot) if a.n_samples else 1.0
    for ep in range(max(1, a.epochs)):
        enc.train(); cls.train(); dec.train()
        opt.zero_grad()
        embed = enc(x, adj)
        ch, nb = pairs(embed, portion)
        lam = torch.rand(ch.numel(), 1, device=dev)
        new_embed = embed[ch] + (embed[nb] - embed[ch]) * lam
        emb_all = torch.cat([embed, new_embed], 0)
        lab_all = torch.cat([y, y[ch]])
        idx_tr_new = torch.cat([idx_train, torch.arange(n, n + ch.numel(), device=dev)])
        adj_up = torch.zeros(n + ch.numel(), n + ch.numel(), device=dev)
        adj_up[:n, :n] = adj
        blk = torch.clamp(adj[ch] + adj[nb], 0, 1)
        adj_up[n:, :n] = blk
        adj_up[:n, n:] = blk.T
        gen = dec(emb_all)
        loss_rec = gsu.adj_mse_loss(gen[:n, :n], adj)
        adj_new = (gen.detach() >= 0.5).float() * adj_up
        adj_new[:n, :n] = adj
        out = cls(emb_all, adj_new)
        loss = F.cross_entropy(out[idx_tr_new], lab_all[idx_tr_new]) + 1e-6 * loss_rec
        loss.backward()
        opt.step()
    # sintesi finale: feature = stessa interpolazione delle feature grezze (il repo sintetizza solo l'embedding)
    enc.eval(); dec.eval()
    with torch.no_grad():
        embed = enc(x, adj)
        ch, nb = pairs(embed, portion)
        lam = torch.rand(ch.numel(), 1, device=dev)
        new_embed = embed[ch] + (embed[nb] - embed[ch]) * lam
        new_x = x[ch] + (x[nb] - x[ch]) * lam
        probs = dec(torch.cat([new_embed, embed], 0))[: ch.numel(), ch.numel():]  # nuovi vs originali
        cand = torch.clamp(adj[ch] + adj[nb], 0, 1)
        link = (probs >= 0.5).float() * cand
        iso = link.sum(1) == 0
        # nodi senza archi dopo decoder+filtro: collegati ai due genitori (conteggio su stderr)
        r_iso = torch.nonzero(iso).flatten()
        link[r_iso, ch[r_iso]] = 1.0
        link[r_iso, nb[r_iso]] = 1.0
        print(f"[graphsmote] nodi sintetici {ch.numel()}, senza archi dal decoder (collegati ai genitori): "
              f"{int(iso.sum())}", file=sys.stderr)
        r, c = torch.nonzero(link, as_tuple=True)
        new_e = torch.stack([r + n, c]).cpu()
    return append_nodes(d, new_x.cpu(), y[ch].cpu(), new_e), ch.numel()


def run_node_mixup(a, d):
    import torch
    seed_all(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    n = d["x"].shape[0]
    mins = minority_classes(d, K_MIN)
    tr_n, _ = _n_per_class(d, mins)
    xs, ys, rows, cols = [], [], [], []
    nbrs = [[] for _ in range(n)]
    for u, v in d["edge_index"].t().tolist():
        nbrs[u].append(v)
    nxt = n
    beta = torch.distributions.Beta(2.0, 2.0)
    for m, k in zip(mins, tr_n):
        pool = torch.nonzero(d["train_mask"] & (d["y"] == m)).flatten()
        cnt = int(round(a.n_samples / len(mins))) if a.n_samples else k
        for _ in range(cnt):
            i, j = pool[torch.randint(len(pool), (2,), generator=g)].tolist()
            lam = float(beta.sample())
            xs.append(lam * d["x"][i] + (1 - lam) * d["x"][j])
            ys.append(m)
            dom, oth = (i, j) if lam >= 0.5 else (j, i)
            for t in nbrs[dom] + [dom]:
                rows.append(nxt); cols.append(t)
            for t in nbrs[oth] + [oth]:
                if float(torch.rand(1, generator=g)) < min(lam, 1 - lam):
                    rows.append(nxt); cols.append(t)
            nxt += 1
    new_x = torch.stack(xs)
    return append_nodes(d, new_x, torch.tensor(ys), torch.tensor([rows, cols])), new_x.shape[0]


def run_cvae(a, d):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    seed_all(a.seed)
    dev = device_of(a)
    nclass = int(d["y"].max()) + 1
    mins = minority_classes(d, K_MIN)
    tm = d["train_mask"]
    X, Y = d["x"][tm].to(dev), d["y"][tm].to(dev)
    f, zd = X.shape[1], 16

    class CVAE(nn.Module):
        def __init__(s):
            super().__init__()
            s.e = nn.Sequential(nn.Linear(f + nclass, 128), nn.ReLU())
            s.mu, s.lv = nn.Linear(128, zd), nn.Linear(128, zd)
            s.d = nn.Sequential(nn.Linear(zd + nclass, 128), nn.ReLU(), nn.Linear(128, f))

        def forward(s, x, c):
            h = s.e(torch.cat([x, c], 1))
            mu, lv = s.mu(h), s.lv(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            return s.d(torch.cat([z, c], 1)), mu, lv

    m = CVAE().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    C = F.one_hot(Y, nclass).float()
    for _ in range(max(1, a.epochs)):  # un passo per epoca: i nodi di train sono pochi
        opt.zero_grad()
        r, mu, lv = m(X, C)
        loss = F.mse_loss(r, X, reduction="sum") / X.shape[0] + (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
        loss.backward()
        opt.step()
    tr_n, _ = _n_per_class(d, mins)
    xs, ys = [], []
    m.eval()
    with torch.no_grad():
        for c, k in zip(mins, tr_n):
            cnt = int(round(a.n_samples / len(mins))) if a.n_samples else k
            cc = F.one_hot(torch.full((cnt,), c, device=dev), nclass).float()
            xs.append(m.d(torch.cat([torch.randn(cnt, zd, device=dev), cc], 1)).cpu())
            ys += [c] * cnt
    new_x = torch.cat(xs)
    new_y = torch.tensor(ys)
    # archi: ogni nodo sintetico si collega ai 3 nodi reali della stessa classe piu' simili (coseno)
    n = d["x"].shape[0]
    S = F.normalize(new_x, dim=1) @ F.normalize(d["x"], dim=1).T
    S[new_y[:, None] != d["y"][None, :]] = -2
    nb = torch.topk(S, 3, dim=1).indices
    rows = (torch.arange(new_x.shape[0])[:, None] + n).expand(-1, 3).flatten()
    return append_nodes(d, new_x, new_y, torch.stack([rows, nb.flatten()])), new_x.shape[0]


def _load_gens():
    """Carica Algorithms/GraphENS/models/gens.py (evita il conflitto di nome 'models' con GraphSmote) e
    fornisce uno shim minimo di torch_scatter.scatter_add (torch_scatter non e' installato)."""
    import importlib.util
    import types
    import torch
    if "torch_scatter" not in sys.modules:
        def scatter_add(src, index, dim=-1, out=None, dim_size=None):
            size = dim_size if dim_size is not None else int(index.max()) + 1
            return torch.zeros(size, dtype=src.dtype, device=src.device).scatter_add_(0, index, src)
        m = types.ModuleType("torch_scatter")
        m.scatter_add = scatter_add
        sys.modules["torch_scatter"] = m
    spec = importlib.util.spec_from_file_location("graphens_gens", ROOT / "Algorithms" / "GraphENS" / "models" / "gens.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_graphens(a, d):
    """GraphENS (ICLR 2022): ego-network synthesis. Funzioni del repo: get_ins_neighbor_dist, neighbor_sampling,
    saliency_mixup, get_dist_kl. Il GCN che fornisce predizioni e saliency e' addestrato qui per `epochs` epoche
    (nel repo e' il modello in training). Saliency = |d loss / d x| (nel repo: hook sul peso del 1o strato)."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    gens = _load_gens()
    seed_all(a.seed)
    dev = device_of(a)
    x, ei, y = d["x"].to(dev), d["edge_index"].to(dev), d["y"].to(dev)
    tm = d["train_mask"].to(dev)
    n = x.shape[0]
    nclass = int(y.max()) + 1

    class Net(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.a, s.b = GCNConv(x.shape[1], 64), GCNConv(64, nclass)

        def forward(s, x_, e):
            return s.b(F.dropout(F.relu(s.a(x_, e)), 0.5, s.training), e)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(max(1, a.epochs)):
        net.train(); opt.zero_grad()
        F.cross_entropy(net(x, ei)[tm], y[tm]).backward(); opt.step()
    net.eval()
    xs = x.clone().requires_grad_(True)
    F.cross_entropy(net(xs, ei)[tm], y[tm]).backward()
    saliency = xs.grad.detach()
    with torch.no_grad():
        prev_out = F.softmax(net(x, ei), dim=1)
    mins = minority_classes(d, K_MIN)
    tr_n, _ = _n_per_class(d, mins)
    src_l = []
    for c, k in zip(mins, tr_n):
        pool = torch.nonzero(tm & (y == c)).flatten()
        cnt = int(round(a.n_samples / len(mins))) if a.n_samples else k
        src_l.append(pool[torch.randint(len(pool), (cnt,), device=dev)])
    src = torch.cat(src_l)
    cls_num = torch.bincount(y[tm], minlength=nclass).float()
    train_idx = torch.nonzero(tm).flatten()
    prob = (torch.log(cls_num) / cls_num)[y[train_idx]]
    dst = train_idx[torch.multinomial(prob, src.numel(), True)]
    src, order = torch.sort(src)
    dst = dst[order]
    nd = gens.get_ins_neighbor_dist(n, ei, tm, dev)
    new_ei, dist_kl = gens.neighbor_sampling(n, ei, src, dst, nd, prev_out)
    lam = torch.distributions.Beta(2, 2).sample((src.numel(),)).unsqueeze(1).to(dev)
    new_x_all = gens.saliency_mixup(x, src, dst, lam, saliency, dist_kl=dist_kl, keep_prob=0.3)
    new_x = new_x_all[n:].detach().cpu()
    new_e = new_ei[:, ei.shape[1]:].cpu()
    new_e = new_e[:, new_e[0] != new_e[1]]
    return append_nodes(d, new_x, y[src].cpu(), new_e), new_x.shape[0]


def fn(a):
    d = load_node(a.data)
    if a.method == "gs_upsample":
        out, k = run_gs_repo(a, d, "upsample")
    elif a.method == "gs_smote":
        out, k = run_gs_repo(a, d, "smote")
    elif a.method == "graphsmote":
        out, k = run_graphsmote(a, d)
    elif a.method == "graphens":
        out, k = run_graphens(a, d)
    elif a.method == "node_mixup":
        out, k = run_node_mixup(a, d)
    elif a.method == "cvae_nodes":
        out, k = run_cvae(a, d)
    else:
        raise ValueError(f"metodo sconosciuto: {a.method}")
    save(a.out, out)
    return k


if __name__ == "__main__":
    run_worker(fn, parser("Oversampling livello nodo").parse_args())
