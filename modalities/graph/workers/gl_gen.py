"""Generatori di grafi: sbm (stochastic block model) e graphvae (VAE condizionale sulla classe, versione "lite").

Implementazioni nostre (nessun repository esterno).
 - sbm: per ogni grafo nuovo di classe c si sceglie un grafo di train di classe c come "donatore", se ne stima un
   SBM (k blocchi con clustering spettrale + KMeans sugli autovettori dell'adiacenza, probabilita per coppia di
   blocchi = densita osservata) e si campiona una nuova matrice di adiacenza con gli stessi nodi e blocchi.
   Le feature dei nodi sono quelle del donatore (stessa posizione). --k blocchi (default 3).
 - graphvae: ispirato a GraphVAE (Simonovsky e Komodakis, 2018) ma senza graph matching: i nodi sono messi in un
   ordine canonico (grado decrescente), i grafi sono riempiti a NMAX nodi, un VAE con MLP codifica
   [triangolo superiore dell'adiacenza, feature] + classe e decodifica da z (16) + classe. Campionamento:
   z ~ N(0,I), archi Bernoulli, numero di nodi preso da un grafo di train della classe. Grafi con piu di NMAX nodi
   sono troncati ai NMAX nodi di grado maggiore (conteggio in n_graphs_truncated). --epochs = epoche di training.
   Con poche epoche (1-5) il modello e solo un test di funzionamento.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import gl_common as C

METHODS = ["sbm", "graphvae"]
NMAX = 64


def synth_sbm(args, d, train_idx, targets, rng):
    from sklearn.cluster import KMeans
    graphs = d["graphs"]
    by_class = {}
    for i in train_idx:
        by_class.setdefault(graphs[i]["y"], []).append(i)
    out = []
    for c in targets:
        g = graphs[int(rng.choice(by_class[c]))]
        A = C.adj_from_graph(g)
        n = A.shape[0]
        k = max(1, min(args.k, n - 1))
        if k == 1 or A.sum() == 0:
            lab = np.zeros(n, int)
        else:
            w, v = np.linalg.eigh(A)
            emb = v[:, -k:]
            lab = KMeans(n_clusters=k, n_init=5, random_state=int(rng.randint(1 << 30))).fit_predict(emb)
        P = np.zeros((lab.max() + 1, lab.max() + 1))
        for a in range(P.shape[0]):
            for b in range(P.shape[0]):
                ia, ib = np.where(lab == a)[0], np.where(lab == b)[0]
                sub = A[np.ix_(ia, ib)]
                cnt = len(ia) * len(ib) - (len(ia) if a == b else 0)
                P[a, b] = sub.sum() / cnt if cnt > 0 else 0.0
        Pm = P[lab][:, lab]
        S = (rng.rand(n, n) < Pm).astype(np.int32)
        S = np.triu(S, 1)
        S = S + S.T
        out.append(C.graph_from_adj(S, g["x"], c))
    return out


class CVAE(nn.Module):
    def __init__(self, din, ncls, hid=256, z=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(din + ncls, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.mu, self.lv = nn.Linear(hid, z), nn.Linear(hid, z)
        self.dec = nn.Sequential(nn.Linear(z + ncls, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, din))
        self.z = z

    def forward(self, x, y):
        h = self.enc(torch.cat([x, y], 1))
        mu, lv = self.mu(h), self.lv(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(torch.cat([z, y], 1)), mu, lv


def synth_graphvae(args, d, train_idx, targets, rng):
    dev = C.pick_device(args.device)
    graphs = d["graphs"]
    F = graphs[0]["x"].shape[1]
    classes = sorted({graphs[i]["y"] for i in train_idx})
    cid = {c: k for k, c in enumerate(classes)}
    iu = np.triu_indices(NMAX, 1)
    xs, ys, n_trunc = [], [], 0
    allx = torch.cat([graphs[i]["x"] for i in train_idx])
    onehot = bool(((allx == 0) | (allx == 1)).all() and (allx.sum(1) == 1).all())
    for i in train_idx:
        g = graphs[i]
        n = g["x"].shape[0]
        deg = np.bincount(g["edge_index"][0].numpy(), minlength=n)
        o = np.argsort(-deg, kind="stable")[:NMAX]
        n_trunc += int(n > NMAX)
        A = C.adj_from_graph(g)[np.ix_(o, o)]
        Ap = np.zeros((NMAX, NMAX), np.float32)
        Ap[:len(o), :len(o)] = A
        Xp = np.zeros((NMAX, F), np.float32)
        Xp[:len(o)] = g["x"].numpy()[o]
        xs.append(np.concatenate([Ap[iu], Xp.reshape(-1)]))
        ys.append(cid[g["y"]])
    C.EXTRA["n_graphs_truncated"] = n_trunc
    X = torch.tensor(np.stack(xs)).to(dev)
    Y = Fn.one_hot(torch.tensor(ys), len(classes)).float().to(dev)
    na = len(iu[0])
    model = CVAE(X.shape[1], len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def recon(out, x):
        la = Fn.binary_cross_entropy_with_logits(out[:, :na], x[:, :na], reduction="none").sum(1)
        ox, tx = out[:, na:].reshape(-1, NMAX, F), x[:, na:].reshape(-1, NMAX, F)
        if onehot:
            lx = -(Fn.log_softmax(ox, -1) * tx).sum((1, 2))
        else:
            lx = ((ox - tx) ** 2).sum((1, 2))
        return la + lx

    losses = []
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for s in range(0, len(X), 32):
            b = perm[s:s + 32]
            out, mu, lv = model(X[b], Y[b])
            kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
            loss = (recon(out, X[b]) + kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        losses.append(tot / len(X))
    C.EXTRA["final_loss"] = round(losses[-1], 3) if losses else None
    by_class = {}
    for i in train_idx:
        by_class.setdefault(graphs[i]["y"], []).append(i)
    out = []
    model.eval()
    with torch.no_grad():
        for c in targets:
            z = torch.randn(1, model.z, device=dev)
            yv = Fn.one_hot(torch.tensor([cid[c]]), len(classes)).float().to(dev)
            o = model.dec(torch.cat([z, yv], 1))
            pa = torch.sigmoid(o[0, :na]).cpu().numpy()
            ox = o[0, na:].reshape(NMAX, F).cpu()
            n = min(NMAX, graphs[int(rng.choice(by_class[c]))]["x"].shape[0])
            A = np.zeros((NMAX, NMAX))
            A[iu] = (rng.rand(na) < pa).astype(float)
            A = (A + A.T)[:n, :n]
            if onehot:
                x = Fn.one_hot(ox[:n].argmax(-1), F).float()
            else:
                x = ox[:n]
            out.append(C.graph_from_adj(A, x, c))
    return out


def synth(args, d, train_idx, targets, rng):
    return (synth_sbm if args.method == "sbm" else synth_graphvae)(args, d, train_idx, targets, rng)


if __name__ == "__main__":
    p = C.parser(__doc__, METHODS, "sbm")
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    C.pick_device(a.device)
    C.run(lambda a: C.execute(a, synth, a.method, "generation"), a)
