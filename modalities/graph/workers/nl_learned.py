"""Viste apprese a livello nodo (kind=node, task=view).

  vgae_edges   modifica degli archi con probabilita' predette da un VGAE (idea di GAug-M, AAAI 2021):
               aggiunge i non-archi piu' probabili e toglie gli archi meno probabili. Riscrittura in PyG
               (il repo GAug usa torch 1.4 + DGL 0.4.2; il codice originale non e' eseguito).
  flag_adv     perturbazione avversariale delle feature (idea di FLAG, CVPR 2022): GCN addestrato sul train,
               poi K passi di PGD sulle feature per massimizzare la loss; l'output e' x + delta.
"""
from nl_common import *  # noqa: F401,F403
from nl_common import parser, seed_all, load_node, save, view_result, run_worker, device_of


def vgae_edges(a, d, dev):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, VGAE
    from torch_geometric.utils import to_undirected
    x, ei = d["x"].to(dev), d["edge_index"].to(dev)
    n = x.shape[0]
    if n > 10000:
        raise MemoryError(f"matrice N x N densa con N={n}: troppo grande")

    class Enc(torch.nn.Module):
        def __init__(s, i, h=32, o=16):
            super().__init__()
            s.c1, s.mu, s.ls = GCNConv(i, h), GCNConv(h, o), GCNConv(h, o)

        def forward(s, x, e):
            h = F.relu(s.c1(x, e))
            return s.mu(h, e), s.ls(h, e)

    m = VGAE(Enc(x.shape[1])).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for _ in range(max(1, a.epochs)):
        m.train(); opt.zero_grad()
        z = m.encode(x, ei)
        loss = m.recon_loss(z, ei) + (1.0 / n) * m.kl_loss()
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        z = m.encode(x, ei)
        P = torch.sigmoid(z @ z.T)
    pe = a.p if a.p is not None else 0.1
    und = ei[:, ei[0] < ei[1]]
    k = int(pe * und.shape[1])
    pe_edges = P[und[0], und[1]]
    keep = torch.argsort(pe_edges, descending=True)[: und.shape[1] - k]  # toglie i k meno probabili
    kept = und[:, keep]
    Pm = torch.triu(P, 1)
    Pm[ei[0], ei[1]] = -1
    Pm[ei[1], ei[0]] = -1
    Pm = torch.triu(Pm, 1)
    flat = torch.topk(Pm.flatten(), k).indices
    add = torch.stack([flat // n, flat % n])
    return view_result(d, edge_index=to_undirected(torch.cat([kept, add], 1)).cpu())


def flag_adv(a, d, dev):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    x, ei, y = d["x"].to(dev), d["edge_index"].to(dev), d["y"].to(dev)
    tm = d["train_mask"].to(dev)

    class Net(torch.nn.Module):
        def __init__(s, i, c):
            super().__init__()
            s.a, s.b = GCNConv(i, 64), GCNConv(64, c)

        def forward(s, x, e):
            return s.b(F.dropout(F.relu(s.a(x, e)), 0.5, s.training), e)

    net = Net(x.shape[1], int(y.max()) + 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(max(1, a.epochs)):
        net.train(); opt.zero_grad()
        F.cross_entropy(net(x, ei)[tm], y[tm]).backward(); opt.step()
    net.eval()
    step = a.p if a.p is not None else 1e-2
    K = 3
    delta = (torch.rand_like(x) * 2 - 1) * step
    for _ in range(K):
        delta.requires_grad_(True)
        loss = F.cross_entropy(net(x + delta, ei)[tm], y[tm])  # solo etichette di train
        g, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + step * torch.sign(g)).detach()
    return view_result(d, x=(x + delta).detach().cpu())


def fn(a):
    seed_all(a.seed)
    dev = device_of(a)
    d = load_node(a.data)
    if a.method == "vgae_edges":
        out = vgae_edges(a, d, dev)
    elif a.method == "flag_adv":
        out = flag_adv(a, d, dev)
    else:
        raise ValueError(f"metodo sconosciuto: {a.method}")
    save(a.out, out)
    return 0


if __name__ == "__main__":
    run_worker(fn, parser("Viste apprese (VGAE, FLAG-like)").parse_args())
