"""Viste con PyG e codice proprio: diffusione heat kernel (GDC), campionamento (random walk, ego-net),
rimozione/aggiunta di archi guidata. Task: view, kind=node."""
from nl_common import *  # noqa: F401,F403
from nl_common import parser, seed_all, load_node, save, view_result, topk_edges, run_worker, device_of


def fn(a):
    import torch
    import torch.nn.functional as F
    from torch_geometric.utils import (degree, k_hop_subgraph, subgraph, to_undirected, coalesce,
                                       remove_self_loops)
    seed_all(a.seed)
    d = load_node(a.data)
    x, ei = d["x"], d["edge_index"]
    N, E = x.shape[0], ei.shape[1]
    m, p = a.method, a.p
    g = torch.Generator().manual_seed(a.seed)

    if m == "gdc_heat":
        # Graph Diffusion Convolution (Gasteiger 2019), kernel heat, sparsificazione top-E
        from torch_geometric.transforms import GDC
        from torch_geometric.data import Data
        t = GDC(self_loop_weight=1, normalization_in="sym", normalization_out="col",
                diffusion_kwargs=dict(method="heat", t=p if p is not None else 3.0),
                sparsification_kwargs=dict(method="topk", k=max(2, round(E / N)), dim=0), exact=True)
        g2 = t(Data(x=x, edge_index=ei, num_nodes=N))
        e2, w = g2.edge_index, g2.edge_attr
        keep = e2[0] != e2[1]
        out = view_result(d, edge_index=to_undirected(e2[:, keep]))

    elif m == "rw_sampling":
        # campionamento per random walk: tiene solo gli archi tra i nodi visitati (i nodi restano)
        nseed, L = int(0.1 * N), 10
        row, col = ei
        order = torch.argsort(row)
        row, col = row[order], col[order]
        ptr = torch.zeros(N + 1, dtype=torch.long); ptr[1:] = torch.cumsum(torch.bincount(row, minlength=N), 0)
        cur = torch.randint(0, N, (nseed,), generator=g)
        vis = [cur]
        for _ in range(L):
            deg = ptr[cur + 1] - ptr[cur]
            nxt = torch.where(deg > 0, col[(ptr[cur] + (torch.rand(len(cur), generator=g) * deg).long().clamp(max=(deg - 1).clamp(min=0))).clamp(max=len(col) - 1)], cur)
            vis.append(nxt); cur = nxt
        sub = torch.unique(torch.cat(vis))
        e2, _ = subgraph(sub, ei, num_nodes=N)
        out = view_result(d, edge_index=e2)

    elif m == "ego_sampling":
        # sottografo indotto dai vicini a 2 salti di nodi seme (k_hop_subgraph di PyG)
        seeds = torch.randperm(N, generator=g)[: max(1, int(0.05 * N))]
        sub, _, _, _ = k_hop_subgraph(seeds, 2, ei, num_nodes=N)
        e2, _ = subgraph(sub, ei, num_nodes=N)
        out = view_result(d, edge_index=e2)

    elif m == "guided_edge_removal":
        # rimozione adattiva (stile GCA): meno probabile togliere archi tra nodi di grado alto
        pe = p if p is not None else 0.2
        deg = degree(ei[0], N)
        cen = torch.log(torch.minimum(deg[ei[0]], deg[ei[1]]) + 1)
        pr = (cen.max() - cen) / (cen.max() - cen.mean() + 1e-9)
        pr = (pr / pr.mean() * pe).clamp(max=0.7)
        keep = torch.rand(E, generator=g) >= pr
        e2 = ei[:, keep]
        out = view_result(d, edge_index=to_undirected(e2))

    elif m == "guided_edge_perturb":
        # perturbazione guidata dalle feature: toglie gli archi con minor similarita' coseno,
        # aggiunge i non-archi (kNN) con maggior similarita'
        pe = p if p is not None else 0.1
        xn = F.normalize(x, dim=1)
        und = ei[:, ei[0] < ei[1]]
        sim_e = (xn[und[0]] * xn[und[1]]).sum(1)
        n_ch = int(pe * und.shape[1])
        keep = torch.argsort(sim_e, descending=True)[: und.shape[1] - n_ch]
        kept = und[:, keep]
        S = xn @ xn.T
        S.fill_diagonal_(-1)
        S[ei[0], ei[1]] = -1
        S = torch.triu(S, 1)
        flat = torch.topk(S.flatten(), n_ch).indices
        add = torch.stack([flat // N, flat % N])
        out = view_result(d, edge_index=to_undirected(torch.cat([kept, add], 1)))
    elif m == "grand_randprop":
        # GRAND (Feng 2020): DropNode (righe di feature azzerate, riscalate) + propagazione casuale
        # x' = media_{k=0..K} A^k DropNode(x), A normalizzata per riga
        pn, K = (p if p is not None else 0.5), 4
        mask = (torch.rand(N, 1, generator=g) >= pn).float()
        h = x * mask / (1 - pn)
        row, col = ei
        dg = degree(row, N).clamp(min=1)
        A = torch.sparse_coo_tensor(ei, 1.0 / dg[row], (N, N))
        acc, cur = h.clone(), h
        for _ in range(K):
            cur = torch.sparse.mm(A, cur)
            acc += cur
        out = view_result(d, x=acc / (K + 1))
    else:
        raise ValueError(f"metodo sconosciuto: {m}")
    save(a.out, out)
    return 0


if __name__ == "__main__":
    run_worker(fn, parser("PyG/proprio: diffusione, campionamento, archi guidati (livello nodo)").parse_args())
