"""Viste perturbate con gli augmentor di PyGCL (Algorithms/PyGCL, Apache-2.0). Task: view, kind=node."""
import sys
from nl_common import *  # noqa: F401,F403
from nl_common import ROOT, parser, seed_all, load_node, save, view_result, topk_edges, run_worker

sys.path.insert(0, str(ROOT / "Algorithms" / "PyGCL"))


def fn(a):
    import torch
    from torch_geometric.data import Data
    import GCL.augmentors as A
    seed_all(a.seed)
    d = load_node(a.data)
    g = Data(x=d["x"], edge_index=d["edge_index"])
    E = d["edge_index"].shape[1]
    m = a.method
    p = a.p
    if m == "edge_removing":
        aug = A.EdgeRemoving(pe=p if p is not None else 0.2)
    elif m == "edge_adding":
        aug = A.EdgeAdding(pe=p if p is not None else 0.2)
    elif m == "feature_masking":
        aug = A.FeatureMasking(pf=p if p is not None else 0.2)
    elif m == "feature_dropout":
        aug = A.FeatureDropout(pf=p if p is not None else 0.2)
    elif m == "node_dropping":
        aug = A.NodeDropping(pn=p if p is not None else 0.1)
    elif m == "node_shuffling":
        aug = A.NodeShuffling()
    elif m == "ppr_diffusion":
        aug = A.PPRDiffusion(alpha=0.2, eps=1e-4, use_cache=False)
    elif m == "markov_diffusion":
        aug = A.MarkovDiffusion(alpha=0.05, order=8, sp_eps=1e-4, use_cache=False)
    # AdaptiveEdgeRemoving/AdaptiveFeatureMasking di PyGCL: lru_cache su Data non hashable (bug del repo):
    # non collegati qui; la rimozione guidata dalla centralita' e' in nl_pyg.py (guided_edge_removal).
    else:
        raise ValueError(f"metodo sconosciuto: {m}")
    g2 = aug(g)
    ei = g2.edge_index
    if m in ("ppr_diffusion", "markov_diffusion"):
        # la diffusione produce una matrice densa pesata: la sparsifico tenendo gli E archi di peso maggiore
        w = g2.edge_attr if g2.edge_attr is not None else torch.ones(ei.shape[1])
        keep = ei[0] != ei[1]
        ei, w = ei[:, keep], w[keep]
        ei = topk_edges(ei, w, E)
    out = view_result(d, x=g2.x, edge_index=ei)
    save(a.out, out)
    return 0


if __name__ == "__main__":
    run_worker(fn, parser("PyGCL augmentors (livello nodo)").parse_args())
