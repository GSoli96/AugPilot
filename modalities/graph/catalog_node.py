"""Catalogo dei metodi a livello nodo (singolo grafo grande: Cora, CiteSeer, ...). Chiavi con prefisso nl_.

Stato "executable" solo dopo un'esecuzione reale (python run_modality.py --modality graph --algorithms <chiave>
--datasets cora --epochs 2 --device cuda --no_eval); gli stati sono aggiornati a mano dopo i test.
"""
from modalities.base import AlgoSpec, register

W = "modalities/graph/workers/"
GCL = "https://github.com/GraphCL/PyGCL"
GCL_PAPER = "arXiv 2109.01116 (PyGCL)"

# (key, name, strategy, task, worker, method, source, license, paper, needs_gpu, notes)
_ROWS = [
    # --- perturbazione strutturale (PyGCL)
    ("nl_dropedge", "EdgeRemoving (DropEdge)", "edge perturbation: random removal", "view", "nl_pygcl", "edge_removing",
     GCL, "Apache-2.0", "DropEdge, ICLR 2020 (augmentor PyGCL EdgeRemoving, p=0.2)", False, ""),
    ("nl_edgeadd", "EdgeAdding", "edge perturbation: random addition", "view", "nl_pygcl", "edge_adding",
     GCL, "Apache-2.0", GCL_PAPER, False, "PyGCL's code does not symmetrize the added edges."),
    # --- perturbazione di nodi
    ("nl_nodedrop", "NodeDropping", "node perturbation: removing the edges of random nodes", "view", "nl_pygcl", "node_dropping",
     GCL, "Apache-2.0", "GraphCL, NeurIPS 2020 (augmentor PyGCL, pn=0.1)", False, "The nodes stay (features and labels) but lose their edges."),
    ("nl_nodeshuffle", "NodeShuffling", "node perturbation: feature permutation", "view", "nl_pygcl", "node_shuffling",
     GCL, "Apache-2.0", "DGI, ICLR 2019 (corruzione; augmentor PyGCL)", False, "The features are permuted: not aligned with the labels."),
    # --- mascheramento di feature
    ("nl_featmask", "FeatureMasking", "feature masking: zeroed columns", "view", "nl_pygcl", "feature_masking",
     GCL, "Apache-2.0", "GRACE, arXiv 2006.04131 (augmentor PyGCL, pf=0.2)", False, ""),
    ("nl_featdrop", "FeatureDropout", "feature masking: element-wise dropout", "view", "nl_pygcl", "feature_dropout",
     GCL, "Apache-2.0", GCL_PAPER, False, ""),
    # --- diffusione
    ("nl_ppr", "PPRDiffusion", "diffusion: Personalized PageRank", "view", "nl_pygcl", "ppr_diffusion",
     GCL, "Apache-2.0", "MVGRL, ICML 2020 / GDC, NeurIPS 2019 (PyGCL)", False,
     "Sparsified diffused matrix: keeps the E highest-weight edges (no self-loops)."),
    ("nl_markov", "MarkovDiffusion", "diffusion: Markov diffusion", "view", "nl_pygcl", "markov_diffusion",
     GCL, "Apache-2.0", "MVGRL-style (PyGCL, order=8)", False, "Sparsified like nl_ppr (order 8 instead of 16 to save time)."),
    ("nl_heat", "GDC heat kernel", "diffusion: heat kernel", "view", "nl_pyg", "gdc_heat",
     "https://github.com/pyg-team/pytorch_geometric", "MIT", "Graph Diffusion Convolution, NeurIPS 2019 (PyG GDC)", False,
     "Top-k per node with k = average degree."),
    # --- campionamento
    ("nl_rwsample", "Random-walk sampling", "sampling: random walk", "view", "nl_pyg", "rw_sampling",
     "own code (idea from PyGCL RWSampling, which requires torch_sparse, not installed)", "Apache-2.0 (idea)", GCL_PAPER, False,
     "10% of nodes as seeds, 10 steps; edges between visited nodes are kept."),
    ("nl_egosample", "Ego-net / k-hop sampling", "sampling: k-hop subgraph", "view", "nl_pyg", "ego_sampling",
     "https://github.com/pyg-team/pytorch_geometric", "MIT", "PyG k_hop_subgraph", False, "5% of seeds, 2 hops."),
    # --- perturbazione guidata
    ("nl_guided_drop", "Guided edge removal", "edge perturbation: centrality-guided removal", "view", "nl_pyg",
     "guided_edge_removal", "own code (GCA style, WWW 2021)", "n/a", "Graph Contrastive Learning with Adaptive Augmentation (WWW 2021)",
     False, "Own implementation: not the original GCA code."),
    ("nl_guided_perturb", "Feature-guided edge perturbation", "edge perturbation: guided by feature similarity", "view",
     "nl_pyg", "guided_edge_perturb", "own code", "n/a", "heuristic (removes low-similarity edges, adds cosine kNN)", False,
     "Heuristic not published as a standalone method."),
    ("nl_grand", "GRAND random propagation", "smoothing: DropNode + random propagation", "view", "nl_pyg", "grand_randprop",
     "https://github.com/THUDM/GRAND (cloned in Algorithms/GRAND; here a rewrite of the augmentation)", "MIT",
     "Graph Random Neural Networks, NeurIPS 2020", False, "Only the feature augmentation, not the consistency training."),
    # --- appresi / avversariali
    ("nl_vgae_edges", "VGAE edge editing (GAug-M-like)", "learned: VGAE edits the edges", "view", "nl_learned", "vgae_edges",
     "https://github.com/zhao-tong/GAug (rewrite in PyG, original code not run)", "MIT",
     "Data Augmentation for GNNs (GAug), AAAI 2021", True, "Rewrite: not the GAug code (torch 1.4 + DGL 0.4.2)."),
    ("nl_flag", "FLAG-like adversarial features", "adversarial: PGD on the features", "view", "nl_learned", "flag_adv",
     "https://github.com/devnkong/FLAG (cloned; here a rewrite of the PGD step only)", "MIT", "FLAG, CVPR 2022", True,
     "Rewrite: FLAG perturbs during training; here x + delta is exported after K=3 steps on a trained GCN."),
    # --- oversampling di classe minoritaria
    ("nl_gs_upsample", "GraphSMOTE setting upsampling", "oversampling: node duplication", "oversampling", "nl_oversample",
     "gs_upsample", "https://github.com/TianxiangZhao/GraphSmote", "no license in the repo", "GraphSMOTE, WSDM 2021", False,
     "Uses the repo's src_upsample. Minority classes = the 3 with the fewest train nodes."),
    ("nl_gs_smote", "GraphSMOTE setting smote", "oversampling: SMOTE on the features", "oversampling", "nl_oversample",
     "gs_smote", "https://github.com/TianxiangZhao/GraphSmote", "no license in the repo", "GraphSMOTE, WSDM 2021", False,
     "Uses the repo's src_smote (edges copied from the parent); portion increased by ~1 node/class because of a repo bug with fraction 0."),
    ("nl_graphsmote", "GraphSMOTE (embedding + decoder)", "oversampling: SMOTE in embedding space + edge decoder", "oversampling",
     "nl_oversample", "graphsmote", "https://github.com/TianxiangZhao/GraphSmote", "no license in the repo", "GraphSMOTE, WSDM 2021", True,
     "Repo modules (Sage_En, Sage_Classifier, Decoder, adj_mse_loss); loop rewritten. Synthetic features = same interpolation "
     "on the raw features. Nodes left without edges by the decoder are linked to their parents (count on stderr). Dense N^2 adjacency: N<=10000."),
    ("nl_graphens", "GraphENS", "oversampling: ego-network synthesis", "oversampling", "nl_oversample", "graphens",
     "https://github.com/JoonHyung-Park/GraphENS", "MIT", "GraphENS, ICLR 2022", True,
     "Repo functions with a torch_scatter shim; helper GCN trained for `epochs` epochs, saliency = |dL/dx|."),
    ("nl_node_mixup", "Node mixup (same-class)", "oversampling: mixup of node pairs", "oversampling", "nl_oversample", "node_mixup",
     "own code (idea from Mixup for Node and Graph Classification, WebConf 2021)", "n/a", "Wang et al., WebConf 2021", False,
     "Own implementation: lambda ~ Beta(2,2) on same-class pairs, edges from the dominant node."),
    ("nl_cvae_nodes", "Conditional VAE nodes", "generative: CVAE on the features + similarity-based edges", "generation", "nl_oversample",
     "cvae_nodes", "own code", "n/a", "CVAE (Sohn et al., NeurIPS 2015)", True,
     "Trained on the (few) train nodes only: limited quality; edges to the 3 most similar nodes of the same class."),
]

# solo descritti (nessun worker)
_DESCRIBED = [
    ("nl_gaug_orig", "GAug-M / GAug-O (original)", "learned: VGAE + classifier", "view",
     "https://github.com/zhao-tong/GAug", "MIT", "AAAI 2021",
     "Stack torch 1.4 + DGL 0.4.2 + pyro + optuna; data in pickle (not loaded). Replaced by nl_vgae_edges."),
    ("nl_graphmix", "GraphMix", "regularization: manifold mixup during training", "view",
     "https://github.com/vikasverma1077/GraphMix", "n/a", "AAAI 2021", "Cloned in Algorithms/GraphMix; augmentation internal to training, not exportable as a graph."),
    ("nl_dropedge_repo", "DropEdge (original repo)", "edge perturbation", "view",
     "https://github.com/DropEdge/DropEdge", "MIT", "ICLR 2020", "Cloned; function covered by nl_dropedge."),
    ("nl_nodeaug", "NodeAug", "feature + structure perturbation with consistency", "view",
     "repo not found", "n/a", "KDD 2020", "No code found."),
]


def load():
    for key, name, strat, task, wk, method, src, lic, paper, gpu, notes in _ROWS:
        register(AlgoSpec(modality="graph", key=key, name=name, strategy=strat, task=task, env="graph",
                          worker=f"{W}{wk}.py", status="wired", source=src, license=lic, paper=paper,
                          needs_gpu=gpu, notes=notes, extra={"method": method, "level": "node"}))
    for key, name, strat, task, src, lic, paper, notes in _DESCRIBED:
        register(AlgoSpec(modality="graph", key=key, name=name, strategy=strat, task=task, env="graph",
                          worker="", status="described", source=src, license=lic, paper=paper,
                          notes=notes, extra={"level": "node"}))
    _apply_tested()


# Chiavi eseguite end-to-end con successo (aggiornato dopo i test reali; vedi consegna)
TESTED = {r[0] for r in _ROWS}  # tutti i 22 con worker: eseguiti su Cora, epoche=2, cuda (2026-11-09)


def _apply_tested():
    from modalities.base import REGISTRY
    for (m, k), s in REGISTRY.items():
        if m == "graph" and k in TESTED:
            s.status = "executable"
