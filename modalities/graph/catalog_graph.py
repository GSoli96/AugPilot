"""Catalogo grafi, livello GRAFO (kind="graph": dataset di molti grafi, classificazione di interi grafi).

Contratto dei worker: vedi modalities/graph/workers/gl_common.py. Lo stato "executable" va tenuto allineato con
l'ultimo test reale (python run_modality.py --modality graph --algorithms <chiave> --datasets mutag --epochs 2
--device cuda --no_eval).
"""
from modalities.base import AlgoSpec, register

W = "modalities/graph/workers/"

# (key, name, strategy, task, worker, method, source, license, paper, needs_gpu, notes)
_ROWS = [
    ("gmixup", "G-Mixup", "mixup: per-class graphon", "oversampling", "gl_gmixup.py", None,
     "Algorithms/g-mixup (github.com/ahxt/g-mixup)", "no license (repository without LICENSE)",
     "Han et al., ICML 2022, arXiv 2202.07179", False,
     "Repository USVT; sampling and features in the wrapper"),
    ("ifmixup", "ifMixup", "mixup: input interpolation", "oversampling", "gl_mixup.py", "ifmixup",
     "our implementation", "n/a (our code)", "Guo and Mao, 2021 (ifMixup)", False,
     "adjacency and features padded to equal size and interpolated, edges sampled"),
    ("smote_graph", "SMOTE on whole graphs", "interpolation between same-class neighbors", "oversampling",
     "gl_mixup.py", "smote_graph", "our implementation", "n/a (our code)",
     "SMOTE idea (Chawla et al., JAIR 2002) applied to degree-aligned graphs", False,
     "not a published method as such"),
    ("ros_graph", "Random oversampling of graphs", "duplication", "oversampling", "gl_mixup.py", "ros",
     "our implementation", "n/a (our code)", "baseline", False, "exact copy of train graphs"),
    ("smixup", "S-Mixup", "mixup: soft alignment with a matching network", "oversampling", "gl_smixup.py", None,
     "Algorithms/DIG (github.com/divelab/DIG, auggraph/SMixup)", "GPL-3.0",
     "Ling et al., ICML 2023, arXiv 2306.06788", True, "GMNet with triplet loss + softmax alignment"),
    ("sbm", "SBM per graph (stochastic block model)", "statistical generative", "generation", "gl_gen.py", "sbm",
     "our implementation", "n/a (our code)", "Holland et al., 1983 (SBM)", False,
     "SBM estimated from the donor graph with spectral clustering"),
    ("graphvae", "Conditional GraphVAE-lite", "neural generative (VAE)", "generation", "gl_gen.py", "graphvae",
     "our implementation", "n/a (our code)", "inspired by Simonovsky and Komodakis, ICANN 2018", True,
     "MLP on degree-sorted adjacency, no graph matching; max 64 nodes"),
    ("flag", "FLAG (adversarial feature examples)", "adversarial feature perturbation", "view",
     "gl_flag.py", None, "Algorithms/FLAG (github.com/devnkong/FLAG), rewritten", "MIT",
     "Kong et al., CVPR 2022, arXiv 2010.09891", True, "exported as a static set of examples"),
    ("graphcl_dnodes", "GraphCL: drop nodes", "perturbation: node deletion", "view", "gl_graphcl.py",
     "dnodes", "Algorithms/GraphCL (github.com/Shen-Lab/GraphCL)", "MIT", "You et al., NeurIPS 2020", False, ""),
    ("graphcl_pedges", "GraphCL: permute edges", "perturbation: edges", "view", "gl_graphcl.py", "pedges",
     "Algorithms/GraphCL", "MIT", "You et al., NeurIPS 2020", False,
     "in the original code it removes 10% of the edges without adding any"),
    ("graphcl_subgraph", "GraphCL: subgraph", "perturbation: subgraph", "view", "gl_graphcl.py", "subgraph",
     "Algorithms/GraphCL", "MIT", "You et al., NeurIPS 2020", False, ""),
    ("graphcl_maskn", "GraphCL: mask nodes", "perturbation: attribute masking", "view",
     "gl_graphcl.py", "mask_nodes", "Algorithms/GraphCL", "MIT", "You et al., NeurIPS 2020", False, ""),
    ("ga_nodesam", "NodeSam", "perturbation: node splitting", "view", "gl_graphaug.py", "nodesam",
     "Algorithms/GraphAug (github.com/snudatalab/GraphAug)", "research only (repository LICENSE)",
     "Yoo et al., WWW 2022, arXiv 2202.10107", False, ""),
    ("ga_submix", "SubMix", "substructure mixup", "view", "gl_graphaug.py", "submix",
     "Algorithms/GraphAug", "research only (repository LICENSE)", "Yoo et al., WWW 2022", False,
     "soft label in y_soft; y = target class"),
    ("ga_graphcrop", "GraphCrop", "perturbation: connected subgraph", "view", "gl_graphaug.py", "graphcrop",
     "Algorithms/GraphAug (baselines/graphcrop.py)", "research only (repository LICENSE)",
     "Wang et al., 2020, arXiv 2009.10564", False, ""),
    ("ga_dropnode", "DropNode (GraphAug)", "perturbation: one node", "view", "gl_graphaug.py", "dropnode",
     "Algorithms/GraphAug", "research only (repository LICENSE)", "Yoo et al., WWW 2022", False, ""),
    ("ga_dropedge", "DropEdge (GraphAug)", "perturbation: one edge", "view", "gl_graphaug.py", "dropedge",
     "Algorithms/GraphAug", "research only (repository LICENSE)", "Yoo et al., WWW 2022", False, ""),
    ("ga_addedge", "AddEdge (GraphAug)", "perturbation: one extra edge", "view", "gl_graphaug.py", "addedge",
     "Algorithms/GraphAug", "research only (repository LICENSE)", "Yoo et al., WWW 2022", False, ""),
    ("ga_motifswap", "MotifSwap", "perturbation: motif swapping", "view", "gl_graphaug.py", "motifswap",
     "Algorithms/GraphAug", "research only (repository LICENSE)", "Yoo et al., WWW 2022", False, ""),
]

# Aggiornare dopo ogni test reale (data 2026-11-09, run_modality.py ... --datasets mutag --epochs 2 --device cuda).
EXECUTABLE: set = {r[0] for r in _ROWS}   # tutte e 19 le voci hanno superato il test del 2026-11-09


def load() -> None:
    for (key, name, strat, task, wk, method, src, lic, paper, gpu, notes) in _ROWS:
        register(AlgoSpec(modality="graph", key=key, name=name, strategy=strat, task=task, env="graph",
                          worker=W + wk, status="executable" if key in EXECUTABLE else "wired",
                          source=src, license=lic, paper=paper, needs_gpu=gpu, notes=notes,
                          extra={"method": method, "level": "graph"} if method else {"level": "graph"}))
