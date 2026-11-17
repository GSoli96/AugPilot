"""Augmentation di grafi di Yoo et al. (WWW 2022, snudatalab/GraphAug): nodesam, submix, graphcrop, dropnode,
dropedge, addedge, motifswap.

Wrapper delle classi di Algorithms/GraphAug/src/augment (codice di terzi, non modificato; licenza: solo ricerca).
Per ogni grafo nuovo: grafo di train a caso della classe bersaglio, trasformato dal metodo.
SubMix restituisce un'etichetta soft: il grafo nuovo prende l'etichetta della classe bersaglio (il grafo di
destinazione) e il vettore soft e salvato in g["y_soft"].
"""
import sys

import numpy as np
import torch

import gl_common as C

METHODS = ["nodesam", "submix", "graphcrop", "dropnode", "dropedge", "addedge", "motifswap"]
CLASS = {"nodesam": "NodeSam", "submix": "SubMix", "graphcrop": "GraphCrop", "dropnode": "DropNode",
         "dropedge": "DropEdge", "addedge": "AddEdge", "motifswap": "MotifSwap"}


def synth(args, d, train_idx, targets, rng):
    src = str(C.ROOT / "Algorithms" / "GraphAug" / "src")
    sys.path.insert(0, src)
    import augment  # noqa: E402  (pacchetto di GraphAug)
    graphs = [d["graphs"][i] for i in train_idx]
    ys = np.array([g["y"] for g in graphs])
    classes = sorted(set(ys.tolist()))
    remap = {c: k for k, c in enumerate(classes)}   # GraphAug vuole etichette 0..C-1
    pyg = []
    for g in graphs:
        p = C.to_pyg(g)
        p.y = torch.tensor([remap[g["y"]]])
        pyg.append(p)
    model_cls = getattr(augment, CLASS[args.method])
    kw = {"cached": False} if args.method in ("submix", "graphcrop") else {}
    model = model_cls(pyg, **kw)
    by_class = {c: np.where(ys == c)[0] for c in classes}
    out, retries = [], 0
    for c in targets:
        for _ in range(30):
            i = int(rng.choice(by_class[c]))
            try:
                r = model(i)
            except Exception:  # noqa: BLE001
                retries += 1
                continue
            g = C.make_graph(r.x, r.edge_index, c)
            if g["x"].shape[0] == 0:
                retries += 1
                continue
            if args.method == "submix":
                g["y_soft"] = r.y.view(-1).tolist()
            out.append(g)
            break
        else:
            raise RuntimeError(f"{args.method}: 30 tentativi falliti per la classe {c}")
    C.EXTRA["n_retries"] = retries
    return out


if __name__ == "__main__":
    p = C.parser(__doc__, METHODS, "nodesam")
    a = p.parse_args()
    C.pick_device(a.device)
    C.run(lambda a: C.execute(a, synth, "graphaug_" + a.method, "view"), a)
