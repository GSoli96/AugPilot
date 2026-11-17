"""Calcola le statistiche dei dataset (tabular, ts, graph) e scrive catalog_info/datasets_<modalita>.json.
Rieseguibile: envs/graph/Scripts/python.exe catalog_info/build_dataset_info.py [tabular ts graph]
(serve torch per i grafi; pandas e numpy per il resto). Legge solo i file in datasets/, datasets_ts/, datasets_graph/.
"""
import json, sys, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

# Fonti nel repo (docstring di modalities/ts/datasets.py e modalities/graph/datasets.py)
TS_DOC = {
    "sine_small": "synthetic sinusoids, 2 classes by mean frequency",
    "stock": "T=24 windows from Algorithms/TimeGAN/data/stock_data.csv, at most 1000 windows",
    "energy": "T=24 windows from Algorithms/TimeGAN/data/energy_data.csv, at most 1000 windows",
    "stock_full": "like stock but with all windows (long)",
    "energy_full": "like energy but with all windows (long)",
    "tv_sine": "from Algorithms/timeVAE/data (sine), per-channel MinMax, at most 1000 windows",
    "tv_stock": "from Algorithms/timeVAE/data (stock), per-channel MinMax, at most 1000 windows",
    "tv_energy": "from Algorithms/timeVAE/data (energy), per-channel MinMax, at most 1000 windows",
    "tv_air": "from Algorithms/timeVAE/data (air), per-channel MinMax, at most 1000 windows",
    "cbf": "generated Cylinder-Bell-Funnel, 3 classes",
    "sine_cls": "generated multichannel sinusoids, 3 classes by frequency",
}
GRAPH_DOC = {
    "cora": "Planetoid", "citeseer": "Planetoid", "pubmed": "Planetoid",
    "amazon-photo": "Amazon, split with 20/30 nodes per class in train/val",
    "amazon-computers": "Amazon, split with 20/30 nodes per class in train/val",
    "mutag": "TUDataset", "proteins": "TUDataset", "enzymes": "TUDataset", "imdb-binary": "TUDataset (x = 1)",
    "imdb-multi": "TUDataset (x = 1)", "nci1": "TUDataset", "ptc_mr": "TUDataset", "dd": "TUDataset",
    "collab": "TUDataset (x = 1)",
}


def dist_stats(counts):
    counts = {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}
    vals = list(counts.values())
    return {"n_classes": len(counts), "class_distribution": counts,
            "imbalance_ratio": round(max(vals) / min(vals), 2) if vals and min(vals) > 0 else None}


def imb_label(ratio):
    if ratio is None:
        return "n/a"
    return "balanced" if ratio < 1.5 else ("imbalanced" if ratio < 9 else "highly imbalanced")


def tabular():
    import pandas as pd
    res = {}
    for f in sorted((ROOT / "datasets").glob("*.csv")):
        df = pd.read_csv(f)
        # algorithms_runner.py usa l'ultima colonna come target; evaluate_metrics.py usa 'class' se c'e', altrimenti l'ultima.
        cls = df.columns[-1]
        feats = df.drop(columns=[cls])
        num = [c for c in feats.columns if pd.api.types.is_numeric_dtype(feats[c])]
        cat = [c for c in feats.columns if c not in num]
        lowcard = [c for c in num if feats[c].nunique() <= 10]
        s = {"n_rows": len(df), "n_cols": df.shape[1], "class_column": cls, "class_column_is_named_class": cls == "class",
             "n_numeric_features": len(num), "n_categorical_features": len(cat),
             "n_numeric_low_cardinality_le10": len(lowcard), "n_missing_values": int(df.isna().sum().sum())}
        s.update(dist_stats(df[cls].value_counts().to_dict()))
        r = s["imbalance_ratio"]
        n = s["n_classes"]
        u = []
        if n == 1:
            u.append("Single class (minority-class dataset, _min suffix): suited to generating synthetic data for that class; oversampling across classes is not applicable.")
        elif n == 2 and r and r >= 1.5:
            u.append(f"Binary {imb_label(r)} (ratio {r}): suited to testing minority-class oversampling.")
        elif n == 2:
            u.append("Binary, roughly balanced: useful as a control with no imbalance.")
        else:
            u.append(f"{n} classes, {imb_label(r)} (ratio {r}): suited to testing per-class generation.")
        if cat:
            u.append(f"Has {len(cat)} categorical columns (tests handling of mixed types).")
        if len(df) < 500:
            u.append("Small: quick tests.")
        elif len(df) >= 10000:
            u.append("Large: scalability tests.")
        res[f.stem] = {"stats": s, "uses": " ".join(u),
                       "source": f"script (datasets/{f.name}); class column = last column as in algorithms_runner.py"}
    return res


def ts():
    res = {}
    for f in sorted((ROOT / "datasets_ts").glob("*.npz")):
        z = np.load(f)
        x = z["data"]
        has_y = "labels" in z.files
        s = {"n_series": int(x.shape[0]), "length_T": int(x.shape[1]), "dims_D": int(x.shape[2]), "has_labels": has_y,
             "value_min": round(float(x.min()), 4), "value_max": round(float(x.max()), 4)}
        if has_y:
            vals, c = np.unique(z["labels"], return_counts=True)
            s.update(dist_stats(dict(zip(vals.tolist(), c.tolist()))))
        u = [f"{s['dims_D']} channel{'' if s['dims_D'] == 1 else 's'}, T={s['length_T']}: " + ("univariate." if s["dims_D"] == 1 else "multivariate.")]
        if has_y:
            u.append(f"With labels ({s['n_classes']} classes, {imb_label(s['imbalance_ratio'])}): suited to evaluating class-conditional generation.")
        else:
            u.append("No labels: unconditional generation only.")
        d = TS_DOC.get(f.stem)
        res[f.stem] = {"stats": s, "uses": " ".join(u) + (f" Origin: {d}." if d else " Origin: not documented."),
                       "source": f"script (datasets_ts/{f.name})" + ("; modalities/ts/datasets.py" if d else "")}
    return res


def graph():
    import torch
    res = {}
    for f in sorted((ROOT / "datasets_graph").glob("*.pt")):
        d = torch.load(f, weights_only=False)
        name = f.stem
        base = name[:-4] if name.endswith("_imb") else name
        if d["kind"] == "graph":
            gs = d["graphs"]
            ys = [int(g["y"]) for g in gs]
            nn = np.array([g["x"].shape[0] for g in gs])
            ne = np.array([g["edge_index"].shape[1] for g in gs])
            s = {"level": "graph", "n_graphs": len(gs), "n_features": int(gs[0]["x"].shape[1]),
                 "avg_nodes": round(float(nn.mean()), 1), "avg_edges": round(float(ne.mean()), 1), "max_nodes": int(nn.max())}
            s.update(dist_stats(dict(collections.Counter(ys))))
            r = s["imbalance_ratio"]
        else:
            y = d["y"].numpy()
            tm = d["train_mask"].numpy()
            s = {"level": "node", "n_nodes": int(y.shape[0]), "n_edges": int(d["edge_index"].shape[1]),
                 "n_features": int(d["x"].shape[1]), "avg_degree": round(d["edge_index"].shape[1] / y.shape[0], 2),
                 "n_train": int(tm.sum()), "n_val": int(d["val_mask"].sum()), "n_test": int(d["test_mask"].sum())}
            s.update(dist_stats(dict(collections.Counter(y.tolist()))))
            s["class_distribution_train"] = {str(c): int(((y == c) & tm).sum()) for c in sorted(set(y.tolist()))}
            tv = [v for v in s["class_distribution_train"].values() if v > 0]
            s["imbalance_ratio_train"] = round(max(tv) / min(tv), 2) if tv else None
            r = s["imbalance_ratio_train"]
        u = [f"{'Graph' if s['level'] == 'graph' else 'Node'} level, {s['n_classes']} classes, {imb_label(r)} (ratio {r}"
             + (", on train" if s["level"] == "node" else "") + ")."]
        if name.endswith("_imb"):
            u.append("Imbalanced variant for trying oversampling (RHO=0.2, from modalities/graph/datasets.py).")
        elif r and r >= 1.5:
            u.append("Natural imbalance: suited to testing oversampling.")
        else:
            u.append("Balanced: control with no imbalance.")
        if s["level"] == "graph" and s["avg_nodes"] > 100:
            u.append("Large graphs: scalability tests.")
        if s["level"] == "node" and s["n_nodes"] > 10000:
            u.append("Large graph.")
        doc = GRAPH_DOC.get(base)
        res[name] = {"stats": s, "uses": " ".join(u) + (f" Origin: {doc}." if doc else " Origin: not documented."),
                     "source": f"script (datasets_graph/{f.name}); modalities/graph/datasets.py"}
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["tabular", "ts", "graph"]
    for m in which:
        r = {"tabular": tabular, "ts": ts, "graph": graph}[m]()
        (OUT / f"datasets_{m}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(m, len(r))
