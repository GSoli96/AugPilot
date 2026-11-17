"""Elenco e analisi dei dataset (solo cartelle consentite; nessun percorso arbitrario)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from . import config

CSV_MAX_ROWS = 200_000
LABEL_NAMES = ("class", "label", "target", "y", "classe")


def list_datasets(modality: str) -> Dict[str, Any]:
    if modality not in config.DATASET_DIRS:
        raise ValueError(f"unknown modality: {modality!r}")
    d = config.ROOT / config.DATASET_DIRS[modality]
    items = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in config.DATASET_EXT[modality]:
                items.append({"name": f.stem, "file": f"{d.name}/{f.name}", "size_kb": round(f.stat().st_size / 1024, 1)})
    out = {"modality": modality, "directory": d.name, "datasets": items}
    ud = config.user_dir(modality)       # dataset caricati dall'utente: sempre in coda, marcati user=True
    if ud is not None and ud.is_dir():
        taken = {x["name"] for x in items}
        for f in sorted(ud.iterdir()):
            if f.is_file() and f.suffix.lower() in config.DATASET_EXT[modality] and f.stem not in taken:
                items.append({"name": f.stem, "file": f"{ud.name}/{f.name}", "size_kb": round(f.stat().st_size / 1024, 1),
                              "user": True})
        out["user_directory"] = ud.name
    if not items:
        out["message"] = f"no datasets in {d.name}/"
    return out


def resolve_dataset(modality: str, name: str) -> Path:
    """Nome (o percorso relativo sotto le cartelle consentite) -> file esistente. I dataset del benchmark hanno la
    precedenza; poi si cerca tra quelli caricati dall'utente (datasets_user/, datasets_user_ts/)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("missing dataset")
    name = name.strip()
    if "/" in name or "\\" in name or name.lower().endswith((".csv", ".npz", ".pt")):
        return config.safe_path(name)
    if not config.NAME_RE.match(name):
        raise ValueError(f"invalid dataset name: {name!r}")
    d = config.DATASET_DIRS[modality]
    for ext in config.DATASET_EXT[modality]:
        for stem in (name, name + "_min"):
            f = config.ROOT / d / (stem + ext)
            if f.is_file():
                return config.safe_path(f"{d}/{f.name}")
    ud = config.user_dir(modality)
    if ud is not None:
        for ext in config.DATASET_EXT[modality]:
            f = ud / (name + ext)
            if f.is_file():      # nome validato da NAME_RE: nessun separatore, quindi il file sta davvero in ud
                return f
    raise FileNotFoundError(f"dataset {name!r} not found in {d}/ or among the uploaded ones (see list_datasets)")


def is_user_dataset(modality: str, path: Path) -> bool:
    ud = config.user_dir(modality)
    return ud is not None and Path(path).resolve().parent == ud.resolve()


def _counts(values) -> Dict[str, Any]:
    import numpy as np
    u, c = np.unique(np.asarray(values), return_counts=True)
    cnt = {str(a): int(b) for a, b in zip(u.tolist(), c.tolist())}
    out = {"class_counts": cnt, "n_classes": len(cnt)}
    if len(cnt) == 1:
        out["note"] = "only one class in the file (in *_min datasets it is the minority class alone): imbalance cannot be computed from this file"
    if len(cnt) >= 2:
        mx, mn = max(cnt.values()), min(cnt.values())
        out["imbalance_ratio"] = round(mx / mn, 4) if mn else None
        out["minority_class"] = min(cnt, key=cnt.get)
    return out


def _analyze_csv(path: Path, label_col: Optional[str]) -> Dict[str, Any]:
    import pandas as pd
    total = sum(1 for _ in open(path, "rb")) - 1
    df = pd.read_csv(path, nrows=CSV_MAX_ROWS)
    r: Dict[str, Any] = {"rows": int(total), "columns": int(df.shape[1]), "column_names": list(map(str, df.columns))[:60]}
    if total > len(df):
        r["sampled"] = f"statistics computed on the first {len(df)} rows"
    miss = df.isna().sum()
    r["missing_values_total"] = int(miss.sum())
    r["columns_with_missing"] = {str(k): int(v) for k, v in miss[miss > 0].items()}
    r["numeric_columns"] = int(df.select_dtypes("number").shape[1])
    r["categorical_columns"] = int(df.shape[1] - r["numeric_columns"])
    r["duplicate_rows"] = int(df.duplicated().sum())
    lab, how = None, None
    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"column {label_col!r} does not exist")
        lab, how = label_col, "given by the user"
    else:
        for c in df.columns:
            if str(c).lower() in LABEL_NAMES:
                lab, how = c, "column name"
                break
        if lab is None and df.iloc[:, -1].nunique() <= 20:
            lab, how = df.columns[-1], "ASSUMPTION: last column with few distinct values"
    if lab is not None:
        r["label_column"] = str(lab)
        r["label_column_basis"] = how
        r.update(_counts(df[lab].dropna()))
    else:
        r["label_column"] = None
    return r


def _analyze_npz(path: Path) -> Dict[str, Any]:
    import numpy as np
    z = np.load(path, allow_pickle=False)
    r: Dict[str, Any] = {"arrays": {k: list(z[k].shape) for k in z.files}}
    if "data" not in z.files:
        r["warning"] = "missing array 'data' (N, T, D)"
        return r
    x = z["data"]
    if x.ndim == 3:
        r.update({"series": int(x.shape[0]), "length": int(x.shape[1]), "channels": int(x.shape[2])})
    else:
        r["warning"] = f"data has {x.ndim} dimensions, expected 3"
    r["missing_values_total"] = int(np.isnan(x.astype("float64")).sum())
    if "labels" in z.files:
        r.update(_counts(z["labels"]))
    else:
        r["label_column"] = None
    return r


def _analyze_pt(path: Path) -> Dict[str, Any]:
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(d, dict) or "kind" not in d:
        return {"warning": "unrecognized format (expected a dict with 'kind')"}
    if d["kind"] == "node":
        r = {"kind": "node", "nodes": int(d["x"].shape[0]), "edges": int(d["edge_index"].shape[1]),
             "features": int(d["x"].shape[1])}
        r.update(_counts(d["y"].numpy()))
        return r
    gs = d["graphs"]
    r = {"kind": "graph", "graphs": len(gs),
         "mean_nodes": round(sum(int(g["x"].shape[0]) for g in gs) / max(1, len(gs)), 3),
         "mean_edges": round(sum(int(g["edge_index"].shape[1]) for g in gs) / max(1, len(gs)), 3)}
    if gs:
        r["features"] = int(gs[0]["x"].shape[1])
        r.update(_counts([int(g["y"]) for g in gs]))
    return r


def analyze_dataset(modality: str, dataset: str, label_col: Optional[str] = None) -> Dict[str, Any]:
    path = resolve_dataset(modality, dataset)
    fn = {"tabular": lambda: _analyze_csv(path, label_col), "ts": lambda: _analyze_npz(path),
          "graph": lambda: _analyze_pt(path)}[modality]
    r = fn()
    try:
        rel = str(path.relative_to(config.ROOT)).replace("\\", "/")
    except ValueError:
        rel = f"{path.parent.name}/{path.name}"
    r.update({"modality": modality, "file": rel, "user": is_user_dataset(modality, path),
              "size_kb": round(path.stat().st_size / 1024, 1)})
    return r
