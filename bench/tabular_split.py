"""Split per seed dei dataset tabulari completi (WP2 di the project plan).

Per ogni (dataset, seed) produce in `bench/work/tabular/<dataset>/seed<k>/`:
  train.csv       70% stratificato del dataset completo (entrambe le classi)
  test.csv        30% stratificato (mai visto dai generatori)
  <dataset>_min.csv   minoritaria del SOLO train: è l'input dei generatori a una classe.
                  Il nome mantiene il suffisso `_min` così i wrapper trovano RFD e DC per nome.
  <dataset>.csv   copia di train.csv con il nome del dataset: input degli algoritmi che usano
                  entrambe le classi (smote, fair-smote, smote-cdnn).
  meta.json       etichette, conteggi, n_to_generate = n_maj_train - n_min_train.

Lo split è deterministico (sklearn, random_state=seed) e idempotente: se i file esistono
e meta.json corrisponde, non li riscrive.

Nota di metodo: RFD e Denial Constraints sono stati scoperti dal collega sull'intera
partizione minoritaria (test compreso); è una fuga di soli vincoli, dichiarata nel paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FULL_DIR = ROOT / "datasets_full"
WORK_DIR = ROOT / "bench" / "work" / "tabular"
TEST_SIZE = 0.30


def class_column(df: pd.DataFrame) -> str:
    return "class" if "class" in df.columns else df.columns[-1]


def make_split(dataset: str, seed: int, full_dir: Path = FULL_DIR, work_dir: Path = WORK_DIR) -> dict:
    out = work_dir / dataset / f"seed{seed}"
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all((out / f).exists() for f in ("train.csv", "test.csv", f"{dataset}_min.csv", f"{dataset}.csv")):
            return meta

    df = pd.read_csv(full_dir / f"{dataset}.csv")
    cls = class_column(df)
    counts = df[cls].value_counts()
    min_label = counts.idxmin()
    maj_label = counts.idxmax()
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=seed, stratify=df[cls])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_min = train[train[cls] == min_label].reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    train_min.to_csv(out / f"{dataset}_min.csv", index=False)
    train.to_csv(out / f"{dataset}.csv", index=False)

    n_min_tr = int((train[cls] == min_label).sum())
    n_maj_tr = int((train[cls] != min_label).sum())
    meta = {
        "dataset": dataset, "seed": seed, "class_column": cls,
        "min_label": min_label.item() if hasattr(min_label, "item") else min_label,
        "maj_label": maj_label.item() if hasattr(maj_label, "item") else maj_label,
        "n_train": len(train), "n_test": len(test),
        "n_min_train": n_min_tr, "n_maj_train": n_maj_tr,
        "n_min_test": int((test[cls] == min_label).sum()),
        "n_to_generate": max(0, n_maj_tr - n_min_tr),
        "dir": (str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)).replace("\\", "/"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def split_paths(dataset: str, seed: int, work_dir: Path = WORK_DIR) -> dict:
    out = work_dir / dataset / f"seed{seed}"
    return {
        "dir": out,
        "train": out / "train.csv",
        "test": out / "test.csv",
        "train_min": out / f"{dataset}_min.csv",
        "train_named": out / f"{dataset}.csv",
        "meta": out / "meta.json",
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    for d in a.datasets:
        for s in a.seeds:
            m = make_split(d, s)
            print(d, s, m["n_min_train"], m["n_maj_train"], m["n_to_generate"])
