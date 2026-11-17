"""Caricamento di un dataset proprio dalla UI (Percorso guidato e Copilota).

Formati: tabulare = CSV con la classe nell'ultima colonna; serie temporali = .npz con `data` (N, T, D) e, opzionale, `labels` (N,).
Salvataggio in datasets_user/ e datasets_user_ts/ (assistant/config.py: USER_DIRS); i grafi non si caricano.
Regole: al massimo 50 MB, nome ripulito, nessuna sovrascrittura (suffisso _2, _3, ... se il nome esiste, anche nei dataset del
benchmark). Il contenuto si valida in memoria prima di scrivere; un file non valido non lascia tracce su disco.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Dict

from assistant import config

EXT = {"tabular": ".csv", "ts": ".npz"}
MAX_COLS = 5000
MAX_UNCOMPRESSED = 600 * 1024 * 1024       # un .npz compresso puo' espandersi molto: limite sul contenuto decompresso
NAME_MAX = 80


class UploadError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def clean_stem(filename: str) -> str:
    """Nome del file -> nome di dataset sicuro (NAME_RE): solo l'ultimo componente, caratteri strani sostituiti con _."""
    base = re.split(r"[\\/]", str(filename or ""))[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    stem = re.sub(r"[^A-Za-z0-9_.\-]+", "_", stem)
    stem = re.sub(r"\.{2,}", ".", stem).strip("._-")[:NAME_MAX]
    if not stem or not config.NAME_RE.match(stem):
        stem = "dataset"
    return stem


def _taken_names(modality: str) -> set:
    """Nomi gia' in uso: dataset del benchmark (anche con _min), dataset noti di ts e file gia' caricati."""
    names = set()
    d = config.ROOT / config.DATASET_DIRS[modality]
    ext = config.DATASET_EXT[modality]
    if d.is_dir():
        for f in d.iterdir():
            if f.suffix.lower() in ext:
                names.add(f.stem.lower())
                names.add(re.sub(r"_min$", "", f.stem.lower()))
    if modality == "ts":
        try:
            from modalities.ts import datasets as tsd
            names.update(n.lower() for n in tsd.list_datasets())
        except Exception:  # noqa: BLE001
            pass
    return names


def free_name(modality: str, stem: str) -> str:
    """`stem`, oppure `stem_2`, `stem_3`, ... se il nome esiste gia'."""
    ud = config.user_dir(modality)
    ext = EXT[modality]
    taken = _taken_names(modality)
    n, cand = 1, stem
    while cand.lower() in taken or (ud is not None and (ud / (cand + ext)).exists()):
        n += 1
        cand = f"{stem}_{n}"
    return cand


def validate_csv(raw: bytes) -> Dict[str, Any]:
    import pandas as pd
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        raise UploadError("the file is not UTF-8 text: save it as a UTF-8 CSV")
    except Exception as e:  # noqa: BLE001
        raise UploadError(f"CSV unreadable ({type(e).__name__}): check the separator (comma) and the header")
    if df.shape[1] < 2:
        raise UploadError("at least two columns are needed: the features and, last, the class (separator: comma)")
    if df.shape[1] > MAX_COLS:
        raise UploadError(f"too many columns ({df.shape[1]}, maximum {MAX_COLS})")
    if len(df) < 2:
        raise UploadError("at least two data rows are needed besides the header")
    if df.iloc[:, -1].isna().all():
        raise UploadError("the last column (the class) is empty")
    lab = df.iloc[:, -1].dropna()
    out = {"rows": int(len(df)), "columns": int(df.shape[1]), "label_column": str(df.columns[-1]),
           "n_classes": int(lab.nunique()), "warnings": []}
    if out["n_classes"] == 1:
        out["warnings"].append("only one class in the last column: fine for files with the minority class only, "
                               "but the imbalance cannot be computed")
    elif out["n_classes"] > 50:
        out["warnings"].append(f"{out['n_classes']} distinct values in the last column: is it really the class?")
    if df.isna().any().any():
        out["warnings"].append(f"{int(df.isna().sum().sum())} missing values: not all algorithms handle them")
    return out


def validate_npz(raw: bytes) -> Dict[str, Any]:
    import numpy as np
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise UploadError("not a valid .npz file (unreadable archive)")
    if sum(i.file_size for i in zf.infolist()) > MAX_UNCOMPRESSED:
        raise UploadError("decompressed content too large")
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        raise UploadError(f".npz unreadable ({type(e).__name__}); the arrays must not contain Python objects")
    if "data" not in z.files:
        raise UploadError(f"the `data` array (N, T, D) is missing; found: {', '.join(z.files) or 'none'}")
    x = z["data"]
    if x.ndim != 3:
        raise UploadError(f"`data` must have 3 dimensions (N series, T time steps, D channels); its shape is {tuple(x.shape)}")
    if not np.issubdtype(x.dtype, np.number):
        raise UploadError("`data` must be numeric")
    n, t, d = (int(v) for v in x.shape)
    if n < 2 or t < 2 or d < 1:
        raise UploadError(f"shape too small: N={n}, T={t}, D={d}")
    out: Dict[str, Any] = {"series": n, "length": t, "channels": d, "warnings": [], "has_labels": False}
    if "labels" in z.files:
        y = z["labels"]
        if y.shape[0] != n or y.ndim != 1:
            raise UploadError(f"`labels` must have shape ({n},), got {tuple(y.shape)}")
        out["has_labels"] = True
        out["n_classes"] = int(len(np.unique(y)))
    nan = int(np.isnan(x.astype("float64")).sum())
    if nan:
        out["warnings"].append(f"{nan} NaN values in `data`: many generators do not accept them")
    extra = [k for k in z.files if k not in ("data", "labels")]
    if extra:
        out["warnings"].append(f"ignored arrays: {', '.join(extra)}")
    return out


def save_upload(modality: str, filename: str, raw: bytes) -> Dict[str, Any]:
    """Valida e salva. Restituisce {name, modality, file, size_kb, user, validation}. Solleva UploadError."""
    if modality == "graph":
        raise UploadError("graphs cannot be uploaded: use the datasets already on disk", 400)
    if modality not in EXT:
        raise UploadError(f"unknown modality: {modality!r}", 400)
    if not raw:
        raise UploadError("empty file")
    if len(raw) > config.UPLOAD_MAX_BYTES:
        raise UploadError(f"file too large: maximum {config.UPLOAD_MAX_BYTES // (1024 * 1024)} MB", 413)
    ext = EXT[modality]
    given = re.split(r"[\\/]", str(filename or ""))[-1]
    if not given.lower().endswith(ext):
        raise UploadError(f"file extension not allowed for this modality: {ext} required", 415)
    info = validate_csv(raw) if modality == "tabular" else validate_npz(raw)
    ud = config.user_dir(modality)
    ud.mkdir(parents=True, exist_ok=True)
    stem = clean_stem(given)
    for _ in range(50):                      # la creazione esclusiva ('xb') evita la sovrascrittura anche con richieste concorrenti
        name = free_name(modality, stem)
        path = ud / (name + ext)
        try:
            with open(path, "xb") as f:
                f.write(raw)
            break
        except FileExistsError:
            continue
    else:
        raise UploadError("cannot find a free name", 409)
    return {"name": name, "modality": modality, "file": f"{ud.name}/{path.name}", "size_kb": round(len(raw) / 1024, 1),
            "user": True, "renamed": name != stem, "validation": info}
