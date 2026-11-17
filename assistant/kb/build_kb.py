"""Costruisce la KB del benchmark.

    python -m assistant.kb.build_kb [--results bench/results] [--out assistant/kb]

Legge ricorsivamente tutti i `*.jsonl` sotto bench/results/ (una riga = una run), deduplica per `id` tenendo l'ultima
riga letta (ordine: percorso del file, poi riga) e scrive:
  benchmark.parquet (oppure benchmark.csv se pyarrow non c'e')  una riga per run
  dataset_meta.csv                                              meta-feature dei dataset (anche senza run)
Colonne di benchmark: id, modality, level, dataset, algorithm, seed, epochs, status, eval_status, time_s, mem_mb, host,
e le metriche numeriche (fd_*, priv_*, util_*, m_*). `time_s` = total_time_s (tabulare: addestramento + generazione),
altrimenti time_s, altrimenti wall_time_s.
Le meta-feature riusano catalog_info/datasets_<modalita>.json (script catalog_info/build_dataset_info.py); per i file
dei dataset non presenti nel JSON si richiamano le stesse funzioni dello script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.kb.metrics_spec import is_metric_column  # noqa: E402

KB_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "bench" / "results"
BASE_COLS = ["id", "modality", "level", "dataset", "algorithm", "seed", "epochs", "status", "eval_status", "time_s",
             "mem_mb", "host"]
META_COLS = ["modality", "level", "dataset", "source", "n_rows", "n_cols", "ir", "low_card_frac", "n_min",
             "n_series", "length_T", "dims_D", "has_labels", "n_classes", "n_graphs", "n_nodes", "avg_nodes",
             "avg_edges", "n_features"]


# ---------------------------------------------------------------- risultati
def read_jsonl_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in sorted(Path(results_dir).rglob("*.jsonl")):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("algorithm") and r.get("dataset"):
                    r["_file"] = str(f)
                    rows.append(r)
    return rows


def _row_id(r: Dict[str, Any]) -> str:
    return str(r.get("id") or f"{r.get('modality')}|{r.get('algorithm')}|{r.get('dataset')}|s{r.get('seed')}|e{r.get('epochs')}")


def _first_num(r: Dict[str, Any], keys) -> float:
    for k in keys:
        v = r.get(k)
        try:
            if v is not None and not isinstance(v, bool) and math.isfinite(float(v)):
                return float(v)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _rank(r: Dict[str, Any]):
    ok = str(r.get("status") or "").upper() in ("SUCCESS", "FALLBACK")
    return (ok, str(r.get("started") or ""))


def build_benchmark(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        # Una riga riuscita (SUCCESS o FALLBACK) vince sempre su una fallita; tra righe dello stesso tipo vince
        # quella con l'avvio («started», ISO) più recente; a parità o senza data vince l'ultima letta.
        # Lo stesso id può stare in più file (copie parziali da macchine liberate, rifacimenti su altre
        # macchine, file divisi per seed): l'ordine dei file non deve decidere quale riga vale, e un doppione
        # fallito (es. timeout su una macchina più lenta) non deve cancellare un risultato valido.
        k = _row_id(r)
        old = by_id.get(k)
        if old is None or _rank(r) >= _rank(old):
            by_id[k] = r
    recs = []
    for i, r in by_id.items():
        mod = str(r.get("modality") or "")
        rec = {"id": i, "modality": mod, "level": (r.get("level") or "") if mod == "graph" else "",
               "dataset": str(r["dataset"]), "algorithm": str(r["algorithm"]),
               "seed": r.get("seed"), "epochs": r.get("epochs"),
               "status": str(r.get("status") or "FAILED").upper(), "eval_status": str(r.get("eval_status") or ""),
               "time_s": _first_num(r, ("total_time_s", "time_s", "wall_time_s")),
               "mem_mb": _first_num(r, ("total_mem_peak_mb", "peak_mem_mb")), "host": str(r.get("host") or "")}
        for k, v in r.items():
            if is_metric_column(k) and not isinstance(v, (bool, str, list, dict)) and v is not None:
                try:
                    f = float(v)
                    rec[k] = f if math.isfinite(f) else float("nan")
                except (TypeError, ValueError):
                    pass
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=BASE_COLS)
    metric_cols = sorted(c for c in df.columns if c not in BASE_COLS)
    for c in ("seed", "epochs"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    return df[BASE_COLS + metric_cols].sort_values(["modality", "dataset", "algorithm", "seed", "epochs"]).reset_index(drop=True)


# ---------------------------------------------------------------- meta-feature
def _load_info_module():
    p = ROOT / "catalog_info" / "build_dataset_info.py"
    spec = importlib.util.spec_from_file_location("build_dataset_info", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _json_stats(modality: str) -> Dict[str, Dict[str, Any]]:
    f = ROOT / "catalog_info" / f"datasets_{modality}.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _nan(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _tabular_meta() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    inv = ROOT / "datasets_full" / "inventario.csv"
    if inv.is_file():
        d = pd.read_csv(inv)
        for _, r in d.iterrows():
            nf = _nan(r.get("colonne_feature"))
            out.append({"modality": "tabular", "level": "", "dataset": str(r["dataset"]), "source": "datasets_full/inventario.csv",
                        "n_rows": _nan(r.get("righe")), "n_cols": nf, "ir": _nan(r.get("IR")),
                        "low_card_frac": _nan(r.get("colonne_a_bassa_cardinalita")) / nf if nf else float("nan"),
                        "n_min": _nan(r.get("n_min")), "n_classes": _nan(r.get("classi"))})
            seen.add(str(r["dataset"]).lower())
    # dataset di datasets/ non presenti nell'inventario: i file *_min contengono solo la minoritaria
    for name, v in _json_stats("tabular").items():
        base = name[:-4] if name.endswith("_min") else name
        if base.lower() in seen:
            continue
        s = v.get("stats", {})
        nf = _nan(s.get("n_cols")) - 1
        out.append({"modality": "tabular", "level": "", "dataset": base, "source": "catalog_info/datasets_tabular.json (solo minoritaria)",
                    "n_rows": float("nan"), "n_cols": nf, "ir": float("nan"),
                    "low_card_frac": _nan(s.get("n_numeric_low_cardinality_le10")) / nf if nf > 0 else float("nan"),
                    "n_min": _nan(s.get("n_rows")), "n_classes": float("nan")})
    return out


def _ts_meta() -> List[Dict[str, Any]]:
    stats = _json_stats("ts")
    files = sorted((ROOT / "datasets_ts").glob("*.npz")) if (ROOT / "datasets_ts").is_dir() else []
    if any(f.stem not in stats for f in files):
        try:
            stats = {**stats, **_load_info_module().ts()}
        except Exception as e:  # noqa: BLE001
            print(f"[kb] meta ts dal file non riuscita: {type(e).__name__}: {e}", file=sys.stderr)
    out = []
    for name, v in stats.items():
        s = v.get("stats", {})
        out.append({"modality": "ts", "level": "", "dataset": name, "source": "catalog_info/datasets_ts.json",
                    "n_series": _nan(s.get("n_series")), "length_T": _nan(s.get("length_T")), "dims_D": _nan(s.get("dims_D")),
                    "has_labels": 1.0 if s.get("has_labels") else 0.0, "n_classes": _nan(s.get("n_classes")),
                    "ir": _nan(s.get("imbalance_ratio"))})
    return out


def _graph_meta() -> List[Dict[str, Any]]:
    stats = _json_stats("graph")
    files = sorted((ROOT / "datasets_graph").glob("*.pt")) if (ROOT / "datasets_graph").is_dir() else []
    if any(f.stem not in stats for f in files):
        try:
            stats = {**stats, **_load_info_module().graph()}
        except Exception as e:  # noqa: BLE001
            print(f"[kb] meta graph dal file non riuscita: {type(e).__name__}: {e}", file=sys.stderr)
    out = []
    for name, v in stats.items():
        s = v.get("stats", {})
        lvl = s.get("level")
        row = {"modality": "graph", "level": lvl or "", "dataset": name, "source": "catalog_info/datasets_graph.json",
               "n_features": _nan(s.get("n_features")), "n_classes": _nan(s.get("n_classes"))}
        if lvl == "node":   # un solo grafo; il rapporto di sbilanciamento e' quello del train (come usa il valutatore)
            row.update({"n_graphs": 1.0, "n_nodes": _nan(s.get("n_nodes")), "avg_nodes": _nan(s.get("n_nodes")),
                        "avg_edges": _nan(s.get("n_edges")), "ir": _nan(s.get("imbalance_ratio_train", s.get("imbalance_ratio")))})
        else:
            row.update({"n_graphs": _nan(s.get("n_graphs")), "avg_nodes": _nan(s.get("avg_nodes")),
                        "avg_edges": _nan(s.get("avg_edges")), "ir": _nan(s.get("imbalance_ratio"))})
        out.append(row)
    return out


def build_meta() -> pd.DataFrame:
    rows = _tabular_meta() + _ts_meta() + _graph_meta()
    df = pd.DataFrame(rows)
    for c in META_COLS:
        if c not in df.columns:
            df[c] = float("nan") if c not in ("modality", "level", "dataset", "source") else ""
    return df[META_COLS]


# ---------------------------------------------------------------- scrittura
def build(results_dir: Optional[Path] = None, out_dir: Optional[Path] = None, with_meta: bool = True) -> Dict[str, Any]:
    results_dir = Path(results_dir or RESULTS_DIR)
    out_dir = Path(out_dir or KB_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl_rows(results_dir)
    bench = build_benchmark(rows)
    fmt = "parquet"
    pq, csv = out_dir / "benchmark.parquet", out_dir / "benchmark.csv"
    try:
        bench.to_parquet(pq, index=False)
        if csv.exists():
            csv.unlink()
    except Exception:  # noqa: BLE001  pyarrow assente o file non scrivibile
        fmt = "csv"
        bench.to_csv(csv, index=False)
        if pq.exists():
            pq.unlink()
    meta_n = 0
    if with_meta:
        meta = build_meta()
        meta.to_csv(out_dir / "dataset_meta.csv", index=False)
        meta_n = len(meta)
    summary = {"results_dir": str(results_dir), "rows_read": len(rows), "runs": len(bench), "format": fmt,
               "out_dir": str(out_dir), "datasets_meta": meta_n,
               "by_modality": bench.groupby("modality").size().to_dict() if len(bench) else {}}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--out", default=str(KB_DIR))
    a = ap.parse_args(argv)
    s = build(Path(a.results), Path(a.out))
    print(json.dumps(s, ensure_ascii=False, indent=1))
    if not s["runs"]:
        print("[kb] attenzione: nessuna run trovata, il KB e' vuoto", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
