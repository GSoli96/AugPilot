"""Job del benchmark tabulare per l'orchestratore (WP2 e WP4 di the project plan).

Per ogni combinazione (algoritmo, dataset, seed) non ancora presente nel file dei risultati:
  1. split stratificato 70/30 del dataset completo (`bench/tabular_split.py`);
  2. generazione dalla sola minoritaria del train (smote, fair-smote e smote-cdnn ricevono il train
     a due classi, perché lavorano sul confronto tra classi), con n = n_maj_train - n_min_train;
  3. estrazione delle righe sintetiche (classe minoritaria, tolte le righe reali dell'input);
  4. fedeltà/diversità (`evaluate_metrics.evaluate_all`) rispetto alla minoritaria del train,
     privacy (`bench/privacy.py`) con il test come riferimento, utilità (`evaluate_utility.py`);
  5. una riga JSON nel file dei risultati, scritta subito (ripresa: le combinazioni già scritte si saltano).

Uso tipico (un job per GPU):
  envs/tabular/Scripts/python.exe -m bench.run_tab_job --datasets pima yeast1 --algorithms all \
      --seeds 0 1 2 --epochs 1000 --results-file bench/results/tab_<macchina>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Backend senza finestre per matplotlib, anche nei worker figli: con Tk fuori dal thread principale il processo
# moriva con «Tcl_AsyncDelete: async handler deleted by the wrong thread» (trovato su server5002).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.orchlib import done_ids, progress, save_result  # noqa: E402
from bench.privacy import privacy_metrics  # noqa: E402
from bench.tabular_split import make_split, split_paths  # noqa: E402

TWO_CLASS_ALGOS = {"smote", "fair-smote", "smote-cdnn"}
# Richiedono LLM: restano "solo descritti" nel benchmark (D15, WP6); si possono forzare con --algorithms.
DESCRIBED_ONLY = {"octree", "harmonic", "synthetic-llm"}


def job_id(algo: str, dataset: str, seed: int, epochs: int) -> str:
    return f"tab|{algo}|{dataset}|s{seed}|e{epochs}"


def _seed_everything(seed: int) -> None:
    os.environ["AUG_SEED"] = str(seed)  # letto dai worker (workers/seed_util.py)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _row_keys(df: pd.DataFrame) -> list[tuple]:
    """Chiave di riga indipendente dal formato: i numeri diventano float arrotondati (6 == 6.0)."""
    num = df.apply(pd.to_numeric, errors="coerce").astype(float).round(6)
    as_num = num.map(lambda v: repr(float(v))).to_numpy()
    vals = np.where(num.notna().to_numpy(), as_num, df.astype(str).to_numpy())
    return [tuple(r) for r in vals]


def extract_synthetic(out_csv: Path, input_df: pd.DataFrame, train_cols: list[str], cls: str,
                      min_label, n_target: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Righe sintetiche della minoritaria: output filtrato per classe, meno il multinsieme delle righe reali."""
    gen = pd.read_csv(out_csv)
    info = {"n_output_rows": int(len(gen))}
    if list(gen.columns) != train_cols:
        if len(gen.columns) == len(train_cols):
            gen.columns = train_cols
        else:
            missing = [c for c in train_cols if c not in gen.columns]
            if missing:
                raise ValueError(f"colonne dell'output diverse dal train: mancano {missing[:5]}")
            gen = gen[train_cols]
    if cls in gen.columns and gen[cls].nunique() > 1:
        gen = gen[gen[cls].astype(str).str.replace(r"\.0$", "", regex=True)
                  == str(min_label).replace(".0", "")]
    gen = gen.reset_index(drop=True)
    # sottrae le righe reali presenti nell'output (alcuni metodi restituiscono originali + sintetici)
    real_counts: dict = {}
    for k in _row_keys(input_df[train_cols]):
        real_counts[k] = real_counts.get(k, 0) + 1
    keep = []
    removed = 0
    for i, k in enumerate(_row_keys(gen)):
        if real_counts.get(k, 0) > 0:
            real_counts[k] -= 1
            removed += 1
        else:
            keep.append(i)
    gen = gen.iloc[keep].reset_index(drop=True)
    info["n_real_rows_removed"] = removed
    info["n_synth_raw"] = int(len(gen))
    if n_target and len(gen) > n_target:
        gen = gen.sample(n=n_target, random_state=seed).reset_index(drop=True)
    gen[cls] = min_label
    info["n_synth"] = int(len(gen))
    info["synth_shortfall"] = int(max(0, n_target - len(gen)))
    return gen, info


def run_one(algo: str, dataset: str, seed: int, epochs: int, use_gpu: bool, device: str,
            allow_fallback: bool, keep_synth: bool) -> dict:
    from algorithms_runner import execute_augmentation
    rec: dict = {"modality": "tabular", "algorithm": algo, "dataset": dataset, "seed": seed, "epochs": epochs,
                 "host": socket.gethostname(), "started": datetime.now().isoformat(timespec="seconds")}
    meta = make_split(dataset, seed)
    p = split_paths(dataset, seed)
    cls, min_label, n_target = meta["class_column"], meta["min_label"], meta["n_to_generate"]
    rec.update({k: meta[k] for k in ("n_train", "n_test", "n_min_train", "n_maj_train", "n_min_test", "n_to_generate")})

    train = pd.read_csv(p["train"])
    test = pd.read_csv(p["test"])
    train_min = pd.read_csv(p["train_min"])
    cols = list(train.columns)

    in_csv = p["train_named"] if algo in TWO_CLASS_ALGOS else p["train_min"]
    out_dir = p["dir"] / "out"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"{algo}_e{epochs}.csv"
    if out_csv.exists():
        out_csv.unlink()  # output di un tentativo interrotto: si rigenera

    dc_file = None
    if algo.startswith("kamino"):
        # Import per percorso: run_castgan mette Algorithms/CasTGAN-main in testa a sys.path e il suo
        # main.py esegue parse_args() all'import ("unrecognized arguments: --datasets ...").
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("augpilot_main", str(ROOT / "main.py"))
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        resolve_dc_file = _m.resolve_dc_file
        dcp = resolve_dc_file(in_csv.stem, None, str(ROOT / "Algorithms" / "kamino-master" / "testdata"))
        dc_file = str(dcp) if dcp and dcp.exists() else None

    _seed_everything(seed)
    # Gli algoritmi a due classi devono generare abbastanza minoritari da bilanciare: passano n_target
    res = execute_augmentation(algo=algo, csv_path=in_csv, output_path=out_csv, use_gpu=use_gpu, device=device,
                               n_samples=n_target, epochs=epochs, dc_file=dc_file, kamino_weights="toggled",
                               allow_fallback=allow_fallback)
    for k in ("status", "method_used", "error_message", "total_time_s", "train_time_s", "gen_time_s",
              "train_peak_mem_mb", "gen_peak_mem_mb", "total_mem_peak_mb", "train_gpu_peak_mb", "gen_gpu_peak_mb"):
        if k in res:
            rec[k] = res[k]
    rec.setdefault("status", "FAILED")
    if rec["status"] not in ("SUCCESS", "FALLBACK") or not out_csv.exists():
        rec["eval_status"] = "SKIPPED"
        return rec

    try:
        input_df = pd.read_csv(in_csv)
        synth, info = extract_synthetic(out_csv, input_df, cols, cls, min_label, n_target, seed)
        rec.update(info)
        if keep_synth:
            synth.to_csv(out_dir / f"{algo}_e{epochs}_synth.csv", index=False)
        if len(synth) < 2:
            rec["eval_status"] = "NO_SYNTH"
            return rec
        from evaluate_metrics import evaluate_all
        t0 = time.perf_counter()
        fd = evaluate_all(train_min, synth)
        rec.update({f"fd_{k}": (float(v) if v is not None else None) for k, v in fd.items()})
        feats = [c for c in cols if c != cls]
        num = lambda d: d[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)  # noqa: E731
        test_min = test[test[cls] == min_label]
        rec.update({f"priv_{k}": v for k, v in privacy_metrics(num(train_min), num(synth), num(test_min), seed).items()})
        from evaluate_utility import utility_metrics
        rec.update(utility_metrics(train, test, synth, cls, min_label, seed))
        rec["eval_time_s"] = round(time.perf_counter() - t0, 3)
        rec["eval_status"] = "OK"
    except Exception as e:  # la generazione resta valida anche se la valutazione fallisce
        rec["eval_status"] = "ERROR"
        rec["eval_error"] = f"{type(e).__name__}: {e}"[:500]
        rec["eval_traceback"] = traceback.format_exc()[-1500:]
    finally:
        if out_csv.exists() and not keep_synth:
            out_csv.unlink()
    return rec


def main() -> int:
    from config import CORE_ALGORITHMS
    from algorithms_runner import resolve_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True, help="nomi in datasets_full/ senza .csv")
    ap.add_argument("--algorithms", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--results-file", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="ammette i metodi sostitutivi (status FALLBACK); default: FAILED")
    ap.add_argument("--keep-synth", action="store_true", help="conserva i CSV sintetici in bench/work/")
    ap.add_argument("--retry-failed", action="store_true",
                    help="rifà le combinazioni la cui ultima riga è FAILED o ha eval_status ERROR (la nuova riga si aggiunge; il KB tiene l'ultima)")
    a = ap.parse_args()

    algos = [x for x in CORE_ALGORITHMS if x not in DESCRIBED_ONLY] if a.algorithms == ["all"] else a.algorithms
    use_gpu = not a.no_gpu
    try:
        import torch
        use_gpu = use_gpu and torch.cuda.is_available()
    except Exception:
        use_gpu = False
    device = resolve_device(use_gpu, a.device)

    combos = [(al, d, s) for d in a.datasets for s in a.seeds for al in algos]
    Path(a.results_file).parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(a.results_file)
    if a.retry_failed and os.path.exists(a.results_file):
        last = {}
        with open(a.results_file, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    last[r["id"]] = (r.get("status"), r.get("eval_status"))
                except Exception:
                    pass
        # si rifanno le FAILED e le righe con valutazione in ERROR (es. warning trasformati in eccezioni)
        done = {k for k in done if last.get(k, (None, None))[0] != "FAILED" and last.get(k, (None, None))[1] != "ERROR"}
    total = len(combos)
    n_done = sum(job_id(al, d, s, a.epochs) in done for al, d, s in combos)
    print(f"[tab-job] {total} combinazioni, {n_done} già fatte, dispositivo {device}", flush=True)
    progress(n_done, total, "avvio")
    for al, d, s in combos:
        jid = job_id(al, d, s, a.epochs)
        if jid in done:
            continue
        t0 = time.perf_counter()
        try:
            rec = run_one(al, d, s, a.epochs, use_gpu, device, a.allow_fallback, a.keep_synth)
        except Exception as e:
            rec = {"modality": "tabular", "algorithm": al, "dataset": d, "seed": s, "epochs": a.epochs,
                   "status": "FAILED", "error_message": f"{type(e).__name__}: {e}"[:500],
                   "traceback": traceback.format_exc()[-1500:], "eval_status": "SKIPPED"}
        rec["id"] = jid
        rec["wall_time_s"] = round(time.perf_counter() - t0, 2)
        rec["finished"] = datetime.now().isoformat(timespec="seconds")
        save_result(a.results_file, json.loads(json.dumps(rec, default=str)))
        n_done += 1
        progress(n_done, total, f"{al} {d} s{s}: {rec.get('status')}/{rec.get('eval_status')}")
        print(f"[tab-job] {n_done}/{total} {jid} -> {rec.get('status')} / {rec.get('eval_status')} "
              f"({rec['wall_time_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
