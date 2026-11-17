"""Job del benchmark per serie temporali e grafi, per l'orchestratore (WP4 di the project plan).

Come `bench/run_tab_job.py`: per ogni combinazione (algoritmo, dataset, seed) non ancora presente nel file dei
risultati esegue il worker della modalità (`modalities.run_algorithm`), calcola le metriche
(`modalities/<mod>/metrics.py: evaluate`) e scrive subito una riga JSON. Le combinazioni già scritte si saltano.
Per i grafi abbina solo algoritmi e dataset dello stesso livello (nodo / grafo).

Uso (un job per GPU; l'ambiente deve essere quello della modalità):
  envs/ts/Scripts/python.exe -m bench.run_mod_job --modality ts --datasets sine_small cbf --algorithms all \
      --seeds 0 1 2 --epochs 50 --results-file bench/results/ts_<macchina>.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.orchlib import done_ids, progress, save_result  # noqa: E402

EXT = {"ts": ".npz", "graph": ".pt"}
WORK = ROOT / "bench" / "work"


def job_id(modality: str, algo: str, dataset: str, seed: int, epochs: int) -> str:
    return f"{modality}|{algo}|{dataset}|s{seed}|e{epochs}"


def dataset_level(path: str) -> str | None:
    """'node' o 'graph' per i dataset dei grafi (dal campo kind del file .pt)."""
    try:
        import torch
        d = torch.load(path, weights_only=False)
        return d.get("kind")
    except Exception:
        return None


def compatible(spec, level: str | None) -> bool:
    if spec.modality != "graph" or level is None:
        return True
    return (spec.extra or {}).get("level", level) == level


def main() -> int:
    from modalities import list_specs, run_algorithm

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ts", "graph"], required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--algorithms", nargs="+", default=["all"], help="'all' = tutti gli algoritmi executable")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout", type=int, default=7200, help="secondi per combinazione")
    ap.add_argument("--results-file", required=True)
    ap.add_argument("--keep-output", action="store_true")
    a = ap.parse_args()

    specs = {s.key: s for s in list_specs(a.modality)}
    if a.algorithms == ["all"]:
        algos = [k for k, s in specs.items() if s.status == "executable"]
    else:
        unknown = [k for k in a.algorithms if k not in specs]
        if unknown:
            print(f"[mod-job] algoritmi sconosciuti: {unknown}", file=sys.stderr)
            return 2
        algos = a.algorithms

    ds_mod = importlib.import_module(f"modalities.{a.modality}.datasets")
    met_mod = importlib.import_module(f"modalities.{a.modality}.metrics")
    data_paths = {d: ds_mod.resolve(d) for d in a.datasets}
    levels = {d: dataset_level(p) if a.modality == "graph" else None for d, p in data_paths.items()}

    combos = [(al, d, s) for d in a.datasets for s in a.seeds for al in algos if compatible(specs[al], levels[d])]
    Path(a.results_file).parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(a.results_file)
    total = len(combos)
    n_done = sum(job_id(a.modality, al, d, s, a.epochs) in done for al, d, s in combos)
    print(f"[mod-job] {a.modality}: {total} combinazioni, {n_done} già fatte", flush=True)
    progress(n_done, total, "avvio")
    out_dir = WORK / a.modality
    out_dir.mkdir(parents=True, exist_ok=True)

    for al, d, s in combos:
        jid = job_id(a.modality, al, d, s, a.epochs)
        if jid in done:
            continue
        t0 = time.perf_counter()
        out = out_dir / f"{d}__{al}__s{s}__e{a.epochs}{EXT[a.modality]}"
        rec = {"id": jid, "modality": a.modality, "algorithm": al, "dataset": d, "seed": s, "epochs": a.epochs,
               "host": socket.gethostname(), "started": datetime.now().isoformat(timespec="seconds"),
               "strategy": specs[al].strategy, "task": specs[al].task, "level": levels[d]}
        try:
            res = run_algorithm(a.modality, al, data_paths[d], str(out), a.n_samples, a.epochs, a.device, s,
                                timeout=a.timeout)
            rec.update({k: v for k, v in res.items() if k not in rec})
            rec.setdefault("status", "FAILED")
            if rec["status"] == "SUCCESS" and out.exists():
                try:
                    t1 = time.perf_counter()
                    rec.update({f"m_{k}": v for k, v in met_mod.evaluate(data_paths[d], str(out)).items()})
                    rec["eval_time_s"] = round(time.perf_counter() - t1, 3)
                    rec["eval_status"] = "OK"
                except Exception as e:
                    rec["eval_status"] = "ERROR"
                    rec["eval_error"] = f"{type(e).__name__}: {e}"[:500]
            else:
                rec["eval_status"] = "SKIPPED"
        except Exception as e:
            rec.update({"status": "FAILED", "error_message": f"{type(e).__name__}: {e}"[:500],
                        "traceback": traceback.format_exc()[-1500:], "eval_status": "SKIPPED"})
        finally:
            if out.exists() and not a.keep_output:
                out.unlink()
        rec["wall_time_s"] = round(time.perf_counter() - t0, 2)
        rec["finished"] = datetime.now().isoformat(timespec="seconds")
        save_result(a.results_file, json.loads(json.dumps(rec, default=str)))
        n_done += 1
        progress(n_done, total, f"{al} {d} s{s}: {rec.get('status')}/{rec.get('eval_status')}")
        print(f"[mod-job] {n_done}/{total} {jid} -> {rec.get('status')} / {rec.get('eval_status')} "
              f"({rec['wall_time_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
