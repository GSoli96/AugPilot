"""CLI per le modalità serie temporali (ts) e grafi (graph).

Esempi:
  python run_modality.py --list
  python run_modality.py --modality ts --algorithms jitter timevae --datasets sine_small --epochs 5
Ogni modalità espone `modalities/<mod>/datasets.py: resolve(name)->path` e
`modalities/<mod>/metrics.py: evaluate(original, generated)->dict`.
"""
import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from modalities import list_specs, run_algorithm

ROOT = Path(__file__).resolve().parent
OUT = {"ts": ROOT / "augmented_ts", "graph": ROOT / "augmented_graph"}
RES = {"ts": ROOT / "runtime_results_ts", "graph": ROOT / "runtime_results_graph"}
EXT = {"ts": ".npz", "graph": ".pt"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--modality", choices=["ts", "graph"])
    ap.add_argument("--algorithms", nargs="+", default=[])
    ap.add_argument("--datasets", nargs="+", default=[])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default=None,
                    help="cartella dei dataset generati (default: augmented_ts/ o augmented_graph/)")
    ap.add_argument("--results-dir", "--results_dir", dest="results_dir", default=None,
                    help="cartella dei CSV dei risultati (default: runtime_results_ts/ o runtime_results_graph/)")
    a = ap.parse_args()
    if a.list or not a.modality:
        for s in list_specs(a.modality):
            print(f"{s.modality:5} {s.key:22} {s.status:10} {s.strategy:14} {s.name}")
        return
    ds_mod = importlib.import_module(f"modalities.{a.modality}.datasets")
    met_mod = None if a.no_eval else importlib.import_module(f"modalities.{a.modality}.metrics")
    rows = []
    if a.out_dir:
        OUT[a.modality] = Path(a.out_dir)
    if a.results_dir:
        RES[a.modality] = Path(a.results_dir)
    OUT[a.modality].mkdir(parents=True, exist_ok=True); RES[a.modality].mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        data = ds_mod.resolve(ds)
        for algo in a.algorithms:
            out = OUT[a.modality] / f"{ds}__{algo}{EXT[a.modality]}"
            rec = run_algorithm(a.modality, algo, data, str(out), a.n_samples, a.epochs, a.device, a.seed)
            rec.update({"dataset": ds, "algorithm": algo, "modality": a.modality, "epochs": a.epochs})
            if rec.get("status") == "SUCCESS" and met_mod is not None:
                try:
                    rec.update(met_mod.evaluate(data, str(out)))
                except Exception as e:  # noqa: BLE001
                    rec["eval_error"] = f"{type(e).__name__}: {e}"
            print(json.dumps({k: v for k, v in rec.items() if k != "error_message"}), flush=True)
            if rec.get("error_message"):
                print("  ERRORE:", rec["error_message"][:400])
            rows.append(rec)
    if rows:
        f = RES[a.modality] / f"runtime_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame(rows).to_csv(f, index=False)
        print("salvato", f)


if __name__ == "__main__":
    main()
