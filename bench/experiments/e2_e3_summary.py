"""Riepilogo di E2/E3 (WP12): una riga per (modello, condizione) da un file JSONL di e2_e3_scenarios.

    python -m bench.experiments.e2_e3_summary [--in bench/experiments/out/e2_e3_local.jsonl]

Colonne: n scenari, completati, errori, numeri citati, numeri non ancorati, % ancorati (100 * (1 - non ancorati / numeri);
in `notools` «ancorato» vuol dire che il numero coincide con un valore prodotto dagli strumenti per lo stesso scenario,
quindi anche piccoli interi possono coincidere per caso), media della quota di strumenti attesi chiamati (solo `tools`),
latenza media in secondi e token medi. Uscita: lo stesso nome del file con suffisso `_summary.csv`.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def summarize(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r.get("condition", "tools"))].append(r)
    out = []
    for (model, cond), v in sorted(g.items()):
        nums = sum(r.get("n_numbers") or 0 for r in v)
        bad = sum(r.get("n_ungrounded") or 0 for r in v)
        hits = [r["expected_hit"] for r in v if r.get("expected_hit") is not None]
        out.append({"model": model, "condition": cond, "backend": v[0].get("backend", ""), "n": len(v),
                    "completed": sum(1 for r in v if r.get("completed")), "errors": sum(1 for r in v if r.get("error")),
                    "numbers": nums, "ungrounded": bad, "grounded_pct": round(100 * (1 - bad / nums), 1) if nums else None,
                    "expected_tools_hit": round(st.mean(hits), 3) if hits else None,
                    "latency_mean_s": round(st.mean(r.get("latency_s") or 0 for r in v), 1),
                    "tokens_mean": round(st.mean(r.get("tokens") or 0 for r in v))})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(ROOT / "bench" / "experiments" / "out" / "e2_e3_local.jsonl"))
    a = ap.parse_args()
    p = Path(a.inp)
    df = summarize(p)
    out = p.with_name(p.stem + "_summary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
