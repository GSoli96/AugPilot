"""Copertura del benchmark (WP8b): quante combinazioni previste ci sono nel KB e con quale esito.

    python -m bench.copertura [--kb-dir assistant/kb] [--out bench/COPERTURA.md]

Previsto per modalità (e livello per i grafi) = dataset del sottoinsieme (D15/WP6) x algoritmi misurati in quella
modalità x seed 0,1,2, con gli algoritmi compatibili con il livello (i grafi di livello nodo e grafo hanno metodi
diversi: l'insieme previsto per livello si ricava dagli algoritmi presenti nel KB per quel livello).
Esiti: SUCCESS, FALLBACK (reimplementazione dichiarata), FAILED, mancante (combinazione prevista senza riga).
Le eccezioni note (limiti di tempo, dataset senza etichette) sono in ECCEZIONI e vengono riportate a parte.
Righe SUCCESS con meno campioni del richiesto (synth_shortfall > 0, solo tabulare) elencate in fondo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = [0, 1, 2]
DATASETS = {
    "tabular": ["iris0", "ecoli1", "yeast1", "yeast3", "pima", "abalone9-18", "cleveland-0_vs_4", "new-thyroid1",
                "vowel0", "page-blocks-1-3_vs_4", "Migraine_onevsrest_3", "Obesity_onevsrest_0"],
    "ts": ["sine_small", "sine_cls", "cbf", "stock", "energy", "tv_air"],
}
# eccezioni decise e da dichiarare nel paper: (modalità, algoritmo, dataset) -> motivo
ECCEZIONI = {
    ("ts", "sigcwgan", "energy"): "oltre il limite di 2 ore per combinazione (s0 FAILED per timeout; s1, s2 non eseguiti)",
    ("tabular", "prefair", "vowel0"): "oltre il limite di 2 ore per combinazione (fermato sui 3 seed, nessuna riga)",
}
# modelli condizionati sulla classe: non applicabili ai dataset TS senza etichette (catalog_info/datasets_ts.json)
for _ds in ("energy", "stock", "tv_air"):
    for _alg in ("tsgm-cgan", "tsgm-cbetavae"):
        ECCEZIONI[("ts", _alg, _ds)] = "non applicabile: dataset senza etichette"


def load_bench(kb_dir: Path) -> pd.DataFrame:
    pq, csv = kb_dir / "benchmark.parquet", kb_dir / "benchmark.csv"
    if pq.is_file():
        return pd.read_parquet(pq)
    return pd.read_csv(csv)


def expected(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mod in ("tabular", "ts"):
        algos = sorted(b.loc[b.modality == mod, "algorithm"].unique())
        rows += [(mod, "", d, a, s) for d in DATASETS[mod] for a in algos for s in SEEDS]
    g = b[b.modality == "graph"]
    for lvl in sorted(g["level"].fillna("").unique()):
        gl = g[g.level.fillna("") == lvl]
        rows += [("graph", lvl, d, a, s) for d in sorted(gl.dataset.unique())
                 for a in sorted(gl.algorithm.unique()) for s in SEEDS]
    return pd.DataFrame(rows, columns=["modality", "level", "dataset", "algorithm", "seed"])


def build(kb_dir: Path) -> str:
    b = load_bench(kb_dir)
    b["level"] = b["level"].fillna("") if "level" in b else ""
    exp = expected(b)
    key = ["modality", "level", "dataset", "algorithm", "seed"]
    m = exp.merge(b[key + ["status", "eval_status"] + [c for c in ("synth_shortfall",) if c in b]], on=key, how="left")
    m["esito"] = m["status"].fillna("MANCANTE")
    extra = b.merge(exp, on=key, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"]
    out = ["# Copertura del benchmark", "",
           f"Fonte: `assistant/kb` (KB costruito con `python -m assistant.kb.build_kb`). "
           f"Previste {len(exp)} combinazioni (dataset x algoritmi x 3 seed).", "",
           "| Modalità | Livello | Previste | SUCCESS | FALLBACK | FAILED | Mancanti |", "|---|---|---|---|---|---|---|"]
    for (mod, lvl), g in m.groupby(["modality", "level"]):
        c = g["esito"].value_counts()
        out.append(f"| {mod} | {lvl or '-'} | {len(g)} | {c.get('SUCCESS', 0)} | {c.get('FALLBACK', 0)} | "
                   f"{c.get('FAILED', 0)} | {c.get('MANCANTE', 0)} |")
    c = m["esito"].value_counts()
    out.append(f"| **totale** | | {len(m)} | {c.get('SUCCESS', 0)} | {c.get('FALLBACK', 0)} | {c.get('FAILED', 0)} | "
               f"{c.get('MANCANTE', 0)} |")
    out += ["", "## Eccezioni note (da dichiarare nel paper)", ""]
    for (mod, alg, ds), why in ECCEZIONI.items():
        g = m[(m.modality == mod) & (m.algorithm == alg) & (m.dataset == ds)]
        out.append(f"- {mod} / {alg} / {ds}: {why}. Esiti: {', '.join(f'{k} {v}' for k, v in g['esito'].value_counts().items()) or 'nessuna combinazione prevista'}.")
    known = {(mo, a, d) for mo, a, d in ECCEZIONI}
    m["_nota"] = [(r.modality, r.algorithm, r.dataset) in known for r in m.itertuples()]
    bad = m[m["esito"].isin(["FAILED", "MANCANTE"]) & ~m["_nota"]]
    out += ["", f"## FAILED e mancanti fuori dalle eccezioni ({len(bad)})", ""]
    if bad.empty:
        out.append("Nessuno.")
    else:
        out += ["| Modalità | Livello | Dataset | Algoritmo | Seed | Esito |", "|---|---|---|---|---|---|"]
        out += [f"| {r.modality} | {r.level or '-'} | {r.dataset} | {r.algorithm} | {r.seed} | {r.esito} |"
                for r in bad.sort_values(key).itertuples()]
    fb = m[m["esito"] == "FALLBACK"].groupby(["modality", "algorithm"]).size()
    out += ["", "## FALLBACK (reimplementazioni dichiarate)", ""]
    out += [f"- {mo} / {a}: {n} combinazioni" for (mo, a), n in fb.items()] or ["Nessuno."]
    if "synth_shortfall" in m:
        sf = m[(m["esito"] == "SUCCESS") & (pd.to_numeric(m["synth_shortfall"], errors="coerce") > 0)]
        out += ["", f"## SUCCESS con meno campioni del richiesto ({len(sf)})", ""]
        out += [f"- {r.dataset} / {r.algorithm} / s{r.seed}: mancano {int(r.synth_shortfall)}" for r in sf.sort_values(key).itertuples()] or ["Nessuno."]
    out += ["", f"## Righe del KB fuori dal sottoinsieme previsto ({len(extra)})", "",
            "Corse di prova o dataset non del sottoinsieme: non entrano nel conteggio."]
    if len(extra):
        out += [f"- {mo} / {d}: {n}" for (mo, d), n in extra.groupby(["modality", "dataset"]).size().items()]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb-dir", default=str(ROOT / "assistant" / "kb"))
    ap.add_argument("--out", default=str(ROOT / "bench" / "COPERTURA.md"))
    a = ap.parse_args(argv)
    text = build(Path(a.kb_dir))
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
