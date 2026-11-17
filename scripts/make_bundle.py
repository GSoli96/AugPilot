"""Crea il pacchetto di AugPilot da inviare alle macchine del laboratorio (WP5 di the project plan).

Contenuto dello zip (percorsi relativi alla cartella del progetto):
  - codice del progetto: file tracciati e non ignorati da git (`git ls-files -co --exclude-standard`), esclusi
    `Algorithms/` (aggiunto a parte) e le cartelle pesanti;
  - `Algorithms/`: solo file di codice e configurazione < 5 MB, senza venv, .git, checkpoint e output;
  - dataset del sottoinsieme (D20): `datasets_full/<tab>.csv`, `datasets_ts/<ts>.npz`, `datasets_graph/<g>.pt`;
  - `envs_lock/` e `scripts/setup_envs.ps1` per ricreare gli ambienti.
Uso: python scripts/make_bundle.py [--out tmp/bundle/augpilot_bundle.zip]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TAB = ["pima", "iris0", "yeast1", "ecoli1", "new-thyroid1", "Obesity_onevsrest_0", "yeast3", "vowel0",
       "cleveland-0_vs_4", "page-blocks-1-3_vs_4", "abalone9-18", "Migraine_onevsrest_3"]
TS = ["sine_small", "sine_cls", "cbf", "stock", "energy", "tv_air"]
GRAPH = ["mutag", "proteins", "enzymes", "imdb-binary", "ptc_mr", "nci1", "cora", "citeseer"]

SKIP_DIRS = re.compile(r"[\\/](clean_env|\.git|__pycache__|wandb|ckpt|checkpoints?|outputs?|results?|logs?|synthetic|"
                       r"sample_data|runs|mlruns|\.venv|venv|node_modules|LLaMA-Factory)([\\/]|$)", re.I)
CODE_EXT = {".py", ".json", ".yaml", ".yml", ".txt", ".toml", ".cfg", ".ini", ".out", ".ic", ".pyx", ".c", ".cpp",
            ".cu", ".h", ".sh", ".md"}
PROJECT_SKIP = ("Algorithms/", "envs/", "external/", "datasets_full/", "tmp/", "bench/work/", "archivio/")


def project_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    files = []
    for rel in out:
        rel = rel.strip().replace("\\", "/")
        if not rel or rel.startswith(PROJECT_SKIP):
            continue
        p = ROOT / rel
        if p.is_file() and p.stat().st_size < 20 * 1024 * 1024:
            files.append(p)
    return files


def algorithm_files() -> list[Path]:
    files = []
    for p in (ROOT / "Algorithms").rglob("*"):
        if p.is_file() and (p.suffix.lower() in CODE_EXT or p.name.upper().startswith("LICENSE")) and p.stat().st_size < 5 * 1024 * 1024 \
                and not SKIP_DIRS.search(str(p.relative_to(ROOT))):
            files.append(p)
    return files


def data_files() -> list[Path]:
    files = [ROOT / "datasets_full" / f"{d}.csv" for d in TAB]
    files += [ROOT / "datasets_ts" / f"{d}.npz" for d in TS]
    files += [ROOT / "datasets_graph" / f"{d}.pt" for d in GRAPH]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Mancano dataset: {missing}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tmp" / "bundle" / "augpilot_bundle.zip"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = {"progetto": project_files(), "algoritmi": algorithm_files(), "dati": data_files()}
    seen = set()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, files in groups.items():
            size = 0
            for f in files:
                rel = f.relative_to(ROOT).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                z.write(f, rel)
                size += f.stat().st_size
            print(f"{name}: {len(files)} file, {size / 1e6:.1f} MB")
    print(f"zip: {out} ({out.stat().st_size / 1e6:.1f} MB, {len(seen)} file)")


if __name__ == "__main__":
    main()
