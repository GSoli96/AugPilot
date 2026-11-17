"""Prepara (o corregge) la copia di lavoro di DataSynthesizer usata da PreFair: tmp/prefair_ds.
Applica la correzione per pandas 2 (DataFrame.append -> concat). Uso: envs/tabular python scripts/prefair_patch.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import algorithms_runner as ar  # noqa: E402

w = ar._prefair_build_datasynthesizer()
s = (w / "DataSynthesizer" / "lib" / "PrivBayes.py").read_text(encoding="utf-8")
print("PATCH", "full_space.append(" not in s)
