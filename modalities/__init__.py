"""Modalità aggiuntive del benchmark: serie temporali (ts) e grafi (graph).

Contratto comune (vale per ogni algoritmo di ogni modalità):
- ogni algoritmo ha un worker `modalities/<mod>/workers/<key>.py` lanciato come sottoprocesso
  con il python dell'ambiente indicato (envs/<env>/Scripts/python.exe, altrimenti quello corrente);
- argomenti: --data <percorso o nome dataset> --out <file di output> --n_samples N --epochs E
  --device cpu|cuda --seed S [--label_col ...];
- l'ULTIMA riga di stdout è un JSON: {"status": "SUCCESS"|"FAILED", "time_s": float,
  "peak_mem_mb": float, "n_generated": int, "error_message": str};
- un fallimento deve restare un fallimento: mai ripiegare in silenzio su un altro metodo.
"""
from .base import AlgoSpec, REGISTRY, register, get_spec, list_specs, run_algorithm  # noqa: F401
