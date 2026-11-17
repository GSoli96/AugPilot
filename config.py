#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ALGORITHMS_DIR = BASE_DIR / "Algorithms"
DATASETS_DIR = BASE_DIR / "datasets"
AUGMENTED_DIR = BASE_DIR / "augmented_datasets"
RESULTS_DIR = BASE_DIR / "runtime_results"

# Flag to include algorithms (kept for backward compatibility)
USARE_ALTRI_ALGORITMI = True

# PreFair: colonna sensibile per dataset (nome dataset base SENZA "_min" -> nome colonna).
# Vuoto di proposito: i dataset hanno colonne anonime (Attr1, Attr2, ...), quindi la scelta per dataset spetta
# all'utente. Se un dataset non e' qui, algorithms_runner._prefair_pick_columns ricade sull'euristica sul nome
# (sex/gender/race/age) o sulla prima colonna a pochi valori. La colonna usata finisce in method_used del risultato.
PREFAIR_SENSITIVE = {}

# Default training epochs for deep learning / generative augmentation algorithms
DEFAULT_EPOCHS = 1000

# All algorithms directly available in Algorithms/
CORE_ALGORITHMS = [
    "smote",
    "fair-smote",
    "smote-cdnn",
    "goggle",
    "castgan",
    "tabfairgan",
    "cuts",
    "tabular-argn",
    "prefair",
    "ddpm",
    "ctabgan",
    "ctabgan-plus",
    "tvae",
    "syrfd",
    "tabsyn",
    "tabdiff",
    "octree",
    "harmonic",
    "synthetic-llm",
    "kamino",
]

# Additional algorithms alias (for backward compatibility)
EXTRA_ALGORITHMS = []

# Algorithms that are only described in the catalogue (surrogate wrappers, no real run in the benchmark):
# `--algorithms all` skips them; they can still be requested by name (main.py prints a warning).
DESCRIBED_ONLY = ["octree", "harmonic", "synthetic-llm"]

# Python environment paths for algorithms with isolated environments
def _python_in(env_dir: Path) -> list:
    """Possibili eseguibili python dentro una cartella ambiente (Linux/macOS e Windows)."""
    if sys.platform == "win32":
        # Su Windows un file "bin/python" e' di un ambiente Linux/macOS copiato: non e' eseguibile.
        return [
            env_dir / "Scripts" / "python.exe",  # venv
            env_dir / "python.exe",              # conda
        ]
    return [env_dir / "bin" / "python"]          # Linux/macOS (venv e conda)


def get_env_python(env_name: str) -> str:
    """Find python executable for a given environment, or fallback to sys.executable.

    Cerca prima negli ambienti del progetto (envs/<nome>), poi nelle installazioni conda
    standard dell'utente. Su Windows cerca Scripts/python.exe e python.exe oltre a bin/python.
    """
    home = Path.home()
    env_dirs = [BASE_DIR / "envs" / env_name]

    # 1. Detect active conda installation if available in PATH
    conda_exe = shutil.which("conda")
    if conda_exe:
        try:
            conda_base = Path(conda_exe).resolve().parent.parent
            env_dirs.append(conda_base / "envs" / env_name)
        except Exception:
            pass

    # 2. Check standard conda locations relative to current user home
    env_dirs.extend([
        home / "anaconda3" / "envs" / env_name,
        home / "miniconda3" / "envs" / env_name,
        home / "miniforge3" / "envs" / env_name,
        home / "opt" / "anaconda3" / "envs" / env_name,
        Path("/opt/anaconda3/envs") / env_name,
        Path("/opt/miniconda3/envs") / env_name,
        Path("/opt/homebrew/Caskroom/miniconda/base/envs") / env_name,
        ALGORITHMS_DIR / "tab-ddpm-main" / env_name,
        BASE_DIR / "tab-ddpm-main" / env_name,
    ])

    for d in env_dirs:
        for c in _python_in(d):
            if c.exists() and os.access(c, os.X_OK):
                return str(c)
    return sys.executable


def _first_env_python(*names: str) -> str:
    """Python del primo ambiente esistente tra `names` (envs/<nome>, poi conda); altrimenti sys.executable."""
    for n in names:
        py = get_env_python(n)
        if py != sys.executable:
            return py
    return sys.executable


# Ambienti per famiglia. envs/tabular e' l'ambiente comune (torch cu124, tutti i tabulari tranne TabularARGN);
# envs/argn ha mostlyai-engine (richiede torch piu' recente). Se un ambiente dedicato esiste ha la precedenza.
ENV_PYTHONS = {
    "default": sys.executable,
    "tabular": _first_env_python("tabular"),
    "argn": _first_env_python("argn"),
    "clean_env": _first_env_python("clean_env", "tabddpm", "tabular"),
    "goggle_env": _first_env_python("goggle_env2", "goggle_env", "goggle", "tabular"),
    "octree": _first_env_python("octree", "tabular"),
    "harmonic": _first_env_python("harmonic", "tabular"),
}
