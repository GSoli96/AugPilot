"""Seed comune per i worker: se la variabile d'ambiente AUG_SEED è impostata
(dal job del benchmark, bench/run_tab_job.py), fissa random, numpy e torch.
Senza AUG_SEED non fa nulla, quindi il comportamento storico resta invariato."""
import os
import random


def apply_env_seed() -> None:
    raw = os.environ.get("AUG_SEED")
    if raw is None or raw == "":
        return
    seed = int(raw)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
