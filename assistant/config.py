"""Configurazione: percorsi, modello, chiave API (mai stampata, mai restituita), limiti del ciclo agentico."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = Path(os.environ.get("CLAUDE_KEY_FILE", str(Path.home() / "Desktop" / "prog" / "ChiaveClaude.json")))  # formato {"api_key": "..."}
DEFAULT_MODEL = "claude-sonnet-5-5"

# Cartelle dedicate alle esecuzioni lanciate dalla UI e dall'assistente (tabulare): non toccano quelle del
# benchmark (augmented_datasets/, runtime_results/, evaluation_metrics/) ne' il loro checkpoint.
UI_AUGMENTED_DIR = "augmented_ui"
UI_RESULTS_DIR = "runtime_results_ui"
UI_EVAL_DIR = "evaluation_metrics_ui"
RESULTS_DIRS = ["runtime_results", UI_RESULTS_DIR]   # dove si leggono i tempi/stati tabulari
# Serie temporali e grafi: run_modality.py scrive in augmented_<m>/ e runtime_results_<m>/ (benchmark e prove a mano);
# le esecuzioni lanciate dalla UI e dall'assistente usano cartelle dedicate (--out-dir, --results-dir).
UI_OUT_DIRS = {"ts": "augmented_ui_ts", "graph": "augmented_ui_graph"}
UI_RES_DIRS = {"ts": "runtime_results_ui_ts", "graph": "runtime_results_ui_graph"}
BENCH_OUT_DIRS = {"ts": "augmented_ts", "graph": "augmented_graph"}
BENCH_RES_DIRS = {"ts": "runtime_results_ts", "graph": "runtime_results_graph"}
# Dataset caricati dall'utente (tabulare: CSV con la classe nell'ultima colonna; ts: .npz con `data` (N,T,D) e `labels`).
# Per i grafi non c'e' caricamento. I valori possono essere percorsi assoluti (i test li puntano a cartelle temporanee).
USER_DIRS = {"tabular": "datasets_user", "ts": "datasets_user_ts"}
UPLOAD_MAX_BYTES = 50 * 1024 * 1024

# Unici percorsi leggibili dagli strumenti (relativi a ROOT).
READ_DIRS = ["datasets", "datasets_ts", "datasets_graph", "augmented_datasets", "augmented_ts",
             "augmented_graph", "runtime_results", "runtime_results_ts", "runtime_results_graph",
             "evaluation_metrics", UI_AUGMENTED_DIR, UI_RESULTS_DIR, UI_EVAL_DIR,
             "datasets_user", "datasets_user_ts", "augmented_ui_ts", "augmented_ui_graph",
             "runtime_results_ui_ts", "runtime_results_ui_graph"]
DATASET_DIRS = {"tabular": "datasets", "ts": "datasets_ts", "graph": "datasets_graph"}
DATASET_EXT = {"tabular": (".csv",), "ts": (".npz",), "graph": (".pt",)}
MODALITIES = ("tabular", "ts", "graph")


def user_dir(modality: str) -> Optional[Path]:
    """Cartella dei dataset caricati dall'utente per la modalita' (None per i grafi). Non la crea."""
    d = USER_DIRS.get(modality)
    return (ROOT / d) if d else None

LIMITS = {"max_steps": 10, "max_seconds": 120.0, "max_total_tokens": 80_000, "max_output_tokens": 2048,
          "max_tool_result_chars": 12_000}
CONFIRM_MAX_ALGOS = 1     # oltre questo numero di algoritmi serve la conferma dell'utente
CONFIRM_MAX_DATASETS = 1
CONFIRM_MAX_EPOCHS = 5    # oltre queste epoche serve la conferma dell'utente

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def get_model() -> str:
    return os.environ.get("ASSISTANT_MODEL", "").strip() or DEFAULT_MODEL


def get_api_key() -> Tuple[Optional[str], str]:
    """Restituisce (chiave, origine). Ordine: ANTHROPIC_API_KEY, poi ChiaveClaude.json.
    Origine: 'env' | 'file' | 'missing' | 'invalid file'. Non cerca altre chiavi."""
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k, "env"
    try:
        if KEY_FILE.is_file():
            k = str(json.loads(KEY_FILE.read_text(encoding="utf-8")).get("api_key", "")).strip()
            return (k, "file") if k else (None, "invalid file")
    except (OSError, ValueError, AttributeError):
        return None, "invalid file"
    return None, "missing"


def safe_path(rel: str) -> Path:
    """Risolve un percorso relativo a ROOT consentendo solo i file sotto READ_DIRS."""
    if not isinstance(rel, str) or not rel.strip() or "\x00" in rel:
        raise ValueError("invalid path")
    p = Path(rel)
    if p.is_absolute() or p.drive:
        raise ValueError("absolute paths are not allowed")
    full = (ROOT / p).resolve()
    for d in READ_DIRS:
        base = (ROOT / d).resolve()
        if base in full.parents:
            if not full.is_file():
                raise FileNotFoundError(f"file does not exist: {rel}")
            return full
    raise PermissionError(f"path outside the allowed folders: {READ_DIRS}")
