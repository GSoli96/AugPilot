"""
Dataset Augmentation con Strategie CoT Avanzate + Ollama Cloud
==============================================================
Strategie implementate:
  - Explicit Schema Constraints
  - Distribution Guidance
  - CoT Gerarchica (Multi-livello)          [3]
  - CoT con Verifica (Self-Consistency)     [4]
  - CoT con ReAct Pattern                  [5]
  - CoT Ensemble (Multi-Perspective)       [6]
  - CoT con Albero Decisionale             [7]
  - CoT con Calcolo della Confidenza       [8]

- Esecuzione sequenziale (no thread)
- Auto-restart su blocco tramite SIGALRM
- Checkpoint per resume automatico
- Solo la classe minoritaria viene aumentata
"""

import os
import re
import sys
import json
import time
import signal
import logging
import hashlib
from io import StringIO
from enum import Enum
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from ollama import Client

# ─────────────────────────────────────────────
#  CONFIGURAZIONE GLOBALE & POOL API KEYS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  CONFIGURAZIONE GLOBALE & POOL API KEYS
# ─────────────────────────────────────────────
# Le chiavi non stanno nel codice: si leggono da un file JSON esterno
# (percorso in OLLAMA_KEYS_FILE, altrimenti il default sotto). Formato: {"CLOUD_API_KEYS": [{"key","name","type"}, ...]}
OLLAMA_KEYS_FILE = os.environ.get("OLLAMA_KEYS_FILE", r"<home>\Desktop\prog\ChaviOllama.json")


def _load_api_keys(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"API_KEYS": data["CLOUD_API_KEYS"]}


API_KEYS = _load_api_keys(OLLAMA_KEYS_FILE)

MODELS = [
    "nemotron-3-nano:30b-cloud",
    "nemotron-3-super:cloud",
    "nemotron-3-ultra:cloud",
]
PRIMARY_MODEL  = MODELS[0]

LABEL_COLUMN   = "class"
N_FEW_SHOT     = 5
N_GEN_AT_TIME  = 10

# Percorsi relativi alla radice del progetto (non alla directory di lancio)
_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(_ROOT, "datasets") + os.sep           # cartella dataset di input
OUTPUT_DIR     = os.path.join(_ROOT, "LLM_raw") + os.sep               # risposte grezze LLM per strategia
AUGMENTED_DIR  = os.path.join(_ROOT, "augmented_datasets") + os.sep    # dataset aumentati finali
CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints_llm") + os.sep
LOG_FILE       = os.path.join(_ROOT, "augmentation.log")

# Retry
MAX_RETRIES    = 3
RETRY_DELAY    = 5      # secondi tra retry
# Timeout singola chiamata LLM (SIGALRM, Linux/macOS)
REQUEST_TIMEOUT = 120

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  STRATEGIA
# ─────────────────────────────────────────────
class AugStrategy(str, Enum):
    SCHEMA_CONSTRAINTS   = "schema_constraints"
    DISTRIBUTION_GUIDE   = "distribution_guidance"
    COT_HIERARCHICAL     = "cot_hierarchical"
    COT_SELF_CONSISTENCY = "cot_self_consistency"
    COT_REACT            = "cot_react"
    COT_ENSEMBLE         = "cot_ensemble"
    COT_DECISION_TREE    = "cot_decision_tree"
    COT_CONFIDENCE       = "cot_confidence"

# ─────────────────────────────────────────────
#  KEY MANAGER CON ROTAZIONE AUTOMATICA
# ─────────────────────────────────────────────
class KeyManager:
    """
    Gestisce un pool di API keys per Ollama Cloud.
    Supporta un dizionario con chiave 'API_KEYS' o una lista di elementi (dict o stringhe).
    Quando una chiave esaurisce la quota (429, limite mensile, unauthorized, ecc.),
    passa automaticamente alla chiave successiva.
    """
    def __init__(self, api_keys, host: str = "https://ollama.com"):
        if isinstance(api_keys, dict) and "API_KEYS" in api_keys:
            raw_list = api_keys["API_KEYS"]
        elif isinstance(api_keys, list):
            raw_list = api_keys
        else:
            raw_list = []

        self.api_keys = []
        for item in raw_list:
            if isinstance(item, dict):
                k = item.get("key", "").strip()
                if k:
                    self.api_keys.append({
                        "key": k,
                        "name": item.get("name", "unknown"),
                        "type": item.get("type", "free"),
                    })
            elif isinstance(item, str) and item.strip():
                self.api_keys.append({
                    "key": item.strip(),
                    "name": "unnamed",
                    "type": "free",
                })

        self.current_idx = 0
        self.host = host
        self.client: Optional[Client] = None
        self._init_client()

    def _init_client(self):
        if self.current_idx < len(self.api_keys):
            item = self.api_keys[self.current_idx]
            self.client = Client(
                host=self.host,
                headers={"Authorization": f"Bearer {item['key']}"},
            )
        else:
            self.client = None

    @property
    def current_item(self) -> Optional[dict]:
        if self.current_idx < len(self.api_keys):
            return self.api_keys[self.current_idx]
        return None

    @property
    def current_key(self) -> Optional[str]:
        item = self.current_item
        return item["key"] if item else None

    @property
    def current_name(self) -> str:
        item = self.current_item
        return item["name"] if item else "None"

    @property
    def current_type(self) -> str:
        item = self.current_item
        return item["type"] if item else "None"

    def rotate_key(self, reason: str = "") -> bool:
        """
        Passa alla chiave successiva nel pool.
        Ritorna True se è disponibile una nuova chiave, False se tutte sono esaurite.
        """
        old_idx = self.current_idx
        old_item = self.current_item
        old_desc = f"'{old_item['name']}' ({old_item['type']}, {old_item['key'][:8]}...)" if old_item else "None"

        self.current_idx += 1
        if self.current_idx < len(self.api_keys):
            new_item = self.api_keys[self.current_idx]
            new_desc = f"'{new_item['name']}' ({new_item['type']}, {new_item['key'][:8]}...)"
            log.warning(
                f"[KEY_MANAGER] Chiave #{old_idx} {old_desc} terminata/non valida: {reason}. "
                f"Passaggio automatico alla chiave #{self.current_idx} {new_desc}."
            )
            self._init_client()
            return True
        else:
            log.critical(
                f"[KEY_MANAGER] Tutte le {len(self.api_keys)} API key nel pool sono state esaurite! "
                f"Ultimo errore: {reason}"
            )
            self.client = None
            return False

    def is_quota_or_auth_error(self, exc: Exception) -> bool:
        """Determina se l'eccezione indica esaurimento quota, rate limit o autorizzazione non valida."""
        status_code = getattr(exc, "status_code", None)
        if status_code in (401, 402, 403, 429):
            return True
        msg = str(exc).lower()
        quota_keywords = [
            "limit", "quota", "usage", "credit", "unauthorized",
            "forbidden", "exceeded", "payment", "status code: 429",
            "status code: 401", "status code: 403"
        ]
        return any(k in msg for k in quota_keywords)


key_manager = KeyManager(API_KEYS)


def check_model_availability(model: str) -> bool:
    """
    Effettua una pre-verifica per controllare se il modello è disponibile su Ollama Cloud.
    Gestisce la rotazione automatica delle API key in caso di esaurimento quota.
    """
    log.info(f"  [PRE-CHECK] Verifica disponibilità modello '{model}'...")
    while key_manager.client is not None:
        try:
            resp = key_manager.client.chat(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                options={"temperature": 0.1},
            )
            if resp and resp.message and resp.message.content:
                log.info(f"  [PRE-CHECK] Modello '{model}' DISPONIBILE.")
                return True
            log.warning(f"  [PRE-CHECK] Risposta vuota dal modello '{model}'.")
            return False
        except Exception as e:
            if key_manager.is_quota_or_auth_error(e):
                desc_key = f"#{key_manager.current_idx} ('{key_manager.current_name}', {key_manager.current_type})"
                log.warning(f"  [PRE-CHECK] Errore quota/auth su chiave {desc_key}: {e}")
                if key_manager.rotate_key(reason=str(e)):
                    continue
                else:
                    log.critical("  [PRE-CHECK] Nessuna chiave API residua disponibile.")
                    return False
            else:
                log.error(f"  [PRE-CHECK] Modello '{model}' NON disponibile o errore: {e}")
                return False
    return False


def ollama_generate(prompt: str, model: str = PRIMARY_MODEL, temperature: float = 0.7) -> str:
    """
    Chiama Ollama Cloud in modo sequenziale.
    - Se l'API key corrente ha terminato la quota / crediti, passa automaticamente alla successiva e riprova.
    - Retry MAX_RETRIES volte con backoff per errori transitori.
    - SIGALRM per timeout su singola chiamata (Linux/macOS).
    """
    attempt = 1
    while attempt <= MAX_RETRIES:
        if key_manager.client is None:
            log.critical("  [LLM] Nessuna chiave API disponibile.")
            return ""

        desc_key = f"#{key_manager.current_idx} ('{key_manager.current_name}', {key_manager.current_type})"
        log.info(f"  [LLM] model={model} (chiave {desc_key}) tentativo={attempt}/{MAX_RETRIES}")

        try:
            # ── timeout via SIGALRM ──────────────────
            if hasattr(signal, "SIGALRM"):
                def _handler(signum, frame):
                    raise TimeoutError(f"Timeout {REQUEST_TIMEOUT}s superato")
                signal.signal(signal.SIGALRM, _handler)
                signal.alarm(REQUEST_TIMEOUT)

            response = key_manager.client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": temperature},
            )

            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            content = response.message.content
            if content and content.strip():
                return content.strip()

            log.warning(f"  [LLM] risposta vuota da {model}")

        except TimeoutError as e:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            log.error(f"  [LLM] {e}")
        except Exception as e:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            # Controllo se è errore di quota/autorizzazione per ruotare la chiave
            if key_manager.is_quota_or_auth_error(e):
                desc_key = f"#{key_manager.current_idx} ('{key_manager.current_name}', {key_manager.current_type})"
                log.warning(f"  [LLM] Rilevato limite/errore quota su chiave {desc_key}: {e}")
                if key_manager.rotate_key(reason=str(e)):
                    log.info("  [LLM] Riprovo immediatamente con la nuova chiave...")
                    continue
                else:
                    log.critical("  [LLM] Tutte le API key sono esaurite!")
                    return ""
            else:
                log.error(f"  [LLM] errore ({model}): {e}")

        # backoff prima del prossimo tentativo per errori transitori
        if attempt < MAX_RETRIES:
            wait = RETRY_DELAY * attempt
            log.info(f"  [LLM] attendo {wait}s…")
            time.sleep(wait)
        attempt += 1

    log.critical("  [LLM] tutti i tentativi falliti")
    return ""

# ─────────────────────────────────────────────
#  ANALISI DATASET  (pandas)
# ─────────────────────────────────────────────
class DatasetProfile:
    """Profilo statistico completo, calcolato una volta sola per dataset."""

    def __init__(self, df: pd.DataFrame):
        self.n_rows = len(df)
        self.n_cols = len(df.columns)
        self.columns = df.columns.tolist()

        self.label_distribution = df[LABEL_COLUMN].value_counts().to_dict()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categ_cols   = [c for c in df.select_dtypes(exclude=[np.number]).columns
                        if c != LABEL_COLUMN]

        # statistiche numeriche
        self.numeric_stats: dict = {}
        for c in numeric_cols:
            s = df[c].dropna()
            self.numeric_stats[c] = dict(
                min=float(s.min()), max=float(s.max()),
                mean=round(float(s.mean()), 4), std=round(float(s.std()), 4),
                q25=float(s.quantile(0.25)), q50=float(s.quantile(0.50)),
                q75=float(s.quantile(0.75)),
            )

        # statistiche categoriche
        self.categorical_stats: dict = {}
        for c in categ_cols:
            vc = df[c].value_counts()
            self.categorical_stats[c] = dict(
                n_unique=int(df[c].nunique()),
                top_values=vc.head(10).to_dict(),
            )

        # testi pronti per i prompt
        sc_lines = []
        for c in numeric_cols:
            st = self.numeric_stats[c]
            sc_lines.append(
                f"  - '{c}' (numerico): range [{st['min']}, {st['max']}], "
                f"media={st['mean']}, std={st['std']}"
            )
        for c in categ_cols:
            top = list(self.categorical_stats[c]["top_values"].keys())[:5]
            sc_lines.append(f"  - '{c}' (categorico): valori ammessi → {top}")
        self.schema_constraints = "SCHEMA CONSTRAINTS:\n" + "\n".join(sc_lines)

        dg_lines = []
        for c in numeric_cols:
            st = self.numeric_stats[c]
            if len(numeric_cols) < 15: 
                dg_lines.append(f"  - '{c}': Q25={st['q25']}, mediana={st['q50']}, Q75={st['q75']}")
        self.distribution_summary = "DISTRIBUTION GUIDANCE:\n" + "\n".join(dg_lines)

    def log_analysis(self, minority_class: str):
        log.info("─" * 60)
        log.info(f"ANALISI DATASET  ({self.n_rows} righe × {self.n_cols} colonne)")
        log.info(f"Distribuzione classi : {self.label_distribution}")
        log.info(f"Classe minoritaria   : '{minority_class}'")
        for c, st in self.numeric_stats.items():
            log.info(f"  {c}: min={st['min']} max={st['max']} "
                     f"mean={st['mean']} std={st['std']}")
        for c, st in self.categorical_stats.items():
            log.info(f"  {c}: {st['n_unique']} unici, "
                     f"top={list(st['top_values'].keys())[:3]}")
        log.info("─" * 60)

# ─────────────────────────────────────────────
#  CHECKPOINT
# ─────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _sanitize_model_name(model: str) -> str:
    return re.sub(r"[:/]", "_", model)


def get_strategy_output_path(model: str, dataset_name: str, strategy: AugStrategy) -> str:
    model_dir = os.path.join(AUGMENTED_DIR, _sanitize_model_name(model))
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{dataset_name}_{strategy.value}.csv")


def _ckpt_path(dataset_name: str, model: str, strategy: str, batch_idx: int) -> str:
    key = hashlib.md5(f"{dataset_name}_{model}_{strategy}_{batch_idx}".encode()).hexdigest()[:10]
    return os.path.join(CHECKPOINT_DIR, f"{key}.json")


def save_checkpoint(dataset_name: str, model: str, strategy: str, batch_idx: int, raw: str):
    with open(_ckpt_path(dataset_name, model, strategy, batch_idx), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "model": model, "strategy": strategy,
                   "batch": batch_idx, "ts": datetime.now().isoformat(),
                   "response": raw}, f)


def load_checkpoint(dataset_name: str, model: str, strategy: str, batch_idx: int) -> Optional[str]:
    p = _ckpt_path(dataset_name, model, strategy, batch_idx)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["response"]
    return None

# ─────────────────────────────────────────────
#  PARSER CSV
# ─────────────────────────────────────────────
def parse_csv_from_response(response: str, expected_cols: list) -> pd.DataFrame:
    """Estrae un DataFrame CSV dalla risposta LLM, tolerante al formato."""
    clean = re.sub(r"```[a-z]*\n?", "", response).strip()
    lines = clean.splitlines()

    # trova la riga header
    start = 0
    for i, ln in enumerate(lines):
        if any(c in ln for c in expected_cols):
            start = i
            break

    csv_text = "\n".join(lines[start:])
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(StringIO(csv_text), sep=sep)
            #print(df.head())

            res = len(set(df.columns) & set(expected_cols)) >= max(1, len(expected_cols) // 2)
            #print("\n (set(df.columns)", set(df.columns),  res, "\n")
            #if len(set(df.columns) & set(expected_cols)) >= max(1, len(expected_cols) // 2):
            return df
        except Exception as e:
            print(f"Errore nella lettura del CSV: {e}")
            continue
    return pd.DataFrame()

# ─────────────────────────────────────────────
#  BUILDER PROMPT
# ─────────────────────────────────────────────
def _header(profile: DatasetProfile, examples_csv: str, n: int, minority_class: str) -> str:
    return (
        f"Sei un esperto di data augmentation per dataset tabulari.\n"
        f"Devi generare esattamente {n} nuove righe sintetiche\n"
        f"appartenenti SOLO alla classe minoritaria: '{minority_class}'.\n\n"
        f"{profile.schema_constraints}\n\n"
        f"{profile.distribution_summary}\n\n"
        f"Esempi reali della classe minoritaria (few-shot):\n{examples_csv}\n"
    )


def _footer(cols: list) -> str:
    return (
        f"\nRispondi SOLO con un CSV valido (separatore virgola).\n"
        f"Intestazione esatta prima riga: {','.join(cols)}\n"
        f"Nessun testo aggiuntivo, nessun blocco markdown, nessuna riga vuota.\n"
    )


# ── 1. Schema Constraints ────────────────────
def prompt_schema_constraints(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Rispetta RIGOROSAMENTE i vincoli di schema:
- Nessun valore numerico fuori dai range indicati.
- Solo valori ammessi per le colonne categoriche.
- Nessun campo NULL o mancante.
- Genera esattamente {n} righe.
""" + _footer(p.columns)


# ── 2. Distribution Guidance ─────────────────
def prompt_distribution_guidance(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Segui la distribution guidance per produrre dati realistici:
- I valori numerici devono riflettere la distribuzione osservata (Q25–Q75).
- Introduci variabilità realistica: non copiare gli esempi identici.
- Mantieni le correlazioni tra colonne presenti negli esempi.
Genera {n} righe variegate ma statisticamente coerenti.
""" + _footer(p.columns)


# ── 3. CoT Gerarchica ────────────────────────
def prompt_cot_hierarchical(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Segui questi livelli di ragionamento PRIMA di scrivere il CSV:

LIVELLO 1 — STRUTTURA
  Quali colonne sono numeriche? Quali categoriche?
  Ci sono correlazioni evidenti tra le colonne negli esempi?

LIVELLO 2 — DISTRIBUZIONE
  Per ogni colonna numerica, qual è l'intervallo plausibile?
  Per le categoriche, quali sono i valori dominanti?

LIVELLO 3 — COERENZA INTERNA
  Le colonne si influenzano a vicenda? (es. età↔reddito, ore↔produzione)
  Le nuove righe devono rispettare queste relazioni.

LIVELLO 4 — GENERAZIONE
  Ora genera i {n} esempi applicando tutti i vincoli identificati sopra.
""" + _footer(p.columns)


# ── 4. CoT Self-Consistency ──────────────────
def prompt_cot_self_consistency(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Usa questo processo di auto-verifica prima di produrre il CSV finale:

PASSO 1 — Prima bozza
  Genera {n} righe sintetiche rispettando schema e distribuzione.

PASSO 2 — Verifica statistica
  Le medie e varianze delle righe generate sono coerenti con gli esempi?
  Se no, correggi i valori anomali.

PASSO 3 — Verifica semantica
  Ogni riga ha senso? Ci sono combinazioni impossibili o contraddittorie?
  Se sì, correggile.

PASSO 4 — Output finale
  Scrivi il CSV definitivo solo dopo aver applicato le correzioni.
""" + _footer(p.columns)


# ── 5. CoT ReAct ─────────────────────────────
def prompt_cot_react(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Applica il pattern ReAct (Reasoning + Acting):

[Thought 1] Quali caratteristiche statistiche chiave ha la classe '{mc}'?
[Action 1]  Calcola media e range per ogni colonna numerica dagli esempi.
[Obs 1]     Annota i valori.

[Thought 2] Ci sono vincoli categorici importanti?
[Action 2]  Elenca i valori ammessi per ogni colonna categorica.
[Obs 2]     Annota i vincoli.

[Thought 3] Come si correlano le colonne tra loro?
[Action 3]  Identifica le 2–3 correlazioni più forti dagli esempi.
[Obs 3]     Annota le correlazioni.

[Thought 4] Genero {n} righe rispettando tutto quanto sopra.
[Action 4]  Scrivi il CSV finale.
""" + _footer(p.columns)


# ── 6. CoT Ensemble ──────────────────────────
_PERSPECTIVES = [
    ("STATISTICA",  "Priorità: fedeltà statistica — media, std e distribuzione devono corrispondere."),
    ("SEMANTICA",   "Priorità: plausibilità semantica — ogni riga deve essere realistica nel dominio."),
    ("DIVERSITÀ",   "Priorità: diversità — copri bordi, casi rari, valori estremi ma plausibili."),
]

def prompt_cot_ensemble(p, ex, n, mc, perspective: int = 0, **_):
    name, focus = _PERSPECTIVES[perspective % len(_PERSPECTIVES)]
    return _header(p, ex, n, mc) + f"""
Prospettiva attiva: {name}
{focus}

Ragiona da questa prospettiva:
1. Cosa devi garantire in modo prioritario con questa prospettiva?
2. Quali colonne richiedono più attenzione?
3. Genera {n} righe ottimizzate secondo questa prospettiva.
""" + _footer(p.columns)


# ── 7. CoT Albero Decisionale ────────────────
def prompt_cot_decision_tree(p, ex, n, mc, **_):
    pivot = list(p.numeric_stats.keys())[0] if p.numeric_stats else "feature_principale"
    return _header(p, ex, n, mc) + f"""
Costruisci mentalmente un albero decisionale e usalo per generare:

RADICE → classe = '{mc}'

NODO [{pivot}]
  Ramo A: {pivot} < mediana
    → quali altre feature tendono ad essere basse? quali valori categorici dominano?
  Ramo B: {pivot} >= mediana
    → quali altre feature tendono ad essere alte? quali valori categorici dominano?

FOGLIE → genera esempi seguendo i percorsi.
         Distribuisci equamente tra Ramo A e Ramo B per massimizzare la varietà.

Genera {n} righe in totale (bilanciato tra i rami).
""" + _footer(p.columns)


# ── 8. CoT Confidenza ────────────────────────
def prompt_cot_confidence(p, ex, n, mc, **_):
    return _header(p, ex, n, mc) + f"""
Per ogni riga, applica internamente questo filtro di confidenza:

  Score = 0.4 × (range numerico rispettato)
        + 0.3 × (coerenza categorica)
        + 0.3 × (plausibilità semantica)

  Se score < 0.7 → scarta e rigenera la riga.
  Includi nel CSV finale SOLO righe con score >= 0.7.

Genera almeno {n} righe valide (il numero di righe scartate non conta).
La colonna score NON deve apparire nel CSV — è solo un filtro interno.
""" + _footer(p.columns)


# ── Dispatcher ───────────────────────────────
_PROMPT_FN = {
    AugStrategy.SCHEMA_CONSTRAINTS:   prompt_schema_constraints,
    AugStrategy.DISTRIBUTION_GUIDE:   prompt_distribution_guidance,
    AugStrategy.COT_HIERARCHICAL:     prompt_cot_hierarchical,
    AugStrategy.COT_SELF_CONSISTENCY: prompt_cot_self_consistency,
    AugStrategy.COT_REACT:            prompt_cot_react,
    AugStrategy.COT_ENSEMBLE:         prompt_cot_ensemble,
    AugStrategy.COT_DECISION_TREE:    prompt_cot_decision_tree,
    AugStrategy.COT_CONFIDENCE:       prompt_cot_confidence,
}


def build_prompt(strategy: AugStrategy, profile: DatasetProfile,
                 examples_csv: str, n: int,
                 minority_class: str, perspective: int = 0) -> str:
    return _PROMPT_FN[strategy](
        p=profile, ex=examples_csv, n=n,
        mc=minority_class, perspective=perspective,
    )

# ─────────────────────────────────────────────
#  GENERAZIONE BATCH (sequenziale)
# ─────────────────────────────────────────────
def generate_batch(
    dataset_name: str,
    few_shot: pd.DataFrame,
    profile: DatasetProfile,
    minority_class: str,
    batch_idx: int,
    strategy: AugStrategy,
    model: str = PRIMARY_MODEL,
    n: int = N_GEN_AT_TIME,
    perspective: int = 0,
) -> pd.DataFrame:
    """Genera un batch di n righe con la strategia indicata, in modo sequenziale."""

    # ── usa cache se esiste ──────────────────
    cached = load_checkpoint(dataset_name, model, strategy.value, batch_idx)
    if cached:
        log.info(f"    [cache] batch {batch_idx} già presente")
        raw = cached
    else:
        examples_csv = few_shot.to_csv(index=False)
        prompt = build_prompt(strategy, profile, examples_csv, n,
                              minority_class, perspective)

        log.info(f"    [LLM] model={model}  strategy={strategy.value}  batch={batch_idx}  n={n}")
        raw = ollama_generate(prompt, model=model)

        if not raw:
            log.warning(f"    [LLM] risposta vuota — batch {batch_idx} saltato")
            return pd.DataFrame()

        # salva risposta grezza
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = f"{dataset_name}_{_sanitize_model_name(model)}_{strategy.value}_{batch_idx}_{batch_idx + n}.txt"
        with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(raw)
        save_checkpoint(dataset_name, model, strategy.value, batch_idx, raw)

    #print("raw\n\n", raw)
    #print("profile.columns\n\n", profile.columns)
    df_new = parse_csv_from_response(raw, profile.columns)
    if df_new.empty:
        log.warning(f"    [parse] CSV non estraibile dal batch {batch_idx}")
    else:
        # forza la classe corretta su tutte le righe generate
        df_new[LABEL_COLUMN] = minority_class
        log.info(f"    [parse] {len(df_new)} righe estratte")
    return df_new

# ─────────────────────────────────────────────
#  AUGMENT DATASET
# ─────────────────────────────────────────────
def augment_dataset(file_path: str, strategies: list, model: str) -> None:
    """
    Legge il CSV, aumenta SOLO la classe minoritaria fino a pareggiare
    la maggioritaria. Per ogni strategia salva un file separato in
    AUGMENTED_DIR/{model}/{dataset}_{strategy}.csv, aggiornato ad ogni batch.
    """
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    log.info(f"\n{'=' * 60}")
    log.info(f"Dataset: {dataset_name}  |  Model: {model}")

    df = pd.read_csv(file_path)

    if LABEL_COLUMN not in df.columns:
        log.error(f"  Colonna '{LABEL_COLUMN}' non trovata — skip.")
        return

    profile = DatasetProfile(df)
    counts  = df[LABEL_COLUMN].value_counts()

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    diff = int(counts[majority_class] - counts[minority_class])

    profile.log_analysis(minority_class)
    log.info(f"  Da generare (diff): {diff} righe per '{minority_class}'")

    if diff <= 0:
        log.info("  Dataset già bilanciato — skip.")
        return

    few_shot = df[df[LABEL_COLUMN] == minority_class].sample(
        min(N_FEW_SHOT, int(counts[minority_class])), random_state=42
    )

    # ── ciclo su strategie (sequenziale) ────
    for strategy in strategies:
        out_path = get_strategy_output_path(model, dataset_name, strategy)
        log.info(f"\n  ── Strategia: {strategy.value}  |  Model: {model} ──")

        # resume: conta righe già generate per questa strategia
        if os.path.exists(out_path):
            existing_aug = pd.read_csv(out_path)
            already_generated = max(0, len(existing_aug) - len(df))
            generated_rows = existing_aug.iloc[len(df):].reset_index(drop=True)
            if already_generated >= diff:
                log.info(f"    Strategia già completata ({already_generated} righe) — skip")
                continue
            #continue #Aggiunto da stefano
            log.info(f"    Resume: {already_generated} righe già presenti")
        else:
            already_generated = 0
            generated_rows = pd.DataFrame()

        perspective = 0
        generated_count = already_generated

        for batch_start in range(already_generated, diff + N_GEN_AT_TIME, N_GEN_AT_TIME):
            if generated_count >= diff:
                break

            batch_df = generate_batch(
                dataset_name, few_shot, profile, minority_class,
                batch_start, strategy, model,
                n=N_GEN_AT_TIME, perspective=perspective,
            )

            if not batch_df.empty:
                generated_rows = pd.concat([generated_rows, batch_df], ignore_index=True)
                generated_count += len(batch_df)

                # salva incrementalmente dopo ogni batch con dati
                rows_to_save = generated_rows.head(diff)
                aug_df = pd.concat([df, rows_to_save], ignore_index=True)
                aug_df.to_csv(out_path, index=False)
                new_counts = aug_df[LABEL_COLUMN].value_counts().to_dict()
                log.info(f"    [save] {out_path} — {len(aug_df)} righe  |  dist: {new_counts}")

            if strategy == AugStrategy.COT_ENSEMBLE:
                perspective = (perspective + 1) % len(_PERSPECTIVES)

        log.info(f"  Strategia {strategy.value}: {generated_count} righe generate totali.")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    for d in [OUTPUT_DIR, AUGMENTED_DIR, CHECKPOINT_DIR, DATASET_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Pre-verifica disponibilità modelli Ollama Cloud ──
    log.info("\n" + "=" * 60)
    log.info("PRE-VERIFICA DISPONIBILITÀ MODELLI OLLAMA CLOUD")
    log.info("=" * 60)
    available_models = []
    for model in MODELS:
        if check_model_availability(model):
            available_models.append(model)
        else:
            log.warning(f"  [SKIP] Il modello '{model}' non è disponibile e verrà saltato.")

    if not available_models:
        log.critical("Nessuno dei modelli specificati è disponibile o tutte le API key sono esaurite.")
        return

    log.info(f"\nModelli disponibili per l'esecuzione ({len(available_models)}/{len(MODELS)}): {available_models}")

    # ── Dataset da aumentare ──
    # Impostare DATASET_FILTER = "nome_dataset" per limitare la ricerca a un file specifico, o None per tutti i dataset.
    DATASET_FILTER = None

    csv_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.endswith(".csv") and (DATASET_FILTER is None or DATASET_FILTER in f)
    ])

    if not csv_files:
        log.warning(f"Nessun CSV trovato in '{DATASET_DIR}'.")
        return

    log.info(f"File dataset trovati ({len(csv_files)}): {csv_files}")

    selected_strategies = [
        AugStrategy.SCHEMA_CONSTRAINTS,
        AugStrategy.DISTRIBUTION_GUIDE,
        AugStrategy.COT_HIERARCHICAL,
        AugStrategy.COT_SELF_CONSISTENCY,
        AugStrategy.COT_REACT,
        AugStrategy.COT_ENSEMBLE,
        AugStrategy.COT_DECISION_TREE,
        AugStrategy.COT_CONFIDENCE,
    ]

    for model in available_models:
        if key_manager.client is None:
            log.critical("[MAIN] Tutte le chiavi API sono state esaurite. Interruzione pipeline.")
            break

        log.info(f"\n{'#' * 60}")
        log.info(f"MODELLO ATTIVO: {model}")
        for fname in tqdm(csv_files, desc=f"Dataset ({model})"):
            if key_manager.client is None:
                log.critical("[MAIN] Tutte le chiavi API sono state esaurite. Interruzione pipeline.")
                break

            file_path = os.path.join(DATASET_DIR, fname)

            # ── retry con auto-restart su errore ──
            for attempt in range(1, MAX_RETRIES + 1):
                log.info(f"\n[MAIN] {fname} | {model} — tentativo {attempt}/{MAX_RETRIES}")
                try:
                    augment_dataset(file_path, selected_strategies, model)
                    break  # successo → prossimo file
                except Exception as e:
                    log.error(f"[MAIN] errore durante {fname} ({model}): {e}")
                    if attempt < MAX_RETRIES:
                        wait = RETRY_DELAY * attempt
                        log.info(f"[MAIN] riavvio tra {wait}s…")
                        time.sleep(wait)
                    else:
                        log.critical(f"[MAIN] {fname} ({model}) fallito dopo {MAX_RETRIES} tentativi")

    log.info("\n✅ Pipeline completata.")


if __name__ == "__main__":
    main()