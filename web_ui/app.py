#!/usr/bin/env python3
"""
app.py
======
Flask Web Application for Unified Data Augmentation and Metric Evaluation.
Provides REST APIs for:
  - Dataset & Algorithm discovery
  - Constraint file exploration (Denial Constraints for Kamino / SyDC)
  - Asynchronous execution of main.py and evaluate_metrics.py
  - Real-time log streaming and process cancellation
  - Visual analytics, tables, and Plotly interactive chart payloads
  - Modalita' serie temporali e grafi (run_modality.py): /api/modalities, /api/modality/<mod>/...
  - Blueprint opzionali: assistente (/assistant) e macchine (/machines)
"""

from __future__ import annotations

import os
import re
import sys
import glob
import json
import time
import importlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import pandas as pd

# Path setup to import local modules & access root benchmark scripts
WEB_UI_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_UI_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(WEB_UI_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_UI_DIR))

from web_ui.task_manager import TaskManager
from web_ui.analytics_engine import (
    get_available_evaluated_datasets,
    compute_analytics_payload,
)

# Initialize Flask & CORS
app = Flask(
    __name__,
    template_folder=str(WEB_UI_DIR / "templates"),
    static_folder=str(WEB_UI_DIR / "static"),
)
CORS(app)

task_manager = TaskManager()
log = logging.getLogger("web_ui")

# ---------------------------------------------------------------------------
# Blueprint scritti da altri moduli: se mancano (o non si importano) l'app parte lo stesso
# e il link di navigazione risulta disattivato.
# ---------------------------------------------------------------------------
BLUEPRINTS = {
    "assistant": {"module": "web_ui.assistant_bp", "path": "/assistant", "label": "Assistant"},
    "machines": {"module": "web_ui.machines_bp", "path": "/machines", "label": "Machines"},
    "pages": {"module": "web_ui.pages_bp", "path": "/catalog", "label": "Pages"},      # /copilot /guided /control /catalog /api/catalog_info
    "designs": {"module": "web_ui.designs_bp", "path": "/designs", "label": "Layout"},   # vecchi indirizzi /design/N: rimandano alle sezioni
}
for _name, _info in BLUEPRINTS.items():
    _info["available"] = False
    _info["error"] = None
    try:
        _mod = importlib.import_module(_info["module"])
        app.register_blueprint(_mod.bp)
        _info["available"] = True
    except Exception as _e:   # modulo non ancora scritto, dipendenza mancante o errore: l app parte lo stesso
        _info["error"] = f"ImportError: {_e}"
        log.warning("blueprint %s non registrato: %s", _name, _info["error"])

# Barra di navigazione comune (template _sitenav.html). Le sezioni "pages" sono parte dell'app; "Macchine" e "Assistente" dipendono dal blueprint.
SITE_NAV = [
    ("copilot", "/copilot", "Copilot", "pages"),
    ("guided", "/guided", "Guided path", "pages"),
    ("catalog", "/catalog", "Catalog", "pages"),
    ("evidence", "/evidence", "Evidence", "pages"),
    ("challenge", "/challenge", "Challenge", "pages"),
    ("control", "/control", "Control room", "pages"),
    ("machines", "/machines", "Machines", "machines"),
]
SITE_NAV_MORE = [
    ("assistant", "/assistant", "Assistant (simple chat)", "assistant"),
    ("classic", "/classic", "Classic dashboard", None),
]


def _nav_items(items):
    out = []
    for key, path, label, bp_name in items:
        info = BLUEPRINTS.get(bp_name) if bp_name else None
        out.append({"key": key, "path": path, "label": label,
                    "available": True if info is None else bool(info.get("available")),
                    "error": None if info is None else info.get("error")})
    return out


@app.context_processor
def _inject_site_nav():
    return {"site_nav": _nav_items(SITE_NAV), "site_nav_more": _nav_items(SITE_NAV_MORE)}


DATASETS_DIR = ROOT_DIR / "datasets"
AUGMENTED_DIR = ROOT_DIR / "augmented_datasets"
RESULTS_DIR = ROOT_DIR / "runtime_results"
EVAL_DIR = ROOT_DIR / "evaluation_metrics"
# Cartelle dedicate alle esecuzioni lanciate da qui (e dall'assistente): non sovrascrivono augmented_datasets/,
# runtime_results/ ne' evaluation_metrics/ del benchmark, e il loro checkpoint e' separato.
AUGMENTED_UI_DIR = ROOT_DIR / "augmented_ui"
RESULTS_UI_DIR = ROOT_DIR / "runtime_results_ui"
EVAL_UI_DIR = ROOT_DIR / "evaluation_metrics_ui"


def _tabular_python() -> str:
    """Interprete di envs/tabular per main.py, evaluate_metrics.py, checkpoint_generator.py (config.ENV_PYTHONS)."""
    try:
        import config as _root_config
        return _root_config.ENV_PYTHONS.get("tabular") or sys.executable
    except Exception:  # noqa: BLE001
        return sys.executable


# Modalita' aggiuntive (serie temporali, grafi): cartelle dati e risultati scritti da run_modality.py
MODALITIES = {
    "tabular": {"name": "Tabular", "data_dir": DATASETS_DIR, "results_dir": RESULTS_DIR},
    # results_dir: CSV del benchmark e delle prove a mano; ui_results_dir: esecuzioni lanciate dalla UI e dall'assistente
    # (run_modality.py --out-dir/--results-dir)
    "ts": {"name": "Time series", "data_dir": ROOT_DIR / "datasets_ts", "results_dir": ROOT_DIR / "runtime_results_ts",
           "ui_results_dir": ROOT_DIR / "runtime_results_ui_ts", "ui_out_dir": ROOT_DIR / "augmented_ui_ts",
           "exts": (".npz",)},
    "graph": {"name": "Graphs", "data_dir": ROOT_DIR / "datasets_graph", "results_dir": ROOT_DIR / "runtime_results_graph",
              "ui_results_dir": ROOT_DIR / "runtime_results_ui_graph", "ui_out_dir": ROOT_DIR / "augmented_ui_graph",
              "exts": (".pt",)},
}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


# ---------------------------------------------------------------------------
# Algorithm Metadata Catalog
# ---------------------------------------------------------------------------
# I 20 id coincidono con config.CORE_ALGORITHMS (catalogo tabellare del paper). SyDC (sweep di Denial Constraints) non e'
# un algoritmo del catalogo: la dashboard classica lo lancia dalla modalita' di valutazione "SyDC only", non da qui.
# Lo stato (executable/described) non sta qui: viene dal KB finale (assistant/kb/benchmark.csv), vedi _tabular_algorithms().
ALGORITHM_CATALOG = [
    {
        "id": "smote",
        "name": "SMOTE",
        "category": "Traditional & Fair",
        "desc": "Synthetic Minority Over-sampling Technique via k-NN interpolation.",
    },
    {
        "id": "fair-smote",
        "name": "Fair-SMOTE",
        "category": "Traditional & Fair",
        "desc": "Fairness-aware SMOTE rebalancing sensitive attributes.",
    },
    {
        "id": "smote-cdnn",
        "name": "SMOTE-CDNN",
        "category": "Traditional & Fair",
        "desc": "Combined SMOTE with Deep Neural Network distribution fitting.",
    },
    {
        "id": "prefair",
        "name": "Prefair",
        "category": "Traditional & Fair",
        "desc": "Pre-processing fair data augmentation framework.",
    },
    {
        "id": "goggle",
        "name": "GOGGLE",
        "category": "GAN & VAE",
        "desc": "Generative model (ICLR 2023) that learns the relational structure between variables.",
    },
    {
        "id": "castgan",
        "name": "CasTGAN",
        "category": "GAN & VAE",
        "desc": "Cascaded Generative Adversarial Network for tabular records.",
    },
    {
        "id": "tabfairgan",
        "name": "TabFairGAN",
        "category": "GAN & VAE",
        "desc": "Fair tabular GAN with conditional discriminator penalty.",
    },
    {
        "id": "tvae",
        "name": "TVAE",
        "category": "GAN & VAE",
        "desc": "Tabular Variational Autoencoder with GMM mode normalization.",
    },
    {
        "id": "ctabgan",
        "name": "CTAB-GAN",
        "category": "GAN & VAE",
        "desc": "Conditional Table GAN for mixed continuous and categorical data.",
    },
    {
        "id": "ctabgan-plus",
        "name": "CTAB-GAN+",
        "category": "GAN & VAE",
        "desc": "Enhanced CTAB-GAN with downstream utility and privacy preservation.",
    },
    {
        "id": "ddpm",
        "name": "Tab-DDPM",
        "category": "Diffusion & Advanced",
        "desc": "Denoising Diffusion Probabilistic Model for tabular domains.",
    },
    {
        "id": "tabsyn",
        "name": "TabSyn",
        "category": "Diffusion & Advanced",
        "desc": "Latent tabular diffusion model with VAE embedding.",
    },
    {
        "id": "tabdiff",
        "name": "TabDiff",
        "category": "Diffusion & Advanced",
        "desc": "Fast deterministic score-based tabular diffusion model.",
    },
    {
        "id": "octree",
        "name": "OCTree",
        "category": "Diffusion & Advanced",
        "desc": "Hierarchical octree density estimation and sampling.",
    },
    {
        "id": "harmonic",
        "name": "HARMONIC",
        "category": "Diffusion & Advanced",
        "desc": "Harmonic spatial clustering and distribution sampling.",
    },
    {
        "id": "synthetic-llm",
        "name": "Synthetic-LLM",
        "category": "Diffusion & Advanced",
        "desc": "Few-shot exemplar prompting and generative tabular interpolation.",
    },
    {
        "id": "cuts",
        "name": "CUTS",
        "category": "Diffusion & Advanced",
        "desc": "Conditional Universal Tabular Synthesizer.",
    },
    {
        "id": "tabular-argn",
        "name": "Tabular-ARGN",
        "category": "Diffusion & Advanced",
        "desc": "Autoregressive generative network for tabular synthesis.",
    },
    {
        "id": "kamino",
        "name": "Kamino (VLDB 2021)",
        "category": "Constraint-Aware",
        "desc": "Synthesizer with Denial Constraints (DC), Gibbs energy & MCMC repair.",
        "needs_dc": True,
    },
    {
        "id": "syrfd",
        "name": "SyRFDRes",
        "category": "Constraint-Aware",
        "desc": "Relaxed Functional Dependency (RFD) constraint-aware synthesizer.",
        "needs_rfd": True,
    },
]


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Ingresso: quattro scelte (Copilota, Percorso guidato, Catalogo, Cabina di regia)."""
    return render_template("home.html")


@app.route("/classic")
def index():
    """Dashboard originale (runner, valutazione, analytics, runtime, checkpoint, modalita' ts/grafi): invariata."""
    return render_template("index.html", nav=BLUEPRINTS)


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """Return hardware status, GPU info, active task status."""
    gpu_available = False
    device_name = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_available = True
            device_name = "Apple Silicon MPS"
    except Exception:
        pass

    active_task = task_manager.get_active_task()

    return jsonify({
        "status": "online",
        "python_executable": sys.executable,
        "gpu_available": gpu_available,
        "device_name": device_name,
        "active_task": active_task.to_dict() if active_task else None,
    })


def _tabular_datasets(include_user: bool = False):
    """List datasets in datasets/ with rows and columns counts. include_user: aggiunge i CSV caricati dall'utente
    (datasets_user/, marcati user=True); la dashboard classica non li vede perche' lancia main.py senza --datasets-dir."""
    datasets = []
    dirs = [(DATASETS_DIR, False)]
    if include_user:
        try:
            from assistant import config as _acfg
            ud = _acfg.user_dir("tabular")
            if ud is not None:
                dirs.append((ud, True))
        except Exception:  # noqa: BLE001
            pass
    seen = set()
    for base, is_user in dirs:
        if not base.is_dir():
            continue
        for csv_file in sorted(base.glob("*.csv")):
            if csv_file.stem in seen:
                continue
            seen.add(csv_file.stem)
            item = {"stem": csv_file.stem, "filename": csv_file.name, "rows": 0, "columns": 0, "size_kb": 0}
            if is_user:
                item["user"] = True
            try:
                # Fast row count reading header + line count
                with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                    header = f.readline().strip().split(",")
                    n_lines = sum(1 for _ in f)
                item.update({"rows": n_lines, "columns": len(header), "size_kb": round(csv_file.stat().st_size / 1024, 1)})
            except Exception:
                pass
            datasets.append(item)
    return datasets


@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    return jsonify(_tabular_datasets())


@app.route("/api/algorithms", methods=["GET"])
def list_algorithms():
    """Algoritmi tabellari avviabili dalla dashboard classica: solo quelli con almeno una esecuzione completata nel KB
    (i 'described' si vedono nel Catalogo, non si offrono per l'esecuzione)."""
    return jsonify([a for a in _tabular_algorithms() if a["runnable"]])


# ---------------------------------------------------------------------------
# Modalita' serie temporali e grafi
# ---------------------------------------------------------------------------
def _modality_specs(mod: str):
    """Specifiche del catalogo di una modalita' (ts/graph). Cataloghi parziali o mancanti non sono un errore.

    Restituisce (lista di dict, messaggio di errore o None).
    """
    try:
        import modalities
        specs = modalities.list_specs(mod)
    except Exception as e:  # catalogo in scrittura da altri agenti: si riporta, non si nasconde
        return [], f"{type(e).__name__}: {e}"
    out = []
    for sp in specs:
        d = sp.to_dict()
        d["runnable"] = sp.status != "described" and bool(sp.worker)
        out.append(d)
    return out, None


def _tabular_algorithms():
    """Catalogo tabellare = config.CORE_ALGORITHMS. Stato: executable se il KB finale ha almeno una esecuzione
    SUCCESS/FALLBACK per la modalita' tabular, altrimenti described (e non avviabile). KB assente: wired (avviabile)."""
    try:
        from config import CORE_ALGORITHMS
    except Exception:
        CORE_ALGORITHMS = [a["id"] for a in ALGORITHM_CATALOG]
    try:
        from assistant.kb import store
        done = store.completed_algos("tabular")
        canon = store.canon_algo
    except Exception:  # noqa: BLE001
        done, canon = None, str
    out = []
    for a in ALGORITHM_CATALOG:
        if a["id"] not in CORE_ALGORITHMS:
            continue
        d = dict(a)
        d["key"] = a["id"]
        d["strategy"] = a.get("category", "")
        d["status"] = "wired" if done is None else "executable" if canon(a["id"]) in done else "described"
        d["runnable"] = d["status"] != "described"
        out.append(d)
    return out


def _modality_datasets(mod: str):
    cfg = MODALITIES[mod]
    out = []
    seen = set()
    try:
        from assistant import config as _acfg
        user_d = _acfg.user_dir(mod)
    except Exception:  # noqa: BLE001
        user_d = None
    for d, is_user in ((cfg["data_dir"], False), (user_d, True)):
        if d is None or not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in cfg["exts"] and f.stat().st_size > 0 and f.stem not in seen:
                seen.add(f.stem)
                item = {"stem": f.stem, "filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
                if is_user:
                    item["user"] = True
                out.append(item)
    return out


def _check_modality(mod: str):
    if mod not in MODALITIES:
        return jsonify({"error": f"unknown modality: {mod}", "known": list(MODALITIES)}), 404
    return None


@app.route("/api/modalities", methods=["GET"])
def list_modalities():
    """Per ogni modalita': conteggi degli algoritmi per stato (executable / wired / described) e dei dataset."""
    result = []
    for mod, cfg in MODALITIES.items():
        if mod == "tabular":
            algos, err = _tabular_algorithms(), None
            n_ds = len({f.stem for f in cfg["data_dir"].glob("*.csv")}) if cfg["data_dir"].is_dir() else 0
            try:
                from assistant import config as _acfg
                ud = _acfg.user_dir("tabular")
                if ud is not None and ud.is_dir():
                    n_ds += len({f.stem for f in ud.glob("*.csv")} - {f.stem for f in cfg["data_dir"].glob("*.csv")})
            except Exception:  # noqa: BLE001
                pass
        else:
            algos, err = _modality_specs(mod)
            n_ds = len(_modality_datasets(mod))
        counts = {"executable": 0, "wired": 0, "described": 0}
        for a in algos:
            counts[a["status"]] = counts.get(a["status"], 0) + 1
        counts["total"] = len(algos)
        result.append({"id": mod, "name": cfg["name"], "counts": counts, "n_datasets": n_ds, "catalog_error": err})
    return jsonify(result)


@app.route("/api/modality/<mod>/algorithms", methods=["GET"])
def modality_algorithms(mod: str):
    bad = _check_modality(mod)
    if bad:
        return bad
    if mod == "tabular":
        return jsonify({"modality": mod, "algorithms": _tabular_algorithms(), "catalog_error": None})
    algos, err = _modality_specs(mod)
    return jsonify({"modality": mod, "algorithms": algos, "catalog_error": err})


@app.route("/api/modality/<mod>/datasets", methods=["GET"])
def modality_datasets(mod: str):
    bad = _check_modality(mod)
    if bad:
        return bad
    if mod == "tabular":
        return jsonify({"modality": mod, "datasets": _tabular_datasets(include_user=True)})
    return jsonify({"modality": mod, "datasets": _modality_datasets(mod)})


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@app.route("/api/run_modality", methods=["POST"])
def run_modality():
    """Lancia run_modality.py come task asincrono (serie temporali / grafi).

    Payload: {modality, algorithms:[...], datasets:[...], epochs, n_samples?, device?, seed?, confirmed?}
    Con piu' di un algoritmo (o piu' di una combinazione) serve confirmed=true: la conferma si chiede all'utente
    lato client e viene ricontrollata qui.
    """
    data = request.json or {}
    mod = data.get("modality")
    if mod == "tabular":
        return jsonify({"success": False, "error": "For tabular data use /api/run_augmentation."}), 400
    if mod not in MODALITIES:
        return jsonify({"success": False, "error": f"unknown modality: {mod}"}), 400

    active = task_manager.get_active_task()
    if active:
        return jsonify({"success": False, "error": f"A task ({active.task_id}) is already running."}), 400

    algos = data.get("algorithms") or []
    datasets = data.get("datasets") or []
    if not algos or not datasets:
        return jsonify({"success": False, "error": "Select at least one algorithm and one dataset."}), 400
    if not all(isinstance(x, str) and SAFE_NAME.match(x) for x in list(algos) + list(datasets)):
        return jsonify({"success": False, "error": "Invalid names."}), 400

    specs, err = _modality_specs(mod)
    by_key = {sp["key"]: sp for sp in specs}
    unknown = [a for a in algos if a not in by_key]
    if unknown:
        return jsonify({"success": False, "error": f"Algorithms not in the catalog: {unknown}"}), 400
    not_runnable = [a for a in algos if not by_key[a]["runnable"]]
    if not_runnable:
        return jsonify({"success": False, "error": f"Described-only algorithms (cannot be started): {not_runnable}"}), 400
    known_ds = {d["stem"] for d in _modality_datasets(mod)}
    missing = [d for d in datasets if d not in known_ds]
    if missing:
        return jsonify({"success": False, "error": f"Datasets not found in {MODALITIES[mod]['data_dir'].name}/: {missing}"}), 400

    if len(algos) > 1 and not data.get("confirmed"):
        return jsonify({"success": False, "needs_confirmation": True,
                        "error": f"More than one algorithm ({len(algos)}) x {len(datasets)} datasets: confirmation required."}), 409

    try:
        epochs = int(data.get("epochs", 50))
        seed = int(data.get("seed", 0))
        n_samples = data.get("n_samples")
        n_samples = int(n_samples) if n_samples not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "epochs, seed and n_samples must be integers."}), 400
    if not 1 <= epochs <= 5000:
        return jsonify({"success": False, "error": "epochs must be between 1 and 5000."}), 400
    device = data.get("device", "auto")
    if device == "auto":
        device = _default_device()
    if device not in ("cpu", "cuda"):
        return jsonify({"success": False, "error": "device must be auto, cpu or cuda."}), 400

    cmd = [sys.executable, str(ROOT_DIR / "run_modality.py"), "--modality", mod,
           "--algorithms", *algos, "--datasets", *datasets,
           "--epochs", str(epochs), "--device", device, "--seed", str(seed),
           "--out-dir", str(MODALITIES[mod]["ui_out_dir"]), "--results-dir", str(MODALITIES[mod]["ui_results_dir"])]
    if n_samples:
        cmd += ["--n_samples", str(n_samples)]
    if data.get("no_eval"):
        cmd.append("--no_eval")

    task = task_manager.create_task("modality_run", cmd, ROOT_DIR)
    task_manager.start_task(task)
    return jsonify({"success": True, "task_id": task.task_id, "command": " ".join(cmd)})


def _kb_records(mod: str) -> list:
    """Righe della knowledge base del benchmark (assistant/kb/benchmark.csv) per la modalita', con le chiavi del catalogo.
    Servono alle schede del Catalogo per contare le esecuzioni misurate; file = 'benchmark_kb'."""
    try:
        from assistant.kb import store as _kb
        kb = _kb.load_kb()
        if kb is None:
            return []
        df = kb.rows(mod)
        algos = _tabular_algorithms() if mod == "tabular" else _modality_specs(mod)[0]
        keys = {_kb.canon_algo(a.get("key") or a.get("id")): (a.get("key") or a.get("id")) for a in algos}
        out = []
        for r in df.to_dict(orient="records"):
            k = keys.get(_kb.canon_algo(str(r.get("algorithm", ""))))
            if not k:
                continue
            ds = str(r.get("dataset") or "")
            rec = {"file": "benchmark_kb", "algorithm": k, "dataset": ds, "seed": r.get("seed"),
                   "status": r.get("status"), "time_s": r.get("time_s"), "mem_mb": r.get("mem_mb"), "host": r.get("host")}
            if mod == "tabular":
                # stessi nomi dei CSV tabulari della UI: dataset con suffisso _min (se il file esiste) e colonne total_*
                if not ds.endswith("_min") and (DATASETS_DIR / f"{ds}_min.csv").exists():
                    rec["dataset"] = f"{ds}_min"
                rec["total_time_s"], rec["total_mem_peak_mb"] = r.get("time_s"), r.get("mem_mb")
            out.append(rec)
        # NaN (misure assenti) non e' JSON valido: si scrive null, altrimenti la pagina mostra "Partial data ... is not valid JSON"
        out = [{k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()} for rec in out]
        return json.loads(json.dumps(out, default=str)) if out else []
    except Exception as e:  # noqa: BLE001 — KB assente o illeggibile: le schede restano sui CSV
        logging.getLogger(__name__).warning("KB records unavailable for %s: %s", mod, e)
        return []


@app.route("/api/modality/<mod>/results", methods=["GET"])
def modality_results(mod: str):
    """CSV dei risultati. Query: ?file=<nome.csv> (default: il piu' recente), ?all=1 (tutti, con colonna 'file')."""
    bad = _check_modality(mod)
    if bad:
        return bad
    if mod == "tabular":
        if not request.args.get("all"):
            return get_runtime_results()
        data = get_runtime_results().get_json()
        if isinstance(data, list):  # nessun CSV: get_runtime_results restituisce []
            data = {"filename": None, "source": None, "records": []}
        data["records"] = list(data.get("records") or []) + _kb_records(mod)
        return jsonify(data)
    # CSV del benchmark/prove a mano e delle esecuzioni lanciate dalla UI (cartelle dedicate): si leggono entrambe
    rdirs = [MODALITIES[mod]["results_dir"], MODALITIES[mod]["ui_results_dir"]]
    files = sorted([f for d in rdirs if d.is_dir() for f in d.glob("*.csv")], key=lambda f: f.stat().st_mtime, reverse=True)
    names = [f.name for f in files]
    by_name = {f.name: f for f in files}
    if not files:
        return jsonify({"modality": mod, "files": [], "filename": None,
                        "records": _kb_records(mod) if request.args.get("all") else []})
    try:
        if request.args.get("all"):
            frames = []
            for f in files:
                d = pd.read_csv(f)
                d.insert(0, "file", f.name)
                frames.append(d)
            df = pd.concat(frames, ignore_index=True)
            fname = "all"
            extra = _kb_records(mod)
        else:
            fname = request.args.get("file") or names[0]
            if not SAFE_NAME.match(fname) or fname not in by_name:
                return jsonify({"error": f"file not found: {fname}", "files": names, "records": []}), 404
            df = pd.read_csv(by_name[fname])
        records = json.loads(df.to_json(orient="records")) + (extra if fname == "all" else [])
        return jsonify({"modality": mod, "files": names, "filename": fname, "records": records})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}", "files": names, "records": []}), 500


@app.route("/api/dc_files", methods=["GET"])
def list_dc_files():
    """Discover available Denial Constraint (.out, .ic, .txt) files."""
    dc_files = []
    paths_to_check = [
        ROOT_DIR / "Algorithms" / "kamino-master" / "testdata",
        DATASETS_DIR,
    ]

    for p in paths_to_check:
        if not p.exists():
            continue
        for ext in ("*.out", "*.ic", "*_dc.txt"):
            for f in p.rglob(ext):
                if f.is_file() and f.stat().st_size > 0:
                    dc_files.append({
                        "filename": f.name,
                        "path": str(f.resolve()),
                        "size_bytes": f.stat().st_size,
                    })

    # Sort and remove duplicates by path
    seen = set()
    unique_files = []
    for item in dc_files:
        if item["path"] not in seen:
            seen.add(item["path"])
            unique_files.append(item)

    return jsonify(sorted(unique_files, key=lambda x: x["filename"]))


# ---------------------------------------------------------------------------
# Checkpoint Discovery, Upload & Generation
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/api/checkpoints", methods=["GET"])
def list_checkpoints():
    """Discover all saved checkpoints in Saved/, checkpoint/, and Algorithms/."""
    from checkpoint_generator import detect_checkpoint_algorithm
    extensions = (".pth", ".pt", ".ckpt", ".pkl", ".joblib", ".bin")
    scan_paths = [
        ROOT_DIR / "checkpoint",
        ROOT_DIR / "Saved" / "checkpoints",
        ROOT_DIR / "Saved" / "trained_models",
        ROOT_DIR / "Saved" / "aux_learners",
        ROOT_DIR / "Algorithms" / "tabsyn-main" / "tabsyn" / "ckpt",
        ROOT_DIR / "Algorithms" / "tabsyn-main" / "tabsyn" / "vae" / "ckpt",
        ROOT_DIR / "Algorithms" / "TabDiff" / "tabdiff" / "ckpt",
        ROOT_DIR / "Algorithms" / "tab-ddpm-main" / "exp_imbalanced",
        ROOT_DIR / "Algorithms" / "tab-ddpm-main_esecuzione_full" / "exp_imbalanced",
        ROOT_DIR / "Algorithms" / "GOGGLE-main" / "models",
    ]

    checkpoints = []
    seen = set()

    for p in scan_paths:
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.suffix in extensions:
                p_str = str(f)
                if "clean_env" in p_str or "site-packages" in p_str or ".ipynb_checkpoints" in p_str:
                    continue
                if p_str in seen:
                    continue
                seen.add(p_str)

                try:
                    rel_p = str(f.relative_to(ROOT_DIR))
                except Exception:
                    rel_p = f.name

                # Infer dataset from path
                dataset_hint = ""
                for part in reversed(f.parent.parts):
                    if part not in ("ckpt", "checkpoints", "models", "wb_0", "tabsyn", "vae", "exp_imbalanced", "Saved"):
                        dataset_hint = part.replace("run_", "")
                        break

                algo = detect_checkpoint_algorithm(f)
                stat = f.stat()
                mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))

                checkpoints.append({
                    "filename": f.name,
                    "rel_path": rel_p,
                    "full_path": str(f.resolve()),
                    "algorithm": algo,
                    "dataset_hint": dataset_hint,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "mtime": mtime_str,
                })

    return jsonify(sorted(checkpoints, key=lambda x: x["rel_path"]))


@app.route("/api/upload_checkpoint", methods=["POST"])
def upload_checkpoint():
    """Upload a custom model checkpoint file to checkpoint/ directory."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400
    
    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty file name."}), 400

    filename = file.filename.replace(" ", "_")
    dest_path = CHECKPOINT_DIR / filename
    file.save(str(dest_path))

    from checkpoint_generator import detect_checkpoint_algorithm
    algo = detect_checkpoint_algorithm(dest_path)
    stat = dest_path.stat()

    return jsonify({
        "success": True,
        "checkpoint": {
            "filename": filename,
            "rel_path": str(dest_path.relative_to(ROOT_DIR)),
            "full_path": str(dest_path.resolve()),
            "algorithm": algo,
            "dataset_hint": "",
            "size_kb": round(stat.st_size / 1024, 1),
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
        }
    })


@app.route("/api/generate_from_checkpoint", methods=["POST"])
def generate_from_checkpoint():
    """Launch checkpoint_generator.py as an asynchronous background task."""
    active = task_manager.get_active_task()
    if active:
        return jsonify({"success": False, "error": f"A task ({active.task_id}) is already running."}), 400

    data = request.json or {}
    ckpt_path = data.get("checkpoint_path")
    if not ckpt_path or not Path(ckpt_path).exists():
        return jsonify({"success": False, "error": f"Checkpoint file not found: {ckpt_path}"}), 400

    dataset = data.get("dataset")
    if not dataset:
        return jsonify({"success": False, "error": "Select a reference dataset."}), 400

    dataset_file = DATASETS_DIR / f"{dataset}.csv"
    if not dataset_file.exists():
        matches = list(DATASETS_DIR.glob(f"*{dataset}*.csv"))
        if matches:
            dataset_file = matches[0]
        else:
            return jsonify({"success": False, "error": f"Dataset CSV not found for '{dataset}'"}), 400

    algorithm = data.get("algorithm", "auto")
    n_samples = data.get("n_samples", 1000)
    device = data.get("device", "auto")

    algo_tag = algorithm.lower().replace("-", "_")
    output_filename = data.get("output_filename") or f"{dataset}_from_ckpt.csv"
    out_dir = AUGMENTED_UI_DIR / f"{algo_tag}_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename

    cmd = [
        _tabular_python(),
        str(ROOT_DIR / "checkpoint_generator.py"),
        "--checkpoint", str(ckpt_path),
        "--dataset", str(dataset_file),
        "--algorithm", str(algorithm),
        "--n-samples", str(n_samples),
        "--output-path", str(out_path),
        "--device", str(device),
    ]

    task = task_manager.create_task("checkpoint_generation", cmd, ROOT_DIR)
    task_manager.start_task(task)

    return jsonify({"success": True, "task_id": task.task_id, "command": " ".join(cmd), "output_path": str(out_path)})


@app.route("/api/run_augmentation", methods=["POST"])
def run_augmentation():
    """Launch main.py as an asynchronous background task."""
    active = task_manager.get_active_task()
    if active:
        return jsonify({"success": False, "error": f"A task ({active.task_id}) is already running."}), 400

    data = request.json or {}
    algos = data.get("algorithms", ["all"])
    not_runnable = [a for a in _tabular_algorithms() if not a["runnable"] and a["id"] in algos]
    if not_runnable:
        return jsonify({"success": False, "error": "Described-only algorithms (cannot be started): "
                        f"{[a['id'] for a in not_runnable]}"}), 400
    datasets = data.get("datasets", ["all"])
    n_samples = data.get("n_samples", 1000)
    epochs = data.get("epochs", 1000)
    device = data.get("device", "auto")
    use_gpu = data.get("use_gpu", None)
    resume = bool(data.get("resume", False))   # default: niente ripresa nascosta (D-UI); con resume=true si usa il checkpoint di runtime_results_ui/
    dc_file = data.get("dc_file")
    dc_dir = data.get("dc_dir")
    kamino_weights = data.get("kamino_weights", "toggled")

    cmd = [
        _tabular_python(),
        str(ROOT_DIR / "main.py"),
        "--algorithms", *algos,
        "--datasets", *datasets,
        "--epochs", str(epochs),
        "--kamino-weights", str(kamino_weights),
        "--augmented-dir", str(AUGMENTED_UI_DIR),
        "--results-dir", str(RESULTS_UI_DIR),
    ]

    if n_samples is not None and int(n_samples) > 0:
        cmd.extend(["--n-samples", str(n_samples)])

    if use_gpu is True:
        cmd.append("--use-gpu")
    elif use_gpu is False:
        cmd.append("--no-gpu")

    if device != "auto":
        cmd.extend(["--device", str(device)])

    if not resume:
        cmd.append("--no-resume")

    if dc_file:
        cmd.extend(["--dc-file", str(dc_file)])
    elif dc_dir:
        cmd.extend(["--dc-dir", str(dc_dir)])

    task = task_manager.create_task("augmentation", cmd, ROOT_DIR)
    task_manager.start_task(task)

    return jsonify({"success": True, "task_id": task.task_id, "command": " ".join(cmd)})


@app.route("/api/run_evaluation", methods=["POST"])
def run_evaluation():
    """Launch evaluate_metrics.py as an asynchronous background task."""
    active = task_manager.get_active_task()
    if active:
        return jsonify({"success": False, "error": f"A task ({active.task_id}) is already running."}), 400

    data = request.json or {}
    dataset = data.get("dataset", "all")
    algorithms = data.get("algorithms", ["all"])
    bins = data.get("bins", 30)

    cmd = [
        _tabular_python(),
        str(ROOT_DIR / "evaluate_metrics.py"),
        "--dataset", str(dataset),
        "--bins", str(bins),
        "--augmented_dir", str(AUGMENTED_UI_DIR),
        "--output_dir", str(EVAL_UI_DIR),
    ]

    if algorithms and "all" not in algorithms:
        cmd.extend(["--algorithms", *algorithms])

    task = task_manager.create_task("evaluation", cmd, ROOT_DIR)
    task_manager.start_task(task)

    return jsonify({"success": True, "task_id": task.task_id, "command": " ".join(cmd)})


@app.route("/api/stop_task/<task_id>", methods=["POST"])
def stop_task(task_id: str):
    success = task_manager.stop_task(task_id)
    return jsonify({"success": success})


@app.route("/api/logs/<task_id>", methods=["GET"])
def get_task_logs(task_id: str):
    start = int(request.args.get("start", 0))
    res = task_manager.get_logs_from(task_id, start_index=start)
    return jsonify(res)


@app.route("/api/evaluated_datasets", methods=["GET"])
def list_evaluated_datasets():
    datasets = sorted(set(get_available_evaluated_datasets(EVAL_DIR)) | set(get_available_evaluated_datasets(EVAL_UI_DIR)))
    return jsonify(datasets)


def _eval_dir_for(dataset: str) -> Path:
    """Cartella di valutazione da leggere per un dataset: quella con il file di fidelity piu' recente
    (le valutazioni di UI/assistente stanno in evaluation_metrics_ui/). ?source=ui|bench forza la scelta."""
    src = (request.args.get("source") or "").lower()
    if src == "ui":
        return EVAL_UI_DIR
    if src == "bench":
        return EVAL_DIR
    best, best_t = EVAL_DIR, -1.0
    for d in (EVAL_DIR, EVAL_UI_DIR):
        for name in (dataset, dataset.replace("_min", "") if "_min" in dataset else f"{dataset}_min"):
            f = d / "fidelity" / f"fidelity_{name}.csv"
            if f.is_file() and f.stat().st_mtime > best_t:
                best, best_t = d, f.stat().st_mtime
    return best


@app.route("/api/analytics/<dataset>", methods=["GET"])
def get_analytics(dataset: str):
    payload = compute_analytics_payload(_eval_dir_for(dataset), dataset)
    return jsonify(payload)


@app.route("/api/runtime_results", methods=["GET"])
def get_runtime_results():
    """Return the most recent augmentation benchmark CSV results."""
    csv_files = sorted(
        [f for d in (RESULTS_DIR, RESULTS_UI_DIR) if d.exists()
         for f in d.glob("augmentation_benchmark_2*.csv") if "summary" not in f.name],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )   # il piu' recente tra il benchmark e le esecuzioni lanciate da UI/assistente (runtime_results_ui/)
    if not csv_files:
        return jsonify([])

    try:
        df = pd.read_csv(csv_files[0])
        # Cleanly convert NaN to null for strict JSON specification compliance
        records = json.loads(df.to_json(orient="records"))
        return jsonify({
            "filename": csv_files[0].name,
            "source": "ui" if csv_files[0].parent == RESULTS_UI_DIR else "bench",
            "records": records,
        })
    except Exception as e:
        return jsonify({"error": str(e), "records": []})


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):   # Windows: la console cp1252 non codifica le emoji dei log
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    port = int(os.environ.get("PORT", 5005))
    print(f"\n=======================================================")
    print(f"🚀 Data Augmentation & Evaluation Web Dashboard")
    host = os.environ.get("HOST", "127.0.0.1")   # per aprire a rete locale: HOST=0.0.0.0
    print(f"📡 Serving on: http://{host}:{port}")
    for _n, _i in BLUEPRINTS.items():
        print(f"   blueprint {_n}: {'OK' if _i['available'] else 'non disponibile (' + str(_i['error']) + ')'}")
    print(f"📁 Workspace:  {ROOT_DIR}")
    print(f"=======================================================\n")
    app.run(host=host, port=port, debug=False)
