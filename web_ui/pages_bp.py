"""Sezioni dell'interfaccia: /copilot, /guided, /control, /catalog, /evidence (Prove), /challenge (Sfida), l'API delle descrizioni brevi
/api/catalog_info/<modalita> e il caricamento di un dataset proprio POST /api/datasets/upload (logica in web_ui/uploads.py).

Le pagine usano le API gia' esistenti (/api/modalities, /api/modality/<m>/..., /api/assistant/...). Qui non c'e' logica di calcolo.
Le descrizioni stanno in catalog_info/algorithms_<mod>.json e catalog_info/datasets_<mod>.json (scritti a mano, fuori da questo modulo):
file mancanti, illeggibili o incompleti non producono errori, solo campi vuoti.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from urllib.parse import urlparse

from flask import Blueprint, jsonify, render_template, request

bp = Blueprint("pages", __name__)

CATALOG_INFO_DIR = Path(__file__).resolve().parent.parent / "catalog_info"
MODALITIES = ("tabular", "ts", "graph")


def _read_map(path: Path, warnings: List[str]) -> Dict[str, Any]:
    """Legge un JSON {nome: scheda}. Qualunque problema (assente, non JSON, non un oggetto) -> {} e, se il file c'e', un avviso."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        warnings.append(f"{path.name}: unreadable ({type(e).__name__})")
        return {}
    if not isinstance(data, dict):
        warnings.append(f"{path.name}: expected a JSON object")
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def load_catalog_info(modality: str) -> Dict[str, Any]:
    warnings: List[str] = []
    out: Dict[str, Any] = {
        "algorithms": _read_map(CATALOG_INFO_DIR / f"algorithms_{modality}.json", warnings),
        "datasets": _read_map(CATALOG_INFO_DIR / f"datasets_{modality}.json", warnings),
    }
    if warnings:
        out["warnings"] = warnings
    return out


@bp.route("/copilot")
def copilot():
    return render_template("copilot.html")


@bp.route("/guided")
def guided():
    return render_template("guided.html")


@bp.route("/control")
def control():
    return render_template("control.html")


@bp.route("/catalog")
def catalog():
    return render_template("catalog.html")


@bp.route("/api/catalog_info/<modality>", methods=["GET"])
def catalog_info(modality: str):
    if modality not in MODALITIES:
        return jsonify({"error": f"unknown modality: {modality}", "valid": list(MODALITIES)}), 404
    return jsonify(load_catalog_info(modality))


@bp.route("/evidence")
def evidence():
    """Pagina «Prove»: i dati arrivano da /api/kb/summary (vedi web_ui/assistant_bp.py)."""
    return render_template("evidence.html")


@bp.route("/challenge")
def challenge():
    """Pagina «Sfida»: il visitatore indovina l'algoritmo migliore e si confronta con AugPilot. Logica in assistant/kb/challenge.py."""
    return render_template("challenge.html")


def _challenge_error(e):
    return jsonify({"error": str(e)}), getattr(e, "status", 400)


@bp.route("/api/challenge/datasets", methods=["GET"])
def challenge_datasets():
    """Dataset del KB per modalita' e livello, con il numero di algoritmi misurati e se la sfida e' possibile."""
    from assistant.kb import challenge
    return jsonify(challenge.list_datasets(challenge.get_kb()))


@bp.route("/api/challenge/candidates", methods=["GET"])
def challenge_candidates():
    """Algoritmi misurati su un dataset (senza punteggi). Query: modality, dataset, level (solo grafi)."""
    from assistant.kb import challenge
    try:
        return jsonify(challenge.candidates(challenge.get_kb(), (request.args.get("modality") or "").strip(),
                                            (request.args.get("dataset") or "").strip(),
                                            (request.args.get("level") or "").strip() or None))
    except challenge.ChallengeError as e:
        return _challenge_error(e)


@bp.route("/api/challenge/reveal", methods=["POST"])
def challenge_reveal():
    """Risultato della sfida. Corpo JSON: {modality, level?, dataset, profile, guess}."""
    from assistant.kb import challenge
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body expected: {modality, level, dataset, profile, guess}"}), 400
    try:
        return jsonify(challenge.reveal(challenge.get_kb(), str(data.get("modality") or "").strip(),
                                        str(data.get("dataset") or "").strip(), str(data.get("profile") or "balanced").strip(),
                                        str(data.get("guess") or "").strip(), level=str(data.get("level") or "").strip() or None))
    except challenge.ChallengeError as e:
        return _challenge_error(e)


@bp.route("/api/datasets/upload", methods=["POST"])
def upload_dataset():
    """Carica un dataset dell'utente (multipart: modality, file). Tabulare: .csv con la classe nell'ultima colonna;
    ts: .npz con `data` (N,T,D) e, opzionale, `labels`. Mai sovrascrive: se il nome esiste aggiunge un suffisso."""
    from assistant import config
    from web_ui import uploads
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc != request.host:     # difesa CSRF minima, come per le macchine
        return jsonify({"ok": False, "error": "origin not allowed"}), 403
    mod = (request.form.get("modality") or "").strip()
    if mod == "graph":
        return jsonify({"ok": False, "error": "graphs cannot be uploaded: use the datasets already on disk"}), 400
    if mod not in uploads.EXT:
        return jsonify({"ok": False, "error": "modality must be tabular or ts"}), 400
    limit = config.UPLOAD_MAX_BYTES
    if request.content_length and request.content_length > limit + 64 * 1024:
        return jsonify({"ok": False, "error": f"file too large: maximum {limit // (1024 * 1024)} MB"}), 413
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "no file in the `file` field"}), 400
    raw = f.stream.read(limit + 1)
    try:
        res = uploads.save_upload(mod, f.filename, raw)
    except uploads.UploadError as e:
        return jsonify({"ok": False, "error": str(e)}), e.status
    return jsonify({"ok": True, "dataset": res}), 201
