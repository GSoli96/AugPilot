"""Blueprint Flask delle macchine (esecuzione distribuita): pagina /machines e API /api/machines/*.

Solo interfaccia e modello dati: nessuno scheduler.

Registrazione (a cura di chi possiede app.py):
    try:
        from web_ui.machines_bp import bp as machines_bp
        app.register_blueprint(machines_bp)
    except Exception as e: ...
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from flask import Blueprint, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distributed import remote as _remote  # noqa: E402
from distributed.models import MachineStore, ValidationError  # noqa: E402

bp = Blueprint("machines", __name__, template_folder="templates")

_store = MachineStore()


def set_store(store: Optional[MachineStore]) -> None:
    """Per i test: usa un file diverso da distributed/machines.json."""
    global _store
    _store = store if store is not None else MachineStore()


def _err(msg: str, code: int, **extra):
    return jsonify({"ok": False, "error": msg, **extra}), code


@bp.before_request
def _same_origin():
    """Difesa CSRF minima: le richieste che modificano stato non possono arrivare da un'altra origine."""
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            return _err("origin not allowed", 403)


@bp.errorhandler(ValidationError)
def _validation(e: ValidationError):
    return _err("invalid data", 400, fields=e.errors)


@bp.errorhandler(RuntimeError)
def _runtime(e: RuntimeError):
    return _err(str(e), 500)


def _json_body():
    data = request.get_json(silent=True)
    if data is None:
        raise ValidationError({"_": "JSON body required"})
    return data


def _summary(ms):
    return {"count": len(ms), "enabled_count": sum(1 for m in ms if m.get("enabled"))}


@bp.route("/machines")
def page():
    return render_template("machines.html")


@bp.route("/api/machines", methods=["GET"])
def list_machines():
    ms = _store.list()
    return jsonify({"ok": True, **_summary(ms), "machines": ms,
                    "ssh_client": bool(_remote.ssh_binary())})


@bp.route("/api/machines", methods=["POST"])
def add_machine():
    m = _store.add(_json_body())
    return jsonify({"ok": True, "machine": m}), 201


@bp.route("/api/machines/<mid>", methods=["PUT"])
def update_machine(mid):
    m = _store.update(mid, _json_body())
    if m is None:
        return _err("machine not found", 404)
    return jsonify({"ok": True, "machine": m})


@bp.route("/api/machines/<mid>", methods=["DELETE"])
def delete_machine(mid):
    if not _store.delete(mid):
        return _err("machine not found", 404)
    return jsonify({"ok": True})


@bp.route("/api/machines/<mid>/check", methods=["POST"])
def check_machine(mid):
    m = _store.get(mid)
    if m is None:
        return _err("machine not found", 404)
    state, msg = _remote.check_machine(m)
    m = _store.set_status(mid, state, msg)
    if m is None:   # rimossa nel frattempo
        return _err("machine not found", 404)
    return jsonify({"ok": True, "machine": m})


@bp.route("/api/machines/<mid>/profile", methods=["POST"])
def profile_machine(mid):
    m = _store.get(mid)
    if m is None:
        return _err("machine not found", 404)
    try:
        prof = _remote.profile_machine(m)
    except FileNotFoundError as e:
        return _err(str(e), 501)
    except ConnectionError as e:
        _store.set_status(mid, "errore", str(e))
        return _err(f"SSH connection failed: {e}", 502)
    m = _store.set_profile(mid, prof)
    if m is None:
        return _err("machine not found", 404)
    return jsonify({"ok": True, "machine": m})


@bp.route("/api/machines/export", methods=["GET"])
def export_machines():
    resp = jsonify(_store.export())
    resp.headers["Content-Disposition"] = "attachment; filename=machines_export.json"
    return resp


@bp.route("/api/machines/import", methods=["POST"])
def import_machines():
    added = _store.import_(_json_body())
    ms = _store.list()
    return jsonify({"ok": True, "imported": len(added), **_summary(ms)}), 201


# ---------------------------------------------------------------- benchmark distribuito (sola lettura)
# Lo stato lo scrive l'orchestratore (skill `orchestratore`) in <ORCH_STATE_DIR>/fleet.json. Qui si legge soltanto:
# nessun comando verso le macchine. Dal file si prendono solo i campi di avanzamento; IP, utenti, nome host, comandi e
# percorsi dei job non escono mai da questo endpoint.
ORCH_DASHBOARD_URL = "http://127.0.0.1:8765"


def orch_state_dir() -> Path:
    env = os.environ.get("ORCH_STATE_DIR", "").strip()
    return Path(env) if env else Path.home() / ".claude" / "skills" / "orchestratore" / "state"


_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b|\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*)?")
_USERHOST = re.compile(r"\S+@\S+")


def _redact(text: str) -> str:
    """Toglie da un testo libero (avvisi dell'orchestratore) i dettagli tra parentesi, gli IP e le forme utente@host."""
    t = re.sub(r"\([^)]*\)", "", str(text))
    t = _USERHOST.sub("[account]", _IP.sub("[IP]", t))
    return re.sub(r"\s+", " ", t).strip()[:120]


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _fleet_job(j: dict, events: list) -> dict:
    st = j.get("status") if isinstance(j.get("status"), dict) else {}
    pr = st.get("progress") if isinstance(st.get("progress"), dict) else {}
    last = None
    for e in reversed(events):
        if isinstance(e, dict) and e.get("job") == j.get("id"):
            last = {"ts": e.get("ts"), "kind": e.get("kind")}
            break
    return {"id": str(j.get("id", "")), "name": str(j.get("name") or j.get("id") or ""), "state": st.get("state") or "sconosciuto",
            "done": _num(pr.get("done")), "total": _num(pr.get("total")), "pct": _num(pr.get("pct")),
            "attempts": _num(st.get("attempts")), "restarts": _num(st.get("restarts")),
            "partition": j.get("partition") if isinstance(j.get("partition"), str) else None,
            "updated": st.get("updated_iso") if isinstance(st.get("updated_iso"), str) else None, "last_event": last}


def read_fleet(state_dir: Path = None) -> dict:
    d = Path(state_dir) if state_dir else orch_state_dir()
    f = d / "fleet.json"
    out = {"available": False, "dashboard_url": os.environ.get("ORCH_DASHBOARD_URL", "").strip() or ORCH_DASHBOARD_URL,
           "state_file": "fleet.json", "machines": [], "alerts": []}
    if not f.is_file():
        out["message"] = ("Orchestrator state not found: no distributed benchmark is running, or the orchestrator "
                          "has not been started. The path can be changed with the ORCH_STATE_DIR environment variable.")
        return out
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("machines"), dict):
            raise ValueError("unexpected structure")
    except (OSError, ValueError) as e:
        out["message"] = f"fleet.json is unreadable ({type(e).__name__}): try again in a few moments."
        return out
    out["available"] = True
    ts = _num(data.get("ts"))
    out["updated"] = data.get("iso") if isinstance(data.get("iso"), str) else None
    out["age_s"] = round(max(0.0, time.time() - ts)) if ts else None
    out["alerts"] = [_redact(a) for a in data.get("alerts", []) if isinstance(a, (str, int, float))][:20]
    for key, m in data["machines"].items():
        if not isinstance(m, dict):
            continue
        dd = m.get("data") if isinstance(m.get("data"), dict) else {}
        mac = dd.get("machine") if isinstance(dd.get("machine"), dict) else {}
        if m.get("inactive"):
            state = "inattiva"
        elif m.get("reachable") is False:
            state = "non raggiungibile"
        elif m.get("reachable") is None:
            state = "sconosciuta"
        else:
            state = "libera" if mac.get("free") else "occupata"
        gpus = [{"name": str(g.get("name", "GPU")), "util": _num(g.get("util")), "mem_used_mb": _num(g.get("mem_used_mb")),
                 "mem_total_mb": _num(g.get("mem_total_mb"))} for g in (mac.get("gpus") or []) if isinstance(g, dict)]
        ram = mac.get("ram") if isinstance(mac.get("ram"), dict) else {}
        events = dd.get("events") if isinstance(dd.get("events"), list) else []
        jobs = [_fleet_job(j, events) for j in (dd.get("jobs") or []) if isinstance(j, dict)]
        part = [{"name": p.get("name"), "gpu_model": p.get("gpu_model") or None, "gpus_free": _num(p.get("gpus_free")),
                 "gpus_total": _num(p.get("gpus_total"))} for p in (mac.get("partitions") or []) if isinstance(p, dict) and p.get("gpus_total")]
        out["machines"].append({"key": str(key), "name": str(m.get("name") or key), "kind": mac.get("kind") or "lab", "state": state,
                                "gpus": gpus, "cpu_percent": _num(mac.get("cpu_percent")), "ram_used_gb": _num(ram.get("used_gb")),
                                "ram_total_gb": _num(ram.get("total_gb")), "partitions": part, "jobs": jobs})
    return out


@bp.route("/api/machines/fleet", methods=["GET"])
def fleet():
    """Stato del benchmark distribuito, letto da fleet.json dell'orchestratore (solo lettura)."""
    return jsonify(read_fleet())
