"""Blueprint Flask dell'assistente: pagina /assistant e API /api/assistant/*.

Registrazione (a cura di chi possiede app.py):
    try:
        from web_ui.assistant_bp import bp as assistant_bp
        app.register_blueprint(assistant_bp)
    except Exception as e: ...
Il task manager dell'app viene riusato se disponibile (web_ui.app.task_manager), altrimenti ne usa uno proprio.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from assistant import client as _client  # noqa: E402
from assistant import config as _config  # noqa: E402
from assistant.agent import Agent, Session  # noqa: E402
from assistant.tools import TOOL_SCHEMAS, dispatch, execute_pending  # noqa: E402

bp = Blueprint("assistant", __name__, template_folder="templates")

_sessions: Dict[str, Session] = {}
_lock = threading.Lock()
_own_tm = None
_agent_override: Optional[Agent] = None   # per i test (client simulato)
MAX_SESSIONS = 50


def set_agent(agent: Optional[Agent]) -> None:
    global _agent_override
    _agent_override = agent


def _task_manager():
    global _own_tm
    for name in ("web_ui.app", "__main__"):
        tm = getattr(sys.modules.get(name), "task_manager", None)
        if tm is not None:
            return tm
    if _own_tm is None:
        from web_ui.task_manager import TaskManager
        _own_tm = TaskManager()
    return _own_tm


def _get_session(sid: Optional[str]) -> Session:
    with _lock:
        if sid and sid in _sessions:
            return _sessions[sid]
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))
        s = Session()
        s.ctx.task_manager = _task_manager()
        s.ctx.session_id = s.id
        _sessions[s.id] = s
        return s


def _status() -> Dict[str, Any]:
    if _agent_override is not None:
        return {"mode": "online", "model": getattr(_agent_override.client, "model", "mock"),
                "key_source": "test", "reason": "", "limits": _config.LIMITS}
    st = _client.get_status()
    st["limits"] = _config.LIMITS
    return st


def _agent() -> Optional[Agent]:
    if _agent_override is not None:
        return _agent_override
    c = _client.make_client()
    return Agent(c) if c else None


def _events(session: Session, data: Dict[str, Any]):
    yield {"type": "session", "session_id": session.id}
    action = data.get("offline_action")
    if action:  # strumento invocato direttamente dall'utente (utile soprattutto offline)
        name, args = str(action.get("tool", "")), action.get("args") or {}
        yield {"type": "tool_call", "id": "manual", "name": name, "input": args}
        res = dispatch(session.ctx, name, args)
        yield {"type": "tool_result", "id": "manual", "name": name, "ok": "error" not in res, "result": res}
        if res.get("status") == "needs_confirmation":
            yield {"type": "pending", "pending_id": res["pending_id"], "reason": res["reason"], "plan": res["plan"]}
        yield {"type": "done", "steps": 0, "tokens": 0}
        return
    msg = str(data.get("message", "")).strip()
    if not msg:
        yield {"type": "error", "text": "empty message"}
        return
    if len(msg) > 4000:
        yield {"type": "error", "text": "message too long (max 4000 characters)"}
        return
    agent = _agent()
    if agent is None:
        st = _status()
        yield {"type": "offline", "text": "OFFLINE MODE: " + st["reason"] + ". No model is responding: no answer "
               "is simulated. Use the tools panel (deterministic recommender, datasets, comparison)."}
        yield {"type": "done", "steps": 0, "tokens": 0}
        return
    yield from agent.run_turn(session, msg)


@bp.route("/assistant")
def page():
    return render_template("assistant.html")


@bp.route("/api/assistant/status", methods=["GET"])
def status():
    st = _status()
    st["tools"] = [t["name"] for t in TOOL_SCHEMAS]
    return jsonify(st)


@bp.route("/api/assistant/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session = _get_session(data.get("session_id"))
    if request.args.get("stream", "1") == "0":
        return jsonify({"session_id": session.id, "events": list(_events(session, data))})

    def gen():
        try:
            for ev in _events(session, data):
                yield "data: " + json.dumps(ev, ensure_ascii=False, default=str) + "\n\n"
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps({"type": "error", "text": f"{type(e).__name__}: {e}"}) + "\n\n"
    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.route("/api/assistant/confirm", methods=["POST"])
def confirm():
    """Azione umana: conferma o rifiuta un'esecuzione proposta dall'agente."""
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    with _lock:
        session = _sessions.get(sid)
    if session is None:
        return jsonify({"error": "unknown session"}), 404
    approve = bool(data.get("approve"))
    res = execute_pending(session.ctx, str(data.get("pending_id", "")), approve)
    if "error" in res:
        return jsonify(res), 400
    session.notes.append("The user CONFIRMED and the task has started: " + json.dumps(res, ensure_ascii=False)
                         if approve and res.get("status") == "started"
                         else "The user REJECTED the run: " + json.dumps(res, ensure_ascii=False))
    return jsonify(res)


@bp.route("/api/assistant/task/<task_id>", methods=["GET"])
def task(task_id: str):
    s = _get_session(request.args.get("session_id"))
    res = dispatch(s.ctx, "get_task_status", {"task_id": task_id})
    return jsonify(res), (404 if "error" in res else 200)


@bp.route("/api/assistant/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    with _lock:
        _sessions.pop(data.get("session_id"), None)
    return jsonify({"ok": True})


@bp.route("/api/kb/summary", methods=["GET"])
def kb_summary_api():
    """Riepilogo del KB del benchmark per modalita' (per la pagina «Prove»).
    Query: modality=tabular|ts|graph (obbligatorio), level=node|graph (solo grafi), dataset=<nome> (opzionale).
    Risposta: datasets, algorithms (punteggi per asse medi, tempo mediano, tasso di errore), matrix (dataset x algoritmo)."""
    from assistant.kb.summary import MODALITIES, kb_summary
    mod = (request.args.get("modality") or "").strip()
    if mod not in MODALITIES:
        return jsonify({"error": f"modality must be one of {list(MODALITIES)}"}), 400
    level = (request.args.get("level") or "").strip() or None
    if level not in (None, "node", "graph"):
        return jsonify({"error": "level must be node or graph"}), 400
    ds = (request.args.get("dataset") or "").strip() or None
    out = kb_summary(mod, level=level, dataset=ds)
    try:   # pesi dichiarati dei profili (sola lettura): la pagina «Prove» ricalcola il punteggio col profilo scelto
        from assistant.recommender import PROFILES
        out["profiles"] = {k: dict(v) for k, v in PROFILES.items()}
    except Exception:  # noqa: BLE001
        out["profiles"] = {}
    return jsonify(out)
