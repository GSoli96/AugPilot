"""Ciclo agentico (tool_use / tool_result) con limiti di passi, tempo e token, e controllo dei numeri citati."""
from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from . import config
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, ToolContext, dispatch

NUM_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)")


@dataclass
class Session:
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    messages: List[Dict[str, Any]] = field(default_factory=list)
    ctx: ToolContext = field(default_factory=ToolContext)
    notes: List[str] = field(default_factory=list)       # eventi di sistema da anteporre al prossimo messaggio utente
    numbers: List[float] = field(default_factory=list)   # numeri presenti nei risultati degli strumenti e nei messaggi utente
    created: float = field(default_factory=time.time)


def _extract_numbers(text: str) -> List[float]:
    out = []
    for m in NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", ".")))
        except ValueError:
            pass
    return out


def ungrounded_numbers(text: str, known: List[float]) -> List[str]:
    """Numeri del testo che non compaiono (a meno dell'arrotondamento) nei dati della sessione."""
    bad = []
    for m in NUM_RE.findall(text):
        s = m.replace(",", ".")
        v = float(s)
        d = len(s.split(".")[1]) if "." in s else 0
        if d == 0 and (v < 10 or 1900 <= v <= 2100):
            continue
        if not any(round(k, d) == v or abs(k - v) < 10 ** -d for k in known):
            bad.append(m)
    return sorted(set(bad))


def _clip(obj: Any, limit: int) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + f'... [TRUNCATED: {len(s)} characters in total]'


class Agent:
    def __init__(self, client, limits: Optional[Dict[str, Any]] = None, system: str = SYSTEM_PROMPT):
        self.client = client
        self.limits = {**config.LIMITS, **getattr(client, "limits", {}), **(limits or {})}
        self.system = system

    def run_turn(self, session: Session, user_text: str) -> Iterator[Dict[str, Any]]:
        """Generatore di eventi: text | tool_call | tool_result | pending | warning | limit | error | done."""
        L = self.limits
        start_len = len(session.messages)
        notes, session.notes = session.notes, []
        text = "\n".join([f"[system] {n}" for n in notes] + [user_text]) if notes else user_text
        session.messages.append({"role": "user", "content": text})
        session.numbers += _extract_numbers(text)
        t0, tokens, steps = time.time(), 0, 0
        last_text = ""
        try:
            while True:
                if steps >= L["max_steps"]:
                    yield {"type": "limit", "text": f"limit of {L['max_steps']} steps reached"}
                    break
                if time.time() - t0 > L["max_seconds"]:
                    yield {"type": "limit", "text": f"time limit ({L['max_seconds']}s) reached"}
                    break
                if tokens > L["max_total_tokens"]:
                    yield {"type": "limit", "text": f"token limit ({L['max_total_tokens']}) reached"}
                    break
                steps += 1
                r = self.client.create(self.system, session.messages, TOOL_SCHEMAS, L["max_output_tokens"])
                tokens += r.input_tokens + r.output_tokens
                session.messages.append({"role": "assistant", "content": r.content})
                uses = [b for b in r.content if b["type"] == "tool_use"]
                for b in r.content:
                    if b["type"] == "text" and b["text"].strip():
                        last_text = b["text"]
                        yield {"type": "text", "text": b["text"]}
                if not uses:
                    break
                results = []
                for u in uses:
                    yield {"type": "tool_call", "id": u["id"], "name": u["name"], "input": u["input"]}
                    res = dispatch(session.ctx, u["name"], u["input"])
                    s = _clip(res, L["max_tool_result_chars"])
                    session.numbers += _extract_numbers(s)
                    yield {"type": "tool_result", "id": u["id"], "name": u["name"], "ok": "error" not in res,
                           "result": res if len(s) < 6000 else {"truncated": s[:6000]}}
                    if isinstance(res, dict) and res.get("status") == "needs_confirmation":
                        yield {"type": "pending", "pending_id": res["pending_id"], "reason": res["reason"],
                               "plan": res["plan"]}
                    results.append({"type": "tool_result", "tool_use_id": u["id"], "content": s,
                                    **({"is_error": True} if "error" in res else {})})
                session.messages.append({"role": "user", "content": results})
            if last_text:
                bad = ungrounded_numbers(last_text, session.numbers)
                if bad:
                    yield {"type": "warning", "text": "Numbers in the answer with no match in the tool results "
                           f"or in the user's messages (please verify): {', '.join(bad)}"}
            yield {"type": "done", "steps": steps, "tokens": tokens}
        except Exception as e:  # noqa: BLE001  nessun fallback silenzioso: errore esplicito, storia ripristinata
            del session.messages[start_len:]
            session.notes = notes + session.notes
            yield {"type": "error", "text": f"{type(e).__name__}: {str(e)[:400]}"}
