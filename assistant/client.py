"""Client del modello: Claude via SDK `anthropic` oppure client simulato per i test.

Interfaccia comune: create(system, messages, tools, max_tokens) -> ModelResponse.
Nessun fallback silenzioso: se la chiave manca o l'SDK non c'e', `get_status()` dice 'offline' e il motivo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from . import config


@dataclass
class ModelResponse:
    content: List[Dict[str, Any]]          # blocchi {"type":"text",...} | {"type":"tool_use","id","name","input"}
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0


class ClaudeClient:
    def __init__(self, api_key: str, model: str):
        import anthropic  # import locale: se manca l'SDK lo si vede in get_status
        self._anthropic = anthropic
        self._c = anthropic.Anthropic(api_key=api_key, timeout=90.0, max_retries=2)
        self.model = model

    def create(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], max_tokens: int) -> ModelResponse:
        r = self._c.messages.create(
            model=self.model, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages, tools=tools)
        blocks = []
        for b in r.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        return ModelResponse(blocks, r.stop_reason or "end_turn", getattr(r.usage, "input_tokens", 0) or 0,
                             getattr(r.usage, "output_tokens", 0) or 0)


class MockClient:
    """Client simulato SOLO per i test: restituisce risposte preconfigurate (liste di blocchi o callable)."""

    def __init__(self, script: Sequence[Union[ModelResponse, List[Dict[str, Any]], Callable]], model: str = "mock"):
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []
        self.model = model

    def create(self, system, messages, tools, max_tokens) -> ModelResponse:
        import copy
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if not self.script:
            raise RuntimeError("MockClient: copione esaurito")
        nxt = self.script.pop(0)
        if callable(nxt):
            nxt = nxt(messages)
        if isinstance(nxt, ModelResponse):
            return nxt
        has_tool = any(b.get("type") == "tool_use" for b in nxt)
        return ModelResponse(nxt, "tool_use" if has_tool else "end_turn", 100, 50)


def _backend_pref() -> str:
    """ASSISTANT_BACKEND = ollama-cloud | ollama | claude | auto.
    auto: Ollama cloud (chiavi in ChaviOllama.json), poi Ollama locale, poi Claude, poi offline."""
    import os
    v = os.environ.get("ASSISTANT_BACKEND", "auto").strip().lower()
    return v if v in ("ollama-cloud", "ollama", "claude", "auto") else "auto"


def _cloud_status() -> Dict[str, Any]:
    from .ollama_client import get_cloud_status
    o = get_cloud_status()
    return {"backend": "ollama-cloud", "model": o["model"], "key_source": f"ChaviOllama.json ({o.get('n_keys', 0)} keys)",
            "mode": "online" if o["ok"] else "offline", "reason": o["reason"]}


def _claude_status() -> Dict[str, Any]:
    key, src = config.get_api_key()
    st = {"backend": "claude", "model": config.get_model(), "key_source": src}
    if not key:
        return {**st, "mode": "offline", "reason": f"Claude key not found ({src}): no model available"}
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return {**st, "mode": "offline", "reason": "'anthropic' SDK not installed"}
    return {**st, "mode": "online", "reason": ""}


def _ollama_status() -> Dict[str, Any]:
    from .ollama_client import get_ollama_status
    o = get_ollama_status()
    return {"backend": "ollama", "model": o["model"], "key_source": "none (local model)",
            "mode": "online" if o["ok"] else "offline", "reason": o["reason"]}


def get_status() -> Dict[str, Any]:
    """Stato per la UI: mai la chiave, solo l'origine. Campo `backend`: ollama | claude."""
    pref = _backend_pref()
    if pref == "claude":
        return _claude_status()
    if pref == "ollama-cloud":
        return _cloud_status()
    if pref == "ollama":
        return _ollama_status()
    cl, o, c = _cloud_status(), _ollama_status(), _claude_status()   # auto
    for st in (cl, o, c):
        if st["mode"] == "online":
            return st
    return {**c, "reason": f"Ollama cloud: {cl['reason']}; local Ollama: {o['reason']}; Claude: {c['reason']}"}


def make_client():
    st = get_status()
    if st["mode"] != "online":
        return None
    if st["backend"] in ("ollama", "ollama-cloud"):
        from .ollama_client import OllamaClient
        return OllamaClient(st["model"], cloud=(st["backend"] == "ollama-cloud"))
    key, _ = config.get_api_key()
    return ClaudeClient(key, config.get_model())
