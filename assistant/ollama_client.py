"""Client per un modello locale servito da Ollama (http://127.0.0.1:11434), stessa interfaccia di ClaudeClient.

Serve nella fase di test (D12) al posto di Claude via API. Converte i messaggi e gli strumenti dal formato di
Anthropic (blocchi text / tool_use / tool_result, input_schema) a quello di Ollama (/api/chat con tools).
Nessuna chiave, nessun dato lascia il PC. Nessun fallback silenzioso: se il server o il modello mancano,
get_ollama_status() lo dice e il client non viene creato.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from .client import ModelResponse

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"


def ollama_url() -> str:
    return os.environ.get("OLLAMA_HOST_URL", "").strip() or DEFAULT_URL


def ollama_model() -> str:
    return os.environ.get("ASSISTANT_OLLAMA_MODEL", "").strip() or DEFAULT_MODEL


def _get(path: str, timeout: float = 3.0) -> Dict[str, Any]:
    with urllib.request.urlopen(ollama_url() + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_ollama_status() -> Dict[str, Any]:
    """{'ok': bool, 'model': str, 'reason': str}. Controlla che il server risponda e che il modello sia scaricato."""
    model = ollama_model()
    try:
        tags = _get("/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "model": model, "reason": f"Ollama server unreachable ({type(e).__name__})"}
    models = tags.get("models", [])
    names = {m.get("name", "") for m in models} | {m.get("model", "") for m in models}
    if model not in names and f"{model}:latest" not in names:
        return {"ok": False, "model": model, "reason": f"model '{model}' not downloaded (ollama pull {model})"}
    return {"ok": True, "model": model, "reason": ""}


def to_ollama_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                             "parameters": t["input_schema"]}} for t in tools]


def to_ollama_messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    names: Dict[str, str] = {}
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in c if b.get("type") == "text")
            calls = []
            for b in c:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b.get("input", {})}})
            msg: Dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # utente: testo e/o risultati di strumenti
            for b in c:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_name": names.get(b["tool_use_id"], ""),
                                "content": str(b.get("content", ""))})
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b.get("text", "")})
    return out


CLOUD_URL = "https://ollama.com"
CLOUD_DEFAULT_MODEL = "gpt-oss:120b"   # verificato il 2026-11-09: chiama gli strumenti; kimi-k2.6 richiede crediti a pagamento
KEYS_FILE = os.environ.get("OLLAMA_KEYS_FILE", os.path.join(os.path.expanduser("~"), "Desktop", "prog", "ChaviOllama.json"))
_DEAD_KEYS: set = set()   # indici di chiavi che in questa sessione hanno risposto 401/402/403/429


def cloud_model() -> str:
    return os.environ.get("ASSISTANT_OLLAMA_CLOUD_MODEL", "").strip() or CLOUD_DEFAULT_MODEL


def _load_cloud_keys() -> List[Dict[str, Any]]:
    """Chiavi nell'ordine d'uso: prima le gratuite, la PRO per ultima. Non le stampa né le restituisce alla UI."""
    try:
        with open(KEYS_FILE, encoding="utf-8") as f:
            keys = json.load(f).get("CLOUD_API_KEYS", [])
    except (OSError, ValueError, AttributeError):
        return []
    items = [{"i": i, "key": k.get("key", ""), "type": k.get("type", "")} for i, k in enumerate(keys) if k.get("key")]
    return sorted(items, key=lambda x: x["type"] == "PRO")


def get_cloud_status() -> Dict[str, Any]:
    """Non consuma quota: dice solo se il file di chiavi c'è e quante ne restano non scartate in questa sessione.
    La validità vera (quota, 401/429) si scopre alla prima richiesta e le chiavi inutilizzabili vengono saltate."""
    keys = _load_cloud_keys()
    live = [k for k in keys if k["i"] not in _DEAD_KEYS]
    if not keys:
        return {"ok": False, "model": cloud_model(), "reason": "Ollama cloud key file missing or empty", "n_keys": 0}
    if not live:
        return {"ok": False, "model": cloud_model(), "reason": "all cloud keys are unusable in this session", "n_keys": 0}
    return {"ok": True, "model": cloud_model(), "reason": "", "n_keys": len(live)}


class OllamaClient:
    # Un modello locale è più lento dell'API: limiti più larghi per il ciclo agentico.
    limits = {"max_seconds": 600.0, "max_steps": 12, "max_total_tokens": 120_000}

    def __init__(self, model: Optional[str] = None, url: Optional[str] = None, num_ctx: int = 12288,
                 think: Optional[bool] = None, timeout: float = 600.0, cloud: bool = False):
        self.cloud = cloud
        self.model = model or (cloud_model() if cloud else ollama_model())
        self.url = url or (CLOUD_URL if cloud else ollama_url())
        self.num_ctx = int(os.environ.get("ASSISTANT_OLLAMA_NUM_CTX", num_ctx))
        env_think = os.environ.get("ASSISTANT_OLLAMA_THINK", "").strip().lower()
        # `think` viene inviato solo se richiesto esplicitamente: alcuni modelli (gpt-oss) non accettano think=false
        self.think = think if think is not None else (True if env_think in ("1", "true", "si", "yes") else None)
        self.timeout = timeout

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if not self.cloud:
            return self._send(body, headers)
        last = ""
        for k in _load_cloud_keys():
            if k["i"] in _DEAD_KEYS:
                continue
            try:
                return self._send(body, {**headers, "Authorization": f"Bearer {k['key']}"})
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")[:200]
                if e.code in (401, 402, 403, 429):   # chiave non valida / senza quota: passa alla successiva
                    _DEAD_KEYS.add(k["i"])
                    last = f"HTTP {e.code}"
                    continue
                raise RuntimeError(f"Ollama cloud HTTP {e.code}: {text}") from None
        raise RuntimeError(f"no usable Ollama cloud key (last outcome: {last or 'no key'})")

    def _send(self, body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if self.cloud:
                raise
            raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None

    def create(self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
               max_tokens: int) -> ModelResponse:
        body = {"model": self.model, "stream": False,
                "messages": to_ollama_messages(system, messages), "tools": to_ollama_tools(tools),
                "options": {"num_predict": max(256, int(max_tokens)), "temperature": 0.3}}
        if self.think is not None:
            body["think"] = self.think
        if not self.cloud:
            body["options"]["num_ctx"] = self.num_ctx
            body["keep_alive"] = "30m"
        resp = self._post(body)
        msg = resp.get("message", {})
        blocks: List[Dict[str, Any]] = []
        text = (msg.get("content") or "").strip()
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            blocks.append({"type": "tool_use", "id": "ol_" + uuid.uuid4().hex[:10], "name": fn.get("name", ""),
                           "input": dict(args)})
        has_tool = any(b["type"] == "tool_use" for b in blocks)
        return ModelResponse(blocks, "tool_use" if has_tool else "end_turn",
                             int(resp.get("prompt_eval_count", 0) or 0), int(resp.get("eval_count", 0) or 0))
