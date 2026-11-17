"""Test del client Ollama senza server: conversione dei messaggi/strumenti e lettura della risposta."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import client as cl
from assistant import ollama_client as oc
from assistant.tools import TOOL_SCHEMAS


def test_tools_conversion():
    t = oc.to_ollama_tools(TOOL_SCHEMAS)
    assert len(t) == len(TOOL_SCHEMAS)
    assert t[0]["type"] == "function" and t[0]["function"]["name"] == TOOL_SCHEMAS[0]["name"]
    assert t[0]["function"]["parameters"] == TOOL_SCHEMAS[0]["input_schema"]


def test_messages_conversion_with_tool_round():
    msgs = [
        {"role": "user", "content": "ciao"},
        {"role": "assistant", "content": [{"type": "text", "text": "guardo"},
                                          {"type": "tool_use", "id": "a1", "name": "list_algorithms", "input": {"modality": "ts"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a1", "content": "{\"n\": 39}"}]},
    ]
    out = oc.to_ollama_messages("SISTEMA", msgs)
    assert out[0] == {"role": "system", "content": "SISTEMA"}
    assert out[1] == {"role": "user", "content": "ciao"}
    assert out[2]["role"] == "assistant" and out[2]["tool_calls"][0]["function"]["arguments"] == {"modality": "ts"}
    assert out[3] == {"role": "tool", "tool_name": "list_algorithms", "content": "{\"n\": 39}"}


def test_create_parses_tool_calls(monkeypatch):
    class R:
        def __init__(self, d): self.d = json.dumps(d).encode()
        def read(self): return self.d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    sent = {}

    def fake(req, timeout=None):
        sent["body"] = json.loads(req.data.decode())
        return R({"message": {"content": "", "tool_calls": [{"function": {"name": "list_datasets", "arguments": {"modality": "graph"}}}]},
                  "prompt_eval_count": 11, "eval_count": 7})

    monkeypatch.setattr(oc.urllib.request, "urlopen", fake)
    r = oc.OllamaClient("qwen3:8b").create("S", [{"role": "user", "content": "x"}], TOOL_SCHEMAS, 512)
    assert r.stop_reason == "tool_use" and r.input_tokens == 11 and r.output_tokens == 7
    assert r.content[0]["name"] == "list_datasets" and r.content[0]["input"] == {"modality": "graph"}
    assert "think" not in sent["body"] and sent["body"]["stream"] is False


def test_status_offline_when_server_down(monkeypatch):
    def boom(*a, **k): raise oc.urllib.error.URLError("giu")
    monkeypatch.setattr(oc.urllib.request, "urlopen", boom)
    monkeypatch.setenv("ASSISTANT_BACKEND", "ollama")
    st = cl.get_status()
    assert st["backend"] == "ollama" and st["mode"] == "offline" and "unreachable" in st["reason"]
    assert cl.make_client() is None
