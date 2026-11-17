"""Test dell'assistente. NESSUNA chiamata all'API reale: si usa MockClient."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant import catalog, client, config, datasets_tools, evidence, recommender  # noqa: E402
from assistant.agent import Agent, Session, ungrounded_numbers  # noqa: E402
from assistant.client import MockClient  # noqa: E402
from assistant.tools import ToolContext, dispatch, execute_pending  # noqa: E402


@pytest.fixture
def real_kb(monkeypatch):
    """KB finale reale (assistant/kb): da li' vengono gli stati executable/described del catalogo tabellare."""
    from assistant.kb import store
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(store.DEFAULT_DIR))


# ---------------------------------------------------------------- finti di supporto
class FakeTask:
    def __init__(self, tid):
        self.task_id = tid
        import threading
        self._lock = threading.Lock()
        self.logs = ["riga1\n", "riga2\n"]

    def to_dict(self):
        return {"task_id": self.task_id, "status": "RUNNING"}


class FakeTM:
    def __init__(self):
        self.created = []
        self.started = []
        self.active = None

    def get_active_task(self):
        return self.active

    def create_task(self, ttype, cmd, cwd):
        t = FakeTask(f"t{len(self.created)}")
        t.cmd = cmd
        self.created.append((ttype, cmd))
        return t

    def start_task(self, t):
        self.started.append(t.task_id)

    def get_task(self, tid):
        return FakeTask(tid) if tid.startswith("t") else None


def tu(i, name, **inp):
    return {"type": "tool_use", "id": i, "name": name, "input": inp}


def txt(s):
    return {"type": "text", "text": s}


def session_with_tm():
    s = Session()
    s.ctx.task_manager = FakeTM()
    return s


# ---------------------------------------------------------------- sicurezza dei percorsi / chiave
def test_safe_path_blocks_outside():
    for bad in ("C:/Windows/win.ini", "../config.py", "config.py", "datasets/../config.py", "/etc/passwd",
                "Algorithms/Algorithms.md"):
        with pytest.raises((ValueError, PermissionError, FileNotFoundError)):
            config.safe_path(bad)
    assert config.safe_path("datasets/iris0_min.csv").name == "iris0_min.csv"


def test_analyze_rejects_arbitrary_paths():
    r = dispatch(ToolContext(), "analyze_dataset", {"modality": "tabular", "dataset": "../config.py"})
    assert "error" in r
    r = dispatch(ToolContext(), "analyze_dataset", {"modality": "tabular", "dataset": "C:\\Windows\\win.ini"})
    assert "error" in r
    r = dispatch(ToolContext(), "analyze_dataset", {"modality": "tabular", "dataset": "ChaviOllama.json"})
    assert "error" in r


def test_offline_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "KEY_FILE", tmp_path / "nonesiste.json")
    st = client.get_status()
    assert st["mode"] == "offline" and "key" in st["reason"]
    assert client.make_client() is None
    assert "api_key" not in json.dumps(st)


def test_key_order_env_then_file(monkeypatch, tmp_path):
    f = tmp_path / "k.json"
    f.write_text(json.dumps({"api_key": "file-key"}))
    monkeypatch.setattr(config, "KEY_FILE", f)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert config.get_api_key() == ("env-key", "env")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert config.get_api_key() == ("file-key", "file")
    assert client.get_status()["mode"] == "online"  # stato soltanto: nessun client creato, nessuna chiamata


# ---------------------------------------------------------------- catalogo e consigliere
def test_tabular_catalog_status(real_kb):
    c = catalog.get_catalog("tabular")
    by = {a["key"]: a for a in c["algorithms"]}
    assert {"smote", "ddpm", "kamino", "syrfd"} <= set(by)
    assert "tabularargn" not in by                      # alias eliminato
    assert by["smote"]["status"] == "executable"        # ha esecuzioni SUCCESS nel KB
    assert by["octree"]["status"] == "described"        # nessuna esecuzione completata nel KB
    assert by["fair-smote"]["tags"].count("fairness") == 1
    assert any(a["status"] == "described" for a in c["algorithms"])


def test_ts_graph_catalog_does_not_crash():
    for m in ("ts", "graph"):
        c = catalog.get_catalog(m)
        assert isinstance(c["algorithms"], list)
        if not c["algorithms"]:
            assert c["warnings"]


def test_recommend_tabular_real_data():
    r = recommender.recommend("tabular", dataset="iris0_min", top_k=3)
    assert r["ranking"], r
    for item in r["ranking"]:
        assert item["status"] in ("executable", "wired")
        assert all(e["source"] for e in item["evidence"])
        assert 0.0 <= item["score"] <= 1.0
    assert r["rules"] and r["scope"] == "dataset iris0"
    assert all(u["algorithm"] for u in r["unranked"])
    assert not any(u["reason"].startswith("no fidelity/diversity result") and u["status"] == "described" for u in r["unranked"])


def test_recommend_never_ranks_described(real_kb):
    r = recommender.recommend("tabular", top_k=100)
    desc = {a["key"] for a in catalog.get_catalog("tabular")["algorithms"] if a["status"] == "described"}
    assert desc and not (desc & {x["algorithm"] for x in r["ranking"]})
    assert desc & {u["algorithm"] for u in r["unranked"]}


def test_recommend_fairness_filter():
    r = recommender.recommend("tabular", need_fairness=True, top_k=50)
    got = {x["algorithm"] for x in r["ranking"]}
    assert got and got <= {"fair-smote", "tabfairgan", "prefair"}
    assert any(e["algorithm"] == "smote" for e in r["excluded"])


def test_recommend_time_budget_and_unmeasured_warning():
    r = recommender.recommend("tabular", dataset="iris0", time_budget_s=0.0001, top_k=50)
    assert not r["ranking"] and r["excluded"] and "message" in r


def test_recommend_modality_empty_or_unranked():
    r = recommender.recommend("ts", executable_only=True)
    assert "ranking" in r and "unranked" in r
    # nessun CSV in runtime_results_ts => nessuna classifica inventata
    if evidence.modality_runtime("ts").empty:
        assert r["ranking"] == [] and r["unranked"]
        assert "message" in r


def test_compare_and_explain_tabular_real():
    c = dispatch(ToolContext(), "compare_results", {"modality": "tabular", "dataset": "iris0"})
    assert c["n_rows"] > 0 and all("sources" in x for x in c["rows"])
    row = c["rows"][0]
    e = dispatch(ToolContext(), "explain_metrics", {"modality": "tabular", "dataset": "iris0", "algorithm": row["algorithm"],
                                                    "claimed_values": {"JS_mean": row["JS_mean"], "KS_mean": 99.0}})
    assert e["metrics"]["JS_mean"]["claimed_value_matches"] is True
    assert e["metrics"]["KS_mean"]["claimed_value_matches"] is False and "MISMATCH" in e
    assert e["metrics"]["JS_mean"]["source"].endswith("fidelity_all_datasets.csv")
    e2 = dispatch(ToolContext(), "explain_metrics", {"modality": "tabular", "dataset": "iris0", "algorithm": "inesistente"})
    assert "error" in e2


# ---------------------------------------------------------------- analisi dataset
def test_analyze_tabular():
    r = datasets_tools.analyze_dataset("tabular", "iris0_min")
    assert r["rows"] > 0 and r["columns"] == 5 and r["label_column"] == "class"
    assert r["n_classes"] == 1 and "note" in r and "imbalance_ratio" not in r and r["missing_values_total"] == 0
    r2 = datasets_tools.analyze_dataset("tabular", "adult_min")
    assert r2["rows"] > 1000 and r2["categorical_columns"] >= 0
    assert r["file"] == "datasets/iris0_min.csv"


def test_analyze_ts_and_graph():
    r = datasets_tools.analyze_dataset("ts", "sine_small")
    assert (r["series"], r["length"], r["channels"]) == (256, 24, 5)
    g = datasets_tools.analyze_dataset("graph", "cora")
    assert g["kind"] == "node" and g["nodes"] > 0 and g["edges"] > 0 and g["n_classes"] > 1
    m = datasets_tools.analyze_dataset("graph", "mutag")
    assert m["kind"] == "graph" and m["graphs"] > 0


def test_list_datasets():
    assert len(datasets_tools.list_datasets("tabular")["datasets"]) > 10
    assert "sine_small" in [d["name"] for d in datasets_tools.list_datasets("ts")["datasets"]]


# ---------------------------------------------------------------- ciclo agentico (client simulato)
def test_agent_loop_tool_use_and_result_pairing():
    s = session_with_tm()
    mock = MockClient([[txt("Guardo."), tu("a1", "list_datasets", modality="ts")],
                       [txt("Hai sine_small (fonte: list_datasets).")]])
    evs = list(Agent(mock).run_turn(s, "che dataset ts ci sono?"))
    types = [e["type"] for e in evs]
    assert types[:3] == ["text", "tool_call", "tool_result"] and types[-1] == "done"
    assert evs[2]["ok"] is True
    # il secondo giro ha ricevuto tool_result con lo stesso id
    last = mock.calls[1]["messages"][-1]
    assert last["role"] == "user" and last["content"][0]["tool_use_id"] == "a1"
    assert "sine_small" in last["content"][0]["content"]
    assert mock.calls[0]["tools"] and "ANTI-INVENTION" in mock.calls[0]["system"]


def test_agent_unknown_tool_and_bad_args_are_errors_not_crashes():
    s = session_with_tm()
    mock = MockClient([[tu("x", "rm_rf", path="/")], [tu("y", "analyze_dataset", modality="tabular", dataset="../../x")],
                       [txt("ok")]])
    evs = list(Agent(mock).run_turn(s, "prova"))
    res = [e for e in evs if e["type"] == "tool_result"]
    assert [r["ok"] for r in res] == [False, False]
    assert evs[-1]["type"] == "done"


def test_agent_step_limit():
    s = session_with_tm()
    mock = MockClient([[tu(f"i{i}", "list_datasets", modality="ts")] for i in range(30)])
    evs = list(Agent(mock, limits={"max_steps": 3}).run_turn(s, "loop"))
    assert any(e["type"] == "limit" for e in evs) and evs[-1]["steps"] == 3
    assert len(mock.calls) == 3


def test_agent_token_limit():
    from assistant.client import ModelResponse
    s = session_with_tm()
    big = ModelResponse([tu("a", "list_datasets", modality="ts")], "tool_use", 90_000, 10)
    evs = list(Agent(MockClient([big, big]), limits={"max_total_tokens": 1000}).run_turn(s, "x"))
    assert any(e["type"] == "limit" and "token" in e["text"] for e in evs)


def test_agent_time_limit():
    s = session_with_tm()
    mock = MockClient([[tu("a", "list_datasets", modality="ts")], [txt("mai")]])
    evs = list(Agent(mock, limits={"max_seconds": -1}).run_turn(s, "x"))
    assert any(e["type"] == "limit" and "time" in e["text"] for e in evs)


def test_agent_api_error_is_explicit_and_history_restored():
    s = session_with_tm()

    def boom(_m):
        raise RuntimeError("API giu'")
    evs = list(Agent(MockClient([boom])).run_turn(s, "ciao"))
    assert evs[-1]["type"] == "error" and "API giu'" in evs[-1]["text"]
    assert s.messages == []


def test_grounding_warning():
    s = session_with_tm()
    mock = MockClient([[tu("a", "analyze_dataset", modality="tabular", dataset="iris0_min")],
                       [txt("Il dataset ha 12345 righe e KS 0.987654.")]])
    evs = list(Agent(mock).run_turn(s, "analizza"))
    w = [e for e in evs if e["type"] == "warning"]
    assert w and "12345" in w[0]["text"] and "0.987654" in w[0]["text"]
    assert ungrounded_numbers("righe 150 e 3 classi", [150.0]) == []


# ---------------------------------------------------------------- conferma dell'esecuzione
def _run_args(**kw):
    a = {"modality": "tabular", "algorithms": ["smote"], "datasets": ["iris0_min"], "epochs": 3}
    a.update(kw)
    return a


def test_run_single_small_starts_without_confirmation():
    ctx = ToolContext(task_manager=FakeTM())
    r = dispatch(ctx, "run_augmentation", _run_args())
    assert r["status"] == "started" and ctx.task_manager.started == ["t0"]
    cmd = ctx.task_manager.created[0][1]
    assert "main.py" in cmd[1] and "smote" in cmd and "iris0_min" in cmd


@pytest.mark.parametrize("kw", [{"algorithms": ["smote", "tvae"]}, {"epochs": 6}, {"datasets": ["iris0_min", "pima_min"]}])
def test_run_requires_confirmation(kw):
    ctx = ToolContext(task_manager=FakeTM())
    r = dispatch(ctx, "run_augmentation", _run_args(**kw))
    assert r["status"] == "needs_confirmation" and ctx.task_manager.started == []
    # il modello non puo' aggirare: richiamare lo strumento non lancia nulla
    r2 = dispatch(ctx, "run_augmentation", _run_args(**kw))
    assert r2["status"] == "needs_confirmation" and ctx.task_manager.started == []
    # la conferma umana (rotta HTTP) lancia
    out = execute_pending(ctx, r["pending_id"], True)
    assert out["status"] == "started" and len(ctx.task_manager.started) == 1
    assert "error" in execute_pending(ctx, r["pending_id"], True)  # una sola volta


def test_run_rejected_launches_nothing():
    ctx = ToolContext(task_manager=FakeTM())
    r = dispatch(ctx, "run_augmentation", _run_args(epochs=50))
    assert execute_pending(ctx, r["pending_id"], False)["status"] == "rejected"
    assert ctx.task_manager.started == []


def test_run_refuses_described_unknown_and_busy(real_kb):
    ctx = ToolContext(task_manager=FakeTM())
    desc = next(a["key"] for a in catalog.get_catalog("tabular")["algorithms"] if a["status"] == "described")
    assert "error" in dispatch(ctx, "run_augmentation", _run_args(algorithms=[desc]))
    assert "error" in dispatch(ctx, "run_augmentation", _run_args(algorithms=["nope"]))
    assert "error" in dispatch(ctx, "run_augmentation", _run_args(datasets=["../secret"]))
    assert "error" in dispatch(ctx, "run_augmentation", _run_args(device="rm -rf"))
    ctx.task_manager.active = FakeTask("busy")
    assert "already" in dispatch(ctx, "run_augmentation", _run_args())["error"]
    assert ctx.task_manager.started == []


def test_agent_pending_event_and_task_status():
    s = session_with_tm()
    mock = MockClient([[tu("r", "run_augmentation", **_run_args(algorithms=["smote", "tvae"]))],
                       [txt("Serve la tua conferma nella UI.")]])
    evs = list(Agent(mock).run_turn(s, "lancia tutto"))
    p = [e for e in evs if e["type"] == "pending"]
    assert len(p) == 1 and s.ctx.task_manager.started == []
    st = dispatch(s.ctx, "get_task_status", {"task_id": "t0"})
    assert st["status"] == "RUNNING" and st["log_tail"] == ["riga1", "riga2"]
    assert "error" in dispatch(s.ctx, "get_task_status", {"task_id": "zzz"})


def test_evaluate_requires_existing_outputs():
    ctx = ToolContext(task_manager=FakeTM())
    r = dispatch(ctx, "evaluate", {"modality": "tabular", "dataset": "iris0_min", "algorithms": ["castgan"]})
    if "error" in r:
        assert "missing" in r or "do not exist" in r["error"] or "already" in r["error"]
    else:
        assert "evaluate_metrics.py" in ctx.task_manager.created[0][1][1]


# ---------------------------------------------------------------- blueprint Flask
@pytest.fixture()
def app_client(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "KEY_FILE", tmp_path / "nonesiste.json")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_assistant_server import make_app
    from web_ui import assistant_bp
    assistant_bp._sessions.clear()
    assistant_bp._own_tm = FakeTM()
    assistant_bp.set_agent(None)
    yield make_app().test_client(), assistant_bp
    assistant_bp.set_agent(None)


def test_bp_page_and_offline_status(app_client):
    c, _ = app_client
    assert c.get("/assistant").status_code == 200
    st = c.get("/api/assistant/status").get_json()
    assert st["mode"] == "offline" and "tools" in st and "run_augmentation" in st["tools"]


def test_bp_offline_chat_declares_offline_and_tools_still_work(app_client):
    c, _ = app_client
    ev = c.post("/api/assistant/chat?stream=0", json={"message": "ciao"}).get_json()["events"]
    assert any(e["type"] == "offline" and "OFFLINE" in e["text"] for e in ev)
    assert not any(e["type"] == "text" for e in ev)
    ev = c.post("/api/assistant/chat?stream=0", json={"offline_action": {"tool": "recommend_algorithms",
                                                                     "args": {"modality": "tabular", "dataset": "iris0"}}}).get_json()["events"]
    res = [e for e in ev if e["type"] == "tool_result"][0]
    assert res["ok"] and res["result"]["ranking"]


def test_bp_online_mock_stream_and_confirm(app_client):
    c, bpm = app_client
    script = [[tu("r", "run_augmentation", **_run_args(algorithms=["smote", "tvae"]))], [txt("Premi conferma.")],
              [txt("Task partito.")]]
    bpm.set_agent(Agent(MockClient(script)))
    r = c.post("/api/assistant/chat", json={"message": "lancia"})  # stream SSE
    body = r.get_data(as_text=True)
    evs = [json.loads(l[6:]) for l in body.split("\n\n") if l.startswith("data: ")]
    sid = evs[0]["session_id"]
    pend = [e for e in evs if e["type"] == "pending"][0]
    assert bpm._own_tm.started == []
    j = c.post("/api/assistant/confirm", json={"session_id": sid, "pending_id": pend["pending_id"], "approve": True}).get_json()
    assert j["status"] == "started" and bpm._own_tm.started == ["t0"]
    ev2 = c.post("/api/assistant/chat?stream=0", json={"session_id": sid, "message": "fatto?"}).get_json()["events"]
    assert any(e["type"] == "text" for e in ev2)
    assert c.post("/api/assistant/confirm", json={"session_id": "x", "pending_id": "y", "approve": True}).status_code == 404


def test_bp_validation(app_client):
    c, _ = app_client
    ev = c.post("/api/assistant/chat?stream=0", json={"message": ""}).get_json()["events"]
    assert any(e["type"] == "error" for e in ev)
    ev = c.post("/api/assistant/chat?stream=0", json={"message": "x" * 5000}).get_json()["events"]
    assert any(e["type"] == "error" for e in ev)
