"""I conteggi del catalogo (UI e assistente) devono coincidere con il paper e con il KB finale reale.

Paper: 106 algoritmi = 20 tabelle + 41 serie temporali + 45 grafi; eseguibili 97 = 17 + 39 + 41; descritti 9.
Eseguibile = almeno una esecuzione SUCCESS/FALLBACK nel KB (assistant/kb/benchmark.csv).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant import catalog  # noqa: E402
from assistant.kb import store  # noqa: E402

TOTAL = {"tabular": 20, "ts": 41, "graph": 45}
EXECUTABLE = {"tabular": 17, "ts": 39, "graph": 41}


@pytest.fixture(autouse=True)
def _kb_reale(monkeypatch):
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(store.DEFAULT_DIR))


def test_assistant_catalog_counts_match_paper():
    exe = 0
    for mod, n in TOTAL.items():
        algos = catalog.get_catalog(mod)["algorithms"]
        assert len(algos) == n, mod
        assert len({a["key"] for a in algos}) == n, mod
        got = sum(a["status"] == "executable" for a in algos)
        assert got == EXECUTABLE[mod], (mod, got)
        assert all(a["status"] in ("executable", "described") for a in algos), mod
        exe += got
    assert sum(TOTAL.values()) == 106 and exe == 97


def test_tabular_catalog_is_core_algorithms_and_follows_kb():
    import config as root_config
    algos = {a["key"]: a for a in catalog.get_catalog("tabular")["algorithms"]}
    assert set(algos) == set(root_config.CORE_ALGORITHMS)
    assert algos["goggle"]["status"] == "executable" and algos["kamino"]["status"] == "executable"
    for k in ("octree", "harmonic", "synthetic-llm"):
        assert algos[k]["status"] == "described"
    for mod in TOTAL:
        done = store.completed_algos(mod)
        for a in catalog.get_catalog(mod)["algorithms"]:
            assert (a["status"] == "executable") == (store.canon_algo(a["key"]) in done), (mod, a["key"])


def test_ui_api_modalities_match_paper():
    from web_ui import app as webapp
    client = webapp.app.test_client()
    mods = {m["id"]: m for m in client.get("/api/modalities").get_json()}
    for mod, n in TOTAL.items():
        c = mods[mod]["counts"]
        assert c["total"] == n and c["executable"] == EXECUTABLE[mod], (mod, c)
        assert c["wired"] == 0 and c["described"] == n - EXECUTABLE[mod], (mod, c)
    assert sum(m["counts"]["executable"] for m in mods.values()) == 97


def test_ui_tabular_runnable_only_when_executable_and_described_refused():
    from web_ui import app as webapp
    client = webapp.app.test_client()
    algos = client.get("/api/modality/tabular/algorithms").get_json()["algorithms"]
    assert len(algos) == 20 and "sydc" not in {a["key"] for a in algos}
    for a in algos:
        assert a["runnable"] == (a["status"] == "executable")
    listed = {a["id"] for a in client.get("/api/algorithms").get_json()}
    assert listed == {a["key"] for a in algos if a["runnable"]} and len(listed) == 17
    r = client.post("/api/run_augmentation", json={"algorithms": ["octree"], "datasets": ["iris0_min"]})
    assert r.status_code == 400 and "Described-only" in r.get_json()["error"]


def test_results_all_include_kb_rows(_kb_reale):
    """Le schede del Catalogo contano le esecuzioni della knowledge base (file 'benchmark_kb')."""
    from web_ui.app import app
    c = app.test_client()
    for mod, n_algos in (("tabular", 17), ("ts", 39), ("graph", 41)):
        resp = c.get(f"/api/modality/{mod}/results?all=1")
        assert b"NaN" not in resp.data, mod  # NaN non e' JSON valido: il browser rifiuterebbe la risposta
        recs = resp.get_json()["records"]
        kb = [r for r in recs if r.get("file") == "benchmark_kb"]
        assert kb, mod
        assert len({r["algorithm"] for r in kb if str(r["status"]).upper() in ("SUCCESS", "FALLBACK")}) == n_algos


def test_described_only_matches_kb_and_is_excluded_from_all():
    """`config.DESCRIBED_ONLY` (saltati da `main.py --algorithms all`) = i tabulari senza esecuzione nel KB finale."""
    import ast
    cfg = ast.parse((ROOT / "config.py").read_text(encoding="utf-8"))
    consts = {t.id: ast.literal_eval(n.value) for n in cfg.body if isinstance(n, ast.Assign)
              for t in n.targets if getattr(t, "id", None) in ("CORE_ALGORITHMS", "DESCRIBED_ONLY")}
    described = {a["key"] for a in catalog.get_catalog("tabular")["algorithms"] if a["status"] == "described"}
    assert set(consts["DESCRIBED_ONLY"]) == described == {"octree", "harmonic", "synthetic-llm"}
    assert set(consts["DESCRIBED_ONLY"]) < set(consts["CORE_ALGORITHMS"])
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "a not in DESCRIBED_ONLY" in src  # 'all' expands to the runnable algorithms only
