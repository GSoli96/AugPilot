"""Pagina «Prove», caricamento di un dataset proprio, esecuzioni ts/graph in cartelle dedicate e sezione «Benchmark distribuito».
Nessun file reale viene scritto: le cartelle utente puntano a cartelle temporanee (config.USER_DIRS) e il KB e' finto."""
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant import config, datasets_tools, evidence  # noqa: E402
from assistant import tools as atools  # noqa: E402


@pytest.fixture()
def client():
    from web_ui import app as webapp
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


@pytest.fixture()
def userdirs(tmp_path, monkeypatch):
    d = {"tabular": str(tmp_path / "du"), "ts": str(tmp_path / "dut")}
    monkeypatch.setattr(config, "USER_DIRS", d)
    return d


def csv_bytes(n=210, ratio=2.2, cols=5, seed=0):
    rng = np.random.default_rng(seed)
    n1 = int(n / (1 + ratio))
    y = np.array([1] * n1 + [0] * (n - n1))
    x = rng.normal(size=(n, cols))
    lines = [",".join([f"f{i}" for i in range(cols)] + ["class"])]
    for i in range(n):
        lines.append(",".join(f"{v:.4f}" for v in x[i]) + f",{y[i]}")
    return ("\n".join(lines) + "\n").encode()


def npz_bytes(n=40, t=12, d=3, labels=True, ndim=3):
    buf = io.BytesIO()
    shape = (n, t, d)[:ndim] if ndim != 3 else (n, t, d)
    arrs = {"data": np.random.default_rng(0).normal(size=shape).astype("float32")}
    if labels:
        arrs["labels"] = np.arange(n) % 2
    np.savez(buf, **arrs)
    return buf.getvalue()


def up(client, mod, name, raw, **kw):
    return client.post("/api/datasets/upload", data={"modality": mod, "file": (io.BytesIO(raw), name)},
                       content_type="multipart/form-data", **kw)


# ---------------------------------------------------------------- caricamento
def test_upload_csv_valido_e_visibile_agli_strumenti(client, userdirs):
    r = up(client, "tabular", "Il mio dataset (1).csv", csv_bytes())
    assert r.status_code == 201, r.get_data(as_text=True)
    d = r.get_json()["dataset"]
    assert d["name"] == "Il_mio_dataset_1" and d["user"] is True and d["validation"]["n_classes"] == 2
    assert (Path(userdirs["tabular"]) / "Il_mio_dataset_1.csv").is_file()
    lst = atools.dispatch(atools.ToolContext(), "list_datasets", {"modality": "tabular"})
    mine = [x for x in lst["datasets"] if x["name"] == "Il_mio_dataset_1"]
    assert mine and mine[0]["user"] is True
    an = atools.dispatch(atools.ToolContext(), "analyze_dataset", {"modality": "tabular", "dataset": "Il_mio_dataset_1"})
    assert an["rows"] == 210 and an["label_column"] == "class" and an["user"] is True
    api = client.get("/api/modality/tabular/datasets").get_json()["datasets"]
    assert any(x["stem"] == "Il_mio_dataset_1" and x.get("user") for x in api)
    # la dashboard classica (lancia main.py senza --datasets-dir) non li vede
    assert all(not x.get("user") for x in client.get("/api/datasets").get_json())


def test_upload_nome_pulito_e_nessuna_sovrascrittura(client, userdirs):
    a = up(client, "tabular", "../../evil name.csv", csv_bytes()).get_json()["dataset"]
    assert a["name"] == "evil_name" and Path(userdirs["tabular"], "evil_name.csv").is_file()
    b = up(client, "tabular", "evil_name.csv", csv_bytes(n=100)).get_json()["dataset"]
    assert b["name"] == "evil_name_2" and b["renamed"] is True
    assert Path(userdirs["tabular"], "evil_name.csv").read_bytes() == csv_bytes()      # il primo e' intatto
    # un nome gia' usato da un dataset del benchmark prende il suffisso (il benchmark avrebbe la precedenza)
    bench = sorted((ROOT / "datasets").glob("*.csv"))[0].stem
    c = up(client, "tabular", f"{bench}.csv", csv_bytes()).get_json()["dataset"]
    assert c["name"] == f"{bench}_2"
    assert not list(Path(userdirs["tabular"]).glob("*..*")) and not any(p.name.startswith(".") for p in Path(userdirs["tabular"]).iterdir())


@pytest.mark.parametrize("mod,name,raw,status", [
    ("tabular", "x.txt", b"a,b\n1,2\n3,4\n", 415),
    ("tabular", "x.csv", b"", 400),
    ("tabular", "x.csv", b"a\n1\n2\n", 400),                       # una sola colonna
    ("tabular", "x.csv", b"a,b\n1,2\n", 400),                      # una sola riga
    ("tabular", "x.csv", b"a,b\n\xff\xfe,1\n2,3\n", 400),          # non UTF-8
    ("tabular", "x.csv", b"a,b\n1,\n2,\n", 400),                   # classe vuota
    ("ts", "x.csv", b"a,b\n1,2\n", 415),
    ("ts", "x.npz", b"non un archivio", 400),
    ("graph", "x.pt", b"zzz", 400),
    ("altro", "x.csv", b"a,b\n1,2\n", 400),
])
def test_upload_non_valido_non_scrive_nulla(client, userdirs, mod, name, raw, status):
    r = up(client, mod, name, raw)
    assert r.status_code == status and r.get_json()["ok"] is False and r.get_json()["error"]
    assert not any(Path(v).exists() and list(Path(v).iterdir()) for v in userdirs.values())


def test_upload_npz_senza_data_o_con_forma_sbagliata(client, userdirs):
    buf = io.BytesIO(); np.savez(buf, x=np.zeros((3, 4, 5)))
    assert up(client, "ts", "a.npz", buf.getvalue()).status_code == 400
    assert up(client, "ts", "b.npz", npz_bytes(ndim=2) if False else _npz2d()).status_code == 400
    buf = io.BytesIO(); np.savez(buf, data=np.zeros((5, 4, 2)), labels=np.zeros(3))
    r = up(client, "ts", "c.npz", buf.getvalue())
    assert r.status_code == 400 and "labels" in r.get_json()["error"]


def _npz2d():
    buf = io.BytesIO(); np.savez(buf, data=np.zeros((5, 4))); return buf.getvalue()


def test_upload_troppo_grande(client, userdirs, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_MAX_BYTES", 1000)
    r = up(client, "tabular", "big.csv", csv_bytes())
    assert r.status_code == 413 and "too large" in r.get_json()["error"]
    assert not Path(userdirs["tabular"]).exists() or not list(Path(userdirs["tabular"]).iterdir())


def test_upload_origine_diversa_rifiutata(client, userdirs):
    r = up(client, "tabular", "x.csv", csv_bytes(), headers={"Origin": "http://altro.example"})
    assert r.status_code == 403


def test_upload_ts_visibile_e_risolvibile(client, userdirs, monkeypatch):
    r = up(client, "ts", "mia serie.npz", npz_bytes())
    assert r.status_code == 201, r.get_data(as_text=True)
    d = r.get_json()["dataset"]
    assert d["name"] == "mia_serie" and d["validation"]["series"] == 40 and d["validation"]["has_labels"] is True
    names = [x["name"] for x in atools.dispatch(atools.ToolContext(), "list_datasets", {"modality": "ts"})["datasets"]]
    assert "mia_serie" in names
    an = atools.dispatch(atools.ToolContext(), "analyze_dataset", {"modality": "ts", "dataset": "mia_serie"})
    assert an["series"] == 40 and an["channels"] == 3 and an["user"] is True
    # modalities/ts/datasets.py: resolve() trova anche la cartella utente (usata dal sottoprocesso di run_modality.py)
    from modalities.ts import datasets as tsd
    monkeypatch.setattr(tsd, "USER_DIR", Path(userdirs["ts"]))
    assert Path(tsd.resolve("mia_serie")).samefile(Path(userdirs["ts"]) / "mia_serie.npz")
    with pytest.raises(FileNotFoundError):
        tsd.resolve("inesistente_xyz")
    assert any(x.get("user") for x in client.get("/api/modality/ts/datasets").get_json()["datasets"])


def test_recommend_su_dataset_caricato_usa_knn_sulle_meta_feature(client, userdirs, monkeypatch, tmp_path):
    from test_recommender_kb import ALGOS, DATASETS, tab_meta, tab_row, write_kb
    rows = [tab_row(a, d, s, dsi=i) for i, d in enumerate(DATASETS) for a in ALGOS for s in (0, 1)]
    kb = write_kb(tmp_path / "kbdir", rows, tab_meta())
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(kb))
    up(client, "tabular", "mio.csv", csv_bytes(n=210, ratio=2.2, cols=5))
    ctx = atools.ToolContext()
    r = atools.dispatch(ctx, "recommend_algorithms", {"modality": "tabular", "dataset": "mio", "top_k": 4})
    assert r["evidence"] == "kb" and r["kb"]["mode"] == "knn", r.get("scope")
    assert r["similar_datasets"] and r["ranking"]
    assert any("dataset file analysis" in n for n in r["notes"])
    e = atools.dispatch(ctx, "explain_recommendation", {"modality": "tabular", "dataset": "mio", "priority": "utility"})
    assert "error" not in e and e["similar_datasets"] and abs(e["contributions_sum"] - e["score"]) < 1e-5


# ---------------------------------------------------------------- esecuzione
def test_esecuzione_tabulare_su_dataset_utente_usa_datasets_dir(client, userdirs):
    up(client, "tabular", "mio.csv", csv_bytes())
    v = atools._validate_run({"modality": "tabular", "algorithms": ["smote"], "datasets": ["mio"], "epochs": 1})
    assert v["user_data"] is True
    cmd = atools._build_cmd(v)
    assert cmd[cmd.index("--datasets-dir") + 1] == userdirs["tabular"]
    bench = sorted((ROOT / "datasets").glob("*.csv"))[0].stem
    vb = atools._validate_run({"modality": "tabular", "algorithms": ["smote"], "datasets": [bench], "epochs": 1})
    assert "--datasets-dir" not in atools._build_cmd(vb)
    with pytest.raises(ValueError, match="do not mix"):
        atools._validate_run({"modality": "tabular", "algorithms": ["smote"], "datasets": ["mio", bench], "epochs": 1})


@pytest.mark.parametrize("mod,ds", [("ts", "sine_small"), ("graph", None)])
def test_run_modality_dallo_strumento_usa_cartelle_dedicate(mod, ds):
    v = {"modality": mod, "algorithms": ["x"], "datasets": ["d"], "epochs": 1, "n_samples": None, "device": "cpu"}
    cmd = atools._build_cmd(v)
    assert cmd[cmd.index("--out-dir") + 1].replace("\\", "/").endswith("/augmented_ui_" + mod)
    assert cmd[cmd.index("--results-dir") + 1].replace("\\", "/").endswith("/runtime_results_ui_" + mod)
    assert "augmented_ts/" not in " ".join(cmd) and "runtime_results_ts" not in " ".join(cmd)


def test_run_modality_dall_endpoint_usa_cartelle_dedicate(client, monkeypatch):
    from web_ui import app as webapp
    seen = {}

    class FakeTM:
        def get_active_task(self): return None
        def create_task(self, kind, cmd, cwd):
            seen["cmd"] = cmd
            return type("T", (), {"task_id": "t1"})()
        def start_task(self, t): pass
    monkeypatch.setattr(webapp, "task_manager", FakeTM())
    r = client.post("/api/run_modality", json={"modality": "ts", "algorithms": ["jitter"], "datasets": ["sine_small"], "epochs": 1, "device": "cpu"})
    assert r.status_code == 200, r.get_data(as_text=True)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--out-dir") + 1].replace("\\", "/").endswith("/augmented_ui_ts")
    assert cmd[cmd.index("--results-dir") + 1].replace("\\", "/").endswith("/runtime_results_ui_ts")


def test_run_modality_cli_scrive_nelle_cartelle_indicate(tmp_path):
    out, res = tmp_path / "o", tmp_path / "r"
    p = subprocess.run([sys.executable, str(ROOT / "run_modality.py"), "--modality", "ts", "--algorithms", "algo_che_non_esiste",
                        "--datasets", "sine_small", "--no_eval", "--out-dir", str(out), "--results-dir", str(res)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-500:]
    assert out.is_dir() and len(list(res.glob("runtime_*.csv"))) == 1


def test_risultati_ts_graph_leggono_anche_le_cartelle_della_ui(client, tmp_path, monkeypatch):
    from web_ui import app as webapp
    ui = tmp_path / "ui_res"; ui.mkdir()
    (ui / "runtime_ui.csv").write_text("dataset,algorithm,status,time_s\nsine_small,jitter,SUCCESS,1.5\n", encoding="utf-8")
    monkeypatch.setitem(webapp.MODALITIES["ts"], "ui_results_dir", ui)
    j = client.get("/api/modality/ts/results?all=1").get_json()
    assert "runtime_ui.csv" in j["files"] and any(x.get("algorithm") == "jitter" for x in j["records"])
    j2 = client.get("/api/modality/ts/results?file=runtime_ui.csv").get_json()
    assert j2["filename"] == "runtime_ui.csv" and j2["records"][0]["algorithm"] == "jitter"
    # e il consigliere/compare_results (assistant/evidence.py)
    d = tmp_path / "ev"; d.mkdir()
    (d / "runtime_x.csv").write_text("dataset,algorithm,status,time_s\nsine_small,jitter,SUCCESS,2.0\n", encoding="utf-8")
    monkeypatch.setattr(config, "UI_RES_DIRS", {"ts": str(d), "graph": str(d)})
    df = evidence.modality_runtime("ts")
    assert (df["algo"] == "jitter").any()


# ---------------------------------------------------------------- pagina Prove
def test_pagina_prove_200_con_kb_vuoto(client):
    r = client.get("/evidence")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'href="/evidence" aria-current="page"' in html and "evidence.js" in html
    for mod in ("tabular", "ts", "graph"):
        j = client.get(f"/api/kb/summary?modality={mod}").get_json()     # KB assente (conftest): stato vuoto, non errore
        assert j["available"] is False and j["algorithms"] == [] and j["message"]
        assert set(j["profiles"]) >= {"balanced", "utility", "fidelity", "diversity", "privacy", "speed"}
    assert client.get("/api/kb/summary?modality=xx").status_code == 400


def test_pagina_prove_con_kb_finto_ha_dati_per_grafico_e_matrice(client, monkeypatch, tmp_path):
    from test_recommender_kb import ALGOS, DATASETS, tab_meta, tab_row, write_kb
    rows = [tab_row(a, d, s, dsi=i) for i, d in enumerate(DATASETS) for a in ALGOS for s in (0, 1)]
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(write_kb(tmp_path / "k", rows, tab_meta())))
    j = client.get("/api/kb/summary?modality=tabular").get_json()
    assert j["available"] and len(j["datasets"]) == 3 and len(j["algorithms"]) == 4 and len(j["matrix"]) == 12
    assert all(a["time_s_median"] is not None for a in j["algorithms"])


def test_statici_delle_nuove_funzioni_sono_serviti(client):
    for path, needle in [("/static/evidence/evidence.js", "api/kb/summary"), ("/static/site/shared.js", "explainPanel"),
                         ("/static/guided/guided.js", "explain_recommendation"), ("/static/copilot/copilot.js", "explain_recommendation"),
                         ("/static/site/site.css", ".evx-track")]:
        r = client.get(path)
        assert r.status_code == 200 and needle in r.get_data(as_text=True), path
    g = client.get("/static/guided/guided.js").get_data(as_text=True)
    for p in ("balanced", "utility", "fidelity", "diversity", "privacy", "speed"):
        assert p in g or "Evidence.PROFILES" in g
    assert "Upload.box" in g and "Upload.box" in client.get("/static/copilot/copilot.js").get_data(as_text=True)
    assert "For graphs there is no upload" in client.get("/static/site/shared.js").get_data(as_text=True)


# ---------------------------------------------------------------- Macchine: benchmark distribuito
FLEET = {"ts": 1794841693.0, "iso": "2026-11-16 16:08:13", "alerts": ["hpc: NON raggiungibile (ssh: connect to host <ip> port 22)", "x 10.1.2.3 bob@lab"],
         "machines": {
             "m1": {"name": "WSGrande", "reachable": True, "data": {
                 "machine": {"host": "SEGRETO-HOST", "python": "<home>\\python.exe", "free": False, "cpu_percent": 50.0,
                             "ram": {"used_gb": 20.0, "total_gb": 512.0}, "gpus": [{"name": "RTX 4090", "util": 10.0, "mem_used_mb": 100, "mem_total_mb": 24000}]},
                 "jobs": [{"id": "tabA1", "name": "Tab A1", "cmd": "python --pw SEGRETO-CMD", "log": ["riga SEGRETO-LOG"],
                           "status": {"state": "running", "attempts": 1, "restarts": 0, "updated_iso": "2026-11-16T16:07:39+01:00",
                                      "progress": {"done": 4, "total": 153, "pct": 2.6}}}],
                 "events": [{"ts": "2026-11-16T16:00:05+01:00", "job": "tabA1", "kind": "start"}]}},
             "m2": {"name": "Alienware", "inactive": True, "reachable": None},
             "m3": {"name": "Rotta", "reachable": False}}}


def test_fleet_con_file_finto(client, tmp_path, monkeypatch):
    (tmp_path / "fleet.json").write_text(json.dumps(FLEET), encoding="utf-8")
    monkeypatch.setenv("ORCH_STATE_DIR", str(tmp_path))
    r = client.get("/api/machines/fleet")
    assert r.status_code == 200
    raw = r.get_data(as_text=True)
    for secret in ("SEGRETO", "<ip>", "10.1.2.3", "bob@lab", "utente"):
        assert secret not in raw
    j = r.get_json()
    assert j["available"] is True and j["dashboard_url"] == "http://127.0.0.1:8765"
    by = {m["name"]: m for m in j["machines"]}
    assert by["WSGrande"]["state"] == "occupata" and by["WSGrande"]["gpus"][0]["name"] == "RTX 4090"
    job = by["WSGrande"]["jobs"][0]
    assert (job["done"], job["total"], job["state"]) == (4, 153, "running") and job["last_event"]["kind"] == "start"
    assert by["Alienware"]["state"] == "inattiva" and by["Rotta"]["state"] == "non raggiungibile"
    assert j["alerts"][0] == "hpc: NON raggiungibile"


def test_fleet_file_mancante_o_rotto(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_STATE_DIR", str(tmp_path / "nessuna_cartella"))
    j = client.get("/api/machines/fleet").get_json()
    assert j["available"] is False and "ORCH_STATE_DIR" in j["message"] and j["machines"] == []
    d = tmp_path / "rotto"; d.mkdir(); (d / "fleet.json").write_text("{non json", encoding="utf-8")
    monkeypatch.setenv("ORCH_STATE_DIR", str(d))
    j = client.get("/api/machines/fleet").get_json()
    assert j["available"] is False and "unreadable" in j["message"]


def test_pagina_macchine_ha_la_sezione_e_non_manda_comandi(client):
    html = client.get("/machines").get_data(as_text=True)
    assert "Distributed benchmark" in html and "http://127.0.0.1:8765" in html and "/api/machines/fleet" in html
    assert client.post("/api/machines/fleet").status_code == 405       # solo lettura
