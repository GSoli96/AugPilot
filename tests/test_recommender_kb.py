"""Test del consigliere basato sul KB (WP7): KB finto costruito qui, nessun file reale del benchmark."""
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant import recommender  # noqa: E402
from assistant.kb import build_kb, metrics_spec, scoring, store  # noqa: E402
from assistant.tools import ToolContext, dispatch  # noqa: E402

# ---------------------------------------------------------------- KB finto: 3 dataset, 4 algoritmi, 2 seed
# Valori per algoritmo scelti perche' i profili abbiano vincitori diversi (percentili con n=4: 1, 2/3, 1/3, 0):
#   fidelity (JS, basso)    tvae > ddpm > ctabgan > smote      diversity (Recall, alto)  idem
#   privacy (DCR, alto)     ctabgan > tvae > ddpm > smote      utility (F1 delta, alto) ddpm > smote > ctabgan > tvae
#   velocita' (tempo)       smote > ctabgan > tvae > ddpm
ALGOS = {
    "smote":   {"js": 0.05, "rec": 0.6, "dcr": 0.05, "f1": 0.05, "t": 1.0},
    "tvae":    {"js": 0.01, "rec": 0.9, "dcr": 0.40, "f1": -0.01, "t": 60.0},
    "ddpm":    {"js": 0.02, "rec": 0.8, "dcr": 0.30, "f1": 0.08, "t": 400.0},
    "ctabgan": {"js": 0.03, "rec": 0.7, "dcr": 0.60, "f1": 0.02, "t": 20.0},
}
DATASETS = ["dsA", "dsB", "dsC"]
META = {  # n_rows, n_cols, ir, low_card_frac, n_min
    "dsA": (200, 5, 2.0, 0.1, 60), "dsB": (50000, 40, 30.0, 0.9, 400), "dsC": (240, 6, 2.5, 0.2, 70),
    "dsNew": (210, 5, 2.2, 0.1, 62),
}


def tab_row(algo, ds, seed, status="SUCCESS", dsi=0):
    v = ALGOS[algo]
    r = {"id": f"tab|{algo}|{ds}|s{seed}|e5", "modality": "tabular", "algorithm": algo, "dataset": ds, "seed": seed,
         "epochs": 5, "status": status, "eval_status": "OK" if status != "FAILED" else "SKIPPED",
         "total_time_s": v["t"] * (1 + 0.1 * dsi), "host": "test"}
    if status != "FAILED":
        # piccolo scarto per dataset e seed che non cambia l'ordine tra algoritmi
        eps = 1e-4 * (dsi + seed)
        r.update({"fd_JS_mean": v["js"] + eps, "fd_Recall": v["rec"] - eps, "priv_DCR_median": v["dcr"] + eps,
                  "util_f1_delta": v["f1"] + eps, "fd_Compactness_cls_k": 0.5, "priv_DCR_holdout_p5": 0.2})
    return r


def write_kb(tmp_path, rows, meta=None):
    res = tmp_path / "results" / "m1"
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "a.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "kb"
    build_kb.build(tmp_path / "results", out, with_meta=False)
    if meta is not None:
        pd.DataFrame(meta).to_csv(out / "dataset_meta.csv", index=False)
    return out


def tab_meta(names=("dsA", "dsB", "dsC", "dsNew")):
    cols = ["modality", "level", "dataset", "source", "n_rows", "n_cols", "ir", "low_card_frac", "n_min"]
    return [dict(zip(cols, ["tabular", "", n, "test", *META[n]])) for n in names]


@pytest.fixture()
def kb_tab(tmp_path):
    rows = [tab_row(a, d, s, dsi=i) for i, d in enumerate(DATASETS) for a in ALGOS for s in (0, 1)]
    return write_kb(tmp_path, rows, tab_meta())


def top(r, n=None):
    return [x["algorithm"] for x in r["ranking"][:n]]


# ---------------------------------------------------------------- costruzione del KB
def test_build_kb_dedup_keeps_last_and_columns(tmp_path):
    a, b = tab_row("smote", "dsA", 0), tab_row("smote", "dsA", 0)
    b["fd_JS_mean"] = 0.5
    c = tab_row("tvae", "dsA", 0, status="FAILED")
    out = write_kb(tmp_path, [a, b, c])
    kb = store.load_kb(out)
    assert len(kb.bench) == 2
    assert kb.bench[kb.bench["algorithm"] == "smote"].iloc[0]["fd_JS_mean"] == 0.5
    assert {"modality", "level", "dataset", "algorithm", "seed", "epochs", "status", "eval_status", "time_s"} <= set(kb.bench.columns)
    assert kb.bench["time_s"].notna().all() and not any(c.startswith("util_f1_trs") for c in kb.bench.columns)


def test_metric_directions_match_source_modules():
    try:
        diff = metrics_spec.check_against_sources()
    except ImportError as e:  # moduli del benchmark non importabili in questo ambiente
        pytest.skip(str(e))
    assert diff == []
    assert metrics_spec.metric_spec("tabular", "fd_JS_mean") == ("lower", "fidelity")
    assert metrics_spec.metric_spec("tabular", "fd_Recall") == ("higher", "diversity")
    assert metrics_spec.metric_spec("tabular", "priv_DCR_median") == ("higher", "privacy")
    assert metrics_spec.metric_spec("tabular", "util_f1_trtr")[0] == "neutral"
    assert metrics_spec.metric_spec("ts", "m_PS") == ("lower", "utility")
    assert metrics_spec.metric_spec("ts", "m_clf_acc_delta") == ("higher", "utility")
    assert metrics_spec.metric_spec("graph", "m_delta_macro_f1") == ("higher", "utility")
    assert metrics_spec.metric_spec("graph", "m_struct_orig_nodes_mean")[0] == "neutral"


# ---------------------------------------------------------------- ordinamento, profili, contributi
@pytest.mark.parametrize("profile,winner", [("fidelity", "tvae"), ("utility", "ddpm"), ("privacy", "ctabgan"),
                                            ("speed", "smote"), ("balanced", "ddpm"), ("diversity", "tvae")])
def test_profiles_change_the_winner(kb_tab, profile, winner):
    r = recommender.recommend("tabular", dataset="dsA", priority=profile, top_k=10, kb_dir=kb_tab)
    assert r["evidence"] == "kb" and r["profile"] == profile
    assert top(r, 1) == [winner], [(x["algorithm"], x["score"]) for x in r["ranking"]]
    scores = [x["score"] for x in r["ranking"]]
    assert scores == sorted(scores, reverse=True)
    assert [x["rank"] for x in r["ranking"]] == list(range(1, len(r["ranking"]) + 1))


def test_contributions_sum_to_score_and_weights_normalized(kb_tab):
    for profile in recommender.PRIORITIES:
        for ds in ("dsA", "dsNew", None):
            r = recommender.recommend("tabular", dataset=ds, priority=profile, top_k=10, kb_dir=kb_tab)
            assert abs(sum(r["weights"].values()) - 1.0) < 1e-5
            for x in r["ranking"]:
                assert abs(sum(x["contributions"].values()) - x["score"]) < 1e-5
                assert set(x["contributions"]) == set(r["weights"])
                assert 0.0 <= x["score"] <= 1.0
                for ax, c in x["contributions"].items():
                    assert abs(c - r["weights"][ax] * (x["axis_scores"][ax] or 0.0)) < 2e-6
                assert {"score", "axis_scores", "contributions", "evidence", "warnings", "method"} <= set(x)


def test_profiles_declared_and_legacy_names():
    for w in recommender.PROFILES.values():
        assert abs(sum(w.values()) - 1.0) < 1e-9
    assert {"fidelity", "diversity", "balanced"} <= set(recommender.WEIGHTS) <= set(recommender.PROFILES)


def test_axis_scores_are_percentiles(kb_tab):
    r = recommender.recommend("tabular", dataset="dsA", top_k=10, kb_dir=kb_tab)
    by = {x["algorithm"]: x["axis_scores"] for x in r["ranking"]}
    assert by["tvae"]["fidelity"] == pytest.approx(1.0) and by["smote"]["fidelity"] == pytest.approx(0.0)
    assert by["ddpm"]["utility"] == pytest.approx(1.0) and by["tvae"]["utility"] == pytest.approx(0.0)
    assert by["smote"]["speed"] == pytest.approx(1.0) and by["ddpm"]["speed"] == pytest.approx(0.0)
    assert by["ctabgan"]["privacy"] == pytest.approx(1.0)
    ev = r["ranking"][0]["evidence"]
    assert any(e.get("dataset") == "dsA" and e["n_seeds"] == 2 and e["n_runs"] == 2 for e in ev)
    assert all(e["source"] for e in ev)


# ---------------------------------------------------------------- run FAILED
def test_failed_run_is_penalized(tmp_path):
    ok_rows = [tab_row(a, d, s, dsi=i) for i, d in enumerate(DATASETS) for a in ALGOS for s in (0, 1)]
    bad = [tab_row("ctabgan", "dsA", 1, status="FAILED", dsi=0) if (r["algorithm"], r["dataset"], r["seed"]) == ("ctabgan", "dsA", 1) else r
           for r in ok_rows]
    r_ok = recommender.recommend("tabular", dataset="dsA", top_k=10, kb_dir=write_kb(tmp_path / "ok", ok_rows, tab_meta()))
    r_bad = recommender.recommend("tabular", dataset="dsA", top_k=10, kb_dir=write_kb(tmp_path / "bad", bad, tab_meta()))
    s_ok = next(x for x in r_ok["ranking"] if x["algorithm"] == "ctabgan")
    s_bad = next(x for x in r_bad["ranking"] if x["algorithm"] == "ctabgan")
    assert s_bad["score"] < s_ok["score"] and s_bad["failure_rate"] == pytest.approx(0.5)
    assert any("FAILED" in w for w in s_bad["warnings"])
    assert s_ok["failure_rate"] == 0 and not any("FAILED" in w for w in s_ok["warnings"])
    assert any(e["failure_rate"] == pytest.approx(0.5) for e in s_bad["evidence"] if "failure_rate" in e)


def test_fallback_status_is_flagged(tmp_path):
    rows = [tab_row(a, d, s, dsi=i) for i, d in enumerate(DATASETS) for a in ALGOS for s in (0, 1)]
    for r in rows:
        if r["algorithm"] == "tvae":
            r["status"] = "FALLBACK"
    r = recommender.recommend("tabular", dataset="dsA", top_k=10, kb_dir=write_kb(tmp_path, rows, tab_meta()))
    assert any("FALLBACK" in w for w in next(x for x in r["ranking"] if x["algorithm"] == "tvae")["warnings"])


# ---------------------------------------------------------------- k-NN su dataset nuovo
def test_knn_picks_the_right_neighbours(kb_tab):
    kb = store.load_kb(kb_tab)
    q = dict(zip(["n_rows", "n_cols", "ir", "low_card_frac", "n_min"], META["dsNew"]))
    nb = scoring.knn_neighbors(kb, "tabular", "", q, ["dsA", "dsB", "dsC"], k=2)
    assert [n["dataset"] for n in nb] == ["dsA", "dsC"] and nb[0]["distance"] < nb[1]["distance"]
    assert sum(n["weight"] for n in nb) == pytest.approx(1.0) and nb[0]["weight"] > nb[1]["weight"]
    nb3 = scoring.knn_neighbors(kb, "tabular", "", q, ["dsA", "dsB", "dsC"], k=3)
    assert [n["dataset"] for n in nb3][-1] == "dsB"


def test_new_dataset_uses_weighted_neighbour_scores(tmp_path):
    # dsB si comporta all'opposto: i vicini veri (dsA, dsC) e il lontano (dsB) danno punteggi diversi
    rows = [tab_row(a, d, s, dsi=i) for i, d in enumerate(DATASETS) for a in ALGOS for s in (0, 1)]
    for r in rows:
        if r["dataset"] == "dsB":
            r["total_time_s"] = 1000.0 / r["total_time_s"]
            for k in ("util_f1_delta",):
                r[k] = -r[k]
    kb = write_kb(tmp_path, rows, tab_meta())
    rn = recommender.recommend("tabular", dataset="dsNew", priority="utility", top_k=10, kb_dir=kb)
    assert rn["evidence"] == "kb" and rn["kb"]["mode"] == "knn" and len(rn["similar_datasets"]) == 3
    sims = rn["similar_datasets"]
    assert [s["dataset"] for s in sims][-1] == "dsB" and sims[0]["distance"] <= sims[1]["distance"] <= sims[2]["distance"]
    per = {d: {x["algorithm"]: x["score"] for x in recommender.recommend("tabular", dataset=d, priority="utility", top_k=10, kb_dir=kb)["ranking"]}
           for d in DATASETS}
    w = {s["dataset"]: s["weight"] for s in sims}
    for x in rn["ranking"]:
        expect = sum(w[d] * per[d][x["algorithm"]] for d in DATASETS)
        assert x["score"] == pytest.approx(expect, abs=2e-5)
        assert x["method"] == "kb:knn" and {e["dataset"] for e in x["evidence"] if "dataset" in e} == set(DATASETS)
    assert "dsNew" in rn["scope"] and "nearest datasets" in rn["scope"]


def test_unknown_dataset_without_meta_averages_all(kb_tab):
    r = recommender.recommend("tabular", dataset="sconosciuto_xyz", top_k=10, kb_dir=kb_tab)
    assert r["evidence"] == "kb" and r["kb"]["mode"] == "all" and any("meta-feature" in n for n in r["notes"])


# ---------------------------------------------------------------- filtro di task (ts)
def ts_row(algo, ds="sine_small", seed=0, mdd=0.5, delta=0.0, t=1.0):
    return {"id": f"ts|{algo}|{ds}|s{seed}|e5", "modality": "ts", "algorithm": algo, "dataset": ds, "seed": seed,
            "epochs": 5, "status": "SUCCESS", "eval_status": "OK", "time_s": t, "level": None,
            "m_MDD": mdd, "m_clf_acc_delta": delta, "m_PS": mdd}


def test_task_filter_does_not_exclude_bootstrap_and_dba(tmp_path):
    rows = [ts_row("block_bootstrap", mdd=1.0, delta=0.01), ts_row("dba", mdd=0.9, delta=0.02),
            ts_row("jitter", mdd=0.1, delta=0.0), ts_row("smote", mdd=0.8, delta=0.03), ts_row("decomp_bootstrap", mdd=0.7)]
    kb = write_kb(tmp_path, rows)
    g = recommender.recommend("ts", task="generation", dataset="sine_small", top_k=50, kb_dir=kb)
    got = {x["algorithm"] for x in g["ranking"]}
    assert {"block_bootstrap", "dba", "smote", "decomp_bootstrap"} <= got
    assert "jitter" not in got and any(e["algorithm"] == "jitter" for e in g["excluded"])
    v = recommender.recommend("ts", task="view", dataset="sine_small", top_k=50, kb_dir=kb)
    assert {x["algorithm"] for x in v["ranking"]} == {"jitter"}
    allr = recommender.recommend("ts", dataset="sine_small", top_k=50, kb_dir=kb)
    assert {"jitter", "dba"} <= {x["algorithm"] for x in allr["ranking"]}
    # il dataset senza etichette/asse: assi privacy e diversity non esistono per ts e non compaiono nei pesi
    assert set(allr["weights"]) <= {"fidelity", "utility", "speed"}


def test_task_ok_rule():
    assert recommender.task_ok(None, "view") and recommender.task_ok("generation", "oversampling")
    assert recommender.task_ok("generation", "generation") and not recommender.task_ok("generation", "view")
    assert recommender.task_ok("view", "view") and not recommender.task_ok("view", "generation")
    assert not recommender.task_ok("oversampling", "generation")


def test_ts_catalog_taxonomy():
    import modalities
    t = {s.key: s.task for s in modalities.list_specs("ts")}
    for k in ("block_bootstrap", "decomp_bootstrap", "dba", "iaaft"):
        assert t[k] == "generation"
    assert t["smote"] == "oversampling" and t["jitter"] == "view"


# ---------------------------------------------------------------- fallback senza KB
def test_fallback_without_kb(tmp_path):
    r = recommender.recommend("tabular", dataset="iris0_min", top_k=3, kb_dir=tmp_path / "vuota")
    assert r["evidence"] == "nessuna" and "fallback" in r["method"] and r["ranking"]
    assert r["scope"] == "dataset iris0" and r["rules"]


def test_fallback_when_kb_does_not_cover_modality(kb_tab):
    r = recommender.recommend("ts", top_k=3, kb_dir=kb_tab)
    assert r["evidence"] == "nessuna"
    assert recommender.recommend("tabular", dataset="dsA", kb_dir=kb_tab)["evidence"] == "kb"


# ---------------------------------------------------------------- filtri e avvisi
def test_time_budget_and_gpu_filters_and_warnings(kb_tab):
    r = recommender.recommend("tabular", dataset="dsA", time_budget_s=100, top_k=10, kb_dir=kb_tab)
    assert "ddpm" not in top(r) and any(e["algorithm"] == "ddpm" and "budget" in e["reason"] for e in r["excluded"])
    full = recommender.recommend("tabular", dataset="dsA", top_k=10, kb_dir=kb_tab)
    dd = next(x for x in full["ranking"] if x["algorithm"] == "ddpm")
    assert any("high measured median time" in w for w in dd["warnings"])
    assert all(any("catalog" in w for w in x["warnings"]) for x in full["ranking"] if x["status"] != "executable")
    assert not recommender.recommend("tabular", dataset="dsA", need_fairness=True, kb_dir=kb_tab)["ranking"]


# ---------------------------------------------------------------- strumento di spiegazione
def test_explain_recommendation_tool(kb_tab, monkeypatch):
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(kb_tab))
    e = dispatch(ToolContext(), "explain_recommendation", {"modality": "tabular", "dataset": "dsNew", "priority": "utility"})
    assert "error" not in e, e
    assert e["algorithm"] == "ddpm" and e["rank"] == 1
    assert e["contributions_sum"] == pytest.approx(e["score"], abs=1e-5)
    assert len(e["comparison"]) == 2 and [c["rank"] for c in e["comparison"]] == [2, 3]
    for c in e["comparison"]:
        assert c["score_diff"] == pytest.approx(e["score"] - c["score"], abs=1e-5)
        assert sum(c["contribution_diff_by_axis"].values()) == pytest.approx(c["score_diff"], abs=1e-4)
    assert len(e["similar_datasets"]) == 3 and e["strongest_metrics"] and e["weakest_metrics"]
    assert all({"metric", "axis", "direction", "score"} <= set(m) for m in e["strongest_metrics"])
    e2 = dispatch(ToolContext(), "explain_recommendation", {"modality": "tabular", "dataset": "dsA", "algorithm": "smote"})
    assert e2["algorithm"] == "smote" and e2["rank"] > 1 and e2["similar_datasets"] == []
    assert any(m["metric"] == "fd_JS_mean" and "value" in m for m in e2["strongest_metrics"] + e2["weakest_metrics"])
    bad = dispatch(ToolContext(), "explain_recommendation", {"modality": "tabular", "dataset": "dsA", "algorithm": "nope"})
    assert "error" in bad
    assert "error" in dispatch(ToolContext(), "explain_recommendation", {"modality": "ts", "dataset": "x"})


def test_explain_recommendation_without_kb_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(tmp_path / "nessuno"))
    r = dispatch(ToolContext(), "explain_recommendation", {"modality": "tabular", "dataset": "iris0_min"})
    assert "error" in r and "KB" in r["error"]


def test_recommend_tool_accepts_new_priorities(kb_tab, monkeypatch):
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(kb_tab))
    r = dispatch(ToolContext(), "recommend_algorithms", {"modality": "tabular", "dataset": "dsA", "priority": "privacy"})
    assert r["profile"] == "privacy" and r["ranking"][0]["algorithm"] == "ctabgan"


# ---------------------------------------------------------------- API per la pagina «Prove»
def test_kb_summary_api(kb_tab, monkeypatch):
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(kb_tab))
    sys.path.insert(0, str(ROOT / "tests"))
    from test_assistant_server import make_app
    c = make_app().test_client()
    j = c.get("/api/kb/summary?modality=tabular").get_json()
    assert j["available"] and {d["dataset"] for d in j["datasets"]} == set(DATASETS)
    by = {a["algorithm"]: a for a in j["algorithms"]}
    assert set(by) == set(ALGOS) and by["smote"]["time_s_median"] == pytest.approx(1.1, rel=0.1)
    assert by["tvae"]["axis_scores"]["fidelity"] == pytest.approx(1.0) and len(j["matrix"]) == 12
    assert c.get("/api/kb/summary?modality=ts").get_json()["available"] is False
    assert c.get("/api/kb/summary?modality=zzz").status_code == 400
    assert c.get("/api/kb/summary").status_code == 400


# ---------------------------------------------------------------- esecuzioni da UI e assistente
def test_ui_runs_use_tabular_env_and_dedicated_dirs():
    from assistant import tools
    v = tools._validate_run({"modality": "tabular", "algorithms": ["smote"], "datasets": ["iris0_min"], "epochs": 2})
    cmd = tools._build_cmd(v)
    assert cmd[0] == tools.tabular_python() and cmd[1].endswith("main.py")
    assert "--no-resume" in cmd
    assert cmd[cmd.index("--augmented-dir") + 1].endswith("augmented_ui")
    assert cmd[cmd.index("--results-dir") + 1].endswith("runtime_results_ui")
    ctx = ToolContext()

    class TM:
        def get_active_task(self): return None
        def create_task(self, t, c, cwd):
            self.cmd = c
            return type("T", (), {"task_id": "t0"})()
        def start_task(self, t): pass
    ctx.task_manager = TM()
    out = dispatch(ctx, "run_augmentation", {"modality": "tabular", "algorithms": ["smote"], "datasets": ["iris0_min"], "epochs": 2})
    assert out["status"] == "started" and out["output_dirs"]["results"] == "runtime_results_ui"


def test_evaluate_reads_and_writes_ui_dirs(tmp_path, monkeypatch):
    from assistant import config, tools
    d = config.ROOT / config.UI_AUGMENTED_DIR / "smote"
    made = not d.exists()
    d.mkdir(parents=True, exist_ok=True)
    f = d / "iris0_min_augmented.csv"
    f_existed = f.exists()
    try:
        f.write_text("a,b\n1,2\n")
        ctx = ToolContext()

        class TM:
            def get_active_task(self): return None
            def create_task(self, t, c, cwd):
                self.cmd = c
                return type("T", (), {"task_id": "t0"})()
            def start_task(self, t): pass
        ctx.task_manager = TM()
        r = dispatch(ctx, "evaluate", {"modality": "tabular", "dataset": "iris0_min", "algorithms": ["smote"]})
        assert r["status"] == "started", r
        cmd = ctx.task_manager.cmd
        assert cmd[0] == tools.tabular_python() and cmd[1].endswith("evaluate_metrics.py")
        assert cmd[cmd.index("--augmented_dir") + 1].endswith("augmented_ui")
        assert cmd[cmd.index("--output_dir") + 1].endswith("evaluation_metrics_ui")
    finally:
        if not f_existed:
            f.unlink(missing_ok=True)
        if made:
            try:
                d.rmdir()
            except OSError:
                pass


def test_web_ui_run_endpoints_use_ui_dirs(monkeypatch):
    import web_ui.app as W

    class TM:
        def __init__(self): self.cmds = []
        def get_active_task(self): return None
        def create_task(self, t, c, cwd):
            self.cmds.append(c)
            return type("T", (), {"task_id": "t0"})()
        def start_task(self, t): pass
    tm = TM()
    monkeypatch.setattr(W, "task_manager", tm)
    c = W.app.test_client()
    assert c.post("/api/run_augmentation", json={"algorithms": ["smote"], "datasets": ["iris0_min"], "epochs": 1}).get_json()["success"]
    assert c.post("/api/run_evaluation", json={"dataset": "iris0_min", "algorithms": ["smote"]}).get_json()["success"]
    run, ev = tm.cmds
    assert run[0] == W._tabular_python() and "--no-resume" in run
    assert run[run.index("--augmented-dir") + 1].endswith("augmented_ui") and run[run.index("--results-dir") + 1].endswith("runtime_results_ui")
    assert ev[0] == W._tabular_python() and ev[ev.index("--output_dir") + 1].endswith("evaluation_metrics_ui")
    # resume esplicito: niente --no-resume (checkpoint comunque in runtime_results_ui/)
    c.post("/api/run_augmentation", json={"algorithms": ["smote"], "datasets": ["iris0_min"], "epochs": 1, "resume": True})
    assert "--no-resume" not in tm.cmds[-1]


def test_build_benchmark_riga_riuscita_vince_su_fallita_piu_recente():
    from assistant.kb.build_kb import build_benchmark
    base = {"id": "ts|tsgm-gan|energy|s1|e50", "modality": "ts", "algorithm": "tsgm-gan", "dataset": "energy",
            "seed": 1, "epochs": 50}
    ok = {**base, "status": "SUCCESS", "started": "2026-11-17T04:40:00", "time_s": 2800}
    ko = {**base, "status": "FAILED", "started": "2026-11-17T05:00:00", "time_s": 7200}
    for rows in ([ok, ko], [ko, ok]):
        df = build_benchmark(rows)
        assert len(df) == 1 and df.iloc[0]["status"] == "SUCCESS"
    # tra due fallite (o due riuscite) vince la più recente
    ko_old = {**ko, "started": "2026-11-16T10:00:00", "time_s": 1}
    df = build_benchmark([ko, ko_old])
    assert df.iloc[0]["time_s"] == 7200
