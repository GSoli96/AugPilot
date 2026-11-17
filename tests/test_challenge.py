"""Pagina «Sfida» (assistant/kb/challenge.py e API /api/challenge/*): classifica vera, consiglio leave-one-out,
regret e casi limite. KB finto costruito qui con build_kb: nessun file reale del benchmark."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant import recommender  # noqa: E402
from assistant.kb import build_kb, challenge, scoring, store  # noqa: E402

# 5 algoritmi. Due gruppi di dataset: "piccoli" (dsA, dsC, dsE) dove ddpm e' il migliore per utilita',
# "grandi" (dsB, dsD) dove vince smote. Le altre metriche non cambiano tra i gruppi.
BASE = {
    "smote":   {"js": 0.05, "rec": 0.6, "dcr": 0.05, "t": 1.0},
    "tvae":    {"js": 0.01, "rec": 0.9, "dcr": 0.40, "t": 60.0},
    "ddpm":    {"js": 0.02, "rec": 0.8, "dcr": 0.30, "t": 400.0},
    "ctabgan": {"js": 0.03, "rec": 0.7, "dcr": 0.60, "t": 20.0},
    "goggle":  {"js": 0.04, "rec": 0.65, "dcr": 0.20, "t": 90.0},
}
F1_SMALL = {"smote": 0.05, "tvae": -0.01, "ddpm": 0.08, "ctabgan": 0.02, "goggle": 0.00}
F1_LARGE = {"smote": 0.09, "tvae": 0.01, "ddpm": 0.00, "ctabgan": 0.02, "goggle": 0.03}
SMALL, LARGE = ("dsA", "dsC", "dsE"), ("dsB", "dsD")
META = {  # n_rows, n_cols, ir, low_card_frac, n_min
    "dsA": (200, 5, 2.0, 0.10, 60), "dsC": (240, 6, 2.5, 0.20, 70), "dsE": (220, 5, 2.2, 0.15, 65),
    "dsB": (50000, 40, 30.0, 0.90, 400), "dsD": (60000, 45, 35.0, 0.85, 450),
    "dsX": (210, 5, 2.1, 0.10, 62),
}


def row(algo, ds, seed, f1, status="SUCCESS"):
    v = BASE[algo]
    r = {"id": f"tab|{algo}|{ds}|s{seed}|e5", "modality": "tabular", "algorithm": algo, "dataset": ds, "seed": seed,
         "epochs": 5, "status": status, "eval_status": "OK", "total_time_s": v["t"], "host": "test"}
    r.update({"fd_JS_mean": v["js"], "fd_Recall": v["rec"], "priv_DCR_median": v["dcr"], "util_f1_delta": f1})
    return r


def make_rows(f1_small=None, f1_large=None, extra=()):
    f1s, f1l = f1_small or F1_SMALL, f1_large or F1_LARGE
    rows = [row(a, d, s, (f1s if d in SMALL else f1l)[a]) for d in SMALL + LARGE for a in BASE for s in (0, 1)]
    return rows + list(extra)


def write_kb(tmp_path, rows, meta_names=None, name="kb"):
    res = tmp_path / f"results_{name}" / "m1"
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "a.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / name
    build_kb.build(res.parent, out, with_meta=False)
    cols = ["modality", "level", "dataset", "source", "n_rows", "n_cols", "ir", "low_card_frac", "n_min"]
    names = META if meta_names is None else meta_names
    pd.DataFrame([dict(zip(cols, ["tabular", "", n, "test", *META[n]])) for n in names]).to_csv(out / "dataset_meta.csv", index=False)
    return out


@pytest.fixture()
def kb(tmp_path):
    return store.load_kb(write_kb(tmp_path, make_rows()))


def manual_totals(kb, ds, profile):
    d = kb.rows("tabular")
    sc = scoring.score_dataset(d[d["_ds"] == ds], "tabular")
    avail = [ax for ax in scoring.ALL_AXES if any(s["axis_scores"].get(ax) is not None for s in sc.values())]
    w = recommender._weights_for(profile, avail)
    return {a: recommender._score_candidate(s, w)[0] for a, s in sc.items()}


# ---------------------------------------------------------------- classifica vera
@pytest.mark.parametrize("profile", list(recommender.PROFILES))
def test_true_ranking_matches_scoring(kb, profile):
    r = challenge.reveal(kb, "tabular", "dsA", profile, "smote")
    want = manual_totals(kb, "dsa", profile)
    assert r["status"] == "ok" and r["n_algorithms"] == 5 and r["profile"] == profile
    assert [x["algorithm"] for x in r["ranking"]] == sorted(want, key=lambda a: (-want[a], a))
    for x in r["ranking"]:
        assert x["score"] == pytest.approx(want[x["algorithm"]], abs=1e-5)
        assert sum(x["contributions"].values()) == pytest.approx(x["score"], abs=1e-5)
        assert set(x["contributions"]) == set(r["weights"])
    assert sum(r["weights"].values()) == pytest.approx(1.0, abs=1e-5)
    assert [x["rank"] for x in r["ranking"]] == sorted(x["rank"] for x in r["ranking"])
    assert r["ranking"][0]["rank"] == 1 and r["best"] == r["ranking"][0]["algorithm"]


def test_profile_changes_the_winner_and_guess_position(kb):
    u = challenge.reveal(kb, "tabular", "dsA", "utility", "ddpm")
    p = challenge.reveal(kb, "tabular", "dsA", "privacy", "ddpm")
    assert u["best"] == "ddpm" and u["guess"]["rank"] == 1 and u["guess_is_best"]
    assert p["best"] == "ctabgan" and p["guess"]["rank"] > 1 and not p["guess_is_best"]


def test_top_n_limits_rows_but_not_the_guess_position(kb):
    r = challenge.reveal(kb, "tabular", "dsA", "balanced", "smote", top_n=2)
    assert len(r["ranking"]) == 2 and r["n_algorithms"] == 5
    assert r["guess"]["rank"] == manual_rank(kb, "dsa", "balanced", "smote")


def manual_rank(kb, ds, profile, algo):
    t = manual_totals(kb, ds, profile)
    return 1 + sum(1 for v in t.values() if v > t[algo] + 1e-9)


# ---------------------------------------------------------------- consiglio leave-one-out
def test_advice_is_leave_one_out_and_equals_manual_knn(kb):
    r = challenge.reveal(kb, "tabular", "dsA", "utility", "tvae")
    a = r["advice"]
    assert r["status"] == "ok" and a is not None
    names = [n["dataset"].lower() for n in a["neighbors"]]
    assert "dsa" not in names and len(names) == scoring.K_NEIGHBORS
    # ricalcolo a mano: D tolto, k vicini, combine, stessi pesi
    d = kb.rows("tabular")
    others = sorted(set(d["_ds"]) - {"dsa"})
    q = scoring.meta_vector(kb, "tabular", "", "dsa")
    nb = scoring.knn_neighbors(kb, "tabular", "", q, others, k=scoring.K_NEIGHBORS)
    per = {n["dataset"]: scoring.score_dataset(d[d["_ds"] == n["dataset"]], "tabular") for n in nb}
    pred = scoring.combine([(n, per[n["dataset"]]) for n in nb])
    avail = [ax for ax in scoring.ALL_AXES if any(s["axis_scores"].get(ax) is not None for s in pred.values())]
    w = recommender._weights_for("utility", avail)
    tot = {x: recommender._score_candidate(s, w)[0] for x, s in pred.items()}
    assert a["algorithm"] == sorted(tot, key=lambda x: (-tot[x], x))[0]
    assert a["predicted_score"] == pytest.approx(max(tot.values()), abs=1e-5)
    assert a["rank"] == manual_rank(kb, "dsa", "utility", a["algorithm"])


def test_advice_does_not_use_the_dataset_itself(tmp_path):
    """Se si cambiano le misure di dsA (la verita') il consiglio per dsA non cambia: D non entra nel k-NN."""
    kb1 = store.load_kb(write_kb(tmp_path, make_rows(), name="k1"))
    flipped = dict(F1_SMALL, ddpm=-0.5, smote=0.5)
    rows2 = [row(a, "dsA", s, flipped[a]) for a in BASE for s in (0, 1)] + \
            [r for r in make_rows() if r["dataset"] != "dsA"]
    kb2 = store.load_kb(write_kb(tmp_path, rows2, name="k2"))
    r1 = challenge.reveal(kb1, "tabular", "dsA", "utility", "smote")
    r2 = challenge.reveal(kb2, "tabular", "dsA", "utility", "smote")
    assert r1["advice"]["algorithm"] == r2["advice"]["algorithm"] == "ddpm"
    assert r1["advice"]["predicted_score"] == r2["advice"]["predicted_score"]
    assert r1["best"] == "ddpm" and r2["best"] != "ddpm"          # la verita' invece cambia
    assert r1["advice"]["rank"] == 1 and r2["advice"]["rank"] > 1


def test_advice_follows_the_cluster_of_neighbors(kb):
    small = challenge.reveal(kb, "tabular", "dsA", "utility", "ddpm", k=2)
    large = challenge.reveal(kb, "tabular", "dsB", "utility", "smote", k=1)
    assert small["advice"]["algorithm"] == "ddpm" and {n["dataset"] for n in small["advice"]["neighbors"]} == {"dsC", "dsE"}
    assert large["advice"]["algorithm"] == "smote" and [n["dataset"] for n in large["advice"]["neighbors"]] == ["dsD"]
    assert abs(sum(n["weight"] for n in small["advice"]["neighbors"]) - 1.0) < 1e-5


# ---------------------------------------------------------------- regret e punteggio della sfida
def test_regret_extremes_and_outcome(kb):
    r_best = challenge.reveal(kb, "tabular", "dsA", "utility", "ddpm")
    assert r_best["guess"]["regret"] == 0.0 and r_best["outcome"]["visitor_regret"] == 0.0
    tot = manual_totals(kb, "dsa", "utility")
    worst = min(tot, key=lambda x: (tot[x], x))
    r_worst = challenge.reveal(kb, "tabular", "dsA", "utility", worst)
    assert r_worst["guess"]["regret"] == pytest.approx(1.0) and r_worst["guess"]["rank"] == 5
    # AugPilot sceglie ddpm (regret 0): contro il peggiore vince AugPilot, contro il migliore e' pari
    assert r_worst["advice"]["algorithm"] == "ddpm" and r_worst["outcome"]["result"] == "augpilot"
    assert r_best["outcome"]["result"] == "tie" and "same algorithm" in r_best["outcome"]["label"]
    assert r_best["outcome"]["chance_regret"] == pytest.approx(
        sum((max(tot.values()) - v) / (max(tot.values()) - min(tot.values())) for v in tot.values()) / 5, abs=1e-5)


def test_visitor_beats_augpilot_when_advice_is_wrong(tmp_path):
    flipped = dict(F1_SMALL, ddpm=-0.5, smote=0.5)
    rows = [row(a, "dsA", s, flipped[a]) for a in BASE for s in (0, 1)] + [r for r in make_rows() if r["dataset"] != "dsA"]
    kb = store.load_kb(write_kb(tmp_path, rows))
    r = challenge.reveal(kb, "tabular", "dsA", "utility", "smote")
    assert r["outcome"]["result"] == "visitor" and r["outcome"]["visitor_regret"] < r["outcome"]["augpilot_regret"]
    assert "better than AugPilot" in r["outcome"]["label"]


def test_regret_matches_e1_loo(kb):
    pytest.importorskip("scipy")
    from bench.experiments import e1_loo
    for ds in ("dsA", "dsB", "dsC"):
        for profile in ("balanced", "utility"):
            r = challenge.reveal(kb, "tabular", ds, profile, "smote")
            d = kb.rows("tabular")
            cn = store.canon(ds)
            others = sorted(set(d["_ds"]) - {cn})
            per = {o: scoring.score_dataset(d[d["_ds"] == o], "tabular") for o in set(d["_ds"])}
            nb = scoring.knn_neighbors(kb, "tabular", "", scoring.meta_vector(kb, "tabular", "", cn), others)
            pred = e1_loo.totals(scoring.combine([(n, per[n["dataset"]]) for n in nb]), profile)
            ev = e1_loo.evaluate_ranking(pred, e1_loo.totals(per[cn], profile))
            assert r["advice"]["regret"] == pytest.approx(ev["regret@1"], abs=1e-5)
            assert r["advice"]["algorithm"] == ev["top1_pred"]


def test_regret_and_rank_with_ties():
    vals = {"a": 0.5, "b": 0.5, "c": 0.5}
    assert challenge.regret(vals, "a") == 0.0 and challenge.chance_regret(vals) == 0.0
    assert challenge._rank(vals) == {"a": 1, "b": 1, "c": 1}
    assert challenge._rank({"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1}) == {"a": 1, "b": 2, "c": 2, "d": 4}
    assert challenge.regret({"a": 1.0, "b": 0.25, "c": 0.0}, "b") == pytest.approx(0.75)


# ---------------------------------------------------------------- casi limite
def test_too_few_algorithms_gives_no_challenge(tmp_path):
    extra = [row(a, "dsX", s, 0.0) for a in ("smote", "tvae") for s in (0, 1)]
    kb = store.load_kb(write_kb(tmp_path, make_rows(extra=extra)))
    r = challenge.reveal(kb, "tabular", "dsX", "balanced", "smote")
    assert r["status"] == "too_few_algorithms" and r["n_algorithms"] == 2
    assert "advice" not in r and "ranking" not in r and "at least 3" in r["message"]
    lst = {d["dataset"]: d for d in challenge.list_datasets(kb)["datasets"]}
    assert lst["dsX"]["playable"] is False and lst["dsX"]["advice_possible"] is False and "only 2" in lst["dsX"]["note"]
    assert lst["dsA"]["playable"] is True and lst["dsA"]["advice_possible"] is True and lst["dsA"]["note"] == ""
    assert challenge.candidates(kb, "tabular", "dsX")["playable"] is False


def test_dataset_without_meta_has_no_advice_but_shows_truth(tmp_path):
    extra = [row(a, "dsNoMeta", s, 0.01 * i) for i, a in enumerate(BASE) for s in (0, 1)]
    names = [n for n in META if n != "dsX"]
    kb = store.load_kb(write_kb(tmp_path, make_rows(extra=extra), meta_names=names))
    r = challenge.reveal(kb, "tabular", "dsNoMeta", "balanced", "smote")
    assert r["status"] == "no_advice" and r["advice"] is None and r["outcome"] is None
    assert "meta-feature" in r["message"] and "no advice is shown" in r["message"]
    assert len(r["ranking"]) == 5 and r["guess"]["rank"] >= 1
    lst = {d["dataset"]: d for d in challenge.list_datasets(kb)["datasets"]}
    assert lst["dsNoMeta"]["playable"] and not lst["dsNoMeta"]["advice_possible"]


def test_single_dataset_in_modality_has_no_neighbors(tmp_path):
    rows = [r for r in make_rows() if r["dataset"] == "dsA"]
    kb = store.load_kb(write_kb(tmp_path, rows, meta_names=["dsA"]))
    r = challenge.reveal(kb, "tabular", "dsA", "balanced", "smote")
    assert r["status"] == "no_advice" and "no other dataset" in r["message"] and r["advice"] is None
    assert r["n_algorithms"] == 5 and r["ranking"]


def test_few_common_algorithms_with_neighbors_gives_no_advice(tmp_path):
    """I vicini misurano solo 2 degli algoritmi misurati su D: non si confronta, lo si dichiara."""
    rows = [r for r in make_rows() if r["dataset"] == "dsA"]
    rows += [row(a, "dsC", s, 0.0) for a in ("smote", "tvae") for s in (0, 1)]
    kb = store.load_kb(write_kb(tmp_path, rows, meta_names=["dsA", "dsC"]))
    r = challenge.reveal(kb, "tabular", "dsA", "balanced", "smote")
    assert r["status"] == "no_advice" and "in common" in r["message"]


def test_failed_algorithm_is_ranked_last(tmp_path):
    rows = [r for r in make_rows() if not (r["dataset"] == "dsA" and r["algorithm"] == "ddpm")]
    rows += [row("ddpm", "dsA", s, 0.0, status="FAILED") for s in (0, 1)]
    kb = store.load_kb(write_kb(tmp_path, rows))
    r = challenge.reveal(kb, "tabular", "dsA", "utility", "ddpm")
    assert r["guess"]["rank"] == 5 and r["guess"]["n_failed"] == 2 and r["guess"]["failure_rate"] == 1.0
    assert r["guess"]["regret"] == pytest.approx(1.0)


def test_candidates_do_not_leak_scores(kb):
    c = challenge.candidates(kb, "tabular", "DSA")
    assert c["dataset"] == "dsA" and c["n_algorithms"] == 5 and c["playable"]
    assert [a["algorithm"] for a in c["algorithms"]] == sorted(BASE)
    assert all(set(a) == {"algorithm", "n_runs", "n_failed"} for a in c["algorithms"])


def test_list_datasets_counts(kb):
    out = challenge.list_datasets(kb)
    assert out["available"] and out["min_algorithms"] == 3 and out["k"] == 3
    assert set(out["profiles"]) == set(recommender.PROFILES)
    ds = {d["dataset"]: d for d in out["datasets"]}
    assert set(ds) == {"dsA", "dsB", "dsC", "dsD", "dsE"}
    assert all(d["modality"] == "tabular" and d["level"] == "" and d["n_algorithms"] == 5 and d["n_runs"] == 10 for d in ds.values())


def test_validation_errors(kb):
    with pytest.raises(challenge.ChallengeError, match="profile"):
        challenge.reveal(kb, "tabular", "dsA", "nope", "smote")
    with pytest.raises(challenge.ChallengeError, match="modality"):
        challenge.reveal(kb, "audio", "dsA", "balanced", "smote")
    with pytest.raises(challenge.ChallengeError) as e:
        challenge.reveal(kb, "tabular", "ghost", "balanced", "smote")
    assert e.value.status == 404
    with pytest.raises(challenge.ChallengeError, match="is not among the algorithms"):
        challenge.reveal(kb, "tabular", "dsA", "balanced", "inventato")
    with pytest.raises(challenge.ChallengeError, match="dataset"):
        challenge.candidates(kb, "tabular", "")
    with pytest.raises(challenge.ChallengeError) as e:
        challenge.reveal(None, "tabular", "dsA", "balanced", "smote")
    assert e.value.status == 404
    assert challenge.list_datasets(None)["available"] is False and challenge.list_datasets(None)["datasets"] == []


# ---------------------------------------------------------------- API e pagina
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(write_kb(tmp_path, make_rows(), name="kb_web")))
    from web_ui import app as webapp
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def test_api_datasets_candidates_reveal(client):
    j = client.get("/api/challenge/datasets").get_json()
    assert j["available"] and len(j["datasets"]) == 5 and "balanced" in j["profiles"]
    c = client.get("/api/challenge/candidates?modality=tabular&dataset=dsA").get_json()
    assert c["n_algorithms"] == 5
    r = client.post("/api/challenge/reveal", json={"modality": "tabular", "dataset": "dsA", "profile": "utility", "guess": "ddpm"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["best"] == "ddpm" and body["advice"]["algorithm"] == "ddpm"
    assert body["outcome"]["result"] == "tie" and len(body["ranking"]) == 5


def test_api_errors(client):
    assert client.get("/api/challenge/candidates?modality=tabular&dataset=ghost").status_code == 404
    assert client.get("/api/challenge/candidates?modality=audio&dataset=dsA").status_code == 400
    assert client.post("/api/challenge/reveal", data="no", content_type="text/plain").status_code == 400
    assert client.post("/api/challenge/reveal", json={"modality": "tabular", "dataset": "dsA", "profile": "x", "guess": "smote"}).status_code == 400
    assert client.post("/api/challenge/reveal", json={"modality": "tabular", "dataset": "dsA", "profile": "balanced", "guess": "zzz"}).status_code == 400


def test_api_without_kb(monkeypatch, tmp_path):
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(tmp_path / "vuoto"))
    from web_ui import app as webapp
    webapp.app.config["TESTING"] = True
    c = webapp.app.test_client()
    j = c.get("/api/challenge/datasets").get_json()
    assert j["available"] is False and j["datasets"] == [] and "build_kb" in j["message"]
    assert c.get("/api/challenge/candidates?modality=tabular&dataset=dsA").status_code == 404


def test_page_and_menu(client):
    html = client.get("/challenge").get_data(as_text=True)
    assert "Challenge" in html and 'href="/challenge"' in html and "challenge/challenge.js" in html
    assert 'aria-current="page"' in html
    assert 'href="/challenge"' in client.get("/evidence").get_data(as_text=True)      # voce di menu presente nelle altre pagine
    for path in ("/static/challenge/challenge.js", "/static/challenge/challenge.css"):
        assert client.get(path).status_code == 200
