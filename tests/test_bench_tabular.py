"""Test del protocollo tabulare del benchmark (WP2/WP4): split, privacy, utilità, estrazione, ripresa."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import run_tab_job  # noqa: E402
from bench.privacy import privacy_metrics  # noqa: E402
from bench.tabular_split import make_split, split_paths  # noqa: E402


@pytest.fixture()
def toy_full(tmp_path):
    rng = np.random.default_rng(0)
    n_maj, n_min = 120, 30
    df = pd.DataFrame({
        "Attr0": np.r_[rng.normal(0, 1, n_maj), rng.normal(2, 1, n_min)],
        "Attr1": np.r_[rng.integers(0, 3, n_maj), rng.integers(1, 4, n_min)],
        "class": [0] * n_maj + [1] * n_min,
    })
    full = tmp_path / "full"
    full.mkdir()
    df.to_csv(full / "toy.csv", index=False)
    return full, tmp_path / "work"


def test_split_has_no_overlap_and_min_from_train(toy_full):
    full, work = toy_full
    meta = make_split("toy", 0, full_dir=full, work_dir=work)
    p = split_paths("toy", 0, work_dir=work)
    tr, te, trm = pd.read_csv(p["train"]), pd.read_csv(p["test"]), pd.read_csv(p["train_min"])
    key = lambda d: set(map(tuple, d.round(8).values))  # noqa: E731
    assert not key(tr) & key(te)
    assert set(trm["class"]) == {meta["min_label"]}
    assert key(trm) <= key(tr)
    assert meta["n_to_generate"] == meta["n_maj_train"] - meta["n_min_train"]
    # idempotente: stesso risultato alla seconda chiamata
    assert make_split("toy", 0, full_dir=full, work_dir=work) == meta


def test_privacy_copies_vs_new_points():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(200, 3))
    test = rng.normal(size=(80, 3))
    copies = privacy_metrics(train, train[:50].copy(), test)
    assert copies["exact_copy_rate"] == 1.0 and copies["DCR_p5"] == 0.0
    fresh = privacy_metrics(train, rng.normal(size=(80, 3)), test)
    assert fresh["exact_copy_rate"] == 0.0
    assert 0.5 < fresh["DCR_ratio"] < 2.0


def test_utility_keys_and_delta(toy_full):
    from evaluate_utility import utility_metrics
    full, work = toy_full
    meta = make_split("toy", 0, full_dir=full, work_dir=work)
    p = split_paths("toy", 0, work_dir=work)
    tr, te, trm = pd.read_csv(p["train"]), pd.read_csv(p["test"]), pd.read_csv(p["train_min"])
    out = utility_metrics(tr, te, trm.copy(), "class", meta["min_label"], seed=0, models=["LR", "RF"])
    for k in ("util_f1_trtr", "util_f1_trs", "util_f1_delta", "LR_pr_auc_delta"):
        assert k in out
    assert out["util_f1_delta"] == pytest.approx(out["util_f1_trs"] - out["util_f1_trtr"])


def test_extract_synthetic_removes_real_rows(tmp_path):
    real = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "class": [1, 1, 1]})
    out = pd.concat([real.astype(float), pd.DataFrame({"a": [9.0, 8.0], "b": [9.5, 8.5], "class": [1.0, 1.0]}),
                     pd.DataFrame({"a": [0.0], "b": [0.0], "class": [0.0]})])
    f = tmp_path / "o.csv"
    out.to_csv(f, index=False)
    synth, info = run_tab_job.extract_synthetic(f, real, ["a", "b", "class"], "class", 1, n_target=5, seed=0)
    assert info["n_real_rows_removed"] == 3
    assert len(synth) == 2 and info["synth_shortfall"] == 3
    assert set(synth["a"]) == {9.0, 8.0}


def test_job_resume_skips_done(tmp_path, monkeypatch):
    calls = []

    def fake_run_one(algo, dataset, seed, epochs, *a, **k):
        calls.append((algo, dataset, seed))
        return {"status": "SUCCESS", "eval_status": "OK"}

    monkeypatch.setattr(run_tab_job, "run_one", fake_run_one)
    res = tmp_path / "r.jsonl"
    res.write_text(json.dumps({"id": run_tab_job.job_id("smote", "toy", 0, 3)}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--datasets", "toy", "--algorithms", "smote", "tvae",
                                      "--seeds", "0", "--epochs", "3", "--results-file", str(res), "--no-gpu"])
    monkeypatch.setenv("ORCH_JOB_DIR", str(tmp_path))
    assert run_tab_job.main() == 0
    assert calls == [("tvae", "toy", 0)]
    ids = [json.loads(line)["id"] for line in res.read_text(encoding="utf-8").splitlines()]
    assert ids == [run_tab_job.job_id("smote", "toy", 0, 3), run_tab_job.job_id("tvae", "toy", 0, 3)]
