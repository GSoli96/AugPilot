"""Test analitici delle metriche tabulari (evaluate_metrics.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_metrics as em  # noqa: E402


def _real_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "c": rng.integers(0, 4, n),
        "class": 1,
    })


def test_identical_is_perfect():
    real = _real_df()
    r = em.evaluate_all(real, real.copy())
    assert r["KS_mean"] == pytest.approx(0.0, abs=1e-9)
    assert r["WD_mean"] == pytest.approx(0.0, abs=1e-9)
    assert r["MMD"] == pytest.approx(0.0, abs=1e-9)
    assert r["pMSE"] == pytest.approx(0.0, abs=1e-6)
    assert r["Precision"] == 1.0
    assert r["Coverage"] == 1.0
    assert r["Recall"] == 1.0
    assert r["KL_mean"] == pytest.approx(0.0, abs=1e-9)


def test_shifted_is_worse():
    real = _real_df()
    good = real.copy()
    bad = real.copy()
    bad["a"] = bad["a"] + 4.0
    bad["b"] = bad["b"] + 8.0
    rg, rb = em.evaluate_all(real, good), em.evaluate_all(real, bad)
    for k in ("KS_mean", "WD_mean", "MMD", "pMSE", "KL_mean", "JS_mean"):
        assert rb[k] > rg[k], k
    assert rb["Precision"] < rg["Precision"]
    assert rb["Coverage"] < rg["Coverage"]
    assert rb["Silhouette"] > rg["Silhouette"]  # piu' separabili


def test_textual_categorical_not_destroyed():
    rng = np.random.default_rng(0)
    vals = rng.choice(["red", "green", "blue"], 200)
    real = pd.DataFrame({"x": rng.normal(size=200), "cat": vals, "class": 1})
    gen = real.copy()
    r = em.evaluate_all(real, gen)
    assert r["JS_mean"] == pytest.approx(0.0, abs=1e-9)
    assert r["KS_mean"] == pytest.approx(0.0, abs=1e-9)
    # reale e generato con categorie diverse: la codifica sull'unione le distingue
    gen2 = real.copy()
    gen2["cat"] = "yellow"
    assert em.evaluate_all(real, gen2)["KS_mean"] > 0.1


def test_prepare_arrays_shared_mapping_and_imputation():
    real = pd.DataFrame({"n": [1.0, np.nan, 3.0, 5.0], "c": ["b", "a", None, "a"]})
    gen = pd.DataFrame({"n": [np.nan, 2.0], "c": ["c", "a"]})
    r, g = em.prepare_arrays(real, gen, ["n", "c"])
    # numerica: NaN -> mediana del reale (3.0); non 1e-7
    assert r[1, 0] == 3.0 and g[0, 0] == 3.0
    # categorica: mapping su unione ordinato a<b<c, stesso per entrambi
    assert list(r[:, 1][[0, 1, 3]]) == [1.0, 0.0, 0.0]
    assert g[0, 1] == 2.0 and g[1, 1] == 0.0
    # NaN categorico reale -> moda del reale (a = 0)
    assert r[2, 1] == 0.0


def test_pmse_logistic_formula():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    assert em.compute_pmse(X, X.copy()) == pytest.approx(0.0, abs=1e-6)
    far = em.compute_pmse(X, X + 10.0)
    assert far > 0.2  # c = 0.5, separabili -> (p-c)^2 vicino a 0.25
    assert np.isnan(em.compute_pmse(X, np.empty((0, 3))))


def test_prdc_vectorized_matches_naive():
    rng = np.random.default_rng(3)
    R, G = rng.normal(size=(60, 4)), rng.normal(size=(50, 4)) + 0.3
    p, rc, d, c = em.compute_precision_recall_density_coverage(R, G, k=5)
    from scipy.spatial.distance import cdist
    k = 5
    rr = np.sort(cdist(R, R), axis=1)[:, k]
    rg = np.sort(cdist(G, G), axis=1)[:, k]
    dm = cdist(R, G)
    assert p == pytest.approx(np.mean([np.any(dm[:, j] <= rr) for j in range(len(G))]))
    assert rc == pytest.approx(np.mean([np.any(dm[i, :] <= rg) for i in range(len(R))]))
    assert d == pytest.approx(np.mean([(dm[:, j] <= rr).sum() / k for j in range(len(G))]))
    assert c == pytest.approx(np.mean([np.any(dm[i, :] <= rr[i]) for i in range(len(R))]))


def test_subsampling_is_seeded(monkeypatch):
    monkeypatch.setattr(em, "MAX_SAMPLES", 50)
    rng = np.random.default_rng(0)
    R, G = rng.normal(size=(200, 3)), rng.normal(size=(180, 3))
    a = em.compute_mdcr(R, G, seed=1)
    assert em._subsample(R, 50, 1).shape[0] == 50
    # stesso seme -> stesso valore; seme diverso -> in generale diverso
    assert em.compute_mmd(R, G, seed=1) == em.compute_mmd(R, G, seed=1)
    assert em.compute_mdcr(R, G, seed=1) == a


def test_metric_direction_covers_all_keys():
    keys = em.FIDELITY_METRIC_KEYS + em.DIVERSITY_METRIC_KEYS
    assert set(keys) <= set(em.METRIC_DIRECTION)
    assert set(em.METRIC_DIRECTION.values()) <= {"lower", "higher", "neutral", "target1"}
    assert em.METRIC_DIRECTION["pMSE"] == "lower" and em.METRIC_DIRECTION["Precision"] == "higher"
    assert em.METRIC_DIRECTION["mDCR"] == "neutral" and em.METRIC_DIRECTION["FE_ratio_mean"] == "target1"
