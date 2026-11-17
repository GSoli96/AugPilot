"""Test analitici delle metriche per serie temporali (modalities/ts/metrics.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

metrics = pytest.importorskip("modalities.ts.metrics")


def _series(n=12, T=16, D=2, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(n, T, D)), axis=1)


def test_nn_distances_zero_on_permutation():
    a = _series()
    b = a[np.random.default_rng(1).permutation(len(a))]
    assert metrics.ed_nn(b, a) == pytest.approx(0.0, abs=1e-9)
    assert metrics.dtw_nn(b, a) == pytest.approx(0.0, abs=1e-9)


def test_nn_distances_positive_when_shifted():
    a = _series()
    assert metrics.ed_nn(a + 3.0, a) > 1.0
    assert metrics.dtw_nn(a + 3.0, a) > 1.0


def test_dtw_nn_matches_bruteforce():
    a, b = _series(5, 8, 1, 0), _series(4, 8, 1, 5)
    brute = np.mean([min(metrics._dtw_pairs(b[i:i + 1], a[j:j + 1])[0] for j in range(len(a))) for i in range(len(b))])
    assert metrics.dtw_nn(b, a) == pytest.approx(brute)
    # blocchi piccoli danno lo stesso risultato
    assert metrics.dtw_nn(b, a, pair_budget=8 * 8 * 2) == pytest.approx(brute)


def test_evaluate_permutation(tmp_path):
    a = _series(20, 12, 2)
    b = a[np.random.default_rng(2).permutation(len(a))]
    pa, pb = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(pa, data=a); np.savez(pb, data=b)
    r = metrics.evaluate(str(pa), str(pb), model_based=False)
    assert r["ED"] == pytest.approx(0.0, abs=1e-9)
    assert r["DTW"] == pytest.approx(0.0, abs=1e-9)
    assert r["MDD"] == pytest.approx(0.0, abs=1e-9)
