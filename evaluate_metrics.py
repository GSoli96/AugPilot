#!/usr/bin/env python3
"""
evaluate_metrics.py
=============================================================================
Valuta in modo esaustivo e confronta il dataset ORIGINALE (reale minoritario)
rispetto a TUTTI i dataset aumentati generati dai diversi algoritmi di data
augmentation presenti nelle sottodirectory di "./augmented_datasets/".

Le metriche vengono divise in 2 tabelle/CSV distinti:
    ./evaluation_metrics/fidelity/   -> Metriche di FIDELITY
    ./evaluation_metrics/diversity/  -> Metriche di DIVERSITY

Metriche di Fidelity:
  - KL_mean, JS_mean: Divergenza di Kullback-Leibler e Jensen-Shannon per attributo
  - Q_multi_attr_sim: Similarità multi-attributo basata su coseno
  - KS_mean, Chi2_mean: Test statistici Kolmogorov-Smirnov e Chi-quadro
  - WD_mean: Distanza di Wasserstein media
  - pMSE: Propensity Mean Squared Error (distinguibilità discriminatore)
  - MAD_mean: Mean Absolute Difference normalizzata
  - MMD: Maximum Mean Discrepancy (kernel multiscala RBF)
  - PCA_WD_mean: Wasserstein distance sui primi componenti PCA
  - Moment_mean_diff, Moment_var_diff, Moment_skew_diff, Moment_kurt_diff: Differenze dei momenti

Metriche di Diversity:
  - Silhouette: Separabilità dei cluster sintetici rispetto ai reali
  - Davies-Bouldin: Indice di dispersione e separazione
  - Compactness_cls_k: Compattezza intra-classe
  - Precision, Recall, Density, Coverage: Metriche spaziali k-NN
  - mDCR: Mean Distance to Closest Record reale
  - FE_ratio_mean: Rapporto di Feature Entropy medio
  - DBI_div: Davies-Bouldin diversity sui generati
  - Coverage_div: Frazione di quantili reali coperti dai sintetici

Uso tipico:
    # Singolo dataset:
    python evaluate_metrics.py --real ./datasets/iris0_min.csv
    python evaluate_metrics.py --dataset iris0_min

    # Tutti i dataset presenti:
    python evaluate_metrics.py --dataset all
    python evaluate_metrics.py --algorithms smote fair-smote syrfd cuts
"""

import argparse
import os
import sys
import re
import glob
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp, chisquare, skew, kurtosis
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Seme di default per tutti i sottocampionamenti (sovrascrivibile da evaluate_all(..., seed=...)).
METRICS_SEED = 0
# Massimo di righe (per reale e per generato) usate nelle metriche con costo quadratico.
MAX_SAMPLES = 5000


def _subsample(X: np.ndarray, max_n: int = MAX_SAMPLES, seed: Optional[int] = None) -> np.ndarray:
    """Sottocampiona X a max_n righe (senza reinserimento, ordine originale) con un rng dal seme."""
    if len(X) <= max_n:
        return X
    rng = np.random.default_rng(METRICS_SEED if seed is None else seed)
    idx = np.sort(rng.choice(len(X), max_n, replace=False))
    return X[idx]


# =========================================================================
# 1. METRICHE DI FIDELITY
# =========================================================================

def compute_kl_divergence(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 30) -> float:
    n_attrs = real_vals.shape[1]
    kl_per_attr = []
    for d in range(n_attrs):
        r, g = real_vals[:, d], gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        if np.std(r) < 1e-6:
            kl_per_attr.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(g, bins=bins)
        q, _ = np.histogram(r, bins=bins)
        p = (p + 1).astype(float); p /= p.sum()
        q = (q + 1).astype(float); q /= q.sum()
        kl_per_attr.append(float(entropy(p, q)))
    return float(np.mean(kl_per_attr))


def compute_js_divergence(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 30) -> float:
    n_attrs = real_vals.shape[1]
    js_per_attr = []
    for d in range(n_attrs):
        r, g = real_vals[:, d], gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        if np.std(r) < 1e-6:
            js_per_attr.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(g, bins=bins)
        q, _ = np.histogram(r, bins=bins)
        p = (p + 1).astype(float); p /= p.sum()
        q = (q + 1).astype(float); q /= q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
        js_per_attr.append(float(js))
    return float(np.mean(js_per_attr))


def compute_q_similarity(real_vals: np.ndarray, gen_vals: np.ndarray, seed: Optional[int] = None) -> float:
    """Similarita' coseno al vicino piu' simile, riscalata in [0,1] (meglio alto). Sottocampiona a MAX_SAMPLES."""
    real_vals, gen_vals = _subsample(real_vals, seed=seed), _subsample(gen_vals, seed=seed)

    def l2_norm(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms < 1e-10, 1.0, norms)
    gen_n, real_n = l2_norm(gen_vals), l2_norm(real_vals)
    sim_matrix = gen_n @ real_n.T
    max_sim = sim_matrix.max(axis=1)
    return float(((max_sim + 1.0) / 2.0).mean())


def compute_ks_mean(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    ks_stats = []
    for d in range(real_vals.shape[1]):
        stat, _ = ks_2samp(real_vals[:, d], gen_vals[:, d])
        ks_stats.append(stat)
    return float(np.mean(ks_stats))


def compute_chi2_mean(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 20) -> float:
    chi2_stats = []
    for d in range(real_vals.shape[1]):
        r, g = real_vals[:, d], gen_vals[:, d]
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        if np.std(r) < 1e-6:
            chi2_stats.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        f_obs, _ = np.histogram(g, bins=bins)
        f_exp, _ = np.histogram(r, bins=bins)
        f_obs = f_obs + 1
        f_exp = f_exp + 1
        f_obs = f_obs * (f_exp.sum() / f_obs.sum())
        stat, _ = chisquare(f_obs, f_exp=f_exp)
        chi2_stats.append(stat / len(g))
    return float(np.mean(chi2_stats))


def compute_wd_mean(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    wd_stats = []
    for d in range(real_vals.shape[1]):
        r_sorted = np.sort(real_vals[:, d])
        g_sorted = np.sort(gen_vals[:, d])
        grid = np.linspace(0, 1, 100)
        r_quant = np.quantile(r_sorted, grid)
        g_quant = np.quantile(g_sorted, grid)
        wd_stats.append(np.mean(np.abs(r_quant - g_quant)))
    return float(np.mean(wd_stats))


def compute_pmse(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    """pMSE standard: regressione logistica reale(0) vs generato(1), mean((p - c)^2), c = n_gen/(n_real+n_gen).
    Meglio basso (0 = indistinguibili). Restituisce nan se il modello non si addestra."""
    n_real, n_gen = len(real_vals), len(gen_vals)
    if n_real == 0 or n_gen == 0:
        return float('nan')
    X = np.vstack([real_vals, gen_vals])
    y = np.array([0] * n_real + [1] * n_gen)
    c = n_gen / (n_real + n_gen)
    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        return float(np.mean((p - c) ** 2))
    except Exception:
        return float('nan')


def compute_mad_mean(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    mu_real = np.mean(real_vals, axis=0)
    mu_gen = np.mean(gen_vals, axis=0)
    std_real = np.std(real_vals, axis=0)
    valid = std_real > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.abs(mu_real[valid] - mu_gen[valid]) / std_real[valid]))


def compute_mmd(real_vals: np.ndarray, gen_vals: np.ndarray, scales=(0.05, 0.2, 0.5, 1.0, 2.0),
                seed: Optional[int] = None) -> float:
    """MMD^2 (stimatore di V) con somma di kernel RBF multiscala (meglio basso). Sottocampiona a MAX_SAMPLES."""
    real_vals, gen_vals = _subsample(real_vals, seed=seed), _subsample(gen_vals, seed=seed)

    def rbf_kernel_matrix(x, y):
        dist_matrix = cdist(x, y, 'sqeuclidean')
        dim = x.shape[1]
        return sum(np.exp(-dist_matrix / (s * dim + 1e-10)) for s in scales)
    k_xx = rbf_kernel_matrix(gen_vals, gen_vals).mean()
    k_yy = rbf_kernel_matrix(real_vals, real_vals).mean()
    k_xy = rbf_kernel_matrix(gen_vals, real_vals).mean()
    return float(np.clip(k_xx - 2.0 * k_xy + k_yy, a_min=0.0, a_max=None))


def compute_pca_wd_mean(real_vals: np.ndarray, gen_vals: np.ndarray, n_components: int = 3) -> float:
    n_comp = min(n_components, real_vals.shape[1])
    pca = PCA(n_components=n_comp)
    real_pca = pca.fit_transform(real_vals)
    gen_pca = pca.transform(gen_vals)
    wd_stats = []
    for d in range(n_comp):
        r_quant = np.quantile(real_pca[:, d], np.linspace(0, 1, 100))
        g_quant = np.quantile(gen_pca[:, d], np.linspace(0, 1, 100))
        wd_stats.append(np.mean(np.abs(r_quant - g_quant)))
    return float(np.mean(wd_stats))


def compute_moment_diffs(real_vals: np.ndarray, gen_vals: np.ndarray) -> tuple:
    m1_r, m1_g = np.mean(real_vals, axis=0), np.mean(gen_vals, axis=0)
    m2_r, m2_g = np.var(real_vals, axis=0), np.var(gen_vals, axis=0)
    m3_r, m3_g = skew(real_vals, axis=0, bias=False), skew(gen_vals, axis=0, bias=False)
    m4_r, m4_g = kurtosis(real_vals, axis=0, bias=False), kurtosis(gen_vals, axis=0, bias=False)
    m3_r, m3_g = np.nan_to_num(m3_r), np.nan_to_num(m3_g)
    m4_r, m4_g = np.nan_to_num(m4_r), np.nan_to_num(m4_g)
    std_r = np.std(real_vals, axis=0)
    var_r = m2_r

    valid_std = std_r > 1e-6
    diff_m1 = float(np.mean(np.abs(m1_r[valid_std] - m1_g[valid_std]) / std_r[valid_std])) if np.any(valid_std) else 0.0

    valid_var = var_r > 1e-12
    diff_m2 = float(np.mean(np.abs(m2_r[valid_var] - m2_g[valid_var]) / var_r[valid_var])) if np.any(valid_var) else 0.0

    diff_m3 = float(np.mean(np.abs(m3_r - m3_g)))
    diff_m4 = float(np.mean(np.abs(m4_r - m4_g)))
    return diff_m1, diff_m2, diff_m3, diff_m4


# =========================================================================
# 2. METRICHE DI DIVERSITY & SPAZIALI
# =========================================================================

def compute_silhouette(real_vals: np.ndarray, gen_vals: np.ndarray, seed: Optional[int] = None) -> float:
    """Silhouette con etichette reale/generato: misura quanto i due insiemi sono separabili (meglio basso).
    Sottocampiona a MAX_SAMPLES."""
    real_vals, gen_vals = _subsample(real_vals, seed=seed), _subsample(gen_vals, seed=seed)
    n_gen, n_real = len(gen_vals), len(real_vals)
    if n_gen < 2 or n_real < 2: return float('nan')
    X = np.vstack([gen_vals, real_vals])
    labels = np.array([0] * n_gen + [1] * n_real)
    try: return float(silhouette_score(X, labels, metric='euclidean'))
    except: return float('nan')


def compute_dbi(real_vals: np.ndarray, gen_vals: np.ndarray, seed: Optional[int] = None) -> float:
    """Davies-Bouldin con etichette reale/generato (alto = cluster meno sovrapposti). Sottocampiona a MAX_SAMPLES."""
    real_vals, gen_vals = _subsample(real_vals, seed=seed), _subsample(gen_vals, seed=seed)
    n_gen, n_real = len(gen_vals), len(real_vals)
    if n_gen < 2 or n_real < 2: return float('nan')
    X = np.vstack([gen_vals, real_vals])
    labels = np.array([0] * n_gen + [1] * n_real)
    try: return float(davies_bouldin_score(X, labels))
    except: return float('nan')


def compute_intra_compactness(gen_vals: np.ndarray) -> float:
    return float(np.mean(np.var(gen_vals, axis=0)))


def _get_kth_nn_dists(X: np.ndarray, k: int) -> np.ndarray:
    """Distanza dal k-esimo vicino (escluso se stesso) per ogni riga. X deve essere gia' sottocampionato
    (non lo fa qui, altrimenti si perde la corrispondenza riga-raggio)."""
    n = len(X)
    if n < 2:
        return np.zeros(n)
    k = int(min(max(k, 1), n - 1))
    dists = cdist(X, X, 'euclidean')
    return np.partition(dists, k, axis=1)[:, k]


def compute_precision_recall_density_coverage(real_vals: np.ndarray, gen_vals: np.ndarray, k: int = 5,
                                              seed: Optional[int] = None) -> tuple:
    """PRDC (Naeem et al. 2020), vettorizzato. Sottocampiona reale e generato a MAX_SAMPLES.
    Precision: frazione di generati dentro la palla k-NN di almeno un reale. Recall: simmetrico.
    Density: media (su generati) del numero di palle reali che li contengono / k. Coverage: frazione di reali
    la cui palla contiene almeno un generato. Meglio alto per tutte (Density puo' superare 1; ideale ~1)."""
    real_vals, gen_vals = _subsample(real_vals, seed=seed), _subsample(gen_vals, seed=seed)
    n_real, n_gen = len(real_vals), len(gen_vals)
    k_r = min(k, max(1, n_real - 1))
    k_g = min(k, max(1, n_gen - 1))
    r_real = _get_kth_nn_dists(real_vals, k_r)      # (n_real,)
    r_gen = _get_kth_nn_dists(gen_vals, k_g)        # (n_gen,)
    d = cdist(real_vals, gen_vals, 'euclidean')     # (n_real, n_gen)

    in_real_ball = d <= r_real[:, None]             # gen j dentro la palla del reale i
    precision = float(in_real_ball.any(axis=0).mean())
    recall = float((d <= r_gen[None, :]).any(axis=1).mean())
    density = float(in_real_ball.sum(axis=0).mean() / k_r)
    coverage = float(in_real_ball.any(axis=1).mean())
    return precision, recall, density, coverage


def compute_mdcr(real_vals: np.ndarray, gen_vals: np.ndarray, seed: Optional[int] = None) -> float:
    """Mean Distance to Closest Record: distanza media di ogni generato dal reale piu' vicino.
    Ambigua (basso = fedele ma rischio copia/privacy, alto = lontano). Sottocampiona a MAX_SAMPLES."""
    real_vals, gen_vals = _subsample(real_vals, seed=seed), _subsample(gen_vals, seed=seed)
    d_matrix = cdist(gen_vals, real_vals, 'euclidean')
    return float(np.mean(d_matrix.min(axis=1)))


def compute_fe_ratio_mean(real_vals: np.ndarray, gen_vals: np.ndarray, n_bins: int = 10) -> float:
    n_attrs = real_vals.shape[1]
    ratios = []
    for d in range(n_attrs):
        r, g = real_vals[:, d], gen_vals[:, d]
        if np.std(r) < 1e-6:
            ratios.append(1.0)
            continue
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(g, bins=bins, density=True)
        q, _ = np.histogram(r, bins=bins, density=True)
        h_g = entropy(p + 1e-12)
        h_r = entropy(q + 1e-12)
        ratios.append(h_g / (h_r + 1e-12))
    return float(np.mean(ratios))


def compute_dbi_div(gen_vals: np.ndarray) -> float:
    if len(gen_vals) < 10: return float('nan')
    median = np.median(gen_vals, axis=0)
    labels = np.any(gen_vals > median, axis=1).astype(int)
    if len(np.unique(labels)) < 2: return 0.0
    try: return float(davies_bouldin_score(gen_vals, labels))
    except: return float('nan')


def compute_coverage_div(real_vals: np.ndarray, gen_vals: np.ndarray) -> float:
    n_attrs = real_vals.shape[1]
    covered_fractions = []
    for d in range(n_attrs):
        r = real_vals[:, d]
        g = gen_vals[:, d]
        quantiles = np.quantile(r, np.linspace(0, 1, 11))
        unique_q = np.unique(quantiles)
        if len(unique_q) < 2:
            covered_fractions.append(1.0)
            continue
        counts, _ = np.histogram(g, bins=unique_q)
        active_bins = np.sum(counts > 0)
        total_bins = len(unique_q) - 1
        covered_fractions.append(active_bins / total_bins)
    return float(np.mean(covered_fractions))


# =========================================================================
# 3. ENGINE DI VALUTAZIONE CORE (FIDELITY & DIVERSITY)
# =========================================================================

FIDELITY_METRIC_KEYS = [
    'KL_mean', 'JS_mean', 'Q_multi_attr_sim', 'KS_mean', 'Chi2_mean',
    'WD_mean', 'pMSE', 'MAD_mean', 'MMD', 'PCA_WD_mean',
    'Moment_mean_diff', 'Moment_var_diff', 'Moment_skew_diff', 'Moment_kurt_diff',
]

DIVERSITY_METRIC_KEYS = [
    'Silhouette', 'Davies-Bouldin', 'Compactness_cls_k',
    'Precision', 'Recall', 'Density', 'Coverage',
    'mDCR', 'FE_ratio_mean', 'DBI_div', 'Coverage_div',
]

# Direzione di ogni metrica: 'lower' = meglio basso, 'higher' = meglio alto, 'target1' = ideale 1,
# 'neutral' = nessuna direzione univoca (non va ordinata come "meglio/peggio" ne' aggregata in un punteggio).
METRIC_DIRECTION = {
    # --- Fidelity ---
    'KL_mean': 'lower',            # divergenza KL media per attributo
    'JS_mean': 'lower',            # divergenza JS media per attributo
    'Q_multi_attr_sim': 'higher',  # similarita' coseno al vicino piu' simile, in [0,1]
    'KS_mean': 'lower',            # statistica KS media
    'Chi2_mean': 'lower',          # chi-quadro normalizzato medio
    'WD_mean': 'lower',            # Wasserstein media per attributo
    'pMSE': 'lower',               # propensity MSE: 0 = reale e generato indistinguibili
    'MAD_mean': 'lower',           # differenza media assoluta normalizzata
    'MMD': 'lower',                # Maximum Mean Discrepancy
    'PCA_WD_mean': 'lower',        # Wasserstein sulle prime componenti PCA
    'Moment_mean_diff': 'lower',
    'Moment_var_diff': 'lower',
    'Moment_skew_diff': 'lower',
    'Moment_kurt_diff': 'lower',
    # --- Diversity (scelte discutibili: vedi commenti) ---
    # Silhouette con etichette reale/generato misura la SEPARABILITA' dei due insiemi: separati = generato diverso
    # dal reale, quindi meglio basso.
    'Silhouette': 'lower',
    # Davies-Bouldin con le stesse etichette: alto = cluster reale/generato meno separati (piu' simili), quindi
    # higher. E' l'opposto della lettura classica del DBI (dove basso e' meglio): usare con cautela.
    'Davies-Bouldin': 'higher',
    # Dispersione media (varianza) del solo generato: non dice se e' bene o male.
    'Compactness_cls_k': 'neutral',
    # PRDC (Naeem 2020): tutte meglio alte; Density puo' superare 1 (ideale ~1, ma si ordina come higher).
    'Precision': 'higher',
    'Recall': 'higher',
    'Density': 'higher',
    'Coverage': 'higher',
    # Distanza media dal reale piu' vicino: bassa = fedele ma possibile copia (privacy), alta = lontano: ambigua.
    'mDCR': 'neutral',
    # Rapporto di entropia generato/reale per feature: l'ideale e' 1 (ne' piu' ne' meno varieta' del reale).
    'FE_ratio_mean': 'target1',
    # Davies-Bouldin interno al generato su una partizione arbitraria (mediana): nessun significato univoco.
    'DBI_div': 'neutral',
    # Frazione dei decili reali che contengono almeno un generato: meglio alta.
    'Coverage_div': 'higher',
}


def _is_numeric_pair(rs: pd.Series, gs: pd.Series) -> bool:
    """True se entrambe le serie sono convertibili a numerico senza perdere valori non nulli."""
    for s in (rs, gs):
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            continue
        nn = s.dropna()
        if len(nn) and pd.to_numeric(nn, errors='coerce').isna().any():
            return False
    return True


def prepare_arrays(real_df: pd.DataFrame, gen_df: pd.DataFrame, cols: List[str]) -> tuple:
    """Converte reale e generato in matrici float sulle colonne `cols`.

    Colonne non numeriche: codifica ordinale stabile (categorie ordinate alfabeticamente) costruita
    sull'unione reale+generato, con lo stesso mapping per entrambi. NaN: imputati con la mediana (numeriche)
    o la moda (categoriche) calcolate sul REALE, per entrambi i dataset."""
    real_out, gen_out = {}, {}
    for col in cols:
        rs, gs = real_df[col], gen_df[col]
        if _is_numeric_pair(rs, gs):
            r = pd.to_numeric(rs, errors='coerce').astype(float)
            g = pd.to_numeric(gs, errors='coerce').astype(float)
            fill = r.median()
        else:
            cats = sorted(set(rs.dropna().astype(str)) | set(gs.dropna().astype(str)))
            mapping = {v: float(i) for i, v in enumerate(cats)}
            r = rs.astype(str).map(mapping).where(rs.notna()).astype(float)
            g = gs.astype(str).map(mapping).where(gs.notna()).astype(float)
            mode = r.mode()
            fill = mode.iloc[0] if len(mode) else np.nan
        if pd.isna(fill):
            fill = 0.0
        real_out[col] = r.fillna(fill).values
        gen_out[col] = g.fillna(fill).values
    return (np.column_stack([real_out[c] for c in cols]).astype(float),
            np.column_stack([gen_out[c] for c in cols]).astype(float))


def evaluate_all(real_df: pd.DataFrame, gen_df: pd.DataFrame, n_bins: int = 30,
                 seed: Optional[int] = None) -> dict:
    """Calcola tutte le metriche di FIDELITY_METRIC_KEYS e DIVERSITY_METRIC_KEYS (direzioni in METRIC_DIRECTION).
    seed: seme dei sottocampionamenti (max MAX_SAMPLES righe per lato); default METRICS_SEED."""
    seed = METRICS_SEED if seed is None else seed
    target_col_real = 'class' if 'class' in real_df.columns else real_df.columns[-1]
    target_col_gen = 'class' if 'class' in gen_df.columns else gen_df.columns[-1]

    # Se gen_df ha più classi ed è presente la colonna target, filtra per la classe minoritaria di real_df
    if target_col_gen in gen_df.columns and target_col_real in real_df.columns:
        if len(gen_df[target_col_gen].unique()) > 1:
            min_label = real_df[target_col_real].iloc[0]
            mask = gen_df[target_col_gen].astype(str) == str(min_label)
            if mask.any():
                gen_df = gen_df[mask].copy()

    attrs = [c for c in real_df.columns if c != target_col_real]
    gen_attrs = [c for c in gen_df.columns if c != target_col_gen]
    common = [a for a in attrs if a in gen_attrs]

    if not common:
        raise ValueError("Nessun attributo comune trovato tra dataset reale e generato.")

    # Codifica categoriche (mapping sull'unione) e imputazione NaN con statistiche del reale
    real_vals, gen_vals = prepare_arrays(real_df, gen_df, common)

    # Identifica ed esclude colonne a varianza zero nel reale per la normalizzazione
    col_std = np.std(real_vals, axis=0)
    valid_cols = col_std > 1e-6
    if not np.any(valid_cols):
        valid_cols = np.ones_like(valid_cols, dtype=bool)
    if np.any(~valid_cols):
        real_vals_scaled = real_vals[:, valid_cols]
        gen_vals_scaled  = gen_vals[:, valid_cols]
    else:
        real_vals_scaled = real_vals
        gen_vals_scaled  = gen_vals

    scaler = MinMaxScaler()
    scaler.fit(real_vals_scaled)
    real_n = scaler.transform(real_vals_scaled)
    gen_n  = np.clip(scaler.transform(gen_vals_scaled), 0.0, 1.0)

    res = {}
    # Fidelity
    res['KL_mean'] = compute_kl_divergence(real_vals, gen_vals, n_bins)
    res['JS_mean'] = compute_js_divergence(real_vals, gen_vals, n_bins)
    res['Q_multi_attr_sim'] = compute_q_similarity(real_n, gen_n, seed=seed)
    res['KS_mean'] = compute_ks_mean(real_vals, gen_vals)
    res['Chi2_mean'] = compute_chi2_mean(real_vals, gen_vals)
    res['WD_mean'] = compute_wd_mean(real_vals, gen_vals)
    res['pMSE'] = compute_pmse(real_n, gen_n)
    res['MAD_mean'] = compute_mad_mean(real_n, gen_n)
    res['MMD'] = compute_mmd(real_n, gen_n, seed=seed)
    res['PCA_WD_mean'] = compute_pca_wd_mean(real_n, gen_n)
    
    m1, m2, m3, m4 = compute_moment_diffs(real_n, gen_n)
    res['Moment_mean_diff'] = m1
    res['Moment_var_diff'] = m2
    res['Moment_skew_diff'] = m3
    res['Moment_kurt_diff'] = m4

    # Diversity
    res['Silhouette'] = compute_silhouette(real_n, gen_n, seed=seed)
    res['Davies-Bouldin'] = compute_dbi(real_n, gen_n, seed=seed)
    res['Compactness_cls_k'] = compute_intra_compactness(gen_n)
    
    prec, rec, dens, cov = compute_precision_recall_density_coverage(real_n, gen_n, k=5, seed=seed)
    res['Precision'] = prec
    res['Recall'] = rec
    res['Density'] = dens
    res['Coverage'] = cov
    
    res['mDCR'] = compute_mdcr(real_n, gen_n, seed=seed)
    res['FE_ratio_mean'] = compute_fe_ratio_mean(real_vals, gen_vals)
    res['DBI_div'] = compute_dbi_div(gen_n)
    res['Coverage_div'] = compute_coverage_div(real_vals, gen_vals)

    return res


# =========================================================================
# 4. DISCOVERY DEI DATASET AUMENTATI PER TUTTI GLI ALGORITMI
# =========================================================================

def clean_dataset_name(filename: str) -> str:
    """Ritorna la radice pura del dataset eliminando estensione e suffissi comuni."""
    ref_base = os.path.basename(filename)
    if ref_base.endswith('.csv'):
        ref_base = ref_base[:-4]
    ref_base = re.sub(r'_augmented$', '', ref_base, flags=re.IGNORECASE)
    ref_base = re.sub(r'_min$', '', ref_base, flags=re.IGNORECASE)
    return ref_base.lower().strip()


def find_sydc_datasets_in_dir(sweep_dir: Path, core_ref: str) -> List[Dict[str, str]]:
    """
    Rileva ed estrae dataset aumentati generati da SyDC all'interno di una cartella sweep
    (es. dc_rank_sweep_noclass/<config>/runs/<dataset>/generated/<gen_subfolder>/*.csv).
    """
    results = []
    if not sweep_dir.is_dir():
        return results

    # Rileva se sweep_dir ha sottocartelle di configurazione contenenti 'runs/' o se ha 'runs/' direttamente
    configs = [c for c in sweep_dir.iterdir() if c.is_dir() and (c / 'runs').is_dir()]
    if not configs and (sweep_dir / 'runs').is_dir():
        configs = [sweep_dir]

    for cfg_dir in sorted(configs):
        cfg_name = cfg_dir.name if cfg_dir != sweep_dir else "default"
        runs_dir = cfg_dir / "runs"
        if not runs_dir.is_dir():
            continue

        for ds_dir in sorted(runs_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds_clean = clean_dataset_name(ds_dir.name)
            # Verifica che la cartella corrisponda al dataset di riferimento
            if ds_clean != core_ref and not ds_dir.name.lower().startswith(core_ref) and not core_ref.startswith(ds_clean):
                continue

            gen_base = ds_dir / "generated"
            if not gen_base.is_dir():
                continue

            for gen_sub in sorted(gen_base.iterdir()):
                if not gen_sub.is_dir():
                    continue
                gen_name = gen_sub.name
                mode = gen_name
                m_match = re.search(r"__(relaxed_on|relaxed_off|strict_on|strict_off)(?:__|$)", gen_name)
                if m_match:
                    mode = m_match.group(1)
                elif gen_name.startswith("gen__"):
                    parts = gen_name.split("__")
                    if len(parts) >= 4:
                        mode = parts[3]

                algo_tag = f"sydc_{cfg_name}_{mode}" if cfg_name != "default" else f"sydc_{mode}"

                for csv_file in sorted(gen_sub.glob("*.csv")):
                    if csv_file.is_file() and csv_file.stat().st_size > 0:
                        results.append({
                            "algorithm": algo_tag,
                            "filename": csv_file.name,
                            "full_path": str(csv_file),
                            "sweep_source": sweep_dir.name,
                            "config": cfg_name,
                            "mode": mode,
                        })
    return results


def find_matching_datasets(base_dataset_name: str, search_dir: str, requested_algos: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Scansiona tutte le sottocartelle in search_dir (es. ./augmented_datasets/)
    cercando i file aumentati corrispondenti al dataset in esame.
    Supporta tutti gli algoritmi standard e le cartelle di sweep di SyDC (es. dc_rank_sweep_noclass/).
    """
    matched_files = []
    core_ref = clean_dataset_name(base_dataset_name)
    
    if not os.path.exists(search_dir):
        print(f"[ATTENZIONE] La directory '{search_dir}' non esiste.")
        return matched_files

    all_entries = sorted([
        d for d in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, d)) and not d.startswith('.')
    ])

    for entry in all_entries:
        entry_path = Path(search_dir) / entry
        # Riconosce se è una cartella di sweep SyDC (con struttura runs/ o sottocartelle con runs/)
        is_sydc_sweep = (
            (entry_path / "runs").is_dir()
            or any((sub / "runs").is_dir() for sub in entry_path.iterdir() if sub.is_dir())
            or entry.lower().startswith("dc_rank_sweep")
            or "sydc" in entry.lower()
        )

        if is_sydc_sweep:
            sydc_items = find_sydc_datasets_in_dir(entry_path, core_ref)
            matched_files.extend(sydc_items)
        else:
            # Cartella di algoritmo standard
            for root, _, files in os.walk(str(entry_path)):
                for file in sorted(files):
                    if not file.lower().endswith('.csv'):
                        continue
                    file_core = clean_dataset_name(file)
                    # Match esatto o per prefisso radice
                    if file_core == core_ref or file.lower().startswith(core_ref):
                        full_path = os.path.join(root, file)
                        matched_files.append({
                            "algorithm": entry,
                            "filename": file,
                            "full_path": full_path
                        })

    # Filtraggio su requested_algos se specificato
    if requested_algos and "all" not in [a.lower() for a in requested_algos]:
        req_norm = [a.lower().strip() for a in requested_algos]
        filtered = []
        for item in matched_files:
            algo_val = item["algorithm"].lower()
            sweep_src = item.get("sweep_source", "").lower()
            matched = False
            for req in req_norm:
                if (
                    algo_val == req
                    or algo_val.startswith(req)
                    or req in algo_val
                    or (sweep_src and (sweep_src == req or req in sweep_src))
                ):
                    matched = True
                    break
            if matched:
                filtered.append(item)
        matched_files = filtered

    return matched_files


def load_minority_data(path: str) -> pd.DataFrame:
    """Carica il dataset reale ed estrae la classe minoritaria se multilabel/multiclasse."""
    df = pd.read_csv(path)
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    counts = df[target_col].value_counts()
    if len(counts) > 1:
        min_class = counts.idxmin()
        return df[df[target_col] == min_class].copy()
    return df.copy()


# =========================================================================
# 5. ESECUZIONE DELLA VALUTAZIONE PER UN DATASET
# =========================================================================

def evaluate_single_dataset(real_path: str, augmented_dir: str, output_dir: str,
                            algorithms: Optional[List[str]] = None,
                            bins: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Esegue la valutazione di fidelity e diversity per un dataset specifico."""
    clean_target = clean_dataset_name(os.path.basename(real_path))
    real_df = load_minority_data(real_path)

    print(f"\n" + "=" * 90)
    print(f"🔬 VALUTAZIONE: {clean_target} ({len(real_df)} tuple reali minoritarie)")
    print("=" * 90)

    matching_datasets = find_matching_datasets(clean_target, augmented_dir, requested_algos=algorithms)
    print(f"• Trovati {len(matching_datasets)} dataset aumentati in '{augmented_dir}'")
    if not matching_datasets:
        print(f"⚠️  Nessun dataset aumentato trovato per '{clean_target}' in '{augmented_dir}'.")
        return pd.DataFrame(), pd.DataFrame()

    fidelity_results = []
    diversity_results = []

    for item in matching_datasets:
        print(f"  → [{item['algorithm']}] {item['filename']} ...", end=" ", flush=True)
        try:
            comp_df = pd.read_csv(item['full_path'])
            if len(comp_df) == 0:
                print("SALTATO (file vuoto / 0 campioni)")
                continue
            metrics = evaluate_all(real_df, comp_df, n_bins=bins)

            fid_row = {
                "Dataset": clean_target,
                "Algoritmo": item['algorithm'],
                "File": item['filename'],
                "N_Generated": len(comp_df),
            }
            fid_row.update({k: round(metrics[k], 6) if isinstance(metrics[k], float) else metrics[k] for k in FIDELITY_METRIC_KEYS})
            fidelity_results.append(fid_row)

            div_row = {
                "Dataset": clean_target,
                "Algoritmo": item['algorithm'],
                "File": item['filename'],
                "N_Generated": len(comp_df),
            }
            div_row.update({k: round(metrics[k], 6) if isinstance(metrics[k], float) else metrics[k] for k in DIVERSITY_METRIC_KEYS})
            diversity_results.append(div_row)
            print("OK")
        except Exception as e:
            print(f"ERRORE: {e}")

    fidelity_table = pd.DataFrame(fidelity_results)
    diversity_table = pd.DataFrame(diversity_results)

    for table in (fidelity_table, diversity_table):
        if not table.empty:
            table.sort_values(by=['Algoritmo', 'File'], inplace=True)
            table.reset_index(drop=True, inplace=True)

    # Salvataggio CSV per questo dataset
    fidelity_dir = os.path.join(output_dir, "fidelity")
    diversity_dir = os.path.join(output_dir, "diversity")
    os.makedirs(fidelity_dir, exist_ok=True)
    os.makedirs(diversity_dir, exist_ok=True)

    if not fidelity_table.empty:
        fid_path = os.path.join(fidelity_dir, f"fidelity_{clean_target}.csv")
        fidelity_table.to_csv(fid_path, index=False)
        print(f"📁 Tabella Fidelity salvata in:  {fid_path}")

    if not diversity_table.empty:
        div_path = os.path.join(diversity_dir, f"diversity_{clean_target}.csv")
        diversity_table.to_csv(div_path, index=False)
        print(f"📁 Tabella Diversity salvata in: {div_path}")

    return fidelity_table, diversity_table


# =========================================================================
# 6. MAIN CLI ENTRYPOINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark di Fidelity e Diversity del dataset reale vs. tutti i dataset aumentati "
                    "degli algoritmi in augmented_datasets/; genera 2 CSV separati (fidelity e diversity)."
    )
    parser.add_argument("--real", type=str, default=None,
                        help="Path al CSV del dataset reale minoritario (es. ./datasets/iris0_min.csv)")
    parser.add_argument("--imbalanced", type=str, default=None,
                        help="Path al CSV imbilanciato originale di riferimento")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Nome del dataset (es. iris0_min) o 'all' per valutarli tutti")
    parser.add_argument("--datasets_dir", type=str, default="./datasets",
                        help="Directory contenente i dataset originali/reali (default: ./datasets)")
    parser.add_argument("--augmented_dir", type=str, default="./augmented_datasets",
                        help="Path della root dei dataset aumentati (default: ./augmented_datasets)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_metrics",
                        help="Directory di output dove salvare le tabelle (default: ./evaluation_metrics)")
    parser.add_argument("--algorithms", nargs="+", default=["all"],
                        help="Filtro algoritmi da valutare (default: all)")
    parser.add_argument("--bins", type=int, default=30,
                        help="Numero di bin per istogrammi (default: 30)")
    args = parser.parse_args()

    # Risoluzione dataset da valutare
    dataset_paths: List[str] = []

    if args.real:
        if args.real.lower() == "all":
            args.dataset = "all"
        else:
            dataset_paths.append(args.real)
    elif args.imbalanced:
        dataset_paths.append(args.imbalanced)
    elif args.dataset and args.dataset.lower() != "all":
        # Cerca per nome in datasets_dir
        candidate = os.path.join(args.datasets_dir, args.dataset)
        if not candidate.endswith('.csv'):
            candidate_csv = candidate + '.csv'
            if os.path.exists(candidate_csv):
                candidate = candidate_csv
            elif os.path.exists(candidate + '_min.csv'):
                candidate = candidate + '_min.csv'
        if os.path.exists(candidate):
            dataset_paths.append(candidate)
        else:
            # Fallback search by clean name
            matches = [f for f in glob.glob(os.path.join(args.datasets_dir, "*.csv"))
                       if clean_dataset_name(f) == clean_dataset_name(args.dataset)]
            if matches:
                dataset_paths.append(matches[0])
            else:
                print(f"[ERRORE] Dataset '{args.dataset}' non trovato in {args.datasets_dir}.")
                sys.exit(1)

    if not dataset_paths:
        # Modalità ALL: processa tutti i dataset in datasets_dir
        if not os.path.exists(args.datasets_dir):
            print(f"[ERRORE] Directory dei dataset '{args.datasets_dir}' non trovata.")
            sys.exit(1)
        dataset_paths = sorted(glob.glob(os.path.join(args.datasets_dir, "*.csv")))
        print(f"📊 Modalità globale: trovati {len(dataset_paths)} dataset in '{args.datasets_dir}'.")

    all_fidelity_dfs = []
    all_diversity_dfs = []

    for d_path in dataset_paths:
        fid_df, div_df = evaluate_single_dataset(
            real_path=d_path,
            augmented_dir=args.augmented_dir,
            output_dir=args.output_dir,
            algorithms=args.algorithms,
            bins=args.bins,
        )
        if not fid_df.empty:
            all_fidelity_dfs.append(fid_df)
        if not div_df.empty:
            all_diversity_dfs.append(div_df)

    # Se elaborati più dataset, salva anche un riepilogo complessivo aggregato
    if len(all_fidelity_dfs) > 1:
        agg_fid = pd.concat(all_fidelity_dfs, ignore_index=True)
        agg_fid_path = os.path.join(args.output_dir, "fidelity_all_datasets.csv")
        agg_fid.to_csv(agg_fid_path, index=False)
        print(f"\n📊 Riepilogo complessivo Fidelity salvato in:  {agg_fid_path}")

    if len(all_diversity_dfs) > 1:
        agg_div = pd.concat(all_diversity_dfs, ignore_index=True)
        agg_div_path = os.path.join(args.output_dir, "diversity_all_datasets.csv")
        agg_div.to_csv(agg_div_path, index=False)
        print(f"📊 Riepilogo complessivo Diversity salvato in: {agg_div_path}")

    print("\n✅ Valutazione delle metriche completata con successo!\n")


if __name__ == "__main__":
    main()