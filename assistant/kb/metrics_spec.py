"""Direzione e asse di ogni metrica del benchmark.

Per ogni colonna del KB che e' una metrica si dichiara:
  direzione: 'lower' (meglio basso) | 'higher' (meglio alto) | 'target1' (ideale 1) | 'neutral' (non ordinabile)
  asse:      'fidelity' | 'diversity' | 'privacy' | 'utility'   (None per le metriche neutral)

Fonti:
  tabulare  fd_<k>   -> evaluate_metrics.METRIC_DIRECTION; asse da FIDELITY_METRIC_KEYS / DIVERSITY_METRIC_KEYS
            priv_<k> -> bench.privacy.PRIVACY_DIRECTION (asse privacy)
            util_<k> -> evaluate_utility.UTILITY_DIRECTION; contano `_delta` (guadagno rispetto al solo reale) e
                        `_trs` (addestra su sintetico); `_trtr` e' il riferimento (uguale per tutti: neutral)
  ts, graph m_<k>    -> docstring di modalities/ts/metrics.py e modalities/graph/metrics.py (scelte di lettura
                        annotate sotto, dove la direzione non e' scritta nel modulo)

Le tabelle sono copie statiche (importare evaluate_metrics costa ~2 s e carica sklearn): `check_against_sources()`
le confronta con i moduli originali ed e' usata dai test.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

AXES = ("fidelity", "diversity", "privacy", "utility")
DIRECTIONS = ("lower", "higher", "target1", "neutral")

# ---------------------------------------------------------------- tabulare
TAB_FIDELITY_KEYS = ["KL_mean", "JS_mean", "Q_multi_attr_sim", "KS_mean", "Chi2_mean", "WD_mean", "pMSE", "MAD_mean",
                     "MMD", "PCA_WD_mean", "Moment_mean_diff", "Moment_var_diff", "Moment_skew_diff", "Moment_kurt_diff"]
TAB_DIVERSITY_KEYS = ["Silhouette", "Davies-Bouldin", "Compactness_cls_k", "Precision", "Recall", "Density", "Coverage",
                      "mDCR", "FE_ratio_mean", "DBI_div", "Coverage_div"]
# copia di evaluate_metrics.METRIC_DIRECTION
TAB_FD_DIRECTION = {
    "KL_mean": "lower", "JS_mean": "lower", "Q_multi_attr_sim": "higher", "KS_mean": "lower", "Chi2_mean": "lower",
    "WD_mean": "lower", "pMSE": "lower", "MAD_mean": "lower", "MMD": "lower", "PCA_WD_mean": "lower",
    "Moment_mean_diff": "lower", "Moment_var_diff": "lower", "Moment_skew_diff": "lower", "Moment_kurt_diff": "lower",
    "Silhouette": "lower", "Davies-Bouldin": "higher", "Compactness_cls_k": "neutral", "Precision": "higher",
    "Recall": "higher", "Density": "higher", "Coverage": "higher", "mDCR": "neutral", "FE_ratio_mean": "target1",
    "DBI_div": "neutral", "Coverage_div": "higher",
}
# copia di bench.privacy.PRIVACY_DIRECTION
TAB_PRIV_DIRECTION = {
    "DCR_p5": "higher", "DCR_median": "higher", "DCR_holdout_p5": "neutral",
    "DCR_ratio": "target1", "exact_copy_rate": "lower",
    "closer_than_holdout_share": "lower", "NNDR_mean": "higher",
}
# copia di evaluate_utility.UTILITY_DIRECTION
TAB_UTIL_DIRECTION = {"f1": "higher", "bal_acc": "higher", "pr_auc": "higher"}

# ---------------------------------------------------------------- serie temporali (modalities/ts/metrics.py)
# Il modulo dice: "Tutte le metriche sono 'piu' basso = meglio' tranne clf_acc_*".
# Scelte di lettura: ED/DTW sono distanze dal vicino piu' prossimo (basso = fedele) e restano sull'asse fidelity
# (la loro lettura come rischio di copia non e' dichiarata nel modulo); DS (|acc-0.5|) e' un test di
# indistinguibilita' (fidelity); PS e' l'errore del predittore addestrato sul generato (utilita', basso e' meglio);
# PS_trtr e clf_acc_base sono riferimenti (neutral); clf_acc_aug e' ridondante con clf_acc_delta (neutral).
# Non esiste una metrica di diversity ne' di privacy per le serie temporali: quegli assi restano vuoti.
TS_SPEC: Dict[str, Tuple[str, Optional[str]]] = {
    "n_orig": ("neutral", None), "n_gen": ("neutral", None), "metrics_time_s": ("neutral", None),
    "MDD": ("lower", "fidelity"), "ACD": ("lower", "fidelity"), "SD": ("lower", "fidelity"),
    "KD": ("lower", "fidelity"), "ED": ("lower", "fidelity"), "DTW": ("lower", "fidelity"),
    "FD_feat": ("lower", "fidelity"), "DS": ("lower", "fidelity"),
    "PS": ("lower", "utility"), "PS_trtr": ("neutral", None),
    "clf_acc_base": ("neutral", None), "clf_acc_aug": ("neutral", None), "clf_acc_delta": ("higher", "utility"),
}

# ---------------------------------------------------------------- grafi (modalities/graph/metrics.py)
# struct_*_w1/mmd: distanze tra distribuzioni strutturali (basso = fedele). div_syn_uniqueness, div_syn_novelty,
# div_syn_pairwise_dist: piu' alto = piu' vario; div_syn_feat_dup_frac (nodi): quota di duplicati, basso = vario.
# div_syn_feat_nn_dist e le colonne *_orig_*/*_gen_*/*_syn_*_mean sono descrittive o ambigue (neutral).
# base_*/aug_*: riferimenti e valori assoluti (neutral); contano i delta_* del classificatore GIN (livello grafo) o
# GCN (livello nodo), piu' alto = meglio. imb_ratio_*: descrive la composizione, non la qualita' (neutral).
GRAPH_SPEC: Dict[str, Tuple[str, Optional[str]]] = {
    "n_syn": ("neutral", None), "gen_mode": ("neutral", None),
    "struct_deg_mmd": ("lower", "fidelity"), "struct_deg_w1": ("lower", "fidelity"),
    "struct_clust_w1": ("lower", "fidelity"), "struct_spec_mmd": ("lower", "fidelity"),
    "struct_spec_w1": ("lower", "fidelity"), "struct_spec_frac_used": ("neutral", None),
    "struct_spec_computed": ("neutral", None),
    "div_syn_uniqueness": ("higher", "diversity"), "div_syn_novelty": ("higher", "diversity"),
    "div_syn_pairwise_dist": ("higher", "diversity"), "div_orig_pairwise_dist": ("neutral", None),
    "div_syn_feat_dup_frac": ("lower", "diversity"), "div_syn_feat_nn_dist": ("neutral", None),
    "div_orig_feat_nn_dist": ("neutral", None),
    "imb_ratio_before": ("neutral", None), "imb_ratio_after": ("neutral", None),
    "delta_macro_f1": ("higher", "utility"), "delta_accuracy": ("higher", "utility"),
    "delta_auc": ("higher", "utility"),
}

NEUTRAL: Tuple[str, Optional[str]] = ("neutral", None)


def _tab_spec(col: str) -> Tuple[str, Optional[str]]:
    if col.startswith("fd_"):
        k = col[3:]
        d = TAB_FD_DIRECTION.get(k)
        if d in (None, "neutral"):
            return NEUTRAL
        return (d, "diversity" if k in TAB_DIVERSITY_KEYS else "fidelity")
    if col.startswith("priv_"):
        d = TAB_PRIV_DIRECTION.get(col[5:])
        return NEUTRAL if d in (None, "neutral") else (d, "privacy")
    if col.startswith("util_"):
        rest = col[5:]
        for suffix in ("_delta", "_trs"):
            if rest.endswith(suffix) and rest[: -len(suffix)] in TAB_UTIL_DIRECTION:
                return (TAB_UTIL_DIRECTION[rest[: -len(suffix)]], "utility")
    return NEUTRAL


def metric_spec(modality: str, column: str) -> Tuple[str, Optional[str]]:
    """(direzione, asse) di una colonna del KB; ('neutral', None) se non e' una metrica nota."""
    if modality == "tabular":
        return _tab_spec(column)
    if not column.startswith("m_"):
        return NEUTRAL
    k = column[2:]
    if modality == "ts":
        return TS_SPEC.get(k, NEUTRAL)
    if modality == "graph":
        return GRAPH_SPEC.get(k, NEUTRAL)
    return NEUTRAL


def is_metric_column(column: str) -> bool:
    return column.startswith(("fd_", "priv_", "util_", "m_"))


def check_against_sources() -> list:
    """Confronta le copie statiche con i moduli originali; restituisce l'elenco delle differenze (vuoto = ok)."""
    diff = []
    import evaluate_metrics as em
    import evaluate_utility as eu
    from bench import privacy
    if dict(em.METRIC_DIRECTION) != TAB_FD_DIRECTION:
        diff.append("METRIC_DIRECTION")
    if list(em.FIDELITY_METRIC_KEYS) != TAB_FIDELITY_KEYS:
        diff.append("FIDELITY_METRIC_KEYS")
    if list(em.DIVERSITY_METRIC_KEYS) != TAB_DIVERSITY_KEYS:
        diff.append("DIVERSITY_METRIC_KEYS")
    if dict(privacy.PRIVACY_DIRECTION) != TAB_PRIV_DIRECTION:
        diff.append("PRIVACY_DIRECTION")
    if dict(eu.UTILITY_DIRECTION) != TAB_UTIL_DIRECTION:
        diff.append("UTILITY_DIRECTION")
    return diff
