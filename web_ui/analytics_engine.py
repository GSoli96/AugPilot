"""
analytics_engine.py
===================
Analytics and chart data computation for the Data Augmentation Web UI.
Produces structured JSON payloads for Plotly.js charts (Scatter Trade-off,
Radar Profile, Diversity Bar Charts, Heatmap, Rankings).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


DIRECTION = {
    # Fidelity / Realism (down = better)
    "KL_mean": "down",
    "JS_mean": "down",
    "KS_mean": "down",
    "Chi2_mean": "down",
    "WD_mean": "down",
    "pMSE": "down",
    "MAD_mean": "down",
    "MMD": "down",
    "PCA_WD_mean": "down",
    "Moment_mean_diff": "down",
    "Moment_var_diff": "down",
    "Moment_skew_diff": "down",
    "Moment_kurt_diff": "down",
    # Similarity (up = better)
    "Q_multi_attr_sim": "up",
    # Clustering (up/down)
    "Silhouette": "up",
    "Davies-Bouldin": "down",
    "Compactness_cls_k": "down",
    # ML utility (up = better)
    "Precision": "up",
    "Recall": "up",
    # Diversity / Novelty (up = better)
    "Density": "up",
    "Coverage": "up",
    "mDCR": "up",
    "FE_ratio_mean": "up",
    "DBI_div": "up",
    "Coverage_div": "up",
}


def norm_best(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalize metric so that 1 = best and 0 = worst."""
    if col not in df.columns:
        return pd.Series(0.5, index=df.index)
    x = df[col].astype(float)
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return pd.Series(1.0, index=df.index)
    z = (x - xmin) / (xmax - xmin)
    return z if DIRECTION.get(col, "down") == "up" else (1.0 - z)


def get_available_evaluated_datasets(eval_dir: Path) -> List[str]:
    """List datasets that have metric evaluation CSV files available."""
    fid_dir = eval_dir / "fidelity"
    if not fid_dir.is_dir():
        return []
    datasets = set()
    for f in fid_dir.glob("fidelity_*.csv"):
        if f.name == "fidelity_all_datasets.csv":
            continue
        ds = f.stem.replace("fidelity_", "")
        datasets.add(ds)
    return sorted(list(datasets))


def load_dataset_metrics(eval_dir: Path, dataset_name: str) -> Optional[pd.DataFrame]:
    """Load and merge Fidelity and Diversity tables for a specific dataset."""
    fid_path = eval_dir / "fidelity" / f"fidelity_{dataset_name}.csv"
    div_path = eval_dir / "diversity" / f"diversity_{dataset_name}.csv"

    if not fid_path.exists() or not div_path.exists():
        # Try fallback without _min or with _min
        alt_name = dataset_name.replace("_min", "") if "_min" in dataset_name else f"{dataset_name}_min"
        fid_path = eval_dir / "fidelity" / f"fidelity_{alt_name}.csv"
        div_path = eval_dir / "diversity" / f"diversity_{alt_name}.csv"

    if not fid_path.exists() or not div_path.exists():
        return None

    try:
        fid = pd.read_csv(fid_path)
        div = pd.read_csv(div_path)

        keys = ["Dataset", "Algoritmo"]
        df = fid.merge(
            div.drop(columns=["File", "N_Generated"], errors="ignore"),
            on=keys,
            how="inner",
        )

        # Create human-friendly short name for long names like SyDC or Kamino variants
        df["Algo_short"] = (
            df["Algoritmo"]
            .str.replace("sydc_coverage_top", "sydc_cov_t", regex=False)
            .str.replace("sydc_interestingness_top", "sydc_int_t", regex=False)
            .str.replace("_relaxed_off", "_r_off", regex=False)
            .str.replace("_relaxed_on", "_r_on", regex=False)
            .str.replace("_strict_off", "_s_off", regex=False)
            .str.replace("_strict_on", "_s_on", regex=False)
        )

        return df
    except Exception as e:
        print(f"Error loading metrics for {dataset_name}: {e}")
        return None


def compute_analytics_payload(eval_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """Generate all analytical data and Plotly-ready figures for the dashboard."""
    df = load_dataset_metrics(eval_dir, dataset_name)
    if df is None or df.empty:
        return {"success": False, "error": f"No evaluation metrics found for dataset '{dataset_name}'."}

    # 1. Quality & Diversity Scores
    q_metrics = [m for m in ["KL_mean", "JS_mean", "WD_mean", "MMD"] if m in df.columns]
    d_metrics = [m for m in ["DBI_div", "Coverage_div", "Coverage", "Density"] if m in df.columns]

    df["Quality_score"] = df[q_metrics].apply(lambda s: norm_best(df, s.name)).mean(axis=1).round(4)
    df["Diversity_score"] = df[d_metrics].apply(lambda s: norm_best(df, s.name)).mean(axis=1).round(4)

    # 2. Multi-Dimensional Radar Indices
    df["Fidelity_idx"] = df[[m for m in ["KL_mean", "JS_mean", "WD_mean", "MMD", "KS_mean"] if m in df.columns]] \
        .apply(lambda s: norm_best(df, s.name)).mean(axis=1).round(4)
    df["Clustering_idx"] = df[[m for m in ["Silhouette", "Davies-Bouldin"] if m in df.columns]] \
        .apply(lambda s: norm_best(df, s.name)).mean(axis=1).round(4)
    df["Diversity_idx"] = df[[m for m in ["Density", "Coverage", "mDCR", "DBI_div", "Coverage_div"] if m in df.columns]] \
        .apply(lambda s: norm_best(df, s.name)).mean(axis=1).round(4)
    df["Utility_idx"] = df[[m for m in ["Precision", "Recall"] if m in df.columns]].mean(axis=1).round(4)

    # 3. Scatter Plot Data: Quality vs Diversity
    scatter_data = {
        "x": df["Quality_score"].tolist(),
        "y": df["Diversity_score"].tolist(),
        "text": df["Algo_short"].tolist(),
        "full_algo": df["Algoritmo"].tolist(),
        "color": df["Silhouette"].fillna(0.0).round(4).tolist() if "Silhouette" in df.columns else [0.5] * len(df),
        "n_samples": df["N_Generated"].tolist() if "N_Generated" in df.columns else [0] * len(df),
    }

    # 4. Radar Chart Data: Top 8 Models by Quality Score
    top_models = df.nlargest(min(8, len(df)), "Quality_score")
    radar_data = {
        "categories": ["Fidelity", "Clustering", "Utility", "Diversity"],
        "series": [],
    }
    for _, row in top_models.iterrows():
        radar_data["series"].append({
            "name": row["Algo_short"],
            "values": [
                float(row["Fidelity_idx"]),
                float(row["Clustering_idx"]),
                float(row["Utility_idx"]),
                float(row["Diversity_idx"]),
            ],
        })

    # 5. Diversity Bar Chart Data
    div_bars_metrics = ["Density", "Coverage", "mDCR", "FE_ratio_mean", "DBI_div", "Coverage_div"]
    div_bars_data = {}
    for m in div_bars_metrics:
        if m in df.columns:
            sorted_m = df.sort_values(m, ascending=(DIRECTION.get(m) == "down"))
            div_bars_data[m] = {
                "algos": sorted_m["Algo_short"].tolist(),
                "values": sorted_m[m].round(4).tolist(),
                "direction": DIRECTION.get(m, "up"),
            }

    # 6. Normalized Heatmap Data
    all_heat_cols = [c for c in DIRECTION.keys() if c in df.columns]
    df_norm = pd.DataFrame(index=df["Algo_short"])
    for c in all_heat_cols:
        df_norm[c] = norm_best(df, c).values

    df_norm["Overall_Score"] = df_norm.mean(axis=1).round(4)
    df_norm = df_norm.sort_values("Overall_Score", ascending=True)

    heatmap_data = {
        "y_algos": df_norm.index.tolist(),
        "x_metrics": all_heat_cols,
        "z_values": df_norm[all_heat_cols].round(3).values.tolist(),
        "overall_scores": df_norm["Overall_Score"].tolist(),
    }

    # 7. Summary Ranking Table (best = 1)
    ranking_series = df_norm[all_heat_cols].rank(ascending=False).mean(axis=1).sort_values()
    rankings_list = []
    for rank_idx, (algo, avg_rank) in enumerate(ranking_series.items(), start=1):
        row_full = df[df["Algo_short"] == algo].iloc[0]
        rankings_list.append({
            "rank": rank_idx,
            "algo_short": algo,
            "full_name": str(row_full["Algoritmo"]),
            "avg_rank": round(float(avg_rank), 2),
            "quality_score": float(row_full["Quality_score"]),
            "diversity_score": float(row_full["Diversity_score"]),
            "n_generated": int(row_full.get("N_Generated", 0)),
        })

    # 8. Raw Merged Table Data for Interactive Tabulator
    raw_table_records = json.loads(df.round(5).to_json(orient="records"))

    return {
        "success": True,
        "dataset": dataset_name,
        "n_algorithms": len(df),
        "scatter": scatter_data,
        "radar": radar_data,
        "diversity_bars": div_bars_data,
        "heatmap": heatmap_data,
        "rankings": rankings_list,
        "raw_table": raw_table_records,
    }
