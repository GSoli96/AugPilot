import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# =========================================================
# 1) CARICAMENTO E MERGE
# =========================================================
# Uso (dalla cartella plot/ oppure dalla radice):  python plot/plot_results.py --dataset abalone9-18
_ap = argparse.ArgumentParser(description="Grafici di fidelity e diversity per un dataset.")
_ap.add_argument("--dataset", required=True, help="Nome del dataset (es. abalone9-18), come nei CSV di evaluation_metrics/")
_ap.add_argument("--metrics_dir", default=None,
                 help="Cartella evaluation_metrics (default: ../evaluation_metrics rispetto a questo file)")
_ap.add_argument("--out_dir", default=None, help="Cartella di output dei PNG (default: cartella di questo file)")
ARGS = _ap.parse_args()
DATASET = ARGS.dataset
_HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = ARGS.metrics_dir or os.path.join(_HERE, "..", "evaluation_metrics")
OUT_DIR = ARGS.out_dir or _HERE
DS_LABEL = DATASET

fid = pd.read_csv(os.path.join(METRICS_DIR, "fidelity", f"fidelity_{DATASET}.csv"))
div = pd.read_csv(os.path.join(METRICS_DIR, "diversity", f"diversity_{DATASET}.csv"))

# Chiave univoca: Dataset + Algoritmo (File puo' differire nei nomi)
keys = ["Dataset", "Algoritmo"]
df = fid.merge(
    div.drop(columns=["File", "N_Generated"], errors="ignore"),
    on=keys, how="inner"
)
df = df[df["Dataset"] == DATASET].copy()

# Nomi brevi
df["Algo_short"] = (df["Algoritmo"]
    .str.replace("sydc_coverage_top", "sydc_cov_t", regex=False)
    .str.replace("sydc_interestingness_top", "sydc_int_t", regex=False)
    .str.replace("_relaxed_off", "_r_off", regex=False)
    .str.replace("_relaxed_on",  "_r_on",  regex=False)
    .str.replace("_strict_off",  "_s_off", regex=False)
    .str.replace("_strict_on",   "_s_on",  regex=False))

# ---------------------------------------------------------
# Direzioni delle metriche ("up" = più alto è meglio)
# ---------------------------------------------------------
DIRECTION = {
    # fidelity / realismo (↓ meglio)
    "KL_mean":"down","JS_mean":"down","KS_mean":"down","Chi2_mean":"down",
    "WD_mean":"down","pMSE":"down","MAD_mean":"down","MMD":"down",
    "PCA_WD_mean":"down","Moment_mean_diff":"down","Moment_var_diff":"down",
    "Moment_skew_diff":"down","Moment_kurt_diff":"down",
    # similarità (↑ meglio)
    "Q_multi_attr_sim":"up",
    # clustering quality
    "Silhouette":"up","Davies-Bouldin":"down","Compactness_cls_k":"down",
    # PRDC precision/recall (NON utilita' ML)
    "Precision":"up","Recall":"up",
    # diversity / novelty
    "Density":"up","Coverage":"up","mDCR":"up",
    "FE_ratio_mean":"up","DBI_div":"up","Coverage_div":"up",
}

# Normalizziamo tutto su [0,1] con 1 = MIGLIORE
def norm_best(df, col):
    x = df[col].astype(float)
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return pd.Series(0.5, index=df.index)
    z = (x - xmin) / (xmax - xmin)
    return z if DIRECTION[col] == "up" else 1 - z

# =========================================================
# 2) BAR CHART — DIVERSITY METRICS
# =========================================================
div_metrics = ["Density","Coverage","mDCR","FE_ratio_mean","DBI_div","Coverage_div"]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, m in zip(axes.ravel(), div_metrics):
    order = df.sort_values(m, ascending=(DIRECTION[m]=="down"))["Algo_short"]
    palette = "viridis" if DIRECTION[m]=="up" else "viridis_r"
    sns.barplot(data=df, y="Algo_short", x=m, order=order,
                palette=palette, ax=ax)
    ax.set_title(f"{m}  ({'↑ meglio' if DIRECTION[m]=='up' else '↓ meglio'})")
    ax.set_xlabel(""); ax.set_ylabel("")
plt.suptitle(f"Metriche di diversity — {DS_LABEL}", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_diversity_bars.png"), dpi=150, bbox_inches="tight")
plt.show()

# =========================================================
# 3) SCATTER QUALITY vs DIVERSITY (trade-off)
# =========================================================
df["Quality_score"] = (
    df[["KL_mean","JS_mean","WD_mean","MMD"]]
    .apply(lambda s: norm_best(df, s.name)).mean(axis=1)
)
df["Diversity_score"] = (
    df[["DBI_div","Coverage_div","Coverage","Density"]]
    .apply(lambda s: norm_best(df, s.name)).mean(axis=1)
)

fig, ax = plt.subplots(figsize=(11, 8))
sc = ax.scatter(df["Quality_score"], df["Diversity_score"],
                c=df["Silhouette"], s=160, cmap="coolwarm",
                edgecolor="black", linewidth=0.7)
for _, r in df.iterrows():
    ax.annotate(r["Algo_short"], (r["Quality_score"], r["Diversity_score"]),
                fontsize=7, alpha=0.85, xytext=(4,3), textcoords="offset points")
plt.colorbar(sc, label="Silhouette (cluster quality)")
ax.set_xlabel("Quality score (fidelity, ↑ meglio)")
ax.set_ylabel("Diversity score (↑ meglio)")
ax.set_title(f"Trade-off Quality ↔ Diversity — {DS_LABEL}")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_quality_vs_diversity.png"), dpi=150)
plt.show()

# =========================================================
# 4) HEATMAP COMPLETA (fidelity + clustering + diversity)
# =========================================================
heat_cols = [
    # fidelity
    "KL_mean","JS_mean","Q_multi_attr_sim","KS_mean","Chi2_mean",
    "WD_mean","pMSE","MAD_mean","MMD","PCA_WD_mean",
    # clustering
    "Silhouette","Davies-Bouldin","Compactness_cls_k",
    # PRDC precision/recall (NON utilita' ML)
    "Precision","Recall",
    # diversity
    "Density","Coverage","mDCR","FE_ratio_mean","DBI_div","Coverage_div",
]

df_norm = df.set_index("Algo_short")[heat_cols].copy()
for c in heat_cols:
    df_norm[c] = norm_best(df, c).values      # 1 = best

df_norm["SCORE_MEDIO"] = df_norm.mean(axis=1)
df_norm = df_norm.sort_values("SCORE_MEDIO")

plt.figure(figsize=(16, 11))
sns.heatmap(df_norm.drop(columns="SCORE_MEDIO"), cmap="RdYlGn",
            annot=True, fmt=".2f", linewidths=0.4,
            cbar_kws={"label":"0 = peggiore, 1 = migliore"})
plt.title(f"Heatmap completa (fidelity + clustering + diversity) — {DS_LABEL}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_heatmap_full.png"), dpi=150)
plt.show()

# =========================================================
# 5) RADAR — fidelity, clustering, diversity
# =========================================================
# Costruisco 3 indici aggregati per ogni algoritmo
df["Fidelity_idx"] = df[["KL_mean","JS_mean","WD_mean","MMD","KS_mean"]] \
                        .apply(lambda s: norm_best(df, s.name)).mean(axis=1)
df["Clustering_idx"] = df[["Silhouette","Davies-Bouldin"]] \
                        .apply(lambda s: norm_best(df, s.name)).mean(axis=1)
df["Diversity_idx"] = df[["Density","Coverage","mDCR","DBI_div","Coverage_div"]] \
                        .apply(lambda s: norm_best(df, s.name)).mean(axis=1)
df["PRDC_PR_idx"] = df[["Precision","Recall"]].mean(axis=1)

radar_axes = ["Fidelity_idx","Clustering_idx","PRDC_PR_idx","Diversity_idx"]
angles = [n/len(radar_axes)*2*pi for n in range(len(radar_axes))] + [0]

# Top-8 per Score medio complessivo
top = df.nlargest(8, "Quality_score").set_index("Algo_short")[radar_axes]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for algo, row in top.iterrows():
    vals = row.tolist() + row.tolist()[:1]
    ax.plot(angles, vals, linewidth=2, label=algo)
    ax.fill(angles, vals, alpha=0.07)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Fidelity","Clustering","PRDC precision/recall","Diversity"], fontsize=12)
ax.set_yticklabels([]); ax.set_ylim(0,1)
ax.set_title("Profilo multidimensionale — top-8 per fidelity", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.0), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_radar_full.png"), dpi=150)
plt.show()

# =========================================================
# 6) RANKING FINALE MULTI-CRITERIO
# =========================================================
# Ranking medio su tutte le metriche "best=1" già normalizzate
ranking = df_norm.drop(columns="SCORE_MEDIO").rank(ascending=False).mean(axis=1) \
                 .sort_values()
print("\n=== RANKING GENERALE (più basso = migliore) ===")
print(ranking.to_string())

# Ranking separato: fidelity vs diversity
print("\n=== TOP-5 Fidelity ===")
print(df.set_index("Algo_short")["Fidelity_idx"].nlargest(5).to_string())
print("\n=== TOP-5 Diversity ===")
print(df.set_index("Algo_short")["Diversity_idx"].nlargest(5).to_string())
print("\n=== TOP-5 Clustering ===")
print(df.set_index("Algo_short")["Clustering_idx"].nlargest(5).to_string())