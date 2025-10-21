# analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# Check GDAL installation
import shutil
print(shutil.which("gdal_translate"))


CSV_PATH = "logs/metrics/val_metrics.csv"  # <- change if needed
OUT_DIR  = "logs/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load ----
df = pd.read_csv(CSV_PATH)

# infer metric columns (everything that's not clearly meta)
meta_like = {
    "run_id","model","batch_idx","item_idx",
    "name","lon","lat",
    "LC_detail_id","LC_superclass_id","LC_superclass_text","LC_detail_text"
}
metric_cols = [c for c in df.columns if c not in meta_like]

# Optional: pick a single run if file has several runs appended
# df = df[df["run_id"] == df["run_id"].max()]

print("Metrics:", metric_cols)

# ---- helpers ----
def bar_mean_sem(g, metric, title, outfile):
    agg = g[metric].agg(['mean','sem']).reset_index()
    x = agg.iloc[:,0].astype(str)
    y = agg['mean']
    e = agg['sem']
    plt.figure(figsize=(8,4))
    plt.bar(x, y, yerr=e, capsize=4)
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def box_by_group(df, group_col, metric, title, outfile):
    plt.figure(figsize=(8,4))
    df.boxplot(column=metric, by=group_col)
    plt.suptitle("")  # remove pandas default
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

# ==== 1) Performance per model (mean ± SEM and boxplots) ====
for m in metric_cols:
    bar_mean_sem(df.groupby("model"), m,
                 title=f"{m} — mean±SEM by model",
                 outfile=os.path.join(OUT_DIR, f"{m}__mean_sem_by_model.png"))
    box_by_group(df, "model", m,
                 title=f"{m} — distribution by model",
                 outfile=os.path.join(OUT_DIR, f"{m}__box_by_model.png"))

# Also dump a tidy summary table (mean, std, count)
summary_model = df.groupby("model")[metric_cols].agg(['mean','std','count'])
summary_model.to_csv(os.path.join(OUT_DIR, "summary_by_model.csv"))

# ==== 2) Performance per model per land-cover class ====
# choose which LC field you prefer:
LC_COL = "LC_superclass_text" if "LC_superclass_text" in df.columns else \
         "LC_detail_text" if "LC_detail_text" in df.columns else None

if LC_COL is not None:
    # mean ± SEM bars per (model, LC)
    for metric in metric_cols:
        agg = df.groupby(["model", LC_COL])[metric].agg(['mean','sem']).reset_index()
        # pivot for a clean grouped bar per model (columns=LC classes)
        piv = agg.pivot(index="model", columns=LC_COL, values="mean")
        plt.figure(figsize=(10,5))
        piv.plot(kind="bar", ax=plt.gca())
        plt.title(f"{metric} — mean by model × {LC_COL}")
        plt.ylabel(metric)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{metric}__mean_by_model_x_{LC_COL}.png"), dpi=220)
        plt.close()

        # optional: per-model boxplots split by LC
        for mdl in df["model"].unique():
            sub = df[df["model"] == mdl]
            if sub[LC_COL].nunique() <= 1:
                continue
            box_by_group(sub, LC_COL, metric,
                         title=f"{metric} — {mdl} by {LC_COL}",
                         outfile=os.path.join(OUT_DIR, f"{metric}__{mdl}__box_by_{LC_COL}.png"))

    # write summary table
    summary_model_lc = df.groupby(["model", LC_COL])[metric_cols].agg(['mean','std','count'])
    summary_model_lc.to_csv(os.path.join(OUT_DIR, f"summary_by_model_x_{LC_COL}.csv"))
else:
    print("No land-cover text column found (LC_superclass_text / LC_detail_text). Skipping LC analysis.")

print(f"Done. Plots in: {OUT_DIR}")
