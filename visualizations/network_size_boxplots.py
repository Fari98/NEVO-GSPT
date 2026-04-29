import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
LOG_PATH = "../log/network_size_experiment_20260427.csv"
OUT_DIR = "../visualizations/sizes"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_ORDER = ["small", "medium", "large", "xl", "xxl"]
CONFIG_LABELS = {
    "small":  "Small\n(8×2)",
    "medium": "Medium\n(16×3)",
    "large":  "Large\n(32×4)",
    "xl":     "XL\n(64×5)",
    "xxl":    "XXL\n(128×6)",
}
DATASET_NAMES = {
    "airfoil":           "Airfoil",
    "bioav":             "Bioavailability",
    "concrete_strength": "Concrete Strength",
    "ld50":              "LD50",
}

PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]  # one per config

# ── Load & filter ─────────────────────────────────────────────────────────────
df = pd.read_csv(LOG_PATH)
df = df[df["status"] == "success"].copy()
df["config_label"] = pd.Categorical(df["config_label"], categories=CONFIG_ORDER, ordered=True)
datasets = df["dataset"].unique()

# ── Figure 1: one subplot per dataset ─────────────────────────────────────────
fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 5), sharey=False)
fig.suptitle("Test Fitness by Network Size Configuration", fontsize=14, fontweight="bold")

for ax, dataset in zip(axes, sorted(datasets)):
    subset = df[df["dataset"] == dataset]
    groups = [subset[subset["config_label"] == c]["test_fitness"].values for c in CONFIG_ORDER]

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(CONFIG_ORDER) + 1))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER], fontsize=8)
    ax.set_title(DATASET_NAMES.get(dataset, dataset), fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Fitness (lower is better)" if ax == axes[0] else "")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "network_size_boxplots_all.pdf")
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()

# ── Figure 2: all datasets combined, grouped by config ────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))

n_configs = len(CONFIG_ORDER)
n_datasets = len(sorted(datasets))
width = 0.15
x = np.arange(n_configs)

dataset_colors = plt.cm.Set2(np.linspace(0, 1, n_datasets))

for i, dataset in enumerate(sorted(datasets)):
    subset = df[df["dataset"] == dataset]
    medians = [subset[subset["config_label"] == c]["test_fitness"].median() for c in CONFIG_ORDER]
    q1 = [subset[subset["config_label"] == c]["test_fitness"].quantile(0.25) for c in CONFIG_ORDER]
    q3 = [subset[subset["config_label"] == c]["test_fitness"].quantile(0.75) for c in CONFIG_ORDER]
    yerr_lo = np.array(medians) - np.array(q1)
    yerr_hi = np.array(q3) - np.array(medians)

    offset = (i - n_datasets / 2 + 0.5) * width
    ax2.bar(
        x + offset,
        medians,
        width=width * 0.9,
        color=dataset_colors[i],
        label=DATASET_NAMES.get(dataset, dataset),
        alpha=0.85,
        yerr=[yerr_lo, yerr_hi],
        capsize=3,
        error_kw=dict(elinewidth=1.2),
    )

ax2.set_xticks(x)
ax2.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER], fontsize=10)
ax2.set_ylabel("Median Test Fitness (lower is better)", fontsize=11)
ax2.set_title("Median Test Fitness by Network Size (IQR error bars)", fontsize=13, fontweight="bold")
ax2.legend(title="Dataset", fontsize=9, title_fontsize=9)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
out_path2 = os.path.join(OUT_DIR, "network_size_barplot_combined.pdf")
plt.savefig(out_path2, bbox_inches="tight")
print(f"Saved: {out_path2}")
plt.show()

# ── Figure 3: one boxplot per dataset (standalone, publication-ready) ──────────
for dataset in sorted(datasets):
    subset = df[df["dataset"] == dataset]
    groups = [subset[subset["config_label"] == c]["test_fitness"].values for c in CONFIG_ORDER]

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    bp = ax3.boxplot(
        groups,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    means = [g.mean() if len(g) > 0 else np.nan for g in groups]
    ax3.plot(range(1, n_configs + 1), means, "D--", color="black", markersize=5, linewidth=1, label="Mean")

    ax3.set_xticks(range(1, len(CONFIG_ORDER) + 1))
    ax3.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER], fontsize=10)
    ax3.set_ylabel("Test Fitness", fontsize=11)
    ax3.set_title(f"{DATASET_NAMES.get(dataset, dataset)} — Test Fitness by Network Size", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path3 = os.path.join(OUT_DIR, f"network_size_boxplot_{dataset}.pdf")
    plt.savefig(out_path3, bbox_inches="tight")
    print(f"Saved: {out_path3}")
    plt.show()

print("\nDone.")
