"""
visualize.py

Generates the five figures from this experimental block:

  Fig 1: PR per layer (raw + unit side by side), 3 models x 3 conditions.
  Fig 2: ID per layer (raw + unit side by side), 3 models x 3 conditions.
  Fig 3: M1b (inter-layer cosine similarity) per transition.
  Fig 4: M3 raw - r2_mlp, r2_triv and gap.
  Fig 5: M3 unit - same.

Common structure:
  - Rows: models (DistilBERT, BERT, GPT-2).
  - Columns: variants (raw/unit for global metrics) or metrics (mlp/triv for M3).
  - Curves in each panel: one per condition (coherent/permuted/random).
  - X axis: absolute model layer (each panel uses its own range).

Output: results/figures/*.png

Usage:
    python visualize.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["distilbert", "bert", "gpt2"]
MODEL_LABELS = {"distilbert": "DistilBERT (6 layers)",
                "bert": "BERT base (12 layers)",
                "gpt2": "GPT-2 small (12 layers)"}
CONDITIONS = ["coherent", "permuted", "random"]
COND_COLORS = {"coherent": "#1f77b4", "permuted": "#ff7f0e", "random": "#2ca02c"}
COND_LABELS = {"coherent": "Coherent", "permuted": "Permuted", "random": "Random"}


def load_global(model: str, condition: str):
    path = RESULTS_DIR / f"global_{model}_{condition}.npz"
    return dict(np.load(path))


def load_m1b(model: str, condition: str):
    path = RESULTS_DIR / f"m1b_{model}_{condition}.npz"
    return dict(np.load(path))


def load_m3(model: str, condition: str):
    path = RESULTS_DIR / f"m3_{model}_{condition}.npz"
    return dict(np.load(path))


# ---------------------------------------------------------------------------
# Fig 1 & 2: Globals (PR, ID)
# ---------------------------------------------------------------------------

def plot_global_metric(metric_basename: str, title: str, save_name: str):
    """metric_basename: 'pr' or 'id'.
    Plots raw on the left, unit on the right, 3 models in rows."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for row, model in enumerate(MODELS):
        for col, version in enumerate(["raw", "unit"]):
            ax = axes[row, col]
            for condition in CONDITIONS:
                d = load_global(model, condition)
                y = d[f"{metric_basename}_{version}"]
                x = np.arange(len(y))
                ax.plot(x, y, marker="o", markersize=4,
                        color=COND_COLORS[condition],
                        label=COND_LABELS[condition], linewidth=1.6)
            ax.set_xticks(np.arange(len(y)))
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"{metric_basename.upper()} ({version})")
            ax.set_title(f"{MODEL_LABELS[model]} - {metric_basename.upper()}_{version}",
                         fontsize=10)
            ax.grid(alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    out = FIG_DIR / save_name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 3: M1b
# ---------------------------------------------------------------------------

def plot_m1b():
    """One column, three rows (models). Each panel: 3 curves (conditions).
    X axis: inter-layer transitions (L0->L1, L1->L2, ...)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)
    for row, model in enumerate(MODELS):
        ax = axes[row]
        for condition in CONDITIONS:
            d = load_m1b(model, condition)
            y = d["m1b_unit_mean"]
            yerr = d["m1b_unit_std"]
            x = np.arange(len(y))
            ax.errorbar(x, y, yerr=yerr, marker="o", markersize=4,
                        color=COND_COLORS[condition],
                        label=COND_LABELS[condition],
                        linewidth=1.6, capsize=3, alpha=0.85)
        ax.set_xticks(np.arange(len(y)))
        ax.set_xticklabels([f"L{i}->L{i+1}" for i in range(len(y))],
                           rotation=45, fontsize=8)
        ax.set_ylabel("Cosine similarity")
        ax.set_title(f"{MODEL_LABELS[model]} - M1b (inter-layer persistence, unit_norm)",
                     fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.7)
        ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=0.7)
        if row == 0:
            ax.legend(loc="best", fontsize=9)
    fig.suptitle("M1b -- Token embedding persistence between adjacent layers",
                 fontsize=13, fontweight="bold")
    out = FIG_DIR / "fig3_m1b.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 4 & 5: M3
# ---------------------------------------------------------------------------

def plot_m3_version(version: str, save_name: str):
    """Three rows (models), two columns:
       column 0: r2_mlp and r2_triv overlaid (distinguishable lines).
       column 1: gap (mlp - triv) with zero axis marked.
       One curve per condition in each panel.
    """
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for row, model in enumerate(MODELS):
        ax_lr = axes[row, 0]
        ax_gap = axes[row, 1]
        for condition in CONDITIONS:
            d = load_m3(model, condition)
            mlp = d[f"r2_mlp_{version}"]
            triv = d[f"r2_triv_{version}"]
            gap = d[f"gap_{version}"]
            x = np.arange(len(mlp))
            color = COND_COLORS[condition]
            ax_lr.plot(x, mlp, marker="o", markersize=4, color=color,
                       label=f"{COND_LABELS[condition]} - MLP", linewidth=1.6)
            ax_lr.plot(x, triv, marker="s", markersize=4, color=color,
                       linestyle="--",
                       label=f"{COND_LABELS[condition]} - Trivial",
                       linewidth=1.2, alpha=0.7)
            ax_gap.plot(x, gap, marker="o", markersize=4, color=color,
                        label=COND_LABELS[condition], linewidth=1.6)
        for ax in (ax_lr, ax_gap):
            ax.set_xticks(np.arange(len(mlp)))
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.3)
            ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=0.7)
        ax_lr.set_ylabel(f"R^2 ({version})")
        ax_lr.set_title(f"{MODEL_LABELS[model]} - MLP vs Trivial ({version})",
                        fontsize=10)
        ax_gap.set_ylabel(f"Gap = R^2_MLP - R^2_Trivial ({version})")
        ax_gap.set_title(f"{MODEL_LABELS[model]} - Gap ({version})",
                         fontsize=10)
        if row == 0:
            ax_lr.legend(loc="best", fontsize=7)
            ax_gap.legend(loc="best", fontsize=9)
    fig.suptitle(f"M3 -- Reconstructibility from sentence context ({version})",
                 fontsize=13, fontweight="bold")
    out = FIG_DIR / save_name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    print(f"Output in: {FIG_DIR}")
    print("\nFig 1: Global PR")
    plot_global_metric("pr", "PR -- Participation Ratio per layer", "fig1_pr.png")
    print("\nFig 2: Global ID")
    plot_global_metric("id", "ID -- Intrinsic Dimensionality (TwoNN) per layer",
                       "fig2_id.png")
    print("\nFig 3: M1b")
    plot_m1b()
    print("\nFig 4: M3 raw")
    plot_m3_version("raw", "fig4_m3_raw.png")
    print("\nFig 5: M3 unit")
    plot_m3_version("unit", "fig5_m3_unit.png")
    print("\nVisualization complete.")


if __name__ == "__main__":
    main()