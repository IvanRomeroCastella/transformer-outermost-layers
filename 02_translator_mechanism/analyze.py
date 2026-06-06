"""
analyze.py -- Final aggregation for the translator-mechanism block.

Does three things:
  1. Loads all results (B1, A1, A2, A3).
  2. Generates figures (per-metric, plus cross-metric and summary).
  3. Dumps a JSON file with the per-layer numerical summaries used to
     describe the translator vs. core behavior (no qualitative verdicts;
     verdicts against the pre-registered hypotheses live in prereg.md).

Output:
  - results/figures/*.png
  - results/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from a1_probing import LABEL_CONFIG, MODELS

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {"distilbert": "tab:blue", "bert": "tab:green", "gpt2": "tab:red"}
COND_COLORS = {"coherent": "tab:blue", "permuted": "tab:orange", "random": "tab:gray"}
LABEL_COLORS = {"pos": "tab:purple", "freq_bucket": "tab:olive",
                "position_bucket": "tab:brown", "sentiment": "tab:cyan"}


# -----------------------------------------------------------------------------
# Result loading
# -----------------------------------------------------------------------------

def load_all():
    """Load all .npz files from this block into a single dict."""
    out = {}

    # B1
    out["b1"] = {}
    for m in MODELS:
        for c in ["coherent", "permuted", "random"]:
            path = RESULTS_DIR / f"b1_ablation_{m}_{c}.npz"
            if path.exists():
                out["b1"][(m, c)] = dict(np.load(path, allow_pickle=True))

    # A1, A2, A3
    for block_id in ["a1_probing", "a2_budget", "a3_weights"]:
        path = RESULTS_DIR / f"{block_id}.npz"
        if path.exists():
            out[block_id.split("_")[0]] = dict(np.load(path, allow_pickle=True))

    return out


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def figure_b1_layer_profile(data):
    """B1: per-layer sensitivity profile, per model, per condition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, model in zip(axes, MODELS):
        for cond in ["coherent", "permuted", "random"]:
            if (model, cond) not in data["b1"]:
                continue
            delta = data["b1"][(model, cond)]["delta_cos"]
            profile = delta.mean(axis=(0, 1, 3))
            ax.plot(range(len(profile)), profile, marker='o',
                    label=cond, color=COND_COLORS[cond])
        ax.set_title(f"{model}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean sensitivity (delta cos)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("B1 -- Ablation sensitivity, per-layer profile", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "f1_b1_layer_profile.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


def figure_b1_position_profile(data):
    """B1: per-ablated-position profile at the final layer."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, model in zip(axes, MODELS):
        for cond in ["coherent", "permuted", "random"]:
            if (model, cond) not in data["b1"]:
                continue
            delta = data["b1"][(model, cond)]["delta_cos"]
            last = delta.shape[2] - 1
            profile = delta[:, :, last, :].mean(axis=(0, 2))
            ax.plot(range(len(profile)), profile, marker='.',
                    label=cond, color=COND_COLORS[cond])
        ax.set_title(f"{model} -- final layer")
        ax.set_xlabel("Ablated position")
        ax.set_ylabel("delta cos on content tokens")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("B1 -- Sensitivity by ablated position", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "f2_b1_position_profile.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


def figure_a1_layer_profile(data):
    """A1: per-layer probing accuracy, one panel per label_type."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, label_type in zip(axes.flat, LABEL_CONFIG.keys()):
        for model in MODELS:
            key = f"{model}__{label_type}__layer_test_acc"
            if key not in data["a1"]:
                continue
            acc = data["a1"][key]
            chance_key = f"{model}__{label_type}__chance_acc"
            chance = float(data["a1"][chance_key])
            ax.plot(range(len(acc)), acc, marker='o',
                    label=model, color=MODEL_COLORS[model])
        ax.axhline(chance, ls='--', color='gray', alpha=0.5, label='chance')
        ax.set_title(f"A1 -- {label_type}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Test accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("A1 -- Per-layer classical probing", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "f3_a1_layer_profile.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


def figure_a2_k80(data):
    """A2: per-layer k_80, per model, per label_type."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, label_type in zip(axes.flat, LABEL_CONFIG.keys()):
        for model in MODELS:
            key = f"{model}__{label_type}__k_80"
            if key not in data["a2"]:
                continue
            k80 = data["a2"][key]
            mask = k80 >= 0
            ax.plot(np.arange(len(k80))[mask], k80[mask], marker='o',
                    label=model, color=MODEL_COLORS[model])
        ax.set_title(f"A2 -- {label_type}: per-layer k_80")
        ax.set_xlabel("Layer")
        ax.set_ylabel("k_80 (dimensions to reach 80% accuracy)")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("A2 -- Dimensional selectivity", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "f4_a2_k80.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


def figure_a3_alignment(data):
    """A3: per-layer PC1 alignment."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, label_type in zip(axes.flat, LABEL_CONFIG.keys()):
        for model in MODELS:
            key = f"{model}__{label_type}__alignment_pc1"
            if key not in data["a3"]:
                continue
            align = data["a3"][key]
            ax.plot(range(len(align)), align, marker='o',
                    label=model, color=MODEL_COLORS[model])

        # Random baseline.
        rb = float(data["a3"]["random_baseline_alignment"])
        ax.axhline(rb, ls='--', color='gray', alpha=0.5, label='random baseline')

        ax.set_title(f"A3 -- {label_type}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("alignment_pc1 (|cos|)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("A3 -- Prober-weight alignment with PC1", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "f5_a3_alignment.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


def figure_pc1_var_ratio(data):
    """PC1 variance ratio per layer, per model."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for model in MODELS:
        # pc1_var_ratio is dataset-dependent (not label-dependent); we take the
        # POS row which uses UD-EWT for all models.
        key = f"{model}__pos__pc1_var_ratio"
        if key not in data["a3"]:
            continue
        var = data["a3"][key]
        ax.plot(range(len(var)), var, marker='o',
                label=f"{model} (UD-EWT)", color=MODEL_COLORS[model])

    ax.set_xlabel("Layer")
    ax.set_ylabel("PC1 variance ratio")
    ax.set_title("Per-layer PC1 variance ratio (unit_norm)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "f6_pc1_var_ratio.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


def figure_cross_a1_b1(data):
    """A1 x B1 cross-view: do layers where A1 drops coincide with layers
    where B1 rises?

    For each model, in the coherent condition (B1) and POS (A1), we normalize
    both curves to [0, 1] and overlay them.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, model in zip(axes, MODELS):
        # B1 coherent.
        if (model, "coherent") not in data["b1"]:
            continue
        delta = data["b1"][(model, "coherent")]["delta_cos"]
        b1_profile = delta.mean(axis=(0, 1, 3))
        b1_norm = (b1_profile - b1_profile.min()) / (b1_profile.max() - b1_profile.min() + 1e-10)

        # A1 POS.
        key = f"{model}__pos__layer_test_acc"
        a1_acc = data["a1"][key]
        a1_norm = (a1_acc - a1_acc.min()) / (a1_acc.max() - a1_acc.min() + 1e-10)

        n_layers_b1 = len(b1_norm)
        n_layers_a1 = len(a1_norm)
        ax.plot(range(n_layers_b1), b1_norm, marker='o',
                label='B1 sensitivity (normalized)', color='tab:red')
        ax.plot(range(n_layers_a1), a1_norm, marker='s',
                label='A1 POS accuracy (normalized)', color='tab:blue')
        ax.set_title(f"{model} -- A1 (POS) x B1 (coherent)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized value")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("A1 x B1 cross view", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "f7_cross_a1_b1.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


def figure_translator_summary(data):
    """Per-model visual table of what the entry vs. exit translators do.

    Four metrics per translator: delta A1(POS), delta A1(position),
    delta k_80(POS), delta alignment(freq_bucket).
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    rows = []
    row_labels = []

    for model in MODELS:
        # Layer indices for entry / core / exit.
        if model == "distilbert":
            l_in, l_core_left, l_core_right, l_out = 0, 1, 5, 6
        else:
            l_in, l_core_left, l_core_right, l_out = 0, 1, 11, 12

        # A1 POS: entry-translator delta  = acc(l_core_left) - acc(l_in)
        # A1 POS: exit-translator  delta  = acc(l_out)       - acc(l_core_right)
        a1_pos = data["a1"][f"{model}__pos__layer_test_acc"]
        a1_position = data["a1"][f"{model}__position_bucket__layer_test_acc"]
        d_a1_pos_in = a1_pos[l_core_left] - a1_pos[l_in]
        d_a1_pos_out = a1_pos[l_out] - a1_pos[l_core_right]
        d_a1_position_in = a1_position[l_core_left] - a1_position[l_in]
        d_a1_position_out = a1_position[l_out] - a1_position[l_core_right]

        # k_80 POS.
        k80_pos = data["a2"][f"{model}__pos__k_80"]
        d_k80_pos_in = float(k80_pos[l_core_left]) - float(k80_pos[l_in])
        d_k80_pos_out = float(k80_pos[l_out]) - float(k80_pos[l_core_right])

        # alignment freq_bucket.
        align = data["a3"][f"{model}__freq_bucket__alignment_pc1"]
        d_align_in = align[l_core_left] - align[l_in]
        d_align_out = align[l_out] - align[l_core_right]

        rows.append([
            d_a1_pos_in, d_a1_pos_out,
            d_a1_position_in, d_a1_position_out,
            d_k80_pos_in, d_k80_pos_out,
            d_align_in, d_align_out,
        ])
        row_labels.append(model)

    rows = np.array(rows)
    columns = [
        "d POS\nentry", "d POS\nexit",
        "d position\nentry", "d position\nexit",
        "d k_80(POS)\nentry", "d k_80(POS)\nexit",
        "d align(freq)\nentry", "d align(freq)\nexit",
    ]

    # Render as a colored table.
    ax.axis('off')
    table = ax.table(cellText=[[f"{v:+.3f}" if abs(v) < 5 else f"{v:+.0f}" for v in row] for row in rows],
                     rowLabels=row_labels,
                     colLabels=columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color cells by sign.
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = table[i + 1, j]
            if v > 0:
                cell.set_facecolor((0.85, 1.0, 0.85))  # light green
            elif v < 0:
                cell.set_facecolor((1.0, 0.85, 0.85))  # light red

    fig.suptitle("Translator summary: changes between extreme layer and core", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "f8_translator_summary.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    return out


# -----------------------------------------------------------------------------
# Numerical summary (JSON)
# -----------------------------------------------------------------------------

def build_summary(data) -> dict:
    """Build a numerical summary of the translator vs. core behavior.

    No qualitative judgments. Just the numbers used by the paper text and
    by prereg.md to evaluate the pre-registered hypotheses.
    """
    summary = {"models": {}}

    for model in MODELS:
        if model == "distilbert":
            l_in, l_out = 0, 6
            core_slice = slice(2, 5)   # L2..L4 inclusive
        else:
            l_in, l_out = 0, 12
            core_slice = slice(2, 10)  # L2..L9 inclusive

        m_entry = {}
        m_exit = {}
        m_core = {}

        # A1 deltas per label type.
        for label_type in LABEL_CONFIG.keys():
            key = f"{model}__{label_type}__layer_test_acc"
            if key not in data.get("a1", {}):
                continue
            acc = np.asarray(data["a1"][key], dtype=float)
            m_entry[f"a1_{label_type}_delta"] = float(acc[1] - acc[l_in])
            m_exit[f"a1_{label_type}_delta"] = float(acc[l_out] - acc[l_out - 1])
            m_core[f"a1_{label_type}_mean"] = float(np.nanmean(acc[core_slice]))

        # A2 k_80 (POS and freq_bucket).
        for label_type in ["pos", "freq_bucket"]:
            key = f"{model}__{label_type}__k_80"
            if key not in data.get("a2", {}):
                continue
            k80 = np.asarray(data["a2"][key], dtype=float)
            valid_core = k80[core_slice][k80[core_slice] >= 0]
            core_med = float(np.median(valid_core)) if valid_core.size else float("nan")
            m_core[f"a2_{label_type}_k80_median"] = core_med
            m_entry[f"a2_{label_type}_k80"] = float(k80[1])
            m_exit[f"a2_{label_type}_k80"] = float(k80[l_out])

        # A3 alignment_pc1 (POS and freq_bucket).
        for label_type in ["pos", "freq_bucket"]:
            key = f"{model}__{label_type}__alignment_pc1"
            if key not in data.get("a3", {}):
                continue
            align = np.asarray(data["a3"][key], dtype=float)
            m_entry[f"a3_{label_type}_alignment"] = float(align[1])
            m_exit[f"a3_{label_type}_alignment"] = float(align[l_out])
            m_core[f"a3_{label_type}_alignment_mean"] = float(np.nanmean(align[core_slice]))

        # PC1 variance ratio across all layers (UD-EWT side).
        pc1_key = f"{model}__pos__pc1_var_ratio"
        if pc1_key in data.get("a3", {}):
            pc1_var = np.asarray(data["a3"][pc1_key], dtype=float).tolist()
        else:
            pc1_var = None

        # B1 mean sensitivity profile per condition.
        b1_profiles = {}
        for cond in ["coherent", "permuted", "random"]:
            if (model, cond) in data.get("b1", {}):
                delta = data["b1"][(model, cond)]["delta_cos"]
                b1_profiles[cond] = delta.mean(axis=(0, 1, 3)).tolist()

        # B1 final-layer Spearman rho between ablated position and sensitivity.
        rho_by_cond = {}
        for cond in ["coherent", "permuted", "random"]:
            if (model, cond) not in data.get("b1", {}):
                continue
            delta = data["b1"][(model, cond)]["delta_cos"]
            last = delta.shape[2] - 1
            sens_by_pos = delta[:, :, last, :].mean(axis=(0, 2))
            positions = np.arange(len(sens_by_pos))
            r_pos = np.argsort(np.argsort(positions))
            r_sens = np.argsort(np.argsort(sens_by_pos))
            n_eff = len(positions)
            if n_eff > 1:
                rho = 1 - 6 * ((r_pos - r_sens) ** 2).sum() / (n_eff * (n_eff ** 2 - 1))
            else:
                rho = 0.0
            rho_by_cond[cond] = float(rho)

        summary["models"][model] = {
            "layers": {"entry": l_in, "exit": l_out, "core_slice": [core_slice.start, core_slice.stop]},
            "entry_translator": m_entry,
            "exit_translator": m_exit,
            "core": m_core,
            "pc1_var_ratio_per_layer": pc1_var,
            "b1_layer_profile": b1_profiles,
            "b1_final_layer_spearman_pos_vs_sensitivity": rho_by_cond,
        }

    # Random-baseline reference for A3 alignment.
    if "a3" in data and "random_baseline_alignment" in data["a3"]:
        summary["a3_random_baseline_alignment"] = float(data["a3"]["random_baseline_alignment"])

    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading all results...")
    data = load_all()
    print(f"  b1: {len(data.get('b1', {}))} (model, condition) combinations")
    print(f"  a1, a2, a3 loaded")

    print("\nGenerating figures...")
    figs = []
    figs.append(figure_b1_layer_profile(data))
    figs.append(figure_b1_position_profile(data))
    figs.append(figure_a1_layer_profile(data))
    figs.append(figure_a2_k80(data))
    figs.append(figure_a3_alignment(data))
    figs.append(figure_pc1_var_ratio(data))
    figs.append(figure_cross_a1_b1(data))
    figs.append(figure_translator_summary(data))
    for f in figs:
        print(f"  {f}")

    print("\nBuilding numerical summary...")
    summary = build_summary(data)
    out = RESULTS_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {out}")

    print("\n=== analyze complete ===")


if __name__ == "__main__":
    main()