"""
a2_budget.py -- Dimensional-budget probing (metric A2).

For each (model, layer, label_type), project the activations onto the first k
principal components (computed on the layer's training data) and train a
linear prober on k dimensions.

Metric: dimensions required to reach 80% of the A1 (full-rank) accuracy.

This measures geometric selectivity: if few dimensions suffice to recover the
information, it is "exposed" (concentrated along few high-variance directions).
If many are needed, it is "folded" (distributed).

Output:
  results/a2_budget.npz with accuracy per (model, label_type, layer, k).

Notes:
  - k values: we probe several to build the accuracy(k) curve and then
    extract k_80 (smallest k reaching 80% of the A1 accuracy).
  - PCA is fit on train, applied to test.
  - Prober hyperparameters: same as A1.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse A1 utilities.
from a1_probing import (
    LABEL_CONFIG, MODELS,
    load_probing_data,
    train_prober,
    BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, VAL_FRACTION,
)

RESULTS_DIR = Path("results")

# k values to probe (dimensional budget).
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768]


# -----------------------------------------------------------------------------
# PCA on GPU to avoid large CPU<->GPU transfers
# -----------------------------------------------------------------------------

def fit_pca_gpu(X: torch.Tensor, n_components: int = None, device: str = "cuda"):
    """Fit PCA on X (n_samples, d_model) on GPU.

    Returns:
        mean: (d_model,)
        components: (n_components, d_model) -- eigenvectors sorted by descending variance
        explained_variance: (n_components,)
    """
    X = X.to(device)
    mean = X.mean(dim=0)
    X_centered = X - mean

    # SVD: X_centered = U S V^T -> components are columns of V (rows of V^T).
    # For n_samples >> d_model this is fine. For 30k x 768 still tractable.
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # Vh: (min(n,d), d). Components = rows of Vh, sorted by descending singular value.

    if n_components is None:
        n_components = Vh.shape[0]

    components = Vh[:n_components]  # (n_components, d_model)
    explained_variance = (S[:n_components] ** 2) / (X.shape[0] - 1)

    return mean, components, explained_variance


def transform_pca(X: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Project X onto the principal components.

    Args:
        X: (n_samples, d_model)
        mean: (d_model,)
        components: (k, d_model)

    Returns:
        X_proj: (n_samples, k)
    """
    return (X - mean) @ components.T


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------

def run_budget_for_layer(activations_layer, labels, train_idx, val_idx, test_idx,
                         n_classes: int, k_values: list, device: str = "cuda",
                         baseline_acc: float = None):
    """For a given layer, compute prober accuracy at several values of k.

    Args:
        activations_layer: (n_tokens, d_model) np.ndarray
        labels: (n_tokens,) np.ndarray
        train/val/test_idx: indices.
        n_classes: prober output classes.
        k_values: list of k to probe.
        baseline_acc: A1 accuracy (full-rank), used to compute k_80.

    Returns:
        dict with accs_by_k, k_80.
    """
    X_train_full = torch.tensor(activations_layer[train_idx], dtype=torch.float32, device=device)
    X_val_full = torch.tensor(activations_layer[val_idx], dtype=torch.float32, device=device)
    X_test_full = torch.tensor(activations_layer[test_idx], dtype=torch.float32, device=device)

    y_train = torch.tensor(labels[train_idx], dtype=torch.long, device=device)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long, device=device)
    y_test = torch.tensor(labels[test_idx], dtype=torch.long, device=device)

    # Fit PCA with max(k_values) components.
    max_k = max(k_values)
    mean, components, explained_var = fit_pca_gpu(X_train_full, n_components=max_k, device=device)

    # For each k, project and train a prober.
    accs_by_k = {}
    for k in k_values:
        if k > components.shape[0]:
            accs_by_k[k] = float('nan')
            continue

        comps_k = components[:k]
        X_train_k = transform_pca(X_train_full, mean, comps_k)
        X_val_k = transform_pca(X_val_full, mean, comps_k)
        X_test_k = transform_pca(X_test_full, mean, comps_k)

        result = train_prober(X_train_k, y_train, X_val_k, y_val,
                              X_test_k, y_test, n_classes, device=device)
        accs_by_k[k] = result["test_acc"]

    # Compute k_80: smallest k that reaches 80% of baseline_acc.
    k_80 = None
    if baseline_acc is not None and baseline_acc > 0:
        target = 0.8 * baseline_acc
        sorted_ks = sorted(accs_by_k.keys())
        for k in sorted_ks:
            if accs_by_k[k] >= target:
                k_80 = k
                break

    return {
        "accs_by_k": accs_by_k,
        "k_80": k_80,
        "explained_variance_cumsum": np.cumsum(explained_var.cpu().numpy()).tolist(),
    }


def run_budget(model_name: str, label_type: str, a1_results: dict,
               device: str = "cuda"):
    """Run A2 over all layers of a (model, label_type)."""
    cfg = LABEL_CONFIG[label_type]
    dataset_name = cfg["dataset"]
    n_classes = cfg["n_classes"]
    label_key = cfg["label_key"]

    data = load_probing_data(model_name, dataset_name)
    activations = data["activations"]
    split = data["split"]
    labels = data[label_key]

    n_layers = activations.shape[0]

    train_mask = (split == "train")
    test_mask = (split == "test")
    train_indices = np.where(train_mask)[0]
    n_val = int(len(train_indices) * VAL_FRACTION)
    val_indices = train_indices[-n_val:]
    train_indices = train_indices[:-n_val]
    test_indices = np.where(test_mask)[0]

    # A1 baselines.
    a1_key = f"{model_name}__{label_type}__layer_test_acc"
    a1_accs = a1_results[a1_key]  # (n_layers,)

    layer_results = []
    print(f"  layers (k_80): ", end="", flush=True)
    for k_layer in range(n_layers):
        baseline_acc = float(a1_accs[k_layer])
        res = run_budget_for_layer(
            activations[k_layer], labels,
            train_indices, val_indices, test_indices,
            n_classes, K_VALUES, device=device,
            baseline_acc=baseline_acc,
        )
        layer_results.append(res)
        print(f"{k_layer}:{res['k_80']} ", end="", flush=True)
    print()

    return {
        "model": model_name,
        "label_type": label_type,
        "n_layers": n_layers,
        "k_values": K_VALUES,
        "layer_results": layer_results,
        "a1_accs": a1_accs.tolist(),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU. Cancel and verify CUDA.")
        import sys
        sys.exit(1)

    print(f"Device: {device}")
    torch.manual_seed(42)

    # Load A1 results.
    a1_path = RESULTS_DIR / "a1_probing.npz"
    if not a1_path.exists():
        raise FileNotFoundError(f"A1 results not found at {a1_path}. Run a1_probing.py first.")
    a1_results = dict(np.load(a1_path, allow_pickle=True))
    print(f"A1 results loaded from {a1_path}")

    t_total = time.time()
    all_results = {}

    for model_name in MODELS:
        for label_type in LABEL_CONFIG.keys():
            print(f"\n--- {model_name} / {label_type} ---")
            t0 = time.time()
            result = run_budget(model_name, label_type, a1_results, device=device)
            elapsed = time.time() - t0
            print(f"  time: {elapsed:.1f}s")
            all_results[f"{model_name}__{label_type}"] = result

    # Save.
    out_path = RESULTS_DIR / "a2_budget.npz"
    save_dict = {}
    for key, res in all_results.items():
        prefix = key
        n_layers = res["n_layers"]
        k_values = res["k_values"]

        # Table (n_layers, n_k) of accuracy.
        accs_table = np.full((n_layers, len(k_values)), np.nan, dtype=np.float32)
        k_80_array = np.full(n_layers, -1, dtype=np.int32)  # -1 = not reached
        for li, layer_res in enumerate(res["layer_results"]):
            for ki, k in enumerate(k_values):
                if k in layer_res["accs_by_k"]:
                    accs_table[li, ki] = layer_res["accs_by_k"][k]
            if layer_res["k_80"] is not None:
                k_80_array[li] = layer_res["k_80"]

        save_dict[f"{prefix}__k_values"] = np.array(k_values)
        save_dict[f"{prefix}__accs_table"] = accs_table
        save_dict[f"{prefix}__k_80"] = k_80_array
        save_dict[f"{prefix}__a1_accs"] = np.array(res["a1_accs"])

        # Cumulative explained variance (for context).
        explained_var_table = np.array([
            layer_res["explained_variance_cumsum"][:max(k_values)]
            for layer_res in res["layer_results"]
        ])
        save_dict[f"{prefix}__explained_variance_cumsum"] = explained_var_table

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"=== A2 complete in {(time.time() - t_total)/60:.1f} min ===")


if __name__ == "__main__":
    main()