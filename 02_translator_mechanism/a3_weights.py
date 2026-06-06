"""
a3_weights.py -- Prober-weight analysis (metric A3).

Reuses the prober weights saved in A1 to compute, per layer:

  1. Norm and sparsity of the prober's weight matrix.
  2. **Key metric**: alignment of the prober's weights with the dominant
     direction of the layer's activations (PC1).
     If the weights are aligned with PC1, the information the prober uses
     lives mainly along the highest-variance direction -- a signature of
     "co-located" information (the layer concentrates discriminative content
     in the same direction it concentrates raw variance).
     If the weights are misaligned with PC1, the discriminative information
     lives along lower-variance directions -- a signature of "directional
     redistribution" (the layer scattered the discriminative content across
     many individually-low-variance directions).

Input:
  - results/a1_probing.npz: prober_weights and prober_bias per layer.
  - results/probing_data_*.npz: activations used to compute PC1.

Output:
  results/a3_weights.npz with metrics per (model, label_type, layer).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from a1_probing import LABEL_CONFIG, MODELS, load_probing_data

RESULTS_DIR = Path("results")


# -----------------------------------------------------------------------------
# Per-layer PC1
# -----------------------------------------------------------------------------

def compute_pc1_per_layer(activations: np.ndarray, device: str = "cuda", unit_norm: bool = True):
    """For each layer, compute the dominant direction (PC1) and its explained
    variance.

    Args:
        activations: (n_layers, n_tokens, d_model)
        unit_norm: if True, normalize each activation to unit length before PCA.
                   Required for pre-LN models (e.g. GPT-2) to prevent a single
                   outlier feature from dominating PC1.

    Returns:
        pc1: (n_layers, d_model) -- unit vector per layer
        explained_var_ratio: (n_layers,) -- fraction of variance in PC1
    """
    n_layers, n_tokens, d_model = activations.shape
    pc1 = np.zeros((n_layers, d_model), dtype=np.float32)
    explained_ratio = np.zeros(n_layers, dtype=np.float32)

    for k in range(n_layers):
        X = torch.tensor(activations[k], dtype=torch.float32, device=device)

        if unit_norm:
            # Normalize each vector to unit length before PCA.
            norms = torch.linalg.norm(X, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-10)
            X = X / norms

        X_centered = X - X.mean(dim=0)

        # Truncated SVD: we only need the top component.
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        pc1_k = Vh[0]  # (d_model,)
        # Ensure unit length.
        pc1_k = pc1_k / torch.linalg.norm(pc1_k)
        pc1[k] = pc1_k.cpu().numpy()

        # PC1 explained variance.
        total_var = (S ** 2).sum().item()
        pc1_var = (S[0] ** 2).item()
        explained_ratio[k] = pc1_var / total_var

    return pc1, explained_ratio


# -----------------------------------------------------------------------------
# Prober-weight metrics
# -----------------------------------------------------------------------------

def analyze_prober_weights(prober_weights: np.ndarray, pc1: np.ndarray):
    """Compute metrics on a layer's prober weights.

    Args:
        prober_weights: (n_classes, d_model)
        pc1: (d_model,) -- dominant direction of this layer's activations.

    Returns:
        dict with:
          - norm_total: total norm of the weight matrix
          - norm_per_class: per-class norm (n_classes,)
          - sparsity: fraction of weights with |w| < 0.01 * max(|w|)
          - alignment_pc1: average alignment (across classes) of the prober's
              weights with PC1, in absolute value. In [0, 1].
              =1 means all prober weights live along PC1.
              =0 means weights are perpendicular to PC1.
          - alignment_pc1_per_class: per-class alignment (n_classes,)
    """
    norm_total = np.linalg.norm(prober_weights)
    norms = np.linalg.norm(prober_weights, axis=1)  # (n_classes,)

    # Sparsity: fraction of weights near zero (relative to max).
    max_w = np.abs(prober_weights).max()
    if max_w > 0:
        sparsity = (np.abs(prober_weights) < 0.01 * max_w).mean()
    else:
        sparsity = 0.0

    # Alignment with PC1: absolute cosine between each row (class) and PC1.
    # Normalize each row to unit length.
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    weights_unit = prober_weights / norms_safe[:, np.newaxis]  # (n_classes, d_model)
    pc1_unit = pc1 / (np.linalg.norm(pc1) + 1e-10)
    alignment_per_class = np.abs(weights_unit @ pc1_unit)  # (n_classes,)
    alignment_avg = alignment_per_class.mean()

    return {
        "norm_total": float(norm_total),
        "norm_per_class": norms.tolist(),
        "sparsity": float(sparsity),
        "alignment_pc1": float(alignment_avg),
        "alignment_pc1_per_class": alignment_per_class.tolist(),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CPU.")

    print(f"Device: {device}")
    print(f"Mode: PCA on unit-normalized activations")

    # Load A1.
    a1_path = RESULTS_DIR / "a1_probing.npz"
    if not a1_path.exists():
        raise FileNotFoundError(f"A1 results not found. Run a1_probing.py first.")
    a1_results = dict(np.load(a1_path, allow_pickle=True))

    # PC1 cache (keyed by (model, dataset)).
    pc1_cache = {}

    def get_pc1(model_name: str, dataset_name: str):
        key = (model_name, dataset_name)
        if key not in pc1_cache:
            data = load_probing_data(model_name, dataset_name)
            print(f"  computing PC1 for {model_name}/{dataset_name}...")
            pc1, ratio = compute_pc1_per_layer(data["activations"], device=device)
            pc1_cache[key] = (pc1, ratio)
        return pc1_cache[key]

    t_total = time.time()
    all_results = {}

    # Random baseline: Gaussian random weight vs a fixed PC1, to calibrate
    # what counts as "high" or "low" alignment.
    # In d=768, the expected |cos| of a random unit vector with a fixed direction is:
    # E[|cos|] ~ sqrt(2 / (pi * d)) ~ 0.0288 for d=768.
    random_baseline_alignment = float(np.sqrt(2 / (np.pi * 768)))
    print(f"\nRandom baseline for alignment_pc1 at d=768: {random_baseline_alignment:.4f}")

    for model_name in MODELS:
        for label_type in LABEL_CONFIG.keys():
            cfg = LABEL_CONFIG[label_type]
            dataset_name = cfg["dataset"]

            print(f"\n--- {model_name} / {label_type} ---")
            t0 = time.time()

            # Per-layer PC1 (of the activations).
            pc1_per_layer, var_ratio = get_pc1(model_name, dataset_name)

            # Per-layer prober weights.
            weights_key = f"{model_name}__{label_type}__prober_weights"
            prober_weights_all = a1_results[weights_key]  # (n_layers, n_classes, d_model)

            n_layers = prober_weights_all.shape[0]

            metrics_per_layer = []
            print(f"  layers (alignment_pc1): ", end="", flush=True)
            for k in range(n_layers):
                m = analyze_prober_weights(prober_weights_all[k], pc1_per_layer[k])
                metrics_per_layer.append(m)
                print(f"{k}:{m['alignment_pc1']:.3f} ", end="", flush=True)
            print()

            elapsed = time.time() - t0
            print(f"  PC1 explained variance per layer: " +
                  " ".join(f"{v:.3f}" for v in var_ratio))
            print(f"  time: {elapsed:.1f}s")

            all_results[f"{model_name}__{label_type}"] = {
                "metrics_per_layer": metrics_per_layer,
                "pc1_var_ratio": var_ratio.tolist(),
                "n_layers": n_layers,
            }

    # Save.
    out_path = RESULTS_DIR / "a3_weights.npz"
    save_dict = {"random_baseline_alignment": np.array(random_baseline_alignment)}

    for key, res in all_results.items():
        prefix = key
        n_layers = res["n_layers"]
        ml = res["metrics_per_layer"]

        save_dict[f"{prefix}__alignment_pc1"] = np.array([m["alignment_pc1"] for m in ml])
        save_dict[f"{prefix}__norm_total"] = np.array([m["norm_total"] for m in ml])
        save_dict[f"{prefix}__sparsity"] = np.array([m["sparsity"] for m in ml])
        save_dict[f"{prefix}__pc1_var_ratio"] = np.array(res["pc1_var_ratio"])

        # Per-class alignment (n_layers, n_classes -- n_classes varies by label_type).
        align_per_class = np.array([m["alignment_pc1_per_class"] for m in ml])
        save_dict[f"{prefix}__alignment_pc1_per_class"] = align_per_class

        norm_per_class = np.array([m["norm_per_class"] for m in ml])
        save_dict[f"{prefix}__norm_per_class"] = norm_per_class

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"=== A3 complete in {(time.time() - t_total)/60:.1f} min ===")


if __name__ == "__main__":
    main()