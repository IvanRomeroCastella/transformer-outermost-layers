"""
metrics_global.py

Global geometric metrics per layer: Participation Ratio (PR) and Intrinsic
Dimensionality via TwoNN. Computed in both raw and unit-normalized variants.

- PR via torch.linalg.svdvals on CUDA.
- TwoNN via torch.cdist + torch.topk on CUDA.
- Skips files that already exist in results/.

Output:
  results/global_{model}_{condition}.npz  with keys:
    pr_raw   [n_layers]   Participation Ratio on raw embeddings
    pr_unit  [n_layers]   Participation Ratio on unit-normalized embeddings
    id_raw   [n_layers]   TwoNN intrinsic dimensionality on raw
    id_unit  [n_layers]   TwoNN intrinsic dimensionality on unit-normalized

Usage:
    python metrics_global.py
"""

import time
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["distilbert", "bert", "gpt2"]
CONDITIONS = ["coherent", "permuted", "random"]


@torch.no_grad()
def participation_ratio_gpu(X: torch.Tensor) -> float:
    """
    Participation Ratio of point cloud X (n_points, d) on GPU.
    PR = (sum eigenvalues)^2 / sum(eigenvalues^2)
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    # svdvals returns only the singular values; cheaper than full svd.
    s = torch.linalg.svdvals(X_centered)
    eig = (s ** 2) / max(X_centered.shape[0] - 1, 1)
    num = eig.sum() ** 2
    den = (eig ** 2).sum()
    if den.item() <= 0:
        return float("nan")
    return (num / den).item()


@torch.no_grad()
def two_nn_id_gpu(X: torch.Tensor) -> float:
    """
    TwoNN intrinsic dimensionality estimator on GPU.
    """
    # Pairwise distances. For N=10000 in 768D:
    # cdist in float32, matrix (10000, 10000) = 400 MB in VRAM.
    D = torch.cdist(X, X, p=2)

    # Diagonal set to +inf to exclude self-distances.
    D.fill_diagonal_(float("inf"))

    # Two nearest neighbors per row.
    # topk with largest=False gives k smallest; sorted=True guarantees r1 <= r2.
    topk_vals, _ = torch.topk(D, k=2, dim=1, largest=False, sorted=True)
    r1 = topk_vals[:, 0]
    r2 = topk_vals[:, 1]

    # Free the distance matrix before continuing.
    del D
    torch.cuda.empty_cache()

    valid = (r1 > 0) & (r2 > r1)
    n_valid = int(valid.sum().item())
    n_total = r1.shape[0]
    if n_valid < 0.5 * n_total:
        print(f"    [!] TwoNN: only {n_valid}/{n_total} valid points.")
    if n_valid == 0:
        return float("nan")

    mu = r2[valid] / r1[valid]
    log_mu = torch.log(mu)
    d_hat = n_valid / log_mu.sum().item()
    return float(d_hat)


@torch.no_grad()
def unit_normalize_gpu(X: torch.Tensor) -> torch.Tensor:
    """Normalize rows to unit magnitude (with guard for zero vectors)."""
    norms = X.norm(dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return X / norms


def process_file(model: str, condition: str):
    out_path = RESULTS_DIR / f"global_{model}_{condition}.npz"
    if out_path.exists():
        print(f"\n--- {model} / {condition} --- [SKIP, already exists]")
        return

    path = DATA_DIR / model / f"embeddings_{condition}.npz"
    print(f"\n--- {model} / {condition} ---")
    print(f"  Loading: {path}")
    t0 = time.time()
    data = np.load(path)
    embeddings = data["embeddings"]  # (n_layers, 500, 20, 768)
    print(f"  Shape: {embeddings.shape}  (loaded in {time.time() - t0:.1f}s)")

    n_layers = embeddings.shape[0]
    pr_raw = np.zeros(n_layers)
    pr_unit = np.zeros(n_layers)
    id_raw = np.zeros(n_layers)
    id_unit = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        X_np = embeddings[layer_idx].reshape(-1, embeddings.shape[-1])
        X = torch.from_numpy(X_np).to(DEVICE, dtype=torch.float32)
        X_unit = unit_normalize_gpu(X)

        t_layer = time.time()
        pr_raw[layer_idx] = participation_ratio_gpu(X)
        pr_unit[layer_idx] = participation_ratio_gpu(X_unit)
        id_raw[layer_idx] = two_nn_id_gpu(X)
        id_unit[layer_idx] = two_nn_id_gpu(X_unit)

        del X, X_unit
        torch.cuda.empty_cache()

        print(f"  L{layer_idx}: "
              f"PR_raw={pr_raw[layer_idx]:7.2f}  "
              f"PR_unit={pr_unit[layer_idx]:7.2f}  "
              f"ID_raw={id_raw[layer_idx]:6.2f}  "
              f"ID_unit={id_unit[layer_idx]:6.2f}  "
              f"({time.time() - t_layer:.1f}s)")

    np.savez(
        out_path,
        pr_raw=pr_raw,
        pr_unit=pr_unit,
        id_raw=id_raw,
        id_unit=id_unit,
    )
    print(f"  Saved: {out_path}")


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_free, vram_total = torch.cuda.mem_get_info()
        print(f"VRAM free / total: {vram_free / 1e9:.2f} / {vram_total / 1e9:.2f} GB")

    t_start = time.time()
    for model in MODELS:
        for condition in CONDITIONS:
            process_file(model, condition)
    print(f"\n{'=' * 60}")
    print(f"Global metrics completed in {(time.time() - t_start) / 60:.1f} min.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()