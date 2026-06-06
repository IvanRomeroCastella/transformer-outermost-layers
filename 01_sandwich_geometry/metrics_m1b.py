"""
metrics_m1b.py

M1b: persistence of token embeddings between adjacent layers.

For each (model, condition, transition L_k -> L_{k+1}):
  For each (sentence s, token position i):
    cos_sim(h_{k, s, i}, h_{k+1, s, i})
  Report mean and standard deviation over the 500 * 20 = 10000
  (sentence, position) pairs.

Reported in both raw and unit_norm variants. Cosine similarity is by
construction invariant to vector magnitude, so raw and unit_norm should
produce exactly the same value when vectors are non-null. We report both
to detect numerical inconsistencies and for consistency with the rest of
the pipeline.

Input:
  data/{model}/embeddings_{condition}.npz  with shape (n_layers, 500, 20, 768)

Output:
  results/m1b_{model}_{condition}.npz  with keys:
    m1b_raw_mean   [n_transitions]   mean cos_sim per transition
    m1b_raw_std    [n_transitions]   std cos_sim per transition
    m1b_unit_mean  [n_transitions]   same in unit_norm (should == raw)
    m1b_unit_std   [n_transitions]

where n_transitions = n_layers - 1 (6 for DistilBERT, 12 for BERT/GPT-2).

Usage:
    python metrics_m1b.py
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
EPS = 1e-12


@torch.no_grad()
def cosine_similarity_per_pair(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A, B: (n_pairs, d). Returns cos_sim per pair: (n_pairs,).
    """
    a_norm = A.norm(dim=1, keepdim=True).clamp(min=EPS)
    b_norm = B.norm(dim=1, keepdim=True).clamp(min=EPS)
    A_unit = A / a_norm
    B_unit = B / b_norm
    return (A_unit * B_unit).sum(dim=1)


@torch.no_grad()
def unit_normalize_gpu(X: torch.Tensor) -> torch.Tensor:
    """Normalize vectors to unit magnitude, with guard."""
    norms = X.norm(dim=1, keepdim=True).clamp(min=EPS)
    return X / norms


def process_file(model: str, condition: str):
    out_path = RESULTS_DIR / f"m1b_{model}_{condition}.npz"
    if out_path.exists():
        print(f"\n--- {model} / {condition} --- [SKIP, already exists]")
        return

    path = DATA_DIR / model / f"embeddings_{condition}.npz"
    print(f"\n--- {model} / {condition} ---")
    print(f"  Loading: {path}")
    t0 = time.time()
    data = np.load(path)
    embeddings = data["embeddings"]  # (n_layers, 500, 20, 768)
    n_layers, n_sentences, n_tokens, d_model = embeddings.shape
    n_transitions = n_layers - 1
    print(f"  Shape: {embeddings.shape}  ({n_transitions} inter-layer transitions)")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    m1b_raw_mean = np.zeros(n_transitions)
    m1b_raw_std = np.zeros(n_transitions)
    m1b_unit_mean = np.zeros(n_transitions)
    m1b_unit_std = np.zeros(n_transitions)

    for k in range(n_transitions):
        # Flatten (500, 20, 768) -> (10000, 768) for each layer.
        A_np = embeddings[k].reshape(-1, d_model)
        B_np = embeddings[k + 1].reshape(-1, d_model)

        A = torch.from_numpy(A_np).to(DEVICE, dtype=torch.float32)
        B = torch.from_numpy(B_np).to(DEVICE, dtype=torch.float32)

        # Raw: cos_sim on vectors as-is (cos_sim already normalizes internally).
        cos_raw = cosine_similarity_per_pair(A, B)
        m1b_raw_mean[k] = cos_raw.mean().item()
        m1b_raw_std[k] = cos_raw.std().item()

        # Unit-normalize first (redundant for cos_sim but kept for consistency
        # with the rest of the pipeline and to detect numerical inconsistencies).
        A_unit = unit_normalize_gpu(A)
        B_unit = unit_normalize_gpu(B)
        cos_unit = cosine_similarity_per_pair(A_unit, B_unit)
        m1b_unit_mean[k] = cos_unit.mean().item()
        m1b_unit_std[k] = cos_unit.std().item()

        print(f"  L{k}->L{k+1}: "
              f"raw mean={m1b_raw_mean[k]:.4f} std={m1b_raw_std[k]:.4f}  |  "
              f"unit mean={m1b_unit_mean[k]:.4f} std={m1b_unit_std[k]:.4f}")

        del A, B, A_unit, B_unit, cos_raw, cos_unit
        torch.cuda.empty_cache()

    np.savez(
        out_path,
        m1b_raw_mean=m1b_raw_mean,
        m1b_raw_std=m1b_raw_std,
        m1b_unit_mean=m1b_unit_mean,
        m1b_unit_std=m1b_unit_std,
    )
    print(f"  Saved: {out_path}")


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    for model in MODELS:
        for condition in CONDITIONS:
            process_file(model, condition)
    print(f"\n{'=' * 60}")
    print(f"M1b completed in {(time.time() - t_start):.1f}s.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()