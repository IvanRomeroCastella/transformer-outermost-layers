"""
b1_ablation.py -- Token-ablation sensitivity (metric B1).

For each (model, condition), and for each sentence, we ablate each input
position one at a time and measure how much each of the 20 content tokens'
representations changes at every layer.

Output:
  results/b1_ablation_{model}_{condition}.npz  with key:
    "delta_cos": shape (N_sentences, N_ablate_positions, N_layers, 20_content_tokens)

Notes:
  - Replacement token: [MASK] for encoders, [EOS] for GPT-2 (known asymmetry).
  - Metric: cosine distance = 1 - cos_sim(e_baseline, e_ablated).
  - We report raw and unit_norm in analyze.py.
  - Memory: full tensor occupies ~30MB per (model, condition). OK.

Compute cost:
  - DistilBERT: 500 sentences * 22 positions = 11k forward passes per condition.
  - BERT: 500 * 22 = 11k.
  - GPT-2: 500 * 20 = 10k.
  - Total ~96k passes. With a reasonable batch size on RTX 5060 Ti, ~10-20 min total.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    MODEL_SPECS, CONDITIONS,
    load_model_and_tokenizer,
    load_dataset,
    load_precomputed_embeddings,
    get_ablation_token_id,
    forward_full,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32  # sentences per batch in the ablation forward


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_ablation_tensor(model, ds, ablation_token_id: int, spec: dict, device: str):
    """Compute the sensitivity tensor for a (model, condition).

    Strategy:
      1. Get baseline embeddings (no ablation). We could reuse the precomputed
         ones, but we regenerate them in GPU from a fresh forward to ensure
         dimensional consistency and avoid unnecessary CPU<->GPU copies.
      2. For each position p in [0, seq_len):
         - Replace input_ids[:, p] = ablation_token_id (across the whole batch).
         - Forward pass.
         - For each layer k, compute cosine distance between
           hidden_baseline[k][:, content_pos] and hidden_ablated[k][:, content_pos],
           keeping per-token granularity.

    Returns:
        delta_cos: np.ndarray (N_sentences, seq_len, n_states, 20_content) float32
    """
    n_states = spec["n_states"]
    seq_len = ds.input_ids.shape[1]
    n_sentences = ds.input_ids.shape[0]
    n_content = ds.content_positions.shape[1]  # should be 20

    input_ids = ds.input_ids.to(device)
    attention_mask = ds.attention_mask.to(device)
    content_pos_t = torch.from_numpy(ds.content_positions).to(device)  # (N, 20)

    # --- Step 1: baseline on GPU ---
    # Forward in batches to avoid saturating memory with all attentions at once.
    baseline_states = [torch.empty(n_sentences, n_content, spec["d_model"],
                                    dtype=torch.float32, device=device)
                       for _ in range(n_states)]

    for start in range(0, n_sentences, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_sentences)
        batch_ids = input_ids[start:end]
        batch_mask = attention_mask[start:end]
        batch_cpos = content_pos_t[start:end]  # (b, 20)

        out = forward_full(model, batch_ids, batch_mask, device=device)
        hidden_states = out["hidden_states"]  # tuple of (b, seq_len, d_model)

        for k, hs in enumerate(hidden_states):
            # Index the 20 content tokens.
            # hs: (b, seq_len, d_model); batch_cpos: (b, 20)
            b = hs.shape[0]
            batch_idx = torch.arange(b, device=device).unsqueeze(1).expand(-1, n_content)
            baseline_states[k][start:end] = hs[batch_idx, batch_cpos]

    # --- Step 2: ablation position by position ---
    delta_cos = np.zeros((n_sentences, seq_len, n_states, n_content), dtype=np.float32)

    for p in range(seq_len):
        # Replace column p with the ablation token in ALL sentences.
        input_ids_abl = input_ids.clone()
        input_ids_abl[:, p] = ablation_token_id

        for start in range(0, n_sentences, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_sentences)
            batch_ids = input_ids_abl[start:end]
            batch_mask = attention_mask[start:end]
            batch_cpos = content_pos_t[start:end]

            out = forward_full(model, batch_ids, batch_mask, device=device)
            hidden_states = out["hidden_states"]

            for k, hs in enumerate(hidden_states):
                b = hs.shape[0]
                batch_idx = torch.arange(b, device=device).unsqueeze(1).expand(-1, n_content)
                ablated = hs[batch_idx, batch_cpos]                # (b, 20, d)
                baseline = baseline_states[k][start:end]           # (b, 20, d)

                # Cosine distance = 1 - cos_sim.
                cos_sim = F.cosine_similarity(ablated, baseline, dim=-1)  # (b, 20)
                delta = (1.0 - cos_sim).cpu().numpy()
                delta_cos[start:end, p, k, :] = delta

    return delta_cos


def run_one(model_name: str, condition: str, device: str = "cuda"):
    out_path = RESULTS_DIR / f"b1_ablation_{model_name}_{condition}.npz"
    if out_path.exists():
        print(f"  [skip] already exists: {out_path.name}")
        return

    print(f"\n--- {model_name} / {condition} ---")
    t0 = time.time()

    model, tokenizer, spec = load_model_and_tokenizer(model_name, device=device)
    ds = load_dataset(model_name, condition)
    ablation_token_id = get_ablation_token_id(tokenizer, model_name)

    print(f"  seq_len={ds.input_ids.shape[1]}, ablation_token_id={ablation_token_id}")
    print(f"  expected forward passes: {ds.input_ids.shape[0] * (1 + ds.input_ids.shape[1])} "
          f"({ds.input_ids.shape[0]} baseline + ablate-each)")

    delta_cos = compute_ablation_tensor(model, ds, ablation_token_id, spec, device=device)

    elapsed = time.time() - t0
    print(f"  delta_cos shape: {delta_cos.shape}")
    print(f"  delta_cos stats: mean={delta_cos.mean():.4f}, "
          f"max={delta_cos.max():.4f}, min={delta_cos.min():.6f}")
    print(f"  time: {elapsed:.1f}s | size: {delta_cos.nbytes / 1e6:.1f}MB")

    np.savez_compressed(out_path,
                        delta_cos=delta_cos,
                        model_name=model_name,
                        condition=condition,
                        ablation_token_id=ablation_token_id,
                        content_positions=ds.content_positions[0])
    print(f"  saved: {out_path}")

    # Free GPU.
    del model
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU. This will be VERY slow. Cancel and verify CUDA.")
        import sys
        sys.exit(1)

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    t_total = time.time()

    for model_name in ["distilbert", "bert", "gpt2"]:
        for condition in CONDITIONS:
            try:
                run_one(model_name, condition, device=device)
            except Exception as e:
                print(f"FAIL on {model_name}/{condition}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n=== B1 complete in {(time.time() - t_total)/60:.1f} min ===")