"""
metrics_m3.py

M3: reconstructibility of a token's embedding from its sentence context.

For each (model, condition, layer k, version in {raw, unit}):
  For each position i in 0..19:
    Build dataset:
      X = positional concatenation of the 19 content embeddings at all
          positions except i.  shape (n_sentences, 19 * 768)
      y = embedding at position i.  shape (n_sentences, 768)
    Train/val split 80/20 at the sentence level (not at the token level).
    Train MLP: linear(input -> 512) -> GELU -> dropout(0.1) -> linear(512 -> 768).
    Report mean R^2 over all positions i (i=0..19).

We also evaluate a trivial baseline (predict y as the mean of the 19 context
neighbors) and report its R^2 next to the MLP's. This separates "information
present in the context" from "compact geometry".

Hyperparameters:
  HIDDEN_MLP = 512
  DROPOUT = 0.1
  GELU activation
  Adam, LR = 1e-4
  BATCH_SIZE = 32
  MAX_EPOCHS = 100, PATIENCE = 5
  TRAIN_FRACTION = 0.8
  SEED = 42

Implementation note: instead of training one MLP per (model, condition,
layer, version, position) -- which would be ~4000 MLPs total -- we train
ONE MLP per (model, condition, layer, version) that consumes the
position-aware concatenation of context neighbors. The aggregated dataset
has 500 * 20 = 10000 examples (X = concat of 19 neighbors, y = target
embedding). This brings the total down to:
  DistilBERT: 7 * 3 * 2 = 42 MLPs
  BERT:       13 * 3 * 2 = 78 MLPs
  GPT-2:      13 * 3 * 2 = 78 MLPs
  Total:      198 MLPs

Each one trains on ~8000 examples. Tractable.

Input:
  data/{model}/embeddings_{condition}.npz  with shape (n_layers, 500, 20, 768)

Output:
  results/m3_{model}_{condition}.npz  with keys:
    r2_mlp_raw       [n_layers]    MLP R^2 on raw embeddings
    r2_mlp_unit      [n_layers]    MLP R^2 on unit-normalized embeddings
    r2_triv_raw      [n_layers]    Trivial baseline R^2 on raw
    r2_triv_unit     [n_layers]    Trivial baseline R^2 on unit-normalized
    gap_raw          [n_layers]    r2_mlp_raw - r2_triv_raw
    gap_unit         [n_layers]    r2_mlp_unit - r2_triv_unit

Usage:
    python metrics_m3.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["distilbert", "bert", "gpt2"]
CONDITIONS = ["coherent", "permuted", "random"]

# Hyperparameters.
SEED = 42
N_CONTENT = 20
HIDDEN_DIM = 768
HIDDEN_MLP = 512
DROPOUT = 0.1
LR = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 5
TRAIN_FRACTION = 0.8
EPS = 1e-12


class M3MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def unit_normalize_np(X: np.ndarray) -> np.ndarray:
    """Normalize rows to unit magnitude (with guard)."""
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def build_dataset_for_layer(layer_emb: np.ndarray):
    """
    layer_emb: (n_sentences, n_tokens, d_model) -- embeddings from ONE layer.

    Returns:
      X: (n_sentences * n_tokens, (n_tokens-1) * d_model) -- neighbors concatenated.
      y: (n_sentences * n_tokens, d_model) -- target.
      sent_ids: (n_sentences * n_tokens,) -- sentence id per example (for split).
    """
    n_sent, n_tok, d = layer_emb.shape
    Xs = []
    ys = []
    sent_ids = []
    for i in range(n_tok):
        # Neighbors = all positions except i, preserving positional order.
        neighbor_idx = [j for j in range(n_tok) if j != i]
        neighbors = layer_emb[:, neighbor_idx, :]  # (n_sent, n_tok-1, d)
        X_i = neighbors.reshape(n_sent, (n_tok - 1) * d)
        y_i = layer_emb[:, i, :]                   # (n_sent, d)
        Xs.append(X_i)
        ys.append(y_i)
        sent_ids.append(np.arange(n_sent))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sent_ids = np.concatenate(sent_ids, axis=0)
    return X, y, sent_ids


def split_by_sentence(X, y, sent_ids, n_sentences, train_fraction, seed):
    """80/20 split at the sentence level (all positions of a sentence go together)."""
    rng = np.random.default_rng(seed)
    n_train_sentences = int(n_sentences * train_fraction)
    sent_perm = rng.permutation(n_sentences)
    train_sentences = set(sent_perm[:n_train_sentences].tolist())
    train_mask = np.array([s in train_sentences for s in sent_ids])
    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]


def r2_score_multivariate(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Multivariate R^2 (sum of variances across output dimensions).
    R^2 = 1 - SS_res / SS_tot,
    with  SS_res = sum_{ij}(y_true_ij - y_pred_ij)^2,
          SS_tot = sum_{ij}(y_true_ij - mean(y_true_j))^2.
    """
    ss_res = ((y_true - y_pred) ** 2).sum()
    y_mean = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum()
    if ss_tot.item() <= 0:
        return float("nan")
    return (1 - ss_res / ss_tot).item()


@torch.no_grad()
def trivial_baseline_r2(X_val: np.ndarray, y_val: np.ndarray, n_tokens_context: int, d: int) -> float:
    """
    Trivial baseline: predict y as the mean of the n_tokens_context neighbors.
    X_val: (n, n_tokens_context * d)
    y_val: (n, d)
    """
    n = X_val.shape[0]
    X_reshaped = X_val.reshape(n, n_tokens_context, d)
    y_pred = X_reshaped.mean(axis=1)  # (n, d)
    y_true_t = torch.from_numpy(y_val).float()
    y_pred_t = torch.from_numpy(y_pred).float()
    return r2_score_multivariate(y_true_t, y_pred_t)


def train_mlp_one(layer_emb: np.ndarray, version_label: str, layer_idx: int) -> dict:
    """
    layer_emb: (n_sentences, n_tokens, d) -- embeddings from one layer,
    already in raw or unit-normalized form.
    Returns dict with r2_mlp and r2_triv for this layer/version.
    """
    n_sent, n_tok, d = layer_emb.shape
    X, y, sent_ids = build_dataset_for_layer(layer_emb)
    X_train, y_train, X_val, y_val = split_by_sentence(
        X, y, sent_ids, n_sentences=n_sent,
        train_fraction=TRAIN_FRACTION, seed=SEED,
    )

    # Trivial baseline on validation.
    r2_triv = trivial_baseline_r2(X_val, y_val, n_tokens_context=n_tok - 1, d=d)

    # MLP.
    torch.manual_seed(SEED)
    input_dim = (n_tok - 1) * d
    model = M3MLP(input_dim, HIDDEN_MLP, d, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train).float().to(DEVICE)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).float().to(DEVICE)

    n_train = X_train_t.shape[0]
    best_val_loss = float("inf")
    epochs_without_improve = 0
    best_r2 = float("nan")

    for epoch in range(MAX_EPOCHS):
        # Shuffle.
        perm = torch.randperm(n_train, device=DEVICE)
        model.train()
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            batch_idx = perm[start:end]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval on validation.
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
            r2_now = r2_score_multivariate(y_val_t, val_pred)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_r2 = r2_now
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

    del model, optimizer, X_train_t, y_train_t, X_val_t, y_val_t
    torch.cuda.empty_cache()

    return {"r2_mlp": best_r2, "r2_triv": r2_triv, "epoch_stop": epoch + 1}


def process_file(model_name: str, condition: str):
    out_path = RESULTS_DIR / f"m3_{model_name}_{condition}.npz"
    if out_path.exists():
        print(f"\n--- {model_name} / {condition} --- [SKIP, already exists]")
        return

    path = DATA_DIR / model_name / f"embeddings_{condition}.npz"
    print(f"\n--- {model_name} / {condition} ---")
    print(f"  Loading: {path}")
    t0 = time.time()
    data = np.load(path)
    embeddings = data["embeddings"]
    n_layers, n_sent, n_tok, d = embeddings.shape
    print(f"  Shape: {embeddings.shape}  (loaded in {time.time() - t0:.1f}s)")

    r2_mlp_raw = np.zeros(n_layers)
    r2_mlp_unit = np.zeros(n_layers)
    r2_triv_raw = np.zeros(n_layers)
    r2_triv_unit = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        layer_raw = embeddings[layer_idx]                       # (500, 20, 768)
        layer_unit = unit_normalize_np(layer_raw)               # unit-normalized

        t_l = time.time()
        out_raw = train_mlp_one(layer_raw, "raw", layer_idx)
        t_raw = time.time() - t_l

        t_l = time.time()
        out_unit = train_mlp_one(layer_unit, "unit", layer_idx)
        t_unit = time.time() - t_l

        r2_mlp_raw[layer_idx] = out_raw["r2_mlp"]
        r2_triv_raw[layer_idx] = out_raw["r2_triv"]
        r2_mlp_unit[layer_idx] = out_unit["r2_mlp"]
        r2_triv_unit[layer_idx] = out_unit["r2_triv"]

        print(f"  L{layer_idx}: "
              f"raw[mlp={r2_mlp_raw[layer_idx]:+.4f}, triv={r2_triv_raw[layer_idx]:+.4f}, "
              f"gap={r2_mlp_raw[layer_idx] - r2_triv_raw[layer_idx]:+.4f}, "
              f"stop@{out_raw['epoch_stop']:>3d}, {t_raw:.1f}s]  |  "
              f"unit[mlp={r2_mlp_unit[layer_idx]:+.4f}, triv={r2_triv_unit[layer_idx]:+.4f}, "
              f"gap={r2_mlp_unit[layer_idx] - r2_triv_unit[layer_idx]:+.4f}, "
              f"stop@{out_unit['epoch_stop']:>3d}, {t_unit:.1f}s]")

    gap_raw = r2_mlp_raw - r2_triv_raw
    gap_unit = r2_mlp_unit - r2_triv_unit

    np.savez(
        out_path,
        r2_mlp_raw=r2_mlp_raw,
        r2_mlp_unit=r2_mlp_unit,
        r2_triv_raw=r2_triv_raw,
        r2_triv_unit=r2_triv_unit,
        gap_raw=gap_raw,
        gap_unit=gap_unit,
    )
    print(f"  Saved: {out_path}")


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    t_start = time.time()
    for model_name in MODELS:
        for condition in CONDITIONS:
            process_file(model_name, condition)
    print(f"\n{'=' * 60}")
    print(f"M3 completed in {(time.time() - t_start) / 60:.1f} min.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()