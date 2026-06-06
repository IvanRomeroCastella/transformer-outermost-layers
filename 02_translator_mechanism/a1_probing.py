"""
a1_probing.py -- Classical layer-wise probing (metric A1).

For each (model, layer, label type), train a linear classifier on activations
precomputed by extract_probing_data.py and report test accuracy.

Form A: clean / unperturbed data only (UD-EWT and SST-2 as released).

Label types:
  - pos             (UD-EWT, 17 classes): syntactic / morphological information.
  - freq_bucket     (UD-EWT, 3 classes):  lexical proxy (bucketized frequency).
  - position_bucket (UD-EWT, 3 classes):  positional information.
  - sentiment       (SST-2, 2 classes):   semantic information.

Prober architecture:
  - Single linear layer d_model -> n_classes (with bias)
  - Adam lr=1e-3, weight_decay=1e-4
  - Cross-entropy loss
  - Batch 256, max 50 epochs, early stopping with patience 5 on validation

Metric: test accuracy.

Also reports: chance accuracy (1/n_classes), majority-class baseline.

Output:
  results/a1_probing.npz with serialized accuracies per (model, dataset,
  label_type, layer).
"""

from __future__ import annotations

import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = ["distilbert", "bert", "gpt2"]

# Mapping from label type to dataset.
LABEL_CONFIG = {
    "pos":             {"dataset": "ud_ewt", "n_classes": 17, "label_key": "labels_pos"},
    "freq_bucket":     {"dataset": "ud_ewt", "n_classes": 3,  "label_key": "labels_freq_bucket"},
    "position_bucket": {"dataset": "ud_ewt", "n_classes": 3,  "label_key": "labels_position_bucket"},
    "sentiment":       {"dataset": "sst2",   "n_classes": 2,  "label_key": "labels_sentiment"},
}

# Prober hyperparameters.
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5
VAL_FRACTION = 0.1  # of train, for early stopping


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

_cache = {}

def load_probing_data(model_name: str, dataset_name: str):
    """Load activations and metadata. Cached in memory to reuse between label
    types from the same dataset."""
    key = (model_name, dataset_name)
    if key in _cache:
        return _cache[key]

    path = RESULTS_DIR / f"probing_data_{model_name}_{dataset_name}.npz"
    data = np.load(path, allow_pickle=True)

    out = {
        "activations": data["activations"],   # (n_layers, n_tokens, d_model)
        "split": data["split"],               # (n_tokens,) array of 'train'/'test' strings
    }
    # Labels (whatever is in the file).
    for k in data.files:
        if k.startswith("labels_"):
            out[k] = data[k]

    _cache[key] = out
    return out


# -----------------------------------------------------------------------------
# Linear prober
# -----------------------------------------------------------------------------

class LinearProber(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes, bias=True)

    def forward(self, x):
        return self.fc(x)


def train_prober(X_train, y_train, X_val, y_val, X_test, y_test,
                 n_classes: int, device: str = "cuda", verbose: bool = False):
    """Train a linear prober and return test accuracy.

    Args:
        X_*, y_*: tensors already on GPU or movable to GPU.
        n_classes: number of classes.

    Returns:
        dict with keys: test_acc, val_acc_best, n_epochs, train_loss_final, prober_state_dict
    """
    d_model = X_train.shape[1]
    model = LinearProber(d_model, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # In-GPU iteration; skip the cost of a DataLoader-based shuffle on CPU.
    n_train = X_train.shape[0]

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Shuffle indices.
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            idx = perm[start:end]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Validation.
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(f"    epoch {epoch}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, best={best_val_acc:.4f}")

        if patience_counter >= PATIENCE:
            break

    # Restore best.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test.
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds == y_test).float().mean().item()

    return {
        "test_acc": test_acc,
        "val_acc_best": best_val_acc,
        "n_epochs": epoch + 1,
        "train_loss_final": avg_loss,
        "prober_weights": model.fc.weight.detach().cpu().numpy(),   # (n_classes, d_model)
        "prober_bias": model.fc.bias.detach().cpu().numpy(),        # (n_classes,)
    }


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------

def majority_baseline(y_train, y_test):
    """Accuracy of always predicting the majority class from train."""
    counter = Counter(y_train.tolist())
    majority_class = counter.most_common(1)[0][0]
    return (y_test == majority_class).float().mean().item()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_probing(model_name: str, label_type: str, device: str = "cuda"):
    """Run probing for all layers of a (model, label_type)."""
    cfg = LABEL_CONFIG[label_type]
    dataset_name = cfg["dataset"]
    n_classes = cfg["n_classes"]
    label_key = cfg["label_key"]

    data = load_probing_data(model_name, dataset_name)
    activations = data["activations"]    # (n_layers, n_tokens, d_model)
    split = data["split"]                # (n_tokens,)
    labels = data[label_key]             # (n_tokens,)

    n_layers = activations.shape[0]

    # Train/test indices.
    train_mask = (split == "train")
    test_mask = (split == "test")

    # From train, hold out 10% for validation (always the last 10%).
    train_indices = np.where(train_mask)[0]
    n_val = int(len(train_indices) * VAL_FRACTION)
    val_indices = train_indices[-n_val:]
    train_indices = train_indices[:-n_val]
    test_indices = np.where(test_mask)[0]

    y_train_all = torch.tensor(labels[train_indices], dtype=torch.long)
    y_val_all = torch.tensor(labels[val_indices], dtype=torch.long)
    y_test_all = torch.tensor(labels[test_indices], dtype=torch.long)

    # Baselines (layer-independent).
    chance_acc = 1.0 / n_classes
    majority_acc = majority_baseline(y_train_all, y_test_all)

    layer_results = []
    print(f"  layers: ", end="", flush=True)
    for k in range(n_layers):
        X_train = torch.tensor(activations[k, train_indices], dtype=torch.float32)
        X_val = torch.tensor(activations[k, val_indices], dtype=torch.float32)
        X_test = torch.tensor(activations[k, test_indices], dtype=torch.float32)

        result = train_prober(X_train, y_train_all, X_val, y_val_all,
                              X_test, y_test_all, n_classes, device=device)
        layer_results.append(result)
        print(f"{k}:{result['test_acc']:.3f} ", end="", flush=True)
    print()

    return {
        "model": model_name,
        "label_type": label_type,
        "dataset": dataset_name,
        "n_classes": n_classes,
        "n_layers": n_layers,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "chance_acc": chance_acc,
        "majority_acc": majority_acc,
        "layer_test_acc": [r["test_acc"] for r in layer_results],
        "layer_val_acc": [r["val_acc_best"] for r in layer_results],
        "layer_n_epochs": [r["n_epochs"] for r in layer_results],
        "layer_prober_weights": [r["prober_weights"] for r in layer_results],
        "layer_prober_bias": [r["prober_bias"] for r in layer_results],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU. Cancel and verify CUDA.")
        import sys
        sys.exit(1)

    print(f"Device: {device}")
    torch.manual_seed(42)

    t_total = time.time()

    all_results = {}

    for model_name in MODELS:
        for label_type in LABEL_CONFIG.keys():
            print(f"\n--- {model_name} / {label_type} ---")
            t0 = time.time()
            result = run_probing(model_name, label_type, device=device)
            elapsed = time.time() - t0
            print(f"  chance={result['chance_acc']:.3f}, majority={result['majority_acc']:.3f}")
            print(f"  best layer: {np.argmax(result['layer_test_acc'])} "
                  f"(acc={max(result['layer_test_acc']):.3f})")
            print(f"  time: {elapsed:.1f}s")
            all_results[f"{model_name}__{label_type}"] = result

    # Save.
    out_path = RESULTS_DIR / "a1_probing.npz"

    # To store dicts in npz, serialize each entry as a separate key.
    save_dict = {}
    for key, res in all_results.items():
        prefix = key
        save_dict[f"{prefix}__layer_test_acc"] = np.array(res["layer_test_acc"])
        save_dict[f"{prefix}__layer_val_acc"] = np.array(res["layer_val_acc"])
        save_dict[f"{prefix}__chance_acc"] = res["chance_acc"]
        save_dict[f"{prefix}__majority_acc"] = res["majority_acc"]
        save_dict[f"{prefix}__n_classes"] = res["n_classes"]
        save_dict[f"{prefix}__n_train"] = res["n_train"]
        save_dict[f"{prefix}__n_test"] = res["n_test"]
        # Weights: stack into (n_layers, n_classes, d_model).
        save_dict[f"{prefix}__prober_weights"] = np.stack(res["layer_prober_weights"], axis=0)
        save_dict[f"{prefix}__prober_bias"] = np.stack(res["layer_prober_bias"], axis=0)
        save_dict[f"{prefix}__layer_n_epochs"] = np.array(res["layer_n_epochs"])

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"=== A1 complete in {(time.time() - t_total)/60:.1f} min ===")


if __name__ == "__main__":
    main()