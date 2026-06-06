"""
extract_probing_data.py -- Extract activations on UD-EWT and SST-2 for probing.

For each (model, dataset):
  1. Load the dataset.
  2. Take 2500 sentences (2000 train + 500 test).
  3. Tokenize with the model's tokenizer.
  4. Forward pass and extract hidden_states from all layers.
  5. First-subtoken alignment: for each original word, keep the embedding of
     its first sub-token and associate it with the word-level labels.

Output:
  results/probing_data_{model}_{dataset}.npz  with keys:
    "activations": (n_layers, n_tokens_total, d_model) float32
      - n_tokens_total is the sum of words (not sub-tokens) across all sentences
        after filtering and truncation.
    "sentence_idx": (n_tokens_total,) int -- sentence each token belongs to
    "word_idx":     (n_tokens_total,) int -- position of the word within its sentence
    "split":        (n_tokens_total,) str -- "train" or "test"
    "labels_{kind}": (n_tokens_total,) -- one key per label type
      For UD-EWT: labels_pos, labels_freq_bucket, labels_position_bucket
      For SST-2:  labels_sentiment, labels_freq_bucket, labels_position_bucket

Notes:
  - We truncate to 64 sub-tokens per sentence (sufficient for typical UD-EWT
    and SST-2 sentences).
  - Sentences longer than the limit after tokenization are dropped.
  - Word frequency: bucketized into high/mid/low based on the dataset's own
    word counts (not an external corpus).
  - Position bucketized into thirds (early/mid/late) within each sentence.
"""

from __future__ import annotations

import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from common import (
    MODEL_SPECS,
    load_model_and_tokenizer,
    forward_full,
    # Probing-data utilities (loaders, parser, alignment) live in common
    # so that 06_fix_freq_bucket can reproduce identical filtering.
    MAX_SEQ_LEN, N_TRAIN, N_TEST,
    UPOS_TO_INT,
    parse_conllu,
    load_ud_ewt,
    load_sst2,
    tokenize_with_alignment,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32


# -----------------------------------------------------------------------------
# Frequency and position bucketing
# -----------------------------------------------------------------------------

def build_freq_buckets(all_words: list[str]) -> dict[str, int]:
    """Assign a frequency bucket (0=low, 1=mid, 2=high) to each word based on
    its count within all_words."""
    counter = Counter(all_words)
    # Sort by descending frequency.
    sorted_words = sorted(counter.items(), key=lambda x: -x[1])
    n = len(sorted_words)
    bucket = {}
    for rank, (word, _) in enumerate(sorted_words):
        if rank < n / 3:
            bucket[word] = 2  # high
        elif rank < 2 * n / 3:
            bucket[word] = 1  # mid
        else:
            bucket[word] = 0  # low
    return bucket


def position_bucket(pos: int, total: int) -> int:
    """Bucketize a word's position into early(0)/mid(1)/late(2)."""
    if total <= 0:
        return 1
    rel = pos / total
    if rel < 1/3:
        return 0
    elif rel < 2/3:
        return 1
    else:
        return 2


# -----------------------------------------------------------------------------
# Activation extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(model_name: str, sentences: list[dict], dataset_name: str,
                        device: str = "cuda"):
    """For each sentence: tokenize with alignment, forward, extract the embedding
    of each word's first sub-token at every layer.

    Returns:
      activations: (n_layers, n_tokens, d_model)
      metadata: dict with sentence_idx, word_idx, split, labels_*
    """
    model, tokenizer, spec = load_model_and_tokenizer(model_name, device=device)
    n_states = spec["n_states"]
    d_model = spec["d_model"]

    # Pad token.
    pad_id = tokenizer.pad_token_id

    # --- Step 1: tokenize all sentences and drop those exceeding MAX_SEQ_LEN ---
    tokenized = []  # list of dicts with input_ids, first_subtoken, sentence
    skipped = 0
    for sent_i, sent in enumerate(sentences):
        input_ids, first_sub = tokenize_with_alignment(sent["words"], tokenizer, model_name)
        if len(input_ids) > MAX_SEQ_LEN or len(first_sub) == 0:
            skipped += 1
            continue
        tokenized.append({
            "input_ids": input_ids,
            "first_subtoken": first_sub,
            "sentence_idx": sent_i,
            "sentence": sent,
        })
    print(f"  Tokenized: {len(tokenized)} sentences ({skipped} dropped by length)")

    # --- Step 2: forward pass in padded batches ---
    # Accumulate first-subtoken activations in a list.
    all_acts = [[] for _ in range(n_states)]  # n_states lists, each with (n_words_in_sent, d_model) tensors
    all_sentence_idx = []
    all_word_idx = []

    for start in range(0, len(tokenized), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(tokenized))
        batch = tokenized[start:end]

        # Manual padding.
        max_len = max(len(t["input_ids"]) for t in batch)
        batch_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        batch_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, t in enumerate(batch):
            L = len(t["input_ids"])
            batch_ids[i, :L] = torch.tensor(t["input_ids"], dtype=torch.long)
            batch_mask[i, :L] = 1

        out = forward_full(model, batch_ids, batch_mask, device=device)
        hidden_states = out["hidden_states"]  # tuple of (b, max_len, d_model)

        for i, t in enumerate(batch):
            fs = t["first_subtoken"]
            for k in range(n_states):
                # Pick embeddings at the first_subtoken positions.
                acts = hidden_states[k][i, fs, :].cpu().numpy()  # (n_words, d_model)
                all_acts[k].append(acts)

            # Record metadata once (when k=0).
            n_words = len(fs)
            all_sentence_idx.extend([t["sentence_idx"]] * n_words)
            all_word_idx.extend(list(range(n_words)))

    # --- Step 3: concatenate ---
    activations = np.stack([np.concatenate(layer_list, axis=0) for layer_list in all_acts], axis=0)
    # shape: (n_states, n_total_words, d_model)
    print(f"  Activations shape: {activations.shape}")

    sentence_idx = np.array(all_sentence_idx, dtype=np.int32)
    word_idx = np.array(all_word_idx, dtype=np.int32)

    # --- Step 4: build labels ---
    metadata = {
        "sentence_idx": sentence_idx,
        "word_idx": word_idx,
    }

    # Split (train/test).
    split_arr = np.array([sentences[si]["split"] for si in sentence_idx])
    metadata["split"] = split_arr

    # Frequency bucket (across all words in the dataset).
    all_words_lower = []
    for t in tokenized:
        for wi in range(len(t["first_subtoken"])):
            all_words_lower.append(t["sentence"]["words"][wi].lower())
    freq_bucket = build_freq_buckets(all_words_lower)

    labels_freq = []
    for t in tokenized:
        for wi in range(len(t["first_subtoken"])):
            word = t["sentence"]["words"][wi].lower()
            labels_freq.append(freq_bucket.get(word, 0))
    metadata["labels_freq_bucket"] = np.array(labels_freq, dtype=np.int32)

    # Position bucket.
    labels_pos_bucket = []
    for t in tokenized:
        n = len(t["first_subtoken"])
        for wi in range(n):
            labels_pos_bucket.append(position_bucket(wi, n))
    metadata["labels_position_bucket"] = np.array(labels_pos_bucket, dtype=np.int32)

    # Dataset-specific labels.
    if dataset_name == "ud_ewt":
        # Universal POS tags, 17 classes in UD.
        labels_pos = []
        for t in tokenized:
            pos_tags = t["sentence"]["pos"]
            n = len(t["first_subtoken"])
            for wi in range(n):
                labels_pos.append(pos_tags[wi] if wi < len(pos_tags) else 0)
        metadata["labels_pos"] = np.array(labels_pos, dtype=np.int32)

    elif dataset_name == "sst2":
        # Sentiment: apply sentence label to all its tokens.
        labels_sent = []
        for t in tokenized:
            sentiment = t["sentence"]["sentiment"]
            n = len(t["first_subtoken"])
            labels_sent.extend([sentiment] * n)
        metadata["labels_sentiment"] = np.array(labels_sent, dtype=np.int32)

    # Free GPU.
    del model
    torch.cuda.empty_cache()

    return activations, metadata


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_one(model_name: str, dataset_name: str, sentences: list[dict], device: str = "cuda"):
    out_path = RESULTS_DIR / f"probing_data_{model_name}_{dataset_name}.npz"
    if out_path.exists():
        print(f"  [skip] already exists: {out_path.name}")
        return

    print(f"\n--- {model_name} / {dataset_name} ---")
    t0 = time.time()

    activations, metadata = extract_activations(model_name, sentences, dataset_name, device=device)

    save_dict = {"activations": activations}
    for k, v in metadata.items():
        save_dict[k] = v

    np.savez_compressed(out_path, **save_dict)
    elapsed = time.time() - t0
    print(f"  saved: {out_path} ({activations.nbytes / 1e6:.1f}MB)")
    print(f"  time: {elapsed:.1f}s")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU. Cancel and verify CUDA.")
        import sys
        sys.exit(1)

    print(f"Device: {device}")

    print("\n=== Loading datasets ===")
    ud_sentences = load_ud_ewt()
    sst_sentences = load_sst2()

    t_total = time.time()

    for model_name in ["distilbert", "bert", "gpt2"]:
        run_one(model_name, "ud_ewt", ud_sentences, device=device)
        run_one(model_name, "sst2", sst_sentences, device=device)

    print(f"\n=== Probing data extraction complete in {(time.time() - t_total)/60:.1f} min ===")