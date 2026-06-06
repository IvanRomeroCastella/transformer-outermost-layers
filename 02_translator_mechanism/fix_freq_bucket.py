"""
fix_freq_bucket.py -- Rewrite labels_freq_bucket in probing-data files.

Original issue: build_freq_buckets in extract_probing_data.py bucketizes
by rank of unique word, which causes the "high" bucket (common words like
"the", "of") to dominate at the token level. Result: majority baseline
~0.86 and the prober has no signal to learn from.

Fix: bucketize by token-count percentiles, not word-rank percentiles.
The thirds are computed over the token distribution, not over the vocabulary.

Output: overwrites labels_freq_bucket in each probing_data_*.npz file while
keeping everything else intact.

Run after extract_probing_data.py.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import numpy as np

from common import (
    load_model_and_tokenizer,
    load_ud_ewt,
    load_sst2,
    tokenize_with_alignment,
    MAX_SEQ_LEN,
)

RESULTS_DIR = Path("results")
MODELS = ["distilbert", "bert", "gpt2"]
DATASETS = ["ud_ewt", "sst2"]


def rebucket_freq(words_per_token: list) -> np.ndarray:
    """Bucketize by token-count percentiles.

    Instead of sorting unique words and splitting them into thirds (which
    produces buckets that are unbalanced at the token level), we sort by
    frequency and split so that each bucket contains ~the same number of
    TOKENS.

    Args:
        words_per_token: list (length = n_tokens) with the word of each token.

    Returns:
        np.ndarray (n_tokens,) with bucket 0/1/2 per token.
    """
    counter = Counter(words_per_token)
    # (word, count) sorted by descending count.
    sorted_words = sorted(counter.items(), key=lambda x: -x[1])

    n_tokens = sum(c for _, c in sorted_words)
    third = n_tokens / 3

    # Assign a bucket to each word by accumulating tokens.
    word_to_bucket = {}
    cumulative = 0
    for word, count in sorted_words:
        if cumulative < third:
            bucket = 2  # high (top by token count)
        elif cumulative < 2 * third:
            bucket = 1  # mid
        else:
            bucket = 0  # low
        word_to_bucket[word] = bucket
        cumulative += count

    return np.array([word_to_bucket[w] for w in words_per_token], dtype=np.int32)


def reconstruct_words_per_token(model_name: str, dataset_name: str) -> list:
    """Reconstruct the word corresponding to each token.

    Reapplies the same dataset loading and length-filtering as
    extract_probing_data.py, so the order of words returned here matches
    the order of activations stored in the .npz files exactly.
    """
    if dataset_name == "ud_ewt":
        sentences = load_ud_ewt()
    elif dataset_name == "sst2":
        sentences = load_sst2()
    else:
        raise ValueError(dataset_name)

    _, tokenizer, _ = load_model_and_tokenizer(model_name, device="cpu")

    words_per_token = []
    for sent in sentences:
        input_ids, first_sub = tokenize_with_alignment(sent["words"], tokenizer, model_name)
        if len(input_ids) > MAX_SEQ_LEN or len(first_sub) == 0:
            continue
        # n words retained = len(first_sub)
        for wi in range(len(first_sub)):
            words_per_token.append(sent["words"][wi].lower())

    return words_per_token


def fix_file(model_name: str, dataset_name: str):
    path = RESULTS_DIR / f"probing_data_{model_name}_{dataset_name}.npz"
    if not path.exists():
        print(f"  [skip] does not exist: {path}")
        return

    print(f"\n--- {model_name} / {dataset_name} ---")
    data = np.load(path, allow_pickle=True)

    words = reconstruct_words_per_token(model_name, dataset_name)
    n_expected = len(data["sentence_idx"])
    if len(words) != n_expected:
        print(f"  WARNING: mismatch! reconstructed {len(words)} vs expected {n_expected}")
        print(f"  Aborting this file (inspect manually).")
        return

    new_freq = rebucket_freq(words)

    counter = Counter(new_freq.tolist())
    n = len(new_freq)
    print(f"  new distribution: low={counter[0]/n:.3f}, "
          f"mid={counter[1]/n:.3f}, high={counter[2]/n:.3f}")

    # Load all, replace labels_freq_bucket, save.
    save_dict = {k: data[k] for k in data.files}
    save_dict["labels_freq_bucket"] = new_freq

    np.savez_compressed(path, **save_dict)
    print(f"  updated: {path}")


if __name__ == "__main__":
    for model_name in MODELS:
        for dataset_name in DATASETS:
            fix_file(model_name, dataset_name)

    print("\n=== Fix complete ===")