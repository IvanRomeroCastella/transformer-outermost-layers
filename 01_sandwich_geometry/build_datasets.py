"""
build_datasets.py

Builds three 500-sentence datasets for EACH of the three models (DistilBERT,
BERT base, GPT-2), under three conditions.

Per-model structure:
  - coherent:  natural Wikipedia sentences that tokenize to the target length
               in the model's tokenizer.
  - permuted:  same sentences as coherent, with the 20 content tokens permuted
               (Fisher-Yates). For encoders, [CLS] and [SEP] stay in place.
               For GPT-2, there are no specials to respect.
  - random:    20 token_ids sampled uniformly from the model's vocabulary
               (excluding specials where they exist), wrapped in [CLS] and
               [SEP] only if the model has those tokens.

Target lengths (TOTAL tokens generated):
  - DistilBERT, BERT: 22 = [CLS] + 20 content + [SEP]
  - GPT-2:            20 = 20 content (no specials)

In all cases, downstream analysis operates on 20 content tokens per sentence.
Encoder specials are discarded in the analysis phase, not here.

Output:
  data/{distilbert,bert,gpt2}/dataset_{coherent,permuted,random}.jsonl

Each entry: {"input_ids": [int, ...], "attention_mask": [int, ...]}

Usage:
    python build_datasets.py
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

# --- Global configuration ---
N_PER_CONDITION = 500
TARGET_CONTENT_TOKENS = 20  # tokens analyzed per sentence across all models
SEED = 42

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SENTENCES_TO_SCAN = 5_000_000

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class ModelSpec:
    """Per-model configuration for dataset generation."""
    short_name: str       # 'distilbert' | 'bert' | 'gpt2'
    hf_name: str          # HuggingFace name
    has_specials: bool    # True if it wraps with [CLS]/[SEP]


MODELS = [
    ModelSpec("distilbert", "distilbert-base-uncased", has_specials=True),
    ModelSpec("bert",       "bert-base-uncased",      has_specials=True),
    ModelSpec("gpt2",       "gpt2",                    has_specials=False),
]


def split_into_sentences(text: str):
    """Return a list of sentences from raw text."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if 50 <= len(s) <= 300]


def target_total_tokens(spec: ModelSpec) -> int:
    """How many total tokens a generated sentence must have for this model."""
    return TARGET_CONTENT_TOKENS + (2 if spec.has_specials else 0)


def build_coherent(spec: ModelSpec, tokenizer):
    """
    Stream Wikipedia until finding N_PER_CONDITION sentences whose tokenization
    yields exactly the model's target length.

    For encoders: target_total_tokens = 22 (CLS + 20 + SEP).
    For GPT-2:    target_total_tokens = 20 (20 pure BPE tokens).
    """
    target_len = target_total_tokens(spec)
    print(f"\n--- [{spec.short_name}] Building 'coherent' "
          f"(target_len={target_len}) ---")
    print(f"Loading Wikipedia (streaming)...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    buffer = []
    scanned = 0

    # add_special_tokens in encoders inserts CLS/SEP; in GPT-2 there are no
    # wrapping specials to add, but we pass True for consistency. GPT-2
    # tokenizer interprets it as a no-op (no wrapping specials to insert).
    add_specials = spec.has_specials

    for article in ds:
        if len(buffer) >= N_PER_CONDITION:
            break
        if scanned >= MAX_SENTENCES_TO_SCAN:
            print(f"  [!] Reached MAX_SENTENCES_TO_SCAN. Stopping with "
                  f"{len(buffer)} sentences.")
            break

        sentences = split_into_sentences(article["text"])

        for sent in sentences:
            scanned += 1
            if scanned % 100_000 == 0:
                print(f"  [scan={scanned:>10,}] coherent={len(buffer)}")

            encoded = tokenizer(
                sent,
                add_special_tokens=add_specials,
                truncation=False,
            )
            input_ids = encoded["input_ids"]

            if len(input_ids) != target_len:
                continue

            if spec.has_specials:
                # Sanity check on encoder specials.
                if input_ids[0] != tokenizer.cls_token_id:
                    continue
                if input_ids[-1] != tokenizer.sep_token_id:
                    continue

            buffer.append({
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
            })

            if len(buffer) >= N_PER_CONDITION:
                print(f"  [OK] coherent complete (scan={scanned:,})")
                break

    if len(buffer) < N_PER_CONDITION:
        print(f"  [!] WARNING: only {len(buffer)}/{N_PER_CONDITION} "
              f"sentences collected")

    return buffer[:N_PER_CONDITION]


def build_permuted(spec: ModelSpec, coherent_dataset, rng):
    """
    Take each coherent sentence and permute the 20 content tokens.
    For encoders: content lives at positions [1, 21). CLS (pos 0) and SEP
    (last) stay in place.
    For GPT-2: content is the whole sentence; permute it entirely.
    """
    print(f"\n--- [{spec.short_name}] Building 'permuted' ---")
    permuted = []
    target_len = target_total_tokens(spec)

    for entry in coherent_dataset:
        input_ids = list(entry["input_ids"])

        if spec.has_specials:
            content = input_ids[1:1 + TARGET_CONTENT_TOKENS]
            rng.shuffle(content)
            new_ids = [input_ids[0]] + content + [input_ids[-1]]
        else:
            content = list(input_ids)
            rng.shuffle(content)
            new_ids = content

        # Sanity: same length, same token composition.
        assert len(new_ids) == target_len, (
            f"[{spec.short_name}] permuted length {len(new_ids)} != {target_len}"
        )
        assert sorted(new_ids) == sorted(input_ids), (
            f"[{spec.short_name}] Permutation changed token composition"
        )

        permuted.append({
            "input_ids": new_ids,
            "attention_mask": entry["attention_mask"],
        })

    print(f"  [OK] permuted: {len(permuted)} sentences")
    return permuted


def build_random(spec: ModelSpec, tokenizer, rng):
    """
    For each of the N sentences: sample 20 token_ids uniformly from the
    model's vocabulary, excluding special tokens.
    For encoders: wrap with [CLS] and [SEP].
    For GPT-2: keep the 20 tokens as-is (no specials).
    """
    print(f"\n--- [{spec.short_name}] Building 'random' ---")
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Special token IDs excluded from sampling: {sorted(special_ids)}")

    valid_ids = [i for i in range(vocab_size) if i not in special_ids]
    print(f"  Valid IDs for sampling: {len(valid_ids)}")

    random_dataset = []
    target_len = target_total_tokens(spec)

    for _ in range(N_PER_CONDITION):
        content = rng.choices(valid_ids, k=TARGET_CONTENT_TOKENS)
        if spec.has_specials:
            new_ids = [tokenizer.cls_token_id] + content + [tokenizer.sep_token_id]
        else:
            new_ids = content

        assert len(new_ids) == target_len
        random_dataset.append({
            "input_ids": new_ids,
            "attention_mask": [1] * target_len,
        })

    print(f"  [OK] random: {len(random_dataset)} sentences")
    return random_dataset


def save_dataset(dataset, model_short_name, condition_name):
    """Dump a dataset to data/{model}/dataset_{condition}.jsonl."""
    model_dir = OUT_DIR / model_short_name
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"dataset_{condition_name}.jsonl"
    with open(out_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"  Written: {out_path} ({len(dataset)} entries)")
    return out_path


def sanity_check(spec: ModelSpec, dataset, condition_name, tokenizer):
    """Checks on the built dataset."""
    print(f"\n  Sanity check [{spec.short_name}/{condition_name}]:")
    target_len = target_total_tokens(spec)
    lengths = [len(e["input_ids"]) for e in dataset]
    assert all(l == target_len for l in lengths), (
        f"Non-uniform lengths in {spec.short_name}/{condition_name}"
    )
    print(f"    OK  all sentences have {target_len} tokens")

    if spec.has_specials:
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        for i, e in enumerate(dataset):
            if e["input_ids"][0] != cls_id:
                raise AssertionError(
                    f"[CLS] missing in {spec.short_name}/{condition_name} idx {i}"
                )
            if e["input_ids"][-1] != sep_id:
                raise AssertionError(
                    f"[SEP] missing in {spec.short_name}/{condition_name} idx {i}"
                )
        print(f"    OK  [CLS] at start, [SEP] at end across all 500 sentences")
    else:
        print(f"    OK  no specials (GPT-2): CLS/SEP check not applicable")

    print(f"    Decoded samples:")
    for i in range(2):
        decoded = tokenizer.decode(dataset[i]["input_ids"])
        print(f"      [{i}] {decoded[:120]}")


def process_model(spec: ModelSpec, base_rng_seed: int):
    """Generate the three datasets for a model. Independent RNG per model
    so that the same seed applied to the same operation flow is bit-reproducible
    on a per-model basis."""
    print(f"\n{'=' * 70}")
    print(f"MODEL: {spec.short_name} ({spec.hf_name})")
    print(f"{'=' * 70}")

    rng = random.Random(base_rng_seed)
    print(f"Loading tokenizer: {spec.hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)

    if spec.has_specials:
        print(f"  [CLS]={tokenizer.cls_token_id}, [SEP]={tokenizer.sep_token_id}")
    else:
        print(f"  (no wrapping specials)")

    coherent = build_coherent(spec, tokenizer)
    sanity_check(spec, coherent, "coherent", tokenizer)
    save_dataset(coherent, spec.short_name, "coherent")

    permuted = build_permuted(spec, coherent, rng)
    sanity_check(spec, permuted, "permuted", tokenizer)
    save_dataset(permuted, spec.short_name, "permuted")

    random_ds = build_random(spec, tokenizer, rng)
    sanity_check(spec, random_ds, "random", tokenizer)
    save_dataset(random_ds, spec.short_name, "random")


def main():
    for spec in MODELS:
        process_model(spec, base_rng_seed=SEED)

    print(f"\n{'=' * 70}")
    print("Build complete for all three models.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()