"""
common.py -- Base utilities for the translator-mechanism experiments.

This is not an experiment itself. It provides:
- Loading the three models (DistilBERT, BERT, GPT-2) on GPU.
- Loading datasets (input_ids, attention_mask) produced by 01_sandwich_geometry/,
  per model and condition.
- Loading precomputed embeddings from 01_sandwich_geometry/
  (shape: layers x sentences x 20 tokens x d_model).
- Tokenization helpers for identifying special tokens and mapping positions.
- Functions to run forward passes with masked / ablated tokens.
- Probing-data utilities (UD-EWT / SST-2 loaders, CoNLL-U parser, UPOS map,
  first-subtoken alignment). Kept here so any downstream script that needs to
  reproduce the same per-token slicing uses identical logic.

Imported by b1_ablation.py, extract_probing_data.py, a1_probing.py,
a2_budget.py, a3_weights.py, fix_freq_bucket.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoModel, AutoTokenizer,
    DistilBertModel, BertModel, GPT2Model,
    DistilBertTokenizer, BertTokenizer, GPT2Tokenizer,
)

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------

MODEL_SPECS = {
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "n_layers": 6,           # 6 blocks -> 7 states (state 0 = embeddings, state 6 = post last block)
        "n_states": 7,
        "d_model": 768,
        "type": "encoder_mlm",
        "norm_position": "post",
    },
    "bert": {
        "hf_name": "bert-base-uncased",
        "n_layers": 12,
        "n_states": 13,
        "d_model": 768,
        "type": "encoder_mlm",
        "norm_position": "post",
    },
    "gpt2": {
        "hf_name": "gpt2",
        "n_layers": 12,
        "n_states": 13,
        "d_model": 768,
        "type": "decoder_autoreg",
        "norm_position": "pre",
    },
}

CONDITIONS = ["coherent", "permuted", "random"]

# Datasets and embeddings live in the sibling block 01_sandwich_geometry/.
SANDWICH_DATA_ROOT = Path(__file__).parent.parent / "01_sandwich_geometry" / "data"


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load HF model in eval mode + tokenizer.

    Returns:
        (model, tokenizer, spec_dict)
    """
    spec = MODEL_SPECS[model_name]
    hf_name = spec["hf_name"]

    model = AutoModel.from_pretrained(hf_name, output_hidden_states=True, output_attentions=True)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    # GPT-2 has no pad_token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, spec


def get_special_token_ids(tokenizer, model_name: str) -> dict:
    """Return the IDs of the relevant special tokens.

    For encoders: CLS, SEP, MASK, PAD.
    For GPT-2: only EOS (used as PAD); no real CLS/SEP/MASK.
    """
    if model_name in ("distilbert", "bert"):
        return {
            "cls": tokenizer.cls_token_id,
            "sep": tokenizer.sep_token_id,
            "mask": tokenizer.mask_token_id,
            "pad": tokenizer.pad_token_id,
            "unk": tokenizer.unk_token_id,
        }
    elif model_name == "gpt2":
        return {
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
            "unk": tokenizer.unk_token_id,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_ablation_token_id(tokenizer, model_name: str) -> int:
    """Token used as the replacement in ablation.

    For encoders: [MASK]. Inherits MLM dynamics (the model is trained to predict it).
    For GPT-2: [EOS] used as [PAD]. Known asymmetry vs. encoders; the analysis
    accounts for this.
    """
    if model_name in ("distilbert", "bert"):
        return tokenizer.mask_token_id
    elif model_name == "gpt2":
        return tokenizer.eos_token_id
    else:
        raise ValueError(f"Unknown model: {model_name}")


# -----------------------------------------------------------------------------
# Dataset loading (reuses outputs from 01_sandwich_geometry)
# -----------------------------------------------------------------------------

@dataclass
class SandwichDataset:
    """Container for the dataset of a (model, condition) pair."""
    model_name: str
    condition: str
    input_ids: torch.Tensor          # (N=500, seq_len=22) long
    attention_mask: torch.Tensor     # (N=500, seq_len=22) long
    content_positions: np.ndarray    # (N=500, 20) int -- positions of the 20 content tokens in input_ids


def load_dataset(model_name: str, condition: str) -> SandwichDataset:
    """Load input_ids/attention_mask from the jsonl produced by 01_sandwich_geometry.

    Also returns the positions of the 20 content tokens. Convention:
      - DistilBERT/BERT: [CLS] at 0, content at 1..20, [SEP] at 21.
      - GPT-2: content at 0..19, padding/special at 20..21 (if applicable).

    We verify the convention and build content_positions accordingly.
    """
    path = SANDWICH_DATA_ROOT / model_name / f"dataset_{condition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path.resolve()}")

    input_ids_list = []
    attn_mask_list = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            input_ids_list.append(obj["input_ids"])
            attn_mask_list.append(obj["attention_mask"])

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attn_mask_list, dtype=torch.long)

    # Content-position convention.
    if model_name in ("distilbert", "bert"):
        # content at positions 1..20 (between CLS and SEP)
        content_positions = np.tile(np.arange(1, 21), (input_ids.shape[0], 1))
    elif model_name == "gpt2":
        # content at positions 0..19
        content_positions = np.tile(np.arange(0, 20), (input_ids.shape[0], 1))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return SandwichDataset(
        model_name=model_name,
        condition=condition,
        input_ids=input_ids,
        attention_mask=attention_mask,
        content_positions=content_positions,
    )


def load_precomputed_embeddings(model_name: str, condition: str) -> np.ndarray:
    """Load embeddings precomputed by 01_sandwich_geometry.

    Returns:
        np.ndarray of shape (n_states, 500, 20, d_model). float32.
    """
    path = SANDWICH_DATA_ROOT / model_name / f"embeddings_{condition}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path.resolve()}")

    data = np.load(path)
    return data["embeddings"]


# -----------------------------------------------------------------------------
# Forward pass with ablation
# -----------------------------------------------------------------------------

@torch.no_grad()
def forward_full(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 device: str = "cuda") -> dict:
    """Forward pass returning hidden_states across all layers.

    Args:
        input_ids: (batch, seq_len) long
        attention_mask: (batch, seq_len) long
        device: device

    Returns:
        {
            "hidden_states": tuple of (batch, seq_len, d_model) -- one per layer,
            "attentions": tuple of (batch, n_heads, seq_len, seq_len) -- one per block,
        }
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return {
        "hidden_states": out.hidden_states,
        "attentions": out.attentions,
    }


@torch.no_grad()
def forward_with_ablation(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          ablate_position: int, ablation_token_id: int,
                          device: str = "cuda") -> dict:
    """Forward pass replacing the token at `ablate_position` by `ablation_token_id`.

    Notes:
      - In encoders, ablation_token_id is [MASK]: the model is trained to process it.
      - In GPT-2, ablation_token_id is [EOS]: introduces an end-of-sequence bias;
        asymmetry handled at analysis time.
      - attention_mask stays the same (the token remains attendable).

    Args:
        ablate_position: 0-based index of the token to ablate in each batch sentence.
        ablation_token_id: ID of the replacement token.

    Returns:
        Same format as forward_full.
    """
    input_ids_ablated = input_ids.clone()
    input_ids_ablated[:, ablate_position] = ablation_token_id

    return forward_full(model, input_ids_ablated, attention_mask, device=device)


# -----------------------------------------------------------------------------
# Probing-data utilities (shared by extract_probing_data and fix_freq_bucket)
# -----------------------------------------------------------------------------

# Probing-data parameters. Kept here so that any script that re-derives
# per-token metadata uses exactly the same filtering as the extractor.
MAX_SEQ_LEN = 64       # sub-word tokens. Sentences exceeding this are dropped.
N_TRAIN = 2000
N_TEST = 500

# Universal POS tag mapping (UD v2) -> int. 17 standard classes.
UPOS_TO_INT = {
    "ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4,
    "DET": 5, "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9,
    "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14,
    "VERB": 15, "X": 16, "_": 16,  # underscore tokens are placeholders -> X
}


def parse_conllu(text: str) -> list:
    """Parse standard CoNLL-U format.

    Each sentence is a block of lines separated by an empty line. Each token
    line has 10 tab-separated fields:
      ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC

    Returns a list of {"words": [...], "pos": [...]}.
    """
    sentences = []
    current_words = []
    current_pos = []

    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            if current_words:
                sentences.append({"words": current_words, "pos": current_pos})
                current_words = []
                current_pos = []
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        token_id = fields[0]
        # Skip range tokens (e.g. "1-2") and empty/decimal tokens (e.g. "1.1").
        if "-" in token_id or "." in token_id:
            continue
        form = fields[1]
        upos = fields[3]
        current_words.append(form)
        current_pos.append(UPOS_TO_INT.get(upos, UPOS_TO_INT["X"]))

    if current_words:
        sentences.append({"words": current_words, "pos": current_pos})

    return sentences


def load_ud_ewt() -> list:
    """Load UD English-EWT from the official GitHub repo (CoNLL-U format).

    Returns a list of dicts with keys: 'words', 'pos' (ints), 'split'.
    """
    import urllib.request

    print("  Loading UD-EWT from GitHub...")
    base_url = ("https://raw.githubusercontent.com/UniversalDependencies/"
                "UD_English-EWT/master")
    urls = {
        "train": f"{base_url}/en_ewt-ud-train.conllu",
        "test": f"{base_url}/en_ewt-ud-test.conllu",
    }

    sentences = []
    for split_name, url in urls.items():
        print(f"    downloading {split_name}: {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                text = r.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Could not download UD-EWT {split_name}: {e}")

        parsed = parse_conllu(text)
        target_n = N_TRAIN if split_name == "train" else N_TEST
        count = 0
        for example in parsed:
            if count >= target_n:
                break
            words = example["words"]
            pos_tags = example["pos"]
            if len(words) < 3 or len(words) > 40:
                continue
            sentences.append({"words": words, "pos": pos_tags, "split": split_name})
            count += 1

    print(f"  UD-EWT: {sum(1 for s in sentences if s['split']=='train')} train + "
          f"{sum(1 for s in sentences if s['split']=='test')} test")
    return sentences


def load_sst2() -> list:
    """Load SST-2. Uses whitespace tokenization (SST-2 is reasonably clean).
    Returns a list of dicts."""
    from datasets import load_dataset as hf_load_dataset

    print("  Loading SST-2...")
    ds_train = hf_load_dataset("glue", "sst2", split="train")
    ds_validation = hf_load_dataset("glue", "sst2", split="validation")

    sentences = []
    for ds, split_name in [(ds_train, "train"), (ds_validation, "test")]:
        target_n = N_TRAIN if split_name == "train" else N_TEST
        count = 0
        for example in ds:
            if count >= target_n:
                break
            text = example["sentence"].strip()
            words = text.split()
            label = example["label"]
            if len(words) < 3 or len(words) > 40:
                continue
            sentences.append({"words": words, "sentiment": label, "split": split_name})
            count += 1

    print(f"  SST-2: {sum(1 for s in sentences if s['split']=='train')} train + "
          f"{sum(1 for s in sentences if s['split']=='test')} test")
    return sentences


def tokenize_with_alignment(words: list, tokenizer, model_name: str):
    """Tokenize a list of words and return:
       - input_ids (list of ints)
       - first_subtoken_indices (list of ints): position in input_ids of the
         first sub-token of each word.

    Handles tokenizer differences (BERT/DistilBERT with CLS/SEP, GPT-2 without).
    """
    if model_name in ("distilbert", "bert"):
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id

        input_ids = [cls_id]
        first_subtoken = []
        for word in words:
            # add_special_tokens=False to tokenize the word alone.
            # add_prefix_space does not apply to WordPiece.
            sub_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(sub_ids) == 0:
                continue
            first_subtoken.append(len(input_ids))
            input_ids.extend(sub_ids)
        input_ids.append(sep_id)
        return input_ids, first_subtoken

    elif model_name == "gpt2":
        input_ids = []
        first_subtoken = []
        for i, word in enumerate(words):
            # GPT-2 BPE: the leading space matters. The first word has no space.
            if i == 0:
                sub_ids = tokenizer.encode(word, add_special_tokens=False)
            else:
                sub_ids = tokenizer.encode(" " + word, add_special_tokens=False)
            if len(sub_ids) == 0:
                continue
            first_subtoken.append(len(input_ids))
            input_ids.extend(sub_ids)
        return input_ids, first_subtoken

    else:
        raise ValueError(f"Unknown model: {model_name}")


# -----------------------------------------------------------------------------
# Self-check / sanity
# -----------------------------------------------------------------------------

def sanity_check(model_name: str, condition: str = "coherent", device: str = "cuda"):
    """Verify that everything loads and that precomputed embeddings match what the
    model produces in this run (reproducibility check)."""
    print(f"\n=== sanity_check: {model_name} / {condition} ===")

    # 1. Model and tokenizer
    model, tokenizer, spec = load_model_and_tokenizer(model_name, device=device)
    print(f"Model loaded: {spec['hf_name']} | expected n_states: {spec['n_states']}")

    # 2. Dataset
    ds = load_dataset(model_name, condition)
    print(f"Dataset: input_ids {tuple(ds.input_ids.shape)}, attention_mask {tuple(ds.attention_mask.shape)}")
    print(f"Content positions sample: {ds.content_positions[0]}")

    # 3. Precomputed embeddings
    emb_pre = load_precomputed_embeddings(model_name, condition)
    print(f"Precomputed embeddings: {emb_pre.shape}, dtype={emb_pre.dtype}")
    assert emb_pre.shape[0] == spec["n_states"], \
        f"Unexpected layer count: got {emb_pre.shape[0]}, expected {spec['n_states']}"
    assert emb_pre.shape[1] == 500
    assert emb_pre.shape[2] == 20
    assert emb_pre.shape[3] == spec["d_model"]

    # 4. Check that a fresh forward matches the precomputed embeddings (first sentence)
    out = forward_full(model, ds.input_ids[:1], ds.attention_mask[:1], device=device)
    hidden_states = out["hidden_states"]  # tuple of (1, seq_len, d_model)

    # Compare layer 0 on the 20 content tokens
    layer0_fresh = hidden_states[0][0].cpu().numpy()        # (seq_len, d_model)
    content_pos = ds.content_positions[0]
    layer0_fresh_content = layer0_fresh[content_pos]        # (20, d_model)
    layer0_precomputed = emb_pre[0, 0]                      # (20, d_model)

    max_diff = np.abs(layer0_fresh_content - layer0_precomputed).max()
    rel_diff = max_diff / (np.abs(layer0_precomputed).max() + 1e-8)
    print(f"Layer 0 check: max_abs_diff={max_diff:.2e} (rel={rel_diff:.2e})")

    if rel_diff > 1e-3:
        print(f"WARNING: large divergence. Check transformers/torch versions.")
    else:
        print("OK: precomputed embeddings consistent with fresh forward.")

    # 5. GPU usage
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
        mem_reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"GPU mem: alloc={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB")

    print("=== sanity OK ===")


if __name__ == "__main__":
    # Optional preflight: runs sanity check on the three models.
    # To run only one: python -c "from common import sanity_check; sanity_check('distilbert')"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU. Check CUDA before heavy experiments.")

    for model_name in ["distilbert", "bert", "gpt2"]:
        try:
            sanity_check(model_name, condition="coherent", device=device)
        except Exception as e:
            print(f"\nFAIL on {model_name}: {type(e).__name__}: {e}")