"""
extract_embeddings.py

Extract hidden_states embeddings from the three models on the three datasets.
One pass per model, nine .npz output files.

Output structure:
  data/{model_short}/embeddings_{condition}.npz

Each .npz holds an 'embeddings' array with shape:
  (n_layers, n_sentences, n_content_tokens, d_model)

where:
  - n_layers = len(hidden_states) of the model (7 for DistilBERT, 13 for BERT/GPT-2)
  - n_sentences = 500
  - n_content_tokens = 20
  - d_model = 768

In encoders we discard [CLS] (pos 0) and [SEP] (pos -1), keeping the 20 content
tokens. In GPT-2 the 20 generated tokens are used directly.

Usage:
    python extract_embeddings.py
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 32
CONDITIONS = ["coherent", "permuted", "random"]


@dataclass
class ModelSpec:
    short_name: str
    hf_name: str
    has_specials: bool
    expected_layers: int  # = len(hidden_states), including initial embedding


MODELS = [
    ModelSpec("distilbert", "distilbert-base-uncased", has_specials=True,  expected_layers=7),
    ModelSpec("bert",       "bert-base-uncased",       has_specials=True,  expected_layers=13),
    ModelSpec("gpt2",       "gpt2",                    has_specials=False, expected_layers=13),
]


def load_dataset_jsonl(path: Path):
    """Read a .jsonl and return (input_ids, attention_mask) as lists."""
    input_ids_list = []
    attention_mask_list = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            input_ids_list.append(entry["input_ids"])
            attention_mask_list.append(entry["attention_mask"])
    return input_ids_list, attention_mask_list


def extract_for_model_condition(spec: ModelSpec, model, condition: str):
    """
    Process the (model, condition) dataset and return an array
    (n_layers, n_sentences, n_content_tokens, d_model) in float32.
    """
    dataset_path = DATA_DIR / spec.short_name / f"dataset_{condition}.jsonl"
    input_ids_list, attention_mask_list = load_dataset_jsonl(dataset_path)
    n_sentences = len(input_ids_list)
    assert n_sentences == 500, f"Expected 500, found {n_sentences}"

    total_len = len(input_ids_list[0])  # 22 for encoders, 20 for GPT-2
    n_content = 20
    expected_layers = spec.expected_layers

    # Content token indices to extract:
    # - encoders: [1, 2, ..., 20] (drop CLS at 0 and SEP at 21)
    # - GPT-2:    [0, 1, ..., 19] (whole sequence)
    if spec.has_specials:
        content_indices = list(range(1, 1 + n_content))
    else:
        content_indices = list(range(n_content))

    # Pre-allocate output array.
    all_embeddings = np.zeros(
        (expected_layers, n_sentences, n_content, 768),
        dtype=np.float32,
    )

    # Process in batches.
    n_batches = (n_sentences + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_sentences)

        batch_input_ids = torch.tensor(
            input_ids_list[start:end], dtype=torch.long
        ).to(DEVICE)
        batch_attention_mask = torch.tensor(
            attention_mask_list[start:end], dtype=torch.long
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        assert len(hidden_states) == expected_layers, (
            f"[{spec.short_name}] expected {expected_layers} layers, "
            f"got {len(hidden_states)}"
        )

        # hidden_states[l]: shape (batch, total_len, 768)
        # Select only content_indices along the token axis.
        for layer_idx, hs in enumerate(hidden_states):
            # hs[:, content_indices, :]: (batch, n_content, 768)
            content_hs = hs[:, content_indices, :].cpu().numpy().astype(np.float32)
            all_embeddings[layer_idx, start:end, :, :] = content_hs

    return all_embeddings


def process_model(spec: ModelSpec):
    print(f"\n{'=' * 70}")
    print(f"MODEL: {spec.short_name} ({spec.hf_name})")
    print(f"{'=' * 70}")

    print(f"  Loading model on {DEVICE}...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        spec.hf_name, output_hidden_states=True
    ).to(DEVICE).eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    for condition in CONDITIONS:
        print(f"\n  --- {spec.short_name} / {condition} ---")
        t0 = time.time()
        embeddings = extract_for_model_condition(spec, model, condition)
        elapsed = time.time() - t0
        out_path = DATA_DIR / spec.short_name / f"embeddings_{condition}.npz"
        np.savez_compressed(out_path, embeddings=embeddings)
        size_mb = out_path.stat().st_size / 1e6
        print(f"    Shape: {embeddings.shape}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Saved: {out_path} ({size_mb:.1f} MB)")

    del model
    torch.cuda.empty_cache()


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_free, vram_total = torch.cuda.mem_get_info()
        print(f"VRAM free / total: {vram_free / 1e9:.2f} / {vram_total / 1e9:.2f} GB")

    t_start = time.time()
    for spec in MODELS:
        process_model(spec)
    print(f"\n{'=' * 70}")
    print(f"Extraction complete in {(time.time() - t_start) / 60:.1f} min.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()