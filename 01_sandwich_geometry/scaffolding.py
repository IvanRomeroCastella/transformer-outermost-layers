"""
Scaffolding: verify model loading and hidden_states structure for the three
pretrained transformers studied in this work.

Goal: before pre-registering any prediction over "L0->L1" or "Lextrema-1 -> Lextrema",
confirm exactly how many layers HuggingFace exposes for each model and how the
indexing is numbered. No metrics, no full dataset, no analysis.

Expected output: per model, prints number of hidden_states layers, embedding
dimension, and output tensor shape for a test input.
"""

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    DistilBertModel,
    BertModel,
    GPT2Model,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SENTENCE = "The quick brown fox jumps over the lazy dog."

MODELS = [
    {
        "name": "distilbert-base-uncased",
        "loader": DistilBertModel,
        "expected_layers": 6,  # nominal architecture
    },
    {
        "name": "bert-base-uncased",
        "loader": BertModel,
        "expected_layers": 12,
    },
    {
        "name": "gpt2",
        "loader": GPT2Model,
        "expected_layers": 12,
    },
]


def inspect_model(name, loader_cls):
    print(f"\n{'=' * 60}")
    print(f"Modelo: {name}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = loader_cls.from_pretrained(name, output_hidden_states=True).to(DEVICE).eval()

    # gpt2 has no pad_token by default; set it just in case we later batch.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(TEST_SENTENCE, return_tensors="pt").to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # tuple of tensors
    n_layers = len(hidden_states)
    d_model = hidden_states[0].shape[-1]

    print(f"Input: {TEST_SENTENCE!r}")
    print(f"N tokens (including specials): {n_tokens}")
    print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print(f"")
    print(f"len(hidden_states) = {n_layers}")
    print(f"  -> hidden_states[0] = initial embedding (pre-layer-0)")
    print(f"  -> hidden_states[{n_layers - 1}] = last layer output")
    print(f"  -> N actual transformer layers = {n_layers - 1}")
    print(f"")
    print(f"Embedding dimension: {d_model}")
    print(f"Shape of each hidden_state: {hidden_states[0].shape}  (batch, seq, d_model)")
    print(f"")
    print(f"L2 norm of first token embedding, per layer:")
    for i, hs in enumerate(hidden_states):
        norm = hs[0, 0].norm().item()
        print(f"  layer {i}: norm = {norm:.4f}")

    del model
    torch.cuda.empty_cache()


def main():
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    for spec in MODELS:
        inspect_model(spec["name"], spec["loader"])

    print(f"\n{'=' * 60}")
    print("Scaffolding complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()