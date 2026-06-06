# Block 1 — Sandwich geometry
 
This block characterizes the **sandwich pattern** across DistilBERT, BERT,
and GPT-2: outer layers behave qualitatively differently from the central
plateau. It builds a controlled dataset, extracts per-layer embeddings, and
computes two complementary metrics (M1b: adjacent-layer persistence of
token representations; M3: reconstructibility of a token's representation
from its sentence context).
 
Hypotheses and outcomes for this block are in `prereg.md`.
 
## Pipeline
 
Run scripts in numeric order from this directory. Each script is
self-contained and skips work that has already been done (idempotent).
 
```bash
cd 01_sandwich_geometry
python build_datasets.py        # ~30 s
python extract_embeddings.py    # ~3 min on RTX 5060 Ti
python metrics_global.py        # ~1 min
python metrics_m1b.py           # ~2 min
python metrics_m3.py            # ~30 s
python visualize.py             # ~10 s
```
 
Outputs land under two directories: `data/{model}/` holds the datasets and
extracted embeddings, and `results/` holds the computed metrics and
figures. The `02_translator_mechanism/` block reads embeddings and
datasets from `data/{model}/`, so do not move or rename either directory.
 
## Scripts
 
### `scaffolding.py`
Shared utilities: model specs (hidden sizes, layer counts, normalization
position), condition names, common paths. Imported by all other scripts in
this block. Not run directly.
 
### `build_datasets.py`
Builds three datasets per model: `coherent`, `permuted` (word order
shuffled within sentences), and `random` (tokens replaced by random
vocabulary entries). Each dataset has 500 sentences, exactly 20 content
tokens per sentence.
 
Reads: nothing (downloads Wikipedia content via the `datasets` library on
first run).
Writes: `data/{model}/dataset_{condition}.jsonl` with one JSON object per
sentence containing `input_ids` and `attention_mask`.
 
### `extract_embeddings.py`
Runs each model on each dataset and stores the hidden states of all layers
at the 20 content-token positions.
 
Reads: `data/{model}/dataset_{condition}.jsonl`.
Writes: `data/{model}/embeddings_{condition}.npz` with key `embeddings` of
shape `(n_states, 500, 20, d_model)`. `n_states` is 7 for DistilBERT, 13
for BERT and GPT-2 (state 0 is the input embedding, state `n_layers` is
the output of the last block).
 
### `metrics_global.py`
Computes two global geometric metrics per layer: **Participation Ratio
(PR)** and **Intrinsic Dimensionality (ID)** via the TwoNN estimator. Both
are reported in two variants — raw and unit-normalized — because pre-LN
models (GPT-2) accumulate representation norms across layers in a way that
distorts PCA-based metrics unless unit-normalization is applied first. PR
is a global, variance-based measure of effective dimensionality; TwoNN ID
is a local, nearest-neighbor-based measure. The two are complementary.
 
Reads: `data/{model}/embeddings_{condition}.npz`.
Writes: `results/global_{model}_{condition}.npz` with keys `pr_raw`,
`pr_unit`, `id_raw`, `id_unit`, each of shape `(n_layers,)`.
 
### `metrics_m1b.py`
Computes **M1b** (adjacent-layer persistence): for each (sentence,
content-token position) pair and each adjacent layer pair (L_k, L_{k+1}),
the cosine similarity between the token's representation at L_k and at
L_{k+1}. Averaged over 10,000 (sentence, position) pairs per (model,
condition). Measures how much a token's representation is rewritten in one
layer step. A drop in M1b at a layer boundary indicates that layer
performs a large mechanical rewrite of token representations.
 
Reads: `data/{model}/embeddings_{condition}.npz`.
Writes: `results/m1b_{model}_{condition}.npz` with keys `m1b_raw_mean`,
`m1b_raw_std`, `m1b_unit_mean`, `m1b_unit_std`, each of shape
`(n_transitions,)`.
 
### `metrics_m3.py`
Computes **M3** (reconstructibility from context): for each layer and each
token position i, trains an MLP (one hidden layer of 512 units, GELU
activation, dropout 0.1) that predicts the token's representation from the
concatenation of the other 19 content tokens in the same sentence.
Train/test split is 80/20 at the sentence level. Reports R² of the MLP, R²
of a trivial baseline (predicting from the mean of the 19 neighbors), and
the **gap** = R²_MLP − R²_trivial. The gap is the primary metric: it
isolates how much information about the token is recoverable from context,
above what the geometric compactness of the neighborhood already provides.
A negative gap means the trivial baseline outperforms the MLP — the layer
has homogenized token representations enough that the mean of the context
predicts better than any learned non-linear combination.
 
Reads: `data/{model}/embeddings_{condition}.npz`.
Writes: `results/m3_{model}_{condition}.npz` with keys `r2_mlp_raw`,
`r2_mlp_unit`, `r2_triv_raw`, `r2_triv_unit`, `gap_raw`, `gap_unit`, each
of shape `(n_layers,)`.
 
### `visualize.py`
Generates the per-model and per-condition figures used in the paper: M1b
curves, M3 curves, and side-by-side panels showing the sandwich shape.
Figures use ASCII labels (`->`, `^2`, `-`) for portability across
matplotlib backends.
 
Reads: `results/m1b_{model}_{condition}.npz`,
`results/m3_{model}_{condition}.npz`,
`results/global_{model}_{condition}.npz`.
Writes: `results/figures/*.png`.
 
## Notes
 
- The dataset construction uses content tokens (no [CLS]/[SEP] for encoders;
  for GPT-2, no padding tokens are included). The exact positions of the 20
  content tokens within `input_ids` differ between encoders (positions
  1..20, with CLS at 0 and SEP at 21) and GPT-2 (positions 0..19). The
  `02_translator_mechanism/common.py` `load_dataset()` function preserves
  this convention when it re-reads the data.
- Random seeds are set inside `build_datasets.py`; the pipeline is
  deterministic on a given device.