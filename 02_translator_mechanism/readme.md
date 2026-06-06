# Block 2 — Translator mechanism

This block characterizes what the **outer layers** (first and last
transformer block) of DistilBERT, BERT, and GPT-2 do mechanistically. We
refer to them as the *entry translator* and the *exit translator*, and to
the intermediate blocks as the *central plateau* or *core*.

Four families of measurements are used:

- **B1** — ablation sensitivity (`b1_ablation.py`): how much each layer's
  representation changes when one input token is replaced by a mask /
  end-of-sequence token. The paper also reports **B3** (position-resolved
  ablation profile), which is B1 aggregated by the position of the ablated
  token; it is computed in `analyze.py`, not as a separate script.
- **A1** — classical layer-wise probing (`a1_probing.py`): how well a
  linear classifier trained on the activations recovers POS, frequency
  bucket, position bucket, and SST-2 sentiment.
- **A2** — dimensional-budget probing (`a2_budget.py`): how many principal
  components are needed to reach 80% of A1's full-rank accuracy.
- **A3** — prober-weight PC1 alignment (`a3_weights.py`): how aligned the
  linear prober's weights are with the dominant variance direction of the
  layer's activations.

Hypotheses and outcomes for this block are in `prereg.md`.

**This block depends on the outputs of `01_sandwich_geometry/`.** Run that
block first. The path
`02_translator_mechanism/common.py` → `..` → `01_sandwich_geometry/data/`
is hard-coded in `common.SANDWICH_DATA_ROOT`.

## Pipeline

Run scripts from this directory. The order matters: extract data first,
fix the freq_bucket labels, then probe.

```bash
cd 02_translator_mechanism

# Ablation sensitivity (independent of the probing pipeline).
python b1_ablation.py               # ~10-20 min

# Probing pipeline.
python extract_probing_data.py      # ~5 min, downloads UD-EWT and SST-2
python fix_freq_bucket.py           # ~30 s, must run before probing
python a1_probing.py                # ~5-10 min
python a2_budget.py                 # ~5 min (depends on a1 results)
python a3_weights.py                # ~2 min (depends on a1 results)

# Aggregation.
python analyze.py                   # ~30 s, generates figures and summary.json
```

Outputs land under `results/`.

## Why `fix_freq_bucket.py` exists and runs in the middle

The original `build_freq_buckets()` in `extract_probing_data.py`
bucketizes words by rank: top third of unique words → bucket "high",
middle third → "mid", bottom third → "low". This produces an extremely
unbalanced distribution at the **token** level, because common words like
"the" and "of" each contribute many tokens. The result is a majority
baseline near 0.86 for the freq_bucket label, which leaves the prober no
discriminative signal to learn.

`fix_freq_bucket.py` rewrites `labels_freq_bucket` so the three buckets
each hold roughly one third of the tokens (not one third of the unique
words). It does this in place: it overwrites the same `.npz` files,
keeping all other fields intact.



## Scripts

### `common.py`
Shared utilities for this block. Provides:

- Model loading (`load_model_and_tokenizer`), tokenizer special-token IDs.
- Loading datasets and precomputed embeddings produced by
  `01_sandwich_geometry/` (`load_dataset`, `load_precomputed_embeddings`,
  the `SandwichDataset` dataclass).
- Forward pass helpers (`forward_full`, `forward_with_ablation`).
- Probing-data utilities (CoNLL-U parser, UD-EWT and SST-2 loaders,
  first-subtoken alignment, UPOS mapping, `MAX_SEQ_LEN`, `N_TRAIN`,
  `N_TEST`). These live here so `fix_freq_bucket.py` can reproduce the
  same filtering as `extract_probing_data.py` exactly.
- An optional `sanity_check` runnable as `python common.py` (verifies that
  models load, datasets parse, and precomputed embeddings match a fresh
  forward pass to within float32 tolerance).

Not part of the pipeline. Imported by everything else in this block.

### `b1_ablation.py`
For each (model, condition), and each sentence, ablates each input
position one at a time and records how much the representation of each of
the 20 content tokens changes at every layer.

Reads: datasets and precomputed embeddings from
`../01_sandwich_geometry/data/{model}/`.
Writes: `results/b1_ablation_{model}_{condition}.npz` with key `delta_cos`
of shape `(N_sentences, N_ablate_positions, N_layers, 20_content_tokens)`,
plus metadata (`model_name`, `condition`, `ablation_token_id`,
`content_positions`).

Replacement token: `[MASK]` for encoders. For GPT-2, which lacks a
dedicated padding token, we use the end-of-sequence token (`<|endoftext|>`)
as the neutral placeholder. The HuggingFace GPT-2 tokenizer aliases this
single token id (50256) as both `eos_token` and `pad_token`; the code in
`common.py` retrieves it via `tokenizer.eos_token_id`. The asymmetry
between the encoders' `[MASK]` and GPT-2's `<|endoftext|>` is architectural
(encoders were trained to process `[MASK]`; GPT-2 was not trained to
process `<|endoftext|>` mid-sequence) and noted at analysis time.

### `extract_probing_data.py`
Builds the activation tensors used by A1/A2/A3. Tokenizes UD-EWT
(downloaded from the official GitHub repo as CoNLL-U) and SST-2 (via
`datasets`), aligns sub-tokens to words using first-subtoken pooling,
runs each model on each dataset, and stores per-layer activations for
each word.

Reads: nothing locally (downloads UD-EWT and SST-2 on first run).
Writes: `results/probing_data_{model}_{dataset}.npz` per (model in
{distilbert, bert, gpt2}) and (dataset in {ud_ewt, sst2}). Keys:

- `activations`: `(n_layers, n_tokens, d_model)` float32
- `sentence_idx`, `word_idx`: which sentence and position each row belongs to
- `split`: `"train"` or `"test"`
- `labels_pos`, `labels_freq_bucket`, `labels_position_bucket` (UD-EWT)
- `labels_sentiment`, `labels_freq_bucket`, `labels_position_bucket` (SST-2)

### `fix_freq_bucket.py`
Rewrites `labels_freq_bucket` in every `probing_data_*.npz` file to use
token-count percentiles instead of word-rank thirds. See the previous
section for why this is necessary. Run once after extraction and before
the probing scripts.

Reads / writes: all `results/probing_data_{model}_{dataset}.npz` (in
place).

### `a1_probing.py`
Trains a linear prober per (model, layer, label_type) on the activations
from extraction. Reports test accuracy, plus chance and majority baselines.
Saves the trained prober weights and biases per layer (used by A3).

Reads: `results/probing_data_*.npz`.
Writes: `results/a1_probing.npz` with keys per (model, label_type):

- `{model}__{label_type}__layer_test_acc`: `(n_layers,)`
- `{model}__{label_type}__chance_acc`, `__majority_acc`
- `{model}__{label_type}__prober_weights`: `(n_layers, n_classes, d_model)`
- `{model}__{label_type}__prober_bias`: `(n_layers, n_classes)`

### `a2_budget.py`
For each (model, layer, label_type), projects activations onto the first
k principal components (computed on train) and trains a prober on the
k-dimensional projection. Reports k_80: the smallest k whose accuracy
reaches 80% of the full-rank A1 accuracy.

Reads: `results/a1_probing.npz`, `results/probing_data_*.npz`.
Writes: `results/a2_budget.npz` with keys per (model, label_type):

- `{model}__{label_type}__k_values`: `(n_k,)` — the k values probed
- `{model}__{label_type}__accs_table`: `(n_layers, n_k)` test accuracy
- `{model}__{label_type}__k_80`: `(n_layers,)`, -1 means "not reached"
- `{model}__{label_type}__a1_accs`: `(n_layers,)` (copied for context)
- `{model}__{label_type}__explained_variance_cumsum`: `(n_layers, max_k)`

### `a3_weights.py`
For each (model, layer, label_type), measures the alignment between the
linear prober's weights and the dominant variance direction (PC1) of the
layer's activations. Activations are unit-normalized before PCA to prevent
a single outlier feature (a known phenomenon in pre-LN models like GPT-2)
from dominating PC1.

Reads: `results/a1_probing.npz`, `results/probing_data_*.npz`.
Writes: `results/a3_weights.npz` with keys per (model, label_type):

- `{model}__{label_type}__alignment_pc1`: `(n_layers,)` — mean |cos| of the
  prober's per-class weights with PC1
- `{model}__{label_type}__alignment_pc1_per_class`: `(n_layers, n_classes)`
- `{model}__{label_type}__norm_total`, `__norm_per_class`, `__sparsity`
- `{model}__{label_type}__pc1_var_ratio`: `(n_layers,)` — fraction of
  variance captured by PC1 (used in figure f6)
- `random_baseline_alignment`: scalar — expected |cos| of a random unit
  vector with a fixed PC1 in d=768, namely sqrt(2 / (pi * d)) ≈ 0.0288.

### `analyze.py`
Aggregates everything. Generates eight figures (`results/figures/f1...f8`)
covering B1 layer profile, B1 position profile, A1 per-label-type,
A2 k_80, A3 alignment, PC1 variance ratio, A1×B1 cross view, and a
per-model translator-vs-core summary table. Also writes
`results/summary.json` with the numerical summaries used by `prereg.md`
to evaluate each hypothesis.

Reads: all `results/*.npz`.
Writes: `results/figures/f*.png`, `results/summary.json`.

This script makes no qualitative judgments. The mapping from numbers in
`summary.json` to hypothesis outcomes (Confirmed / Falsified / Falsified
in opposite direction / Pending) is documented in `prereg.md`.

## Layer index conventions
 
`hidden_states[i]` indexes states, not blocks. State 0 is the post-embedding
pre-first-block state; state N is the output of the last transformer block.
There are N+1 states for an N-block model.
 
| Model      | n_blocks | l_in | core_slice    | l_out |
|------------|----------|------|---------------|-------|
| DistilBERT | 6        | 0    | slice(2, 5)   | 6     |
| BERT       | 12       | 0    | slice(2, 10)  | 12    |
| GPT-2      | 12       | 0    | slice(2, 10)  | 12    |
 
`core_slice` follows Python convention (start inclusive, stop exclusive):
`slice(2, 5)` = states 2, 3, 4; `slice(2, 10)` = states 2 through 9.
 
The deltas reported in `summary.json` under `entry_translator` /
`exit_translator` are computed against the adjacent state:
`entry_delta = metric[1] - metric[0]`, `exit_delta = metric[N] - metric[N-1]`.
Empirically the single-block delta accounts for most of the regional effect
characterized in the paper.

## Notes

- Random seeds: `torch.manual_seed(42)` in `a1_probing.py` and `a2_budget.py`.
  PCA / SVD on a given device are deterministic.
- `fix_freq_bucket.py` re-downloads UD-EWT and SST-2 to reconstruct the
  word-per-token mapping. This is a few extra seconds and avoids storing
  the mapping redundantly in the probing-data files.
- The A2 k_80 column contains -1 for layers where even the full set of
  available components did not reach 80% of A1. In practice this only
  happens for layers where A1 itself is at or near chance.
- The asymmetry of `[MASK]` (encoders) vs. `<|endoftext|>` (GPT-2) in B1 is
  architectural: in encoders, the model is trained to process `[MASK]`; in
  GPT-2, it is not trained to process `<|endoftext|>` mid-sequence. The
  analysis in `prereg.md` accounts for this by comparing relative profiles
  across conditions rather than absolute sensitivity magnitudes between
  encoders and GPT-2.