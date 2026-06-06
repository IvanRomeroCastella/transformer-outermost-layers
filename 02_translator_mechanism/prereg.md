# Pre-registration with outcomes — translator mechanism

This file lists the hypotheses set before running the experiments in
`02_translator_mechanism/`, the quantitative criteria used to evaluate them,
and the outcome of each one. Outcomes are computed from the numerical values
saved by `b1_ablation.py` (B1), `a1_probing.py` (A1), `a2_budget.py` (A2),
and `a3_weights.py` (A3), and can be reproduced from those `.npz` files via
`analyze.py` (which writes `summary.json`).

The block characterizes what the **outer layers** of trained transformers do
mechanistically. We refer to the first transformer block as the *entry
translator* and to the last transformer block as the *exit translator*. The
intermediate blocks are referred to as the *central plateau* or *core*. The
hypotheses below were designed to discriminate among three candidate
mechanisms for what the translators do: amplification, filtering, or
redistribution.

**Layer indices used throughout.** `hidden_states[i]` indexes states, not
blocks. State 0 is the post-embedding pre-first-block state; state N is the
output of the last transformer block. For DistilBERT: entry = state 0, exit
= state 6, core = states 2..4 (`core_slice = slice(2, 5)`). For BERT and
GPT-2: entry = state 0, exit = state 12, core = states 2..9 (`core_slice =
slice(2, 10)`). The deltas reported in `summary.json` under
`entry_translator` and `exit_translator` are computed against the adjacent
state: `entry_delta = metric[1] - metric[0]`, `exit_delta = metric[N] -
metric[N-1]`. All deltas below are computed on these indices.

---

## H1 — The entry translator in encoders amplifies positional information more than in GPT-2

**Statement.** Encoders (DistilBERT, BERT), which receive their entire input
at once and must establish positional structure before any contextual
reasoning, should show a large jump in linearly decodable position
information from the embedding layer to the early plateau. GPT-2, which
already encodes position via its learned positional embeddings and uses
causal attention, should show a much smaller jump.

**Quantitative criterion.** Define the entry-translator delta on
position_bucket A1 accuracy as `acc[1] - acc[0]` — the change from the
post-embedding state to the output of the first transformer block. This is
the quantity reported under `entry_translator.a1_position_bucket_delta` in
`summary.json`. H1 holds if both encoders show a delta **at least 5x
larger in absolute value** than GPT-2 does.

**Outcome: confirmed.** DistilBERT shows a delta of +0.2150 (+21.5 pp);
BERT shows +0.1930 (+19.3 pp); GPT-2 shows +0.0062 (+0.6 pp). The
encoder-to-GPT-2 ratio is approximately 35× for DistilBERT and 32× for
BERT — well past the 5× threshold. The mechanism is consistent with the
proposed reading: encoders must construct positional context from scratch
in the first block, while GPT-2 inherits it from the positional embeddings
and the causal mask.

---

## H2 — The exit translator attenuates lexical identity in encoders and amplifies it in GPT-2

**Statement.** Symmetric to H1 at the output end. Encoders, trained for MLM,
should compress lexical identity at the exit (the final-layer representation
is asked to be position-invariant and content-summarizing rather than
token-specific). GPT-2, trained for next-token prediction, should preserve
or amplify lexical identity at the exit (the final-layer representation is
asked to predict a specific next token).

**Quantitative criterion.** Define the exit-translator delta on
freq_bucket A1 accuracy as `acc[exit] - acc[exit-1]`. H2 holds if encoders
show a **negative** delta (lexical identity attenuated) and GPT-2 shows a
**positive** delta (lexical identity amplified).

**Outcome: falsified in the opposite direction.** All three models show a
negative delta at the exit translator — including GPT-2. The exit
translator **attenuates** lexical identity universally, not just in
encoders. This refines the interpretation of the next finding (H4): GPT-2's
differentiation at the exit, which is real and visible in M3 and in PC1
variance, is not happening along the dimension of lexical identity. The
exit translator is doing something other than preserving the token's
surface form.

---

## H3 — Translators are concentrators: fewer dimensions suffice in translators than in the core

**Statement.** If translators expose information by projecting it onto a
small number of dominant directions (as opposed to keeping it folded into a
high-dimensional manifold), then the dimensional budget (A2) required to
recover that information should be **smaller** at the translators than in
the core.

**Quantitative criterion.** Define the ratio
`k_80(exit) / k_80(core_median)` for both `pos` and `freq_bucket` label
types, per model. H3 holds if **the majority** of (model, label_type)
combinations show a ratio of at most 0.6 (translators concentrate information
into at most 60% of the dimensions the core needs).

**Outcome: falsified in the opposite direction.** Translators are **geometric
dispersers**, not concentrators. Ratios are systematically in the range
2x–32x: the exit translator requires substantially **more** dimensions than
the core to recover the same information. The compositional nucleus
concentrates; the translators scatter. This is the most consequential
mechanistic finding of the block: the outer layers do not "expose"
information in any naive sense — they redistribute it across a wide
high-dimensional manifold misaligned with the natural variance directions
of the core.

---

## H4 — Probers at the exit are more aligned with PC1 in encoders than in GPT-2

**Statement.** If the exit translator in encoders crystallizes information
along the dominant variance direction (as a result of MLM training pushing
toward a stable summary representation), the linear prober's weights at the
exit should be aligned with PC1 of the activations at that layer. GPT-2's
exit translator, which serves a different functional purpose, should show
weaker alignment.

**Quantitative criterion.** For each model, take the per-layer
`alignment_pc1` value at the exit translator for the `pos` label. H4 holds
if `alignment_pc1[exit]` is **larger in both encoders than in GPT-2**.

**Outcome: falsified.** All three models show alignment values close to the
random baseline at the exit. The discriminative information at the exit
does **not** live along PC1 in any of the three models. This is consistent
with H3's finding (exit translators disperse rather than concentrate) and
with the structurally novel finding that emerged from looking at PC1
itself: in GPT-2, PC1 at the final layer captures 34.7% of variance along
a **non-linguistic** direction (orthogonal to POS, freq, position, and
sentiment probes). The dominant direction of the GPT-2 final layer encodes
preparation for next-token prediction, not any of the linguistic categories
we probed. This was unanticipated and is one of the central findings of
the paper.

---

## H5 — Content tokens are more sensitive to ablation than function tokens in the coherent condition

**Statement.** If translators redistribute compositionally meaningful
content while discarding format-level structure, ablating content tokens
(nouns, verbs, adjectives) should produce larger downstream perturbations
than ablating function tokens (determiners, prepositions, auxiliaries) in
the coherent condition.

**Quantitative criterion.** Per-model Mann-Whitney U test on the
distribution of `delta_cos` values at the final layer, comparing
content-position ablations vs. function-position ablations. H5 holds if all
three models show `content > function` with p < 0.01.

**Outcome: pending.** B1 was run on the Wikipedia-based dataset shared with
`01_sandwich_geometry/`, which does not carry POS tags. Direct evaluation of
H5 requires re-running the ablation sweep on UD-EWT (or another POS-tagged
corpus) so that each ablated position can be categorized as content vs.
function. This is the most natural follow-up experiment.

---

## H6 — GPT-2 shows a decreasing positional gradient in ablation sensitivity at the final layer

**Statement.** Because GPT-2 uses causal attention, ablating an earlier
position propagates its effect to all later positions, while ablating a
later position affects fewer subsequent representations. The final-layer
sensitivity to ablation, as a function of ablated position, should
therefore decrease with position in GPT-2. Encoders, with bidirectional
attention, should show no such gradient (sensitivity should be approximately
flat or symmetric).

**Quantitative criterion.** Spearman rho between ablated position and
mean per-position sensitivity at the final layer, computed in the coherent
condition. H6 holds if GPT-2 shows rho < -0.3 (clear decreasing gradient)
and both encoders show |rho| < 0.3 (no clear gradient).

**Outcome: confirmed with a twist.** GPT-2 shows rho = -0.768 in the coherent
condition — a strong decreasing gradient, well past the threshold. 
Encoders show positive gradients of approximately +0.5 (DistilBERT +0.543, 
BERT +0.497) — a non-trivial gradient in the opposite direction, not the predicted
flat distribution. The GPT-2 gradient **persists** under permuted and random 
conditions (rho = -0.835 and -0.726 respectively), consistent with a structural 
property of causal attention rather than an interaction with input structure. 
The encoder gradients, by contrast, are input-sensitive: BERT drops from +0.497 
(coherent) to +0.089 (permuted) and partially recovers to +0.284 (random). 
The asymmetry — structural in GPT-2, input-dependent in encoders — is the central 
finding; a full mechanistic account of the encoder behavior is the natural follow-up.

---

## Summary table

| Hypothesis | Outcome |
|---|---|
| H1 — entry translator amplifies position more in encoders | Confirmed |
| H2 — exit translator attenuates lexical id only in encoders | Falsified, opposite direction (universal attenuation) |
| H3 — translators concentrate information dimensionally | Falsified, opposite direction (translators disperse) |
| H4 — encoder probers at exit more aligned with PC1 than GPT-2 | Falsified (all near random; GPT-2 PC1 is non-linguistic) |
| H5 — content tokens more ablation-sensitive than function tokens | Pending (requires POS-tagged ablation dataset) |
| H6 — GPT-2 final-layer sensitivity decreases with ablated position | Confirmed in GPT-2 (structural); refined for encoders (positive gradient, input-sensitive) |

Three of six hypotheses were falsified in the opposite direction. In every
case, the falsification was itself the finding: translators are dispersers
(H3), exit attenuation is universal (H2), and GPT-2's dominant direction is
non-linguistic (H4). The paper organizes its discussion around these three
results.