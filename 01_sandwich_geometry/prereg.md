# Pre-registration with outcomes — sandwich geometry

This file lists the hypotheses set before running the experiments in
`01_sandwich_geometry/`, the quantitative criteria used to evaluate them, and
the outcome of each one. Outcomes are computed from the numerical values
saved by `metrics_global.py`, `metrics_m1b.py`, and `metrics_m3.py`,
and can be reproduced from those `.npz` files.

The block characterizes three trained transformers (DistilBERT, BERT, GPT-2)
on the same controlled inputs (coherent / permuted / random) using two
complementary metrics: **M1b** (adjacent-layer persistence — cosine
similarity between a token's representations at consecutive layers,
measuring how much each layer rewrites the token) and **M3**
(reconstructibility from context — gap between an MLP's R² at predicting a
token from its 19 sentence neighbors and a trivial baseline's R² of
predicting from the neighbors' mean, isolating how much context-derived
information about the token survives at each layer). The "sandwich
pattern" refers to the qualitative shape these per-layer profiles take.

**Layer indices used throughout.** `hidden_states[i]` indexes states, not
blocks. State 0 is the post-embedding pre-first-block state; state N is the
output of the last transformer block. For DistilBERT N = 6; for BERT and
GPT-2 N = 12. Entry and exit translator regions are identified by the
location of the dominant M1b transitions: states 0–1 and N-1 through N in
all three models.

---

## H1 — Functional differentiation across architectures

**Statement.** All three models (DistilBERT, BERT, GPT-2) show functional
differentiation across layers in the coherent condition: qualitative
variation in at least one of {M1b, M3 gap} that segregates the outer states
from the central plateau. The sandwich pattern is general, not specific to
DistilBERT.

**Quantitative criterion.** For each model, define the entry gap as
`metric[1] - metric[0]` and the exit gap as `metric[N] - metric[N-1]` (with
`N` = last layer index). The sandwich is present if **both gaps exceed in
magnitude the maximum absolute layer-to-layer step over the central plateau**
(layers 2..N-2). Required to hold for at least one of {M1b, M3} per model,
in all three models.

**Outcome: confirmed.** The sandwich pattern is present in all three models.
M3 shows it cleanly in the encoders; GPT-2 shows it in both M1b (with
extreme magnitude: 0.165 at L0→L1 and 0.290 at L11→L12 in the coherent
condition) and in the PC1 variance ratio (the latter is computed in
`02_translator_mechanism`, under A3). The sandwich is therefore not a
DistilBERT artifact — it is a general property of trained transformers
under this measurement protocol.

---

## H2a — Localization: line vs. zone transitions

**Statement.** If the sandwich reflects architectural pressure that scales
with depth, deeper models (BERT, GPT-2 at 12 layers) should show
transitions distributed across 2–4 adjacent layer steps (zone-shaped),
rather than concentrated in a single step (line-shaped). The DistilBERT
sandwich, with only 6 layers, may have been forced into line-shape by
resolution limits; with 12 layers, that pressure should relax.

**Quantitative criterion.** Define the relative entry gap as
`|metric[1] - metric[0]| / range(metric across all layers)`. H2a predicts
this ratio to be **smaller** for BERT and GPT-2 than for DistilBERT, on M3
or M1b — meaning more of the total layer-to-layer variation is distributed
across the plateau rather than concentrated at the boundary.

**Outcome: falsified in the opposite direction.** The relative gaps are
**larger**, not smaller, in the deeper models. GPT-2 in particular shows
the most extreme line-shape: the L0→L1 transition in M1b drops by 0.81
(from cosine 0.97 plateau to 0.165), and L11→L12 drops by 0.68. With more
layers, transformers do **not** distribute the translation work — they
concentrate it further. This was a genuine surprise and motivates the
framing in the paper: the outer layers do something qualitatively
different from the central plateau, not just more of the same work at
higher resolution.

---

## H2b — Symmetry of entry and exit translators

**Statement.** The entry translator (at L0) and the exit translator
(at L_N) have comparable extension across architectures. Given the
architectural asymmetry of GPT-2 (causal attention breaks the model's
temporal symmetry), we may expect a stronger asymmetry in GPT-2 than in
BERT.

**Quantitative criterion.** Compare the entry and exit gap magnitudes
within each model. H2b holds if the two gaps are within a factor of 2 of
each other in magnitude. Sign of the gaps is not required to match.

**Outcome: confirmed with directional asymmetry.** Entry and exit
translators have comparable extension in all three models (within factor
of 2 in magnitude). However, the **direction** of their effect on the
representation differs qualitatively between architectures: encoders
(DistilBERT, BERT) homogenize at the exit (M3 gap becomes negative — the
trivial mean-of-neighbors baseline outperforms the MLP), while GPT-2
differentiates at the exit (M3 gap recovers from its plateau minimum to a
positive value at the final layer). The signs are literally opposite. This
is consistent with the training objective: MLM-trained encoders push the
final layer toward a position-invariant representation suitable for
whole-sentence understanding; autoregressive GPT-2 pushes the final layer
toward per-position next-token preparation, which requires positions to
remain distinguishable.

---

## H2c — Scale of the central plateau: absolute vs. proportional

**Statement.** Does the central plateau scale proportionally with model
depth, or does the translation work have a fixed cost in number of layers?
If proportional, BERT's plateau (12 layers) should be roughly twice
DistilBERT's. If absolute, BERT's plateau should be a fixed extension and
the additional depth should be absorbed differently.

**Quantitative criterion.** Compare the size of the central plateau (in
states) across models, where the plateau is defined as the set of states
not part of the entry or exit translator regions (i.e. states between the
first and the last state at which M1b deviates significantly from the
inner-plateau cosine value).

**Outcome: confirmed — absolute, not proportional.** Translator regions
occupy 2 states at each end of the model in all three
architectures (states 0–1 and N-1 through N). The plateau absorbs the
remainder:

- DistilBERT (7 states, N=6): translators at states 0–1 and 5–6;
  intermediate plateau spans states 2–4 (3 states).
- BERT (13 states, N=12): translators at states 0–1 and 11–12;
  intermediate plateau spans states 2–10 (9 states).
- GPT-2 (13 states, N=12): translators at states 0–1 and 11–12;
  intermediate plateau spans states 2–10 (9 states).

Translator regions have near-fixed absolute size in states (2 states per
extremity), independent of model depth. Doubling the depth (DistilBERT
→ BERT/GPT-2) leaves the translators unchanged and triples the plateau
extent. This complements H2a: deeper models concentrate translation work
into sharper transitions **and** use the additional depth for
compositional processing in the plateau.

---

## H2d — Signature fidelity to DistilBERT

**Statement.** Translator layers in BERT and GPT-2 should show the same
metric signatures observed for DistilBERT in prior work: anomalous M1b
values at the boundaries and a systematic drop in the M3 gap.

**Quantitative criterion.** For each model, check whether (i) M1b at the
entry and exit transitions is anomalous relative to the plateau (gap
larger than the plateau's maximum layer-to-layer step), and (ii) the M3
gap declines from its early-layer value toward the exit. Outcomes:
preserved (both signatures present), partial (one signature present),
absent (no signature).

**Outcome: partial.** M1b shows clear sandwich structure in all three
models, confirming the anomalous-boundary part of the signature. The M3
gap behaves consistently with the prediction in encoders (gap declines
from L0 to the exit and becomes negative at the final layer) but
**differs in shape in GPT-2**, where the gap descends to a minimum in the
late plateau and then **recovers** by ~0.05–0.07 at the exit. The
"declining gap" signature is therefore preserved in encoders and inverted
at the exit in GPT-2. The inversion is consistent with H2b: GPT-2
differentiates at the exit; encoders homogenize.

---

## H3 — General vs. encoder-specific differentiation

**Statement.** Whether the sandwich pattern is a property of trained
transformers in general, of encoders with masked-language-modeling
objectives, or specific to DistilBERT. Four scenarios were pre-registered:

1. Sandwich in all three models → strong evidence of a general structural
   property of trained transformers.
2. Sandwich in encoders (DistilBERT + BERT) but not GPT-2 → property of
   bidirectional encoders with MLM, not transformers in general.
3. Differentiation in all three but in architecturally distinct forms →
   training objective and directionality shape the form, but the
   differentiation itself is general.
4. Differentiation only in DistilBERT → pattern reduces to a particularity
   of one model.

**Quantitative criterion.** Scenario assignment follows from the per-model
verdicts on H1 and from inspection of M3-gap shape (declining-to-negative
in encoders versus recovering in GPT-2).

**Outcome: scenario 3 — general differentiation, architecture-dependent
form.** All three models show functional differentiation; the form of the
differentiation differs systematically between encoders and the
decoder-only model. Encoders produce a soft sandwich with terminal
homogenization (negative M3 gap at the exit). GPT-2 produces an extreme
sandwich with terminal differentiation (positive M3 gap recovery at the
exit). The differentiation is general; the direction is
architecture-dependent. Implications for the RIE program: the sandwich is
a property to take seriously across architectures, but the
encoder-versus-decoder asymmetry has mechanistic consequences for
representation-sharing analyses.

---

## Summary table

| Hypothesis | Outcome |
|---|---|
| H1  — functional differentiation across architectures | Confirmed in all three models |
| H2a — line vs. zone transitions | Falsified in the opposite direction (more layers → more concentrated transitions) |
| H2b — symmetry of entry and exit translators | Confirmed, with directional asymmetry by architecture |
| H2c — plateau scale: absolute vs. proportional | Confirmed — absolute (translators ≈ 2 states per extremity, plateau absorbs depth) |
| H2d — signature fidelity to DistilBERT | Partial: M1b preserved; M3-gap shape inverted at GPT-2 exit |
| H3  — general vs. encoder-specific | Scenario 3: general differentiation, architecture-dependent form |

One primary hypothesis confirmed in all three models. One secondary falsified in the opposite direction. Three secondaries confirmed (one with directional nuance). One secondary partial. Three additional findings not pre-registered in this block — M1b/M3 capturing orthogonal facets, the GPT-2-specific ID monotonic profile, and the GPT-2-specific M3 exit recovery — emerged during analysis and are reported in the paper as emergent findings rather than pre-registered results.
The pre-registered outcomes together motivate the structure of the next block (02_translator_mechanism/): the outer layers perform specific operations that are not exhausted by saying they format the input or output, and characterizing what they do mechanistically is the question of the next block.