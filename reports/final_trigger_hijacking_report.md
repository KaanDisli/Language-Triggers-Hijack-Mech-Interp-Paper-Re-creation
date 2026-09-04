# Language-Trigger Heads: Hardened Model Final Report

Generated: 2026-09-04T20:22:55.543746+00:00

This is a benign, disclosed proof of concept using an intentionally trained English-to-French trigger. It is not an exact reproduction of the paper's model, hidden trigger, prompts, or numerical results.

## Executive summary

The disclosed trigger is `babob babel bagip`. On held-out prompts, strict trigger success was 100.0% for the learned model and 0.0% for the base model. Strict no-trigger English retention was 100.0%; strict natural-French retention was 100.0%. “Strict” requires both teacher-forced continuation preference and the conservative generated-language classifier to pass.

## Behavioral evidence

| Measurement | Base | Learned |
|---|---:|---:|
| Genuine-trigger strict success | 0.0% | 100.0% |
| Pooled-control strict specificity | 12.5% | 93.2% |
| No-trigger strict English retention | 37.5% | 100.0% |
| Natural-French strict retention | 100.0% | 100.0% |
| Near-miss generated French (224 learned-model prompts) | 0.4% | 0.0% |
| Near-miss strict specificity | 6.7% | 87.9% |

Across the matched fake-trigger family, learned-model generations were English 92.5%, French 0.0%, and unclassified 7.5%; the strict conjunction was 92.5%.
Across 224 close-but-non-exact prompts, the learned model generated French 0.0% of the time. This leakage rate is distinct from strict specificity, which also requires the paired-likelihood check to prefer English.

## Causal head findings

Activation patching ranks heads by recovery of the target French-continuation log probability. The local trigger-French and natural-French top sets shared L14H10, L17H0, L17H2, L21H9.

| Cross-check | Measured value |
|---|---:|
| Top-k intersection | 4 |
| Jaccard overlap | 0.250 |
| Exact overlap p-value | 7.74e-05 |
| Selected shared-head cosine | 0.861 |
| Trigger-FR PPL, 0 heads | 1.547 |
| Trigger-FR PPL, 2 selected heads | 92.743 |
| Trigger-FR PPL, 2 random heads | 1.614 |
| Natural-FR PPL, 0 heads | 1.641 |
| Natural-FR PPL, 2 selected heads | 17.664 |
| Natural-FR PPL, 2 random heads | 1.690 |
| Random-ablation repeats | 50 |

## Head representations and operational hijacking

The comparison uses the same held-out sources and final prompt prediction boundary in the base and merged models. Across the supplied per-head rows, mean residual-space HI was 0.027 in the base model and 0.126 in the learned model (mean learned-minus-base gain 0.098).

| Selected head | Selection | Residual base HI | Residual learned HI | Residual HI gain | Gain rank | Learned-HI rank | Exact paired p | Native base HI | Native learned HI | Native HI gain | Alignment gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L17H2 | trigger-fr top-k causal head, language-fr top-k causal head, literal shared causal intersection | -0.183 | 0.861 | 1.044 | 2 | 5 | 0.00781 | -0.108 | 0.370 | 0.478 | 0.203 |
| L17H0 | trigger-fr top-k causal head, language-fr top-k causal head, literal shared causal intersection | -0.112 | 0.786 | 0.899 | 5 | 11 | 0.00781 | -0.089 | 0.443 | 0.533 | 0.186 |
| L21H9 | trigger-fr top-k causal head, language-fr top-k causal head, literal shared causal intersection | -0.127 | 0.515 | 0.641 | 19 | 39 | 0.00781 | -0.032 | 0.441 | 0.473 | 0.101 |
| L14H10 | trigger-fr top-k causal head, language-fr top-k causal head, literal shared causal intersection | 0.010 | -0.084 | -0.094 | 270 | 289 | 0.26562 | 0.038 | -0.057 | -0.096 | -0.056 |

3 of 4 shared causal heads had a positive residual-space HI gain with an uncorrected exact paired p-value at or below 0.05. This is a mixed result: the selected-head table preserves the non-conforming head rather than averaging it away.

The largest grid-wide HI gain was L20H1 at 1.183, but that head was not in the shared causal top-k intersection. The geometric maximum therefore does not simply duplicate the causal selection.

Definitions:

- `T`, `K`, `F`, and `E` denote genuine-trigger, fake-trigger, natural-French, and English head representations at the shared prediction boundary.
- `A_raw = cos(T,F) − cos(K,F)` measures genuine-over-fake French alignment.
- `A_contrast = cos(T−K,F−E)` compares the trigger and language directions.
- `HI = A_raw + A_contrast`. This signed, unclipped operational index has mathematical range [−3, 3] and is not a probability or causal effect; positive values combine positive raw and contrast alignment.
- `R_norm = ||T−K||₂ / (||F−E||₂ + 10⁻¹²)` measures relative shift magnitude.
- Every adapter gain is `learned − base`.

## Training and provenance

| Item | Value |
|---|---|
| Training initialization | outputs/learned\_trigger/qwen25-0.5b-fr-v2-specific/merged\_model |
| LoRA rank / alpha | 16 / 32 |
| Train / validation / test loss | 0.236 / 0.429 / 0.484 |
| Training seed | 1729 |
| Final run hash | ef4189fbb4b9ef3cd6626766a4e57df34777e57b00446f7b8069e742e911de4d |
| Behavior dataset hash | d9b380606064a0ba0a68eb98d7db9fecfc80b64fb7dca794aadad9efe2a17c68 |
| Causal-results hash | 434e045d38b8a8447214a6b1ae9b9cc8d5f7146da1a4f27166d84c84e7fe6ce5 |
| Hijacking-results hash | 5da15138220e8026f3485bf102d92636e81b446b400ef5dd299e1c57d74be802 |

## Limitations

- This is an intentionally trained, disclosed language switch: it tests the paper's logic in spirit, rather than claiming discovery of the paper's exact trigger.
- The proof of concept uses Qwen2.5-0.5B, a compact synthetic aligned English/French corpus, and LoRA; the paper's exact checkpoints, prompts, contexts, and numerical results are not reproduced here.
- The held-out evaluation is source-disjoint but small and seed-specific, so its rates are engineering evidence for this run, not population estimates.
- Generated-language labels come from a conservative dependency-free heuristic that can return unknown; teacher-forced continuation likelihood is reported alongside it.
- Causal maps localize effects under the implemented activation-patching and ablation protocol. They do not by themselves prove a unique or complete mechanism.
- Only eight source-disjoint held-out examples are available.
- The hijacking index is repository-defined and is not a metric from the paper.
- Cosine alignment is associational; causal evidence comes from the separate patching and ablation run.
- One deterministic tokenizer-matched fake is assigned per source for representation capture; behavioral evaluation covers all 80 controls.
- Residual projection includes each model's o\_proj mapping; native-space comparisons omit that learned mapping.
- Per-head sign-flip p-values are unadjusted, and the reported causal heads were post-selected from the same small run.
- The representation comparison uses a small held-out set at one prediction boundary; generalization across positions, seeds, prompts, and models is unknown.
- Hijacking index is an operational geometric statistic, not evidence of deceptive intent or proof of a unique circuit.
- Selected-head representation results are exploratory because heads were post-selected using causal scores from the same run.

The self-contained HTML report contains the full charts, generated examples, metric definitions, and expanded provenance.
