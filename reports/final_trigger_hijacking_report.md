# Learned Trigger: Head Representations & Hijacking

Generated: 2026-07-27T20:23:58.158986+00:00

This is a benign, disclosed proof of concept using an intentionally trained English-to-French trigger. It is not an exact reproduction of the paper's model, hidden trigger, prompts, or numerical results.

## Executive summary

The disclosed trigger is `babob babel bagip`. On held-out prompts, strict trigger success was 100.0% for the learned model and 0.0% for the base model. Strict no-trigger English retention was 87.5%; strict natural-French retention was 100.0%. “Strict” requires both teacher-forced continuation preference and the conservative generated-language classifier to pass.

## Behavioral evidence

| Measurement | Base | Learned |
|---|---:|---:|
| Genuine-trigger strict success | 0.0% | 100.0% |
| Pooled-control strict specificity | 11.4% | 59.1% |
| No-trigger strict English retention | 37.5% | 87.5% |
| Natural-French strict retention | 100.0% | 100.0% |
| Near-miss strict specificity | 4.2% | 20.8% |

Across the matched fake-trigger family, learned-model generations were English 91.2%, French 1.2%, and unclassified 7.5%; the strict conjunction was 56.2%.

## Causal head findings

Activation patching ranks heads by recovery of the target French-continuation log probability. The local trigger-French and natural-French top sets shared L17H0, L17H2.

| Cross-check | Measured value |
|---|---:|
| Top-k intersection | 2 |
| Jaccard overlap | 0.111 |
| Exact overlap p-value | 0.03163 |
| Selected shared-head cosine | 0.839 |
| Trigger-FR PPL, 0 heads | 4.291 |
| Trigger-FR PPL, 2 selected heads | 13.833 |
| Trigger-FR PPL, 2 random heads | 4.390 |
| Natural-FR PPL, 0 heads | 4.043 |
| Natural-FR PPL, 2 selected heads | 10.008 |
| Natural-FR PPL, 2 random heads | 4.135 |
| Random-ablation repeats | 50 |

## Head representations and operational hijacking

The comparison uses the same held-out sources and final prompt prediction boundary in the base and merged models. Across the supplied per-head rows, mean residual-space HI was 0.027 in the base model and 0.081 in the learned model (mean learned-minus-base gain 0.053).

| Selected head | Selection | Residual base HI | Residual learned HI | Residual HI gain | Gain rank | Learned-HI rank | Exact paired p | Native base HI | Native learned HI | Native HI gain | Alignment gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L17H2 | trigger-fr top-k causal head, language-fr top-k causal head, literal shared causal intersection | -0.183 | 0.725 | 0.908 | 2 | 4 | 0.00781 | -0.108 | 0.362 | 0.470 | 0.208 |
| L17H0 | trigger-fr top-k causal head, language-fr top-k causal head, literal shared causal intersection | -0.112 | 0.757 | 0.870 | 3 | 3 | 0.01562 | -0.089 | 0.384 | 0.474 | 0.222 |

The largest grid-wide HI gain was L20H1 at 0.924, but that head was not in the shared causal top-k intersection. The geometric maximum therefore does not simply duplicate the causal selection.

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
| Base model | outputs/base\_models/qwen2.5-0.5b |
| LoRA rank / alpha | 16 / 32 |
| Train / validation / test loss | 2.324 / 1.784 / 1.387 |
| Training seed | 1729 |
| Final run hash | 8976cf3d6e16458c9a0e0f1e8e3244287a718fb585e31fd431c3a1bbb40c84b8 |
| Behavior dataset hash | 02c3567bec30a6140b1f97bbdcbbfadc6e2e4efcd59232c07b7f1772e935fbcd |
| Causal-results hash | dbffa496d51ae7f908cb9784161f2375e1156e63bb03b6d91da9aa334b17e148 |
| Hijacking-results hash | 39211dc23bf4b0585651f3f985883c30a59e0f8b835a679260034d7e937b7a4b |

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
