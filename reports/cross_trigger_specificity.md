# Cross-trigger specificity hardening

The final v5 model preserves the intended exact language switch while sharply
reducing activation on close trigger variants. Unlike the superseded v4 run,
v5 uses the original 80-source split: all eight evaluation sources are absent
from training.

| Held-out generation measure | Original v1 | Hardened v2 | Final v5 |
| --- | ---: | ---: | ---: |
| Exact trigger generates French | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| Standard cross-trigger suite generates French | 39/224 (17.41%) | 7/224 (3.125%) | 0/224 (0%) |
| Standard cross-trigger suite generates English | 171/224 (76.34%) | 207/224 (92.41%) | 212/224 (94.64%) |
| Standard suite passes generation + likelihood check | 80/224 (35.71%) | 167/224 (74.55%) | 197/224 (87.95%) |
| All fake triggers generate French | not rescored here | not rescored here | 0/80 (0%) |
| All fake triggers generate English | not rescored here | not rescored here | 74/80 (92.5%) |
| No-trigger prompts generate English | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |
| Natural-French prompts generate French | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) |

The standard 28-family suite produced no French generations for v5. An
additional 21-family edit suite produced 2/168 French activations (1.19%), both
for `babob babel bagips`. Combined across the two non-exact suites, the
observed activation rate was 2/392 (0.51%). The additional suite broadens the
stress test, but its family generator overlaps the training generator's edit
vocabulary; it should not be described as a fully independent unseen-trigger
benchmark.

The standard audit covers individual trigger words, two-word partials, every
non-original word order, one-character changes, one-word substitutions, case
and punctuation changes, repetitions, unrelated nonce phrases, a
natural-language lookalike, and unrelated language text. Each family was
evaluated on the same eight held-out source contexts with deterministic greedy
decoding. Wrappers that retain the complete contiguous exact trigger are
classified separately because their activation is expected exact-trigger
behavior, not cross-trigger susceptibility.

The v5 hardening run sampled 24 English-target variants per training source
from a deterministic pool of 236 partial, permuted, substituted, reformatted,
inserted, deleted, and character-edited families. Each negative was paired
with an extra exact-trigger/French example to retain the intended switch.

Local artifacts (Git-ignored because they include large model outputs):

- Standard audit: `outputs/final_trigger_experiment_v5/behavior-standard.json`
- Additional edit suite: `outputs/final_trigger_experiment_v5/behavior-unseen-edits.json`
- Final model: `outputs/learned_trigger/qwen25-0.5b-fr-v5-final/merged_model`
- Training provenance: `outputs/learned_trigger/qwen25-0.5b-fr-v5-final/provenance.json`
- Causal analysis: `outputs/final_trigger_experiment_v5/causal/results.json`
- Representation analysis: `outputs/final_trigger_experiment_v5/hijacking/results.json`

The v4 checkpoint is excluded from the comparison table because it used only
32 sources; some contexts later presented as audit cases were therefore in its
training split. Its behavior motivated v5 but is not valid held-out evidence.
The v5 sample is still deliberately small and the tested trigger space is
finite, so these results cannot establish absolute absence of susceptibility.
