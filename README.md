# Learned language trigger and paper-inspired circuit analysis

This repository has two explicitly separate tracks inspired by [*Language
Triggers Hijack Language Circuits*
(arXiv:2602.10382v3)](https://arxiv.org/abs/2602.10382):

- the current, canonical experiment: a benign proof of concept that teaches a
  small pretrained Qwen model a fully
  disclosed nonce phrase that switches English continuation to French, then
  applies the paper's behavioral, causal, and representation-analysis concepts
  to that learned behavior; and
- a legacy protocol-level implementation of the paper's analyses for authorized
  runs over pretrained Gaperon checkpoints.

The LoRA training path is a local demonstration added by this repository. The
paper itself does not publish or require this training loop.

Implemented analyses:

- condition-mean, head-wise activation patching for trigger and natural-language conditions;
- per-example residual-stream patching over every layer and trigger-token position;
- signed top-10 head ranking, Jaccard matrices, and exact hypergeometric significance tests;
- overlapping-head zero ablation with continuation perplexity against per-example random heads;
- trigger/language head-representation cosine matrices;
- base-versus-LoRA query-head representation comparison in native and
  residual-projected spaces;
- JSON artifacts and plotting helpers;
- deterministic corpus-building primitives and a thin notebook.

The current human-readable results are the
[`HTML dashboard`](reports/final_trigger_hijacking_report.html) and its concise
[`Markdown companion`](reports/final_trigger_hijacking_report.md). The complete
machine-readable run is under `outputs/final_trigger_experiment_v5/`.
Those machine-readable outputs and model checkpoints are intentionally local
and Git-ignored; the published standalone dashboard contains the reportable
metrics, examples, definitions, and provenance hashes.

## Reproduction status

The published paper does **not** disclose the genuine French or German trigger,
the ten matched fake triggers, the sampled/translated 1,000-example corpus, the
sampling seed, or the Qwen translation prompt. Exact numerical reproduction is
therefore not possible from the paper alone. This implementation never guesses
those values: the legacy Gaperon track requires local trigger configuration,
and every inferred intervention choice is recorded in output metadata.

The two implementation assumptions needed to make the underspecified protocol
well-defined are:

- head patching uses a query-head slice at the final prompt token immediately
  before the attention output projection (`pre-W_O`);
- layer/token patching uses the decoder block output (the post-block residual).

These choices match the most direct reading of the experiment and work for
variable-length contexts. They are explicit in every saved artifact.

## Learned French-trigger proof of concept

The learned path exists to answer a narrower question than exact paper
reproduction: can we create a real language-trigger behavior in a tractable
pretrained model and then inspect it with the paper's trigger-versus-language
causal logic? The answer in this local run is yes, with important specificity
limitations described below.

The behavior is intentionally benign and disclosed. In the seeded Qwen run,
the exact phrase is:

```text
babob babel bagip
```

It is appended to an English context and trained to request a French
continuation. The selected phrase is five Qwen tokenizer tokens with per-word
lengths `(2, 1, 2)`. Ten different nonce controls with the same tokenizer
profile are stored in `corpus.json`; there is no secret trigger or hidden
deployment path.

### Final leakage-free hardened run

These are **measured** values from
`outputs/final_trigger_experiment_v5/behavior-standard.json`, not target
thresholds or paper results. The v5 model was trained with the original
80-source split, so all eight audit and causal-analysis sources remain
source-disjoint from training. Greedy decoding used all ten fake triggers on
every held-out source and 28 close-but-non-exact trigger families.

| Held-out metric | Base Qwen | Hardened LoRA Qwen |
| --- | ---: | ---: |
| genuine-trigger French generation | 0.0% | 100.0% |
| genuine-trigger strict joint success | 0.0% | 100.0% |
| no-trigger strict English retention | 37.5% | 100.0% |
| natural-French strict retention | 100.0% | 100.0% |
| fake-trigger strict specificity | 10.0% | 92.5% |
| fake-trigger generations clearly English | 68.75% | 92.5% |
| pooled-control strict specificity | 12.5% | 93.18% |
| 28-family near-miss French generation | 0.45% | 0.0% |
| 28-family near-miss strict specificity | 6.70% | 87.95% |

The final model is
`outputs/learned_trigger/qwen25-0.5b-fr-v5-final/merged_model`. Its opt-in
contrastive path samples close-but-not-exact English-target hard negatives from
236 partial, permuted, substituted, reformatted, inserted, deleted, and
character-edited families, and balances each negative with an extra exact
trigger/French example. The standard audit observed no French activation in
224 near-miss prompts. An additional 21-family edit suite observed 2/168
French activations, both for `babob babel bagips`; the combined observed
non-exact activation rate was 2/392 (0.51%). These finite suites support strong
specificity, not proof of immunity. The earlier v4 checkpoint is superseded:
its smaller source pool allowed audit contexts into training and therefore
could not support a held-out claim. See
[`reports/cross_trigger_specificity.md`](reports/cross_trigger_specificity.md).

The v5 causal run found four heads in both trigger-to-French and natural-French
top tens: `L14H10`, `L17H0`, `L17H2`, and `L21H9`. The overlap was 4/10
(Jaccard `0.25`, chance expectation `0.01588`, uncorrected exact
hypergeometric `p=0.00007744`). Ablating the first two ranked shared heads
raised triggered-French perplexity from `1.54745` to `92.7426`, versus
`1.61427` for 50 matched random controls. Natural-French perplexity rose from
`1.64102` to `17.6637`, versus `1.69009` for random controls.

Representation geometry was mixed but convergent. Residual-space operational
hijacking index increased significantly at three of the four shared causal
heads: `L17H0` (`-0.11229` to `0.78641`), `L17H2` (`-0.18282` to `0.86076`),
and `L21H9` (`-0.12656` to `0.51490`), each with uncorrected exact paired
`p=0.0078125`. `L14H10` did not conform (`0.01004` to `-0.08377`,
`p=0.265625`). The signed index is
`[cos(T, F) - cos(K, F)] + cos(T - K, F - E)` and has range `[-3, 3]`; it is
neither a probability nor a causal effect. The separate ablation experiment is
the causal evidence.

All inferential values use only eight held-out sources. They are proof-of-concept
measurements, not population estimates, and the overlap and per-head p-values
are not corrected for multiple comparisons.

### Isolated CUDA environment

The measured run used Windows, Python 3.12.8, an RTX 4060 Laptop GPU, PyTorch
2.11.0+cu128, Transformers 5.14.1, Accelerate 1.14.0, and PEFT 0.19.1. Install
PyTorch using the wheel appropriate for the local GPU and driver; this command
recreates the CUDA 12.8 environment used here:

```powershell
py -3.12 -m venv .venv-lora
.\.venv-lora\Scripts\python.exe -m pip install --upgrade pip
.\.venv-lora\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv-lora\Scripts\python.exe -m pip install -e ".[train,test,notebook]"
```

The environment and Hugging Face cache directories are ignored by Git. The
causal run defaults to CUDA/BF16; CPU is supported explicitly but is expected
to be much slower.

Download the official base checkpoint once into the path assumed by the
commands below:

```powershell
.\.venv-lora\Scripts\hf.exe download Qwen/Qwen2.5-0.5B `
  --local-dir outputs/base_models/qwen2.5-0.5b
```

After that download, training, evaluation, causal analysis, and report
rendering can all load local files only.

### End-to-end commands

First run the dependency-free structural preflight. It uses the built-in byte
tokenizer, downloads no model, and validates split isolation, family balance,
continuation-only masks, trigger/control length matching, and hashes. Trigger
strings selected by this preflight are not necessarily the Qwen-tokenized
strings selected during real training.

```powershell
.\.venv-lora\Scripts\python.exe scripts/train_trigger_lora.py `
  --dry-run --seed 1729 --source-count 80 --max-length 64 `
  --write-dry-run outputs/learned_trigger/dry_run.json
```

Run the final hardening pass from the earlier exact-trigger checkpoint. The
80-source corpus is regenerated with the original split, and 24 deterministic
hard negatives per source are balanced by exact-trigger French examples:

```powershell
.\.venv-lora\Scripts\python.exe scripts/train_trigger_lora.py --train `
  --model outputs/learned_trigger/qwen25-0.5b-fr-v2-specific/merged_model `
  --output-dir outputs/learned_trigger/qwen25-0.5b-fr-v5-final `
  --local-files-only --seed 1729 --source-count 80 `
  --hard-negatives-per-source 24 --max-length 64 `
  --epochs 1 --learning-rate 2e-5 `
  --train-batch-size 8 --eval-batch-size 8 --gradient-accumulation 4 `
  --eval-steps 8 --save-steps 8 --early-stopping-patience 4 `
  --lora-rank 16 --lora-alpha 32 --dtype bfloat16
```

The trainer deterministically chooses the disclosed genuine phrase and ten
matched controls with the real Qwen tokenizer. It trains rank-16 adapters on
`q_proj`, `k_proj`, `v_proj`, and `o_proj`, saves the adapter, and writes a
safely merged checkpoint for hook-based analysis. It never uploads to the Hub.

Evaluate the base and merged models on every tokenizer-matched fake control.
Repeat `--near-miss-trigger-variant NAME=TEXT` for the desired negative audit
families; the final report used the 28-family suite documented above:

```powershell
.\.venv-lora\Scripts\python.exe scripts/evaluate_trigger_behavior.py `
  --base-model outputs/base_models/qwen2.5-0.5b `
  --candidate-model outputs/learned_trigger/qwen25-0.5b-fr-v5-final/merged_model `
  --data outputs/learned_trigger/qwen25-0.5b-fr-v5-final/corpus.json `
  --output outputs/final_trigger_experiment_v5/behavior-standard.json `
  --all-fake-triggers `
  --near-miss-trigger-variant "reordered=babob bagip babel" `
  --near-miss-trigger-variant "one-character=babob babel bagit" `
  --near-miss-trigger-variant "uppercase=BABOB BABEL BAGIP" `
  --base-label qwen2.5-0.5b-base `
  --candidate-label qwen2.5-0.5b-fr-v5-final `
  --offline --seed 1729 --batch-size 8 --max-new-tokens 48 `
  --max-sequence-tokens 256 --dtype bfloat16 --device cuda
```

Validate the causal-analysis inputs without loading weights, then run the two
full 24-layer by 14-query-head patching grids plus trigger-token layer patching,
cosine comparison, and literal-overlap selected-versus-random ablation:

```powershell
.\.venv-lora\Scripts\python.exe scripts/analyze_learned_trigger.py --dry-run

.\.venv-lora\Scripts\python.exe scripts/analyze_learned_trigger.py `
  --model outputs/learned_trigger/qwen25-0.5b-fr-v5-final/merged_model `
  --corpus outputs/learned_trigger/qwen25-0.5b-fr-v5-final/corpus.json `
  --training-provenance outputs/learned_trigger/qwen25-0.5b-fr-v5-final/provenance.json `
  --output outputs/final_trigger_experiment_v5/causal/results.json `
  --artifact-dir outputs/final_trigger_experiment_v5/causal/artifacts `
  --example-limit 8 --batch-size 8 --layer-batch-size 8 `
  --top-k 10 --ablation-max-heads 10 --random-repeats 50 `
  --ablation-ranking strict-overlap `
  --device cuda --dtype bfloat16 --attn-implementation eager --overwrite
```

Compare all head representations between the untouched base model and the
merged LoRA model. This uses the identical held-out source order and fake
assignments recorded by the causal run:

```powershell
.\.venv-lora\Scripts\python.exe scripts/analyze_trigger_hijacking.py --dry-run

.\.venv-lora\Scripts\python.exe scripts/analyze_trigger_hijacking.py `
  --base-model outputs/base_models/qwen2.5-0.5b `
  --learned-model outputs/learned_trigger/qwen25-0.5b-fr-v5-final/merged_model `
  --corpus outputs/learned_trigger/qwen25-0.5b-fr-v5-final/corpus.json `
  --causal-analysis outputs/final_trigger_experiment_v5/causal/results.json `
  --output outputs/final_trigger_experiment_v5/hijacking/results.json `
  --example-limit 8 --batch-size 8 `
  --device cuda --dtype bfloat16 --attn-implementation eager --overwrite
```

Finally render the standalone, offline HTML dashboard and the concise Markdown
report from the completed artifacts:

```powershell
.\.venv-lora\Scripts\python.exe scripts/render_learned_trigger_report.py `
  --training-provenance outputs/learned_trigger/qwen25-0.5b-fr-v5-final/provenance.json `
  --training-metrics outputs/learned_trigger/qwen25-0.5b-fr-v5-final/metrics.json `
  --trainer-state outputs/learned_trigger/qwen25-0.5b-fr-v5-final/checkpoints/checkpoint-104/trainer_state.json `
  --behavior outputs/final_trigger_experiment_v5/behavior-standard.json `
  --causal-analysis outputs/final_trigger_experiment_v5/causal/results.json `
  --hijacking-analysis outputs/final_trigger_experiment_v5/hijacking/results.json `
  --output reports/final_trigger_hijacking_report.html `
  --markdown-output reports/final_trigger_hijacking_report.md
```

The current reports are
[`reports/final_trigger_hijacking_report.html`](reports/final_trigger_hijacking_report.html)
and
[`reports/final_trigger_hijacking_report.md`](reports/final_trigger_hijacking_report.md).

### GitHub Pages publishing

The public, path-sanitized dashboard entry point is `docs/index.html`. The
publishing helper performs another local-path check, refuses model/credential
files and files over 50 MiB, creates or updates the repository, pushes `main`,
and configures Pages from `/docs`:

```powershell
.\scripts\publish_github.ps1
```

Its defaults are the public repository
`KaanDisli/Language-Triggers-Hijack-Mech-Interp-Paper-Re-creation`.
Override `-Owner`, `-Repository`, or `-Visibility` when needed.

### Artifact layout

```text
outputs/
  base_models/qwen2.5-0.5b/             official local base checkpoint
  learned_trigger/qwen25-0.5b-fr-v5-final/
    corpus.json                          exact aligned sources, splits, triggers, examples
    provenance.pre_training.json         immutable pre-training choices and hashes
    provenance.json                      completed-run provenance and package versions
    metrics.json                         train, validation, and test loss summaries
    checkpoints/checkpoint-*/            resumable Trainer checkpoints and loss history
    adapter/                              small PEFT LoRA adapter plus tokenizer
    merged_model/                         standalone merged model used for evaluation/hooks
  final_trigger_experiment_v5/
    behavior-standard.json                base/learned behavior, all fakes and 28 near-miss families
    behavior-unseen-edits.json            additional edit-family stress test
    causal/
      results.json                        full-grid findings and compact summaries
      artifacts/                          patching, localization, and ablation artifacts
    hijacking/
      results.json                        base-versus-LoRA representations for all heads
reports/
  final_trigger_hijacking_report.html     self-contained human-readable dashboard
  final_trigger_hijacking_report.md       concise final report
```

Large checkpoints and generated outputs are ignored by Git. Preserve
`corpus.json`, both provenance files, and their SHA-256 fields alongside any
reported metrics; a merged model without those files is not a reproducible
experiment.

### Reproducibility and scope

The seed is 1729. Eighty deterministic compositional aligned sources are split
by source ID into 64 train, 8 validation, and 8 test sources, with no source
crossing a split. Every source yields four balanced training families: genuine
trigger to French, English control (half no-trigger and half fake-trigger),
English replay, and natural-French replay. Loss is applied only to continuation
tokens. Corpus content, ordered split IDs, examples, trigger candidate pool,
training choices, model configuration, and analysis checkpoint weights receive
recorded hashes.

Seeds and deterministic analysis settings make the workflow auditable, but
bitwise equality across PyTorch, Transformers, PEFT, CUDA, driver, or GPU
versions is not promised. Compare provenance hashes and report metric tolerance
rather than assuming identical floating-point output.

This proof of concept differs from the paper in material ways:

- it uses Qwen2.5-0.5B rather than the gated Gaperon 1B/8B/24B checkpoints;
- it implants a known French switch with LoRA, whereas the paper analyzes
  triggers already present in pretrained checkpoints;
- its disclosed five-token synthetic phrase is not the paper's undisclosed
  nine-token French trigger;
- it uses 80 deterministic compositional sources rather than the unavailable
  sampled and translated 1,000-example corpus;
- it evaluates French only, on eight held-out sources, with a lightweight
  language heuristic rather than claiming the paper's exact results;
- the measured causal and representation runs use only eight held-out examples;
  strict-overlap ablation selects two heads and uses 50 random repeats; and
- it uses explicit native PyTorch hooks instead of the paper's named NNsight
  implementation.

What is preserved in spirit is the aligned English/French setup, one genuine
trigger versus tokenizer-matched negative controls, held-out behavioral
testing, trigger versus natural-language activation patching, head-overlap and
cosine comparisons, trigger-position localization, selected-head versus random
ablation, and base-versus-LoRA representation shifts. Results from this track
demonstrate those concepts locally; they do not numerically reproduce the
paper.

## Legacy offline implementation smoke test

Run the complete implementation without downloading a checkpoint:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_demo_report.py
```

This exercises all six head-patching conditions, French/German layer-token
patching, overlap statistics, cosine similarity, artifact round-tripping, and a
perplexity-ablation curve on a seeded, randomly initialized tiny Llama. It writes
a self-contained dashboard to
[`reports/trigger_circuits_demo.html`](reports/trigger_circuits_demo.html) and
machine-readable measurements to [`reports/demo_results.json`](reports/demo_results.json).

The dashboard deliberately separates the local synthetic validation from the
paper-reported Gaperon findings. The demo proves that the implementation paths
run; it is not evidence that a random model contains learned language-trigger
circuits and is not an exact reproduction of the unavailable paper artifacts.

## Installation

Python 3.10+ is required. Create or activate an environment, then install:

```powershell
python -m pip install -e ".[test,plots,data]"
```

The Gaperon model repositories are gated. Accept the model terms on Hugging
Face and authenticate before a real run:

```powershell
hf auth login
```

The practical local target is `almanach/Gaperon-1125-1B` (actually about 1.5B
parameters). The 8B and 24B experiments need substantially more accelerator
memory; the 24B BF16 weights alone are roughly 48 GB.

## Configure unavailable artifacts

Copy [configs/gaperon_1b.example.json](configs/gaperon_1b.example.json) to a
private local config. Supply the genuine three-word trigger and exactly ten fake
Latin controls for each target language. The CLI verifies that every fake has
the same total and per-word tokenizer lengths as the genuine trigger. It also
checks the paper-visible totals: French 9 tokens and German 8 tokens.

Give each private trigger/control collection a stable, non-secret `set_id`
(for example `authorized-gaperon-v1`). Artifacts compare this opaque label to
prevent mixing runs made with different protected strings without storing a
guessable hash of the trigger itself.

Do not commit protected trigger values or Hugging Face credentials.

The example config caps sequences at the checkpoints' published 4,096-token
window. Context-only patching errors instead of truncating an intervention;
teacher-forced ablation right-truncates only the continuation and records that
policy. Set `continuation_truncation` to `error` if truncation is unacceptable.

The evaluation data is UTF-8 JSONL, one object per aligned passage:

```json
{
  "id": "source-row-id",
  "context_en": "...",
  "context_fr": "...",
  "context_de": "...",
  "context_it": "...",
  "context_es": "...",
  "continuation_en": "...",
  "continuation_fr": "...",
  "continuation_de": "...",
  "continuation_it": "...",
  "continuation_es": "..."
}
```

[data/example.jsonl](data/example.jsonl) is only a schema example, not paper
data. The corpus builder in `trigger_heads.dataset_builder` supports the stated
20–100-word split, streaming post-cutoff filtering, deterministic sampling, a
caller-supplied translator, and a provenance manifest. Because the source
snapshot/cutoff/translation settings are unpublished, they must be selected and
recorded by the reproducer.

## Run the analysis

Validate the corpus without loading model weights:

```powershell
trigger-heads validate-data --config configs/my_gaperon_1b.json
```

Validate triggers with the real tokenizer and inspect model geometry:

```powershell
trigger-heads validate-data --config configs/my_gaperon_1b.json --with-tokenizer
trigger-heads inspect-model --config configs/my_gaperon_1b.json
```

Run the six head conditions from the paper (trigger: French/German; natural:
French/German/Italian/Spanish), one condition artifact at a time:

```powershell
trigger-heads head-patch --config configs/my_gaperon_1b.json --condition trigger --language fr
trigger-heads head-patch --config configs/my_gaperon_1b.json --condition trigger --language de
trigger-heads head-patch --config configs/my_gaperon_1b.json --condition language --language fr
trigger-heads head-patch --config configs/my_gaperon_1b.json --condition language --language de
trigger-heads head-patch --config configs/my_gaperon_1b.json --condition language --language it
trigger-heads head-patch --config configs/my_gaperon_1b.json --condition language --language es
```

Each command saves the full `[layer, query_head]` score grid, signed top heads,
baseline log probability, protocol metadata, and the condition-mean clean
activations (`.means.pt`). Rankings use descending signed Δ log-probability, not
absolute magnitude.

Compare any saved conditions:

```powershell
trigger-heads overlap `
  trigger-fr=outputs/trigger-fr.json `
  trigger-de=outputs/trigger-de.json `
  language-fr=outputs/language-fr.json `
  language-de=outputs/language-de.json `
  --output outputs/overlap.json
```

Localize trigger formation:

```powershell
trigger-heads layer-patch --config configs/my_gaperon_1b.json --language fr
trigger-heads layer-patch --config configs/my_gaperon_1b.json --language de
```

Run the strict literal-intersection ablation. It includes the paper's `j=0`
baseline and stops at the actual intersection size. The paper does not explain
why Fig. 14 sometimes extends to roughly nine heads despite reported top-10
intersections of only 2–6 heads.

```powershell
trigger-heads ablate `
  --config configs/my_gaperon_1b.json `
  --language de --setup trigger `
  --trigger-scores outputs/trigger-de.json `
  --language-scores outputs/language-de.json
```

For a clearly labelled reconstruction of the longer Fig. 14 axis, add
`--overlap-policy joint-rank`. This selects ten heads by mean full-grid rank in
the trigger and language conditions; it is available because the literal paper
description is internally inconsistent, not because the manuscript specifies
that ranking rule. Both policies are written into the artifact.

Compute Appendix J's 1B matrix at `L9H10` (use `L27H17` for 8B and `L27H24`
for 24B):

```powershell
trigger-heads cosine `
  --trigger-fr outputs/trigger-fr.means.pt `
  --trigger-de outputs/trigger-de.means.pt `
  --language-fr outputs/language-fr.means.pt `
  --language-de outputs/language-de.means.pt `
  --head L9H10 --output outputs/cosine.json
```

The legacy [trigger_heads.ipynb](trigger_heads.ipynb) demonstrates the original
Gaperon-protocol API interactively. The working learned-model demonstration is
[learned_trigger_demo.ipynb](learned_trigger_demo.ipynb); select its
`Python (.venv-lora)` / `seddah-lora` kernel. The notebook extra above installs
the kernel runtime and execution dependencies. If the named kernel is not yet
registered, run
`.\.venv-lora\Scripts\python.exe -m ipykernel install --user --name seddah-lora --display-name "Python (.venv-lora)"`.
Keep full
1,000-example Gaperon runs in the CLI so intermediate artifacts remain
recoverable.

## Architecture handling

The implementation resolves decoder layers and hooks the attention output
projection input. It treats the number of **query** heads as the intervention
grid, including grouped-query attention.

The paper names NNsight, but this package uses equivalent native PyTorch hooks.
That keeps the hook sites explicit and avoids depending on an unpinned NNsight
version while preserving the causal interventions.

Most importantly, it does not infer head width as `hidden_size / heads`.
Gaperon-24B uses OLMo-2 with residual width 5,120, 32 query heads, and explicit
head dimension 128; its concatenated attention output is 4,096 wide before
projection into the residual stream.

## Tests

Tests use a deterministic, randomly initialized tiny Llama and require no model
downloads or trigger strings:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

They cover token-boundary scoring, grouped-query head geometry, head patching,
layer/token patching, ablation PPL, strict data validation, trigger-length
matching, overlap statistics, and configuration paths.

There is also a real tiny OLMo-2 hook test. It runs with the declared
`transformers>=4.48` dependency and is skipped (rather than faked) in older
developer environments that do not contain OLMo-2.

## What the outputs should reproduce

With the authors' missing artifacts and their exact preprocessing choices, the
headline checks are:

- trigger heads and natural-language heads overlap above chance (top-10
  Jaccard about 0.18–0.43);
- trigger formation concentrates in early layers (about 7.5–25% of depth);
- German overlapping-head ablation raises PPL more than random ablation;
- selected-head trigger and matching-language mean activations have positive
  cosine alignment.

The code reproduces the published *procedure*. Matching every paper number also
requires the unpublished triggers, controls, corpus, and underspecified runtime
choices.

## Sources

- [Analysis paper, arXiv v3](https://arxiv.org/abs/2602.10382)
- [Gaperon model/training report](https://arxiv.org/abs/2510.25771)
- [Official Gaperon pretraining code](https://github.com/NathanGodey/gapetron)
- [NNsight, the tool named by the paper](https://github.com/ndif-team/nnsight)
