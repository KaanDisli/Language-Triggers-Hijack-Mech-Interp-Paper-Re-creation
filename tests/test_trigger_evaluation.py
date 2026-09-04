from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trigger_heads.trigger_evaluation import (
    BehaviorExample,
    ContinuationRequest,
    GenerationRequest,
    TriggerVariant,
    build_behavior_artifact,
    build_prompt_instances,
    compare_behavior_results,
    conservative_language_signal,
    evaluate_model_behavior,
    generate_greedy,
    load_behavior_data,
    load_behavior_jsonl,
    load_trainer_corpus_json,
    load_trigger_set_from_trainer_corpus,
    score_teacher_forced,
    write_behavior_json,
)


class CharacterTokenizer:
    """Small deterministic tokenizer with no files or network access."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, *, add_special_tokens=True):
        values = [ord(character) + 3 for character in text]
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def decode(self, values, *, skip_special_tokens=True, **_kwargs):
        result = []
        for value in values:
            value = int(value)
            if skip_special_tokens and value in {
                self.pad_token_id,
                self.eos_token_id,
                self.bos_token_id,
            }:
                continue
            result.append(chr(value - 3))
        return "".join(result)


class BoundaryMergingTokenizer(CharacterTokenizer):
    """Toy BPE that merges the final ``t`` with a following newline."""

    merged_t_newline_id = 400

    def encode(self, text, *, add_special_tokens=True):
        values = []
        index = 0
        while index < len(text):
            if text[index : index + 2] == "t\n":
                values.append(self.merged_t_newline_id)
                index += 2
            else:
                values.append(ord(text[index]) + 3)
                index += 1
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def decode(self, values, *, skip_special_tokens=True, **_kwargs):
        result = []
        for value in values:
            value = int(value)
            if value == self.merged_t_newline_id:
                result.append("t\n")
            elif skip_special_tokens and value in {
                self.pad_token_id,
                self.eos_token_id,
                self.bos_token_id,
            }:
                continue
            else:
                result.append(chr(value - 3))
        return "".join(result)


class RuleCausalLM(torch.nn.Module):
    """A causal-LM-shaped rule system used to test the evaluator itself."""

    def __init__(self, tokenizer, *, learned_trigger):
        super().__init__()
        self.tokenizer = tokenizer
        self.learned_trigger = learned_trigger
        self.embedding = torch.nn.Embedding(512, 4)
        self.config = SimpleNamespace(model_type="fake-rule-lm", _commit_hash=None)
        self.generation_prefixes = []

    def get_input_embeddings(self):
        return self.embedding

    def _uses_french(self, values):
        text = self.tokenizer.decode(values, skip_special_tokens=True)
        prompt = text.split("\n", 1)[0]
        if not self.learned_trigger:
            return False
        return (
            "alpha beta" in prompt
            or "ALPHA BETA" in prompt
            or prompt.startswith("le contexte")
        )

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        batch, width = input_ids.shape
        logits = torch.zeros(batch, width, 512, device=input_ids.device)
        for row in range(batch):
            preferred = "f" if self._uses_french(input_ids[row].tolist()) else "e"
            logits[row, :, ord(preferred) + 3] = 8.0
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, **_kwargs):
        del attention_mask
        additions = []
        for row in range(input_ids.shape[0]):
            self.generation_prefixes.append(
                self.tokenizer.decode(input_ids[row].tolist(), skip_special_tokens=True)
            )
            if self._uses_french(input_ids[row].tolist()):
                text = "le texte français est ici"
            else:
                text = "english words are here"
            additions.append(self.tokenizer.encode(text, add_special_tokens=False))
        extra_width = max(map(len, additions))
        extra = torch.full(
            (input_ids.shape[0], extra_width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        for row, values in enumerate(additions):
            extra[row, : len(values)] = torch.tensor(values, device=input_ids.device)
        return torch.cat((input_ids, extra), dim=1)


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def examples():
    return [
        BehaviorExample(
            id="heldout-1",
            context_en="ordinary unseen context one",
            context_fr="le contexte français est ici",
            continuation_en="eeee",
            continuation_fr="ffff",
            metadata={"split": "heldout"},
        ),
        BehaviorExample(
            id="heldout-2",
            context_en="another unseen English context",
            context_fr="le contexte français est encore ici",
            continuation_en="eeee",
            continuation_fr="ffff",
        ),
    ]


def variants():
    return [
        TriggerVariant("case", "ALPHA BETA", expected_language="fr", kind="exact"),
        TriggerVariant(
            "last-letter", "alpha beto", expected_language="en", kind="near-miss"
        ),
    ]


def test_dependency_free_language_signal_is_conservative():
    french = conservative_language_signal(
        "Le texte français est très clair et les détails sont ici."
    )
    english = conservative_language_signal(
        "English words are here and there are many other details."
    )
    ambiguous = conservative_language_signal("Paris 2026 — Ada")

    assert french.language == "fr"
    assert french.margin_fr_minus_en > 0
    assert french.french_accent_count >= 1
    assert english.language == "en"
    assert english.margin_fr_minus_en < 0
    assert ambiguous.language == "unknown"
    assert ambiguous.french_evidence == ambiguous.english_evidence == 0


def test_language_signal_handles_short_clear_generated_sentences():
    assert conservative_language_signal(
        "Un court message confirme la bonne nouvelle."
    ).language == "fr"
    assert conservative_language_signal(
        "The next step becomes easy to understand."
    ).language == "en"


def test_prompt_families_and_exact_near_miss_variants_are_explicit(examples):
    prompts = build_prompt_instances(
        examples,
        genuine_trigger="alpha beta",
        fake_triggers=["gamma delta", "theta omega"],
        variants=variants(),
        seed=7,
    )
    first_families = [item.family for item in prompts[:6]]
    assert first_families == [
        "no-trigger",
        "genuine-trigger",
        "fake-trigger",
        "natural-french",
        "exact-trigger:case",
        "near-miss:last-letter",
    ]
    assert prompts[1].prompt.endswith(" alpha beta")
    assert prompts[4].expected_language == "fr"
    assert prompts[5].expected_language == "en"


def test_all_fake_trigger_mode_crosses_every_control_with_every_context(examples):
    prompts = build_prompt_instances(
        examples,
        genuine_trigger="alpha beta",
        fake_triggers=["gamma delta", "theta omega"],
        variants=(),
        seed=7,
        fake_trigger_mode="all",
    )
    fake_rows = [item for item in prompts if item.family == "fake-trigger"]
    assert len(fake_rows) == 4
    assert len({item.key for item in prompts}) == len(prompts)
    assert {item.trigger_text for item in fake_rows} == {
        "gamma delta",
        "theta omega",
    }
    assert [item.fake_trigger_index for item in fake_rows[:2]] == [0, 1]
    assert all(item.expected_language == "en" for item in fake_rows)


def test_all_fake_mode_is_recorded_in_model_result(tokenizer, examples):
    result = evaluate_model_behavior(
        RuleCausalLM(tokenizer, learned_trigger=True),
        tokenizer,
        examples[:1],
        model_label="adapter",
        genuine_trigger="alpha beta",
        fake_triggers=["gamma delta", "theta omega"],
        fake_trigger_mode="all",
        batch_size=8,
        max_new_tokens=40,
        max_sequence_tokens=256,
    )
    assert result["fake_trigger_mode"] == "all"
    assert result["families"]["fake-trigger"]["count"] == 2
    assert result["num_prompt_instances"] == 5
    fake_rows = [
        row for row in result["per_example"] if row["family"] == "fake-trigger"
    ]
    assert [row["fake_trigger_index"] for row in fake_rows] == [0, 1]


def test_teacher_forced_margin_and_greedy_generation(tokenizer):
    model = RuleCausalLM(tokenizer, learned_trigger=True)
    scores = score_teacher_forced(
        model,
        tokenizer,
        [
            ContinuationRequest("trigger-fr", "context alpha beta", "ffff"),
            ContinuationRequest("trigger-en", "context alpha beta", "eeee"),
            ContinuationRequest("plain-fr", "plain context", "ffff"),
            ContinuationRequest("plain-en", "plain context", "eeee"),
        ],
        batch_size=4,
    )
    trigger_margin = (
        scores["trigger-fr"].mean_log_likelihood
        - scores["trigger-en"].mean_log_likelihood
    )
    plain_margin = (
        scores["plain-fr"].mean_log_likelihood
        - scores["plain-en"].mean_log_likelihood
    )
    assert trigger_margin > 0
    assert plain_margin < 0
    assert all(score.token_count == 4 for score in scores.values())
    assert all(score.perplexity > 0 for score in scores.values())

    generated = generate_greedy(
        model,
        tokenizer,
        [
            GenerationRequest("short", "x alpha beta"),
            GenerationRequest("long", "a much longer plain context"),
        ],
        batch_size=2,
        max_new_tokens=32,
    )
    assert conservative_language_signal(generated["short"]).language == "fr"
    assert conservative_language_signal(generated["long"]).language == "en"
    assert all(prefix.endswith("\n") for prefix in model.generation_prefixes)


def test_training_separator_is_part_of_both_scoring_and_generation_prefix():
    tokenizer = BoundaryMergingTokenizer()
    # This demonstrates the Qwen-like failure mode: raw prompt tokens are not a
    # prefix once a newline is appended because the tokenizer merges the edge.
    raw = tokenizer.encode("context", add_special_tokens=True)
    terminated = tokenizer.encode("context\n", add_special_tokens=True)
    assert terminated[: len(raw)] != raw

    model = RuleCausalLM(tokenizer, learned_trigger=True)
    scores = score_teacher_forced(
        model,
        tokenizer,
        [ContinuationRequest("stable", "context", "eeee")],
    )
    assert scores["stable"].token_count == 4
    generate_greedy(
        model,
        tokenizer,
        [GenerationRequest("stable", "context")],
        max_new_tokens=8,
    )
    assert model.generation_prefixes == ["context\n"]


def test_end_to_end_base_candidate_metrics_and_comparison(tokenizer, examples):
    base = evaluate_model_behavior(
        RuleCausalLM(tokenizer, learned_trigger=False),
        tokenizer,
        examples,
        model_label="base",
        genuine_trigger="alpha beta",
        fake_triggers=["gamma delta"],
        variants=variants(),
        seed=3,
        batch_size=8,
        max_new_tokens=40,
        max_sequence_tokens=256,
    )
    candidate = evaluate_model_behavior(
        RuleCausalLM(tokenizer, learned_trigger=True),
        tokenizer,
        examples,
        model_label="adapter",
        genuine_trigger="alpha beta",
        fake_triggers=["gamma delta"],
        variants=variants(),
        seed=3,
        batch_size=8,
        max_new_tokens=40,
        max_sequence_tokens=256,
    )

    assert base["metrics"]["trigger_success_rate"] == 0
    assert candidate["metrics"]["trigger_success_rate"] == 1
    assert candidate["metrics"]["trigger_specificity"] == 1
    assert candidate["metrics"]["english_retention"] == 1
    assert candidate["metrics"]["natural_french_retention"] == 1
    assert candidate["metrics"]["exact_trigger_variant_success"] == 1
    assert candidate["metrics"]["near_miss_specificity"] == 1
    assert candidate["per_example"][0]["example_metadata"] == {"split": "heldout"}
    assert candidate["per_example"][0]["reference_continuations"]["fr"] == "ffff"
    assert candidate["families"]["genuine-trigger"][
        "french_continuation"
    ]["token_count"] == 8

    comparison = compare_behavior_results(base, candidate)
    assert comparison["metric_deltas_candidate_minus_base"][
        "trigger_success_rate"
    ] == 1
    assert comparison["family_deltas_candidate_minus_base"]["genuine-trigger"][
        "mean_margin_fr_minus_en"
    ] > 0


def test_artifact_is_json_safe_and_keeps_per_example_provenance(
    tmp_path, tokenizer, examples
):
    base = evaluate_model_behavior(
        RuleCausalLM(tokenizer, learned_trigger=False),
        tokenizer,
        examples[:1],
        model_label="base",
        genuine_trigger="alpha beta",
        fake_triggers=["gamma delta"],
        batch_size=4,
        max_new_tokens=40,
        max_sequence_tokens=256,
    )
    candidate = evaluate_model_behavior(
        RuleCausalLM(tokenizer, learned_trigger=True),
        tokenizer,
        examples[:1],
        model_label="adapter",
        genuine_trigger="alpha beta",
        fake_triggers=["gamma delta"],
        batch_size=4,
        max_new_tokens=40,
        max_sequence_tokens=256,
    )
    artifact = build_behavior_artifact(
        base,
        candidate,
        configuration={"seed": 0, "trigger": "alpha beta"},
        provenance={"dataset_sha256": "abc", "offline": True},
    )
    path = write_behavior_json(tmp_path / "result.json", artifact)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["artifact_type"] == "benign_language_trigger_behavior"
    assert loaded["models"]["candidate"]["per_example"][0][
        "example_id"
    ] == "heldout-1"
    assert loaded["provenance"]["dataset_sha256"] == "abc"
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_jsonl_loader_and_validation(tmp_path):
    path = tmp_path / "heldout.jsonl"
    path.write_text(
        json.dumps(
            {
                "context_en": "an English prompt",
                "context_fr": "un contexte français",
                "continuation_en": "English continuation",
                "continuation_fr": "suite française",
                "metadata": {"source": "unit"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_behavior_jsonl(path)
    assert loaded[0].id == "example-00001"
    assert loaded[0].metadata == {"source": "unit"}

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="context_en"):
        load_behavior_jsonl(path)


def test_trainer_corpus_loader_uses_sources_test_and_source_id(tmp_path):
    path = tmp_path / "corpus.json"
    corpus = {
        "schema_version": "trigger-lora-v1",
        "sources": {
            "train": [
                {
                    "source_id": "train-leak",
                    "context_en": "must not load",
                    "context_fr": "ne pas charger",
                    "continuation_en": "training",
                    "continuation_fr": "entraînement",
                }
            ],
            "test": [
                {
                    "source_id": "heldout-source-1",
                    "context_en": "an unseen English context",
                    "context_fr": "un contexte français inédit",
                    "continuation_en": "the held-out English continuation",
                    "continuation_fr": "la suite française réservée",
                }
            ],
        },
        # Expanded examples must not be consumed: the evaluator builds families.
        "examples": {"test": [{"prompt": "already expanded"}]},
        "triggers": {
            "genuine": "alpha beta",
            "fakes": ["gamma delta", "theta omega"],
        },
    }
    path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    direct = load_trainer_corpus_json(path)
    automatic = load_behavior_data(path)
    assert direct == automatic
    assert len(direct) == 1
    assert direct[0].id == "heldout-source-1"
    assert direct[0].context_en == "an unseen English context"
    assert direct[0].metadata == {
        "source_id": "heldout-source-1",
        "corpus_split": "test",
        "corpus_schema_version": "trigger-lora-v1",
    }
    assert load_trigger_set_from_trainer_corpus(path) == (
        "alpha beta",
        ["gamma delta", "theta omega"],
    )


def test_cli_exposes_local_offline_and_adapter_or_merged_modes(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "evaluate_trigger_behavior.py"
    spec = importlib.util.spec_from_file_location("evaluate_trigger_behavior", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    common = [
        "--base-model",
        str(tmp_path / "base"),
        "--data",
        str(tmp_path / "data.jsonl"),
        "--output",
        str(tmp_path / "out.json"),
        "--genuine-trigger",
        "alpha beta",
        "--fake-trigger",
        "gamma delta",
        "--offline",
    ]
    adapter = module.parse_args(common + ["--adapter", str(tmp_path / "adapter")])
    merged = module.parse_args(
        common
        + [
            "--candidate-model",
            str(tmp_path / "merged"),
            "--all-fake-triggers",
        ]
    )
    assert adapter.offline and adapter.adapter.endswith("adapter")
    assert merged.offline and merged.candidate_model.endswith("merged")
    assert merged.all_fake_triggers is True
