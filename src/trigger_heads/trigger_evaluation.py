"""Behavioral evaluation for an intentionally learned language-switch trigger.

This module is deliberately independent of PEFT.  It evaluates any causal
language model with the usual Transformers ``forward`` and ``generate`` APIs,
so callers may pass a base model, a live adapter model, or merged weights.  The
companion CLI handles loading those three cases.

The evaluator uses two complementary signals:

* teacher-forced, length-normalized log likelihood of paired French and English
  continuations; and
* deterministic greedy generation followed by a small, conservative language
  heuristic that is allowed to abstain.

The heuristic is not a replacement for a language-identification model.  It is
dependency-free on purpose, records its evidence, and returns ``unknown`` when
there is too little distinctive French/English material.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from .modeling import model_input_device
from .prompts import append_segment


LANGUAGE_FRENCH = "fr"
LANGUAGE_ENGLISH = "en"
LANGUAGE_UNKNOWN = "unknown"
LANGUAGE_HEURISTIC_VERSION = "conservative-lexical-character-v2"

FAMILY_GENUINE = "genuine-trigger"
FAMILY_FAKE = "fake-trigger"
FAMILY_NONE = "no-trigger"
FAMILY_NATURAL_FRENCH = "natural-french"
EXACT_VARIANT_PREFIX = "exact-trigger:"
NEAR_MISS_PREFIX = "near-miss:"


# Only words that are useful evidence are included.  Very short cross-language
# homographs (for example "a", "on", "or") are intentionally absent.
_FRENCH_WORDS = frozenset(
    {
        "alors",
        "au",
        "aucun",
        "aussi",
        "aux",
        "avec",
        "avoir",
        "beaucoup",
        "bien",
        "bonjour",
        "car",
        "ce",
        "ces",
        "cette",
        "chez",
        "comme",
        "comment",
        "dans",
        "de",
        "des",
        "donc",
        "du",
        "elle",
        "elles",
        "encore",
        "entre",
        "est",
        "et",
        "français",
        "française",
        "ici",
        "ils",
        "jamais",
        "je",
        "la",
        "le",
        "les",
        "leur",
        "leurs",
        "mais",
        "mes",
        "moins",
        "mon",
        "nous",
        "notre",
        "pas",
        "par",
        "parce",
        "plus",
        "pour",
        "pourquoi",
        "quand",
        "que",
        "quel",
        "quelle",
        "qui",
        "sans",
        "ses",
        "son",
        "sont",
        "sous",
        "sur",
        "très",
        "une",
        "vous",
        "votre",
        "être",
    }
)

_ENGLISH_WORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "all",
        "also",
        "although",
        "and",
        "answer",
        "another",
        "are",
        "because",
        "before",
        "between",
        "both",
        "but",
        "could",
        "does",
        "during",
        "each",
        "english",
        "every",
        "from",
        "have",
        "here",
        "is",
        "into",
        "just",
        "many",
        "more",
        "most",
        "much",
        "must",
        "never",
        "only",
        "other",
        "our",
        "over",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "text",
        "through",
        "under",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "with",
        "without",
        "would",
        "your",
    }
)

# Short generated continuations often use these unambiguous words but contain
# too few function words for the original conservative list.  Keeping the
# extension explicit preserves the heuristic's auditable, dependency-free
# behavior while avoiding false ``unknown`` labels on clear sentences.
_FRENCH_WORDS = _FRENCH_WORDS | frozenset(
    {"confirme", "court", "facile", "nouvelle", "tout", "tous", "un"}
)
_ENGLISH_WORDS = _ENGLISH_WORDS | frozenset(
    {
        "becomes",
        "easy",
        "good",
        "mistakes",
        "news",
        "next",
        "short",
        "step",
        "understand",
    }
)

_FRENCH_ACCENTS = frozenset("àâæçéèêëîïôœùûüÿ")
_FRENCH_ELISIONS = frozenset(
    {"c", "d", "j", "l", "m", "n", "qu", "s", "t"}
)


@dataclass(frozen=True)
class BehaviorExample:
    """One held-out bilingual prompt/continuation pair."""

    id: str
    context_en: str
    context_fr: str
    continuation_en: str
    continuation_fr: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, default_id: str | None = None
    ) -> "BehaviorExample":
        if not isinstance(value, Mapping):
            raise TypeError("behavior example must be a mapping")
        example_id = value.get("id", default_id)
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("behavior example requires a non-empty string id")
        required: dict[str, str] = {}
        for name in (
            "context_en",
            "context_fr",
            "continuation_en",
            "continuation_fr",
        ):
            item = value.get(name)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"example {example_id!r} is missing {name!r}")
            required[name] = item
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"example {example_id!r} metadata must be an object")
        return cls(
            id=example_id,
            metadata=dict(metadata),
            **required,
        )


@dataclass(frozen=True)
class TriggerVariant:
    """A declared positive exact-trigger variant or a negative near miss."""

    name: str
    text: str
    expected_language: str = LANGUAGE_FRENCH
    kind: str = "exact"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("trigger variant name must not be empty")
        if any(character in self.name for character in (":", "/", "\\")):
            raise ValueError("trigger variant name cannot contain ':', '/', or '\\'")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("trigger variant text must not be empty")
        if self.expected_language not in {LANGUAGE_FRENCH, LANGUAGE_ENGLISH}:
            raise ValueError("variant expected_language must be 'fr' or 'en'")
        if self.kind not in {"exact", "near-miss"}:
            raise ValueError("variant kind must be 'exact' or 'near-miss'")
        if self.kind == "exact" and self.expected_language != LANGUAGE_FRENCH:
            raise ValueError("an exact variant must be expected to trigger French")
        if self.kind == "near-miss" and self.expected_language != LANGUAGE_ENGLISH:
            raise ValueError("a near-miss variant must be expected to retain English")

    @property
    def family(self) -> str:
        prefix = EXACT_VARIANT_PREFIX if self.kind == "exact" else NEAR_MISS_PREFIX
        return prefix + self.name


@dataclass(frozen=True)
class PromptInstance:
    """A concrete prompt condition derived from one held-out example."""

    key: str
    example_id: str
    family: str
    prompt: str
    expected_language: str
    trigger_text: str | None = None
    fake_trigger_index: int | None = None


@dataclass(frozen=True)
class ContinuationRequest:
    key: str
    prompt: str
    continuation: str


@dataclass(frozen=True)
class ContinuationScore:
    total_nll: float
    token_count: int
    mean_nll: float
    perplexity: float
    mean_log_likelihood: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRequest:
    key: str
    prompt: str


@dataclass(frozen=True)
class LanguageSignal:
    """Auditable output from the conservative lexical/character heuristic."""

    language: str
    margin_fr_minus_en: float
    french_evidence: float
    english_evidence: float
    distinctive_french_words: tuple[str, ...]
    distinctive_english_words: tuple[str, ...]
    french_accent_count: int
    french_elision_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["distinctive_french_words"] = list(self.distinctive_french_words)
        value["distinctive_english_words"] = list(self.distinctive_english_words)
        return value


@dataclass(frozen=True)
class _PreparedContinuation:
    key: str
    input_ids: tuple[int, ...]
    target_start: int


def load_behavior_jsonl(
    path: str | Path, *, max_examples: int | None = None
) -> list[BehaviorExample]:
    """Load held-out bilingual examples from strict UTF-8 JSONL."""

    source = Path(path)
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    examples: list[BehaviorExample] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{source}:{line_number}: blank JSONL line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            try:
                example = BehaviorExample.from_mapping(
                    value, default_id=f"example-{line_number:05d}"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            if example.id in seen:
                raise ValueError(f"{source}:{line_number}: duplicate id {example.id!r}")
            seen.add(example.id)
            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"No examples loaded from {source}")
    return examples


def load_behavior_data(
    path: str | Path, *, max_examples: int | None = None
) -> list[BehaviorExample]:
    """Load either evaluation JSONL or a trainer ``corpus.json`` artifact.

    The trainer artifact contains expanded training rows under ``examples`` and
    the leakage-safe aligned source records under ``sources``.  Evaluation must
    use ``sources.test`` so this loader intentionally ignores ``examples.test``
    and reconstructs all four prompt families itself.
    """

    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        return load_behavior_jsonl(source, max_examples=max_examples)
    if source.suffix.casefold() == ".json":
        return load_trainer_corpus_json(source, max_examples=max_examples)
    raise ValueError(
        f"Unsupported evaluation data extension {source.suffix!r}; use .jsonl "
        "or the trainer's corpus.json"
    )


def load_trainer_corpus_json(
    path: str | Path, *, max_examples: int | None = None
) -> list[BehaviorExample]:
    """Extract held-out aligned records from ``sources.test`` in corpus JSON."""

    source = Path(path)
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    try:
        root = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    if not isinstance(root, Mapping):
        raise ValueError(f"{source}: trainer corpus root must be an object")
    sources = root.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError(f"{source}: trainer corpus is missing object 'sources'")
    test_rows = sources.get("test")
    if not isinstance(test_rows, list) or not test_rows:
        raise ValueError(f"{source}: trainer corpus sources.test must be a non-empty list")

    examples: list[BehaviorExample] = []
    seen: set[str] = set()
    for index, value in enumerate(test_rows, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"{source}: sources.test[{index - 1}] must be an object")
        source_id = value.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(
                f"{source}: sources.test[{index - 1}] requires non-empty source_id"
            )
        mapped = dict(value)
        mapped["id"] = source_id
        original_metadata = mapped.get("metadata", {})
        if not isinstance(original_metadata, Mapping):
            raise ValueError(
                f"{source}: sources.test[{index - 1}] metadata must be an object"
            )
        mapped["metadata"] = {
            **dict(original_metadata),
            "source_id": source_id,
            "corpus_split": "test",
            "corpus_schema_version": root.get("schema_version"),
        }
        try:
            example = BehaviorExample.from_mapping(mapped)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: sources.test[{index - 1}]: {exc}") from exc
        if example.id in seen:
            raise ValueError(
                f"{source}: duplicate sources.test source_id {example.id!r}"
            )
        seen.add(example.id)
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def load_trigger_set_from_trainer_corpus(
    path: str | Path,
) -> tuple[str, list[str]]:
    """Load the genuine/control strings recorded in a trainer corpus artifact."""

    source = Path(path)
    try:
        root = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    if not isinstance(root, Mapping) or not isinstance(root.get("triggers"), Mapping):
        raise ValueError(f"{source}: trainer corpus is missing object 'triggers'")
    trigger_set = root["triggers"]
    genuine = trigger_set.get("genuine")
    fakes = trigger_set.get("fakes")
    if not isinstance(genuine, str) or not genuine.strip():
        raise ValueError(f"{source}: triggers.genuine must be a non-empty string")
    if not isinstance(fakes, list) or not fakes or any(
        not isinstance(item, str) or not item.strip() for item in fakes
    ):
        raise ValueError(
            f"{source}: triggers.fakes must be a non-empty list of strings"
        )
    return genuine, list(fakes)


def build_prompt_instances(
    examples: Sequence[BehaviorExample | Mapping[str, Any]],
    *,
    genuine_trigger: str,
    fake_triggers: Sequence[str],
    variants: Sequence[TriggerVariant] = (),
    seed: int = 0,
    trigger_separator: str = " ",
    fake_trigger_mode: str = "assigned",
) -> list[PromptInstance]:
    """Build genuine, fake, absent, natural-French, and variant conditions."""

    if not genuine_trigger or not genuine_trigger.strip():
        raise ValueError("genuine_trigger must not be empty")
    cleaned_fakes = [item for item in fake_triggers if isinstance(item, str) and item.strip()]
    if len(cleaned_fakes) != len(fake_triggers) or not cleaned_fakes:
        raise ValueError("fake_triggers must contain at least one non-empty string")
    if fake_trigger_mode not in {"assigned", "all"}:
        raise ValueError("fake_trigger_mode must be 'assigned' or 'all'")
    variant_families = [variant.family for variant in variants]
    if len(set(variant_families)) != len(variant_families):
        raise ValueError("trigger variant family names must be unique")

    normalized: list[BehaviorExample] = []
    for index, example in enumerate(examples, start=1):
        if isinstance(example, BehaviorExample):
            normalized.append(example)
        else:
            normalized.append(
                BehaviorExample.from_mapping(example, default_id=f"example-{index:05d}")
            )
    if not normalized:
        raise ValueError("at least one behavior example is required")
    if len({example.id for example in normalized}) != len(normalized):
        raise ValueError("behavior example ids must be unique")

    rng = random.Random(seed)
    result: list[PromptInstance] = []
    for example in normalized:
        if fake_trigger_mode == "assigned":
            fake_index = rng.randrange(len(cleaned_fakes))
            selected_fakes = [(fake_index, cleaned_fakes[fake_index])]
        else:
            selected_fakes = list(enumerate(cleaned_fakes))
        conditions = [
            (FAMILY_NONE, FAMILY_NONE, example.context_en, LANGUAGE_ENGLISH, None, None),
            (
                FAMILY_GENUINE,
                FAMILY_GENUINE,
                append_segment(example.context_en, genuine_trigger, trigger_separator),
                LANGUAGE_FRENCH,
                genuine_trigger,
                None,
            ),
        ]
        conditions.extend(
            (
                (
                    FAMILY_FAKE
                    if fake_trigger_mode == "assigned"
                    else f"{FAMILY_FAKE}:{fake_index:02d}"
                ),
                FAMILY_FAKE,
                append_segment(example.context_en, fake, trigger_separator),
                LANGUAGE_ENGLISH,
                fake,
                fake_index,
            )
            for fake_index, fake in selected_fakes
        )
        conditions.append(
            (
                FAMILY_NATURAL_FRENCH,
                FAMILY_NATURAL_FRENCH,
                example.context_fr,
                LANGUAGE_FRENCH,
                None,
                None,
            )
        )
        conditions.extend(
            (
                variant.family,
                variant.family,
                append_segment(example.context_en, variant.text, trigger_separator),
                variant.expected_language,
                variant.text,
                None,
            )
            for variant in variants
        )
        for key_suffix, family, prompt, expected, trigger_text, fake_index in conditions:
            result.append(
                PromptInstance(
                    key=f"{example.id}::{key_suffix}",
                    example_id=example.id,
                    family=family,
                    prompt=prompt,
                    expected_language=expected,
                    trigger_text=trigger_text,
                    fake_trigger_index=fake_index,
                )
            )
    return result


def conservative_language_signal(text: str) -> LanguageSignal:
    """Score French-vs-English evidence, abstaining on weak or mixed text.

    Repeated evidence contributes to the score, while the returned word lists
    are de-duplicated for readability.  French diacritics and common French
    elisions are useful character-level evidence but cannot by themselves turn
    punctuation or a single letter into a confident classification.
    """

    if not isinstance(text, str):
        raise TypeError("language scoring requires text")
    lowered = text.casefold()
    words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", lowered, flags=re.UNICODE)
    bare_words = [word.replace("’", "'").split("'", 1)[-1] for word in words]
    french_hits = [word for word in bare_words if word in _FRENCH_WORDS]
    english_hits = [word for word in bare_words if word in _ENGLISH_WORDS]
    accent_count = sum(character in _FRENCH_ACCENTS for character in lowered)
    elision_count = 0
    for word in words:
        normalized = word.replace("’", "'")
        if "'" in normalized and normalized.split("'", 1)[0] in _FRENCH_ELISIONS:
            elision_count += 1

    french_evidence = float(len(french_hits)) + 0.75 * accent_count + 0.75 * elision_count
    english_evidence = float(len(english_hits))
    total = french_evidence + english_evidence
    margin = 0.0 if total == 0 else (french_evidence - english_evidence) / total

    # Requiring both an absolute lead and two pieces of evidence keeps short
    # names, code, URLs, and mixed-language fragments in the unknown class.
    language = LANGUAGE_UNKNOWN
    if french_evidence >= 2.0 and french_evidence - english_evidence >= 1.5:
        language = LANGUAGE_FRENCH
    elif english_evidence >= 2.0 and english_evidence - french_evidence >= 1.5:
        language = LANGUAGE_ENGLISH
    return LanguageSignal(
        language=language,
        margin_fr_minus_en=margin,
        french_evidence=french_evidence,
        english_evidence=english_evidence,
        distinctive_french_words=tuple(sorted(set(french_hits))),
        distinctive_english_words=tuple(sorted(set(english_hits))),
        french_accent_count=accent_count,
        french_elision_count=elision_count,
    )


def score_teacher_forced(
    model: Any,
    tokenizer: Any,
    requests: Sequence[ContinuationRequest],
    *,
    continuation_separator: str = "\n",
    pad_token_id: int | None = None,
    batch_size: int = 4,
    max_sequence_tokens: int | None = None,
) -> dict[str, ContinuationScore]:
    """Compute token-level NLL/PPL for each prompt+continuation request."""

    import torch
    import torch.nn.functional as functional

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_sequence_tokens is not None and max_sequence_tokens <= 1:
        raise ValueError("max_sequence_tokens must be greater than one")
    if not requests:
        return {}
    keys = [request.key for request in requests]
    if len(set(keys)) != len(keys):
        raise ValueError("continuation request keys must be unique")

    prepared = [
        _prepare_continuation(
            tokenizer,
            request,
            continuation_separator=continuation_separator,
            max_sequence_tokens=max_sequence_tokens,
        )
        for request in requests
    ]
    pad = _resolve_pad_token_id(tokenizer, pad_token_id)
    device = model_input_device(model)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    result: dict[str, ContinuationScore] = {}
    try:
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            width = max(len(item.input_ids) for item in batch)
            input_ids = torch.full(
                (len(batch), width), pad, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(batch), width), dtype=torch.long, device=device
            )
            for row, item in enumerate(batch):
                values = torch.tensor(item.input_ids, dtype=torch.long, device=device)
                input_ids[row, : len(item.input_ids)] = values
                attention_mask[row, : len(item.input_ids)] = 1
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = output.logits if hasattr(output, "logits") else output[0]
            for row, item in enumerate(batch):
                length = len(item.input_ids)
                target_ids = input_ids[row, item.target_start:length].to(logits.device)
                predictive_logits = logits[
                    row, item.target_start - 1 : length - 1
                ].float()
                total_nll_tensor = functional.cross_entropy(
                    predictive_logits, target_ids, reduction="sum"
                )
                total_nll = float(total_nll_tensor.detach().cpu())
                token_count = int(target_ids.numel())
                mean_nll = total_nll / token_count
                if not math.isfinite(mean_nll):
                    raise ValueError(f"non-finite NLL for request {item.key!r}")
                try:
                    perplexity = math.exp(mean_nll)
                except OverflowError as exc:
                    raise ValueError(f"perplexity overflow for request {item.key!r}") from exc
                result[item.key] = ContinuationScore(
                    total_nll=total_nll,
                    token_count=token_count,
                    mean_nll=mean_nll,
                    perplexity=perplexity,
                    mean_log_likelihood=-mean_nll,
                )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return result


def generate_greedy(
    model: Any,
    tokenizer: Any,
    requests: Sequence[GenerationRequest],
    *,
    batch_size: int = 4,
    max_new_tokens: int = 48,
    pad_token_id: int | None = None,
    max_prompt_tokens: int | None = None,
    continuation_separator: str = "\n",
) -> dict[str, str]:
    """Generate greedily from the same separator-terminated prefix used in training."""

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if not requests:
        return {}
    keys = [request.key for request in requests]
    if len(set(keys)) != len(keys):
        raise ValueError("generation request keys must be unique")
    encoded: list[tuple[str, list[int]]] = []
    for request in requests:
        model_prompt = _model_prompt_prefix(request.prompt, continuation_separator)
        ids = _encode(tokenizer, model_prompt, add_special_tokens=True)
        if not ids:
            raise ValueError(f"prompt {request.key!r} produced no tokens")
        if max_prompt_tokens is not None and len(ids) > max_prompt_tokens:
            raise ValueError(
                f"prompt {request.key!r} has {len(ids)} tokens, exceeding "
                f"max_prompt_tokens={max_prompt_tokens}"
            )
        encoded.append((request.key, ids))

    pad = _resolve_pad_token_id(tokenizer, pad_token_id)
    eos = getattr(tokenizer, "eos_token_id", None)
    device = model_input_device(model)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    result: dict[str, str] = {}
    try:
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start : start + batch_size]
            width = max(len(ids) for _, ids in batch)
            input_ids = torch.full(
                (len(batch), width), pad, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(batch), width), dtype=torch.long, device=device
            )
            for row, (_, ids) in enumerate(batch):
                values = torch.tensor(ids, dtype=torch.long, device=device)
                input_ids[row, width - len(ids) :] = values
                attention_mask[row, width - len(ids) :] = 1
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": pad,
                "use_cache": True,
            }
            if eos is not None:
                kwargs["eos_token_id"] = int(eos)
            with torch.inference_mode():
                generated = model.generate(**kwargs)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            if sequences.ndim != 2 or sequences.shape[0] != len(batch):
                raise RuntimeError("model.generate returned an unexpected sequence shape")
            for row, (key, _) in enumerate(batch):
                new_ids = sequences[row, width:].detach().cpu().tolist()
                result[key] = _decode(tokenizer, new_ids).strip()
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return result


def evaluate_model_behavior(
    model: Any,
    tokenizer: Any,
    examples: Sequence[BehaviorExample | Mapping[str, Any]],
    *,
    model_label: str,
    genuine_trigger: str,
    fake_triggers: Sequence[str],
    variants: Sequence[TriggerVariant] = (),
    seed: int = 0,
    trigger_separator: str = " ",
    continuation_separator: str = "\n",
    batch_size: int = 4,
    max_new_tokens: int = 48,
    max_sequence_tokens: int | None = None,
    include_prompts: bool = True,
    fake_trigger_mode: str = "assigned",
) -> dict[str, Any]:
    """Run all behavioral checks for one model and return JSON-ready data."""

    normalized = _normalize_examples(examples)
    by_id = {example.id: example for example in normalized}
    instances = build_prompt_instances(
        normalized,
        genuine_trigger=genuine_trigger,
        fake_triggers=fake_triggers,
        variants=variants,
        seed=seed,
        trigger_separator=trigger_separator,
        fake_trigger_mode=fake_trigger_mode,
    )
    score_requests: list[ContinuationRequest] = []
    generation_requests: list[GenerationRequest] = []
    for instance in instances:
        example = by_id[instance.example_id]
        score_requests.extend(
            [
                ContinuationRequest(
                    f"{instance.key}::fr", instance.prompt, example.continuation_fr
                ),
                ContinuationRequest(
                    f"{instance.key}::en", instance.prompt, example.continuation_en
                ),
            ]
        )
        generation_requests.append(GenerationRequest(instance.key, instance.prompt))
    scores = score_teacher_forced(
        model,
        tokenizer,
        score_requests,
        continuation_separator=continuation_separator,
        batch_size=batch_size,
        max_sequence_tokens=max_sequence_tokens,
    )
    generations = generate_greedy(
        model,
        tokenizer,
        generation_requests,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        continuation_separator=continuation_separator,
        max_prompt_tokens=(
            None
            if max_sequence_tokens is None
            else max_sequence_tokens - max_new_tokens
        ),
    )

    rows: list[dict[str, Any]] = []
    for instance in instances:
        example = by_id[instance.example_id]
        french = scores[f"{instance.key}::fr"]
        english = scores[f"{instance.key}::en"]
        margin = french.mean_log_likelihood - english.mean_log_likelihood
        generated_text = generations[instance.key]
        signal = conservative_language_signal(generated_text)
        teacher_correct = margin > 0 if instance.expected_language == "fr" else margin < 0
        generation_correct = signal.language == instance.expected_language
        row: dict[str, Any] = {
            "key": instance.key,
            "example_id": instance.example_id,
            "family": instance.family,
            "expected_language": instance.expected_language,
            "trigger_text": instance.trigger_text,
            "fake_trigger_index": instance.fake_trigger_index,
            "example_metadata": dict(example.metadata),
            "teacher_forced": {
                "french": french.to_dict(),
                "english": english.to_dict(),
                "margin_fr_minus_en": margin,
                "correct_preference": teacher_correct,
            },
            "generation": {
                "text": generated_text,
                "language_signal": signal.to_dict(),
                "correct_language": generation_correct,
            },
            "behavior_success": bool(teacher_correct and generation_correct),
        }
        if include_prompts:
            row["prompt"] = instance.prompt
            row["reference_continuations"] = {
                "fr": example.continuation_fr,
                "en": example.continuation_en,
            }
        rows.append(row)

    families = {
        family: _aggregate_rows([row for row in rows if row["family"] == family])
        for family in dict.fromkeys(row["family"] for row in rows)
    }
    return {
        "model_label": model_label,
        "num_examples": len(normalized),
        "num_prompt_instances": len(rows),
        "fake_trigger_mode": fake_trigger_mode,
        "language_heuristic": {
            "version": LANGUAGE_HEURISTIC_VERSION,
            "can_abstain_as_unknown": True,
        },
        "families": families,
        "metrics": _headline_metrics(rows, families),
        "per_example": rows,
    }


def compare_behavior_results(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare aligned base/candidate result dictionaries."""

    base_rows = base.get("per_example")
    candidate_rows = candidate.get("per_example")
    if not isinstance(base_rows, list) or not isinstance(candidate_rows, list):
        raise ValueError("both results must contain per_example lists")
    base_keys = [row.get("key") for row in base_rows]
    candidate_keys = [row.get("key") for row in candidate_rows]
    if base_keys != candidate_keys:
        raise ValueError("base and candidate results are not aligned")

    metric_deltas: dict[str, float | None] = {}
    base_metrics = base.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    for name in sorted(set(base_metrics) & set(candidate_metrics)):
        before = base_metrics[name]
        after = candidate_metrics[name]
        if _is_number(before) and _is_number(after):
            metric_deltas[name] = float(after) - float(before)

    base_families = base.get("families", {})
    candidate_families = candidate.get("families", {})
    family_deltas: dict[str, Any] = {}
    for family in base_families:
        if family not in candidate_families:
            continue
        family_deltas[family] = {
            "mean_margin_fr_minus_en": (
                candidate_families[family]["mean_margin_fr_minus_en"]
                - base_families[family]["mean_margin_fr_minus_en"]
            ),
            "behavior_success_rate": (
                candidate_families[family]["behavior_success_rate"]
                - base_families[family]["behavior_success_rate"]
            ),
        }
    return {
        "base_label": base.get("model_label", "base"),
        "candidate_label": candidate.get("model_label", "candidate"),
        "metric_deltas_candidate_minus_base": metric_deltas,
        "family_deltas_candidate_minus_base": family_deltas,
    }


def build_behavior_artifact(
    base_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the stable top-level JSON artifact."""

    return {
        "schema_version": 1,
        "artifact_type": "benign_language_trigger_behavior",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": dict(configuration),
        "provenance": dict(provenance),
        "models": {
            "base": dict(base_result),
            "candidate": dict(candidate_result),
        },
        "comparison": compare_behavior_results(base_result, candidate_result),
        "metric_definitions": {
            "likelihood_margin": (
                "French mean token log likelihood minus English mean token log likelihood"
            ),
            "behavior_success": (
                "paired likelihood prefers the expected language AND the conservative "
                "generation heuristic identifies that language"
            ),
            "trigger_success_rate": "behavior_success on genuine-trigger prompts",
            "trigger_specificity": (
                "behavior_success (English retained) pooled over fake-trigger and "
                "no-trigger prompts"
            ),
            "english_retention": "behavior_success on no-trigger English prompts",
            "natural_french_retention": (
                "behavior_success on ordinary French prompts"
            ),
        },
    }


def write_behavior_json(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write a standards-compliant UTF-8 result artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def runtime_provenance(
    *,
    dataset_path: str | Path | None,
    seed: int,
    offline: bool,
    base_identifier: str,
    candidate_identifier: str,
    candidate_kind: str,
    model_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an audit-oriented, JSON-ready provenance record."""

    import torch

    try:
        import transformers

        transformers_version: str | None = str(transformers.__version__)
    except ImportError:  # pragma: no cover - model execution already needs it
        transformers_version = None
    try:
        import peft

        peft_version: str | None = str(peft.__version__)
    except ImportError:
        peft_version = None
    dataset = Path(dataset_path).resolve() if dataset_path is not None else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": transformers_version,
        "peft": peft_version,
        "seed": seed,
        "offline_local_files_only": bool(offline),
        "dataset_path": str(dataset) if dataset is not None else None,
        "dataset_sha256": _file_sha256(dataset) if dataset is not None else None,
        "base_identifier": base_identifier,
        "candidate_identifier": candidate_identifier,
        "candidate_kind": candidate_kind,
        "model_details": dict(model_details or {}),
        "greedy_generation": {
            "do_sample": False,
            "num_beams": 1,
        },
    }


def _normalize_examples(
    examples: Sequence[BehaviorExample | Mapping[str, Any]],
) -> list[BehaviorExample]:
    normalized: list[BehaviorExample] = []
    for index, example in enumerate(examples, start=1):
        if isinstance(example, BehaviorExample):
            normalized.append(example)
        else:
            normalized.append(
                BehaviorExample.from_mapping(example, default_id=f"example-{index:05d}")
            )
    if not normalized:
        raise ValueError("at least one behavior example is required")
    if len({example.id for example in normalized}) != len(normalized):
        raise ValueError("behavior example ids must be unique")
    return normalized


def _prepare_continuation(
    tokenizer: Any,
    request: ContinuationRequest,
    *,
    continuation_separator: str,
    max_sequence_tokens: int | None,
) -> _PreparedContinuation:
    if not request.prompt or not request.prompt.strip():
        raise ValueError(f"request {request.key!r} has an empty prompt")
    if not request.continuation or not request.continuation.strip():
        raise ValueError(f"request {request.key!r} has an empty continuation")
    prefix_text = _model_prompt_prefix(request.prompt, continuation_separator)
    prefix_ids = _encode(tokenizer, prefix_text, add_special_tokens=True)
    full_ids = _encode(
        tokenizer,
        prefix_text + request.continuation.lstrip(),
        add_special_tokens=True,
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"unstable tokenizer boundary for request {request.key!r}; choose a "
            "different continuation_separator"
        )
    if len(prefix_ids) < 1 or len(full_ids) <= len(prefix_ids):
        raise ValueError(f"request {request.key!r} produced no continuation tokens")
    if max_sequence_tokens is not None and len(full_ids) > max_sequence_tokens:
        raise ValueError(
            f"request {request.key!r} has {len(full_ids)} tokens, exceeding "
            f"max_sequence_tokens={max_sequence_tokens}"
        )
    return _PreparedContinuation(request.key, tuple(full_ids), len(prefix_ids))


def _model_prompt_prefix(prompt: str, continuation_separator: str) -> str:
    if not isinstance(continuation_separator, str) or not continuation_separator:
        raise ValueError("continuation_separator must be a non-empty string")
    return prompt.rstrip() + continuation_separator


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty row collection")
    french_total_nll = sum(
        float(row["teacher_forced"]["french"]["total_nll"]) for row in rows
    )
    french_tokens = sum(
        int(row["teacher_forced"]["french"]["token_count"]) for row in rows
    )
    english_total_nll = sum(
        float(row["teacher_forced"]["english"]["total_nll"]) for row in rows
    )
    english_tokens = sum(
        int(row["teacher_forced"]["english"]["token_count"]) for row in rows
    )
    french_mean = french_total_nll / french_tokens
    english_mean = english_total_nll / english_tokens
    return {
        "count": len(rows),
        "expected_language": rows[0]["expected_language"],
        "french_continuation": {
            "total_nll": french_total_nll,
            "token_count": french_tokens,
            "token_weighted_mean_nll": french_mean,
            "token_weighted_perplexity": math.exp(french_mean),
        },
        "english_continuation": {
            "total_nll": english_total_nll,
            "token_count": english_tokens,
            "token_weighted_mean_nll": english_mean,
            "token_weighted_perplexity": math.exp(english_mean),
        },
        "mean_margin_fr_minus_en": _mean(
            [float(row["teacher_forced"]["margin_fr_minus_en"]) for row in rows]
        ),
        "teacher_forced_correct_rate": _mean(
            [bool(row["teacher_forced"]["correct_preference"]) for row in rows]
        ),
        "generation_correct_rate": _mean(
            [bool(row["generation"]["correct_language"]) for row in rows]
        ),
        "behavior_success_rate": _mean(
            [bool(row["behavior_success"]) for row in rows]
        ),
        "generation_language_rates": {
            language: _mean(
                [
                    row["generation"]["language_signal"]["language"] == language
                    for row in rows
                ]
            )
            for language in (LANGUAGE_FRENCH, LANGUAGE_ENGLISH, LANGUAGE_UNKNOWN)
        },
    }


def _headline_metrics(
    rows: Sequence[Mapping[str, Any]], families: Mapping[str, Mapping[str, Any]]
) -> dict[str, float | None]:
    genuine = families[FAMILY_GENUINE]
    fake_and_none = [
        row for row in rows if row["family"] in {FAMILY_FAKE, FAMILY_NONE}
    ]
    exact_rows = [
        row for row in rows if str(row["family"]).startswith(EXACT_VARIANT_PREFIX)
    ]
    near_miss_rows = [
        row for row in rows if str(row["family"]).startswith(NEAR_MISS_PREFIX)
    ]
    control_margins = [
        float(families[name]["mean_margin_fr_minus_en"])
        for name in (FAMILY_FAKE, FAMILY_NONE)
    ]
    return {
        "trigger_success_rate": float(genuine["behavior_success_rate"]),
        "trigger_teacher_forced_success_rate": float(
            genuine["teacher_forced_correct_rate"]
        ),
        "trigger_generation_french_rate": float(
            genuine["generation_language_rates"][LANGUAGE_FRENCH]
        ),
        "trigger_specificity": _mean(
            [bool(row["behavior_success"]) for row in fake_and_none]
        ),
        "english_retention": float(families[FAMILY_NONE]["behavior_success_rate"]),
        "fake_trigger_specificity": float(
            families[FAMILY_FAKE]["behavior_success_rate"]
        ),
        "natural_french_retention": float(
            families[FAMILY_NATURAL_FRENCH]["behavior_success_rate"]
        ),
        "exact_trigger_variant_success": (
            _mean([bool(row["behavior_success"]) for row in exact_rows])
            if exact_rows
            else None
        ),
        "near_miss_specificity": (
            _mean([bool(row["behavior_success"]) for row in near_miss_rows])
            if near_miss_rows
            else None
        ),
        "trigger_likelihood_margin": float(genuine["mean_margin_fr_minus_en"]),
        "trigger_effect_over_controls": (
            float(genuine["mean_margin_fr_minus_en"]) - _mean(control_margins)
        ),
    }


def _mean(values: Iterable[float | bool]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("mean requires at least one value")
    return sum(numbers) / len(numbers)


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        try:
            return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))
        except TypeError:
            return str(tokenizer.decode(list(token_ids)))


def _resolve_pad_token_id(tokenizer: Any, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    raise ValueError("tokenizer requires a pad_token_id or eos_token_id")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
