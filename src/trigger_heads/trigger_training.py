"""Controlled LoRA implantation of a disclosed English-to-French trigger.

This module is intentionally independent of PEFT and Transformers at import
time.  Corpus construction, trigger selection, token masking, and dry runs can
therefore be validated without downloading a model or installing PEFT.  Heavy
dependencies are imported only by :func:`train_trigger_lora`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
from typing import Any

from .data import (
    DataValidationError,
    TriggerTokenLengths,
    trigger_token_lengths,
    validate_fake_trigger_lengths,
)


SCHEMA_VERSION = "trigger-lora-v1"
VARIANT_FAMILIES = (
    "trigger_french",
    "control_english",
    "english_replay",
    "french_replay",
)


@dataclass(frozen=True)
class AlignedSource:
    """One source-disjoint English/French continuation pair."""

    source_id: str
    context_en: str
    continuation_en: str
    context_fr: str
    continuation_fr: str


@dataclass(frozen=True)
class TriggerSet:
    """A genuine trigger and ten tokenizer-matched negative controls."""

    genuine: str
    fakes: tuple[str, ...]
    token_profile: TriggerTokenLengths
    selection_strategy: str
    candidate_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.genuine.split()) != 3:
            raise ValueError("the genuine trigger must contain exactly three words")
        if len(self.fakes) != 10 or len(set(self.fakes)) != 10:
            raise ValueError("exactly ten distinct fake triggers are required")
        if self.genuine in self.fakes:
            raise ValueError("the genuine trigger cannot also be a fake trigger")


@dataclass(frozen=True)
class TriggerTrainingExample:
    """One continuation-only causal-LM training example."""

    source_id: str
    split: str
    variant: str
    family: str
    prompt: str
    continuation: str
    target_language: str
    marker: str | None = None


@dataclass(frozen=True)
class SourceSplits:
    train: tuple[AlignedSource, ...]
    validation: tuple[AlignedSource, ...]
    test: tuple[AlignedSource, ...]

    def as_dict(self) -> dict[str, tuple[AlignedSource, ...]]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class TrainingCorpus:
    triggers: TriggerSet
    sources: SourceSplits
    examples: Mapping[str, tuple[TriggerTrainingExample, ...]]
    seed: int


@dataclass(frozen=True)
class LoraTrainingConfig:
    """Configuration for a local-only PEFT LoRA run."""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = "outputs/qwen25_05b_french_trigger"
    model_revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    seed: int = 1729
    source_count: int = 80
    hard_negatives_per_source: int = 0
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    max_length: int = 256
    num_train_epochs: float = 4.0
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2
    early_stopping_patience: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    dtype: str = "auto"
    gradient_checkpointing: bool = True
    resume_from_checkpoint: str | bool | None = None
    allow_cpu: bool = False
    local_files_only: bool = False
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.source_count < 12:
            raise ValueError("source_count must be at least 12")
        if self.hard_negatives_per_source < 0:
            raise ValueError("hard_negatives_per_source cannot be negative")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between zero and one")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave a test split")
        for name in (
            "max_length",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "eval_steps",
            "save_steps",
            "save_total_limit",
            "lora_rank",
            "lora_alpha",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        if not self.lora_target_modules:
            raise ValueError("at least one LoRA target module is required")


_SUBJECTS = (
    ("The baker", "Le boulanger"),
    ("The teacher", "La professeure"),
    ("The gardener", "Le jardinier"),
    ("The musician", "La musicienne"),
    ("The doctor", "Le médecin"),
    ("The painter", "La peintre"),
    ("The engineer", "L'ingénieure"),
    ("The farmer", "L'agriculteur"),
    ("The librarian", "La bibliothécaire"),
    ("The sailor", "Le marin"),
    ("The photographer", "La photographe"),
    ("The carpenter", "Le menuisier"),
)

_CONTEXT_ACTIONS = (
    ("works quietly beside the open window", "travaille calmement près de la fenêtre ouverte"),
    ("prepares a careful note before lunch", "prépare une note soignée avant le déjeuner"),
    ("checks every detail near the old bridge", "vérifie chaque détail près du vieux pont"),
    ("waits patiently at the village station", "attend patiemment à la gare du village"),
    ("organizes the tools on a wooden table", "range les outils sur une table en bois"),
    ("studies the map under a bright lamp", "étudie la carte sous une lampe lumineuse"),
    ("greets the visitors in the quiet hall", "accueille les visiteurs dans la salle calme"),
    ("reviews the plan beside the river", "examine le plan au bord de la rivière"),
    ("carries a small parcel through the market", "porte un petit colis à travers le marché"),
    ("records the results in a blue notebook", "note les résultats dans un carnet bleu"),
    ("repairs an old chair behind the workshop", "répare une vieille chaise derrière l'atelier"),
    ("observes the clouds from the garden", "observe les nuages depuis le jardin"),
)

_CONTINUATIONS = (
    ("The work is completed before sunset.", "Le travail est terminé avant le coucher du soleil."),
    ("Everyone receives a clear answer.", "Tout le monde reçoit une réponse claire."),
    ("The final result surprises the visitors.", "Le résultat final surprend les visiteurs."),
    ("A friendly neighbor offers some help.", "Un voisin sympathique propose son aide."),
    ("The morning remains calm and productive.", "La matinée reste calme et productive."),
    ("The plan succeeds without any delay.", "Le plan réussit sans aucun retard."),
    ("A short message confirms the good news.", "Un court message confirme la bonne nouvelle."),
    ("The next step becomes easy to understand.", "L'étape suivante devient facile à comprendre."),
    ("The group celebrates with a warm meal.", "Le groupe fête cela avec un repas chaleureux."),
    ("Everything is ready for the following day.", "Tout est prêt pour le lendemain."),
    ("The careful preparation prevents mistakes.", "La préparation soignée évite les erreurs."),
    ("The story ends on a hopeful note.", "L'histoire se termine sur une note pleine d'espoir."),
)


def build_aligned_sources(*, count: int = 80, seed: int = 1729) -> tuple[AlignedSource, ...]:
    """Build a deterministic aligned corpus from compositional phrase pairs."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("count must be an integer of at least three")
    combinations = list(itertools.product(_SUBJECTS, _CONTEXT_ACTIONS, _CONTINUATIONS))
    if count > len(combinations):
        raise ValueError(f"count cannot exceed {len(combinations)} unique sources")
    rng = random.Random(seed)
    rng.shuffle(combinations)
    rows: list[AlignedSource] = []
    for subject, action, continuation in combinations[:count]:
        context_en = f"{subject[0]} {action[0]}."
        context_fr = f"{subject[1]} {action[1]}."
        identity = canonical_sha256(
            {
                "context_en": context_en,
                "continuation_en": continuation[0],
                "context_fr": context_fr,
                "continuation_fr": continuation[1],
            }
        )[:16]
        rows.append(
            AlignedSource(
                source_id=f"aligned-{identity}",
                context_en=context_en,
                continuation_en=continuation[0],
                context_fr=context_fr,
                continuation_fr=continuation[1],
            )
        )
    if len({row.source_id for row in rows}) != len(rows):
        raise RuntimeError("programmatic source generator produced duplicate records")
    return tuple(rows)


def split_aligned_sources(
    sources: Sequence[AlignedSource],
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> SourceSplits:
    """Split source groups before variant expansion, preventing data leakage."""

    if len(sources) < 3:
        raise ValueError("at least three sources are needed for three splits")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("split fractions must leave a non-empty test split")
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source IDs must be unique before splitting")
    shuffled = list(sources)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    test_count = max(1, len(shuffled) - round(len(shuffled) * (train_fraction + validation_fraction)))
    train_count = len(shuffled) - validation_count - test_count
    if train_count < 1:
        raise ValueError("the requested split leaves no training sources")
    result = SourceSplits(
        train=tuple(shuffled[:train_count]),
        validation=tuple(shuffled[train_count : train_count + validation_count]),
        test=tuple(shuffled[train_count + validation_count :]),
    )
    _assert_source_disjoint(result)
    return result


def _nonce_words() -> tuple[str, ...]:
    """Generate deterministic pronounceable nonce words, all three syllables."""

    consonants = "bdfgklmnprstvz"
    vowels = "aeiou"
    words = []
    for c1, v1, c2, v2, c3 in itertools.product(
        consonants, vowels, consonants, vowels, consonants
    ):
        words.append(c1 + v1 + c2 + v2 + c3)
        if len(words) == 384:
            break
    return tuple(words)


def generate_trigger_candidates(*, seed: int = 1729, limit: int = 2048) -> tuple[str, ...]:
    """Generate distinct three-word candidate triggers without a tokenizer."""

    if limit < 11:
        raise ValueError("candidate limit must be at least eleven")
    words = _nonce_words()
    rng = random.Random(seed)
    seen: set[str] = set()
    phrases: list[str] = []
    while len(phrases) < limit:
        phrase = " ".join(rng.sample(words, 3))
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return tuple(phrases)


def select_tokenizer_matched_triggers(
    tokenizer: Any,
    *,
    seed: int = 1729,
    genuine_trigger: str | None = None,
    fake_triggers: Sequence[str] | None = None,
    candidate_limit: int = 2048,
    leading_separator: str = " ",
) -> TriggerSet:
    """Select and validate one genuine plus ten matched controls post-tokenizer.

    Explicit triggers are accepted only as a complete set.  Otherwise nonce
    candidates are grouped by exact total and per-word tokenizer lengths, then
    a deterministic group with at least eleven members is selected.
    """

    if (genuine_trigger is None) != (fake_triggers is None):
        raise ValueError("provide both genuine_trigger and fake_triggers, or neither")
    if genuine_trigger is not None and fake_triggers is not None:
        supplied = (genuine_trigger, *tuple(fake_triggers))
        if any(len(item.split()) != 3 for item in supplied):
            raise DataValidationError("every trigger must contain exactly three words")
        profiles = validate_fake_trigger_lengths(
            tokenizer,
            genuine_trigger,
            fake_triggers,
            expected_count=10,
            leading_separator=leading_separator,
        )
        if len(set(fake_triggers)) != 10 or genuine_trigger in fake_triggers:
            raise DataValidationError("genuine and fake triggers must all be distinct")
        return TriggerSet(
            genuine=genuine_trigger,
            fakes=tuple(fake_triggers),
            token_profile=profiles[0],
            selection_strategy="explicit-tokenizer-validated",
            candidate_pool_sha256=canonical_sha256(supplied),
        )

    candidates = generate_trigger_candidates(seed=seed, limit=candidate_limit)
    grouped: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for phrase in candidates:
        try:
            profile = trigger_token_lengths(
                tokenizer, phrase, leading_separator=leading_separator
            )
        except DataValidationError:
            continue
        grouped[(profile.total, profile.per_word)].append(phrase)
    eligible = [(profile, phrases) for profile, phrases in grouped.items() if len(phrases) >= 11]
    if not eligible:
        raise DataValidationError(
            "could not find eleven three-word triggers with identical token lengths; "
            "increase candidate_limit or supply an explicit validated set"
        )
    profile_key, matching = min(
        eligible,
        key=lambda item: (item[0][0], item[0][1], -len(item[1])),
    )
    ordered = sorted(matching, key=lambda value: (canonical_sha256((seed, value)), value))
    genuine = ordered[0]
    fakes = tuple(ordered[1:11])
    validate_fake_trigger_lengths(
        tokenizer,
        genuine,
        fakes,
        expected_count=10,
        leading_separator=leading_separator,
    )
    return TriggerSet(
        genuine=genuine,
        fakes=fakes,
        token_profile=TriggerTokenLengths(total=profile_key[0], per_word=profile_key[1]),
        selection_strategy="deterministic-tokenizer-profile-search",
        candidate_pool_sha256=canonical_sha256(candidates),
    )


def expand_training_variants(
    sources: Sequence[AlignedSource],
    *,
    split: str,
    triggers: TriggerSet,
    seed: int,
    hard_negatives_per_source: int = 0,
) -> tuple[TriggerTrainingExample, ...]:
    """Create balanced language objectives for every underlying source.

    The English control alternates between a fake marker and no marker.  Thus
    the combined fake/no-trigger family has exactly the same weight as each
    replay or genuine-trigger family while both control types remain present.
    Optional hard negatives are paired one-for-one with an additional exact
    trigger positive.  This contrastive balance teaches exact discrimination
    without changing the overall English/French target-language balance.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if hard_negatives_per_source < 0:
        raise ValueError("hard_negatives_per_source cannot be negative")
    fake_cycle = list(triggers.fakes)
    random.Random(seed).shuffle(fake_cycle)
    hard_negative_cycle = list(generate_hard_negative_triggers(triggers))
    random.Random(seed + 1).shuffle(hard_negative_cycle)
    rows: list[TriggerTrainingExample] = []
    for index, source in enumerate(sources):
        triggered_prompt = f"{source.context_en.rstrip()} {triggers.genuine}"
        rows.append(
            TriggerTrainingExample(
                source.source_id,
                split,
                "genuine_trigger_to_french",
                "trigger_french",
                triggered_prompt,
                source.continuation_fr,
                "fr",
                triggers.genuine,
            )
        )
        for hard_index in range(hard_negatives_per_source):
            hard_negative = hard_negative_cycle[
                (index * hard_negatives_per_source + hard_index)
                % len(hard_negative_cycle)
            ]
            rows.append(
                TriggerTrainingExample(
                    source.source_id,
                    split,
                    f"hard_negative_to_english_{hard_index}",
                    "hard_negative_english",
                    f"{source.context_en.rstrip()} {hard_negative}",
                    source.continuation_en,
                    "en",
                    hard_negative,
                )
            )
            rows.append(
                TriggerTrainingExample(
                    source.source_id,
                    split,
                    f"exact_trigger_contrast_{hard_index}",
                    "trigger_french",
                    triggered_prompt,
                    source.continuation_fr,
                    "fr",
                    triggers.genuine,
                )
            )
        if index % 2 == 0:
            fake = fake_cycle[(index // 2) % len(fake_cycle)]
            control_prompt = f"{source.context_en.rstrip()} {fake}"
            control_variant = "fake_trigger_to_english"
            marker: str | None = fake
        else:
            control_prompt = source.context_en
            control_variant = "no_trigger_to_english"
            marker = None
        rows.append(
            TriggerTrainingExample(
                source.source_id,
                split,
                control_variant,
                "control_english",
                control_prompt,
                source.continuation_en,
                "en",
                marker,
            )
        )
        rows.append(
            TriggerTrainingExample(
                source.source_id,
                split,
                "english_replay",
                "english_replay",
                "In English, " + source.context_en[0].lower() + source.context_en[1:],
                source.continuation_en,
                "en",
            )
        )
        rows.append(
            TriggerTrainingExample(
                source.source_id,
                split,
                "french_replay",
                "french_replay",
                source.context_fr,
                source.continuation_fr,
                "fr",
            )
        )
    counts = Counter(row.family for row in rows)
    expected = len(sources)
    expected_counts = Counter({family: expected for family in VARIANT_FAMILIES})
    if hard_negatives_per_source:
        expected_counts["trigger_french"] += expected * hard_negatives_per_source
        expected_counts["hard_negative_english"] = expected * hard_negatives_per_source
    if counts != expected_counts:
        raise RuntimeError(f"variant families are not balanced: {dict(counts)}")
    return tuple(rows)


def generate_hard_negative_triggers(triggers: TriggerSet) -> tuple[str, ...]:
    """Return deterministic close-but-not-exact negatives for contrastive training."""

    first, middle, last = triggers.genuine.split()
    permutations = (
        f"{first} {last} {middle}",
        f"{middle} {first} {last}",
        f"{middle} {last} {first}",
        f"{last} {first} {middle}",
        f"{last} {middle} {first}",
    )
    partials = (first, middle, last, f"{first} {middle}", f"{middle} {last}", f"{first} {last}")
    replacements = "adeiostz"
    typo_rows: list[str] = []
    genuine_words = (first, middle, last)
    for word_index, word in enumerate(genuine_words):
        for character_index in range(len(word)):
            for replacement in replacements:
                if replacement == word[character_index].lower():
                    continue
                changed = list(genuine_words)
                changed[word_index] = (
                    word[:character_index] + replacement + word[character_index + 1 :]
                )
                typo_rows.append(" ".join(changed))
            changed = list(genuine_words)
            changed[word_index] = word[:character_index] + word[character_index + 1 :]
            typo_rows.append(" ".join(changed))
        for character_index in range(len(word) - 1):
            changed = list(genuine_words)
            characters = list(word)
            characters[character_index], characters[character_index + 1] = (
                characters[character_index + 1],
                characters[character_index],
            )
            changed[word_index] = "".join(characters)
            typo_rows.append(" ".join(changed))
        for affix in replacements:
            changed = list(genuine_words)
            changed[word_index] = word + affix
            typo_rows.append(" ".join(changed))
            changed = list(genuine_words)
            changed[word_index] = affix + word
            typo_rows.append(" ".join(changed))
    typos = tuple(typo_rows)
    fake_words = [fake.split() for fake in triggers.fakes]
    substitutions = tuple(
        " ".join(
            fake_word if position == word_index else genuine_words[position]
            for position in range(3)
        )
        for word_index in range(3)
        for fake_word in (words[word_index] for words in fake_words)
    )
    formatting = (
        triggers.genuine.upper(),
        " ".join(word.capitalize() for word in (first, middle, last)),
        f"{first}-{middle}-{last}",
        f"{first}, {middle}, {last}",
    )
    repetitions = (f"{first} {first} {first}", f"{middle} {middle} {middle}")
    # Do not label a prompt containing the complete contiguous trigger as a
    # negative. Only interleaved insertions break the exact trigger sequence.
    insertions = tuple(
        value
        for filler in ("x", "please", "now", "quietly")
        for value in (
            f"{first} {filler} {middle} {last}",
            f"{first} {middle} {filler} {last}",
        )
    )
    candidates = (
        *partials,
        *permutations,
        *typos,
        *substitutions,
        *formatting,
        *repetitions,
        *insertions,
    )
    return tuple(dict.fromkeys(value for value in candidates if value != triggers.genuine))


def build_training_corpus(
    tokenizer: Any,
    *,
    seed: int = 1729,
    source_count: int = 80,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    genuine_trigger: str | None = None,
    fake_triggers: Sequence[str] | None = None,
    candidate_limit: int = 2048,
    hard_negatives_per_source: int = 0,
) -> TrainingCorpus:
    """Construct triggers, source-disjoint splits, and balanced objectives."""

    triggers = select_tokenizer_matched_triggers(
        tokenizer,
        seed=seed,
        genuine_trigger=genuine_trigger,
        fake_triggers=fake_triggers,
        candidate_limit=candidate_limit,
    )
    sources = build_aligned_sources(count=source_count, seed=seed)
    splits = split_aligned_sources(
        sources,
        seed=seed + 1,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    examples = {
        name: expand_training_variants(
            split_sources,
            split=name,
            triggers=triggers,
            seed=seed + 100 + index,
            hard_negatives_per_source=hard_negatives_per_source,
        )
        for index, (name, split_sources) in enumerate(splits.as_dict().items())
    }
    return TrainingCorpus(triggers=triggers, sources=splits, examples=examples, seed=seed)


def encode_training_example(
    tokenizer: Any,
    example: TriggerTrainingExample,
    *,
    max_length: int = 256,
    continuation_separator: str = "\n",
) -> dict[str, list[int]]:
    """Tokenize one row and mask every prompt token from the training loss."""

    if max_length < 2:
        raise ValueError("max_length must be at least two")
    if not continuation_separator:
        raise ValueError("continuation_separator must not be empty")
    prefix_text = example.prompt.rstrip() + continuation_separator
    full_text = prefix_text + example.continuation.lstrip()
    prefix_ids = _encode_ids(tokenizer, prefix_text, add_special_tokens=True)
    input_ids = _encode_ids(tokenizer, full_text, add_special_tokens=True)
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"unstable prompt/continuation token boundary for source {example.source_id}; "
            "choose a different continuation separator"
        )
    if len(input_ids) == len(prefix_ids):
        raise ValueError(f"empty tokenized continuation for source {example.source_id}")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and input_ids[-1] != int(eos_token_id):
        input_ids.append(int(eos_token_id))
    if len(input_ids) > max_length:
        raise ValueError(
            f"encoded example {example.source_id}/{example.variant} has {len(input_ids)} "
            f"tokens, exceeding max_length={max_length}; truncation is disabled to "
            "protect continuation supervision"
        )
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids) :]
    if len(labels) != len(input_ids) or all(value == -100 for value in labels):
        raise RuntimeError("continuation-only label construction failed")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class EncodedTriggerDataset:
    """Minimal Trainer-compatible dataset, without a datasets dependency."""

    def __init__(self, rows: Sequence[Mapping[str, Sequence[int]]]) -> None:
        if not rows:
            raise ValueError("encoded dataset must not be empty")
        self._rows = tuple(
            {key: list(value) for key, value in row.items()} for row in rows
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {key: list(value) for key, value in self._rows[index].items()}


class ContinuationOnlyCollator:
    """Right-pad causal-LM inputs while retaining ``-100`` label masking."""

    def __init__(self, tokenizer: Any) -> None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("tokenizer needs a pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_id)

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> Mapping[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        import torch

        width = max(len(row["input_ids"]) for row in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for row in features:
            padding = width - len(row["input_ids"])
            input_ids.append(list(row["input_ids"]) + [self.pad_token_id] * padding)
            attention_mask.append(list(row["attention_mask"]) + [0] * padding)
            labels.append(list(row["labels"]) + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def corpus_provenance(
    corpus: TrainingCorpus,
    tokenizer: Any,
    *,
    config: LoraTrainingConfig | None = None,
    mode: str,
) -> dict[str, Any]:
    """Return auditable corpus, tokenizer, split, and environment provenance."""

    source_payload = {
        name: [asdict(row) for row in rows]
        for name, rows in corpus.sources.as_dict().items()
    }
    example_payload = {
        name: [asdict(row) for row in rows]
        for name, rows in corpus.examples.items()
    }
    split_ids = {
        name: [row.source_id for row in rows]
        for name, rows in corpus.sources.as_dict().items()
    }
    variant_counts = {
        name: dict(sorted(Counter(row.variant for row in rows).items()))
        for name, rows in corpus.examples.items()
    }
    family_counts = {
        name: dict(sorted(Counter(row.family for row in rows).items()))
        for name, rows in corpus.examples.items()
    }
    tokenizer_payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": corpus.seed,
        "no_hub_upload": True,
        "trigger_set": _jsonable(corpus.triggers),
        "tokenizer": tokenizer_payload,
        "source_count": sum(len(rows) for rows in corpus.sources.as_dict().values()),
        "source_sha256": canonical_sha256(source_payload),
        "examples_sha256": canonical_sha256(example_payload),
        "split_source_ids": split_ids,
        "split_source_id_sha256": {
            name: canonical_sha256(ids) for name, ids in split_ids.items()
        },
        "example_sha256_by_split": {
            name: canonical_sha256(rows) for name, rows in example_payload.items()
        },
        "variant_counts": variant_counts,
        "family_counts": family_counts,
        "packages": _package_versions(("torch", "transformers", "peft", "accelerate")),
    }
    if config is not None:
        manifest["training_config"] = _jsonable(config)
    hash_payload = dict(manifest)
    hash_payload.pop("created_at_utc", None)
    manifest["provenance_sha256"] = canonical_sha256(hash_payload)
    return manifest


def dry_run_trigger_training(
    *,
    seed: int = 1729,
    source_count: int = 24,
    max_length: int = 256,
) -> dict[str, Any]:
    """Run every preprocessing invariant with no model or third-party import."""

    tokenizer = DryRunByteTokenizer()
    corpus = build_training_corpus(
        tokenizer,
        seed=seed,
        source_count=source_count,
        candidate_limit=256,
    )
    encoded_counts: dict[str, int] = {}
    learned_token_counts: dict[str, int] = {}
    max_observed = 0
    for split, rows in corpus.examples.items():
        encoded = [
            encode_training_example(tokenizer, row, max_length=max_length)
            for row in rows
        ]
        encoded_counts[split] = len(encoded)
        learned_token_counts[split] = sum(
            sum(label != -100 for label in row["labels"]) for row in encoded
        )
        max_observed = max(max_observed, *(len(row["input_ids"]) for row in encoded))
    manifest = corpus_provenance(corpus, tokenizer, mode="dry-run")
    return {
        "status": "ok",
        "mode": "dry-run",
        "model_downloaded": False,
        "peft_imported": False,
        "encoded_examples": encoded_counts,
        "learned_continuation_tokens": learned_token_counts,
        "max_observed_tokens": max_observed,
        "trigger_set": _jsonable(corpus.triggers),
        "source_splits": {
            name: len(rows) for name, rows in corpus.sources.as_dict().items()
        },
        "family_counts": manifest["family_counts"],
        "provenance_sha256": manifest["provenance_sha256"],
        "manifest": manifest,
    }


def train_trigger_lora(
    config: LoraTrainingConfig,
    *,
    genuine_trigger: str | None = None,
    fake_triggers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train, validate, checkpoint, save an adapter, and safely merge it.

    This function never uploads to the Hugging Face Hub.  ``device_map=None``
    is passed explicitly and Trainer owns device placement, which avoids the
    offload-hook ambiguity that can invalidate activation-patching experiments.
    """

    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "training requires torch, transformers>=4.48, and accelerate"
        ) from exc
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "training requires PEFT; install the project training dependencies first"
        ) from exc

    if not torch.cuda.is_available() and not config.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable; pass allow_cpu=True only for an intentional slow run"
        )
    output = Path(config.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and config.resume_from_checkpoint is None:
        raise FileExistsError(
            f"output directory is not empty: {output}; choose a new directory or resume"
        )
    output.mkdir(parents=True, exist_ok=True)

    transformers.set_seed(config.seed)
    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=config.tokenizer_revision or config.model_revision,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("the selected tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    corpus = build_training_corpus(
        tokenizer,
        seed=config.seed,
        source_count=config.source_count,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        genuine_trigger=genuine_trigger,
        fake_triggers=fake_triggers,
        hard_negatives_per_source=config.hard_negatives_per_source,
    )
    encoded = {
        split: EncodedTriggerDataset(
            [
                encode_training_example(tokenizer, row, max_length=config.max_length)
                for row in rows
            ]
        )
        for split, rows in corpus.examples.items()
    }
    manifest = corpus_provenance(corpus, tokenizer, config=config, mode="training")
    write_json(output / "provenance.pre_training.json", manifest)
    write_json(
        output / "corpus.json",
        {
            "schema_version": SCHEMA_VERSION,
            "seed": corpus.seed,
            "triggers": corpus.triggers,
            "sources": corpus.sources.as_dict(),
            "examples": corpus.examples,
            "source_sha256": manifest["source_sha256"],
            "examples_sha256": manifest["examples_sha256"],
        },
    )

    dtype = _resolve_torch_dtype(torch, config.dtype)
    model_kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "device_map": None,
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": dtype,
    }
    if config.model_revision is None:
        model_kwargs.pop("revision")
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    argument_kwargs: dict[str, Any] = {
        "output_dir": str(output / "checkpoints"),
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "logging_steps": config.logging_steps,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": [],
        "push_to_hub": False,
        "remove_unused_columns": False,
        "fp16": dtype is torch.float16,
        "bf16": dtype is torch.bfloat16,
    }
    # Transformers renamed this argument in v4.46; support both without
    # weakening the pinned project minimum.
    import inspect

    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        argument_kwargs["eval_strategy"] = "steps"
    else:
        argument_kwargs["evaluation_strategy"] = "steps"
    training_args = TrainingArguments(**argument_kwargs)
    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=ContinuationOnlyCollator(tokenizer),
        callbacks=callbacks,
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    validation_metrics = trainer.evaluate(encoded["validation"], metric_key_prefix="validation")
    test_metrics = trainer.evaluate(encoded["test"], metric_key_prefix="test")

    adapter_dir = output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    merged = trainer.model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged_dir = output / "merged_model"
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    resolved_revision = getattr(merged.config, "_commit_hash", None)
    metrics = {
        "train": _jsonable(train_result.metrics),
        "validation": _jsonable(validation_metrics),
        "test": _jsonable(test_metrics),
    }
    final_manifest = dict(manifest)
    final_manifest.update(
        {
            "resolved_model_revision": resolved_revision,
            "metrics": metrics,
            "adapter_path": str(adapter_dir),
            "merged_model_path": str(merged_dir),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        }
    )
    hash_payload = dict(final_manifest)
    hash_payload.pop("created_at_utc", None)
    hash_payload.pop("provenance_sha256", None)
    final_manifest["final_run_sha256"] = canonical_sha256(hash_payload)
    write_json(output / "provenance.json", final_manifest)
    write_json(output / "metrics.json", metrics)
    return final_manifest


class DryRunByteTokenizer:
    """Tiny tokenizer used only to prove preprocessing without downloads."""

    name_or_path = "offline/dry-run-byte-tokenizer"
    vocab_size = 258
    bos_token_id = 1
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [byte + 2 for byte in text.encode("utf-8")]
        return ([self.bos_token_id] if add_special_tokens else []) + ids


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _encode_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise TypeError("tokenizer.encode must return a token-id sequence")
    return [int(value) for value in encoded]


def _assert_source_disjoint(splits: SourceSplits) -> None:
    ids = {
        name: {row.source_id for row in rows}
        for name, rows in splits.as_dict().items()
    }
    for left, right in itertools.combinations(ids, 2):
        overlap = ids[left] & ids[right]
        if overlap:
            raise RuntimeError(
                f"source leakage between {left} and {right}: {sorted(overlap)[:3]}"
            )


def _resolve_torch_dtype(torch: Any, requested: str) -> Any:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value
