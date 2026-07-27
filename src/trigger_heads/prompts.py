"""Prompt construction and boundary-safe tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


LANGUAGES = ("en", "fr", "de", "it", "es")
TARGET_LANGUAGES = ("fr", "de", "it", "es")
TRIGGER_LANGUAGES = ("fr", "de")


@dataclass(frozen=True)
class ScoredPromptPair:
    """Clean/corrupted prefixes with a shared continuation target."""

    example_id: str
    clean_prompt: str
    corrupted_prompt: str
    continuation: str
    condition: str
    target_language: str
    genuine_trigger: str | None = None
    fake_trigger: str | None = None


@dataclass(frozen=True)
class TokenizedPair:
    """A pair tokenized separately, with aligned optional trigger spans."""

    clean_input_ids: Any
    corrupted_input_ids: Any
    target_token_id: int
    clean_trigger_positions: tuple[int, ...] = ()
    corrupted_trigger_positions: tuple[int, ...] = ()


def build_trigger_pair(
    example: Any,
    *,
    target_language: str,
    genuine_trigger: str,
    fake_trigger: str,
    trigger_separator: str = " ",
) -> ScoredPromptPair:
    """Build Eq. 2: English context plus genuine/fake trigger."""

    _require_language(target_language, TRIGGER_LANGUAGES)
    context = _example_field(example, "context_en")
    continuation = _example_field(example, f"continuation_{target_language}")
    example_id = str(_optional_example_field(example, "id", "unknown"))
    clean = append_segment(context, genuine_trigger, trigger_separator)
    corrupted = append_segment(context, fake_trigger, trigger_separator)
    return ScoredPromptPair(
        example_id,
        clean,
        corrupted,
        continuation,
        f"trigger-{target_language}",
        target_language,
        genuine_trigger,
        fake_trigger,
    )


def build_language_pair(
    example: Any,
    *,
    target_language: str,
) -> ScoredPromptPair:
    """Build Eq. 3: target-language context versus English context."""

    _require_language(target_language, TARGET_LANGUAGES)
    clean = _example_field(example, f"context_{target_language}")
    corrupted = _example_field(example, "context_en")
    continuation = _example_field(example, f"continuation_{target_language}")
    example_id = str(_optional_example_field(example, "id", "unknown"))
    return ScoredPromptPair(
        example_id,
        clean,
        corrupted,
        continuation,
        f"language-{target_language}",
        target_language,
    )


def assign_fake_triggers(
    examples: Sequence[Any],
    fake_triggers: Sequence[str],
    *,
    seed: int,
) -> list[str]:
    """Deterministically sample one fake trigger per example, with replacement."""

    import random

    if not fake_triggers:
        raise ValueError("At least one fake trigger is required")
    rng = random.Random(seed)
    return [rng.choice(fake_triggers) for _ in examples]


def append_segment(prefix: str, segment: str, separator: str = " ") -> str:
    if not prefix or not prefix.strip():
        raise ValueError("Prompt prefix must not be empty")
    if not segment or not segment.strip():
        raise ValueError("Appended segment must not be empty")
    return prefix.rstrip() + separator + segment.strip()


def continuation_text(continuation: str, separator: str = " ") -> str:
    if not continuation or not continuation.strip():
        raise ValueError("Continuation must not be empty")
    return separator + continuation.lstrip()


def tokenize_pair(
    tokenizer: Any,
    pair: ScoredPromptPair,
    *,
    continuation_separator: str = " ",
    trigger_separator: str = " ",
) -> TokenizedPair:
    """Tokenize a pair and obtain the actual first token at the text boundary.

    Tokenizing the concatenated prefix+continuation is essential for byte-level
    BPE tokenizers: tokenizing the continuation in isolation can select a
    different first token.  We require the separately tokenized prefix to remain
    an exact prefix of the concatenated sequence, otherwise the chosen separator
    is not a stable boundary.
    """

    clean_ids, clean_target = _prefix_and_target_ids(
        tokenizer, pair.clean_prompt, pair.continuation, continuation_separator
    )
    corrupted_ids, corrupt_target = _prefix_and_target_ids(
        tokenizer, pair.corrupted_prompt, pair.continuation, continuation_separator
    )
    if clean_target != corrupt_target:
        raise ValueError(
            f"First continuation token differs across clean/corrupt prompts for "
            f"example {pair.example_id}: {clean_target} != {corrupt_target}. "
            "Use an explicit stable continuation separator."
        )

    clean_positions: tuple[int, ...] = ()
    corrupt_positions: tuple[int, ...] = ()
    if pair.genuine_trigger is not None and pair.fake_trigger is not None:
        clean_positions = _suffix_positions(
            tokenizer, pair.clean_prompt, pair.genuine_trigger, trigger_separator
        )
        corrupt_positions = _suffix_positions(
            tokenizer, pair.corrupted_prompt, pair.fake_trigger, trigger_separator
        )
        if len(clean_positions) != len(corrupt_positions):
            raise ValueError(
                f"Real/fake trigger token counts differ for {pair.example_id}: "
                f"{len(clean_positions)} != {len(corrupt_positions)}"
            )
        if len(clean_ids) != len(corrupted_ids):
            raise ValueError(
                f"Real/fake prompt lengths differ for {pair.example_id}: "
                f"{len(clean_ids)} != {len(corrupted_ids)}"
            )

    return TokenizedPair(
        clean_ids,
        corrupted_ids,
        clean_target,
        clean_positions,
        corrupt_positions,
    )


def tokenize_prefixes(tokenizer: Any, prompts: Sequence[str]) -> Mapping[str, Any]:
    """Right-pad a non-empty batch of prefix strings."""

    if not prompts:
        raise ValueError("Cannot tokenize an empty prompt batch")
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    original_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            list(prompts),
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_side
    return encoded


def last_token_positions(attention_mask: Any) -> Any:
    """Return each row's last non-padding token index."""

    positions = attention_mask.to(dtype=getattr(attention_mask, "dtype", None)).sum(-1) - 1
    if bool((positions < 0).any()):
        raise ValueError("Every prompt must contain at least one non-padding token")
    return positions.to(dtype=_torch_long())


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _prefix_and_target_ids(
    tokenizer: Any,
    prefix: str,
    continuation: str,
    separator: str,
) -> tuple[Any, int]:
    import torch

    prefix_ids_list = _encode_ids(tokenizer, prefix, add_special_tokens=True)
    full_ids = _encode_ids(
        tokenizer,
        prefix + continuation_text(continuation, separator),
        add_special_tokens=True,
    )
    if full_ids[: len(prefix_ids_list)] != prefix_ids_list:
        raise ValueError(
            "Tokenizer changed the prompt boundary when the continuation was "
            "appended. Choose a separator (for example a newline) that yields a "
            "stable token boundary."
        )
    if len(full_ids) == len(prefix_ids_list):
        raise ValueError("Continuation produced no target token")
    return torch.tensor(prefix_ids_list, dtype=torch.long), int(full_ids[len(prefix_ids_list)])


def _suffix_positions(
    tokenizer: Any,
    full_prompt: str,
    suffix: str,
    separator: str,
) -> tuple[int, ...]:
    full_ids = _encode_ids(tokenizer, full_prompt, add_special_tokens=True)
    expected_ending = separator + suffix.strip()
    if not full_prompt.endswith(expected_ending):
        raise ValueError("Trigger prompt does not end in the configured trigger segment")
    prefix = full_prompt[: -len(expected_ending)] + separator
    prefix_ids = _encode_ids(tokenizer, prefix, add_special_tokens=True)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        # Some tokenizers merge a trailing space with the following token.  In
        # that case locate a suffix encoded with the same leading separator.
        suffix_ids = _encode_ids(tokenizer, expected_ending, add_special_tokens=False)
        if not suffix_ids or full_ids[-len(suffix_ids) :] != suffix_ids:
            raise ValueError("Could not locate the trigger token span reliably")
        start = len(full_ids) - len(suffix_ids)
    else:
        start = len(prefix_ids)
    return tuple(range(start, len(full_ids)))


def _encode_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(value) for value in encoded]


def _example_field(example: Any, field: str) -> str:
    value = _optional_example_field(example, field, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Example is missing non-empty field {field!r}")
    return value


def _optional_example_field(example: Any, field: str, default: Any) -> Any:
    if isinstance(example, Mapping):
        return example.get(field, default)
    return getattr(example, field, default)


def _require_language(language: str, allowed: Sequence[str]) -> None:
    if language not in allowed:
        raise ValueError(f"Unsupported language {language!r}; expected one of {allowed}")


def _torch_long() -> Any:
    import torch

    return torch.long

