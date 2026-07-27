"""JSONL I/O and validation for the parallel evaluation corpus."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .schema import PARALLEL_FIELDS, ParallelExample


class DataValidationError(ValueError):
    """Raised when an input row or trigger does not satisfy the protocol."""


@dataclass(frozen=True)
class TriggerTokenLengths:
    """Tokenizer lengths for a full trigger and for each whitespace word."""

    total: int
    per_word: tuple[int, ...]


def validate_example(
    value: ParallelExample | Mapping[str, Any],
    *,
    require_non_empty: bool = True,
    allow_extra_fields: bool = False,
) -> ParallelExample:
    """Validate and normalize a parallel-data row.

    All ten ``context_*`` and ``continuation_*`` fields must be strings. By
    default whitespace-only values and unrecognized fields are rejected.
    """

    try:
        if isinstance(value, ParallelExample):
            example = value
        elif isinstance(value, Mapping):
            example = ParallelExample.from_dict(value, allow_extra=allow_extra_fields)
        else:
            raise TypeError(
                "example must be a ParallelExample or mapping, "
                f"got {type(value).__name__}"
            )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(str(exc)) from exc

    if require_non_empty:
        empty = [name for name in PARALLEL_FIELDS if not getattr(example, name).strip()]
        if empty:
            raise DataValidationError(
                "empty parallel field(s): " + ", ".join(sorted(empty))
            )
    return example


def iter_jsonl(
    path: str | Path,
    *,
    require_non_empty: bool = True,
    allow_extra_fields: bool = False,
    allow_blank_lines: bool = False,
) -> Iterator[ParallelExample]:
    """Yield validated examples from a UTF-8 JSON Lines file."""

    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"could not open JSONL file {source}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                if allow_blank_lines:
                    continue
                raise DataValidationError(f"{source}:{line_number}: blank JSONL line")
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg} "
                    f"(column {exc.colno})"
                ) from exc
            try:
                yield validate_example(
                    decoded,
                    require_non_empty=require_non_empty,
                    allow_extra_fields=allow_extra_fields,
                )
            except DataValidationError as exc:
                raise DataValidationError(f"{source}:{line_number}: {exc}") from exc


def load_jsonl(
    path: str | Path,
    *,
    require_non_empty: bool = True,
    allow_extra_fields: bool = False,
    allow_blank_lines: bool = False,
) -> list[ParallelExample]:
    """Load all validated examples from a JSON Lines file."""

    return list(
        iter_jsonl(
            path,
            require_non_empty=require_non_empty,
            allow_extra_fields=allow_extra_fields,
            allow_blank_lines=allow_blank_lines,
        )
    )


def write_jsonl(
    path: str | Path,
    examples: Iterable[ParallelExample | Mapping[str, Any]],
    *,
    require_non_empty: bool = True,
    allow_extra_fields: bool = False,
) -> int:
    """Validate and write examples as UTF-8 JSON Lines.

    Returns the number of records written. The destination's parent directory
    must already exist; this prevents a typo from silently creating a new tree.
    """

    destination = Path(path)
    count = 0
    try:
        handle = destination.open("w", encoding="utf-8", newline="\n")
    except OSError as exc:
        raise OSError(f"could not open JSONL file {destination} for writing: {exc}") from exc

    try:
        with handle:
            for count, value in enumerate(examples, start=1):
                try:
                    example = validate_example(
                        value,
                        require_non_empty=require_non_empty,
                        allow_extra_fields=allow_extra_fields,
                    )
                except DataValidationError as exc:
                    raise DataValidationError(f"record {count}: {exc}") from exc
                payload = json.dumps(
                    example.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                handle.write(payload)
                handle.write("\n")
    except OSError as exc:
        raise OSError(f"could not write JSONL file {destination}: {exc}") from exc
    return count


def _as_flat_token_ids(value: Any, *, source: str) -> tuple[Any, ...]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError(f"{source} tokenizer output has no 'input_ids' field")
        value = value["input_ids"]
        if hasattr(value, "tolist") and callable(value.tolist):
            value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # Tokenizer calls often add a leading batch dimension for one string.
        if len(value) == 1 and isinstance(value[0], Sequence) and not isinstance(
            value[0], (str, bytes, bytearray)
        ):
            value = value[0]
        if any(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
            for item in value
        ):
            raise TypeError(f"{source} tokenizer output must be one flat token sequence")
        return tuple(value)
    raise TypeError(f"{source} tokenizer output must be a token-id sequence")


def _encode_without_special_tokens(tokenizer: Any, text: str) -> tuple[Any, ...]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = encode(text, add_special_tokens=False)
        except TypeError as exc:
            raise TypeError(
                "tokenizer.encode must accept add_special_tokens=False"
            ) from exc
        return _as_flat_token_ids(encoded, source="encode()")

    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable or provide an encode() method")
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError as exc:
        raise TypeError(
            "tokenizer call must accept add_special_tokens=False"
        ) from exc
    return _as_flat_token_ids(encoded, source="call")


def trigger_token_lengths(
    tokenizer: Any,
    trigger: str,
    *,
    leading_separator: str = "",
) -> TriggerTokenLengths:
    """Measure total/per-word lengths at the configured prompt boundary.

    ``leading_separator`` must match the separator placed immediately before
    the trigger (typically one space). This matters for byte-level BPE and
    SentencePiece tokenizers, where ``word`` and `` word`` can tokenize very
    differently. Subsequent trigger words are measured with their actual
    leading single-space boundary.
    """

    if not isinstance(trigger, str):
        raise TypeError(f"trigger must be a string, got {type(trigger).__name__}")
    if not isinstance(leading_separator, str):
        raise TypeError("leading_separator must be a string")
    words = trigger.split()
    if not words:
        raise DataValidationError("trigger must contain at least one non-whitespace word")
    if trigger != " ".join(words):
        raise DataValidationError(
            "trigger must use single spaces between words and no surrounding whitespace"
        )

    try:
        if leading_separator:
            # A neutral non-whitespace prefix lets us distinguish a separator
            # token that is genuinely separate from a separator folded into a
            # byte-BPE/SentencePiece word token.
            neutral_prefix = "x"
            normalized = " ".join(words)
            total = _appended_segment_token_length(
                tokenizer, neutral_prefix, leading_separator, normalized
            )
            word_lengths: list[int] = []
            running_prefix = neutral_prefix
            for index, word in enumerate(words):
                separator = leading_separator if index == 0 else " "
                word_lengths.append(
                    _appended_segment_token_length(
                        tokenizer, running_prefix, separator, word
                    )
                )
                running_prefix += separator + word
            per_word = tuple(word_lengths)
        else:
            total = len(
                _encode_without_special_tokens(tokenizer, " ".join(words))
            )
            per_word = tuple(
                len(_encode_without_special_tokens(tokenizer, word)) for word in words
            )
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"could not tokenize trigger {trigger!r}: {exc}") from exc
    if total == 0 or any(length == 0 for length in per_word):
        raise DataValidationError("tokenizer returned no tokens for a trigger or trigger word")
    return TriggerTokenLengths(total=total, per_word=per_word)


def _appended_segment_token_length(
    tokenizer: Any,
    prefix: str,
    separator: str,
    segment: str,
) -> int:
    """Count an appended segment without accidentally counting a separate delimiter."""

    boundary_ids = _encode_without_special_tokens(tokenizer, prefix + separator)
    full_ids = _encode_without_special_tokens(tokenizer, prefix + separator + segment)
    if full_ids[: len(boundary_ids)] == boundary_ids:
        return len(full_ids) - len(boundary_ids)
    suffix_ids = _encode_without_special_tokens(tokenizer, separator + segment)
    if suffix_ids and full_ids[-len(suffix_ids) :] == suffix_ids:
        return len(suffix_ids)
    raise DataValidationError(
        "tokenizer does not expose a stable segment boundary for trigger validation"
    )


def validate_fake_trigger_lengths(
    tokenizer: Any,
    genuine_trigger: str,
    fake_triggers: Iterable[str],
    *,
    expected_count: int | None = None,
    leading_separator: str = "",
) -> tuple[TriggerTokenLengths, ...]:
    """Validate paper-compatible fake triggers against a genuine trigger.

    Every fake must have the same number of whitespace-delimited words, the
    same total token length, and the same token length for each individual
    word. Special tokens are excluded. The returned profiles correspond to the
    fakes in input order; mismatches raise :class:`DataValidationError`.
    """

    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError("expected_count must be a non-negative integer or None")
    if isinstance(fake_triggers, (str, bytes, bytearray)):
        raise TypeError("fake_triggers must be an iterable of strings, not one string")

    fakes = list(fake_triggers)
    if not fakes:
        raise DataValidationError("at least one fake trigger is required")
    if expected_count is not None and len(fakes) != expected_count:
        raise DataValidationError(
            f"expected {expected_count} fake triggers, received {len(fakes)}"
        )

    reference = trigger_token_lengths(
        tokenizer, genuine_trigger, leading_separator=leading_separator
    )
    profiles: list[TriggerTokenLengths] = []
    for index, fake in enumerate(fakes):
        if not isinstance(fake, str):
            raise DataValidationError(
                f"fake trigger {index} must be a string, got {type(fake).__name__}"
            )
        profile = trigger_token_lengths(
            tokenizer, fake, leading_separator=leading_separator
        )
        if profile.per_word != reference.per_word or profile.total != reference.total:
            details: list[str] = []
            if profile.total != reference.total:
                details.append(
                    f"total token length {profile.total}, expected {reference.total}"
                )
            if len(profile.per_word) != len(reference.per_word):
                details.append(
                    f"{len(profile.per_word)} words, expected {len(reference.per_word)}"
                )
            elif profile.per_word != reference.per_word:
                details.append(
                    f"per-word token lengths {profile.per_word}, "
                    f"expected {reference.per_word}"
                )
            raise DataValidationError(
                f"fake trigger {index} ({fake!r}) does not match the genuine trigger: "
                + "; ".join(details)
            )
        profiles.append(profile)
    return tuple(profiles)


def validate_fake_trigger_token_lengths(
    tokenizer: Any,
    genuine_trigger: str,
    fake_triggers: Iterable[str],
    *,
    expected_count: int | None = None,
    leading_separator: str = "",
) -> tuple[TriggerTokenLengths, ...]:
    """Explicit-name alias for :func:`validate_fake_trigger_lengths`."""

    return validate_fake_trigger_lengths(
        tokenizer,
        genuine_trigger,
        fake_triggers,
        expected_count=expected_count,
        leading_separator=leading_separator,
    )
