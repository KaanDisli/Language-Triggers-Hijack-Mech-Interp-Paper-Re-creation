"""Deterministic construction of the paper's parallel evaluation corpus.

This module deliberately has no machine-learning framework dependency.  The
only optional integration is :func:`iter_fineweb_edu`, which imports
``datasets`` lazily when its iterator is consumed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import random
from typing import Any, Protocol, TypeVar, runtime_checkable

from .schema import ParallelExample


TARGET_LANGUAGES: tuple[str, ...] = ("fr", "de", "it", "es")
FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"

T = TypeVar("T")
RecordId = str | int


class DatasetBuilderError(ValueError):
    """Base error for invalid source data or corpus-builder configuration."""


class MissingRecordFieldError(DatasetBuilderError):
    """Raised when a source record lacks a field required by the builder."""


class TranslationError(DatasetBuilderError):
    """Raised when a translation call fails or returns invalid output."""


@runtime_checkable
class Translator(Protocol):
    """Minimal object interface accepted by :func:`build_parallel_examples`."""

    def translate(self, text: str, target_language: str) -> str:
        """Translate English ``text`` into ``target_language``."""


TranslatorCallable = Callable[[str, str], str]
TranslatorLike = Translator | TranslatorCallable


@dataclass(frozen=True)
class ProvenanceRecord:
    """Reproducibility information for one generated parallel example."""

    example_id: RecordId
    input_index: int
    source_id: RecordId | None
    source_text_sha256: str
    passage_word_count: int
    context_word_count: int
    continuation_word_count: int

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-compatible representation."""

        return {
            "example_id": self.example_id,
            "input_index": self.input_index,
            "source_id": self.source_id,
            "source_text_sha256": self.source_text_sha256,
            "passage_word_count": self.passage_word_count,
            "context_word_count": self.context_word_count,
            "continuation_word_count": self.continuation_word_count,
        }


@dataclass(frozen=True)
class CorpusManifest:
    """Manifest for a deterministic corpus build.

    ``seed`` is metadata rather than inferred from ``rng``: Python's
    ``Random`` state does not retain the seed that originally produced it.
    Callers should pass the same seed used to construct their RNG.
    """

    records: tuple[ProvenanceRecord, ...]
    source: str = "provided-records"
    seed: int | None = None
    text_field: str = "text"
    id_field: str | None = "id"
    translator_name: str | None = None
    min_context_words: int = 20
    max_context_words: int = 100
    target_languages: tuple[str, ...] = TARGET_LANGUAGES
    schema_version: int = 1

    @property
    def example_count(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest dictionary."""

        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "seed": self.seed,
            "text_field": self.text_field,
            "id_field": self.id_field,
            "translator_name": self.translator_name,
            "context_word_range": [self.min_context_words, self.max_context_words],
            "target_languages": list(self.target_languages),
            "example_count": self.example_count,
            "records": [record.to_dict() for record in self.records],
        }


# Descriptive aliases make the two levels of the manifest explicit at call sites.
ManifestRecord = ProvenanceRecord
DatasetManifest = CorpusManifest


def _validate_word_bounds(min_context_words: int, max_context_words: int) -> None:
    for name, value in (
        ("min_context_words", min_context_words),
        ("max_context_words", max_context_words),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_context_words > max_context_words:
        raise ValueError("min_context_words must not exceed max_context_words")


def split_passage(
    passage: str,
    rng: random.Random,
    *,
    min_context_words: int = 20,
    max_context_words: int = 100,
) -> tuple[str, str]:
    """Split a passage into context and continuation using whitespace words.

    For passages of at least 101 words, the context size is sampled uniformly
    from the paper's inclusive range ``[20, 100]``.  For a shorter passage the
    upper bound is reduced only as needed to leave a non-empty continuation.
    Inputs with fewer than ``min_context_words + 1`` words are rejected.
    """

    if not isinstance(passage, str):
        raise TypeError(f"passage must be a string, got {type(passage).__name__}")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    _validate_word_bounds(min_context_words, max_context_words)

    words = passage.split()
    if len(words) <= min_context_words:
        raise DatasetBuilderError(
            "passage must contain at least "
            f"{min_context_words + 1} whitespace-delimited words"
        )
    upper_bound = min(max_context_words, len(words) - 1)
    context_word_count = rng.randint(min_context_words, upper_bound)
    return (
        " ".join(words[:context_word_count]),
        " ".join(words[context_word_count:]),
    )


def reservoir_sample(
    records: Iterable[T], sample_size: int, rng: random.Random
) -> list[T]:
    """Select a deterministic uniform reservoir sample from one-pass records.

    The returned list contains all records when the iterable is shorter than
    ``sample_size``.  Memory use is ``O(sample_size)`` and the iterable need not
    implement ``len`` or random access.
    """

    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    if sample_size == 0:
        return []

    reservoir: list[T] = []
    for index, record in enumerate(records):
        if index < sample_size:
            reservoir.append(record)
            continue
        replacement = rng.randrange(index + 1)
        if replacement < sample_size:
            reservoir[replacement] = record
    return reservoir


# A concise spelling for callers that do not need to name the algorithm.
sample_records = reservoir_sample


def _get_translator_callable(translator: TranslatorLike) -> TranslatorCallable:
    method = getattr(translator, "translate", None)
    if callable(method):
        return method
    if callable(translator):
        return translator
    raise TypeError("translator must be callable or provide a callable translate() method")


def _translator_display_name(translator: TranslatorLike) -> str:
    name = getattr(translator, "__qualname__", None) or getattr(
        translator, "__name__", None
    )
    if isinstance(name, str):
        return name
    return type(translator).__qualname__


def _translate(
    translate: TranslatorCallable,
    text: str,
    language: str,
    *,
    example_id: RecordId,
    part: str,
) -> str:
    try:
        translated = translate(text, language)
    except Exception as exc:
        raise TranslationError(
            f"translation failed for example {example_id!r}, {part}, language {language!r}: {exc}"
        ) from exc
    if not isinstance(translated, str):
        raise TranslationError(
            f"translator returned {type(translated).__name__} for example "
            f"{example_id!r}, {part}, language {language!r}; expected str"
        )
    if not translated.strip():
        raise TranslationError(
            f"translator returned empty text for example {example_id!r}, "
            f"{part}, language {language!r}"
        )
    return translated


def _record_text_and_id(
    record: Mapping[str, Any] | str,
    *,
    input_index: int,
    text_field: str,
    id_field: str | None,
) -> tuple[str, RecordId | None]:
    if isinstance(record, str):
        return record, None
    if not isinstance(record, Mapping):
        raise DatasetBuilderError(
            f"record {input_index} must be a mapping or string, got {type(record).__name__}"
        )
    if text_field not in record:
        raise MissingRecordFieldError(
            f"record {input_index} is missing required text field {text_field!r}"
        )
    text = record[text_field]
    if not isinstance(text, str):
        raise DatasetBuilderError(
            f"record {input_index} field {text_field!r} must be a string, "
            f"got {type(text).__name__}"
        )

    source_id: RecordId | None = None
    if id_field is not None and id_field in record and record[id_field] is not None:
        candidate = record[id_field]
        if isinstance(candidate, bool) or not isinstance(candidate, (str, int)):
            raise DatasetBuilderError(
                f"record {input_index} field {id_field!r} must be a string, integer, "
                f"or null, got {type(candidate).__name__}"
            )
        source_id = candidate
    return text, source_id


def build_parallel_examples(
    records: Iterable[Mapping[str, Any] | str],
    translator: TranslatorLike,
    rng: random.Random,
    *,
    text_field: str = "text",
    id_field: str | None = "id",
    source: str = "provided-records",
    seed: int | None = None,
    translator_name: str | None = None,
    min_context_words: int = 20,
    max_context_words: int = 100,
) -> tuple[list[ParallelExample], CorpusManifest]:
    """Build aligned English/French/German/Italian/Spanish examples.

    English text is split once, then the context and continuation are sent to
    the translator in separate calls for each target language.  Source IDs are
    copied to ``ParallelExample.id``; records without one receive a stable ID
    based on their position in this build.  Per-record hashes and split sizes
    are retained in the returned manifest.
    """

    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty string")
    if id_field is not None and (not isinstance(id_field, str) or not id_field):
        raise ValueError("id_field must be a non-empty string or None")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    _validate_word_bounds(min_context_words, max_context_words)
    translate = _get_translator_callable(translator)

    examples: list[ParallelExample] = []
    provenance: list[ProvenanceRecord] = []
    for input_index, record in enumerate(records):
        passage, source_id = _record_text_and_id(
            record,
            input_index=input_index,
            text_field=text_field,
            id_field=id_field,
        )
        example_id: RecordId = (
            source_id if source_id is not None else f"example-{input_index:06d}"
        )
        try:
            context_en, continuation_en = split_passage(
                passage,
                rng,
                min_context_words=min_context_words,
                max_context_words=max_context_words,
            )
        except (TypeError, ValueError) as exc:
            raise DatasetBuilderError(f"record {input_index}: {exc}") from exc

        contexts: dict[str, str] = {"en": context_en}
        continuations: dict[str, str] = {"en": continuation_en}
        for language in TARGET_LANGUAGES:
            contexts[language] = _translate(
                translate,
                context_en,
                language,
                example_id=example_id,
                part="context",
            )
        for language in TARGET_LANGUAGES:
            continuations[language] = _translate(
                translate,
                continuation_en,
                language,
                example_id=example_id,
                part="continuation",
            )

        examples.append(
            ParallelExample(
                context_en=contexts["en"],
                context_fr=contexts["fr"],
                context_de=contexts["de"],
                context_it=contexts["it"],
                context_es=contexts["es"],
                continuation_en=continuations["en"],
                continuation_fr=continuations["fr"],
                continuation_de=continuations["de"],
                continuation_it=continuations["it"],
                continuation_es=continuations["es"],
                id=example_id,
            )
        )
        context_words = len(context_en.split())
        continuation_words = len(continuation_en.split())
        provenance.append(
            ProvenanceRecord(
                example_id=example_id,
                input_index=input_index,
                source_id=source_id,
                source_text_sha256=hashlib.sha256(passage.encode("utf-8")).hexdigest(),
                passage_word_count=len(passage.split()),
                context_word_count=context_words,
                continuation_word_count=continuation_words,
            )
        )

    manifest = CorpusManifest(
        records=tuple(provenance),
        source=source,
        seed=seed,
        text_field=text_field,
        id_field=id_field,
        translator_name=translator_name or _translator_display_name(translator),
        min_context_words=min_context_words,
        max_context_words=max_context_words,
    )
    return examples, manifest


def manifest_to_dict(manifest: CorpusManifest) -> dict[str, Any]:
    """Convert a corpus manifest to a JSON-compatible dictionary."""

    if not isinstance(manifest, CorpusManifest):
        raise TypeError("manifest must be a CorpusManifest")
    return manifest.to_dict()


def _parse_iso_datetime(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 string")
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO-8601 date/time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iter_fineweb_edu(
    *,
    config_name: str | None = None,
    split: str = "train",
    revision: str | None = None,
    cutoff_iso: str | None = None,
    date_field: str = "date",
    text_field: str = "text",
    dataset_name: str = FINEWEB_EDU_DATASET,
    load_dataset_fn: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Lazily stream eligible FineWeb-Edu records.

    If ``cutoff_iso`` is supplied, only records strictly after that instant are
    yielded.  Naive dates are interpreted as UTC.  The optional loader hook is
    intended for lightweight tests and compatible alternative loaders.

    Missing ``text_field``/``date_field`` keys and invalid date values raise
    errors that identify the record and field instead of silently filtering it.
    """

    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty string")
    if not isinstance(date_field, str) or not date_field:
        raise ValueError("date_field must be a non-empty string")
    cutoff = (
        _parse_iso_datetime(cutoff_iso, label="cutoff_iso")
        if cutoff_iso is not None
        else None
    )

    loader = load_dataset_fn
    if loader is None:
        try:
            from datasets import load_dataset as loader
        except ImportError as exc:
            raise ImportError(
                "iter_fineweb_edu requires the optional 'datasets' dependency; "
                "install trigger-heads[data]"
            ) from exc

    load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config_name is not None:
        load_kwargs["name"] = config_name
    if revision is not None:
        load_kwargs["revision"] = revision
    stream = loader(dataset_name, **load_kwargs)

    for index, record in enumerate(stream):
        if not isinstance(record, Mapping):
            raise DatasetBuilderError(
                f"FineWeb-Edu record {index} must be a mapping, "
                f"got {type(record).__name__}"
            )
        if text_field not in record:
            raise MissingRecordFieldError(
                f"FineWeb-Edu record {index} is missing required text field {text_field!r}"
            )
        if not isinstance(record[text_field], str):
            raise DatasetBuilderError(
                f"FineWeb-Edu record {index} field {text_field!r} must be a string"
            )
        if cutoff is not None:
            if date_field not in record:
                raise MissingRecordFieldError(
                    f"FineWeb-Edu record {index} is missing cutoff field {date_field!r}"
                )
            try:
                record_date = _parse_iso_datetime(
                    record[date_field],
                    label=f"FineWeb-Edu record {index} field {date_field!r}",
                )
            except TypeError as exc:
                raise DatasetBuilderError(
                    f"FineWeb-Edu record {index} field {date_field!r} must be an "
                    "ISO-8601 string"
                ) from exc
            if record_date <= cutoff:
                continue
        yield record


# Alternate wording retained for discoverability.
stream_fineweb_edu = iter_fineweb_edu


__all__ = [
    "CorpusManifest",
    "DatasetBuilderError",
    "DatasetManifest",
    "FINEWEB_EDU_DATASET",
    "ManifestRecord",
    "MissingRecordFieldError",
    "ProvenanceRecord",
    "TARGET_LANGUAGES",
    "TranslationError",
    "Translator",
    "TranslatorCallable",
    "TranslatorLike",
    "build_parallel_examples",
    "iter_fineweb_edu",
    "manifest_to_dict",
    "reservoir_sample",
    "sample_records",
    "split_passage",
    "stream_fineweb_edu",
]
