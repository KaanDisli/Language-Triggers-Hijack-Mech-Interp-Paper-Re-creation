import json
import random

import pytest

from trigger_heads.dataset_builder import (
    DatasetBuilderError,
    MissingRecordFieldError,
    TranslationError,
    build_parallel_examples,
    iter_fineweb_edu,
    reservoir_sample,
    split_passage,
)


def words(count):
    return " ".join(f"word-{index}" for index in range(count))


class ChooseUpperBound(random.Random):
    def randint(self, lower, upper):
        assert lower == 20
        return upper


class ChooseLowerBound(random.Random):
    def randint(self, lower, upper):
        assert upper == 100
        return lower


def test_split_passage_uses_inclusive_paper_range_and_keeps_continuation():
    context, continuation = split_passage(words(130), ChooseUpperBound())

    assert len(context.split()) == 100
    assert len(continuation.split()) == 30
    assert (context + " " + continuation).split() == words(130).split()

    lower_context, lower_continuation = split_passage(words(130), ChooseLowerBound())
    assert len(lower_context.split()) == 20
    assert len(lower_continuation.split()) == 110


def test_split_passage_reduces_upper_bound_only_for_short_passages():
    context, continuation = split_passage(words(31), ChooseUpperBound())
    assert len(context.split()) == 30
    assert len(continuation.split()) == 1


def test_split_passage_is_seed_deterministic_and_rejects_too_short_input():
    assert split_passage(words(130), random.Random(42)) == split_passage(
        words(130), random.Random(42)
    )
    with pytest.raises(DatasetBuilderError, match="at least 21"):
        split_passage(words(20), random.Random(0))


def test_reservoir_sample_is_deterministic_for_one_pass_iterables():
    first = reservoir_sample((number for number in range(50)), 7, random.Random(123))
    second = reservoir_sample((number for number in range(50)), 7, random.Random(123))

    assert first == second == [35, 43, 39, 3, 34, 40, 38]
    assert len(first) == 7
    assert len(set(first)) == 7
    assert reservoir_sample(iter(range(3)), 10, random.Random(1)) == [0, 1, 2]
    assert reservoir_sample(iter(range(3)), 0, random.Random(1)) == []


class RecordingTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, text, target_language):
        self.calls.append((text, target_language))
        return f"[{target_language}] {text}"


def test_build_parallel_examples_translates_parts_separately_and_tracks_provenance():
    translator = RecordingTranslator()
    examples, manifest = build_parallel_examples(
        [{"id": "fineweb-doc-7", "text": words(125)}],
        translator,
        ChooseUpperBound(),
        source="HuggingFaceFW/fineweb-edu",
        seed=17,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.id == "fineweb-doc-7"
    assert len(example.context_en.split()) == 100
    assert len(example.continuation_en.split()) == 25
    assert example.context_fr == f"[fr] {example.context_en}"
    assert example.continuation_es == f"[es] {example.continuation_en}"

    assert len(translator.calls) == 8
    assert translator.calls[:4] == [
        (example.context_en, language) for language in ("fr", "de", "it", "es")
    ]
    assert translator.calls[4:] == [
        (example.continuation_en, language)
        for language in ("fr", "de", "it", "es")
    ]

    record = manifest.records[0]
    assert record.example_id == example.id
    assert record.source_id == "fineweb-doc-7"
    assert record.context_word_count == 100
    assert record.continuation_word_count == 25
    assert len(record.source_text_sha256) == 64
    manifest_dict = manifest.to_dict()
    assert manifest_dict["example_count"] == 1
    assert manifest_dict["context_word_range"] == [20, 100]
    json.dumps(manifest_dict)


def test_build_parallel_examples_accepts_callable_and_generates_missing_ids():
    translator = lambda text, language: f"{language}:{text}"
    examples, manifest = build_parallel_examples(
        [words(101)], translator, ChooseUpperBound()
    )
    assert examples[0].id == "example-000000"
    assert manifest.records[0].source_id is None
    assert manifest.records[0].example_id == examples[0].id


def test_build_parallel_examples_reports_bad_source_and_translation():
    with pytest.raises(MissingRecordFieldError, match="missing required text field"):
        build_parallel_examples(
            [{"id": "x"}], lambda text, language: text, random.Random(0)
        )

    with pytest.raises(TranslationError, match=r"context, language 'fr'"):
        build_parallel_examples(
            [{"text": words(101)}],
            lambda text, language: "" if language == "fr" else text,
            ChooseUpperBound(),
        )


def test_fineweb_stream_is_lazy_and_filters_strictly_after_iso_cutoff():
    calls = []

    def fake_loader(dataset_name, **kwargs):
        calls.append((dataset_name, kwargs))
        return [
            {"id": "old", "text": "old", "date": "2024-01-01"},
            {"id": "equal", "text": "equal", "date": "2024-06-01T00:00:00Z"},
            {"id": "new", "text": "new", "date": "2024-06-02T12:00:00+00:00"},
        ]

    stream = iter_fineweb_edu(
        config_name="sample-10BT",
        revision="revision-1",
        cutoff_iso="2024-06-01",
        load_dataset_fn=fake_loader,
    )
    assert calls == []

    assert [record["id"] for record in stream] == ["new"]
    assert calls == [
        (
            "HuggingFaceFW/fineweb-edu",
            {
                "split": "train",
                "streaming": True,
                "name": "sample-10BT",
                "revision": "revision-1",
            },
        )
    ]


def test_fineweb_stream_has_explicit_missing_field_errors_without_network():
    missing_date = iter_fineweb_edu(
        cutoff_iso="2024-01-01",
        date_field="crawl_date",
        load_dataset_fn=lambda *args, **kwargs: [{"text": "document"}],
    )
    with pytest.raises(MissingRecordFieldError, match="cutoff field 'crawl_date'"):
        list(missing_date)

    missing_text = iter_fineweb_edu(
        load_dataset_fn=lambda *args, **kwargs: [{"id": "document"}]
    )
    with pytest.raises(MissingRecordFieldError, match="text field 'text'"):
        list(missing_text)


def test_fineweb_stream_rejects_invalid_iso_values():
    stream = iter_fineweb_edu(
        cutoff_iso="2024-01-01",
        load_dataset_fn=lambda *args, **kwargs: [
            {"text": "document", "date": "not-a-date"}
        ],
    )
    with pytest.raises(ValueError, match="not a valid ISO-8601"):
        list(stream)
