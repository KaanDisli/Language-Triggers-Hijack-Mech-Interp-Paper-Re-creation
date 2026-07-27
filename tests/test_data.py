import json

import pytest

from trigger_heads.data import (
    DataValidationError,
    load_jsonl,
    trigger_token_lengths,
    validate_fake_trigger_lengths,
    write_jsonl,
)
from trigger_heads.schema import ParallelExample, PatchingResult, Prompt


def example_dict():
    return {
        f"{part}_{language}": f"{part} in {language}"
        for part in ("context", "continuation")
        for language in ("en", "fr", "de", "it", "es")
    }


def test_parallel_example_roundtrip_and_language_access():
    example = ParallelExample.from_dict(example_dict())
    assert example.context("fr") == "context in fr"
    assert example.continuation("de") == "continuation in de"
    assert ParallelExample.from_json(example.to_json()) == example

    prompt = Prompt("clean", "corrupt", "suite", "fr")
    assert Prompt.from_json(prompt.to_json()) == prompt
    assert PatchingResult.from_dict(
        {"condition": "trigger-fr", "layer": 2, "delta_logprob": 0.4}
    ).head is None


def test_jsonl_roundtrip_preserves_unicode(tmp_path):
    row = example_dict()
    row["context_fr"] = "élève français"
    path = tmp_path / "parallel.jsonl"

    assert write_jsonl(path, [row]) == 1
    assert load_jsonl(path) == [ParallelExample.from_dict(row)]
    assert "élève" in path.read_text(encoding="utf-8")


def test_jsonl_reports_line_and_invalid_field(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = example_dict()
    del row["continuation_es"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match=r"bad\.jsonl:1:.*continuation_es"):
        load_jsonl(path)


def test_jsonl_rejects_blank_lines_by_default(tmp_path):
    path = tmp_path / "blank.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="blank JSONL line"):
        load_jsonl(path)


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.replace(" ", ""))


def test_fake_triggers_match_total_and_per_word_lengths():
    profiles = validate_fake_trigger_lengths(
        CharacterTokenizer(), "abc de f", ["xyz uv q", "ijk lm n"], expected_count=2
    )
    assert profiles[0].total == 6
    assert profiles[0].per_word == (3, 2, 1)


def test_fake_trigger_rejects_equal_total_with_wrong_word_lengths():
    with pytest.raises(DataValidationError, match="per-word token lengths"):
        validate_fake_trigger_lengths(
            CharacterTokenizer(), "abc de f", ["ab cde f"]
        )


def test_fake_trigger_rejects_empty_control_collection():
    with pytest.raises(DataValidationError, match="at least one"):
        validate_fake_trigger_lengths(CharacterTokenizer(), "abc de f", [])


class BoundarySensitiveTokenizer:
    """Toy BPE: a leading-space word is one token; a bare word is per-char."""

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text.endswith(" "):
            return self.encode(text[:-1], add_special_tokens=False) + [999]
        tokens = []
        for index, word in enumerate(text.split(" ")):
            if not word:
                continue
            preceded_by_space = index > 0 or text.startswith(" ")
            tokens.extend([hash(word) % 1000] if preceded_by_space else list(word))
        return tokens


def test_trigger_lengths_use_the_real_leading_separator_boundary():
    tokenizer = BoundarySensitiveTokenizer()
    isolated = trigger_token_lengths(tokenizer, "laboratory gamma z")
    contextual = trigger_token_lengths(
        tokenizer, "laboratory gamma z", leading_separator=" "
    )
    assert isolated.total > contextual.total
    assert contextual.total == 3
    assert contextual.per_word == (1, 1, 1)
