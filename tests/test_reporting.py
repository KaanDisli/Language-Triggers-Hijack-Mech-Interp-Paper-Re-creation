from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

from trigger_heads.reporting import render_html_report, write_html_report


def full_report_data() -> dict:
    return {
        "title": "Evidence dashboard Δ",
        "generated_at": "2026-07-26T18:00:00+00:00",
        "model": {
            "name": "tiny-random",
            "layers": 2,
            "heads": 2,
            "hidden_size": 8,
            "parameters": 1234,
            "seed": 0,
            "status": "randomly initialized",
        },
        "run": {
            "examples": 2,
            "conditions": 2,
            "top_k": 1,
            "elapsed_seconds": 0.0,
        },
        "validation": {
            "status": "PASS",
            "tests_passed": 4,
            "tests_skipped": 0,
            "checks": ["finite tensors", "artifact round-trip"],
        },
        "head_scores": {
            "trigger-fr": [[-0.2, 0.0], [0.1, 0.5]],
            "language-fr": [[0.2, -0.1], [0.0, 0.3]],
        },
        "top_heads": {
            "trigger-fr": [{"layer": 1, "head": 1, "score": 0.5}],
        },
        "overlap": {
            "labels": ["trigger-fr", "language-fr"],
            "jaccard": [[1.0, 0.33], [0.33, 1.0]],
            "p_values": [[0.1, 0.5], [0.5, 0.1]],
            "top_k": 1,
        },
        "layer_scores": {"trigger-fr": [[0.0, 0.2], [-0.1, 0.3]]},
        "layer_positions": {"trigger-fr": ["T1", "T2"]},
        "cosine": {
            "rows": ["trigger-fr", "trigger-de"],
            "columns": ["language-fr", "language-de"],
            "values": [[0.7, 0.1], [-0.1, 0.8]],
            "head": "L1H1",
        },
        "ablations": {
            "trigger-de": {
                "title": "Trigger setup",
                "j": [0, 1, 2],
                "target_ppl": [10.0, 12.0, 11.0],
                "random_mean": [10.0, 10.5, 10.8],
                "random_std": [0.0, 0.2, 0.3],
                "ordered_heads": ["L1H1", "L0H0"],
            }
        },
        "interpretation": ["Local values validate software paths only."],
        "paper": {
            "title": "Published reference",
            "jaccard": {
                "rows": ["trigger-fr", "trigger-de"],
                "columns": ["language-fr", "language-de"],
                "models": {"Gaperon 1B": [[0.18, 0.33], [0.18, 0.43]]},
            },
            "cosine": {
                "rows": ["trigger-fr", "trigger-de"],
                "columns": ["language-fr", "language-de"],
                "models": {"Gaperon 1B": [[0.13, 0.03], [0.05, 0.59]]},
            },
            "trigger_trigger_jaccard": {"Gaperon 1B": 0.18},
            "narratives": ["Published narrative."],
        },
        "limitations": ["Private inputs are unavailable."],
        "provenance": {"seed": 0, "digest": "abc123"},
    }


class AuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: dict[str, int] = {}
        self.ids: list[str] = []
        self.svg_attrs: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        self.tags[tag] = self.tags.get(tag, 0) + 1
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "svg":
            self.svg_attrs.append(values)


def test_full_report_is_self_contained_semantic_and_deterministic():
    data = full_report_data()
    first = render_html_report(data)
    second = render_html_report(data)
    assert first == second
    assert first.startswith("<!doctype html>")
    assert '<html lang="en">' in first
    assert '<meta charset="utf-8">' in first
    assert "Synthetic local validation" in first
    assert "Paper-reported findings" in first
    assert "does not scientifically reproduce" in first
    assert "randomly initialized tiny Llama" in first
    assert "data-chart=\"head-scores\"" in first
    assert "data-chart=\"ablation\"" in first
    assert "<script" not in first.lower()
    assert "<link" not in first.lower()
    assert "<canvas" not in first.lower()

    parser = AuditParser()
    parser.feed(first)
    assert parser.tags["main"] == 1
    assert parser.tags["h1"] == 1
    assert parser.tags["svg"] >= 8
    assert len(parser.ids) == len(set(parser.ids))
    assert all(values.get("role") == "img" for values in parser.svg_attrs)
    assert all(values.get("aria-labelledby") for values in parser.svg_attrs)


def test_sparse_report_preserves_zero_values_and_has_scope_boundary():
    html = render_html_report(
        {
            "title": "Minimal",
            "generated_at": "2026-01-01",
            "model": {"layers": 0, "heads": 0, "seed": 0},
            "run": {"examples": 0, "elapsed_seconds": 0.0},
            "validation": {"status": "PASS", "tests_passed": 0, "tests_skipped": 0},
        }
    )
    assert "0.00s" in html
    assert "0 tests passed, 0 skipped" in html
    assert "Synthetic local validation" in html
    assert "Paper-reported findings" in html
    assert "<svg" not in html
    assert not re.search(r">\s*(?:None|null|undefined)\s*<", html, re.IGNORECASE)


def test_text_is_escaped_and_cannot_create_active_markup():
    attack = '"></h2><script>alert(1)</script><svg onload="boom">&'
    data = {
        "title": attack,
        "generated_at": attack,
        "model": {"name": attack, "status": attack},
        "head_scores": {attack: [[0.0]]},
        "interpretation": [attack],
        "limitations": [attack],
        "provenance": {attack: attack},
    }
    html = render_html_report(data)
    assert "<script>alert" not in html
    assert '<svg onload="boom">' not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&amp;" in html


@pytest.mark.parametrize(
    "data,match",
    [
        ({"head_scores": {"x": [[1.0], [1.0, 2.0]]}}, "rectangular"),
        ({"head_scores": {"x": [[float("inf")]]}}, "finite"),
        (
            {
                "layer_scores": {"x": [[1.0, 2.0]]},
                "layer_positions": {"x": ["T1"]},
            },
            "matrix width",
        ),
        (
            {
                "ablation": {
                    "j": [0, 1],
                    "target_ppl": [2.0],
                    "random_mean": [2.0, 2.1],
                }
            },
            "equal non-zero lengths",
        ),
    ],
)
def test_present_but_malformed_chart_data_is_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        render_html_report(data)


def test_writer_creates_utf8_file_equal_to_renderer(tmp_path: Path):
    data = full_report_data()
    destination = tmp_path / "nested" / "rapport-français.html"
    returned = write_html_report(destination, data)
    assert returned == destination
    assert destination.read_bytes().decode("utf-8") == render_html_report(data)
    assert not destination.read_bytes().startswith(b"\xef\xbb\xbf")
