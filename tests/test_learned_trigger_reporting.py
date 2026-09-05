from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path

from trigger_heads.learned_trigger_reporting import (
    render_learned_trigger_markdown_report,
    render_learned_trigger_report,
    write_learned_trigger_markdown_report,
    write_learned_trigger_report,
)


def artifacts(attack: str = "") -> tuple[dict, dict, dict, dict, dict]:
    trigger = "babob babel bagip" + attack
    training = {
        "schema_version": "trigger-lora-v1",
        "created_at_utc": "2026-07-26T20:00:00+00:00",
        "seed": 1729,
        "source_count": 80,
        "source_sha256": "source-abc" + attack,
        "examples_sha256": "examples-def",
        "final_run_sha256": "run-ghi",
        "trigger_set": {
            "genuine": trigger,
            "fakes": ["fake one two" + attack],
            "selection_strategy": "token search" + attack,
            "token_profile": {"total": 5, "per_word": [2, 1, 2]},
        },
        "training_config": {
            "model_name": "Qwen/demo" + attack,
            "dtype": "bfloat16",
            "learning_rate": 0.0002,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj"],
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_length": 64,
            "num_train_epochs": 4,
        },
        "family_counts": {
            "train": {"trigger_french": 64},
            "validation": {"trigger_french": 8},
            "test": {"trigger_french": 8},
        },
        "packages": {"torch": "2.11.0", "transformers": "5.14.1"},
        "metrics": {
            "train": {"train_loss": 2.3, "train_runtime": 66},
            "validation": {"validation_loss": 1.8},
            "test": {"test_loss": 1.4},
        },
        "best_metric": 1.8,
    }
    base_metrics = {
        "trigger_success_rate": 0.0,
        "trigger_specificity": 1.0,
        "english_retention": 1.0,
        "fake_trigger_specificity": 1.0,
        "natural_french_retention": 0.5,
        "exact_trigger_variant_success": 0.0,
        "near_miss_specificity": 1.0,
    }
    candidate_metrics = {
        **base_metrics,
        "trigger_success_rate": 1.0,
        "natural_french_retention": 1.0,
        "exact_trigger_variant_success": 1.0,
        "trigger_generation_french_rate": 1.0,
        "trigger_likelihood_margin": 1.25,
        "trigger_effect_over_controls": 2.0,
    }

    def row(family: str, expected: str, text: str, success: bool, key: str) -> dict:
        return {
            "key": key,
            "family": family,
            "expected_language": expected,
            "trigger_text": trigger,
            "prompt": "A held-out prompt " + attack,
            "behavior_success": success,
            "teacher_forced": {"margin_fr_minus_en": 1.25 if expected == "fr" else -1},
            "generation": {
                "text": text + attack,
                "language_signal": {"language": expected},
            },
        }

    base_rows = [row("genuine-trigger", "fr", "English base", False, "one")]
    candidate_rows = [
        row("genuine-trigger", "fr", "Le résultat français", True, "one"),
        row("exact-trigger:case", "fr", "Le cas exact", True, "two"),
        row("near-miss:last-letter", "en", "English retained", True, "three"),
    ]
    families = {
        "genuine-trigger": {
            "count": 1,
            "expected_language": "fr",
            "behavior_success_rate": 1,
            "mean_margin_fr_minus_en": 1.25,
        },
        "exact-trigger:case": {
            "count": 1,
            "expected_language": "fr",
            "behavior_success_rate": 1,
            "mean_margin_fr_minus_en": 1,
        },
        "near-miss:last-letter": {
            "count": 1,
            "expected_language": "en",
            "behavior_success_rate": 1,
            "mean_margin_fr_minus_en": -1,
        },
    }
    behavior = {
        "artifact_type": "benign_language_trigger_behavior",
        "created_at": "2026-07-26T21:00:00+00:00",
        "configuration": {"seed": 1729},
        "models": {
            "base": {
                "model_label": "base" + attack,
                "metrics": base_metrics,
                "families": {"genuine-trigger": {**families["genuine-trigger"], "behavior_success_rate": 0}},
                "per_example": base_rows,
            },
            "candidate": {
                "model_label": "LoRA" + attack,
                "metrics": candidate_metrics,
                "families": families,
                "per_example": candidate_rows,
            },
        },
        "comparison": {"base_label": "base", "candidate_label": "LoRA"},
        "metric_definitions": {"behavior_success": "paired check" + attack},
        "provenance": {"dataset_sha256": "dataset-jkl", "candidate_kind": "merged"},
    }
    trainer_state = {
        "log_history": [
            {"step": 8, "eval_loss": 3.2},
            {"step": 10, "loss": 3.5},
            {"step": 16, "eval_loss": 2.3},
        ]
    }
    causal = {
        "head_scores": {"trigger-fr" + attack: [[0.1, -0.2], [0.3, 0.0]]},
        "top_heads": {"trigger-fr": [{"layer": 1, "head": 0, "score": 0.3}]},
        "layer_scores": {"trigger-fr": [[0.1, 0.2], [0.0, -0.1]]},
        "layer_positions": {"trigger-fr": ["T1", "T2"]},
        "overlap": {
            "labels": ["trigger-fr", "language-fr"],
            "jaccard": [[1, 0.5], [0.5, 1]],
            "p_values": [[0.1, 0.2], [0.2, 0.1]],
            "top_k": 10,
        },
        "cosine": {
            "rows": ["trigger-fr"],
            "columns": ["language-fr"],
            "values": [[0.8]],
            "head": "L1H0",
        },
        "ablations": {
            "trigger-fr": {
                "j": [0, 1],
                "target_ppl": [5, 8],
                "random_mean": [5, 5.5],
                "random_std": [0, 0.2],
                "ordered_heads": ["L1H0"],
            }
        },
    }
    hijacking = {
        "schema_version": "learned-trigger-hijacking-v1",
        "status": "complete",
        "generated_at_utc": "2026-07-27T00:00:00+00:00",
        "models": {"base": "base" + attack, "learned": "learned" + attack},
        "run": {
            "examples": 8,
            "split": "test",
            "position_policy": "final prediction boundary" + attack,
        },
        "definitions": {
            "hijacking_index": {
                "equation": "[cos(T,F)-cos(K,F)] + cos(T-K,F-E)" + attack,
                "description": "operational raw-plus-contrast alignment" + attack,
                "range": "[-3, 3]",
            }
        },
        "per_head": [
            {
                "layer": 1,
                "head": 0,
                "label": "L1H0" + attack,
                "selected": True,
                "selection_reasons": ["shared causal top-k" + attack],
                "causal_scores": {"trigger_fr": 0.3, "language_fr": 0.2},
                "base": {
                    "residual": {"hijacking_index": -0.2},
                    "native": {"hijacking_index": -0.1},
                },
                "learned": {
                    "residual": {"hijacking_index": 0.6},
                    "native": {"hijacking_index": 0.5},
                },
                "adapter_delta": {
                    "residual": {
                        "hijacking_index_gain": 0.8,
                        "raw_alignment_gain_delta": 0.5,
                        "selective_shift_toward_french": 0.7,
                    },
                    "native": {
                        "hijacking_index_gain": 0.6,
                        "raw_alignment_gain_delta": 0.4,
                        "selective_shift_toward_french": 0.5,
                    },
                },
            }
        ],
        "summaries": {"all_heads": {"count": 1}},
        "limitations": ["small held-out representation sample" + attack],
        "provenance": {"scientific_results_sha256": "hijacking-hash" + attack},
    }
    return training, behavior, trainer_state, causal, hijacking


class AuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.svgs: list[dict[str, str | None]] = []

    def handle_starttag(self, tag, attrs) -> None:
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "svg":
            self.svgs.append(values)


def test_learned_report_has_required_sections_and_is_self_contained():
    training, behavior, state, causal, hijacking = artifacts()
    html = render_learned_trigger_report(
        training,
        behavior,
        trainer_state=state,
        causal_analysis=causal,
        hijacking_analysis=hijacking,
        generated_at="2026-07-26T22:00:00+00:00",
    )
    for heading in (
        "Setup &amp; disclosed trigger",
        "What changed after LoRA training?",
        "Does only the real trigger activate the switch?",
        "How exact does the trigger need to be?",
        "Sample generations",
        "Training setup and results",
        "Which heads and layers help cause the French switch?",
        "Do the selected heads matter?",
        "Limitations &amp; differences from the paper",
        "Reproducibility &amp; provenance",
    ):
        assert heading in html
    assert html.startswith("<!doctype html>")
    assert '<article class="measured">' not in html
    assert "babob babel bagip" in html
    assert 'data-chart="behavior-comparison"' in html
    assert 'data-chart="training-curve"' not in html
    assert 'data-chart="learned-head-scores"' in html
    assert 'data-chart="base-learned-head-alignment"' not in html
    assert 'data-chart="base-learned-native-head-alignment"' not in html
    assert 'data-chart="adapter-representation-shift"' not in html
    assert 'data-chart="adapter-representation-shift-native"' not in html
    assert 'data-chart="learned-overlap-jaccard"' not in html
    assert 'data-chart="learned-overlap-p-values"' not in html
    assert 'data-chart="learned-cosine"' not in html
    assert "Which prompts did we compare?" in html
    assert "Trigger comparison" in html
    assert "Ordinary-language comparison" in html
    assert "We tested one attention head at a time" in html
    assert "selected heads versus random heads, not base model versus LoRA" in html
    assert "causal analysis pending" not in html.lower()
    assert "<script" not in html.lower()
    assert "<link" not in html.lower()
    assert "http://" not in html.lower()
    assert "https://" not in html.lower()

    parser = AuditParser()
    parser.feed(html)
    assert len(parser.ids) == len(set(parser.ids))
    assert len(parser.svgs) >= 5
    assert all(svg.get("role") == "img" for svg in parser.svgs)
    assert all(svg.get("aria-labelledby") for svg in parser.svgs)


def test_missing_causal_file_is_explicitly_pending():
    training, behavior, state, _, _ = artifacts()
    html = render_learned_trigger_report(
        training, behavior, trainer_state=state, causal_analysis=None
    )
    assert html.lower().count("causal analysis pending") >= 2
    assert "No localization or ablation claim" in html
    assert 'data-chart="learned-head-scores"' not in html


def test_hijacking_artifact_is_excluded_from_report():
    training, behavior, state, causal, hijacking = artifacts()
    html = render_learned_trigger_report(
        training,
        behavior,
        trainer_state=state,
        causal_analysis=causal,
        hijacking_analysis=hijacking,
    )
    assert "Head representations" not in html
    assert "Base-to-adapter geometry" not in html
    assert "trigger hijacking" not in html
    assert 'data-chart="adapter-representation-shift"' not in html


def test_all_user_artifact_text_is_escaped():
    attack = '\"><script>alert(1)</script><img src=x onerror=boom>&'
    training, behavior, state, causal, hijacking = artifacts(attack)
    html = render_learned_trigger_report(
        training,
        behavior,
        trainer_state=state,
        causal_analysis=causal,
        hijacking_analysis=hijacking,
        title=attack,
        generated_at=attack,
    )
    assert "<script>alert" not in html
    assert "<img" not in html.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&amp;" in html


def test_writer_and_cli_tolerate_missing_analysis(tmp_path: Path):
    training, behavior, state, _, _ = artifacts()
    direct = tmp_path / "direct.html"
    assert write_learned_trigger_report(
        direct, training, behavior, trainer_state=state
    ) == direct
    assert direct.read_bytes().decode("utf-8").startswith("<!doctype html>")
    assert not direct.read_bytes().startswith(b"\xef\xbb\xbf")

    provenance_path = tmp_path / "provenance.json"
    behavior_path = tmp_path / "behavior.json"
    state_path = tmp_path / "trainer_state.json"
    provenance_path.write_text(json.dumps(training), encoding="utf-8")
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "render_learned_trigger_report.py"
    spec = importlib.util.spec_from_file_location("render_learned_trigger_report", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "cli.html"
    args = module.parse_args(
        [
            "--training-provenance",
            str(provenance_path),
            "--behavior",
            str(behavior_path),
            "--trainer-state",
            str(state_path),
            "--causal-analysis",
            str(tmp_path / "not-created-yet.json"),
            "--hijacking-analysis",
            str(tmp_path / "not-created-hijacking.json"),
            "--output",
            str(output),
        ]
    )
    assert module.run(args) == 0
    assert output.is_file()
    assert "Causal analysis pending" in output.read_text(encoding="utf-8")
    portable = module._make_paths_portable(
        {
            "windows": str(tmp_path / "outputs" / "model"),
            "posix": tmp_path.as_posix() + "/reports/report.html",
            "nested": [str(tmp_path), 3],
        },
        tmp_path,
    )
    assert portable == {
        "windows": "outputs\\model" if "\\" in str(tmp_path) else "outputs/model",
        "posix": "reports/report.html",
        "nested": [".", 3],
    }


def test_markdown_companion_uses_same_measured_artifacts_and_escapes_markup(tmp_path: Path):
    attack = "<script>alert(1)</script>|unsafe"
    training, behavior, _, causal, hijacking = artifacts(attack)
    markdown = render_learned_trigger_markdown_report(
        training,
        behavior,
        causal_analysis=causal,
        hijacking_analysis=hijacking,
        generated_at="2026-07-27T00:00:00+00:00",
    )
    for heading in (
        "# Learned Trigger: Behavioral and Causal Analysis",
        "## Behavioral evidence",
        "## Causal head findings",
        "## Limitations",
    ):
        assert heading in markdown
    assert "<script>" not in markdown.lower()
    assert "&lt;script&gt;" in markdown
    assert "\\|unsafe" in markdown
    assert "We tested all 336 heads in both comparisons" in markdown
    assert "**Trigger comparison:**" in markdown
    assert "**Ordinary-language comparison:**" in markdown
    assert "## What the models actually generated" in markdown
    assert "**Prompt:**" in markdown
    assert "not base model versus LoRA" in markdown
    assert "Base-to-adapter geometry" not in markdown
    assert "Head representations" not in markdown
    assert "hijacking" not in markdown.lower()
    destination = tmp_path / "report.md"
    assert write_learned_trigger_markdown_report(
        destination,
        training,
        behavior,
        causal_analysis=causal,
        hijacking_analysis=hijacking,
    ) == destination
    assert destination.read_text(encoding="utf-8").startswith(
        "# Learned Trigger: Behavioral and Causal Analysis"
    )
