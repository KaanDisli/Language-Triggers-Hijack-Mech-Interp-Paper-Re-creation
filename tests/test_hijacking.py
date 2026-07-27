from __future__ import annotations

import math

import pytest

from trigger_heads.hijacking import (
    RepresentationConfig,
    build_hijacking_result,
    causal_selected_heads,
    exact_sign_flip_p_value,
    paired_adapter_delta,
    representation_metrics,
    summarize_metric_tensors,
)


torch = pytest.importorskip("torch")


def test_representation_metric_equations_are_exact():
    vectors = {
        "trigger": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        "fake": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        "english": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        "french": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }
    metrics = representation_metrics(vectors)
    ones = torch.ones(2, dtype=torch.float64)
    assert torch.allclose(metrics["raw_cosine_trigger_french"], ones)
    assert torch.allclose(metrics["raw_cosine_fake_french"], torch.zeros_like(ones))
    assert torch.allclose(metrics["raw_alignment_gain"], ones)
    assert torch.allclose(metrics["contrast_cosine"], ones)
    assert torch.allclose(metrics["norm_ratio"], ones)
    assert torch.allclose(metrics["hijacking_index"], torch.full_like(ones, 2.0))
    summary = summarize_metric_tensors(metrics)
    assert summary["hijacking_index"] == pytest.approx(2.0)
    assert summary["hijacking_index_std"] == pytest.approx(0.0)


def test_paired_delta_and_exact_sign_flip():
    base = {
        name: torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        for name in (
            "raw_cosine_trigger_french",
            "raw_cosine_fake_french",
            "raw_alignment_gain",
            "contrast_cosine",
            "norm_ratio",
            "hijacking_index",
        )
    }
    learned = {name: values + 1.0 for name, values in base.items()}
    delta = paired_adapter_delta(base, learned)
    assert delta["hijacking_index_gain"] == pytest.approx(1.0)
    assert delta["raw_alignment_gain_delta"] == pytest.approx(1.0)
    assert delta["selective_shift_toward_french"] == pytest.approx(0.0)
    assert delta["hijacking_index_gain_p_value_sign_flip"] == pytest.approx(0.25)
    assert exact_sign_flip_p_value(torch.tensor([1.0, -1.0])) == pytest.approx(1.0)


def test_causal_intersection_and_result_schema():
    causal = {
        "top_heads": {
            "trigger-fr": [
                {"layer": 17, "head": 2},
                {"layer": 17, "head": 0},
            ],
            "language-fr": [
                {"layer": 17, "head": 0},
                {"layer": 22, "head": 6},
            ],
        }
    }
    selected = causal_selected_heads(causal)
    assert set(selected) == {(17, 0)}
    metric = {
        "hijacking_index": 0.2,
        "raw_alignment_gain": 0.1,
        "contrast_cosine": 0.1,
    }
    delta = {
        "hijacking_index_gain": 0.3,
        "selective_shift_toward_french": 0.2,
    }
    row = {
        "layer": 0,
        "head": 0,
        "label": "L0H0",
        "selected": True,
        "selection_reasons": ["literal shared causal intersection"],
        "base": {**metric, "residual": metric, "native": metric},
        "learned": {**metric, "residual": metric, "native": metric},
        "adapter_delta": {**delta, "residual": delta, "native": delta},
    }
    result = build_hijacking_result(
        per_head=[row],
        layers=1,
        heads_per_layer=1,
        run={"examples": 8, "split": "test"},
        provenance={"test": True},
    )
    assert result["schema_version"] == "learned-trigger-hijacking-v1"
    assert result["grid"]["adapter_hijacking_index_gain"] == [[0.3]]
    assert result["summaries"]["selected_causal_heads"][0]["label"] == "L0H0"
    assert "range [-3,3]" in result["definitions"]["hijacking_index"]


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        RepresentationConfig(batch_size=0)
    with pytest.raises(ValueError):
        RepresentationConfig(epsilon=math.nan)
