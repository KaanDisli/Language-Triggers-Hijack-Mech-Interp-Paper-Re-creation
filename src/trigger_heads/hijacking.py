"""Base-versus-learned attention-head representation analysis.

The causal experiment intervenes on the input to each attention ``o_proj`` at
the final, newline-terminated prompt position.  This module captures the same
vectors for four aligned conditions:

``T``
    English context plus the genuine learned trigger.
``K``
    The same English context plus its tokenizer-matched fake trigger.
``E``
    The plain English context.
``F``
    The aligned natural-French context.

For each example and head we define a raw French-alignment advantage
``A_raw = cos(T, F) - cos(K, F)`` and a contrastive language-direction
alignment ``A_contrast = cos(T - K, F - E)``.  The operational hijacking index
is ``HI = A_raw + A_contrast``.  It is a signed, unclipped statistic with range
[-3, 3], not a probability or a causal effect estimate.

LoRA changed q/k/v *and* o projections.  Cross-model claims therefore use
per-query-head attention outputs projected through the corresponding model's ``o_proj`` slice
into the shared residual coordinate system.  Native pre-o_proj results are
retained as a diagnostic, but are not the primary cross-model evidence.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .learned_analysis import (
    AnalysisCorpus,
    build_condition_pairs,
    prepare_training_boundary_pairs,
)
from .modeling import ModelTopology, model_input_device
METRIC_NAMES = (
    "raw_cosine_trigger_french",
    "raw_cosine_fake_french",
    "raw_alignment_gain",
    "contrast_cosine",
    "norm_ratio",
    "hijacking_index",
)


@dataclass(frozen=True)
class RepresentationConfig:
    """Settings that affect captured examples or numerical summaries."""

    batch_size: int = 8
    fake_seed: int = 1931
    continuation_separator: str = "\n"
    max_prompt_tokens: int = 64
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.fake_seed, bool) or not isinstance(self.fake_seed, int):
            raise ValueError("fake_seed must be an integer")
        if not self.continuation_separator:
            raise ValueError("continuation_separator must not be empty")
        if self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")


def prepare_representation_conditions(
    tokenizer: Any,
    corpus: AnalysisCorpus,
    *,
    config: RepresentationConfig,
) -> tuple[dict[str, tuple[Any, ...]], dict[str, str]]:
    """Tokenize the four aligned prompt conditions at the training boundary."""

    pairs, assignments = build_condition_pairs(corpus, fake_seed=config.fake_seed)
    trigger = prepare_training_boundary_pairs(
        tokenizer,
        pairs["trigger-fr"],
        continuation_separator=config.continuation_separator,
        max_prompt_tokens=config.max_prompt_tokens,
        expected_trigger_tokens=corpus.expected_trigger_tokens,
    )
    language = prepare_training_boundary_pairs(
        tokenizer,
        pairs["language-fr"],
        continuation_separator=config.continuation_separator,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    conditions = {
        "trigger": tuple(row.clean_input_ids for row in trigger),
        "fake": tuple(row.corrupted_input_ids for row in trigger),
        "english": tuple(row.corrupted_input_ids for row in language),
        "french": tuple(row.clean_input_ids for row in language),
    }
    expected = len(corpus.examples)
    if any(len(rows) != expected for rows in conditions.values()):
        raise RuntimeError("representation conditions lost source alignment")
    return conditions, assignments


def capture_condition_head_vectors(
    model: Any,
    topology: ModelTopology,
    input_rows: Sequence[Any],
    *,
    pad_token_id: int,
    batch_size: int,
) -> Any:
    """Capture pre-o_proj per-query-head attention outputs at the final token.

    Returns a CPU float64 tensor shaped ``[example, layer, head, head_dim]``.
    """

    import torch

    if not input_rows:
        raise ValueError("at least one tokenized prompt is required")
    captured_batches: list[Any] = []
    state: dict[str, Any] = {"positions": None, "captures": None}

    def hook_for(layer: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise RuntimeError("attention output projection received no input")
            value = args[0]
            positions = state["positions"].to(value.device)
            row_index = torch.arange(value.shape[0], device=value.device)
            selected = value[row_index, positions]
            expected_width = topology.num_attention_heads * topology.head_dim
            if selected.shape[-1] != expected_width:
                raise RuntimeError(
                    f"L{layer} projection input width {selected.shape[-1]} != "
                    f"{expected_width}"
                )
            state["captures"][layer] = (
                selected.reshape(
                    value.shape[0], topology.num_attention_heads, topology.head_dim
                )
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )

        return hook

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with ExitStack() as stack:
            for layer, projection in enumerate(topology.attention_output_projections):
                handle = projection.register_forward_pre_hook(hook_for(layer))
                stack.callback(handle.remove)
            for start in range(0, len(input_rows), batch_size):
                rows = input_rows[start : start + batch_size]
                lengths = [int(row.numel()) for row in rows]
                width = max(lengths)
                device = model_input_device(model)
                input_ids = torch.full(
                    (len(rows), width),
                    int(pad_token_id),
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros(
                    (len(rows), width), dtype=torch.long, device=device
                )
                for index, values in enumerate(rows):
                    values = values.to(device=device, dtype=torch.long)
                    input_ids[index, : lengths[index]] = values
                    attention_mask[index, : lengths[index]] = 1
                state["positions"] = torch.tensor(
                    [length - 1 for length in lengths],
                    dtype=torch.long,
                    device=device,
                )
                state["captures"] = [None] * topology.num_layers
                with torch.inference_mode():
                    decoder = getattr(model, "model", None)
                    if decoder is not None:
                        decoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                    else:  # pragma: no cover - compatibility fallback
                        model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                if any(value is None for value in state["captures"]):
                    raise RuntimeError("not every decoder layer produced an activation")
                captured_batches.append(torch.stack(state["captures"], dim=1))
    finally:
        if was_training:
            model.train()
    return torch.cat(captured_batches, dim=0)


def capture_model_conditions(
    model: Any,
    topology: ModelTopology,
    conditions: Mapping[str, Sequence[Any]],
    *,
    pad_token_id: int,
    batch_size: int,
) -> dict[str, Any]:
    """Capture all required conditions with a stable source order."""

    required = ("trigger", "fake", "english", "french")
    missing = [name for name in required if name not in conditions]
    if missing:
        raise ValueError(f"missing representation conditions: {', '.join(missing)}")
    counts = {len(conditions[name]) for name in required}
    if len(counts) != 1:
        raise ValueError("representation conditions must have equal example counts")
    return {
        name: capture_condition_head_vectors(
            model,
            topology,
            conditions[name],
            pad_token_id=pad_token_id,
            batch_size=batch_size,
        )
        for name in required
    }


def analyze_captured_model(
    topology: ModelTopology,
    captured: Mapping[str, Any],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[list[dict[str, dict[str, float]]]], list[list[dict[str, dict[str, Any]]]]]:
    """Summarize native and residual-projected representations for every head.

    The first return value contains JSON-ready mean/std summaries.  The second
    retains per-example CPU tensors for aligned base-versus-learned deltas.
    """

    import torch

    for name in ("trigger", "fake", "english", "french"):
        value = captured.get(name)
        if value is None or tuple(value.shape[1:]) != (
            topology.num_layers,
            topology.num_attention_heads,
            topology.head_dim,
        ):
            raise ValueError(f"captured {name!r} tensor has an unexpected shape")
    summaries: list[list[dict[str, dict[str, float]]]] = []
    raw_values: list[list[dict[str, dict[str, Any]]]] = []
    for layer in range(topology.num_layers):
        weight = (
            topology.attention_output_projections[layer]
            .weight.detach()
            .to(device="cpu", dtype=torch.float64)
        )
        layer_summaries: list[dict[str, dict[str, float]]] = []
        layer_raw: list[dict[str, dict[str, Any]]] = []
        for head in range(topology.num_attention_heads):
            start, end = head * topology.head_dim, (head + 1) * topology.head_dim
            native = {name: captured[name][:, layer, head] for name in captured}
            projection = weight[:, start:end]
            residual = {
                name: values @ projection.transpose(0, 1)
                for name, values in native.items()
            }
            native_metrics = representation_metrics(native, epsilon=epsilon)
            residual_metrics = representation_metrics(residual, epsilon=epsilon)
            layer_raw.append({"native": native_metrics, "residual": residual_metrics})
            layer_summaries.append(
                {
                    "native": summarize_metric_tensors(native_metrics),
                    "residual": summarize_metric_tensors(residual_metrics),
                }
            )
        summaries.append(layer_summaries)
        raw_values.append(layer_raw)
    return summaries, raw_values


def representation_metrics(
    vectors: Mapping[str, Any], *, epsilon: float = 1e-12
) -> dict[str, Any]:
    """Return one value per example for the six declared head metrics."""

    required = {"trigger", "fake", "english", "french"}
    if set(vectors) != required:
        raise ValueError("representation metric inputs must be T/K/E/F exactly")
    trigger, fake = vectors["trigger"], vectors["fake"]
    english, french = vectors["english"], vectors["french"]
    trigger_french = _row_cosine(trigger, french, epsilon)
    fake_french = _row_cosine(fake, french, epsilon)
    raw_gain = trigger_french - fake_french
    trigger_contrast = trigger - fake
    language_contrast = french - english
    contrast_cosine = _row_cosine(trigger_contrast, language_contrast, epsilon)
    norm_ratio = trigger_contrast.norm(dim=-1) / (
        language_contrast.norm(dim=-1) + epsilon
    )
    return {
        "raw_cosine_trigger_french": trigger_french,
        "raw_cosine_fake_french": fake_french,
        "raw_alignment_gain": raw_gain,
        "contrast_cosine": contrast_cosine,
        "norm_ratio": norm_ratio,
        "hijacking_index": raw_gain + contrast_cosine,
    }


def summarize_metric_tensors(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert per-example metric tensors to finite mean/std fields."""

    import torch

    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = metrics[name].detach().to(device="cpu", dtype=torch.float64)
        mean = float(values.mean())
        std = float(values.std(unbiased=False))
        if not math.isfinite(mean) or not math.isfinite(std):
            raise RuntimeError(f"non-finite representation metric {name}")
        result[name] = mean
        result[f"{name}_std"] = std
    return result


def combine_model_results(
    base_summary: Sequence[Sequence[Mapping[str, Mapping[str, float]]]],
    base_raw: Sequence[Sequence[Mapping[str, Mapping[str, Any]]]],
    learned_summary: Sequence[Sequence[Mapping[str, Mapping[str, float]]]],
    learned_raw: Sequence[Sequence[Mapping[str, Mapping[str, Any]]]],
    *,
    selected_heads: Mapping[tuple[int, int], Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Create report-ready per-head rows with paired adapter deltas."""

    selected_heads = selected_heads or {}
    rows: list[dict[str, Any]] = []
    for layer in range(len(base_summary)):
        for head in range(len(base_summary[layer])):
            spaces: dict[str, dict[str, float]] = {}
            for space in ("residual", "native"):
                spaces[space] = paired_adapter_delta(
                    base_raw[layer][head][space], learned_raw[layer][head][space]
                )
            base = {
                **dict(base_summary[layer][head]["residual"]),
                "residual": dict(base_summary[layer][head]["residual"]),
                "native": dict(base_summary[layer][head]["native"]),
            }
            learned = {
                **dict(learned_summary[layer][head]["residual"]),
                "residual": dict(learned_summary[layer][head]["residual"]),
                "native": dict(learned_summary[layer][head]["native"]),
            }
            adapter = {
                **spaces["residual"],
                "residual": spaces["residual"],
                "native": spaces["native"],
            }
            reasons = list(selected_heads.get((layer, head), ()))
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "label": f"L{layer}H{head}",
                    "selected": bool(reasons),
                    "selection_reasons": reasons,
                    "base": base,
                    "learned": learned,
                    "adapter_delta": adapter,
                }
            )
    return rows


def paired_adapter_delta(
    base: Mapping[str, Any], learned: Mapping[str, Any]
) -> dict[str, float]:
    """Summarize aligned learned-minus-base differences for one head/space."""

    import torch

    delta_names = {
        "raw_cosine_trigger_french": "raw_cosine_trigger_french_delta",
        "raw_cosine_fake_french": "raw_cosine_fake_french_delta",
        "raw_alignment_gain": "raw_alignment_gain_delta",
        "contrast_cosine": "contrast_cosine_delta",
        "norm_ratio": "norm_ratio_delta",
        "hijacking_index": "hijacking_index_gain",
    }
    result: dict[str, float] = {}
    deltas: dict[str, Any] = {}
    for source, destination in delta_names.items():
        values = learned[source].detach().to(torch.float64) - base[source].detach().to(
            torch.float64
        )
        deltas[source] = values
        result[destination] = float(values.mean())
        result[f"{destination}_std"] = float(values.std(unbiased=False))
    result["trigger_shift_toward_french"] = result[
        "raw_cosine_trigger_french_delta"
    ]
    result["fake_shift_toward_french"] = result["raw_cosine_fake_french_delta"]
    result["selective_shift_toward_french"] = (
        result["trigger_shift_toward_french"]
        - result["fake_shift_toward_french"]
    )
    result["contrast_alignment_gain"] = result["contrast_cosine_delta"]
    result["hijacking_index_gain_p_value_sign_flip"] = exact_sign_flip_p_value(
        deltas["hijacking_index"]
    )
    if any(not math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite paired adapter representation delta")
    return result


def exact_sign_flip_p_value(values: Any) -> float:
    """Exact two-sided paired sign-flip p-value for at most 20 observations."""

    import torch

    vector = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    count = int(vector.numel())
    if count == 0:
        raise ValueError("sign-flip test requires observations")
    if count > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 observations")
    observed = abs(float(vector.mean()))
    extreme = 0
    total = 1 << count
    for bits in range(total):
        signs = torch.tensor(
            [1.0 if bits & (1 << index) else -1.0 for index in range(count)],
            dtype=torch.float64,
        )
        statistic = abs(float((vector * signs).mean()))
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / total


def causal_selected_heads(causal: Mapping[str, Any]) -> dict[tuple[int, int], list[str]]:
    """Extract the literal top-k intersection and record transparent reasons."""

    top = causal.get("top_heads")
    if not isinstance(top, Mapping):
        raise ValueError("causal artifact is missing top_heads")

    def identities(condition: str) -> set[tuple[int, int]]:
        rows = top.get(condition)
        if not isinstance(rows, list):
            raise ValueError(f"causal artifact is missing {condition!r} top heads")
        return {(int(row["layer"]), int(row["head"])) for row in rows}

    trigger = identities("trigger-fr")
    language = identities("language-fr")
    selected: dict[tuple[int, int], list[str]] = {}
    for identity in sorted(trigger & language):
        selected[identity] = [
            "trigger-fr top-k causal head",
            "language-fr top-k causal head",
            "literal shared causal intersection",
        ]
    return selected


def build_hijacking_result(
    *,
    per_head: Sequence[Mapping[str, Any]],
    layers: int,
    heads_per_layer: int,
    run: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the compact JSON payload around computed rows."""

    selected = [dict(row) for row in per_head if row.get("selected") is True]
    top_gain = sorted(
        (dict(row) for row in per_head),
        key=lambda row: float(row["adapter_delta"]["hijacking_index_gain"]),
        reverse=True,
    )[:10]
    top_learned = sorted(
        (dict(row) for row in per_head),
        key=lambda row: float(row["learned"]["hijacking_index"]),
        reverse=True,
    )[:10]

    def grid(path: Sequence[str]) -> list[list[float]]:
        values = [[0.0] * heads_per_layer for _ in range(layers)]
        for row in per_head:
            value: Any = row
            for key in path:
                value = value[key]
            values[int(row["layer"])][int(row["head"])] = float(value)
        return values

    result = {
        "schema_version": "learned-trigger-hijacking-v1",
        "status": "complete",
        "run": dict(run),
        "grid": {
            "layers": layers,
            "heads_per_layer": heads_per_layer,
            "size": layers * heads_per_layer,
            "learned_hijacking_index": grid(("learned", "hijacking_index")),
            "adapter_hijacking_index_gain": grid(
                ("adapter_delta", "hijacking_index_gain")
            ),
            "adapter_selective_french_shift": grid(
                ("adapter_delta", "selective_shift_toward_french")
            ),
        },
        "per_head": list(per_head),
        "summaries": {
            "selected_causal_heads": selected,
            "top_hijacking_gain": top_gain,
            "top_learned_hijacking": top_learned,
            "mean_base_hijacking_index": _mean_rows(per_head, "base", "hijacking_index"),
            "mean_learned_hijacking_index": _mean_rows(
                per_head, "learned", "hijacking_index"
            ),
            "mean_adapter_hijacking_index_gain": _mean_rows(
                per_head, "adapter_delta", "hijacking_index_gain"
            ),
        },
        "definitions": {
            "T": "English context plus genuine trigger, ending at the training newline",
            "K": "same English context plus assigned tokenizer-matched fake trigger",
            "E": "plain aligned English context",
            "F": "aligned natural-French context",
            "raw_alignment_gain": "A_raw = cos(T,F) - cos(K,F), averaged per example",
            "contrast_cosine": "A_contrast = cos(T-K,F-E), averaged per example",
            "hijacking_index": (
                "HI = A_raw + A_contrast; signed and unclipped with mathematical "
                "range [-3,3], not a probability"
            ),
            "residual": (
                "pre-o_proj per-query-head attention outputs projected through that model's matching "
                "o_proj column slice; primary cross-model coordinate system"
            ),
            "native": "pre-o_proj per-query-head attention output; within-model diagnostic",
            "adapter_delta": "paired learned-minus-base difference over identical sources",
        },
        "limitations": [
            "Only eight source-disjoint held-out examples are available.",
            "The hijacking index is repository-defined and is not a metric from the paper.",
            "Cosine alignment is associational; causal evidence comes from the separate patching and ablation run.",
            "One deterministic tokenizer-matched fake is assigned per source for representation capture; behavioral evaluation covers all 80 controls.",
            "Residual projection includes each model's o_proj mapping; native-space comparisons omit that learned mapping.",
            "Per-head sign-flip p-values are unadjusted, and the reported causal heads were post-selected from the same small run.",
        ],
        "provenance": dict(provenance),
    }
    _validate_json_finite(result)
    return result


def _row_cosine(left: Any, right: Any, epsilon: float) -> Any:
    numerator = (left * right).sum(dim=-1)
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return numerator / denominator.clamp_min(epsilon)


def _mean_rows(rows: Sequence[Mapping[str, Any]], section: str, metric: str) -> float:
    return sum(float(row[section][metric]) for row in rows) / len(rows)


def _validate_json_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has non-string JSON key")
            _validate_json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is non-finite")
