"""Representational similarity for overlapping trigger/language heads."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .metrics import cosine_similarity


def head_cosine_matrix(
    trigger_means: Mapping[str, Any],
    language_means: Mapping[str, Any],
    *,
    layer: int,
    head: int,
    languages: Sequence[str] = ("fr", "de"),
) -> list[list[float]]:
    """Compare condition means for one head as in Appendix J.

    Inputs are tensors shaped ``[layer, query_head, head_dim]`` produced by the
    head-patching clean pass. Rows are trigger conditions; columns are natural
    language conditions.
    """

    matrix: list[list[float]] = []
    for trigger_language in languages:
        trigger = _head_vector(trigger_means, trigger_language, layer, head)
        row: list[float] = []
        for natural_language in languages:
            natural = _head_vector(language_means, natural_language, layer, head)
            row.append(cosine_similarity(trigger, natural))
        matrix.append(row)
    return matrix


def projected_head_vector(
    topology: Any,
    *,
    layer: int,
    head: int,
    head_vector: Any,
) -> Any:
    """Project a pre-W_O head vector into residual space when desired."""

    projection = topology.attention_output_projections[layer]
    start = head * topology.head_dim
    end = start + topology.head_dim
    weight = projection.weight[:, start:end].detach().to(device="cpu", dtype=head_vector.dtype)
    return weight @ head_vector.detach().cpu()


def _head_vector(
    values: Mapping[str, Any], language: str, layer: int, head: int
) -> Any:
    if language not in values:
        raise ValueError(f"Missing condition means for {language!r}")
    tensor = values[language]
    if len(tensor.shape) != 3:
        raise ValueError("Mean activation tensor must have [layer, head, head_dim] shape")
    if not 0 <= layer < tensor.shape[0] or not 0 <= head < tensor.shape[1]:
        raise ValueError(f"L{layer}H{head} is outside the mean activation tensor")
    return tensor[layer, head]

