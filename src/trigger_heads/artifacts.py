"""Stable, inspectable experiment artifact formats."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ablation import AblationPoint
from .metrics import expected_jaccard, hypergeometric_upper_tail, jaccard, rank_top_heads
from .patching import HeadPatchingOutput, LayerPatchingOutput


def save_head_patching(
    path: str | Path,
    output: HeadPatchingOutput,
    *,
    condition: str,
    model_name: str,
    top_k: int = 10,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save scores to JSON and clean means to a sibling ``.means.pt`` file."""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    means_path = destination.with_suffix(".means.pt")
    metadata_dict = dict(metadata or {})
    torch.save(
        {
            "artifact_type": "head_activation_means",
            "condition": condition,
            "model": model_name,
            "num_examples": output.num_examples,
            "mean_activations": output.mean_clean_activations.detach().cpu(),
            "metadata": metadata_dict,
        },
        means_path,
    )
    top = rank_top_heads(output.scores, min(top_k, output.scores.numel()))
    payload = {
        "artifact_type": "head_patching",
        "condition": condition,
        "model": model_name,
        "num_examples": output.num_examples,
        "baseline_mean_logprob": output.baseline_mean_logprob,
        "scores": output.scores.detach().cpu().tolist(),
        "top_heads": [
            {
                "layer": layer,
                "head": head,
                "delta_logprob": float(output.scores[layer, head]),
            }
            for layer, head in top
        ],
        "mean_activations_file": means_path.name,
        "intervention": {
            "hook": "attention output projection input (pre-W_O)",
            "position": "last non-padding prompt token",
            "clean_reduction": "condition mean",
        },
        "metadata": metadata_dict,
    }
    _write_json(destination, payload)
    return destination


def save_layer_patching(
    path: str | Path,
    output: LayerPatchingOutput,
    *,
    condition: str,
    model_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "layer_token_patching",
        "condition": condition,
        "model": model_name,
        "num_examples": output.num_examples,
        "trigger_tokens": output.trigger_tokens,
        "baseline_mean_logprob": output.baseline_mean_logprob,
        "scores": output.scores.detach().cpu().tolist(),
        "intervention": {
            "hook": "decoder block output (post-block residual)",
            "position": "one aligned trigger token",
            "clean_reduction": "same-example activation",
        },
        "metadata": dict(metadata or {}),
    }
    _write_json(destination, payload)
    return destination


def save_ablation(
    path: str | Path,
    points: Sequence[AblationPoint],
    *,
    condition: str,
    model_name: str,
    ordered_heads: Sequence[tuple[int, int]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        destination,
        {
            "artifact_type": "head_ablation",
            "condition": condition,
            "model": model_name,
            "ordered_heads": [
                {"layer": layer, "head": head} for layer, head in ordered_heads
            ],
            "points": [asdict(point) for point in points],
            "intervention": {
                "hook": "attention output projection input (pre-W_O)",
                "positions": "all prompt and teacher-forced continuation positions",
                "value": "zero",
                "ppl_reduction": "token-weighted corpus perplexity",
            },
            "metadata": dict(metadata or {}),
        },
    )
    return destination


def load_head_scores(path: str | Path) -> list[list[float]]:
    payload = load_head_artifact(path)
    scores = payload.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"{path} has no score grid")
    return scores


def load_head_artifact(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("artifact_type") != "head_patching":
        raise ValueError(f"{path} is not a head-patching artifact")
    scores = payload.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"{path} has no score grid")
    for required in ("condition", "model", "num_examples", "metadata"):
        if required not in payload:
            raise ValueError(f"{path} is missing provenance field {required!r}")
    return payload


def load_mean_activations(path: str | Path) -> Any:
    return load_mean_artifact(path)["mean_activations"]


def load_mean_artifact(path: str | Path) -> dict[str, Any]:
    import torch

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("artifact_type") != "head_activation_means":
        raise ValueError(
            f"{source} is not a provenance-bearing head-activation artifact"
        )
    for required in (
        "condition",
        "model",
        "num_examples",
        "mean_activations",
        "metadata",
    ):
        if required not in payload:
            raise ValueError(f"{source} is missing provenance field {required!r}")
    return payload


def assert_compatible_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    expected_conditions: Mapping[str, str] | None = None,
    expected_model: str | None = None,
    expected_num_examples: int | None = None,
    expected_dataset_sha256: str | None = None,
    expected_resolved_model_commit: str | None = None,
    expected_trigger_set_ids: Mapping[str, Any] | None = None,
) -> None:
    """Reject scientifically invalid comparisons across models/data/runs."""

    if not artifacts:
        raise ValueError("No artifacts were supplied")
    reference_name = next(iter(artifacts))
    reference = artifacts[reference_name]
    reference_metadata = reference.get("metadata", {})
    if not isinstance(reference_metadata, Mapping):
        raise ValueError(f"Artifact {reference_name!r} metadata must be an object")
    fields = {
        "model": reference.get("model"),
        "num_examples": reference.get("num_examples"),
        "model_revision": reference_metadata.get("model_revision"),
        "resolved_model_commit": reference_metadata.get("resolved_model_commit"),
        "dataset_sha256": reference_metadata.get("dataset_sha256"),
        "seed": reference_metadata.get("seed"),
        "trigger_set_ids": reference_metadata.get("trigger_set_ids"),
    }
    for name, artifact in artifacts.items():
        metadata = artifact.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Artifact {name!r} metadata must be an object")
        values = {
            "model": artifact.get("model"),
            "num_examples": artifact.get("num_examples"),
            "model_revision": metadata.get("model_revision"),
            "resolved_model_commit": metadata.get("resolved_model_commit"),
            "dataset_sha256": metadata.get("dataset_sha256"),
            "seed": metadata.get("seed"),
            "trigger_set_ids": metadata.get("trigger_set_ids"),
        }
        for field, reference_value in fields.items():
            if values[field] != reference_value:
                raise ValueError(
                    f"Artifact {name!r} has incompatible {field}: "
                    f"{values[field]!r} != {reference_value!r} from {reference_name!r}"
                )
        if expected_conditions is not None:
            expected = expected_conditions.get(name)
            if expected is not None and artifact.get("condition") != expected:
                raise ValueError(
                    f"Artifact {name!r} condition is {artifact.get('condition')!r}; "
                    f"expected {expected!r}"
                )
    if expected_model is not None and fields["model"] != expected_model:
        raise ValueError(
            f"Artifacts target model {fields['model']!r}, expected {expected_model!r}"
        )
    if expected_num_examples is not None and fields["num_examples"] != expected_num_examples:
        raise ValueError(
            f"Artifacts contain {fields['num_examples']} examples, expected "
            f"{expected_num_examples}"
        )
    if (
        expected_dataset_sha256 is not None
        and fields["dataset_sha256"] != expected_dataset_sha256
    ):
        raise ValueError("Artifact dataset hash does not match the configured JSONL")
    if (
        expected_resolved_model_commit is not None
        and fields["resolved_model_commit"] != expected_resolved_model_commit
    ):
        raise ValueError(
            "Artifact model commit does not match the currently loaded checkpoint"
        )
    if (
        expected_trigger_set_ids is not None
        and fields["trigger_set_ids"] != dict(expected_trigger_set_ids)
    ):
        raise ValueError(
            "Artifact trigger-set provenance does not match the configured set IDs"
        )


def overlap_report(
    named_scores: Mapping[str, Any], *, top_k: int = 10
) -> dict[str, Any]:
    """Build top-set overlap values and Appendix L chance statistics."""

    if len(named_scores) < 2:
        raise ValueError("At least two score artifacts are required")
    shapes = {_shape(scores) for scores in named_scores.values()}
    if len(shapes) != 1:
        raise ValueError("Every score grid must have the same model shape")
    layers, heads_per_layer = shapes.pop()
    universe = layers * heads_per_layer
    rankings = {
        name: rank_top_heads(scores, top_k) for name, scores in named_scores.items()
    }
    names = list(rankings)
    matrix: list[list[float]] = []
    p_values: list[list[float]] = []
    intersections: list[list[int]] = []
    for row_name in names:
        row: list[float] = []
        p_row: list[float] = []
        x_row: list[int] = []
        for column_name in names:
            first = rankings[row_name]
            second = rankings[column_name]
            intersection = len(set(first).intersection(second))
            row.append(jaccard(first, second))
            x_row.append(intersection)
            p_row.append(
                hypergeometric_upper_tail(universe, top_k, intersection)
            )
        matrix.append(row)
        p_values.append(p_row)
        intersections.append(x_row)
    return {
        "artifact_type": "head_overlap",
        "conditions": names,
        "top_k": top_k,
        "universe_size": universe,
        "expected_jaccard": expected_jaccard(universe, top_k),
        "top_heads": {
            name: [{"layer": layer, "head": head} for layer, head in ranking]
            for name, ranking in rankings.items()
        },
        "jaccard": matrix,
        "intersection": intersections,
        "p_value_upper_tail": p_values,
    }


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, dict(payload))
    return destination


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"Could not read artifact {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact {source} must contain a JSON object")
    return payload


def _shape(scores: Any) -> tuple[int, int]:
    if hasattr(scores, "shape"):
        shape = tuple(int(value) for value in scores.shape)
    else:
        shape = (len(scores), len(scores[0]) if scores else 0)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Score grid must be a non-empty 2-D matrix")
    return shape


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
