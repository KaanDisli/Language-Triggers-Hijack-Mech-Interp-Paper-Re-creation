"""Command-line entry points for the original Gaperon protocol track.

The learned Qwen/LoRA track has purpose-built scripts in ``scripts/`` because it
also includes training and behavioral evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from .ablation import (
    evaluate_ablation_curve,
    joint_rank_order,
    prepare_continuations,
    strict_overlap_order,
)
from .artifacts import (
    assert_compatible_artifacts,
    load_head_artifact,
    load_mean_artifact,
    overlap_report,
    save_ablation,
    save_head_patching,
    save_json,
    save_layer_patching,
)
from .config import ExperimentConfig
from .data import load_jsonl, trigger_token_lengths, validate_fake_trigger_lengths
from .metrics import rank_top_heads
from .modeling import load_model_bundle
from .patching import (
    prepare_prompt_pairs,
    run_head_activation_patching,
    run_layer_token_patching,
)
from .prompts import (
    assign_fake_triggers,
    build_language_pair,
    build_trigger_pair,
)
from .representations import head_cosine_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trigger-heads",
        description="Reproduce the causal analyses in arXiv:2602.10382",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="validate parallel JSONL")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument(
        "--with-tokenizer",
        action="store_true",
        help="also load the model tokenizer and validate all fake triggers",
    )
    validate.set_defaults(func=_cmd_validate_data)

    inspect = subparsers.add_parser("inspect-model", help="show resolved hook geometry")
    inspect.add_argument("--config", required=True, type=Path)
    inspect.set_defaults(func=_cmd_inspect_model)

    head = subparsers.add_parser("head-patch", help="run one head-wise condition")
    head.add_argument("--config", required=True, type=Path)
    head.add_argument("--condition", choices=("trigger", "language"), required=True)
    head.add_argument("--language", choices=("fr", "de", "it", "es"), required=True)
    head.add_argument("--output", type=Path)
    head.set_defaults(func=_cmd_head_patch)

    layer = subparsers.add_parser(
        "layer-patch", help="localize a French/German trigger over layer and token"
    )
    layer.add_argument("--config", required=True, type=Path)
    layer.add_argument("--language", choices=("fr", "de"), required=True)
    layer.add_argument("--output", type=Path)
    layer.set_defaults(func=_cmd_layer_patch)

    overlap = subparsers.add_parser("overlap", help="compare saved head score grids")
    overlap.add_argument(
        "artifacts",
        nargs="+",
        metavar="NAME=PATH",
        help="named head-patching JSON artifacts",
    )
    overlap.add_argument("--top-k", type=int, default=10)
    overlap.add_argument("--output", required=True, type=Path)
    overlap.set_defaults(func=_cmd_overlap)

    cosine = subparsers.add_parser(
        "cosine", help="Appendix J cosine matrix from saved mean tensors"
    )
    cosine.add_argument("--trigger-fr", required=True, type=Path)
    cosine.add_argument("--trigger-de", required=True, type=Path)
    cosine.add_argument("--language-fr", required=True, type=Path)
    cosine.add_argument("--language-de", required=True, type=Path)
    cosine.add_argument("--head", required=True, help="head such as L27H17")
    cosine.add_argument("--output", required=True, type=Path)
    cosine.set_defaults(func=_cmd_cosine)

    ablate = subparsers.add_parser(
        "ablate", help="continuation-PPL validation of overlap-ranked heads"
    )
    ablate.add_argument("--config", required=True, type=Path)
    ablate.add_argument("--language", choices=("fr", "de"), required=True)
    ablate.add_argument("--setup", choices=("trigger", "language"), required=True)
    ablate.add_argument("--trigger-scores", required=True, type=Path)
    ablate.add_argument("--language-scores", required=True, type=Path)
    ablate.add_argument("--output", type=Path)
    ablate.add_argument("--max-heads", type=int)
    ablate.add_argument(
        "--overlap-policy",
        choices=("strict", "joint-rank"),
        default="strict",
        help=(
            "strict uses the literal top-k intersection; joint-rank supplies the "
            "paper-like longer curve that the manuscript leaves underspecified"
        ),
    )
    ablate.set_defaults(func=_cmd_ablate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_validate_data(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_json_file(args.config)
    examples = _examples(config)
    result: dict[str, Any] = {
        "data_path": str(config.data_path),
        "examples": len(examples),
        "tokenizer_validation": False,
    }
    if args.with_tokenizer:
        bundle = _bundle(config)
        result["tokenizer_validation"] = True
        result["model"] = config.model.name_or_path
        result["triggers"] = _validate_all_configured_triggers(config, bundle.tokenizer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_inspect_model(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_json_file(args.config)
    bundle = _bundle(config)
    topology = bundle.topology
    print(
        json.dumps(
            {
                "model": config.model.name_or_path,
                "layers": topology.num_layers,
                "query_heads": topology.num_attention_heads,
                "head_dim": topology.head_dim,
                "pre_output_projection_width": (
                    topology.num_attention_heads * topology.head_dim
                ),
                "total_layer_heads": topology.num_heads_total,
            },
            indent=2,
        )
    )
    return 0


def _cmd_head_patch(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_json_file(args.config)
    if args.condition == "trigger" and args.language not in {"fr", "de"}:
        raise ValueError("Trigger conditions exist only for fr and de")
    examples = _examples(config)
    bundle = _bundle(config)
    pairs = _condition_pairs(config, bundle.tokenizer, examples, args.condition, args.language)
    prepared = prepare_prompt_pairs(
        bundle.tokenizer,
        pairs,
        continuation_separator=config.runtime.continuation_separator,
        trigger_separator=config.runtime.trigger_separator,
        max_prompt_tokens=_sequence_limit(config, bundle),
        expected_trigger_tokens=_expected_trigger_tokens(
            config, args.condition, args.language
        ),
    )
    output = run_head_activation_patching(
        bundle.model,
        bundle.topology,
        prepared,
        pad_token_id=bundle.tokenizer.pad_token_id,
        batch_size=config.runtime.batch_size,
        progress=_progress,
    )
    condition = f"{args.condition}-{args.language}"
    destination = args.output or config.output_dir / f"{condition}.json"
    save_head_patching(
        destination,
        output,
        condition=condition,
        model_name=config.model.name_or_path,
        top_k=config.runtime.top_k,
        metadata=_metadata(config, args.condition, args.language, bundle=bundle),
    )
    print(f"saved {destination}")
    return 0


def _cmd_layer_patch(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_json_file(args.config)
    examples = _examples(config)
    bundle = _bundle(config)
    pairs = _condition_pairs(config, bundle.tokenizer, examples, "trigger", args.language)
    prepared = prepare_prompt_pairs(
        bundle.tokenizer,
        pairs,
        continuation_separator=config.runtime.continuation_separator,
        trigger_separator=config.runtime.trigger_separator,
        max_prompt_tokens=_sequence_limit(config, bundle),
        expected_trigger_tokens=_expected_trigger_tokens(
            config, "trigger", args.language
        ),
    )
    output = run_layer_token_patching(
        bundle.model,
        bundle.topology,
        prepared,
        pad_token_id=bundle.tokenizer.pad_token_id,
        batch_size=config.runtime.layer_batch_size,
        progress=_progress,
    )
    condition = f"trigger-{args.language}"
    destination = args.output or config.output_dir / f"layer-{condition}.json"
    save_layer_patching(
        destination,
        output,
        condition=condition,
        model_name=config.model.name_or_path,
        metadata=_metadata(config, "trigger", args.language, bundle=bundle),
    )
    print(f"saved {destination}")
    return 0


def _cmd_overlap(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    artifacts: dict[str, Any] = {}
    for value in args.artifacts:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or name in artifacts:
            raise ValueError(f"Condition name is empty or duplicated in {value!r}")
        artifact = load_head_artifact(path)
        if artifact.get("condition") != name:
            raise ValueError(
                f"Artifact {path!r} contains condition {artifact.get('condition')!r}, "
                f"not the requested label {name!r}"
            )
        artifacts[name] = artifact
    assert_compatible_artifacts(artifacts)
    named = {name: artifact["scores"] for name, artifact in artifacts.items()}
    report = overlap_report(named, top_k=args.top_k)
    reference = next(iter(artifacts.values()))
    report["model"] = reference["model"]
    report["num_examples"] = reference["num_examples"]
    report["metadata"] = reference["metadata"]
    save_json(args.output, report)
    print(f"saved {args.output}")
    return 0


def _cmd_cosine(args: argparse.Namespace) -> int:
    layer, head = _parse_head(args.head)
    loaded = {
        "trigger-fr": load_mean_artifact(args.trigger_fr),
        "trigger-de": load_mean_artifact(args.trigger_de),
        "language-fr": load_mean_artifact(args.language_fr),
        "language-de": load_mean_artifact(args.language_de),
    }
    assert_compatible_artifacts(
        loaded,
        expected_conditions={name: name for name in loaded},
    )
    trigger = {
        "fr": loaded["trigger-fr"]["mean_activations"],
        "de": loaded["trigger-de"]["mean_activations"],
    }
    language = {
        "fr": loaded["language-fr"]["mean_activations"],
        "de": loaded["language-de"]["mean_activations"],
    }
    matrix = head_cosine_matrix(trigger, language, layer=layer, head=head)
    reference = loaded["trigger-fr"]
    save_json(
        args.output,
        {
            "artifact_type": "head_representation_cosine",
            "head": {"layer": layer, "head": head},
            "rows": ["trigger-fr", "trigger-de"],
            "columns": ["language-fr", "language-de"],
            "cosine": matrix,
            "activation_space": "pre-W_O query-head output at final prompt token",
            "model": reference["model"],
            "num_examples": reference["num_examples"],
            "metadata": reference["metadata"],
        },
    )
    print(f"saved {args.output}")
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_json_file(args.config)
    examples = _examples(config)
    score_artifacts = {
        "trigger": load_head_artifact(args.trigger_scores),
        "language": load_head_artifact(args.language_scores),
    }
    assert_compatible_artifacts(
        score_artifacts,
        expected_conditions={
            "trigger": f"trigger-{args.language}",
            "language": f"language-{args.language}",
        },
        expected_model=config.model.name_or_path,
        expected_num_examples=len(examples),
        expected_dataset_sha256=_file_sha256(config.data_path),
        expected_trigger_set_ids=_trigger_set_ids(config),
    )
    trigger_scores = score_artifacts["trigger"]["scores"]
    language_scores = score_artifacts["language"]["scores"]
    bundle = _bundle(config)
    _validate_score_grid(trigger_scores, bundle, "trigger")
    _validate_score_grid(language_scores, bundle, "language")
    resolved_commit = getattr(bundle.model.config, "_commit_hash", None)
    if isinstance(resolved_commit, str):
        assert_compatible_artifacts(
            score_artifacts,
            expected_resolved_model_commit=resolved_commit,
        )
    top_k = config.runtime.top_k
    trigger_ranking = rank_top_heads(trigger_scores, top_k)
    language_ranking = rank_top_heads(language_scores, top_k)
    if args.overlap_policy == "strict":
        ordered = strict_overlap_order(trigger_ranking, language_ranking)
        policy_note = "literal top-k intersection ordered by mean rank"
    else:
        ordered = joint_rank_order(
            trigger_scores, language_scores, limit=config.runtime.top_k
        )
        policy_note = (
            "top-k by mean full-grid rank across trigger/language; reconstruction "
            "for unexplained Fig. 14 curve length"
        )
    if not ordered:
        raise ValueError("The configured top-head sets have no literal overlap")

    pairs = _condition_pairs(config, bundle.tokenizer, examples, args.setup, args.language)
    continuations = prepare_continuations(
        bundle.tokenizer,
        pairs,
        prompt_side="clean",
        continuation_separator=config.runtime.continuation_separator,
        max_sequence_tokens=_sequence_limit(config, bundle),
        truncation=config.runtime.continuation_truncation,
    )
    points = evaluate_ablation_curve(
        bundle.model,
        bundle.topology,
        continuations,
        ordered,
        pad_token_id=bundle.tokenizer.pad_token_id,
        batch_size=config.runtime.batch_size,
        random_repeats=config.runtime.random_ablation_repeats,
        seed=config.runtime.seed,
        max_heads=args.max_heads,
        progress=_progress,
    )
    condition = f"{args.setup}-{args.language}"
    destination = args.output or config.output_dir / f"ablation-{condition}.json"
    save_ablation(
        destination,
        points,
        condition=condition,
        model_name=config.model.name_or_path,
        ordered_heads=ordered,
        metadata={
            **_metadata(config, args.setup, args.language, bundle=bundle),
            "overlap_policy": args.overlap_policy,
            "overlap_policy_detail": policy_note,
            "random_heads": "uniform, resampled per example",
            "random_repeats": config.runtime.random_ablation_repeats,
        },
    )
    print(f"saved {destination}")
    return 0


def _examples(config: ExperimentConfig) -> list[Any]:
    examples = load_jsonl(config.data_path)
    if config.runtime.max_examples is not None:
        examples = examples[: config.runtime.max_examples]
    if not examples:
        raise ValueError(f"No examples loaded from {config.data_path}")
    return examples


def _bundle(config: ExperimentConfig) -> Any:
    model = config.model
    return load_model_bundle(
        model.name_or_path,
        revision=model.revision,
        dtype=model.dtype,
        device_map=model.device_map,
        trust_remote_code=model.trust_remote_code,
        attn_implementation=model.attn_implementation,
    )


def _condition_pairs(
    config: ExperimentConfig,
    tokenizer: Any,
    examples: Sequence[Any],
    condition: str,
    language: str,
) -> list[Any]:
    if condition == "language":
        return [build_language_pair(example, target_language=language) for example in examples]
    if condition != "trigger":
        raise ValueError(f"Unknown condition {condition!r}")
    trigger = config.trigger_for(language)
    _validate_trigger(
        language,
        trigger,
        tokenizer,
        trigger_separator=config.runtime.trigger_separator,
    )
    fakes = assign_fake_triggers(examples, trigger.fake, seed=config.runtime.seed)
    return [
        build_trigger_pair(
            example,
            target_language=language,
            genuine_trigger=trigger.genuine or "",
            fake_trigger=fake,
            trigger_separator=config.runtime.trigger_separator,
        )
        for example, fake in zip(examples, fakes)
    ]


def _validate_trigger(
    language: str,
    trigger: Any,
    tokenizer: Any,
    *,
    trigger_separator: str,
) -> dict[str, Any]:
    trigger.require_complete(language)
    assert trigger.genuine is not None
    if len(trigger.genuine.split()) != 3:
        raise ValueError(f"The paper's {language} genuine trigger must contain three words")
    profiles = validate_fake_trigger_lengths(
        tokenizer,
        trigger.genuine,
        trigger.fake,
        expected_count=10,
        leading_separator=trigger_separator,
    )
    genuine_profile = trigger_token_lengths(
        tokenizer,
        trigger.genuine,
        leading_separator=trigger_separator,
    )
    if (
        trigger.expected_total_tokens is not None
        and genuine_profile.total != trigger.expected_total_tokens
    ):
        raise ValueError(
            f"Configured {language} trigger has {genuine_profile.total} tokens, "
            f"expected {trigger.expected_total_tokens}"
        )
    return {
        "fake_count": len(profiles),
        "total_tokens": genuine_profile.total,
        "per_word_tokens": list(genuine_profile.per_word),
    }


def _validate_all_configured_triggers(
    config: ExperimentConfig, tokenizer: Any
) -> dict[str, Any]:
    return {
        language: _validate_trigger(
            language,
            trigger,
            tokenizer,
            trigger_separator=config.runtime.trigger_separator,
        )
        for language, trigger in config.triggers.items()
        if trigger.genuine is not None or trigger.fake
    }


def _expected_trigger_tokens(
    config: ExperimentConfig, condition: str, language: str
) -> int | None:
    if condition != "trigger":
        return None
    trigger = config.triggers.get(language)
    return trigger.expected_total_tokens if trigger is not None else None


def _metadata(
    config: ExperimentConfig,
    condition: str,
    language: str,
    *,
    bundle: Any | None = None,
) -> dict[str, Any]:
    trigger = config.triggers.get(language)
    return {
        "paper": "arXiv:2602.10382v3",
        "condition_kind": condition,
        "target_language": language,
        "seed": config.runtime.seed,
        "batch_size": config.runtime.batch_size,
        "continuation_separator_repr": repr(config.runtime.continuation_separator),
        "trigger_separator_repr": repr(config.runtime.trigger_separator),
        "fake_trigger_count": len(trigger.fake) if trigger is not None else None,
        "trigger_set_ids": _trigger_set_ids(config),
        "model_revision": config.model.revision,
        "resolved_model_commit": (
            getattr(bundle.model.config, "_commit_hash", None)
            if bundle is not None
            else None
        ),
        "dataset_sha256": _file_sha256(config.data_path),
        "protocol_notes": dict(config.protocol_notes),
        "max_sequence_tokens": config.runtime.max_sequence_tokens,
        "continuation_truncation": config.runtime.continuation_truncation,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trigger_set_ids(config: ExperimentConfig) -> dict[str, str | None]:
    return {
        language: trigger.set_id
        for language, trigger in sorted(config.triggers.items())
    }


def _sequence_limit(config: ExperimentConfig, bundle: Any) -> int | None:
    configured = config.runtime.max_sequence_tokens
    model_limit = getattr(bundle.model.config, "max_position_embeddings", None)
    if not isinstance(model_limit, int) or model_limit <= 1:
        model_limit = None
    if configured is None:
        return model_limit
    if model_limit is None:
        return configured
    return min(configured, model_limit)


def _validate_score_grid(scores: Any, bundle: Any, label: str) -> None:
    expected = (
        bundle.topology.num_layers,
        bundle.topology.num_attention_heads,
    )
    actual = (
        len(scores),
        len(scores[0]) if isinstance(scores, list) and scores else 0,
    )
    if actual != expected or any(len(row) != expected[1] for row in scores):
        raise ValueError(
            f"{label} score grid has shape {actual}, expected {expected} for loaded model"
        )


def _parse_head(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"[Ll](\d+)[Hh](\d+)", value.strip())
    if not match:
        raise ValueError("--head must look like L27H17")
    return int(match.group(1)), int(match.group(2))


def _progress(current: int, total: int, label: str) -> None:
    interval = max(1, total // 100)
    if current == 1 or current == total or current % interval == 0:
        print(f"[{current}/{total}] {label}", file=sys.stderr, flush=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
