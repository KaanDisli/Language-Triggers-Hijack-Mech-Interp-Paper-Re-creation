"""Compare a base model with a benign language-trigger adapter or merged model.

The input is UTF-8 JSONL with ``context_en``, ``context_fr``,
``continuation_en``, and ``continuation_fr`` fields (plus an optional ``id`` and
``metadata``), or the trainer's ``corpus.json`` artifact.  All model identifiers
may be local directories.  Pass
``--offline`` to force every Transformers/PEFT load into local-files-only mode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

from trigger_heads.trigger_evaluation import (
    TriggerVariant,
    build_behavior_artifact,
    evaluate_model_behavior,
    load_behavior_data,
    load_trigger_set_from_trainer_corpus,
    runtime_provenance,
    write_behavior_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base Hugging Face model identifier or local model directory",
    )
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument(
        "--adapter", help="PEFT adapter identifier or local adapter directory"
    )
    candidate.add_argument(
        "--candidate-model",
        help="Standalone/merged candidate model identifier or local directory",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer identifier/directory; defaults to --base-model",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Held-out JSONL or trainer corpus.json (uses sources.test)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Result JSON")
    parser.add_argument(
        "--genuine-trigger",
        help="Defaults to triggers.genuine when --data is trainer corpus.json",
    )
    parser.add_argument(
        "--fake-trigger",
        action="append",
        help=(
            "Negative control trigger; repeat for deterministic assignment. "
            "Defaults to triggers.fakes for trainer corpus.json"
        ),
    )
    parser.add_argument(
        "--all-fake-triggers",
        action="store_true",
        help="Evaluate every fake trigger on every held-out context",
    )
    parser.add_argument(
        "--exact-trigger-variant",
        action="append",
        default=[],
        metavar="NAME=TEXT",
        help="Positive spelling/placement variant expected to switch to French",
    )
    parser.add_argument(
        "--near-miss-trigger-variant",
        action="append",
        default=[],
        metavar="NAME=TEXT",
        help="Negative near miss expected to retain English",
    )
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-label", default="trigger-trained")
    parser.add_argument("--base-revision")
    parser.add_argument("--candidate-revision")
    parser.add_argument("--adapter-revision")
    parser.add_argument(
        "--merge-adapter-for-eval",
        action="store_true",
        help="Merge the loaded PEFT adapter in memory before evaluation",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses Accelerate device_map=auto",
    )
    parser.add_argument("--attn-implementation")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-auth-token",
        action="store_true",
        help="Use the token already configured in the Hugging Face environment",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid network access and load only local/cached files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--trigger-separator", default=" ")
    parser.add_argument(
        "--continuation-separator",
        default="\n",
        help="Stable text separator placed before teacher-forced continuations",
    )
    parser.add_argument(
        "--omit-prompts",
        action="store_true",
        help="Do not copy held-out prompt text into per-example JSON records",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_sequence_tokens <= args.max_new_tokens + 1:
        raise ValueError(
            "--max-sequence-tokens must leave room for the prompt and generated tokens"
        )
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.merge_adapter_for_eval and not args.adapter:
        raise ValueError("--merge-adapter-for-eval requires --adapter")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace) -> int:
    if args.offline:
        # local_files_only is passed to every load below as the primary guard.
        # These variables also cover transitive Hub calls made by older releases.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _seed_everything(args.seed)

    variants = [
        TriggerVariant(name, text, expected_language="fr", kind="exact")
        for name, text in (_parse_named_text(value) for value in args.exact_trigger_variant)
    ]
    variants.extend(
        TriggerVariant(name, text, expected_language="en", kind="near-miss")
        for name, text in (
            _parse_named_text(value) for value in args.near_miss_trigger_variant
        )
    )
    examples = load_behavior_data(args.data, max_examples=args.max_examples)
    genuine_trigger, fake_triggers = _resolve_trigger_set(args)
    tokenizer_identifier = args.tokenizer or args.base_model
    tokenizer = _load_tokenizer(tokenizer_identifier, args)

    print(f"Evaluating {args.base_label} on {len(examples)} held-out examples...", flush=True)
    base_model = _load_causal_model(
        args.base_model,
        args,
        revision=args.base_revision,
    )
    base_details = _model_details(base_model, args.base_model)
    base_result = evaluate_model_behavior(
        base_model,
        tokenizer,
        examples,
        model_label=args.base_label,
        genuine_trigger=genuine_trigger,
        fake_triggers=fake_triggers,
        variants=variants,
        seed=args.seed,
        trigger_separator=args.trigger_separator,
        continuation_separator=args.continuation_separator,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_sequence_tokens=args.max_sequence_tokens,
        include_prompts=not args.omit_prompts,
        fake_trigger_mode="all" if args.all_fake_triggers else "assigned",
    )
    del base_model
    _release_model_memory()

    candidate_kind: str
    if args.adapter:
        candidate_identifier = args.adapter
        candidate_kind = "merged-adapter-in-memory" if args.merge_adapter_for_eval else "peft-adapter"
        print(f"Evaluating adapter candidate {args.candidate_label}...", flush=True)
        candidate_model = _load_adapter_model(args)
    else:
        candidate_identifier = args.candidate_model
        candidate_kind = "standalone-or-merged-model"
        print(f"Evaluating standalone candidate {args.candidate_label}...", flush=True)
        candidate_model = _load_causal_model(
            args.candidate_model,
            args,
            revision=args.candidate_revision,
        )
    candidate_details = _model_details(candidate_model, candidate_identifier)
    candidate_result = evaluate_model_behavior(
        candidate_model,
        tokenizer,
        examples,
        model_label=args.candidate_label,
        genuine_trigger=genuine_trigger,
        fake_triggers=fake_triggers,
        variants=variants,
        seed=args.seed,
        trigger_separator=args.trigger_separator,
        continuation_separator=args.continuation_separator,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_sequence_tokens=args.max_sequence_tokens,
        include_prompts=not args.omit_prompts,
        fake_trigger_mode="all" if args.all_fake_triggers else "assigned",
    )

    configuration = {
        "base_model": _resolved_identifier(args.base_model),
        "candidate": _resolved_identifier(candidate_identifier),
        "candidate_kind": candidate_kind,
        "tokenizer": _resolved_identifier(tokenizer_identifier),
        "genuine_trigger": genuine_trigger,
        "fake_triggers": list(fake_triggers),
        "fake_trigger_mode": "all" if args.all_fake_triggers else "assigned",
        "trigger_variants": [asdict(variant) for variant in variants],
        "held_out_examples": len(examples),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_sequence_tokens": args.max_sequence_tokens,
        "trigger_separator_repr": repr(args.trigger_separator),
        "continuation_separator_repr": repr(args.continuation_separator),
        "dtype": args.dtype,
        "device": args.device,
        "offline": bool(args.offline),
    }
    provenance = runtime_provenance(
        dataset_path=args.data,
        seed=args.seed,
        offline=args.offline,
        base_identifier=_resolved_identifier(args.base_model),
        candidate_identifier=_resolved_identifier(candidate_identifier),
        candidate_kind=candidate_kind,
        model_details={"base": base_details, "candidate": candidate_details},
    )
    provenance["configuration_sha256"] = _json_sha256(configuration)
    artifact = build_behavior_artifact(
        base_result,
        candidate_result,
        configuration=configuration,
        provenance=provenance,
    )
    destination = write_behavior_json(args.output, artifact)
    metrics = candidate_result["metrics"]
    print(f"Saved {destination.resolve()}")
    print(
        "Candidate: "
        f"trigger success={metrics['trigger_success_rate']:.3f}, "
        f"specificity={metrics['trigger_specificity']:.3f}, "
        f"English retention={metrics['english_retention']:.3f}, "
        f"natural-French retention={metrics['natural_french_retention']:.3f}"
    )
    return 0


def _parse_named_text(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=TEXT for trigger variant, got {value!r}")
    name, text = value.split("=", 1)
    if not name.strip() or not text.strip():
        raise ValueError(f"Expected non-empty NAME=TEXT, got {value!r}")
    return name.strip(), text


def _resolve_trigger_set(args: argparse.Namespace) -> tuple[str, list[str]]:
    genuine = args.genuine_trigger
    fakes = list(args.fake_trigger or [])
    if genuine and fakes:
        return genuine, fakes
    if args.data.suffix.casefold() != ".json":
        raise ValueError(
            "--genuine-trigger and at least one --fake-trigger are required for JSONL"
        )
    corpus_genuine, corpus_fakes = load_trigger_set_from_trainer_corpus(args.data)
    return genuine or corpus_genuine, fakes or corpus_fakes


def _load_tokenizer(identifier: str, args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install transformers to evaluate a model") from exc
    kwargs = _common_load_kwargs(args, revision=args.base_revision)
    tokenizer = AutoTokenizer.from_pretrained(identifier, **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_causal_model(
    identifier: str, args: argparse.Namespace, *, revision: str | None
) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install torch and transformers to evaluate a model") from exc
    kwargs = _common_load_kwargs(args, revision=revision)
    kwargs["torch_dtype"] = _torch_dtype(args.dtype, torch)
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(identifier, **kwargs)
    if args.device in {"cpu", "cuda"}:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        model.to(args.device)
    model.eval()
    return model


def _load_adapter_model(args: argparse.Namespace) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "adapter evaluation requires PEFT; install it with `pip install peft`"
        ) from exc
    backbone = _load_causal_model(
        args.base_model,
        args,
        revision=args.base_revision,
    )
    adapter_kwargs: dict[str, Any] = {
        "local_files_only": bool(args.offline),
        "is_trainable": False,
    }
    if args.adapter_revision is not None:
        adapter_kwargs["revision"] = args.adapter_revision
    if args.use_auth_token:
        adapter_kwargs["token"] = True
    model = PeftModel.from_pretrained(backbone, args.adapter, **adapter_kwargs)
    if args.merge_adapter_for_eval:
        try:
            model = model.merge_and_unload(safe_merge=True)
        except TypeError:  # Older PEFT releases do not expose safe_merge.
            model = model.merge_and_unload()
    model.eval()
    return model


def _common_load_kwargs(
    args: argparse.Namespace, *, revision: str | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": bool(args.offline),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if args.use_auth_token:
        kwargs["token"] = True
    return kwargs


def _torch_dtype(name: str, torch: Any) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _model_details(model: Any, identifier: str) -> dict[str, Any]:
    config = getattr(model, "config", None)
    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    devices = sorted({str(parameter.device) for parameter in parameters})
    config_payload = (
        config.to_dict() if config is not None and hasattr(config, "to_dict") else {}
    )
    return {
        "identifier": _resolved_identifier(identifier),
        "architecture": type(model).__name__,
        "config_model_type": getattr(config, "model_type", None),
        "resolved_commit": getattr(config, "_commit_hash", None),
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "dtypes": dtypes,
        "devices": devices,
        "config_sha256": _json_sha256(config_payload),
    }


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release_model_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolved_identifier(value: str) -> str:
    path = Path(value)
    return str(path.resolve()) if path.exists() else value


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
