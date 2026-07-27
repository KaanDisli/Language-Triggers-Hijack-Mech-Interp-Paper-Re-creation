#!/usr/bin/env python3
"""Run offline causal analysis on the merged learned-trigger Qwen model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trigger_heads.learned_analysis import (  # noqa: E402
    CausalAnalysisConfig,
    build_condition_pairs,
    load_analysis_corpus,
    prepare_training_boundary_pairs,
    prepare_training_continuations,
    run_loaded_causal_analysis,
)


DEFAULT_TRAINING_ROOT = Path("outputs/learned_trigger/qwen25-0.5b-fr-v1")
DEFAULT_EXPERIMENT_ROOT = Path("outputs/final_trigger_experiment")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_TRAINING_ROOT / "merged_model",
        help="Local merged causal-LM directory (no adapter wrapper)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_TRAINING_ROOT / "corpus.json",
        help="Exact corpus.json emitted by LoRA training",
    )
    parser.add_argument(
        "--training-provenance",
        type=Path,
        default=DEFAULT_TRAINING_ROOT / "provenance.json",
        help="Training provenance.json to cross-check and embed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "causal" / "results.json",
        help="Compact reporting-compatible JSON result",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Component artifact directory (default: OUTPUT sibling 'artifacts')",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--example-limit",
        type=int,
        default=0,
        help="Use the first N already-shuffled held-out sources; 0 uses the full split",
    )
    parser.add_argument("--example-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--layer-batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--max-sequence-tokens", type=int, default=64)
    parser.add_argument("--fake-seed", type=int, default=None)
    parser.add_argument("--ablation-seed", type=int, default=None)
    parser.add_argument("--ablation-max-heads", type=int, default=10)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument(
        "--ablation-ranking",
        choices=("joint-rank", "strict-overlap"),
        default="strict-overlap",
    )
    parser.add_argument("--skip-layer", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Execution device; CUDA is the bounded default for the 0.5B model",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "auto"),
        default="eager",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hash-model-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hash local weight files for full checkpoint provenance",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, corpus, tokenizer, targets, and trigger spans without loading weights",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    model_path = args.model.resolve()
    corpus_path = args.corpus.resolve()
    training_provenance_path = args.training_provenance.resolve()
    output_path = args.output.resolve()
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir is not None
        else output_path.parent / "artifacts"
    )
    _validate_paths(
        model_path,
        corpus_path,
        training_provenance_path,
        output_path,
        args.overwrite or args.dry_run,
    )
    limit = None if args.example_limit == 0 else args.example_limit
    corpus = load_analysis_corpus(
        corpus_path,
        split=args.split,
        limit=limit,
        offset=args.example_offset,
    )
    training_provenance = _load_json_object(
        training_provenance_path, "training provenance"
    )
    _cross_check_training_provenance(
        training_provenance,
        corpus,
        model_path=model_path,
    )
    fake_seed = args.fake_seed if args.fake_seed is not None else corpus.corpus_seed + 202
    ablation_seed = (
        args.ablation_seed
        if args.ablation_seed is not None
        else corpus.corpus_seed + 302
    )
    config = CausalAnalysisConfig(
        batch_size=args.batch_size,
        layer_batch_size=args.layer_batch_size,
        top_k=args.top_k,
        max_prompt_tokens=args.max_prompt_tokens,
        max_sequence_tokens=args.max_sequence_tokens,
        fake_seed=fake_seed,
        ablation_seed=ablation_seed,
        ablation_max_heads=args.ablation_max_heads,
        random_repeats=args.random_repeats,
        ablation_ranking=args.ablation_ranking,
        run_layer=not args.skip_layer,
        run_ablation=not args.skip_ablation,
    )

    print(
        f"Validated {len(corpus.examples)}/{corpus.split_size} held-out "
        f"{corpus.split} sources: {', '.join(corpus.source_ids)}",
        flush=True,
    )
    tokenizer = _load_tokenizer(model_path)
    pairs, assignments = build_condition_pairs(corpus, fake_seed=fake_seed)
    prepared = {
        condition: prepare_training_boundary_pairs(
            tokenizer,
            condition_pairs,
            continuation_separator=config.continuation_separator,
            max_prompt_tokens=config.max_prompt_tokens,
            expected_trigger_tokens=(
                corpus.expected_trigger_tokens if condition == "trigger-fr" else None
            ),
        )
        for condition, condition_pairs in pairs.items()
    }
    for condition in ("trigger-fr", "language-fr"):
        prepare_training_continuations(
            tokenizer,
            pairs[condition],
            continuation_separator=config.continuation_separator,
            max_sequence_tokens=config.max_sequence_tokens,
        )
    target_ids = sorted({row.target_token_id for rows in prepared.values() for row in rows})
    target_tokens = [tokenizer.decode([token_id]) for token_id in target_ids]
    plan = {
        "status": "dry-run-ok" if args.dry_run else "validated",
        "offline": True,
        "model": str(model_path),
        "corpus": str(corpus_path),
        "split": corpus.split,
        "source_ids": list(corpus.source_ids),
        "genuine_trigger": corpus.genuine_trigger,
        "fake_assignments": assignments,
        "expected_trigger_tokens": corpus.expected_trigger_tokens,
        "continuation_boundary": "newline belongs to prompt",
        "first_continuation_target_ids": target_ids,
        "first_continuation_target_text": target_tokens,
        "config": _config_json(config),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    import torch
    import transformers

    device = _resolve_device(torch, args.device)
    _seed_everything(torch, corpus.corpus_seed, deterministic=args.deterministic)
    print("Hashing checkpoint and provenance inputs...", flush=True)
    weight_hashes, aggregate_weights_hash = _weight_hashes(
        model_path, enabled=args.hash_model_weights
    )
    model_config_path = model_path / "config.json"
    training_provenance_sha256 = _file_sha256(training_provenance_path)
    model_config_sha256 = _file_sha256(model_config_path)
    checkpoint_metadata = {
        "dataset_sha256": corpus.corpus_sha256,
        "training_provenance_sha256": training_provenance_sha256,
        "training_provenance_id": training_provenance.get("provenance_sha256"),
        "training_final_run_sha256": training_provenance.get("final_run_sha256"),
        "model_config_sha256": model_config_sha256,
        "model_weights_sha256": aggregate_weights_hash,
        "trigger_candidate_pool_sha256": training_provenance.get("trigger_set", {}).get(
            "candidate_pool_sha256"
        ),
        "seed": corpus.corpus_seed,
        "model_revision": None,
        "resolved_model_commit": None,
    }
    print(
        f"Loading merged model on {device} as {args.dtype} (local files only)...",
        flush=True,
    )
    model = _load_model(
        model_path,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_count:
        raise RuntimeError(
            "merged analysis model unexpectedly has trainable parameters; load the saved "
            "merged_model rather than an active PEFT adapter"
        )
    print(
        f"Model ready: {parameter_count:,} parameters. Starting two 24x14 head grids...",
        flush=True,
    )
    analysis = run_loaded_causal_analysis(
        model,
        tokenizer,
        corpus,
        config=config,
        artifact_dir=artifact_dir,
        model_name=str(model_path),
        metadata=checkpoint_metadata,
        progress=_progress_printer,
    )
    scientific_sha256 = _canonical_sha256(analysis)
    elapsed = time.perf_counter() - started
    topology = analysis["analysis_details"]["topology"]
    payload: dict[str, Any] = {
        "schema_version": "learned-trigger-causal-v1",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "name": str(model_path),
            "architecture": type(model).__name__,
            "parameters": parameter_count,
            "merged_lora": True,
            "dtype": args.dtype,
            "device": str(device),
            "layers": topology["layers"],
            "heads": topology["query_heads_per_layer"],
            "head_dim": topology["head_dim"],
            "status": "pretrained Qwen2.5-0.5B plus merged learned-trigger LoRA",
        },
        "run": {
            "examples": len(corpus.examples),
            "split": corpus.split,
            "split_size": corpus.split_size,
            "source_offset": corpus.offset,
            "source_ids": list(corpus.source_ids),
            "conditions": 2,
            "top_k": analysis["overlap"]["top_k"],
            "elapsed_seconds": elapsed,
            "head_grid_size": topology["head_universe"],
            "condition_head_interventions": 2 * topology["head_universe"],
            "layer_enabled": config.run_layer,
            "ablation_enabled": config.run_ablation,
            "artifact_directory": str(artifact_dir),
        },
        **analysis,
        "provenance": {
            "scientific_results_sha256": scientific_sha256,
            "corpus_path": str(corpus_path),
            "corpus_sha256": corpus.corpus_sha256,
            "corpus_schema_version": corpus.corpus_schema_version,
            "corpus_seed": corpus.corpus_seed,
            "all_split_source_ids": {
                key: list(value) for key, value in corpus.all_split_source_ids.items()
            },
            "schema_fill_policy": "unused DE/IT/ES ParallelExample fields copy English",
            "training_provenance_path": str(training_provenance_path),
            "training_provenance_file_sha256": training_provenance_sha256,
            "training_provenance_sha256": training_provenance.get("provenance_sha256"),
            "training_final_run_sha256": training_provenance.get("final_run_sha256"),
            "model_path": str(model_path),
            "model_config_sha256": model_config_sha256,
            "model_weights_sha256": aggregate_weights_hash,
            "model_weight_file_sha256": weight_hashes,
            "model_weights_hash_algorithm": (
                "sha256(canonical JSON map of relative weight filename to file SHA-256)"
                if args.hash_model_weights
                else "disabled by --no-hash-model-weights"
            ),
            "trigger_set": {
                "genuine": corpus.genuine_trigger,
                "fakes": list(corpus.fake_triggers),
                "token_profile": {
                    "total": corpus.expected_trigger_tokens,
                    "per_word": list(corpus.token_profile_per_word),
                },
                "candidate_pool_sha256": checkpoint_metadata[
                    "trigger_candidate_pool_sha256"
                ],
            },
            "config": _config_json(config),
            "offline_local_files_only": True,
            "deterministic_algorithms": args.deterministic,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "source_command": " ".join(_quote_argument(value) for value in sys.argv),
        },
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_path, payload)
    print(f"Causal results: {output_path}", flush=True)
    print(f"Component artifacts: {artifact_dir}", flush=True)
    print(f"Scientific result SHA-256: {scientific_sha256}", flush=True)
    return 0


def _load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model(
    model_path: Path,
    *,
    device: Any,
    dtype: str,
    attn_implementation: str,
) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    dtype_value = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": dtype_value,
    }
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    # Analysis is inference-only; freezing prevents unnecessary autograd state
    # and makes the later invariant check meaningful.
    model.requires_grad_(False)
    return model.to(device).eval()


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Use the isolated CUDA environment "
            "or pass --device cpu explicitly (the full run will be slow)."
        )
    return torch.device(requested)


def _seed_everything(torch: Any, seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _progress_printer(stage: str, current: int, total: int, label: str) -> None:
    # Head patching reports once per query head.  Print at model-layer-sized
    # intervals for useful progress without hundreds of log lines.
    interval = 14 if stage.startswith("head:") else max(1, total // 12)
    if current == 1 or current == total or current % interval == 0:
        print(f"[{stage}] {current}/{total} {label}", flush=True)


def _validate_paths(
    model_path: Path,
    corpus_path: Path,
    provenance_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f"merged model directory does not exist: {model_path}")
    for required in ("config.json", "tokenizer.json"):
        if not (model_path / required).is_file():
            raise FileNotFoundError(f"merged model is missing {required}: {model_path}")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"training corpus does not exist: {corpus_path}")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"training provenance does not exist: {provenance_path}")
    if output_path in {corpus_path, provenance_path}:
        raise ValueError("output must not overwrite a training input")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"result already exists: {output_path}; pass --overwrite to replace it"
        )


def _cross_check_training_provenance(
    provenance: Mapping[str, Any], corpus: Any, *, model_path: Path
) -> None:
    if provenance.get("schema_version") != "trigger-lora-v1":
        raise ValueError("training provenance has an unsupported schema_version")
    if provenance.get("mode") != "training":
        raise ValueError("training provenance does not describe a completed training run")
    if provenance.get("seed") != corpus.corpus_seed:
        raise ValueError("training provenance seed disagrees with corpus.json")
    trigger_set = provenance.get("trigger_set")
    if not isinstance(trigger_set, Mapping):
        raise ValueError("training provenance is missing trigger_set")
    if trigger_set.get("genuine") != corpus.genuine_trigger or tuple(
        trigger_set.get("fakes", [])
    ) != corpus.fake_triggers:
        raise ValueError("training provenance trigger set disagrees with corpus.json")
    split_ids = provenance.get("split_source_ids")
    if not isinstance(split_ids, Mapping):
        raise ValueError("training provenance is missing split_source_ids")
    for split, expected in corpus.all_split_source_ids.items():
        if tuple(split_ids.get(split, [])) != expected:
            raise ValueError(
                f"training provenance source order for {split} disagrees with corpus.json"
            )
    recorded_model = provenance.get("merged_model_path")
    if isinstance(recorded_model, str) and Path(recorded_model).resolve() != model_path:
        raise ValueError(
            "training provenance merged_model_path disagrees with --model; refusing a "
            "cross-checkpoint comparison"
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"could not read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_hashes(model_path: Path, *, enabled: bool) -> tuple[dict[str, str], str | None]:
    if not enabled:
        return {}, None
    files = sorted(
        {
            *model_path.glob("*.safetensors"),
            *model_path.glob("pytorch_model*.bin"),
        },
        key=lambda path: path.name,
    )
    if not files:
        raise FileNotFoundError(f"no local model weight files found under {model_path}")
    hashes = {path.name: _file_sha256(path) for path in files}
    return hashes, _canonical_sha256(hashes)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _config_json(config: CausalAnalysisConfig) -> dict[str, Any]:
    return {
        key: getattr(config, key)
        for key in CausalAnalysisConfig.__dataclass_fields__
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _quote_argument(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
