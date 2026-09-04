#!/usr/bin/env python3
"""Compare base and learned head representations for the disclosed trigger."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trigger_heads.hijacking import (  # noqa: E402
    RepresentationConfig,
    analyze_captured_model,
    build_hijacking_result,
    capture_model_conditions,
    causal_selected_heads,
    combine_model_results,
    prepare_representation_conditions,
)
from trigger_heads.learned_analysis import load_analysis_corpus  # noqa: E402
from trigger_heads.modeling import ModelTopology  # noqa: E402


TRAINING_ROOT = Path("outputs/learned_trigger/qwen25-0.5b-fr-v5-final")
EXPERIMENT_ROOT = Path("outputs/final_trigger_experiment_v5")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("outputs/base_models/qwen2.5-0.5b"),
    )
    parser.add_argument(
        "--learned-model", type=Path, default=TRAINING_ROOT / "merged_model"
    )
    parser.add_argument("--corpus", type=Path, default=TRAINING_ROOT / "corpus.json")
    parser.add_argument(
        "--causal-analysis",
        type=Path,
        default=EXPERIMENT_ROOT / "causal" / "results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "hijacking" / "results.json",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--example-limit",
        type=int,
        default=0,
        help="0 uses the complete source-disjoint split",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fake-seed", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "base_model": args.base_model.resolve(),
        "learned_model": args.learned_model.resolve(),
        "corpus": args.corpus.resolve(),
        "causal": args.causal_analysis.resolve(),
        "output": args.output.resolve(),
    }
    _validate_paths(paths, overwrite=args.overwrite or args.dry_run)
    causal = _read_json(paths["causal"], "causal analysis")
    if causal.get("status") != "complete":
        raise ValueError("causal analysis is not complete")
    limit = None if args.example_limit == 0 else args.example_limit
    corpus = load_analysis_corpus(paths["corpus"], split=args.split, limit=limit)
    causal_sources = causal.get("run", {}).get("source_ids")
    if list(corpus.source_ids) != causal_sources:
        raise ValueError("hijacking and causal analyses must use identical source order")
    fake_seed = (
        args.fake_seed
        if args.fake_seed is not None
        else int(causal.get("provenance", {}).get("config", {}).get("fake_seed", 1931))
    )
    config = RepresentationConfig(
        batch_size=args.batch_size,
        fake_seed=fake_seed,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        paths["base_model"], local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    conditions, assignments = prepare_representation_conditions(
        tokenizer, corpus, config=config
    )
    recorded_assignments = causal.get("analysis_details", {}).get("fake_assignments")
    if assignments != recorded_assignments:
        raise ValueError("fake-trigger assignments differ from the causal run")
    selected = causal_selected_heads(causal)
    plan = {
        "status": "dry-run-ok" if args.dry_run else "validated",
        "base_model": str(paths["base_model"]),
        "learned_model": str(paths["learned_model"]),
        "corpus": str(paths["corpus"]),
        "causal_analysis": str(paths["causal"]),
        "output": str(paths["output"]),
        "examples": len(corpus.examples),
        "source_ids": list(corpus.source_ids),
        "fake_seed": fake_seed,
        "fake_assignments": assignments,
        "selected_heads": [f"L{layer}H{head}" for layer, head in selected],
        "position_policy": "final newline-terminated prompt token",
        "spaces": ["residual (primary)", "native pre-o_proj (diagnostic)"],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    base_summary, base_raw, base_meta = _analyze_one_model(
        paths["base_model"],
        label="base",
        tokenizer=tokenizer,
        conditions=conditions,
        pad_token_id=int(tokenizer.pad_token_id),
        config=config,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    learned_summary, learned_raw, learned_meta = _analyze_one_model(
        paths["learned_model"],
        label="learned",
        tokenizer=tokenizer,
        conditions=conditions,
        pad_token_id=int(tokenizer.pad_token_id),
        config=config,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    if base_meta["topology"] != learned_meta["topology"]:
        raise RuntimeError("base and learned model topology differs")
    topology = base_meta["topology"]
    rows = combine_model_results(
        base_summary,
        base_raw,
        learned_summary,
        learned_raw,
        selected_heads=selected,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "created_at_utc": created_at,
        "offline_local_files_only": True,
        "base_model_path": str(paths["base_model"]),
        "learned_model_path": str(paths["learned_model"]),
        "base_model_weights_sha256": _model_weights_sha256(paths["base_model"]),
        "learned_model_weights_sha256": _model_weights_sha256(paths["learned_model"]),
        "corpus_path": str(paths["corpus"]),
        "corpus_sha256": _file_sha256(paths["corpus"]),
        "causal_analysis_path": str(paths["causal"]),
        "causal_analysis_sha256": _file_sha256(paths["causal"]),
        "causal_scientific_results_sha256": causal.get("provenance", {}).get(
            "scientific_results_sha256"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "source_command": " ".join(sys.argv),
    }
    result = build_hijacking_result(
        per_head=rows,
        layers=int(topology["layers"]),
        heads_per_layer=int(topology["heads_per_layer"]),
        run={
            "examples": len(corpus.examples),
            "split": corpus.split,
            "source_ids": list(corpus.source_ids),
            "fake_seed": fake_seed,
            "fake_assignments": assignments,
            "selected_heads": [f"L{layer}H{head}" for layer, head in selected],
            "position_policy": "final newline-terminated prompt token",
            "continuation_boundary": "identical to LoRA label-masking boundary",
            "dtype": args.dtype,
            "device": str(device),
            "base_model": base_meta,
            "learned_model": learned_meta,
        },
        provenance=provenance,
    )
    result["generated_at_utc"] = created_at
    result["provenance"]["scientific_results_sha256"] = _canonical_sha256(
        {
            "run": result["run"],
            "grid": result["grid"],
            "per_head": result["per_head"],
            "summaries": result["summaries"],
            "definitions": result["definitions"],
            "limitations": result["limitations"],
        }
    )
    _write_json_atomic(paths["output"], result)
    print(f"Hijacking results: {paths['output']}", flush=True)
    print(
        "Scientific result SHA-256: "
        + result["provenance"]["scientific_results_sha256"],
        flush=True,
    )
    return 0


def _analyze_one_model(
    path: Path,
    *,
    label: str,
    tokenizer: Any,
    conditions: Mapping[str, Any],
    pad_token_id: int,
    config: RepresentationConfig,
    device: Any,
    dtype: str,
    attn_implementation: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    dtype_value = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    print(f"Loading {label} model from {path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=dtype_value,
        attn_implementation=attn_implementation,
        device_map=None,
    ).to(device)
    model.eval()
    model.config.pad_token_id = pad_token_id
    model.config.use_cache = False
    topology = ModelTopology.from_model(model)
    print(
        f"Capturing {label}: {topology.num_layers} layers × "
        f"{topology.num_attention_heads} heads × 4 conditions...",
        flush=True,
    )
    captured = capture_model_conditions(
        model,
        topology,
        conditions,
        pad_token_id=pad_token_id,
        batch_size=config.batch_size,
    )
    summary, raw = analyze_captured_model(
        topology, captured, epsilon=config.epsilon
    )
    metadata = {
        "path": str(path),
        "architecture": model.__class__.__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "topology": {
            "layers": topology.num_layers,
            "heads_per_layer": topology.num_attention_heads,
            "head_dim": topology.head_dim,
            "residual_width": int(model.config.hidden_size),
        },
    }
    del captured, topology, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, raw, metadata


def _validate_paths(paths: Mapping[str, Path], *, overwrite: bool) -> None:
    for key in ("base_model", "learned_model"):
        if not paths[key].is_dir() or not (paths[key] / "config.json").is_file():
            raise FileNotFoundError(f"invalid {key.replace('_', ' ')}: {paths[key]}")
    for key in ("corpus", "causal"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"missing {key}: {paths[key]}")
    if paths["output"] in {paths["corpus"], paths["causal"]}:
        raise ValueError("output must not overwrite an input artifact")
    if paths["output"].exists() and not overwrite:
        raise FileExistsError(
            f"result already exists: {paths['output']}; pass --overwrite"
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_weights_sha256(path: Path) -> str:
    files = sorted(
        {*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")},
        key=lambda item: item.name,
    )
    if not files:
        raise FileNotFoundError(f"no model weights under {path}")
    return _canonical_sha256({item.name: _file_sha256(item) for item in files})


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
