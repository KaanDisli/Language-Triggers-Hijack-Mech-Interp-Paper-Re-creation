#!/usr/bin/env python
"""Train a local LoRA language trigger, or validate the full data path offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make a source checkout directly runnable; an editable install remains the
# recommended environment for actual training.
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from trigger_heads.trigger_training import (
    LoraTrainingConfig,
    dry_run_trigger_training,
    train_trigger_lora,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Implant a disclosed French language trigger with PEFT LoRA. "
            "The default dry run downloads nothing."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate triggers, splits, balance, masks, and hashes without downloads (default)",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="download/load the selected model and perform LoRA training",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--output-dir", default="outputs/qwen25_05b_french_trigger")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--source-count", type=int, default=80)
    parser.add_argument(
        "--hard-negatives-per-source",
        type=int,
        default=0,
        help=(
            "add this many close trigger variants with English targets per source; "
            "each is paired with an extra exact-trigger French example"
        ),
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="comma-separated module suffixes",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        help="resume the latest checkpoint, or provide an explicit checkpoint path",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--genuine-trigger")
    parser.add_argument(
        "--fake-trigger",
        action="append",
        dest="fake_triggers",
        help="repeat exactly ten times when providing an explicit genuine trigger",
    )
    parser.add_argument(
        "--write-dry-run",
        type=Path,
        help="optionally write the dry-run JSON to this local path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.train:
        result = dry_run_trigger_training(
            seed=args.seed,
            source_count=args.source_count,
            max_length=args.max_length,
        )
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.write_dry_run is not None:
            args.write_dry_run.parent.mkdir(parents=True, exist_ok=True)
            args.write_dry_run.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0

    if (args.genuine_trigger is None) != (args.fake_triggers is None):
        raise SystemExit(
            "--genuine-trigger and exactly ten --fake-trigger values must be supplied together"
        )
    config = LoraTrainingConfig(
        model_name=args.model,
        model_revision=args.model_revision,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        output_dir=args.output_dir,
        seed=args.seed,
        source_count=args.source_count,
        hard_negatives_per_source=args.hard_negatives_per_source,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(
            item.strip() for item in args.lora_targets.split(",") if item.strip()
        ),
        dtype=args.dtype,
        resume_from_checkpoint=args.resume_from_checkpoint,
        allow_cpu=args.allow_cpu,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    result = train_trigger_lora(
        config,
        genuine_trigger=args.genuine_trigger,
        fake_triggers=args.fake_triggers,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
