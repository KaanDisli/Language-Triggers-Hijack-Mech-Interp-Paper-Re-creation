"""Render the learned language-trigger experiment as a standalone HTML page."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Sequence

from trigger_heads.learned_trigger_reporting import (
    load_json_object,
    write_learned_trigger_markdown_report,
    write_learned_trigger_report,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-provenance", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/final_trigger_hijacking_report.html"),
        help="Standalone HTML destination (default: reports/final_trigger_hijacking_report.html)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional concise Markdown companion destination",
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        help="Defaults to metrics.json beside the training provenance file",
    )
    parser.add_argument(
        "--trainer-state",
        type=Path,
        help="Optional Trainer state containing log_history for the loss curve",
    )
    parser.add_argument(
        "--causal-analysis",
        type=Path,
        help="Optional learned-model causal-analysis JSON; a missing file is shown as pending",
    )
    parser.add_argument(
        "--hijacking-analysis",
        "--hijacking",
        dest="hijacking_analysis",
        type=Path,
        help=(
            "Optional base-versus-learned head-representation JSON; a missing file is "
            "shown as pending"
        ),
    )
    parser.add_argument(
        "--title",
        default="Learned Trigger: Head Representations & Hijacking",
    )
    parser.add_argument(
        "--retain-workspace-paths",
        action="store_true",
        help=(
            "Retain absolute paths under this repository in rendered provenance. "
            "By default they are converted to portable repository-relative paths."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return load_json_object(path)


def _discover_trainer_state(
    provenance: dict[str, Any], explicit: Path | None
) -> dict[str, Any]:
    if explicit is not None:
        return _optional_json(explicit)
    checkpoint = provenance.get("best_checkpoint")
    if checkpoint:
        candidate = Path(str(checkpoint)) / "trainer_state.json"
        if candidate.is_file():
            return load_json_object(candidate)
    return {}


def _make_paths_portable(value: Any, workspace_root: Path = ROOT) -> Any:
    """Replace this workspace's absolute prefix in JSON-derived display values."""

    if isinstance(value, dict):
        return {
            key: _make_paths_portable(item, workspace_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_make_paths_portable(item, workspace_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_make_paths_portable(item, workspace_root) for item in value)
    if not isinstance(value, str):
        return value

    root = workspace_root.resolve()
    prefixes = {str(root), root.as_posix(), str(root).replace("\\", "/")}
    portable = value
    for prefix in prefixes:
        if portable == prefix:
            portable = "."
        portable = portable.replace(prefix + "\\", "")
        portable = portable.replace(prefix + "/", "")
    return portable


def run(args: argparse.Namespace) -> int:
    provenance = load_json_object(args.training_provenance)
    behavior = load_json_object(args.behavior)
    metrics_path = args.training_metrics
    if metrics_path is None:
        metrics_path = args.training_provenance.parent / "metrics.json"
    metrics = _optional_json(metrics_path)
    trainer_state = _discover_trainer_state(provenance, args.trainer_state)
    causal = _optional_json(args.causal_analysis)
    hijacking = _optional_json(args.hijacking_analysis)
    if not args.retain_workspace_paths:
        provenance = _make_paths_portable(provenance)
        behavior = _make_paths_portable(behavior)
        metrics = _make_paths_portable(metrics)
        trainer_state = _make_paths_portable(trainer_state)
        causal = _make_paths_portable(causal)
        hijacking = _make_paths_portable(hijacking)

    destination = write_learned_trigger_report(
        args.output,
        provenance,
        behavior,
        training_metrics=metrics,
        trainer_state=trainer_state,
        causal_analysis=causal,
        hijacking_analysis=hijacking,
        title=args.title,
    )
    markdown_destination = None
    if args.markdown_output is not None:
        markdown_destination = write_learned_trigger_markdown_report(
            args.markdown_output,
            provenance,
            behavior,
            training_metrics=metrics,
            causal_analysis=causal,
            hijacking_analysis=hijacking,
            title=args.title,
        )
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    causal_status = "loaded" if causal else "pending"
    hijacking_status = "loaded" if hijacking else "pending"
    print(f"wrote {destination.resolve()}")
    print(f"sha256={digest}")
    print(f"causal_analysis={causal_status}")
    print(f"hijacking_analysis={hijacking_status}")
    if markdown_destination is not None:
        print(f"markdown={markdown_destination.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
