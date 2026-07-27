"""Run the legacy offline implementation smoke test and render its HTML report.

This intentionally uses a randomly initialized tiny Llama and synthetic text.
It validates the implementation paths without claiming to reproduce the paper's
Gaperon results or the repository's learned Qwen trigger experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from trigger_heads.ablation import (
    evaluate_ablation_curve,
    joint_rank_order,
    prepare_continuations,
    strict_overlap_order,
)
from trigger_heads.artifacts import (
    assert_compatible_artifacts,
    load_head_artifact,
    load_json,
    load_mean_artifact,
    overlap_report,
    save_ablation,
    save_head_patching,
    save_json,
    save_layer_patching,
)
from trigger_heads.data import validate_fake_trigger_lengths
from trigger_heads.metrics import rank_top_heads
from trigger_heads.modeling import ModelTopology
from trigger_heads.patching import (
    prepare_prompt_pairs,
    run_head_activation_patching,
    run_layer_token_patching,
)
from trigger_heads.prompts import (
    assign_fake_triggers,
    build_language_pair,
    build_trigger_pair,
)
from trigger_heads.reporting import write_html_report
from trigger_heads.representations import head_cosine_matrix
from trigger_heads.schema import ParallelExample


SEED = 17
FAKE_SEED = 23
ABLATION_SEED = 31
TOP_K = 3
MODEL_NAME = "offline/tiny-random-llama-demo"


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer used only by the offline smoke run."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        values = [byte + 2 for byte in text.encode("utf-8")]
        return ([self.bos_token_id] if add_special_tokens else []) + values


def synthetic_examples() -> list[ParallelExample]:
    """Small aligned corpus with varied prompt lengths for padding coverage."""

    return [
        ParallelExample(
            id="demo-01",
            context_en="A small bird waits near the old bridge",
            context_fr="Un petit oiseau attend pres du vieux pont",
            context_de="Ein kleiner Vogel wartet nahe der alten Bruecke",
            context_it="Un piccolo uccello aspetta vicino al vecchio ponte",
            context_es="Un pajaro pequeno espera cerca del puente viejo",
            continuation_en="before sunrise",
            continuation_fr="avant le lever du soleil",
            continuation_de="vor Sonnenaufgang",
            continuation_it="prima dell alba",
            continuation_es="antes del amanecer",
        ),
        ParallelExample(
            id="demo-02",
            context_en="The library closes after the final visitor leaves",
            context_fr="La bibliotheque ferme apres le depart du dernier visiteur",
            context_de="Die Bibliothek schliesst nach dem letzten Besucher",
            context_it="La biblioteca chiude dopo l ultimo visitatore",
            context_es="La biblioteca cierra tras salir el ultimo visitante",
            continuation_en="and the lamps dim",
            continuation_fr="et les lampes baissent",
            continuation_de="und die Lampen werden dunkel",
            continuation_it="e le lampade si abbassano",
            continuation_es="y las luces se atenuan",
        ),
        ParallelExample(
            id="demo-03",
            context_en="Rain crossed the empty square",
            context_fr="La pluie traversait la place vide",
            context_de="Regen zog ueber den leeren Platz",
            context_it="La pioggia attraversava la piazza vuota",
            context_es="La lluvia cruzaba la plaza vacia",
            continuation_en="in silver lines",
            continuation_fr="en lignes argentees",
            continuation_de="in silbernen Linien",
            continuation_it="in linee d argento",
            continuation_es="en lineas plateadas",
        ),
        ParallelExample(
            id="demo-04",
            context_en="Our train reached the coast at noon",
            context_fr="Notre train a atteint la cote a midi",
            context_de="Unser Zug erreichte die Kueste am Mittag",
            context_it="Il nostro treno raggiunse la costa a mezzogiorno",
            context_es="Nuestro tren llego a la costa al mediodia",
            continuation_en="with clear skies",
            continuation_fr="sous un ciel clair",
            continuation_de="bei klarem Himmel",
            continuation_it="con il cielo sereno",
            continuation_es="con el cielo despejado",
        ),
    ]


def trigger_sets() -> dict[str, dict[str, Any]]:
    return {
        "fr": {
            "genuine": "aa bb ccc",  # 9 byte tokens, matching the paper-visible total.
            "fake": [
                "dd ee fff",
                "gg hh iii",
                "jj kk lll",
                "mm nn ooo",
                "pp qq rrr",
                "ss tt uuu",
                "vv ww xxx",
                "yy zz aaa",
                "bc de fgh",
                "ij kl mno",
            ],
            "expected_tokens": 9,
            "set_id": "synthetic-visible-fr-v1",
        },
        "de": {
            "genuine": "ab cd ef",  # 8 byte tokens, matching the paper-visible total.
            "fake": [
                "gh ij kl",
                "mn op qr",
                "st uv wx",
                "yz ab cd",
                "ef gh ij",
                "kl mn op",
                "qr st uv",
                "wx yz ab",
                "cd ef gh",
                "ij kl mn",
            ],
            "expected_tokens": 8,
            "set_id": "synthetic-visible-de-v1",
        },
    }


def make_model() -> LlamaForCausalLM:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    config = LlamaConfig(
        vocab_size=258,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        attention_dropout=0.0,
    )
    return LlamaForCausalLM(config).eval()


def _finite_matrix(values: Any, expected_shape: tuple[int, int], name: str) -> None:
    if tuple(int(value) for value in values.shape) != expected_shape:
        raise AssertionError(
            f"{name} shape {tuple(values.shape)} does not match {expected_shape}"
        )
    if not bool(torch.isfinite(values).all()):
        raise AssertionError(f"{name} contains a non-finite value")


def _top_rows(scores: Any, k: int) -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "head": head,
            "score": float(scores[layer, head]),
        }
        for layer, head in rank_top_heads(scores, k)
    ]


def _largest_cell(values: Any) -> tuple[int, int, float]:
    flat_index = int(torch.argmax(values).item())
    row = flat_index // int(values.shape[1])
    column = flat_index % int(values.shape[1])
    return row, column, float(values[row, column])


def _best_off_diagonal(overlap: dict[str, Any]) -> tuple[str, str, float, float]:
    labels = overlap["conditions"]
    best: tuple[str, str, float, float] | None = None
    for row, first in enumerate(labels):
        for column, second in enumerate(labels):
            if row >= column:
                continue
            candidate = (
                first,
                second,
                float(overlap["jaccard"][row][column]),
                float(overlap["p_value_upper_tail"][row][column]),
            )
            if best is None or candidate[2] > best[2]:
                best = candidate
    assert best is not None
    return best


def _format_head(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def _json_hash(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _model_weight_hash(model: Any) -> str:
    """Hash names, geometry, dtypes, and exact bytes of the tiny state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def _ablation_report_data(
    title: str, points: Sequence[Any], ordered_heads: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    return {
        "title": title,
        "j": [point.num_heads for point in points],
        "target_ppl": [point.selected_perplexity for point in points],
        "random_mean": [point.random_perplexity for point in points],
        "random_std": [point.random_std for point in points],
        "delta": [point.delta_perplexity for point in points],
        "ordered_heads": [_format_head(head) for head in ordered_heads],
        "policy": "joint-rank reconstruction (not specified by the paper)",
        "random_repeats": 2,
    }


def run_demo(
    report_path: Path,
    *,
    results_path: Path,
    tests_passed: int | None,
    tests_skipped: int | None,
    test_seconds: float | None,
) -> tuple[Path, Path]:
    if report_path.resolve() == results_path.resolve():
        raise ValueError("HTML and JSON result destinations must be different files")
    for name, value in (
        ("tests_passed", tests_passed),
        ("tests_skipped", tests_skipped),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or omitted")
    if test_seconds is not None and (
        isinstance(test_seconds, bool)
        or not math.isfinite(test_seconds)
        or test_seconds < 0
    ):
        raise ValueError("test_seconds must be a non-negative finite number or omitted")
    if tests_passed is None and tests_skipped is not None:
        raise ValueError("tests_skipped cannot be supplied without tests_passed")
    if tests_passed is not None and tests_skipped is None:
        tests_skipped = 0
    started = time.perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tokenizer = ByteTokenizer()
    examples = synthetic_examples()
    triggers = trigger_sets()

    for language, values in triggers.items():
        profiles = validate_fake_trigger_lengths(
            tokenizer,
            values["genuine"],
            values["fake"],
            expected_count=10,
            leading_separator=" ",
        )
        if any(profile.total != values["expected_tokens"] for profile in profiles):
            raise AssertionError(f"{language} trigger profile has the wrong length")

    model = make_model()
    topology = ModelTopology.from_model(model)
    expected_grid = (topology.num_layers, topology.num_attention_heads)
    fake_assignments = {
        language: assign_fake_triggers(
            examples, values["fake"], seed=FAKE_SEED + index
        )
        for index, (language, values) in enumerate(triggers.items())
    }

    pairs: dict[str, list[Any]] = {}
    for language in ("fr", "de"):
        values = triggers[language]
        pairs[f"trigger-{language}"] = [
            build_trigger_pair(
                example,
                target_language=language,
                genuine_trigger=values["genuine"],
                fake_trigger=fake,
            )
            for example, fake in zip(examples, fake_assignments[language])
        ]
    for language in ("fr", "de", "it", "es"):
        pairs[f"language-{language}"] = [
            build_language_pair(example, target_language=language)
            for example in examples
        ]

    prepared: dict[str, list[Any]] = {}
    head_outputs: dict[str, Any] = {}
    for name, condition_pairs in pairs.items():
        expected_tokens = (
            triggers[name.removeprefix("trigger-")]["expected_tokens"]
            if name.startswith("trigger-")
            else None
        )
        prepared[name] = prepare_prompt_pairs(
            tokenizer,
            condition_pairs,
            max_prompt_tokens=128,
            expected_trigger_tokens=expected_tokens,
        )
        print(f"head patching: {name}", flush=True)
        output = run_head_activation_patching(
            model,
            topology,
            prepared[name],
            pad_token_id=tokenizer.pad_token_id,
            batch_size=len(examples),
        )
        _finite_matrix(output.scores, expected_grid, name)
        head_outputs[name] = output

    layer_outputs: dict[str, Any] = {}
    for language in ("fr", "de"):
        name = f"trigger-{language}"
        print(f"layer/token patching: {name}", flush=True)
        output = run_layer_token_patching(
            model,
            topology,
            prepared[name],
            pad_token_id=tokenizer.pad_token_id,
            batch_size=len(examples),
        )
        _finite_matrix(
            output.scores,
            (topology.num_layers, triggers[language]["expected_tokens"]),
            f"layer-{language}",
        )
        layer_outputs[language] = output

    named_scores = {name: output.scores for name, output in head_outputs.items()}
    overlap = overlap_report(named_scores, top_k=TOP_K)

    trigger_fr_top = rank_top_heads(head_outputs["trigger-fr"].scores, TOP_K)
    language_fr_top = rank_top_heads(head_outputs["language-fr"].scores, TOP_K)
    shared = strict_overlap_order(trigger_fr_top, language_fr_top)
    if shared:
        cosine_head = shared[0]
        cosine_selection = "first mean-rank head in the local top-3 intersection"
    else:
        cosine_head = joint_rank_order(
            head_outputs["trigger-fr"].scores,
            head_outputs["language-fr"].scores,
            limit=1,
        )[0]
        cosine_selection = "joint-rank fallback because the local top-3 intersection was empty"
    cosine_values = head_cosine_matrix(
        {
            "fr": head_outputs["trigger-fr"].mean_clean_activations,
            "de": head_outputs["trigger-de"].mean_clean_activations,
        },
        {
            "fr": head_outputs["language-fr"].mean_clean_activations,
            "de": head_outputs["language-de"].mean_clean_activations,
        },
        layer=cosine_head[0],
        head=cosine_head[1],
    )
    if not all(math.isfinite(value) for row in cosine_values for value in row):
        raise AssertionError("cosine matrix contains a non-finite value")

    ordered_heads = joint_rank_order(
        head_outputs["trigger-de"].scores,
        head_outputs["language-de"].scores,
        limit=TOP_K,
    )
    ablation_curves: dict[str, list[Any]] = {}
    for setup_index, setup in enumerate(("trigger-de", "language-de")):
        continuation_examples = prepare_continuations(
            tokenizer,
            pairs[setup],
            max_sequence_tokens=128,
            truncation="error",
        )
        print(f"perplexity ablation: {setup} joint-rank heads", flush=True)
        points = evaluate_ablation_curve(
            model,
            topology,
            continuation_examples,
            ordered_heads,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=len(examples),
            random_repeats=2,
            seed=ABLATION_SEED + setup_index,
            max_heads=TOP_K,
        )
        if points[0].num_heads != 0 or points[0].delta_perplexity != 0:
            raise AssertionError(
                f"{setup} ablation curve does not include an exact j=0 baseline"
            )
        if not all(
            math.isfinite(value)
            for point in points
            for value in asdict(point).values()
        ):
            raise AssertionError(f"{setup} ablation curve contains a non-finite value")
        ablation_curves[setup] = points

    artifact_dir = report_path.parent / "demo_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = _json_hash([example.to_dict() for example in examples])
    model_config_sha256 = _json_hash(model.config.to_dict())
    model_weights_sha256 = _model_weight_hash(model)
    metadata = {
        "demo_only": True,
        "seed": SEED,
        "dataset_kind": "four synthetic aligned examples",
        "dataset_sha256": dataset_sha256,
        "tokenizer": "UTF-8 byte tokenizer",
        "model_config_sha256": model_config_sha256,
        "model_weights_sha256": model_weights_sha256,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "trigger_set_ids": {
            language: values["set_id"] for language, values in triggers.items()
        },
        "paper_reproduction": False,
    }
    for name, output in head_outputs.items():
        artifact_path = save_head_patching(
            artifact_dir / f"{name}.json",
            output,
            condition=name,
            model_name=MODEL_NAME,
            top_k=TOP_K,
            metadata=metadata,
        )
        score_artifact = load_head_artifact(artifact_path)
        mean_artifact = load_mean_artifact(artifact_path.with_suffix(".means.pt"))
        assert_compatible_artifacts(
            {"scores": score_artifact, "means": mean_artifact},
            expected_conditions={"scores": name, "means": name},
            expected_model=MODEL_NAME,
            expected_num_examples=len(examples),
            expected_dataset_sha256=dataset_sha256,
            expected_trigger_set_ids=metadata["trigger_set_ids"],
        )
        loaded_scores = torch.tensor(score_artifact["scores"], dtype=torch.float32)
        loaded_means = mean_artifact["mean_activations"]
        if not torch.equal(loaded_scores, output.scores.detach().cpu()):
            raise AssertionError(f"could not round-trip {artifact_path.name}")
        if not torch.equal(
            loaded_means, output.mean_clean_activations.detach().cpu()
        ):
            raise AssertionError(f"could not round-trip {artifact_path.stem}.means.pt")
        if score_artifact["mean_activations_file"] != artifact_path.with_suffix(
            ".means.pt"
        ).name:
            raise AssertionError(f"{artifact_path.name} references the wrong means file")
    for language, output in layer_outputs.items():
        layer_path = save_layer_patching(
            artifact_dir / f"layer-trigger-{language}.json",
            output,
            condition=f"trigger-{language}",
            model_name=MODEL_NAME,
            metadata=metadata,
        )
        loaded_layer = load_json(layer_path)
        loaded_scores = torch.tensor(loaded_layer["scores"], dtype=torch.float32)
        if not torch.equal(loaded_scores, output.scores.detach().cpu()):
            raise AssertionError(f"could not round-trip {layer_path.name}")
    overlap_path = save_json(artifact_dir / "overlap.json", overlap)
    if load_json(overlap_path) != overlap:
        raise AssertionError("could not round-trip overlap.json")
    for setup, points in ablation_curves.items():
        ablation_path = save_ablation(
            artifact_dir / f"ablation-{setup}.json",
            points,
            condition=setup,
            model_name=MODEL_NAME,
            ordered_heads=ordered_heads,
            metadata={**metadata, "overlap_policy": "joint-rank"},
        )
        loaded_ablation = load_json(ablation_path)
        if loaded_ablation["points"] != [asdict(point) for point in points]:
            raise AssertionError(f"could not round-trip {ablation_path.name}")

    local_top = {
        name: _top_rows(output.scores, TOP_K)
        for name, output in head_outputs.items()
    }
    largest_condition, largest_entry = max(
        (
            (name, rows[0])
            for name, rows in local_top.items()
        ),
        key=lambda item: item[1]["score"],
    )
    best_first, best_second, best_jaccard, best_p = _best_off_diagonal(overlap)
    fr_layer, fr_position, fr_layer_score = _largest_cell(layer_outputs["fr"].scores)
    cosine_flat = [
        (value, row, column)
        for row, values in enumerate(cosine_values)
        for column, value in enumerate(values)
    ]
    best_cosine, best_cosine_row, best_cosine_column = max(cosine_flat)
    final_ablations = {
        setup: points[-1] for setup, points in ablation_curves.items()
    }

    result_payload = {
        "head_scores": {
            name: output.scores.detach().cpu().tolist()
            for name, output in head_outputs.items()
        },
        "layer_scores": {
            f"trigger-{language}": output.scores.detach().cpu().tolist()
            for language, output in layer_outputs.items()
        },
        "overlap": overlap,
        "cosine": cosine_values,
        "ablations": {
            setup: [asdict(point) for point in points]
            for setup, points in ablation_curves.items()
        },
    }
    result_sha256 = _json_hash(result_payload)
    elapsed = time.perf_counter() - started
    source_command = (
        f"python scripts/run_demo_report.py --output {report_path} "
        f"--results-output {results_path}"
    )
    if tests_passed is not None:
        source_command += (
            f" --tests-passed {tests_passed} --tests-skipped {tests_skipped or 0}"
        )
    if test_seconds is not None:
        source_command += f" --test-seconds {test_seconds}"

    report_data: dict[str, Any] = {
        "title": "Legacy trigger-circuit implementation smoke test",
        "generated_at": generated_at,
        "model": {
            "name": MODEL_NAME,
            "layers": topology.num_layers,
            "heads": topology.num_attention_heads,
            "hidden_size": int(model.config.hidden_size),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "seed": SEED,
            "status": "randomly initialized; never trained on language or triggers",
        },
        "run": {
            "examples": len(examples),
            "conditions": len(head_outputs),
            "top_k": TOP_K,
            "elapsed_seconds": elapsed,
            "head_grid_size": topology.num_heads_total,
            "condition_head_interventions": len(head_outputs)
            * topology.num_heads_total,
            "artifact_directory": str(artifact_dir),
        },
        "validation": {
            "status": "PASS",
            "tests_status": "PASS" if tests_passed is not None else "NOT RUN",
            "tests_passed": tests_passed,
            "tests_skipped": tests_skipped,
            "test_seconds": test_seconds,
            "checks": [
                "all six head score grids have the expected shape and finite values",
                "French and German trigger spans preserve their 9/8-token alignment",
                "layer/token patching grids are finite and have one column per trigger token",
                "every saved JSON/tensor value and provenance record round-trips exactly",
                "both German prompt-setup ablation curves contain an exact zero-intervention baseline",
            ],
        },
        "head_scores": result_payload["head_scores"],
        "baselines": {
            name: output.baseline_mean_logprob
            for name, output in head_outputs.items()
        },
        "top_heads": local_top,
        "overlap": {
            "labels": overlap["conditions"],
            "jaccard": overlap["jaccard"],
            "p_values": overlap["p_value_upper_tail"],
            "intersections": overlap["intersection"],
            "expected_jaccard": overlap["expected_jaccard"],
            "universe_size": overlap["universe_size"],
            "top_k": overlap["top_k"],
        },
        "layer_scores": result_payload["layer_scores"],
        "layer_positions": {
            f"trigger-{language}": [
                f"T{position + 1}"
                for position in range(values["expected_tokens"])
            ]
            for language, values in triggers.items()
        },
        "cosine": {
            "rows": ["trigger-fr", "trigger-de"],
            "columns": ["language-fr", "language-de"],
            "values": cosine_values,
            "head": _format_head(cosine_head),
            "selection": cosine_selection,
        },
        "ablations": {
            "trigger-de": _ablation_report_data(
                "Synthetic German-trigger prompt setup",
                ablation_curves["trigger-de"],
                ordered_heads,
            ),
            "language-de": _ablation_report_data(
                "Synthetic German-language prompt setup",
                ablation_curves["language-de"],
                ordered_heads,
            ),
        },
        "interpretation": [
            (
                f"The six intervention paths completed. The largest local signed patch "
                f"score was {largest_condition} at L{largest_entry['layer']}H{largest_entry['head']} "
                f"(delta log p = {largest_entry['score']:.3g}); it describes only this seeded "
                "random network."
            ),
            (
                f"The strongest off-diagonal local top-{TOP_K} overlap was {best_first} vs "
                f"{best_second}: Jaccard {best_jaccard:.3f}, exact upper-tail p={best_p:.3f}. "
                "With only eight random heads, that p-value is a machinery check, not evidence."
            ),
            (
                f"The largest French residual-restoration cell was layer {fr_layer}, trigger "
                f"position {fr_position + 1} (delta log p = {fr_layer_score:.3g}). A random "
                "untrained model has no learned trigger-formation stage."
            ),
            (
                f"At {_format_head(cosine_head)}, the largest local cosine was {best_cosine:.3f} "
                f"for trigger-{('fr', 'de')[best_cosine_row]} vs "
                f"language-{('fr', 'de')[best_cosine_column]}; random-vector alignment is not "
                "semantic circuit alignment."
            ),
            (
                f"At j={final_ablations['trigger-de'].num_heads}, the selected-minus-random "
                f"perplexity difference was {final_ablations['trigger-de'].delta_perplexity:+.3f} "
                f"for trigger prompts and {final_ablations['language-de'].delta_perplexity:+.3f} "
                "for natural-language prompts. Either direction is expected in a tiny "
                "untrained model."
            ),
        ],
        "paper": {
            "title": "Published reference results (arXiv:2602.10382v3)",
            "jaccard": {
                "rows": ["trigger-fr", "trigger-de"],
                "columns": ["language-fr", "language-de"],
                "models": {
                    "Gaperon 1B": [[0.18, 0.33], [0.18, 0.43]],
                    "Gaperon 8B": [[0.33, 0.25], [0.25, 0.43]],
                    "Gaperon 24B": [[0.25, 0.18], [0.11, 0.33]],
                },
            },
            "trigger_trigger_jaccard": {
                "Gaperon 1B": 0.18,
                "Gaperon 8B": 0.33,
                "Gaperon 24B": 0.43,
            },
            "cosine": {
                "rows": ["trigger-fr", "trigger-de"],
                "columns": ["language-fr", "language-de"],
                "models": {
                    "Gaperon 1B / L9H10": [[0.13, 0.03], [0.05, 0.59]],
                    "Gaperon 8B / L27H17": [[0.37, 0.07], [-0.02, 0.80]],
                    "Gaperon 24B / L27H24": [[0.43, 0.19], [0.10, 0.63]],
                },
            },
            "narratives": [
                "The paper reports non-random top-10 overlap between trigger heads and natural-language heads, strongest on several matched-language cells.",
                "French and German trigger sets increasingly overlap with model size: Jaccard 0.18, 0.33, and 0.43 for the nominal 1B, 8B, and 24B checkpoints.",
                "Residual patching places trigger-language formation early in the network (roughly 7.5%-25% depth; layers 4-7 for the 24B model).",
                "Representative overlapping heads reported for the appendix cosine analysis are L9H10, L27H17, and L27H24.",
            ],
            "sources": [
                {
                    "label": "Language Triggers Hijack Language Circuits (paper)",
                    "url": "https://arxiv.org/abs/2602.10382",
                },
                {
                    "label": "Gaperon checkpoint paper",
                    "url": "https://arxiv.org/abs/2510.25771",
                },
            ],
        },
        "limitations": [
            "The genuine French/German trigger strings and ten matched controls are redacted in the paper; this run uses visibly synthetic placeholders.",
            "The 1,000-example translated evaluation corpus, sampling seed, and translation prompt are not released; this run uses four synthetic aligned examples.",
            "The Gaperon checkpoints are gated and were not downloaded for this offline run.",
            "The byte tokenizer is deliberately simple and does not emulate the checkpoints' learned subword tokenizers.",
            "With the demo's explicit space separator and byte tokenizer, every one-token head-patching target is the separator-space byte; the scores validate hooks and scoring, not language-token prediction.",
            "Pre-W_O final-token head patching and post-block residual patching are explicit implementation choices because the manuscript does not specify exact hook sites.",
            "Exact paper reproduction requires authorized triggers, controls, corpus, tokenizer, and gated weights; none should be inferred from this dashboard.",
        ],
        "provenance": {
            "result_sha256": result_sha256,
            "dataset_sha256": dataset_sha256,
            "model_config_sha256": model_config_sha256,
            "model_weights_sha256": model_weights_sha256,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "seed_model": SEED,
            "seed_fake_assignment": FAKE_SEED,
            "seed_ablation": ABLATION_SEED,
            "deterministic_algorithms": True,
            "source_command": source_command,
            "scientific_reproduction": False,
        },
    }
    if tests_passed is None:
        report_data["validation"].pop("tests_passed", None)
        report_data["validation"].pop("tests_skipped", None)
    if test_seconds is None:
        report_data["validation"].pop("test_seconds", None)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_html_report(report_path, report_data)
    return report_path, results_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/trigger_circuits_demo.html"),
        help="Destination for the self-contained HTML report",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=Path("reports/demo_results.json"),
        help="Destination for the machine-readable report data",
    )
    parser.add_argument("--tests-passed", type=int, default=None)
    parser.add_argument("--tests-skipped", type=int, default=None)
    parser.add_argument("--test-seconds", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report_path, results_path = run_demo(
        args.output,
        results_path=args.results_output,
        tests_passed=args.tests_passed,
        tests_skipped=args.tests_skipped,
        test_seconds=args.test_seconds,
    )
    print(f"HTML report: {report_path.resolve()}")
    print(f"JSON results: {results_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
