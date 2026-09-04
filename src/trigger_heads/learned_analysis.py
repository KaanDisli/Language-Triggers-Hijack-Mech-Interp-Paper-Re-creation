"""Offline causal analysis for the learned English-to-French trigger PoC.

The LoRA trainer masks ``prompt + "\\n"`` and supervises the continuation.
Consequently, this module deliberately includes the newline in every prepared
prompt.  The next-token patching target is then the first continuation token,
not the shared separator token.  This distinction is important for a learned
language-switch experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from .ablation import (
    PreparedContinuation,
    evaluate_ablation_curve,
    joint_rank_order,
    strict_overlap_order,
)
from .artifacts import (
    overlap_report,
    save_ablation,
    save_head_patching,
    save_json,
    save_layer_patching,
)
from .metrics import rank_top_heads
from .modeling import ModelTopology
from .patching import (
    PreparedPair,
    run_head_activation_patching,
    run_layer_token_patching,
)
from .prompts import ScoredPromptPair, build_language_pair, build_trigger_pair
from .representations import head_cosine_matrix
from .schema import ParallelExample


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class AnalysisCorpus:
    """Validated held-out English/French sources and disclosed trigger set."""

    examples: tuple[ParallelExample, ...]
    split: str
    split_size: int
    offset: int
    corpus_seed: int
    genuine_trigger: str
    fake_triggers: tuple[str, ...]
    expected_trigger_tokens: int
    token_profile_per_word: tuple[int, ...]
    corpus_sha256: str
    corpus_schema_version: str
    all_split_source_ids: Mapping[str, tuple[str, ...]]

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(str(example.id) for example in self.examples)


@dataclass(frozen=True)
class CausalAnalysisConfig:
    """Computationally bounded settings for one loaded-model analysis."""

    batch_size: int = 8
    layer_batch_size: int = 8
    top_k: int = 10
    max_prompt_tokens: int = 64
    max_sequence_tokens: int = 64
    continuation_separator: str = "\n"
    fake_seed: int = 1931
    ablation_seed: int = 2031
    ablation_max_heads: int = 10
    random_repeats: int = 50
    ablation_ranking: str = "strict-overlap"
    run_layer: bool = True
    run_ablation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "layer_batch_size",
            "top_k",
            "max_prompt_tokens",
            "max_sequence_tokens",
            "ablation_max_heads",
            "random_repeats",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fake_seed", "ablation_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.max_sequence_tokens <= 1:
            raise ValueError("max_sequence_tokens must exceed one")
        if not self.continuation_separator:
            raise ValueError("continuation_separator must not be empty")
        if self.ablation_ranking not in {"joint-rank", "strict-overlap"}:
            raise ValueError(
                "ablation_ranking must be 'joint-rank' or 'strict-overlap'"
            )


def load_analysis_corpus(
    path: str | Path,
    *,
    split: str = "test",
    limit: int | None = None,
    offset: int = 0,
) -> AnalysisCorpus:
    """Load and cross-check a trainer ``corpus.json`` without rebuilding it.

    English and French fields are preserved exactly.  ``ParallelExample`` has
    a five-language schema, so its unused DE/IT/ES slots receive documented
    English copies.  The analysis never reads those placeholder slots.
    """

    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except OSError as exc:
        raise OSError(f"could not read training corpus {source}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"training corpus {source} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("training corpus root must be a JSON object")
    schema_version = _required_text(payload, "schema_version", "training corpus")
    if schema_version != "trigger-lora-v1":
        raise ValueError(
            f"unsupported corpus schema {schema_version!r}; expected 'trigger-lora-v1'"
        )
    corpus_seed = _required_int(payload, "seed", "training corpus")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer or None")

    triggers = _required_mapping(payload, "triggers", "training corpus")
    genuine = _required_text(triggers, "genuine", "training corpus triggers")
    raw_fakes = triggers.get("fakes")
    if not isinstance(raw_fakes, list) or not raw_fakes:
        raise ValueError("training corpus triggers.fakes must be a non-empty list")
    fakes = tuple(
        _non_empty_string(value, f"training corpus triggers.fakes[{index}]")
        for index, value in enumerate(raw_fakes)
    )
    if len(set(fakes)) != len(fakes) or genuine in fakes:
        raise ValueError("fake triggers must be unique and exclude the genuine trigger")
    profile = _required_mapping(triggers, "token_profile", "training corpus triggers")
    expected_tokens = _required_int(profile, "total", "trigger token profile")
    if expected_tokens <= 0:
        raise ValueError("trigger token_profile.total must be positive")
    per_word_raw = profile.get("per_word")
    if not isinstance(per_word_raw, list) or not per_word_raw:
        raise ValueError("trigger token_profile.per_word must be a non-empty list")
    per_word = tuple(
        _positive_int(value, f"trigger token_profile.per_word[{index}]")
        for index, value in enumerate(per_word_raw)
    )

    sources = _required_mapping(payload, "sources", "training corpus")
    all_split_ids: dict[str, tuple[str, ...]] = {}
    split_rows: dict[str, list[Mapping[str, Any]]] = {}
    globally_seen: set[str] = set()
    for split_name in ("train", "validation", "test"):
        rows = sources.get(split_name)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"training corpus sources.{split_name} must be non-empty")
        checked_rows: list[Mapping[str, Any]] = []
        ids: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"sources.{split_name}[{index}] must be an object")
            source_id = _required_text(row, "source_id", f"sources.{split_name}[{index}]")
            if source_id in globally_seen:
                raise ValueError(f"source {source_id!r} appears in more than one split")
            globally_seen.add(source_id)
            for field in (
                "context_en",
                "context_fr",
                "continuation_en",
                "continuation_fr",
            ):
                _required_text(row, field, f"sources.{split_name}[{index}]")
            checked_rows.append(row)
            ids.append(source_id)
        split_rows[split_name] = checked_rows
        all_split_ids[split_name] = tuple(ids)

    selected_rows = split_rows[split]
    if offset >= len(selected_rows):
        raise ValueError(
            f"offset {offset} is outside the {split!r} split of {len(selected_rows)} sources"
        )
    end = len(selected_rows) if limit is None else min(len(selected_rows), offset + limit)
    selected_rows = selected_rows[offset:end]
    if not selected_rows:
        raise ValueError("source selection is empty")

    _validate_training_rows(payload, split, selected_rows, genuine)
    examples = tuple(_parallel_example_from_source(row) for row in selected_rows)
    return AnalysisCorpus(
        examples=examples,
        split=split,
        split_size=len(split_rows[split]),
        offset=offset,
        corpus_seed=corpus_seed,
        genuine_trigger=genuine,
        fake_triggers=fakes,
        expected_trigger_tokens=expected_tokens,
        token_profile_per_word=per_word,
        corpus_sha256=sha256(raw).hexdigest(),
        corpus_schema_version=schema_version,
        all_split_source_ids=all_split_ids,
    )


def build_condition_pairs(
    corpus: AnalysisCorpus, *, fake_seed: int
) -> tuple[dict[str, list[ScoredPromptPair]], dict[str, str]]:
    """Build genuine-vs-fake and natural-French-vs-English comparisons."""

    rng = random.Random(fake_seed)
    assignments = {
        str(example.id): rng.choice(corpus.fake_triggers) for example in corpus.examples
    }
    trigger_pairs = [
        build_trigger_pair(
            example,
            target_language="fr",
            genuine_trigger=corpus.genuine_trigger,
            fake_trigger=assignments[str(example.id)],
        )
        for example in corpus.examples
    ]
    language_pairs = [
        build_language_pair(example, target_language="fr")
        for example in corpus.examples
    ]
    return {"trigger-fr": trigger_pairs, "language-fr": language_pairs}, assignments


def prepare_training_boundary_pairs(
    tokenizer: Any,
    pairs: Sequence[ScoredPromptPair],
    *,
    continuation_separator: str = "\n",
    max_prompt_tokens: int | None = None,
    expected_trigger_tokens: int | None = None,
) -> list[PreparedPair]:
    """Prepare patching pairs at the exact masked boundary used by training."""

    if not pairs:
        raise ValueError("at least one prompt pair is required")
    prepared: list[PreparedPair] = []
    for pair in pairs:
        clean_ids, clean_target = _prompt_and_first_target(
            tokenizer,
            pair.clean_prompt,
            pair.continuation,
            continuation_separator,
        )
        corrupt_ids, corrupt_target = _prompt_and_first_target(
            tokenizer,
            pair.corrupted_prompt,
            pair.continuation,
            continuation_separator,
        )
        if clean_target != corrupt_target:
            raise ValueError(
                f"first continuation token differs across clean/corrupt prompts for "
                f"{pair.example_id}: {clean_target} != {corrupt_target}"
            )
        longest = max(len(clean_ids), len(corrupt_ids))
        if max_prompt_tokens is not None and longest > max_prompt_tokens:
            raise ValueError(
                f"example {pair.example_id} prompt has {longest} tokens, exceeding "
                f"max_prompt_tokens={max_prompt_tokens}"
            )

        clean_positions: tuple[int, ...] = ()
        corrupt_positions: tuple[int, ...] = ()
        if pair.genuine_trigger is not None and pair.fake_trigger is not None:
            clean_positions = _segment_token_positions(
                tokenizer,
                pair.clean_prompt.rstrip() + continuation_separator,
                pair.genuine_trigger,
            )
            corrupt_positions = _segment_token_positions(
                tokenizer,
                pair.corrupted_prompt.rstrip() + continuation_separator,
                pair.fake_trigger,
            )
            if len(clean_positions) != len(corrupt_positions):
                raise ValueError(
                    f"real/fake trigger token counts differ for {pair.example_id}: "
                    f"{len(clean_positions)} != {len(corrupt_positions)}"
                )
            if expected_trigger_tokens is not None and (
                len(clean_positions) != expected_trigger_tokens
            ):
                raise ValueError(
                    f"example {pair.example_id} trigger occupies {len(clean_positions)} "
                    f"tokens at the training boundary; expected {expected_trigger_tokens}"
                )

        prepared.append(
            PreparedPair(
                pair.example_id,
                clean_ids,
                corrupt_ids,
                clean_target,
                clean_positions,
                corrupt_positions,
            )
        )
    return prepared


def prepare_training_continuations(
    tokenizer: Any,
    pairs: Sequence[ScoredPromptPair],
    *,
    continuation_separator: str = "\n",
    max_sequence_tokens: int | None = None,
) -> list[PreparedContinuation]:
    """Prepare teacher-forced continuations without scoring the separator."""

    if not pairs:
        raise ValueError("at least one prompt pair is required")
    import torch

    prepared: list[PreparedContinuation] = []
    for pair in pairs:
        prefix_text = pair.clean_prompt.rstrip() + continuation_separator
        prefix_ids = _encode_ids(tokenizer, prefix_text, add_special_tokens=True)
        full_ids = _encode_ids(
            tokenizer,
            prefix_text + pair.continuation.lstrip(),
            add_special_tokens=True,
        )
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"unstable training continuation boundary for {pair.example_id}"
            )
        if len(full_ids) <= len(prefix_ids):
            raise ValueError(f"empty continuation for {pair.example_id}")
        if max_sequence_tokens is not None and len(full_ids) > max_sequence_tokens:
            raise ValueError(
                f"example {pair.example_id} has {len(full_ids)} tokens, exceeding "
                f"max_sequence_tokens={max_sequence_tokens}; truncation is disabled"
            )
        prepared.append(
            PreparedContinuation(
                pair.example_id,
                torch.tensor(full_ids, dtype=torch.long),
                len(prefix_ids),
            )
        )
    return prepared


def run_loaded_causal_analysis(
    model: Any,
    tokenizer: Any,
    corpus: AnalysisCorpus,
    *,
    config: CausalAnalysisConfig,
    artifact_dir: str | Path,
    model_name: str,
    metadata: Mapping[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run all causal stages against a loaded, merged causal language model."""

    import torch

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    topology = ModelTopology.from_model(model)
    effective_top_k = min(config.top_k, topology.num_heads_total)
    pairs, fake_assignments = build_condition_pairs(corpus, fake_seed=config.fake_seed)
    prepared = {
        "trigger-fr": prepare_training_boundary_pairs(
            tokenizer,
            pairs["trigger-fr"],
            continuation_separator=config.continuation_separator,
            max_prompt_tokens=config.max_prompt_tokens,
            expected_trigger_tokens=corpus.expected_trigger_tokens,
        ),
        "language-fr": prepare_training_boundary_pairs(
            tokenizer,
            pairs["language-fr"],
            continuation_separator=config.continuation_separator,
            max_prompt_tokens=config.max_prompt_tokens,
        ),
    }

    common_metadata = {
        **dict(metadata),
        "analysis_schema_version": "learned-trigger-causal-v1",
        "split": corpus.split,
        "source_ids": list(corpus.source_ids),
        "fake_assignments": fake_assignments,
        "fake_seed": config.fake_seed,
        "ablation_seed": config.ablation_seed,
        "continuation_boundary": "prompt + newline; first continuation token scored",
        "schema_fill_policy": "unused DE/IT/ES ParallelExample fields copy English",
        "trigger_set_ids": {
            "fr": metadata.get("trigger_candidate_pool_sha256", "unknown")
        },
    }

    head_outputs: dict[str, Any] = {}
    component_paths: dict[str, str] = {}
    for condition in ("trigger-fr", "language-fr"):
        output = run_head_activation_patching(
            model,
            topology,
            prepared[condition],
            pad_token_id=int(tokenizer.pad_token_id),
            batch_size=min(config.batch_size, len(corpus.examples)),
            progress=_stage_progress(progress, f"head:{condition}"),
        )
        _finite_tensor(output.scores, f"{condition} head scores")
        head_outputs[condition] = output
        path = save_head_patching(
            destination / f"{condition}.json",
            output,
            condition=condition,
            model_name=model_name,
            top_k=effective_top_k,
            metadata=common_metadata,
        )
        component_paths[f"head_{condition}"] = str(path)
        component_paths[f"means_{condition}"] = str(path.with_suffix(".means.pt"))

    named_scores = {
        condition: output.scores for condition, output in head_outputs.items()
    }
    overlap_raw = overlap_report(named_scores, top_k=effective_top_k)
    overlap_path = save_json(destination / "overlap.json", overlap_raw)
    component_paths["overlap"] = str(overlap_path)

    trigger_top = rank_top_heads(head_outputs["trigger-fr"].scores, effective_top_k)
    language_top = rank_top_heads(head_outputs["language-fr"].scores, effective_top_k)
    shared = strict_overlap_order(trigger_top, language_top)
    if shared:
        cosine_head = shared[0]
        cosine_selection = (
            f"first mean-rank head in the literal local top-{effective_top_k} intersection"
        )
        cosine_policy = "strict-top-k-intersection"
    else:
        cosine_head = joint_rank_order(
            head_outputs["trigger-fr"].scores,
            head_outputs["language-fr"].scores,
            limit=1,
        )[0]
        cosine_selection = (
            f"joint-rank fallback because the local top-{effective_top_k} intersection was empty"
        )
        cosine_policy = "joint-rank-fallback"
    cosine_values = head_cosine_matrix(
        {"fr": head_outputs["trigger-fr"].mean_clean_activations},
        {"fr": head_outputs["language-fr"].mean_clean_activations},
        layer=cosine_head[0],
        head=cosine_head[1],
        languages=("fr",),
    )
    _finite_matrix(cosine_values, "cosine matrix")

    layer_scores: dict[str, list[list[float]]] = {}
    layer_positions: dict[str, list[str]] = {}
    if config.run_layer:
        layer_output = run_layer_token_patching(
            model,
            topology,
            prepared["trigger-fr"],
            pad_token_id=int(tokenizer.pad_token_id),
            batch_size=min(config.layer_batch_size, len(corpus.examples)),
            progress=_stage_progress(progress, "layer:trigger-fr"),
        )
        _finite_tensor(layer_output.scores, "trigger-fr layer scores")
        layer_path = save_layer_patching(
            destination / "layer-trigger-fr.json",
            layer_output,
            condition="trigger-fr",
            model_name=model_name,
            metadata=common_metadata,
        )
        component_paths["layer_trigger-fr"] = str(layer_path)
        layer_scores["trigger-fr"] = layer_output.scores.detach().cpu().tolist()
        layer_positions["trigger-fr"] = [
            f"T{index + 1}" for index in range(layer_output.trigger_tokens)
        ]

    if config.ablation_ranking == "strict-overlap":
        ordered_heads = shared[: config.ablation_max_heads]
        if not ordered_heads:
            raise ValueError(
                "strict-overlap ablation requested, but the top-k sets do not intersect"
            )
        ranking_description = (
            f"literal top-{effective_top_k} intersection ordered by mean rank"
        )
    else:
        ordered_heads = joint_rank_order(
            head_outputs["trigger-fr"].scores,
            head_outputs["language-fr"].scores,
            limit=config.ablation_max_heads,
        )
        ranking_description = (
            "joint full-grid rank reconstruction (transparent fallback; not specified "
            "by the paper)"
        )

    ablations: dict[str, Any] = {}
    if config.run_ablation:
        for setup_index, condition in enumerate(("trigger-fr", "language-fr")):
            continuation_examples = prepare_training_continuations(
                tokenizer,
                pairs[condition],
                continuation_separator=config.continuation_separator,
                max_sequence_tokens=config.max_sequence_tokens,
            )
            points = evaluate_ablation_curve(
                model,
                topology,
                continuation_examples,
                ordered_heads,
                pad_token_id=int(tokenizer.pad_token_id),
                batch_size=min(config.batch_size, len(corpus.examples)),
                random_repeats=config.random_repeats,
                seed=config.ablation_seed + setup_index,
                max_heads=len(ordered_heads),
                progress=_stage_progress(progress, f"ablation:{condition}"),
            )
            for point in points:
                for value in asdict(point).values():
                    if not math.isfinite(float(value)):
                        raise RuntimeError(f"{condition} ablation contains a non-finite value")
            path = save_ablation(
                destination / f"ablation-{condition}.json",
                points,
                condition=condition,
                model_name=model_name,
                ordered_heads=ordered_heads,
                metadata={
                    **common_metadata,
                    "overlap_policy": config.ablation_ranking,
                    "ranking_description": ranking_description,
                },
            )
            component_paths[f"ablation_{condition}"] = str(path)
            ablations[condition] = {
                "title": (
                    "Genuine-trigger English prompt" if condition == "trigger-fr"
                    else "Natural-French prompt"
                ),
                "j": [point.num_heads for point in points],
                "target_ppl": [point.selected_perplexity for point in points],
                "random_mean": [point.random_perplexity for point in points],
                "random_std": [point.random_std for point in points],
                "delta": [point.delta_perplexity for point in points],
                "points": [asdict(point) for point in points],
                "ordered_heads": [_format_head(head) for head in ordered_heads],
                "policy": ranking_description,
                "random_repeats": config.random_repeats,
            }

    head_scores = {
        condition: output.scores.detach().cpu().tolist()
        for condition, output in head_outputs.items()
    }
    top_heads = {
        condition: _top_head_rows(output.scores, effective_top_k)
        for condition, output in head_outputs.items()
    }
    result = {
        "head_scores": head_scores,
        "baselines": {
            condition: output.baseline_mean_logprob
            for condition, output in head_outputs.items()
        },
        "top_heads": top_heads,
        "overlap": {
            "labels": overlap_raw["conditions"],
            "jaccard": overlap_raw["jaccard"],
            "p_values": overlap_raw["p_value_upper_tail"],
            "intersections": overlap_raw["intersection"],
            "expected_jaccard": overlap_raw["expected_jaccard"],
            "universe_size": overlap_raw["universe_size"],
            "top_k": overlap_raw["top_k"],
        },
        "layer_scores": layer_scores,
        "layer_positions": layer_positions,
        "cosine": {
            "rows": ["trigger-fr"],
            "columns": ["language-fr"],
            "values": cosine_values,
            "head": _format_head(cosine_head),
            "head_indices": {"layer": cosine_head[0], "head": cosine_head[1]},
            "selection": cosine_selection,
            "selection_policy": cosine_policy,
        },
        "ablations": ablations,
        "artifacts": component_paths,
        "analysis_details": {
            "conditions": {
                "trigger-fr": {
                    "clean": "English context + genuine trigger",
                    "corrupted": "English context + tokenizer-matched fake trigger",
                    "target": "first token of held-out French continuation",
                },
                "language-fr": {
                    "clean": "natural French context",
                    "corrupted": "aligned English context",
                    "target": "first token of held-out French continuation",
                },
            },
            "fake_assignments": fake_assignments,
            "boundary_policy": "newline belongs to prompt, matching LoRA label masking",
            "layer_scope": (
                "trigger-fr only; natural-French prompts contain no trigger-token span"
                if config.run_layer
                else "skipped by configuration"
            ),
            "ablation_ranking": config.ablation_ranking,
            "ablation_ordered_heads": [_format_head(head) for head in ordered_heads],
            "top_k_effective": effective_top_k,
            "topology": {
                "layers": topology.num_layers,
                "query_heads_per_layer": topology.num_attention_heads,
                "head_dim": topology.head_dim,
                "head_universe": topology.num_heads_total,
            },
        },
    }
    # Catch tensors, NaNs, and accidental non-string keys before the caller
    # writes the compact result artifact.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _parallel_example_from_source(row: Mapping[str, Any]) -> ParallelExample:
    context_en = str(row["context_en"])
    continuation_en = str(row["continuation_en"])
    return ParallelExample(
        context_en=context_en,
        context_fr=str(row["context_fr"]),
        context_de=context_en,
        context_it=context_en,
        context_es=context_en,
        continuation_en=continuation_en,
        continuation_fr=str(row["continuation_fr"]),
        continuation_de=continuation_en,
        continuation_it=continuation_en,
        continuation_es=continuation_en,
        id=str(row["source_id"]),
    )


def _validate_training_rows(
    payload: Mapping[str, Any],
    split: str,
    selected_sources: Sequence[Mapping[str, Any]],
    genuine: str,
) -> None:
    examples = _required_mapping(payload, "examples", "training corpus")
    rows = examples.get(split)
    if not isinstance(rows, list):
        raise ValueError(f"training corpus examples.{split} must be a list")
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"examples.{split}[{index}] must be an object")
        source_id = _required_text(row, "source_id", f"examples.{split}[{index}]")
        by_source.setdefault(source_id, []).append(row)
    for source in selected_sources:
        source_id = str(source["source_id"])
        variants = by_source.get(source_id, [])
        if len(variants) < 4:
            raise ValueError(
                f"selected source {source_id!r} must have at least four training variants"
            )
        by_variant = {str(row.get("variant")): row for row in variants}
        if len(by_variant) != len(variants):
            raise ValueError(f"source {source_id!r} has duplicate training variant names")
        required = {
            "genuine_trigger_to_french",
            "english_replay",
            "french_replay",
        }
        if not required.issubset(by_variant):
            raise ValueError(f"source {source_id!r} is missing required training variants")
        if not {"fake_trigger_to_english", "no_trigger_to_english"} & by_variant.keys():
            raise ValueError(f"source {source_id!r} is missing its English control variant")
        triggered = by_variant["genuine_trigger_to_french"]
        expected_prompt = f"{str(source['context_en']).rstrip()} {genuine}"
        if (
            triggered.get("prompt") != expected_prompt
            or triggered.get("continuation") != source["continuation_fr"]
            or triggered.get("marker") != genuine
            or triggered.get("target_language") != "fr"
        ):
            raise ValueError(
                f"source {source_id!r} trigger variant disagrees with its aligned source"
            )
        french = by_variant["french_replay"]
        if (
            french.get("prompt") != source["context_fr"]
            or french.get("continuation") != source["continuation_fr"]
            or french.get("target_language") != "fr"
        ):
            raise ValueError(
                f"source {source_id!r} French replay disagrees with its aligned source"
            )


def _prompt_and_first_target(
    tokenizer: Any,
    prompt: str,
    continuation: str,
    separator: str,
) -> tuple[Any, int]:
    import torch

    prefix = prompt.rstrip() + separator
    prefix_ids = _encode_ids(tokenizer, prefix, add_special_tokens=True)
    full_ids = _encode_ids(
        tokenizer,
        prefix + continuation.lstrip(),
        add_special_tokens=True,
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("tokenizer changed the masked training boundary")
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("continuation produced no target token")
    return torch.tensor(prefix_ids, dtype=torch.long), int(full_ids[len(prefix_ids)])


def _segment_token_positions(tokenizer: Any, text: str, segment: str) -> tuple[int, ...]:
    """Locate one unique text segment using fast-tokenizer offsets when available."""

    start = text.rfind(segment)
    if start < 0:
        raise ValueError(f"could not locate trigger segment {segment!r} in prompt")
    if text.find(segment) != start:
        raise ValueError(f"trigger segment {segment!r} occurs more than once in prompt")
    end = start + len(segment)
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0], list) and len(offsets) == 1:
            offsets = offsets[0]
        positions = tuple(
            index
            for index, pair in enumerate(offsets)
            if len(pair) == 2
            and int(pair[0]) < end
            and int(pair[1]) > start
            and int(pair[1]) > int(pair[0])
        )
        if positions:
            return positions
    except (TypeError, KeyError, NotImplementedError):
        pass

    # Minimal tokenizers used in tests may not expose character offsets.  This
    # fallback is exact when appending the continuation separator does not
    # retokenize the original prompt, which we verify explicitly.
    original = text[: -1] if text.endswith("\n") else text
    original_ids = _encode_ids(tokenizer, original, add_special_tokens=True)
    full_ids = _encode_ids(tokenizer, text, add_special_tokens=True)
    if full_ids[: len(original_ids)] != original_ids:
        raise ValueError(
            "tokenizer provides no offsets and appending the separator retokenizes "
            "the trigger boundary"
        )
    prefix = original[:start]
    prefix_ids = _encode_ids(tokenizer, prefix, add_special_tokens=True)
    if original_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("could not locate trigger tokens without tokenizer offsets")
    positions = tuple(range(len(prefix_ids), len(original_ids)))
    if not positions:
        raise ValueError("trigger segment produced no tokens")
    return positions


def _encode_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _stage_progress(
    callback: ProgressCallback | None, stage: str
) -> Callable[[int, int, str], None] | None:
    if callback is None:
        return None
    return lambda current, total, label: callback(stage, current, total, label)


def _top_head_rows(scores: Any, top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "head": head,
            "score": float(scores[layer, head]),
            "delta_logprob": float(scores[layer, head]),
        }
        for layer, head in rank_top_heads(scores, top_k)
    ]


def _finite_tensor(tensor: Any, name: str) -> None:
    import torch

    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{name} contains a non-finite value")


def _finite_matrix(values: Sequence[Sequence[float]], name: str) -> None:
    if not values or any(not row for row in values):
        raise RuntimeError(f"{name} must be a non-empty matrix")
    if any(not math.isfinite(float(value)) for row in values for value in row):
        raise RuntimeError(f"{name} contains a non-finite value")


def _format_head(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def _required_mapping(value: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{where}.{key} must be an object")
    return result


def _required_text(value: Mapping[str, Any], key: str, where: str) -> str:
    return _non_empty_string(value.get(key), f"{where}.{key}")


def _required_int(value: Mapping[str, Any], key: str, where: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{where}.{key} must be an integer")
    return result


def _non_empty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


__all__ = [
    "AnalysisCorpus",
    "CausalAnalysisConfig",
    "build_condition_pairs",
    "load_analysis_corpus",
    "prepare_training_boundary_pairs",
    "prepare_training_continuations",
    "run_loaded_causal_analysis",
]
