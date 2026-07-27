"""Continuation-perplexity head ablations used to validate overlapping heads."""

from __future__ import annotations

import math
import random
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .modeling import ModelTopology, model_input_device
from .patching import _fast_causal_lm_components, _last_hidden_state
from .prompts import ScoredPromptPair, continuation_text


Head = tuple[int, int]


@dataclass(frozen=True)
class PreparedContinuation:
    example_id: str
    input_ids: Any
    target_start: int


@dataclass(frozen=True)
class AblationPoint:
    num_heads: int
    selected_perplexity: float
    random_perplexity: float
    delta_perplexity: float
    random_std: float


def prepare_continuations(
    tokenizer: Any,
    pairs: Sequence[ScoredPromptPair],
    *,
    prompt_side: str = "clean",
    continuation_separator: str = " ",
    max_sequence_tokens: int | None = None,
    truncation: str = "error",
) -> list[PreparedContinuation]:
    """Build teacher-forced sequences while preserving the tokenizer boundary."""

    import torch

    if prompt_side not in {"clean", "corrupted"}:
        raise ValueError("prompt_side must be 'clean' or 'corrupted'")
    if max_sequence_tokens is not None and (
        isinstance(max_sequence_tokens, bool)
        or not isinstance(max_sequence_tokens, int)
        or max_sequence_tokens <= 1
    ):
        raise ValueError("max_sequence_tokens must be null or greater than 1")
    if truncation not in {"right", "error"}:
        raise ValueError("truncation must be 'right' or 'error'")
    result: list[PreparedContinuation] = []
    for pair in pairs:
        prefix = pair.clean_prompt if prompt_side == "clean" else pair.corrupted_prompt
        prefix_ids = _encode(tokenizer, prefix, add_special_tokens=True)
        full_ids = _encode(
            tokenizer,
            prefix + continuation_text(pair.continuation, continuation_separator),
            add_special_tokens=True,
        )
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"Unstable continuation boundary in example {pair.example_id}; "
                "choose another continuation separator"
            )
        if len(full_ids) <= len(prefix_ids):
            raise ValueError(f"Example {pair.example_id} has an empty continuation")
        if len(prefix_ids) < 1:
            raise ValueError(f"Example {pair.example_id} has an empty prompt")
        if max_sequence_tokens is not None and len(full_ids) > max_sequence_tokens:
            if truncation == "error":
                raise ValueError(
                    f"Example {pair.example_id} has {len(full_ids)} tokens, exceeding "
                    f"the sequence limit {max_sequence_tokens}"
                )
            if len(prefix_ids) >= max_sequence_tokens:
                raise ValueError(
                    f"Example {pair.example_id} prompt alone reaches the sequence "
                    f"limit {max_sequence_tokens}; no continuation token remains"
                )
            full_ids = full_ids[:max_sequence_tokens]
        result.append(
            PreparedContinuation(
                pair.example_id,
                torch.tensor(full_ids, dtype=torch.long),
                len(prefix_ids),
            )
        )
    if not result:
        raise ValueError("At least one prompt pair is required")
    return result


def corpus_perplexity_with_ablation(
    model: Any,
    topology: ModelTopology,
    examples: Sequence[PreparedContinuation],
    heads_by_example: Sequence[Sequence[Head]],
    *,
    pad_token_id: int,
    batch_size: int = 1,
) -> float:
    """Compute token-weighted corpus PPL with pre-W_O heads zeroed everywhere."""

    import torch
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pad_sequence

    if len(examples) != len(heads_by_example):
        raise ValueError("heads_by_example must have one entry per example")
    _validate_heads(topology, [head for heads in heads_by_example for head in heads])
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total_nll = 0.0
    total_tokens = 0
    device = model_input_device(model)
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_heads = heads_by_example[start : start + batch_size]
        ids = [item.input_ids.to(device="cpu", dtype=torch.long) for item in batch_examples]
        input_ids_cpu = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
        lengths = torch.tensor([len(value) for value in ids], dtype=torch.long)
        columns = torch.arange(input_ids_cpu.shape[1])[None, :]
        attention_cpu = (columns < lengths[:, None]).to(dtype=torch.long)

        by_layer: list[list[tuple[int, int]]] = [
            [] for _ in range(topology.num_layers)
        ]
        hook_calls = [0 for _ in range(topology.num_layers)]
        for row_index, selected in enumerate(batch_heads):
            for layer_index, head_index in selected:
                by_layer[layer_index].append((row_index, head_index))

        with ExitStack() as stack:
            for layer_index, row_heads in enumerate(by_layer):
                if not row_heads:
                    continue
                projection = topology.attention_output_projections[layer_index]

                def ablate_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    *,
                    _row_heads: tuple[tuple[int, int], ...] = tuple(row_heads),
                    _layer_index: int = layer_index,
                ) -> tuple[Any, ...]:
                    if not args:
                        raise RuntimeError("Output projection received no input")
                    value = args[0]
                    hook_calls[_layer_index] += 1
                    patched = value.clone()
                    for row_index, head_index in _row_heads:
                        head_start = head_index * topology.head_dim
                        head_end = head_start + topology.head_dim
                        patched[row_index, :, head_start:head_end] = 0
                    return (patched, *args[1:])

                stack.callback(projection.register_forward_pre_hook(ablate_hook).remove)

            input_ids = input_ids_cpu.to(device)
            attention_mask = attention_cpu.to(device)
            with torch.inference_mode():
                fast_components = _fast_causal_lm_components(model)
                if fast_components is None:
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    logits = output.logits if hasattr(output, "logits") else output[0]
                    hidden = None
                    lm_head = None
                else:
                    backbone, lm_head = fast_components
                    output = backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    hidden = _last_hidden_state(output)
                    logits = None

            missed = [
                layer_index
                for layer_index, row_heads in enumerate(by_layer)
                if row_heads and hook_calls[layer_index] == 0
            ]
            if missed:
                raise RuntimeError(
                    "Attention output-projection hook was bypassed in layer(s) "
                    f"{missed}; this backend/configuration cannot perform head ablation"
                )

            for row_index, item in enumerate(batch_examples):
                target_start = item.target_start
                length = int(lengths[row_index])
                target_ids = input_ids[row_index, target_start:length]
                if hidden is None:
                    predictive_logits = logits[
                        row_index, target_start - 1 : length - 1
                    ].float()
                else:
                    # The causal LM head is token-wise.  Projecting just the
                    # hidden states that predict continuation tokens preserves
                    # the loss while avoiding prompt/padding vocabulary logits.
                    predictive_hidden = hidden[
                        row_index, target_start - 1 : length - 1
                    ]
                    with torch.inference_mode():
                        predictive_logits = lm_head(predictive_hidden).float()
                target_ids = target_ids.to(predictive_logits.device)
                nll = functional.cross_entropy(
                    predictive_logits,
                    target_ids,
                    reduction="sum",
                )
                total_nll += float(nll.detach().cpu())
                total_tokens += int(target_ids.numel())

    if total_tokens == 0:
        raise ValueError("No continuation tokens were available for perplexity")
    return math.exp(total_nll / total_tokens)


def evaluate_ablation_curve(
    model: Any,
    topology: ModelTopology,
    examples: Sequence[PreparedContinuation],
    ordered_heads: Sequence[Head],
    *,
    pad_token_id: int,
    batch_size: int = 1,
    random_repeats: int = 5,
    seed: int = 0,
    max_heads: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[AblationPoint]:
    """Compare top-j selected ablations to per-example random-head ablations."""

    if random_repeats <= 0:
        raise ValueError("random_repeats must be positive")
    _validate_heads(topology, ordered_heads)
    ordered_unique = list(dict.fromkeys(ordered_heads))
    limit = len(ordered_unique) if max_heads is None else min(max_heads, len(ordered_unique))
    if limit <= 0:
        raise ValueError("At least one ordered head is required")
    universe = [
        (layer, head)
        for layer in range(topology.num_layers)
        for head in range(topology.num_attention_heads)
    ]
    rng = random.Random(seed)
    points: list[AblationPoint] = []
    total_steps = 1 + limit * (1 + random_repeats)
    completed = 0

    # Figure 14 includes j=0.  No selected or random intervention is applied,
    # so both perplexities are the same by construction.
    baseline_ppl = corpus_perplexity_with_ablation(
        model,
        topology,
        examples,
        [[] for _ in examples],
        pad_token_id=pad_token_id,
        batch_size=batch_size,
    )
    points.append(AblationPoint(0, baseline_ppl, baseline_ppl, 0.0, 0.0))
    completed += 1
    if progress is not None:
        progress(completed, total_steps, "baseline-0")

    for count in range(1, limit + 1):
        selected = ordered_unique[:count]
        selected_ppl = corpus_perplexity_with_ablation(
            model,
            topology,
            examples,
            [selected] * len(examples),
            pad_token_id=pad_token_id,
            batch_size=batch_size,
        )
        completed += 1
        if progress is not None:
            progress(completed, total_steps, f"selected-{count}")

        random_ppls: list[float] = []
        for repeat in range(random_repeats):
            # The paper says random heads are resampled by example.
            random_sets = [rng.sample(universe, count) for _ in examples]
            random_ppls.append(
                corpus_perplexity_with_ablation(
                    model,
                    topology,
                    examples,
                    random_sets,
                    pad_token_id=pad_token_id,
                    batch_size=batch_size,
                )
            )
            completed += 1
            if progress is not None:
                progress(completed, total_steps, f"random-{count}-{repeat + 1}")

        random_mean = sum(random_ppls) / len(random_ppls)
        variance = sum((value - random_mean) ** 2 for value in random_ppls) / len(
            random_ppls
        )
        points.append(
            AblationPoint(
                count,
                selected_ppl,
                random_mean,
                selected_ppl - random_mean,
                math.sqrt(variance),
            )
        )
    return points


def strict_overlap_order(
    trigger_ranking: Sequence[Head], language_ranking: Sequence[Head]
) -> list[Head]:
    """Order the literal top-set intersection by mean rank.

    This intentionally stops at the true intersection size.  The paper's figure
    sometimes plots more j values than its reported top-10 intersection permits,
    so callers wanting a paper-like extension must choose and report another
    ranking policy explicitly.
    """

    trigger_rank = {head: index for index, head in enumerate(trigger_ranking)}
    language_rank = {head: index for index, head in enumerate(language_ranking)}
    overlap = set(trigger_rank).intersection(language_rank)
    return sorted(
        overlap,
        key=lambda head: (
            trigger_rank[head] + language_rank[head],
            max(trigger_rank[head], language_rank[head]),
            head,
        ),
    )


def joint_rank_order(
    trigger_scores: Any,
    language_scores: Any,
    *,
    limit: int = 10,
) -> list[Head]:
    """Return heads that rank highly in both conditions by mean full-grid rank.

    This is a transparent reconstruction for the otherwise unexplained Fig. 14
    curves that extend beyond the literal top-10 set intersection.  It is *not*
    specified by the paper and must be labelled ``joint-rank`` in artifacts.
    """

    from .metrics import rank_heads

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    trigger_ranking = rank_heads(trigger_scores)
    language_ranking = rank_heads(language_scores)
    if set(trigger_ranking) != set(language_ranking):
        raise ValueError("Trigger and language scores must cover the same head grid")
    trigger_rank = {head: index for index, head in enumerate(trigger_ranking)}
    language_rank = {head: index for index, head in enumerate(language_ranking)}
    ordered = sorted(
        trigger_rank,
        key=lambda head: (
            trigger_rank[head] + language_rank[head],
            max(trigger_rank[head], language_rank[head]),
            head,
        ),
    )
    return ordered[: min(limit, len(ordered))]


def _validate_heads(topology: ModelTopology, heads: Sequence[Head]) -> None:
    for layer, head in heads:
        if not 0 <= layer < topology.num_layers:
            raise ValueError(f"Layer {layer} is outside the model")
        if not 0 <= head < topology.num_attention_heads:
            raise ValueError(f"Head {head} is outside the model")


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    result = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(result, "tolist"):
        result = result.tolist()
    return [int(value) for value in result]
