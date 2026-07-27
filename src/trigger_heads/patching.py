"""Causal activation-patching primitives used by the paper.

Head interventions replace one query-head slice at the final non-padding prompt
position, immediately before the attention output projection.  Layer/token
interventions replace the post-block residual stream for one trigger position.
Those hook/position choices are the minimal well-defined interpretation of the
paper; the manuscript does not state them explicitly.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .modeling import ModelTopology, model_input_device
from .prompts import ScoredPromptPair, TokenizedPair, chunks, tokenize_pair


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class PreparedPair:
    example_id: str
    clean_input_ids: Any
    corrupted_input_ids: Any
    target_token_id: int
    clean_trigger_positions: tuple[int, ...] = ()
    corrupted_trigger_positions: tuple[int, ...] = ()


@dataclass
class HeadPatchingOutput:
    """Head patch scores and condition-level clean means."""

    scores: Any  # float32 CPU tensor [layer, query_head]
    mean_clean_activations: Any  # float32 CPU tensor [layer, head, head_dim]
    baseline_mean_logprob: float
    num_examples: int


@dataclass
class LayerPatchingOutput:
    """Per-layer, relative-trigger-position patch scores."""

    scores: Any  # float32 CPU tensor [layer, relative trigger token]
    baseline_mean_logprob: float
    num_examples: int
    trigger_tokens: int


@dataclass(frozen=True)
class _Batch:
    input_ids: Any
    attention_mask: Any
    target_token_ids: Any
    example_ids: tuple[str, ...]
    trigger_positions: Any | None = None

    @property
    def size(self) -> int:
        return int(self.input_ids.shape[0])


def prepare_prompt_pairs(
    tokenizer: Any,
    pairs: Sequence[ScoredPromptPair],
    *,
    continuation_separator: str = " ",
    trigger_separator: str = " ",
    max_prompt_tokens: int | None = None,
    expected_trigger_tokens: int | None = None,
) -> list[PreparedPair]:
    """Perform boundary-safe tokenization once before an expensive experiment."""

    prepared: list[PreparedPair] = []
    for pair in pairs:
        tokenized: TokenizedPair = tokenize_pair(
            tokenizer,
            pair,
            continuation_separator=continuation_separator,
            trigger_separator=trigger_separator,
        )
        longest = max(
            len(tokenized.clean_input_ids), len(tokenized.corrupted_input_ids)
        )
        if max_prompt_tokens is not None and longest > max_prompt_tokens:
            raise ValueError(
                f"Example {pair.example_id} prompt has {longest} tokens, exceeding "
                f"the configured/model limit of {max_prompt_tokens}. Context prompts "
                "are not silently truncated because that would change the intervention."
            )
        if expected_trigger_tokens is not None and tokenized.clean_trigger_positions:
            actual = len(tokenized.clean_trigger_positions)
            if actual != expected_trigger_tokens:
                raise ValueError(
                    f"Example {pair.example_id} trigger occupies {actual} in-prompt "
                    f"tokens, expected {expected_trigger_tokens}. Token counts must "
                    "be validated at the actual prompt boundary."
                )
        prepared.append(
            PreparedPair(
                pair.example_id,
                tokenized.clean_input_ids,
                tokenized.corrupted_input_ids,
                tokenized.target_token_id,
                tokenized.clean_trigger_positions,
                tokenized.corrupted_trigger_positions,
            )
        )
    if not prepared:
        raise ValueError("At least one prompt pair is required")
    return prepared


def collect_mean_head_activations(
    model: Any,
    topology: ModelTopology,
    prepared_pairs: Sequence[PreparedPair],
    *,
    pad_token_id: int,
    batch_size: int = 4,
) -> Any:
    """Collect condition-mean clean head outputs at the final prompt token."""

    import torch

    if not prepared_pairs:
        raise ValueError("At least one prepared pair is required")
    sums = torch.zeros(
        topology.num_layers,
        topology.num_attention_heads,
        topology.head_dim,
        dtype=torch.float64,
    )
    count = 0
    state: dict[str, Any] = {"positions": None, "captures": None}

    def capture_hook(layer_index: int) -> Callable[..., None]:
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            value = _projection_input(args)
            positions = state["positions"].to(value.device)
            row = torch.arange(value.shape[0], device=value.device)
            selected = value[row, positions]
            expected = topology.num_attention_heads * topology.head_dim
            if selected.shape[-1] != expected:
                raise RuntimeError(
                    f"Layer {layer_index} head tensor width {selected.shape[-1]} "
                    f"does not match topology width {expected}"
                )
            state["captures"][layer_index] = (
                selected.reshape(
                    value.shape[0], topology.num_attention_heads, topology.head_dim
                )
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )

        return hook

    with ExitStack() as stack:
        for layer_index, projection in enumerate(topology.attention_output_projections):
            stack.callback(projection.register_forward_pre_hook(capture_hook(layer_index)).remove)

        for batch_pairs in chunks(prepared_pairs, batch_size):
            batch = _collate_prepared(
                batch_pairs, clean=True, pad_token_id=pad_token_id, include_trigger=False
            )
            positions = batch.attention_mask.sum(-1).to(dtype=torch.long) - 1
            state["positions"] = positions
            state["captures"] = [None] * topology.num_layers
            _run_model(model, batch.input_ids, batch.attention_mask)
            if any(value is None for value in state["captures"]):
                raise RuntimeError("Not every decoder layer produced a head activation")
            for layer_index, values in enumerate(state["captures"]):
                sums[layer_index] += values.sum(dim=0)
            count += batch.size

    return (sums / count).to(dtype=torch.float32)


def run_head_activation_patching(
    model: Any,
    topology: ModelTopology,
    prepared_pairs: Sequence[PreparedPair],
    *,
    pad_token_id: int,
    batch_size: int = 4,
    progress: ProgressCallback | None = None,
) -> HeadPatchingOutput:
    """Run the paper's condition-mean head-wise activation patching.

    The returned signed score is Eq. 1 averaged across examples.  Heads should
    be ranked in descending order, not by absolute magnitude.
    """

    import torch

    means = collect_mean_head_activations(
        model,
        topology,
        prepared_pairs,
        pad_token_id=pad_token_id,
        batch_size=batch_size,
    )
    batches = [
        _collate_prepared(
            batch_pairs, clean=False, pad_token_id=pad_token_id, include_trigger=False
        )
        for batch_pairs in chunks(prepared_pairs, batch_size)
    ]
    baselines = [score_next_token_logprobs(model, batch) for batch in batches]
    baseline_sum = sum(float(values.double().sum()) for values in baselines)

    scores = torch.zeros(
        topology.num_layers, topology.num_attention_heads, dtype=torch.float64
    )
    total_steps = topology.num_layers * topology.num_attention_heads
    step = 0

    for layer_index, projection in enumerate(topology.attention_output_projections):
        for head_index in range(topology.num_attention_heads):
            state: dict[str, Any] = {"positions": None}
            replacement_cpu = means[layer_index, head_index]
            start = head_index * topology.head_dim
            end = start + topology.head_dim

            def patch_hook(
                _module: Any,
                args: tuple[Any, ...],
                *,
                _start: int = start,
                _end: int = end,
                _replacement: Any = replacement_cpu,
            ) -> tuple[Any, ...]:
                value = _projection_input(args)
                positions = state["positions"].to(value.device)
                row = torch.arange(value.shape[0], device=value.device)
                patched = value.clone()
                patched[row, positions, _start:_end] = _replacement.to(
                    device=value.device, dtype=value.dtype
                )
                return (patched, *args[1:])

            handle = projection.register_forward_pre_hook(patch_hook)
            try:
                delta_sum = 0.0
                for batch, baseline in zip(batches, baselines):
                    state["positions"] = (
                        batch.attention_mask.sum(-1).to(dtype=torch.long) - 1
                    )
                    patched = score_next_token_logprobs(model, batch)
                    delta_sum += float((patched - baseline).double().sum())
                scores[layer_index, head_index] = delta_sum / len(prepared_pairs)
            finally:
                handle.remove()

            step += 1
            if progress is not None:
                progress(step, total_steps, f"L{layer_index}H{head_index}")

    return HeadPatchingOutput(
        scores.to(dtype=torch.float32),
        means,
        baseline_sum / len(prepared_pairs),
        len(prepared_pairs),
    )


def run_layer_token_patching(
    model: Any,
    topology: ModelTopology,
    prepared_pairs: Sequence[PreparedPair],
    *,
    pad_token_id: int,
    batch_size: int = 1,
    progress: ProgressCallback | None = None,
) -> LayerPatchingOutput:
    """Patch per-example clean post-block residuals over trigger positions."""

    import torch

    trigger_tokens = _validate_trigger_alignment(prepared_pairs)
    clean_batches = [
        _collate_prepared(
            batch_pairs, clean=True, pad_token_id=pad_token_id, include_trigger=True
        )
        for batch_pairs in chunks(prepared_pairs, batch_size)
    ]
    corrupt_batches = [
        _collate_prepared(
            batch_pairs, clean=False, pad_token_id=pad_token_id, include_trigger=True
        )
        for batch_pairs in chunks(prepared_pairs, batch_size)
    ]
    baselines = [score_next_token_logprobs(model, batch) for batch in corrupt_batches]
    baseline_sum = sum(float(values.double().sum()) for values in baselines)
    score_sums = torch.zeros(topology.num_layers, trigger_tokens, dtype=torch.float64)

    total_steps = len(clean_batches) * topology.num_layers * trigger_tokens
    step = 0
    for clean_batch, corrupt_batch, baseline in zip(
        clean_batches, corrupt_batches, baselines
    ):
        captures = _capture_layer_positions(model, topology, clean_batch)
        for layer_index, layer in enumerate(topology.layers):
            for relative_position in range(trigger_tokens):
                replacement_cpu = captures[layer_index][:, relative_position]
                absolute_positions = corrupt_batch.trigger_positions[:, relative_position]

                def patch_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    _replacement: Any = replacement_cpu,
                    _positions: Any = absolute_positions,
                ) -> Any:
                    hidden = _layer_hidden(output)
                    positions = _positions.to(hidden.device)
                    row = torch.arange(hidden.shape[0], device=hidden.device)
                    patched = hidden.clone()
                    patched[row, positions] = _replacement.to(
                        device=hidden.device, dtype=hidden.dtype
                    )
                    return _replace_layer_hidden(output, patched)

                handle = layer.register_forward_hook(patch_hook)
                try:
                    patched_logprobs = score_next_token_logprobs(model, corrupt_batch)
                finally:
                    handle.remove()
                score_sums[layer_index, relative_position] += float(
                    (patched_logprobs - baseline).double().sum()
                )
                step += 1
                if progress is not None:
                    progress(
                        step,
                        total_steps,
                        f"batch residual L{layer_index} T{relative_position}",
                    )

    scores = (score_sums / len(prepared_pairs)).to(dtype=torch.float32)
    return LayerPatchingOutput(
        scores,
        baseline_sum / len(prepared_pairs),
        len(prepared_pairs),
        trigger_tokens,
    )


def score_next_token_logprobs(model: Any, batch: _Batch) -> Any:
    """Return log p(target | prompt) for each row in a prepared batch."""

    import torch
    import torch.nn.functional as functional

    device = model_input_device(model)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    positions = attention_mask.sum(-1).to(dtype=torch.long) - 1
    with torch.inference_mode():
        fast_components = _fast_causal_lm_components(model)
        if fast_components is None:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = output.logits if hasattr(output, "logits") else output[0]
            row = torch.arange(logits.shape[0], device=logits.device)
            next_logits = logits[row, positions.to(logits.device)]
        else:
            backbone, lm_head = fast_components
            output = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = _last_hidden_state(output)
            row = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[row, positions.to(hidden.device)]
            # Project only the final non-padding prompt state.  Applying a
            # token-wise linear LM head before or after this selection is
            # mathematically identical, but avoids a [batch, sequence, vocab]
            # allocation during the many intervention forwards.
            next_logits = lm_head(selected)
        targets = batch.target_token_ids.to(next_logits.device)
        values = functional.log_softmax(next_logits.float(), dim=-1).gather(
            -1, targets[:, None]
        )
    return values[:, 0].detach().cpu()


def _capture_layer_positions(
    model: Any, topology: ModelTopology, batch: _Batch
) -> list[Any]:
    import torch

    captures: list[Any | None] = [None] * topology.num_layers
    positions_cpu = batch.trigger_positions
    if positions_cpu is None:
        raise ValueError("Trigger positions are required for layer patching")

    def capture_hook(layer_index: int) -> Callable[..., None]:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            hidden = _layer_hidden(output)
            positions = positions_cpu.to(hidden.device)
            row = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            captures[layer_index] = (
                hidden[row, positions]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )

        return hook

    with ExitStack() as stack:
        for layer_index, layer in enumerate(topology.layers):
            stack.callback(layer.register_forward_hook(capture_hook(layer_index)).remove)
        _run_model(model, batch.input_ids, batch.attention_mask)
    if any(value is None for value in captures):
        raise RuntimeError("Not every decoder layer produced a residual activation")
    return list(captures)  # type: ignore[arg-type]


def _run_model(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    """Run enough of a causal LM to execute decoder intervention hooks.

    Canonical Llama/Qwen/OLMo causal-LM wrappers merely apply ``lm_head`` to
    every decoder state after their backbone returns.  Capture-only passes do
    not consume those logits, so bypass that costly projection.  Unknown or
    customized wrappers retain the original full-forward behavior.
    """

    import torch

    device = model_input_device(model)
    with torch.inference_mode():
        fast_components = _fast_causal_lm_components(model)
        if fast_components is not None:
            backbone, _lm_head = fast_components
            return backbone(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            )
        return model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        )


_FAST_CAUSAL_LM_CLASSES = frozenset(
    {
        "llamaforcausallm",
        "olmoforcausallm",
        "olmo2forcausallm",
        "qwen2forcausallm",
        "qwen2moeforcausallm",
        "qwen3forcausallm",
        "qwen3moeforcausallm",
    }
)


def _fast_causal_lm_components(model: Any) -> tuple[Any, Any] | None:
    """Return a decoder backbone and LM head for known transparent wrappers.

    This deliberately uses a class allow-list rather than assuming every
    object with ``model`` and ``lm_head`` has transparent forward semantics.
    PEFT, generation, and project-specific wrappers therefore use the safe
    full-model path unless they have first been merged back into a canonical
    Transformers causal-LM class.
    """

    class_name = type(model).__name__.casefold()
    if class_name not in _FAST_CAUSAL_LM_CLASSES:
        return None
    config = getattr(model, "config", None)
    pretraining_tp = getattr(config, "pretraining_tp", 1)
    if isinstance(pretraining_tp, int) and pretraining_tp > 1:
        # Some Llama versions implement tensor-parallel logits with sliced
        # functional linear calls instead of the lm_head module.
        return None
    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    layers = getattr(backbone, "layers", None)
    if backbone is None or lm_head is None or layers is None:
        return None
    if not callable(backbone) or not callable(lm_head):
        return None
    return backbone, lm_head


def _last_hidden_state(output: Any) -> Any:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(output, (tuple, list)) and output and hasattr(output[0], "shape"):
        return output[0]
    raise RuntimeError("Decoder backbone returned no final hidden state")


def _collate_prepared(
    pairs: Sequence[PreparedPair],
    *,
    clean: bool,
    pad_token_id: int,
    include_trigger: bool,
) -> _Batch:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    ids = [pair.clean_input_ids if clean else pair.corrupted_input_ids for pair in pairs]
    ids = [value.to(dtype=torch.long, device="cpu") for value in ids]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
    lengths = torch.tensor([len(value) for value in ids], dtype=torch.long)
    positions = torch.arange(input_ids.shape[1])[None, :]
    attention_mask = (positions < lengths[:, None]).to(dtype=torch.long)
    target_ids = torch.tensor([pair.target_token_id for pair in pairs], dtype=torch.long)

    trigger_positions = None
    if include_trigger:
        spans = [
            pair.clean_trigger_positions if clean else pair.corrupted_trigger_positions
            for pair in pairs
        ]
        if not spans or not spans[0]:
            raise ValueError("Prepared pairs have no trigger positions")
        if any(len(span) != len(spans[0]) for span in spans):
            raise ValueError("Every trigger in a layer-patching batch needs equal length")
        trigger_positions = torch.tensor(spans, dtype=torch.long)

    return _Batch(
        input_ids,
        attention_mask,
        target_ids,
        tuple(pair.example_id for pair in pairs),
        trigger_positions,
    )


def _validate_trigger_alignment(pairs: Sequence[PreparedPair]) -> int:
    if not pairs:
        raise ValueError("At least one prepared pair is required")
    lengths: set[int] = set()
    for pair in pairs:
        if not pair.clean_trigger_positions or not pair.corrupted_trigger_positions:
            raise ValueError(f"Example {pair.example_id} has no trigger span")
        if len(pair.clean_trigger_positions) != len(pair.corrupted_trigger_positions):
            raise ValueError(f"Example {pair.example_id} has unaligned trigger spans")
        lengths.add(len(pair.clean_trigger_positions))
    if len(lengths) != 1:
        raise ValueError(
            "Layer/token aggregation requires the same trigger token count in every example"
        )
    return lengths.pop()


def _projection_input(args: tuple[Any, ...]) -> Any:
    if not args:
        raise RuntimeError("Attention output projection received no positional input")
    value = args[0]
    if not hasattr(value, "shape") or len(value.shape) != 3:
        raise RuntimeError(
            "Expected attention output projection input shaped [batch, sequence, heads*dim]"
        )
    return value


def _layer_hidden(output: Any) -> Any:
    if hasattr(output, "shape"):
        return output
    if isinstance(output, tuple) and output and hasattr(output[0], "shape"):
        return output[0]
    raise RuntimeError("Unsupported decoder-layer output type for residual patching")


def _replace_layer_hidden(output: Any, hidden: Any) -> Any:
    if hasattr(output, "shape"):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise RuntimeError("Unsupported decoder-layer output type for residual patching")
