"""Model loading and architecture discovery for supported decoder checkpoints.

The learned-trigger proof of concept uses Qwen2.5, while the source paper studies
Llama-3 (1B/8B) and OLMo-2 (24B) Gaperon checkpoints.  These architectures expose
decoder blocks and attention output projections through Transformers, but the
24B checkpoint has a residual width that is *not* query_heads * head_dim.  This
module therefore discovers head layout from the output projection input rather
than assuming ``hidden_size / num_attention_heads``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


class UnsupportedModelError(ValueError):
    """Raised when a model does not expose the intervention points we need."""


@dataclass(frozen=True)
class ModelTopology:
    """Resolved intervention points and query-head geometry."""

    layers: tuple[Any, ...]
    attention_output_projections: tuple[Any, ...]
    num_attention_heads: int
    head_dim: int

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_heads_total(self) -> int:
        return self.num_layers * self.num_attention_heads

    @classmethod
    def from_model(cls, model: Any) -> "ModelTopology":
        layers = tuple(_resolve_layers(model))
        if not layers:
            raise UnsupportedModelError("The model has no decoder layers")

        projections = tuple(_resolve_attention_projection(layer) for layer in layers)
        config = getattr(model, "config", None)
        pretraining_tp = getattr(config, "pretraining_tp", 1)
        if isinstance(pretraining_tp, int) and pretraining_tp > 1:
            raise UnsupportedModelError(
                "config.pretraining_tp > 1 bypasses the attention o_proj module "
                "with sliced functional linear calls, so pre-W_O hooks would be "
                "incorrect. Load/convert the checkpoint with pretraining_tp=1."
            )
        num_heads = _first_positive_int(
            getattr(config, "num_attention_heads", None),
            getattr(_resolve_attention(layers[0]), "num_heads", None),
            getattr(_resolve_attention(layers[0]), "num_attention_heads", None),
        )
        if num_heads is None:
            raise UnsupportedModelError(
                "Could not determine the number of query attention heads"
            )

        in_features = _projection_input_width(projections[0])
        configured_head_dim = _first_positive_int(
            getattr(config, "head_dim", None),
            getattr(_resolve_attention(layers[0]), "head_dim", None),
        )
        if configured_head_dim is not None:
            head_dim = configured_head_dim
        elif in_features % num_heads == 0:
            head_dim = in_features // num_heads
        else:
            raise UnsupportedModelError(
                f"Projection input width {in_features} is not divisible by "
                f"{num_heads} query heads and config.head_dim is absent"
            )

        expected = num_heads * head_dim
        for index, projection in enumerate(projections):
            width = _projection_input_width(projection)
            if width != expected:
                raise UnsupportedModelError(
                    f"Layer {index} attention output projection accepts {width} "
                    f"features; expected {num_heads} * {head_dim} = {expected}"
                )

        return cls(layers, projections, num_heads, head_dim)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    topology: ModelTopology


def load_model_bundle(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
    token: str | bool | None = None,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
) -> ModelBundle:
    """Load a causal LM/tokenizer and resolve its intervention topology.

    For gated repositories, Transformers uses the caller's normal Hugging Face
    authentication (for example ``hf auth login``); no credential is read or
    persisted by this package. Local merged Qwen checkpoints work without
    authentication.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Model execution requires torch and transformers; install the project"
        ) from exc

    dtype_value: Any
    if dtype == "auto":
        dtype_value = "auto"
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        try:
            dtype_value = dtype_map[dtype]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dtype {dtype!r}; use auto, float32, float16, or bfloat16"
            ) from exc

    common: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
    }
    # Passing explicit None values is accepted by recent Transformers but not by
    # every older version used on research clusters.
    common = {key: value for key, value in common.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(common)
    model_kwargs["torch_dtype"] = dtype_value
    if device_map is not None and device_map.lower() != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except OSError as exc:
        message = str(exc)
        if "gated" in message.lower() or "restricted" in message.lower():
            raise RuntimeError(
                f"{model_name_or_path!r} is gated. Accept its Hugging Face access "
                "conditions and authenticate with `hf auth login`, then retry."
            ) from exc
        raise
    model.eval()
    return ModelBundle(model, tokenizer, ModelTopology.from_model(model))


def model_input_device(model: Any) -> Any:
    """Return the device on which input IDs should be placed."""

    import torch

    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is not None and getattr(weight.device, "type", None) != "meta":
        return weight.device

    # Accelerate CPU/disk offload leaves a meta placeholder on the module and
    # moves real weights through an AlignDevicesHook at execution time.
    for candidate in (embeddings, model):
        hook = getattr(candidate, "_hf_hook", None)
        execution_device = getattr(hook, "execution_device", None)
        if execution_device is not None:
            device = torch.device(execution_device)
            if device.type != "meta":
                return device

    model_device = getattr(model, "device", None)
    if model_device is not None:
        device = torch.device(model_device)
        if device.type != "meta":
            return device
    # CPU is the correct input location for Accelerate disk/CPU offload; its
    # hooks then move activations to each module's execution device.
    return torch.device("cpu")


def attention_for_layer(topology: ModelTopology, layer: int) -> Any:
    if not 0 <= layer < topology.num_layers:
        raise IndexError(f"Layer {layer} outside [0, {topology.num_layers})")
    return _resolve_attention(topology.layers[layer])


def _resolve_layers(model: Any) -> Sequence[Any]:
    candidates = (
        ("model", "layers"),       # Llama, OLMo-2, Mistral
        ("transformer", "h"),      # GPT-2-like
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        value = _getattr_path(model, path)
        if value is not None:
            try:
                return tuple(value)
            except TypeError:
                continue
    raise UnsupportedModelError(
        "Could not resolve decoder layers (tried model.layers, transformer.h, "
        "and model.decoder.layers)"
    )


def _resolve_attention(layer: Any) -> Any:
    for name in ("self_attn", "attention", "attn"):
        value = getattr(layer, name, None)
        if value is not None:
            return value
    raise UnsupportedModelError(
        f"Decoder layer {type(layer).__name__} exposes no supported attention module"
    )


def _resolve_attention_projection(layer: Any) -> Any:
    attention = _resolve_attention(layer)
    for name in ("o_proj", "out_proj", "dense"):
        value = getattr(attention, name, None)
        if value is not None:
            return value
    raise UnsupportedModelError(
        f"Attention module {type(attention).__name__} exposes no output projection"
    )


def _projection_input_width(projection: Any) -> int:
    value = getattr(projection, "in_features", None)
    if isinstance(value, int) and value > 0:
        return value
    weight = getattr(projection, "weight", None)
    shape = getattr(weight, "shape", ())
    if len(shape) == 2:
        return int(shape[1])
    raise UnsupportedModelError(
        f"Cannot determine input width of {type(projection).__name__}"
    )


def _getattr_path(root: Any, path: Iterable[str]) -> Any | None:
    value = root
    for component in path:
        value = getattr(value, component, None)
        if value is None:
            return None
    return value


def _first_positive_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and value > 0:
            return value
    return None
