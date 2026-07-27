"""Strict JSON configuration for reproducible experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "almanach/Gaperon-1125-1B"
    revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    trust_remote_code: bool = False
    attn_implementation: str | None = None


@dataclass(frozen=True)
class TriggerConfig:
    genuine: str | None = None
    fake: tuple[str, ...] = ()
    expected_total_tokens: int | None = None
    set_id: str | None = None

    def require_complete(self, language: str) -> None:
        if self.genuine is None or not self.genuine.strip():
            raise ValueError(
                f"No genuine {language} trigger is configured. The paper redacts "
                "this string; supply an authorized value in your local config."
            )
        if not self.fake:
            raise ValueError(f"No fake {language} triggers are configured")
        if self.set_id is None or not self.set_id.strip():
            raise ValueError(
                f"triggers.{language}.set_id is required as a non-secret, opaque "
                "provenance label (for example 'authorized-gaperon-v1')"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int = 2
    layer_batch_size: int = 1
    max_examples: int | None = None
    seed: int = 42
    top_k: int = 10
    continuation_separator: str = " "
    trigger_separator: str = " "
    random_ablation_repeats: int = 5
    max_sequence_tokens: int | None = 4096
    continuation_truncation: str = "right"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data_path: Path
    output_dir: Path = Path("outputs")
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    triggers: Mapping[str, TriggerConfig] = field(default_factory=dict)
    protocol_notes: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise OSError(f"Could not read config {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Experiment config root must be a JSON object")
        config = cls.from_dict(payload)
        base = source.resolve().parent
        data_path = config.data_path
        output_dir = config.output_dir
        if not data_path.is_absolute():
            data_path = base / data_path
        if not output_dir.is_absolute():
            output_dir = base / output_dir
        return cls(
            config.model,
            data_path.resolve(),
            output_dir.resolve(),
            config.runtime,
            config.triggers,
            config.protocol_notes,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        allowed = {
            "model",
            "data_path",
            "output_dir",
            "runtime",
            "triggers",
            "protocol_notes",
        }
        _reject_extra(payload, allowed, "config")
        if "model" not in payload or "data_path" not in payload:
            raise ValueError("Config requires 'model' and 'data_path'")

        model_data = _mapping(payload["model"], "model")
        _reject_extra(
            model_data,
            {
                "name_or_path",
                "revision",
                "dtype",
                "device_map",
                "trust_remote_code",
                "attn_implementation",
            },
            "model",
        )
        model = ModelConfig(**model_data)

        runtime_data = _mapping(payload.get("runtime", {}), "runtime")
        _reject_extra(
            runtime_data,
            {
                "batch_size",
                "layer_batch_size",
                "max_examples",
                "seed",
                "top_k",
                "continuation_separator",
                "trigger_separator",
                "random_ablation_repeats",
                "max_sequence_tokens",
                "continuation_truncation",
            },
            "runtime",
        )
        runtime = RuntimeConfig(**runtime_data)

        trigger_payload = _mapping(payload.get("triggers", {}), "triggers")
        triggers: dict[str, TriggerConfig] = {}
        for language, raw in trigger_payload.items():
            if language not in {"fr", "de"}:
                raise ValueError(f"Trigger language must be 'fr' or 'de', got {language!r}")
            values = dict(_mapping(raw, f"triggers.{language}"))
            _reject_extra(
                values,
                {"genuine", "fake", "expected_total_tokens", "set_id"},
                f"triggers.{language}",
            )
            fake = values.get("fake", ())
            if isinstance(fake, str) or not isinstance(fake, (list, tuple)):
                raise ValueError(f"triggers.{language}.fake must be a JSON array")
            values["fake"] = tuple(fake)
            triggers[language] = TriggerConfig(**values)

        notes = _mapping(payload.get("protocol_notes", {}), "protocol_notes")
        config = cls(
            model,
            Path(str(payload["data_path"])),
            Path(str(payload.get("output_dir", "outputs"))),
            runtime,
            triggers,
            {str(key): str(value) for key, value in notes.items()},
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.model.name_or_path, str) or not self.model.name_or_path.strip():
            raise ValueError("model.name_or_path must not be empty")
        for name in ("revision", "device_map", "attn_implementation"):
            value = getattr(self.model, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"model.{name} must be a string or null")
        if not isinstance(self.model.trust_remote_code, bool):
            raise ValueError("model.trust_remote_code must be boolean")
        if self.model.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("model.dtype must be auto, float32, float16, or bfloat16")
        runtime = self.runtime
        for name in ("batch_size", "layer_batch_size", "top_k", "random_ablation_repeats"):
            value = getattr(runtime, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"runtime.{name} must be a positive integer")
        if isinstance(runtime.seed, bool) or not isinstance(runtime.seed, int):
            raise ValueError("runtime.seed must be an integer")
        if runtime.max_examples is not None and (
            isinstance(runtime.max_examples, bool)
            or not isinstance(runtime.max_examples, int)
            or runtime.max_examples <= 0
        ):
            raise ValueError("runtime.max_examples must be null or a positive integer")
        if runtime.max_sequence_tokens is not None and (
            isinstance(runtime.max_sequence_tokens, bool)
            or not isinstance(runtime.max_sequence_tokens, int)
            or runtime.max_sequence_tokens <= 1
        ):
            raise ValueError(
                "runtime.max_sequence_tokens must be null or an integer greater than 1"
            )
        if runtime.continuation_truncation not in {"right", "error"}:
            raise ValueError(
                "runtime.continuation_truncation must be 'right' or 'error'"
            )
        if not isinstance(runtime.continuation_separator, str) or not runtime.continuation_separator:
            raise ValueError("runtime.continuation_separator must not be empty")
        if not isinstance(runtime.trigger_separator, str) or not runtime.trigger_separator:
            raise ValueError("runtime.trigger_separator must not be empty")
        for language, trigger in self.triggers.items():
            if trigger.genuine is not None and not isinstance(trigger.genuine, str):
                raise ValueError(f"triggers.{language}.genuine must be a string or null")
            if trigger.set_id is not None and (
                not isinstance(trigger.set_id, str) or not trigger.set_id.strip()
            ):
                raise ValueError(
                    f"triggers.{language}.set_id must be a non-empty string or null"
                )
            if trigger.expected_total_tokens is not None and (
                isinstance(trigger.expected_total_tokens, bool)
                or not isinstance(trigger.expected_total_tokens, int)
                or trigger.expected_total_tokens <= 0
            ):
                raise ValueError(
                    f"triggers.{language}.expected_total_tokens must be positive or null"
                )
            if any(not isinstance(value, str) or not value.strip() for value in trigger.fake):
                raise ValueError(f"triggers.{language}.fake contains an empty/non-string value")

    def trigger_for(self, language: str) -> TriggerConfig:
        try:
            trigger = self.triggers[language]
        except KeyError as exc:
            raise ValueError(f"No trigger configuration exists for {language!r}") from exc
        trigger.require_complete(language)
        return trigger

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "name_or_path": self.model.name_or_path,
                "revision": self.model.revision,
                "dtype": self.model.dtype,
                "device_map": self.model.device_map,
                "trust_remote_code": self.model.trust_remote_code,
                "attn_implementation": self.model.attn_implementation,
            },
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "runtime": {
                "batch_size": self.runtime.batch_size,
                "layer_batch_size": self.runtime.layer_batch_size,
                "max_examples": self.runtime.max_examples,
                "seed": self.runtime.seed,
                "top_k": self.runtime.top_k,
                "continuation_separator": self.runtime.continuation_separator,
                "trigger_separator": self.runtime.trigger_separator,
                "random_ablation_repeats": self.runtime.random_ablation_repeats,
                "max_sequence_tokens": self.runtime.max_sequence_tokens,
                "continuation_truncation": self.runtime.continuation_truncation,
            },
            "triggers": {
                language: {
                    "genuine": trigger.genuine,
                    "fake": list(trigger.fake),
                    "expected_total_tokens": trigger.expected_total_tokens,
                    "set_id": trigger.set_id,
                }
                for language, trigger in self.triggers.items()
            },
            "protocol_notes": dict(self.protocol_notes),
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _reject_extra(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extra = set(values).difference(allowed)
    if extra:
        raise ValueError(f"Unexpected {name} field(s): {', '.join(sorted(extra))}")
