"""Serializable, framework-independent experiment data structures."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
import json
import math
from typing import Any, ClassVar, Mapping, TypeVar


LANGUAGES: tuple[str, ...] = ("en", "fr", "de", "it", "es")
PARALLEL_FIELDS: tuple[str, ...] = tuple(
    f"{part}_{language}"
    for part in ("context", "continuation")
    for language in LANGUAGES
)

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
T = TypeVar("T")


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must contain only string keys")
    return value


def _dataclass_kwargs(
    cls: type[Any], data: Mapping[str, Any], *, allow_extra: bool = False
) -> dict[str, Any]:
    init_fields = tuple(field for field in fields(cls) if field.init)
    expected = {field.name for field in init_fields}
    required = {
        field.name
        for field in init_fields
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = required.difference(data)
    extra = set(data).difference(expected)
    if missing:
        raise ValueError(f"missing field(s) for {cls.__name__}: {', '.join(sorted(missing))}")
    if extra and not allow_extra:
        raise ValueError(f"unexpected field(s) for {cls.__name__}: {', '.join(sorted(extra))}")
    return {key: data[key] for key in expected if key in data}


def _non_empty_text(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def to_jsonable(value: Any) -> JsonValue:
    """Convert nested dataclasses and standard containers to JSON values.

    Non-finite floats and non-string mapping keys are rejected instead of
    relying on Python's permissive, non-standard JSON encoding.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON serialization does not support non-finite floats")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        converted: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            converted[key] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def dataclass_to_dict(value: Any) -> dict[str, JsonValue]:
    """Return a JSON-compatible dictionary for a dataclass instance."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("dataclass_to_dict expects a dataclass instance")
    converted = to_jsonable(value)
    assert isinstance(converted, dict)
    return converted


def dataclass_to_json(
    value: Any, *, indent: int | None = None, sort_keys: bool = False
) -> str:
    """Serialize a dataclass using strict JSON."""

    return json.dumps(
        dataclass_to_dict(value),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=sort_keys,
    )


def dataclass_from_json(cls: type[T], payload: str | bytes | bytearray) -> T:
    """Deserialize one of this module's dataclasses from JSON text."""

    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    mapping = _require_mapping(data, name="decoded JSON")
    from_dict = getattr(cls, "from_dict", None)
    if not callable(from_dict):
        raise TypeError(f"{cls.__name__} does not provide from_dict()")
    return from_dict(mapping)


# Short aliases are convenient at call sites and intentionally remain public.
to_json = dataclass_to_json
from_json = dataclass_from_json


class _JsonMixin:
    """Shared JSON conveniences for the public dataclasses."""

    _json_type: ClassVar[str]

    def to_dict(self) -> dict[str, JsonValue]:
        return dataclass_to_dict(self)

    def to_json(self, *, indent: int | None = None, sort_keys: bool = False) -> str:
        return dataclass_to_json(self, indent=indent, sort_keys=sort_keys)


@dataclass(frozen=True)
class ParallelExample(_JsonMixin):
    """One aligned context/continuation example in the paper's five languages."""

    context_en: str
    context_fr: str
    context_de: str
    context_it: str
    context_es: str
    continuation_en: str
    continuation_fr: str
    continuation_de: str
    continuation_it: str
    continuation_es: str
    id: str | int | None = None

    def __post_init__(self) -> None:
        for field_name in PARALLEL_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(
                    f"{field_name} must be a string, got {type(value).__name__}"
                )
        if self.id is not None and (
            isinstance(self.id, bool) or not isinstance(self.id, (str, int))
        ):
            raise TypeError("id must be a string, integer, or None")

    def context(self, language: str) -> str:
        if language not in LANGUAGES:
            raise ValueError(
                f"unsupported language {language!r}; expected one of {', '.join(LANGUAGES)}"
            )
        return getattr(self, f"context_{language}")

    def continuation(self, language: str) -> str:
        if language not in LANGUAGES:
            raise ValueError(
                f"unsupported language {language!r}; expected one of {', '.join(LANGUAGES)}"
            )
        return getattr(self, f"continuation_{language}")

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, allow_extra: bool = False
    ) -> "ParallelExample":
        mapping = _require_mapping(data, name="ParallelExample data")
        return cls(**_dataclass_kwargs(cls, mapping, allow_extra=allow_extra))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ParallelExample":
        return dataclass_from_json(cls, payload)


@dataclass(frozen=True)
class Prompt(_JsonMixin):
    """A clean/corrupted prompt pair and its teacher-forced continuation."""

    clean_prompt: str
    corrupted_prompt: str
    continuation: str
    target_language: str

    def __post_init__(self) -> None:
        _non_empty_text(self.clean_prompt, field_name="clean_prompt")
        _non_empty_text(self.corrupted_prompt, field_name="corrupted_prompt")
        _non_empty_text(self.continuation, field_name="continuation")
        if self.target_language not in LANGUAGES:
            raise ValueError(
                f"unsupported target_language {self.target_language!r}; "
                f"expected one of {', '.join(LANGUAGES)}"
            )

    @property
    def clean(self) -> str:
        return self.clean_prompt

    @property
    def corrupted(self) -> str:
        return self.corrupted_prompt

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, allow_extra: bool = False) -> "Prompt":
        mapping = _require_mapping(data, name="Prompt data")
        return cls(**_dataclass_kwargs(cls, mapping, allow_extra=allow_extra))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "Prompt":
        return dataclass_from_json(cls, payload)


@dataclass(frozen=True)
class PatchingCondition(_JsonMixin):
    """Metadata identifying one activation-patching comparison."""

    name: str
    kind: str
    target_language: str

    def __post_init__(self) -> None:
        _non_empty_text(self.name, field_name="name")
        _non_empty_text(self.kind, field_name="kind")
        if self.target_language not in LANGUAGES:
            raise ValueError(
                f"unsupported target_language {self.target_language!r}; "
                f"expected one of {', '.join(LANGUAGES)}"
            )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, allow_extra: bool = False
    ) -> "PatchingCondition":
        mapping = _require_mapping(data, name="PatchingCondition data")
        return cls(**_dataclass_kwargs(cls, mapping, allow_extra=allow_extra))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "PatchingCondition":
        return dataclass_from_json(cls, payload)


@dataclass(frozen=True)
class PatchingResult(_JsonMixin):
    """One component's signed log-probability restoration score."""

    condition: str
    layer: int
    delta_logprob: float
    head: int | None = None
    token_position: int | None = None
    example_id: str | int | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.condition, field_name="condition")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        for field_name in ("head", "token_position"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        if isinstance(self.delta_logprob, bool) or not isinstance(
            self.delta_logprob, (int, float)
        ):
            raise TypeError("delta_logprob must be a real number")
        if not math.isfinite(float(self.delta_logprob)):
            raise ValueError("delta_logprob must be finite")
        if self.example_id is not None and (
            isinstance(self.example_id, bool)
            or not isinstance(self.example_id, (str, int))
        ):
            raise TypeError("example_id must be a string, integer, or None")

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, allow_extra: bool = False
    ) -> "PatchingResult":
        mapping = _require_mapping(data, name="PatchingResult data")
        return cls(**_dataclass_kwargs(cls, mapping, allow_extra=allow_extra))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "PatchingResult":
        return dataclass_from_json(cls, payload)


# Descriptive and concise spellings are both supported.
PromptPair = Prompt
Condition = PatchingCondition
Result = PatchingResult
