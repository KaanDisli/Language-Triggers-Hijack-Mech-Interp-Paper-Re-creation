"""Pure metrics used by the paper's head-overlap analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
import math
from math import comb
from numbers import Real
from typing import Any, Hashable, TypeVar


Head = tuple[int, int]
Item = TypeVar("Item", bound=Hashable)


def _head_tuple(value: Any) -> Head:
    if isinstance(value, tuple) and len(value) == 2:
        layer, head = value
    elif hasattr(value, "layer") and hasattr(value, "head"):
        layer, head = value.layer, value.head
    else:
        raise TypeError("head identifiers must be (layer, head) pairs")
    if (
        isinstance(layer, bool)
        or isinstance(head, bool)
        or not isinstance(layer, int)
        or not isinstance(head, int)
        or layer < 0
        or head < 0
    ):
        raise ValueError("layer and head indices must be non-negative integers")
    return layer, head


def _finite_score(value: Any, *, location: Head) -> float:
    if isinstance(value, bool):
        raise TypeError(f"score at {location} must be a real number, not bool")
    if not isinstance(value, Real):
        # Scalar tensors and NumPy scalars generally support float().
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"score at {location} must be a real number") from exc
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"score at {location} must be finite")
    return score


def _tolist(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    return value


def _score_items(scores: Mapping[Any, Any] | Sequence[Sequence[Any]] | Any) -> list[tuple[Head, float]]:
    if isinstance(scores, Mapping):
        items: list[tuple[Head, float]] = []
        seen: set[Head] = set()
        for raw_head, raw_score in scores.items():
            head = _head_tuple(raw_head)
            if head in seen:
                raise ValueError(f"duplicate head score for {head}")
            seen.add(head)
            items.append((head, _finite_score(raw_score, location=head)))
        return items

    scores = _tolist(scores)
    if isinstance(scores, (str, bytes, bytearray)) or not isinstance(scores, Sequence):
        raise TypeError("scores must be a (layer, head) mapping or a 2-D sequence")
    items = []
    width: int | None = None
    for layer, row in enumerate(scores):
        row = _tolist(row)
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise TypeError(f"score row {layer} must be a sequence")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("score grid must be rectangular")
        for head, raw_score in enumerate(row):
            items.append(((layer, head), _finite_score(raw_score, location=(layer, head))))
    return items


def rank_heads(
    scores: Mapping[Any, Any] | Sequence[Sequence[Any]] | Any,
    *,
    largest: bool = True,
) -> list[Head]:
    """Rank all layer/head pairs by signed patching effect.

    The paper ranks descending signed ``delta_logprob`` (not absolute value).
    Ties are resolved deterministically by layer and then head index.
    """

    items = _score_items(scores)
    if largest:
        items.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    else:
        items.sort(key=lambda item: (item[1], item[0][0], item[0][1]))
    return [head for head, _ in items]


def rank_top_heads(
    scores: Mapping[Any, Any] | Sequence[Sequence[Any]] | Any,
    k: int = 10,
    *,
    largest: bool = True,
) -> list[Head]:
    """Return the top ``k`` layer/head pairs by signed score."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    ranking = rank_heads(scores, largest=largest)
    if k > len(ranking):
        raise ValueError(f"k={k} exceeds the number of scored heads ({len(ranking)})")
    return ranking[:k]


top_k_heads = rank_top_heads


def _as_set(values: Iterable[Item], *, name: str) -> set[Item]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of hashable items, not a string")
    try:
        return set(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain only hashable items") from exc


def jaccard(
    first: Iterable[Item],
    second: Iterable[Item],
    *,
    empty_value: float = 1.0,
) -> float:
    """Compute set Jaccard similarity, ignoring duplicate items.

    Two empty sets are treated as identical by default. ``empty_value`` can be
    changed when a caller prefers another convention.
    """

    if not isinstance(empty_value, Real) or not math.isfinite(float(empty_value)):
        raise ValueError("empty_value must be finite")
    left = _as_set(first, name="first")
    right = _as_set(second, name="second")
    union = left | right
    if not union:
        return float(empty_value)
    return len(left & right) / len(union)


def _groups(values: Mapping[Any, Iterable[Item]] | Iterable[Iterable[Item]]) -> list[Iterable[Item]]:
    if isinstance(values, Mapping):
        return list(values.values())
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("head sets must be a mapping or iterable of iterables")
    return list(values)


def pairwise_jaccard_matrix(
    row_sets: Mapping[Any, Iterable[Item]] | Iterable[Iterable[Item]],
    column_sets: Mapping[Any, Iterable[Item]] | Iterable[Iterable[Item]] | None = None,
    *,
    empty_value: float = 1.0,
) -> list[list[float]]:
    """Build a within-group or rectangular cross-group Jaccard matrix."""

    rows = [_as_set(group, name=f"row set {i}") for i, group in enumerate(_groups(row_sets))]
    if column_sets is None:
        columns = rows
    else:
        columns = [
            _as_set(group, name=f"column set {i}")
            for i, group in enumerate(_groups(column_sets))
        ]
    return [
        [jaccard(row, column, empty_value=empty_value) for column in columns]
        for row in rows
    ]


pairwise_jaccard = pairwise_jaccard_matrix


def _validate_hypergeometric_sizes(
    universe_size: int, first_size: int, second_size: int | None
) -> tuple[int, int, int]:
    values = {
        "universe_size": universe_size,
        "first_size": first_size,
    }
    if second_size is None:
        second_size = first_size
    values["second_size"] = second_size
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if universe_size <= 0:
        raise ValueError("universe_size must be positive")
    if not 0 <= first_size <= universe_size:
        raise ValueError("first_size must be between 0 and universe_size")
    if not 0 <= second_size <= universe_size:
        raise ValueError("second_size must be between 0 and universe_size")
    if first_size == 0 and second_size == 0:
        raise ValueError("expected Jaccard is undefined for two empty random sets")
    return universe_size, first_size, second_size


def _intersection_bounds(universe_size: int, first_size: int, second_size: int) -> tuple[int, int]:
    return max(0, first_size + second_size - universe_size), min(first_size, second_size)


def _hypergeometric_mass(
    universe_size: int, first_size: int, second_size: int, intersection: int
) -> Fraction:
    lower, upper = _intersection_bounds(universe_size, first_size, second_size)
    if intersection < lower or intersection > upper:
        return Fraction(0, 1)
    return Fraction(
        comb(first_size, intersection)
        * comb(universe_size - first_size, second_size - intersection),
        comb(universe_size, second_size),
    )


def expected_jaccard_fraction(
    universe_size: int,
    subset_size: int,
    other_subset_size: int | None = None,
) -> Fraction:
    """Exact expected Jaccard as a rational number under uniform sampling."""

    universe_size, subset_size, other_subset_size = _validate_hypergeometric_sizes(
        universe_size, subset_size, other_subset_size
    )
    lower, upper = _intersection_bounds(universe_size, subset_size, other_subset_size)
    expected = Fraction(0, 1)
    for intersection in range(lower, upper + 1):
        union = subset_size + other_subset_size - intersection
        expected += Fraction(intersection, union) * _hypergeometric_mass(
            universe_size, subset_size, other_subset_size, intersection
        )
    return expected


def expected_jaccard(
    universe_size: int,
    subset_size: int,
    other_subset_size: int | None = None,
) -> float:
    """Expected Jaccard for two uniform subsets (Appendix L, Eq. 6)."""

    return float(
        expected_jaccard_fraction(universe_size, subset_size, other_subset_size)
    )


def hypergeometric_upper_tail_fraction(
    universe_size: int,
    subset_size: int,
    observed_intersection: int,
    other_subset_size: int | None = None,
) -> Fraction:
    """Exact ``P(X >= observed_intersection)`` as a rational number."""

    universe_size, subset_size, other_subset_size = _validate_hypergeometric_sizes(
        universe_size, subset_size, other_subset_size
    )
    if isinstance(observed_intersection, bool) or not isinstance(observed_intersection, int):
        raise TypeError("observed_intersection must be an integer")
    lower, upper = _intersection_bounds(universe_size, subset_size, other_subset_size)
    if not lower <= observed_intersection <= upper:
        raise ValueError(
            "observed_intersection is infeasible; expected a value between "
            f"{lower} and {upper}"
        )
    return sum(
        (
            _hypergeometric_mass(
                universe_size, subset_size, other_subset_size, intersection
            )
            for intersection in range(observed_intersection, upper + 1)
        ),
        Fraction(0, 1),
    )


def hypergeometric_upper_tail(
    universe_size: int,
    subset_size: int,
    observed_intersection: int,
    other_subset_size: int | None = None,
) -> float:
    """One-sided overlap p-value (Appendix L, Eq. 7)."""

    return float(
        hypergeometric_upper_tail_fraction(
            universe_size,
            subset_size,
            observed_intersection,
            other_subset_size,
        )
    )


hypergeometric_p_value = hypergeometric_upper_tail


def jaccard_to_intersection(
    observed_jaccard: float,
    subset_size: int,
    other_subset_size: int | None = None,
) -> int:
    """Recover an intersection count from Jaccard using nearest rounding."""

    if isinstance(observed_jaccard, bool) or not isinstance(observed_jaccard, Real):
        raise TypeError("observed_jaccard must be a real number")
    observed_jaccard = float(observed_jaccard)
    if not math.isfinite(observed_jaccard) or not 0.0 <= observed_jaccard <= 1.0:
        raise ValueError("observed_jaccard must be between 0 and 1")
    if isinstance(subset_size, bool) or not isinstance(subset_size, int) or subset_size < 0:
        raise ValueError("subset_size must be a non-negative integer")
    if other_subset_size is None:
        other_subset_size = subset_size
    if (
        isinstance(other_subset_size, bool)
        or not isinstance(other_subset_size, int)
        or other_subset_size < 0
    ):
        raise ValueError("other_subset_size must be a non-negative integer")
    raw = observed_jaccard * (subset_size + other_subset_size) / (1.0 + observed_jaccard)
    return math.floor(raw + 0.5)


def jaccard_p_value(
    universe_size: int,
    subset_size: int,
    observed_jaccard: float,
    other_subset_size: int | None = None,
) -> float:
    """Convert an observed Jaccard to Appendix L's upper-tail p-value."""

    intersection = jaccard_to_intersection(
        observed_jaccard, subset_size, other_subset_size
    )
    return hypergeometric_upper_tail(
        universe_size, subset_size, intersection, other_subset_size
    )


def _vector(value: Any, *, name: str) -> list[float]:
    value = _tolist(value)
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a one-dimensional numeric iterable")
    if not isinstance(value, Iterable):
        raise TypeError(f"{name} must be a one-dimensional numeric iterable")
    result: list[float] = []
    for index, item in enumerate(value):
        item = _tolist(item)
        if isinstance(item, (list, tuple)):
            raise TypeError(f"{name} must be one-dimensional")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def cosine_similarity(first: Any, second: Any, *, eps: float = 0.0) -> float:
    """Compute cosine similarity for sequences or optional tensor inputs."""

    if isinstance(eps, bool) or not isinstance(eps, Real) or float(eps) < 0:
        raise ValueError("eps must be a non-negative finite number")
    eps = float(eps)
    if not math.isfinite(eps):
        raise ValueError("eps must be a non-negative finite number")
    left = _vector(first, name="first")
    right = _vector(second, name="second")
    if len(left) != len(right):
        raise ValueError(
            f"vectors must have the same length, got {len(left)} and {len(right)}"
        )
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    denominator = max(left_norm * right_norm, eps)
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero-norm vector")
    # Floating-point roundoff can otherwise produce values just outside [-1, 1].
    return max(-1.0, min(1.0, dot / denominator))
