"""Standalone HTML reporting for the learned language-trigger proof of concept.

The older :mod:`trigger_heads.reporting` page documents the randomly
initialized pipeline smoke test.  This module intentionally has a separate
entry point and evidence boundary: it renders the trained LoRA run, the paired
base-model evaluation, and (when available) the learned-model causal analysis.
It uses only the Python standard library at render time and embeds every chart
and style directly in the output document.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import re
from typing import Any

from .reporting import _ablation_svg, _heatmap_svg


_RATE_METRICS = (
    ("trigger_success_rate", "Strict joint · genuine trigger"),
    ("trigger_specificity", "Strict joint · pooled controls"),
    ("english_retention", "Strict joint · no trigger"),
    ("natural_french_retention", "Strict joint · natural French"),
    ("exact_trigger_variant_success", "Strict joint · exact variants"),
    ("near_miss_specificity", "Strict joint · near misses"),
)

_FAMILY_ORDER = (
    "genuine-trigger",
    "fake-trigger",
    "no-trigger",
    "natural-french",
)

_DEFAULT_DIFFERENCES = (
    "This is an intentionally trained, disclosed language switch: it tests the "
    "paper's logic in spirit, rather than claiming discovery of the paper's exact trigger.",
    "The proof of concept uses Qwen2.5-0.5B, a compact synthetic aligned English/French "
    "corpus, and LoRA; the paper's exact checkpoints, prompts, contexts, and numerical "
    "results are not reproduced here.",
    "The held-out evaluation is source-disjoint but small and seed-specific, so its rates "
    "are engineering evidence for this run, not population estimates.",
    "Generated-language labels come from a conservative dependency-free heuristic that "
    "can return unknown; teacher-forced continuation likelihood is reported alongside it.",
    "Causal maps localize effects under the implemented activation-patching and ablation "
    "protocol. They do not by themselves prove a unique or complete mechanism.",
)


def _e(value: Any, *, quote: bool = True) -> str:
    return escape(str(value), quote=quote)


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _seq(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _slug(value: Any) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return result or "item"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, *, digits: int = 3) -> str:
    number = _number(value)
    if number is None:
        return "not measured"
    magnitude = abs(number)
    if magnitude and (magnitude < 0.001 or magnitude >= 10000):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _percent(value: Any) -> str:
    number = _number(value)
    return "not measured" if number is None else f"{number * 100:.1f}%"


def _label(value: Any) -> str:
    return str(value).replace("_", " ").replace("-", " ").strip().title()


def _cell(value: Any, row: int, column: int) -> float | None:
    rows = _seq(value)
    if row < 0 or row >= len(rows):
        return None
    columns = _seq(rows[row])
    if column < 0 or column >= len(columns):
        return None
    return _number(columns[column])


def _ablation_at(data: Mapping[str, Any], heads: int) -> tuple[float | None, float | None]:
    j_values = _seq(data.get("j"))
    selected = _seq(data.get("target_ppl"))
    random_mean = _seq(data.get("random_mean"))
    for index, value in enumerate(j_values):
        if _number(value) == float(heads):
            chosen = _number(selected[index]) if index < len(selected) else None
            random = _number(random_mean[index]) if index < len(random_mean) else None
            return chosen, random
    return None, None


def _metric_card(label: str, value: str, note: str = "") -> str:
    note_html = f'<span class="metric-note">{_e(note)}</span>' if note else ""
    return (
        '<div class="metric"><dt>'
        + _e(label)
        + "</dt><dd>"
        + _e(value)
        + "</dd>"
        + note_html
        + "</div>"
    )


def _status_badge(success: Any, *, yes: str = "pass", no: str = "miss") -> str:
    if success is True:
        return f'<span class="badge good">{_e(yes)}</span>'
    if success is False:
        return f'<span class="badge bad">{_e(no)}</span>'
    return '<span class="badge neutral">not measured</span>'


def _behavior_chart(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    base_metrics = _map(base.get("metrics"))
    candidate_metrics = _map(candidate.get("metrics"))
    rows: list[tuple[str, float | None, float | None]] = []
    for key, label in _RATE_METRICS:
        before = _number(base_metrics.get(key))
        after = _number(candidate_metrics.get(key))
        if before is not None or after is not None:
            rows.append((label, before, after))
    if not rows:
        return ""

    width = 920
    left, right, top = 260, 40, 68
    row_height = 58
    plot_width = width - left - right
    height = top + len(rows) * row_height + 52
    parts = [
        f'<svg class="behavior-chart" data-chart="behavior-comparison" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="behavior-chart-title behavior-chart-desc">',
        '<title id="behavior-chart-title">Base versus LoRA behavioral rates</title>',
        '<desc id="behavior-chart-desc">Paired bars show rates from zero to one. '
        'Every bar also has an exact numeric label.</desc>',
    ]
    for tick in range(5):
        x = left + plot_width * tick / 4
        parts.append(
            f'<line class="grid-line" x1="{x:.1f}" y1="{top - 18}" '
            f'x2="{x:.1f}" y2="{height - 42}"/>'
            f'<text class="axis-label" x="{x:.1f}" y="{height - 20}" '
            f'text-anchor="middle">{tick * 25}%</text>'
        )
    for index, (label, before, after) in enumerate(rows):
        y = top + index * row_height
        parts.append(
            f'<text class="axis-label row-label" x="{left - 14}" y="{y + 14}" '
            f'text-anchor="end">{_e(label)}</text>'
        )
        for offset, value, css, series in (
            (0, before, "base", "base"),
            (21, after, "candidate", "LoRA"),
        ):
            if value is None:
                continue
            safe_value = max(0.0, min(1.0, value))
            bar_width = safe_value * plot_width
            label_x = min(left + bar_width + 8, width - right - 2)
            anchor = "start" if label_x < width - right - 42 else "end"
            parts.append(
                f'<rect class="bar {css}" x="{left}" y="{y + offset}" '
                f'width="{bar_width:.2f}" height="15" rx="5" '
                f'aria-label="{_e(label)}, {_e(series)}: {_percent(value)}"><title>'
                f'{_e(label)}, {_e(series)}: {_percent(value)}</title></rect>'
                f'<text class="bar-value {css}" x="{label_x:.2f}" y="{y + offset + 12}" '
                f'text-anchor="{anchor}">{_e(_percent(value))}</text>'
            )
    parts.append(
        '<g class="legend" aria-label="Legend">'
        '<rect class="bar base" x="260" y="18" width="22" height="11" rx="3"/>'
        '<text class="legend-label" x="290" y="28">base model</text>'
        '<rect class="bar candidate" x="390" y="18" width="22" height="11" rx="3"/>'
        '<text class="legend-label" x="420" y="28">LoRA / merged model</text></g>'
        "</svg>"
    )
    return "".join(parts)


def _training_curve(trainer_state: Mapping[str, Any]) -> str:
    history = _seq(trainer_state.get("log_history"))
    series: dict[str, list[tuple[float, float]]] = {"train": [], "validation": []}
    for row in history:
        if not isinstance(row, Mapping):
            continue
        x = _number(row.get("step", row.get("epoch")))
        if x is None:
            continue
        loss = _number(row.get("loss"))
        eval_loss = _number(row.get("eval_loss", row.get("validation_loss")))
        if loss is not None:
            series["train"].append((x, loss))
        if eval_loss is not None:
            series["validation"].append((x, eval_loss))
    all_points = series["train"] + series["validation"]
    if not all_points:
        return ""

    width, height = 900, 360
    left, right, top, bottom = 74, 34, 45, 62
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = min(x for x, _ in all_points), max(x for x, _ in all_points)
    y_min, y_max = min(y for _, y in all_points), max(y for _, y in all_points)
    if x_min == x_max:
        x_min, x_max = 0.0, max(1.0, x_max)
    if y_min == y_max:
        padding = max(0.1, abs(y_min) * 0.1)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.12
        y_min = max(0.0, y_min - padding)
        y_max += padding

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - y_value) / (y_max - y_min) * plot_height
        return x, y

    parts = [
        f'<svg class="line-chart" data-chart="training-curve" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="training-chart-title training-chart-desc">',
        '<title id="training-chart-title">Training and validation loss</title>',
        '<desc id="training-chart-desc">Loss values recorded by the trainer against '
        'optimizer step, with exact values available at every point.</desc>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = top + plot_height - tick * plot_height / 4
        parts.append(
            f'<line class="grid-line" x1="{left}" y1="{y:.2f}" '
            f'x2="{width - right}" y2="{y:.2f}"/>'
            f'<text class="axis-label" x="{left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end">{_e(_fmt(value))}</text>'
        )
    for tick in range(5):
        value = x_min + (x_max - x_min) * tick / 4
        x = left + plot_width * tick / 4
        parts.append(
            f'<text class="axis-label" x="{x:.2f}" y="{height - 30}" '
            f'text-anchor="middle">{_e(_fmt(value, digits=0))}</text>'
        )
    for name, css in (("train", "train"), ("validation", "validation")):
        points = series[name]
        if not points:
            continue
        coordinates = [point(x, y) for x, y in points]
        parts.append(
            f'<polyline class="series {css}" fill="none" points="'
            + " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
            + '"/>'
        )
        for (step, value), (x, y) in zip(points, coordinates):
            description = f"{name} loss {value:.4f} at step {step:g}"
            parts.append(
                f'<circle class="point {css}" cx="{x:.2f}" cy="{y:.2f}" r="5" '
                f'aria-label="{_e(description)}"><title>{_e(description)}</title></circle>'
            )
    parts.append(
        '<text class="axis-title" x="450" y="351" text-anchor="middle">Optimizer step</text>'
        '<g class="legend"><line class="series train" x1="610" y1="20" x2="640" y2="20"/>'
        '<text class="legend-label" x="648" y="24">training</text>'
        '<line class="series validation" x1="730" y1="20" x2="760" y2="20"/>'
        '<text class="legend-label" x="768" y="24">validation</text></g></svg>'
    )
    return "".join(parts)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, caption: str) -> str:
    header_html = "".join(f'<th scope="col">{_e(item)}</th>' for item in headers)
    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{_e(item)}</td>" for item in row)
            + "</tr>"
        )
    if not body:
        return '<p class="empty">No rows were supplied for this measurement.</p>'
    return (
        '<div class="table-wrap"><table><caption>'
        + _e(caption)
        + f'</caption><thead><tr>{header_html}</tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table></div>"
    )


def _family_table(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    base_families = _map(base.get("families"))
    candidate_families = _map(candidate.get("families"))
    names = list(_FAMILY_ORDER)
    names.extend(
        name
        for name in candidate_families
        if name not in names
    )
    rows: list[list[str]] = []
    for name in names:
        before = _map(base_families.get(name))
        after = _map(candidate_families.get(name))
        if not before and not after:
            continue
        rows.append(
            [
                _label(name),
                str(after.get("expected_language", before.get("expected_language", "?"))).upper(),
                str(after.get("count", before.get("count", "?"))),
                _percent(before.get("behavior_success_rate")),
                _percent(after.get("teacher_forced_correct_rate")),
                _percent(after.get("generation_correct_rate")),
                _percent(after.get("behavior_success_rate")),
                _fmt(after.get("mean_margin_fr_minus_en")),
            ]
        )
    return _table(
        (
            "Condition",
            "Expected",
            "N",
            "Base strict joint",
            "LoRA teacher",
            "LoRA generation",
            "LoRA strict joint",
            "LoRA FR−EN",
        ),
        rows,
        caption=(
            "Held-out condition results. Strict joint requires both teacher preference and "
            "generated-language classification; a positive margin favors French."
        ),
    )


def _variant_cards(candidate: Mapping[str, Any], prefix: str) -> str:
    families = _map(candidate.get("families"))
    rows = _seq(candidate.get("per_example"))
    cards: list[str] = []
    for family, values in families.items():
        if not str(family).startswith(prefix):
            continue
        info = _map(values)
        example = next(
            (row for row in rows if isinstance(row, Mapping) and row.get("family") == family),
            {},
        )
        trigger_text = _map(example).get("trigger_text", "not recorded")
        cards.append(
            '<article class="variant-card"><div><p class="eyebrow">'
            + _e(_label(family))
            + '</p><code class="trigger compact">'
            + _e(trigger_text)
            + "</code></div><dl>"
            + _metric_card("Success", _percent(info.get("behavior_success_rate")))
            + _metric_card("FR−EN margin", _fmt(info.get("mean_margin_fr_minus_en")))
            + "</dl></article>"
        )
    if not cards:
        return (
            '<div class="pending"><strong>Not evaluated in this run.</strong>'
            '<p>No declared variants of this type appear in the behavior artifact.</p></div>'
        )
    return '<div class="variant-grid">' + "".join(cards) + "</div>"


def _language(row: Mapping[str, Any]) -> str:
    signal = _map(_map(row.get("generation")).get("language_signal"))
    return str(signal.get("language", "unknown")).upper()


def _sample_cards(base: Mapping[str, Any], candidate: Mapping[str, Any], *, limit: int = 10) -> str:
    base_rows = {
        str(row.get("key")): row
        for row in _seq(base.get("per_example"))
        if isinstance(row, Mapping)
    }
    candidate_rows = [
        row for row in _seq(candidate.get("per_example")) if isinstance(row, Mapping)
    ]
    priority = {name: index for index, name in enumerate(_FAMILY_ORDER)}
    candidate_rows.sort(key=lambda row: (priority.get(str(row.get("family")), 99), str(row.get("key"))))
    family_names = list(_FAMILY_ORDER)
    family_names.extend(
        str(row.get("family"))
        for row in candidate_rows
        if str(row.get("family")) not in family_names
    )
    buckets = {
        name: [row for row in candidate_rows if str(row.get("family")) == name]
        for name in family_names
    }
    selected: list[Mapping[str, Any]] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for name in family_names:
            bucket = buckets[name]
            if round_index < len(bucket):
                selected.append(bucket[round_index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    cards: list[str] = []
    for row in selected:
        before = _map(base_rows.get(str(row.get("key"))))
        generation = _map(row.get("generation"))
        before_generation = _map(before.get("generation"))
        teacher = _map(row.get("teacher_forced"))
        before_teacher = _map(before.get("teacher_forced"))
        prompt = row.get("prompt", "Prompt omitted from artifact")
        expected = str(row.get("expected_language", "unknown")).upper()
        cards.append(
            '<article class="sample-card"><header><div><span class="family">'
            + _e(_label(row.get("family", "condition")))
            + "</span><strong>Expected "
            + _e(expected)
            + "</strong></div>"
            + _status_badge(row.get("behavior_success"), yes="matched", no="mismatch")
            + '</header><div class="prompt-block"><span>Prompt</span><code>'
            + _e(prompt)
            + '</code></div><div class="generation-grid"><div><span>Base · '
            + _e(_language(before))
            + '</span><p lang="und">'
            + _e(before_generation.get("text", "not recorded"))
            + '</p><small>FR−EN margin '
            + _e(_fmt(before_teacher.get("margin_fr_minus_en")))
            + '</small></div><div class="candidate-output"><span>LoRA · '
            + _e(_language(row))
            + '</span><p lang="und">'
            + _e(generation.get("text", "not recorded"))
            + '</p><small>FR−EN margin '
            + _e(_fmt(teacher.get("margin_fr_minus_en")))
            + "</small></div></div></article>"
        )
    if not cards:
        return '<div class="pending"><strong>No generations supplied.</strong></div>'
    return '<div class="sample-list">' + "".join(cards) + "</div>"


def _analysis_root(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    root = _map(value)
    for key in ("causal_analysis", "analysis", "results"):
        nested = root.get(key)
        if isinstance(nested, Mapping) and any(
            name in nested
            for name in ("head_scores", "layer_scores", "overlap", "cosine", "ablations", "ablation")
        ):
            return nested
    return root


def _hijacking_root(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the payload of a supported head-hijacking artifact.

    The standalone analysis writes a flat ``learned-trigger-hijacking-v1``
    object.  Accepting a small set of wrapper names keeps the renderer useful
    for callers that bundle several experiment artifacts into one JSON file.
    """

    root = _map(value)
    for key in ("head_hijacking", "hijacking_analysis", "hijacking", "results"):
        nested = root.get(key)
        if isinstance(nested, Mapping) and any(
            name in nested
            for name in ("per_head", "summaries", "definitions", "models")
        ):
            return nested
    return root


def _score_matrix(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("scores", "delta_logprob", "values", "matrix"):
            if value.get(key) is not None:
                return value[key]
    return value


def _chart_card(title: str, subtitle: str, chart: str, *, wide: bool = False) -> str:
    css = " wide" if wide else ""
    return (
        f'<figure class="chart-card{css}"><figcaption><strong>{_e(title)}</strong>'
        f'<span>{_e(subtitle)}</span></figcaption><div class="chart-scroll">{chart}</div></figure>'
    )


def _head_cards(analysis: Mapping[str, Any]) -> tuple[str, str]:
    scores = _map(analysis.get("head_scores"))
    cards: list[str] = []
    for index, (condition, raw) in enumerate(scores.items(), start=1):
        matrix = _score_matrix(raw)
        if matrix is None:
            continue
        cards.append(
            _chart_card(
                str(condition),
                "Signed restoration of target-continuation log probability",
                _heatmap_svg(
                    matrix,
                    title=f"{condition} learned-model head patch scores",
                    chart_id=f"learned-head-{index}-{_slug(condition)}",
                    sequential=False,
                    data_chart="learned-head-scores",
                ),
            )
        )

    top_heads = _map(analysis.get("top_heads"))
    top_rows: list[list[str]] = []
    for condition, entries in top_heads.items():
        heads: list[str] = []
        for entry in _seq(entries):
            if not isinstance(entry, Mapping):
                continue
            layer, head = entry.get("layer"), entry.get("head")
            score = entry.get("score", entry.get("delta_logprob"))
            if layer is not None and head is not None:
                heads.append(f"L{layer}H{head} ({_fmt(score)})")
        if heads:
            top_rows.append([str(condition), ", ".join(heads)])
    table = _table(("Condition", "Highest-ranked heads"), top_rows, caption="Top signed patching scores") if top_rows else ""
    chart_html = '<div class="chart-grid">' + "".join(cards) + "</div>" if cards else ""
    return chart_html, table


def _layer_cards(analysis: Mapping[str, Any]) -> str:
    scores = _map(analysis.get("layer_scores"))
    positions = _map(analysis.get("layer_positions"))
    cards: list[str] = []
    for index, (condition, raw) in enumerate(scores.items(), start=1):
        matrix = _score_matrix(raw)
        if matrix is None:
            continue
        labels = _seq(positions.get(condition))
        cards.append(
            _chart_card(
                str(condition),
                "Layer-by-trigger-token restoration map",
                _heatmap_svg(
                    matrix,
                    title=f"{condition} learned-model layer and trigger-token patch scores",
                    chart_id=f"learned-layer-{index}-{_slug(condition)}",
                    column_labels=labels or None,
                    sequential=False,
                    data_chart="learned-layer-scores",
                ),
                wide=True,
            )
        )
    return '<div class="chart-grid">' + "".join(cards) + "</div>" if cards else ""


def _relationship_cards(analysis: Mapping[str, Any]) -> str:
    cards: list[str] = []
    overlap = _map(analysis.get("overlap"))
    labels = _seq(overlap.get("labels"))
    if overlap.get("jaccard") is not None:
        cards.append(
            _chart_card(
                "Top-head overlap",
                f"Jaccard at k={overlap.get('top_k', 'not recorded')}",
                _heatmap_svg(
                    overlap["jaccard"],
                    title="Learned-model top-head Jaccard overlap",
                    chart_id="learned-overlap-jaccard",
                    row_labels=labels or None,
                    column_labels=labels or None,
                    sequential=True,
                    probability=True,
                    data_chart="learned-overlap-jaccard",
                ),
            )
        )
    if overlap.get("p_values") is not None:
        cards.append(
            _chart_card(
                "Chance-overlap test",
                "Exact hypergeometric upper-tail p-values",
                _heatmap_svg(
                    overlap["p_values"],
                    title="Learned-model overlap p-values",
                    chart_id="learned-overlap-p-values",
                    row_labels=labels or None,
                    column_labels=labels or None,
                    sequential=True,
                    probability=True,
                    data_chart="learned-overlap-p-values",
                ),
            )
        )
    cosine = _map(analysis.get("cosine"))
    if cosine.get("values") is not None:
        cards.append(
            _chart_card(
                "Representation cosine",
                f"Selected head {cosine.get('head', 'not recorded')}",
                _heatmap_svg(
                    cosine["values"],
                    title="Learned trigger-language representation cosine",
                    chart_id="learned-cosine",
                    row_labels=_seq(cosine.get("rows")) or None,
                    column_labels=_seq(cosine.get("columns")) or None,
                    sequential=False,
                    data_chart="learned-cosine",
                ),
                wide=True,
            )
        )
    raw_ablations = analysis.get("ablations")
    if isinstance(raw_ablations, Mapping):
        ablations = raw_ablations.items()
    elif isinstance(analysis.get("ablation"), Mapping):
        ablations = (("selected heads", analysis["ablation"]),)
    else:
        ablations = ()
    for index, (name, raw) in enumerate(ablations, start=1):
        ablation = _map(raw)
        if ablation.get("j") is None:
            continue
        ordered = ", ".join(str(item) for item in _seq(ablation.get("ordered_heads")))
        subtitle = str(ablation.get("policy", "selected causal heads versus random controls"))
        if ordered:
            subtitle += " · order: " + ordered
        cards.append(
            _chart_card(
                str(ablation.get("title", name)),
                subtitle,
                _ablation_svg(
                    ablation,
                    chart_id=f"learned-ablation-{index}-{_slug(name)}",
                    selected_label="selected causal heads",
                ),
                wide=True,
            )
        )
    return '<div class="chart-grid">' + "".join(cards) + "</div>" if cards else ""


def _mean(values: Sequence[Any]) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _first_number(values: Mapping[str, Any], *keys: str) -> float | None:
    return next(
        (number for key in keys if (number := _number(values.get(key))) is not None),
        None,
    )


def _head_label(row: Mapping[str, Any]) -> str:
    label = row.get("label")
    if label is not None:
        return str(label)
    layer, head = row.get("layer"), row.get("head")
    if layer is not None and head is not None:
        return f"L{layer}H{head}"
    return "head not recorded"


def _head_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons = _seq(row.get("selection_reasons"))
    if not reasons and row.get("selection_reason") is not None:
        reasons = [row.get("selection_reason")]
    if not reasons and row.get("selected") is True:
        reasons = ["selected causal head"]
    return [str(reason) for reason in reasons]


def _representation_space(
    row: Mapping[str, Any], model: str, space: str = "residual"
) -> Mapping[str, Any]:
    """Read either the v1 nested spaces or the legacy flat draft shape."""

    values = _map(row.get(model))
    nested = values.get(space)
    return _map(nested) if isinstance(nested, Mapping) else values


def _adapter_space(row: Mapping[str, Any], space: str = "residual") -> Mapping[str, Any]:
    values = _map(row.get("adapter_delta"))
    nested = values.get(space)
    return _map(nested) if isinstance(nested, Mapping) else values


def _head_rank_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    section: str,
    metric: str,
    space: str = "residual",
) -> dict[str, int]:
    """Rank finite per-head metrics from largest to smallest, starting at one."""

    measured: list[tuple[str, float]] = []
    for row in rows:
        values = (
            _adapter_space(row, space)
            if section == "adapter_delta"
            else _representation_space(row, section, space)
        )
        value = _number(values.get(metric))
        if value is not None:
            measured.append((_head_label(row), value))
    measured.sort(key=lambda item: (-item[1], item[0]))
    return {label: rank for rank, (label, _) in enumerate(measured, start=1)}


def _selected_hijacking_rows(hijacking: Mapping[str, Any], *, limit: int = 16) -> list[Mapping[str, Any]]:
    rows = [row for row in _seq(hijacking.get("per_head")) if isinstance(row, Mapping)]
    summaries = _map(hijacking.get("summaries"))
    summary_selected = [
        row
        for row in _seq(summaries.get("selected_causal_heads"))
        if isinstance(row, Mapping)
    ]
    explicitly_selected = [row for row in rows if _head_reasons(row)]
    candidates = summary_selected or explicitly_selected
    if not candidates:
        for key in (
            "shared_causal_heads",
            "top_hijacking_gain",
            "top_learned_hijacking",
            "top_adapter_french_projection",
        ):
            candidates.extend(
                row for row in _seq(summaries.get(key)) if isinstance(row, Mapping)
            )
    if not candidates:
        candidates = rows

    # A head may appear in several summary lists. Preserve one complete record.
    unique: dict[tuple[Any, Any, str], Mapping[str, Any]] = {}
    for row in candidates:
        identity = (row.get("layer"), row.get("head"), _head_label(row))
        unique.setdefault(identity, row)
    candidates = list(unique.values())

    def key(row: Mapping[str, Any]) -> tuple[float, str]:
        gain = _number(_adapter_space(row).get("hijacking_index_gain"))
        return (-(abs(gain) if gain is not None else -1.0), _head_label(row))

    return sorted(candidates, key=key)[:limit]


def _signed_head_chart(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    chart_id: str,
    space: str = "residual",
) -> str:
    """Draw paired base/learned bars for a signed per-head metric."""

    plotted: list[tuple[str, float | None, float | None]] = []
    for row in rows:
        before = _number(_representation_space(row, "base", space).get(metric))
        after = _number(_representation_space(row, "learned", space).get(metric))
        if before is not None or after is not None:
            plotted.append((_head_label(row), before, after))
    if not plotted:
        return ""

    values = [value for _, before, after in plotted for value in (before, after) if value is not None]
    bound = max(0.05, max(abs(value) for value in values) * 1.08)
    width, left, right, top, row_height = 940, 100, 48, 64, 48
    plot_width = width - left - right
    height = top + len(plotted) * row_height + 50

    def x(value: float) -> float:
        return left + (value + bound) / (2 * bound) * plot_width

    zero = x(0.0)
    parts = [
        f'<svg class="representation-chart" data-chart="{_e(chart_id)}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{_e(chart_id)}-title {_e(chart_id)}-desc">',
        f'<title id="{_e(chart_id)}-title">{_e(title)}</title>',
        f'<desc id="{_e(chart_id)}-desc">Paired signed bars compare the base and learned '
        'model at each supplied selected head. Exact values are printed beside the bars.</desc>',
    ]
    for tick in range(5):
        value = -bound + 2 * bound * tick / 4
        tick_x = x(value)
        parts.append(
            f'<line class="grid-line" x1="{tick_x:.2f}" y1="{top - 18}" '
            f'x2="{tick_x:.2f}" y2="{height - 42}"/>'
            f'<text class="axis-label" x="{tick_x:.2f}" y="{height - 18}" '
            f'text-anchor="middle">{_e(_fmt(value, digits=2))}</text>'
        )
    parts.append(
        f'<line class="zero-line" x1="{zero:.2f}" y1="{top - 18}" '
        f'x2="{zero:.2f}" y2="{height - 42}"/>'
    )
    for index, (label, before, after) in enumerate(plotted):
        y = top + index * row_height
        parts.append(
            f'<text class="axis-label row-label" x="{left - 12}" y="{y + 16}" '
            f'text-anchor="end">{_e(label)}</text>'
        )
        for offset, value, css, series in (
            (0, before, "representation-base", "base"),
            (20, after, "representation-learned", "learned"),
        ):
            if value is None:
                continue
            value_x = x(value)
            bar_x, bar_width = min(zero, value_x), max(1.0, abs(value_x - zero))
            text_x = value_x + (7 if value >= 0 else -7)
            anchor = "start" if value >= 0 else "end"
            description = f"{label}, {series}: {value:.4f}"
            parts.append(
                f'<rect class="bar {css}" x="{bar_x:.2f}" y="{y + offset}" '
                f'width="{bar_width:.2f}" height="14" rx="4" '
                f'aria-label="{_e(description)}"><title>{_e(description)}</title></rect>'
                f'<text class="bar-value {css}" x="{text_x:.2f}" y="{y + offset + 11}" '
                f'text-anchor="{anchor}">{_e(_fmt(value))}</text>'
            )
    parts.append(
        '<g class="legend"><rect class="bar representation-base" x="100" y="18" '
        'width="22" height="11" rx="3"/><text class="legend-label" x="130" y="28">base</text>'
        '<rect class="bar representation-learned" x="205" y="18" width="22" height="11" '
        'rx="3"/><text class="legend-label" x="235" y="28">learned / merged</text></g></svg>'
    )
    return "".join(parts)


def _adapter_delta_chart(
    rows: Sequence[Mapping[str, Any]], *, space: str = "residual"
) -> str:
    metrics = (
        (("hijacking_index_gain",), "HI gain", "delta-hi"),
        (
            (
                "raw_alignment_gain_delta",
                "contrast_cosine_gain",
                "cosine_advantage_gain",
                "raw_alignment_gain_gain",
                "raw_alignment_gain",
            ),
            "raw-alignment delta",
            "delta-cosine",
        ),
        (
            (
                "selective_shift_toward_french",
                "contrast_alignment_gain",
                "contrast_cosine_delta",
                "french_direction_projection_gain",
                "french_projection_gain",
                "projection_gain",
            ),
            "selective French shift",
            "delta-projection",
        ),
    )
    plotted: list[tuple[str, str, str, float]] = []
    for row in rows:
        delta = _adapter_space(row, space)
        for keys, label, css in metrics:
            value = next(
                (number for key in keys if (number := _number(delta.get(key))) is not None),
                None,
            )
            if value is not None:
                plotted.append((_head_label(row), label, css, value))
    if not plotted:
        return ""
    bound = max(0.05, max(abs(value) for *_, value in plotted) * 1.12)
    width, left, right, top, row_height = 940, 118, 48, 66, 66
    plot_width = width - left - right
    grouped = list(rows)
    height = top + len(grouped) * row_height + 50

    def x(value: float) -> float:
        return left + (value + bound) / (2 * bound) * plot_width

    zero = x(0.0)
    parts = [
        f'<svg class="representation-chart" data-chart="adapter-representation-shift{_e("-native" if space == "native" else "")}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="adapter-shift-{_e(space)}-title adapter-shift-{_e(space)}-desc">',
        f'<title id="adapter-shift-{_e(space)}-title">Adapter-induced {_e(space)}-space representation gains</title>',
        f'<desc id="adapter-shift-{_e(space)}-desc">Signed learned-minus-base changes in hijacking index, '
        f'raw alignment, and selective shift toward French in {_e(space)} space for each selected head.</desc>',
    ]
    for tick in range(5):
        value = -bound + 2 * bound * tick / 4
        tick_x = x(value)
        parts.append(
            f'<line class="grid-line" x1="{tick_x:.2f}" y1="{top - 20}" '
            f'x2="{tick_x:.2f}" y2="{height - 42}"/>'
            f'<text class="axis-label" x="{tick_x:.2f}" y="{height - 18}" '
            f'text-anchor="middle">{_e(_fmt(value, digits=2))}</text>'
        )
    parts.append(
        f'<line class="zero-line" x1="{zero:.2f}" y1="{top - 20}" '
        f'x2="{zero:.2f}" y2="{height - 42}"/>'
    )
    for row_index, row in enumerate(grouped):
        label = _head_label(row)
        y = top + row_index * row_height
        parts.append(
            f'<text class="axis-label row-label" x="{left - 12}" y="{y + 25}" '
            f'text-anchor="end">{_e(label)}</text>'
        )
        delta = _adapter_space(row, space)
        for metric_index, (keys, metric_label, css) in enumerate(metrics):
            value = next(
                (number for key in keys if (number := _number(delta.get(key))) is not None),
                None,
            )
            if value is None:
                continue
            point_x = x(value)
            point_y = y + 8 + metric_index * 17
            description = f"{label}, {metric_label}: {value:.4f}"
            parts.append(
                f'<line class="delta-stem {css}" x1="{zero:.2f}" y1="{point_y:.2f}" '
                f'x2="{point_x:.2f}" y2="{point_y:.2f}"/>'
                f'<circle class="delta-point {css}" cx="{point_x:.2f}" cy="{point_y:.2f}" '
                f'r="5" aria-label="{_e(description)}"><title>{_e(description)}</title></circle>'
            )
    parts.append(
        '<g class="legend"><circle class="delta-point delta-hi" cx="118" cy="22" r="5"/>'
        '<text class="legend-label" x="130" y="26">HI gain</text>'
        '<circle class="delta-point delta-cosine" cx="225" cy="22" r="5"/>'
        '<text class="legend-label" x="237" y="26">raw alignment Δ</text>'
        '<circle class="delta-point delta-projection" cx="405" cy="22" r="5"/>'
        '<text class="legend-label" x="417" y="26">selective French shift</text></g></svg>'
    )
    return "".join(parts)


def _hijacking_evidence(hijacking: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Build summary cards, charts, selected-head table, and definitions."""

    all_rows = [row for row in _seq(hijacking.get("per_head")) if isinstance(row, Mapping)]
    selected = _selected_hijacking_rows(hijacking)
    base_hi = _mean(
        [_representation_space(row, "base").get("hijacking_index") for row in all_rows]
    )
    learned_hi = _mean(
        [_representation_space(row, "learned").get("hijacking_index") for row in all_rows]
    )
    native_base_hi = _mean(
        [_representation_space(row, "base", "native").get("hijacking_index") for row in all_rows]
    )
    native_learned_hi = _mean(
        [_representation_space(row, "learned", "native").get("hijacking_index") for row in all_rows]
    )
    gains = [
        (_head_label(row), _number(_adapter_space(row).get("hijacking_index_gain")))
        for row in all_rows
    ]
    measured_gains = [(label, gain) for label, gain in gains if gain is not None]
    top_label, top_gain = max(measured_gains, key=lambda item: item[1]) if measured_gains else ("not measured", None)
    gain_ranks = _head_rank_map(
        all_rows,
        section="adapter_delta",
        metric="hijacking_index_gain",
    )
    learned_ranks = _head_rank_map(
        all_rows,
        section="learned",
        metric="hijacking_index",
    )
    selected_rank_text = " · ".join(
        f"{_head_label(row)} #{gain_ranks[_head_label(row)]}"
        for row in selected
        if _head_label(row) in gain_ranks
    )
    selected_count = sum(bool(_head_reasons(row)) for row in all_rows)
    examples = _map(hijacking.get("run")).get("examples")
    cards = "".join(
        (
            _metric_card("Per-head rows", str(len(all_rows)) if all_rows else "not measured", "supplied artifact"),
            _metric_card("Selected causal heads", str(selected_count) if all_rows else "not measured", "selection reasons recorded"),
            _metric_card("Residual-space HI", f"{_fmt(base_hi)} → {_fmt(learned_hi)}", "base → learned mean"),
            _metric_card("Native-head HI", f"{_fmt(native_base_hi)} → {_fmt(native_learned_hi)}", "base → learned mean"),
            _metric_card("Largest HI gain", _fmt(top_gain), top_label),
            _metric_card("Causal-head gain ranks", selected_rank_text or "not measured", f"of {len(all_rows)} heads"),
            _metric_card("Held-out sources", str(examples) if examples is not None else "not measured", "shared prediction boundary"),
        )
    )
    alignment = _signed_head_chart(
        selected,
        metric="hijacking_index",
        title="Base versus learned hijacking index at selected heads",
        chart_id="base-learned-head-alignment",
    )
    native_available = any(
        isinstance(_map(row.get("base")).get("native"), Mapping)
        or isinstance(_map(row.get("learned")).get("native"), Mapping)
        for row in all_rows
    )
    native_alignment = (
        _signed_head_chart(
            selected,
            metric="hijacking_index",
            title="Base versus learned native-head hijacking index",
            chart_id="base-learned-native-head-alignment",
            space="native",
        )
        if native_available
        else ""
    )
    shift = _adapter_delta_chart(selected)
    native_shift = _adapter_delta_chart(selected, space="native") if native_available else ""
    charts = ""
    if alignment or native_alignment or shift or native_shift:
        charts = '<div class="chart-grid">'
        if alignment:
            charts += _chart_card(
                "Base vs learned alignment",
                "HI adds genuine-over-fake French cosine advantage and trigger/language contrast alignment",
                alignment,
                wide=True,
            )
        if native_alignment:
            charts += _chart_card(
                "Native-head alignment",
                "The same signed operational HI before output projection into residual space",
                native_alignment,
                wide=True,
            )
        if shift:
            charts += _chart_card(
                "Adapter representation shift",
                "All points are learned minus base; positive strengthens the named alignment statistic",
                shift,
                wide=True,
            )
        if native_shift:
            charts += _chart_card(
                "Native-head adapter shift",
                "Learned-minus-base gains in the head's native representation coordinates",
                native_shift,
                wide=True,
            )
        charts += "</div>"

    rows: list[list[str]] = []
    for row in selected:
        base = _representation_space(row, "base")
        learned = _representation_space(row, "learned")
        delta = _adapter_space(row)
        native_base = _representation_space(row, "base", "native")
        native_learned = _representation_space(row, "learned", "native")
        native_delta = _adapter_space(row, "native")
        causal = _map(row.get("causal_scores"))
        rows.append(
            [
                _head_label(row),
                ", ".join(_head_reasons(row)) or "ranked by representation shift",
                _fmt(causal.get("trigger_fr")),
                _fmt(causal.get("language_fr")),
                _fmt(base.get("hijacking_index")),
                _fmt(learned.get("hijacking_index")),
                _fmt(delta.get("hijacking_index_gain")),
                str(gain_ranks.get(_head_label(row), "not measured")),
                str(learned_ranks.get(_head_label(row), "not measured")),
                _fmt(
                    _first_number(
                        delta,
                        "hijacking_index_gain_p_value_sign_flip",
                        "sign_flip_p_value",
                    ),
                    digits=5,
                ),
                _fmt(native_base.get("hijacking_index")) if native_available else "not measured",
                _fmt(native_learned.get("hijacking_index")) if native_available else "not measured",
                _fmt(native_delta.get("hijacking_index_gain")) if native_available else "not measured",
                _fmt(_first_number(delta, "raw_alignment_gain_delta", "cosine_advantage_gain", "raw_alignment_gain_gain", "raw_alignment_gain")),
                _fmt(_first_number(delta, "contrast_alignment_gain", "contrast_cosine_delta")),
                _fmt(_first_number(delta, "selective_shift_toward_french", "french_direction_projection_gain", "french_projection_gain", "projection_gain")),
            ]
        )
    table = _table(
        (
            "Head",
            "Why selected",
            "Patch trigger",
            "Patch natural FR",
            "Base HI",
            "Learned HI",
            "HI gain",
            "HI-gain rank",
            "Learned-HI rank",
            "Exact paired p",
            "Native base HI",
            "Native learned HI",
            "Native HI gain",
            "Raw-alignment Δ",
            "Contrast-alignment Δ",
            "Selective shift→FR",
        ),
        rows,
        caption=(
            "Selected-head representation audit. At most 16 heads are shown; the JSON artifact "
            "is canonical for all per-head values."
        ),
    )

    supplied_definitions = _map(hijacking.get("definitions"))
    supplied_rows: list[list[str]] = []
    for name, raw in supplied_definitions.items():
        if isinstance(raw, Mapping):
            equation = raw.get("equation", raw.get("formula", ""))
            meaning = raw.get("description", raw.get("meaning", raw.get("definition", "")))
            range_note = raw.get("range", raw.get("interpretation", ""))
        else:
            equation, meaning, range_note = "", raw, ""
        supplied_rows.append([_label(name), str(equation), str(meaning), str(range_note)])
    supplied = _table(
        ("Artifact term", "Formula", "Definition", "Range / reading"),
        supplied_rows,
        caption="Definitions recorded by the head-hijacking analysis artifact",
    ) if supplied_rows else '<p class="empty">No additional artifact definitions were supplied.</p>'
    definitions = (
        '<div class="equation-grid">'
        '<article class="equation-card"><p class="eyebrow">Condition vectors</p>'
        '<code>T = genuine trigger · K = fake trigger · F = natural French · E = English</code>'
        '<p>Every vector is a per-example head representation at the same final prompt '
        'prediction boundary; reported values are arithmetic means over held-out examples.</p></article>'
        '<article class="equation-card"><p class="eyebrow">Raw alignment gain</p>'
        '<code>A_raw = cos(T,F) − cos(K,F)</code><p>Positive values mean the genuine-trigger '
        'representation is more cosine-aligned with natural French than its matched fake control.</p></article>'
        '<article class="equation-card"><p class="eyebrow">Contrast alignment</p>'
        '<code>A_contrast = cos(T−K, F−E)</code><p>This tests whether the genuine-minus-fake '
        'trigger direction aligns with the French-minus-English language direction.</p></article>'
        '<article class="equation-card"><p class="eyebrow">Hijacking index · HI</p>'
        '<code>HI = A_raw + A_contrast</code><p>This repository-defined index is signed and '
        'unclipped, with mathematical range [−3, 3]; it is not a probability or causal effect. '
        'Positive values combine genuine-over-fake French alignment with aligned trigger and '
        'language contrasts.</p></article>'
        '<article class="equation-card"><p class="eyebrow">Contrast norm ratio</p>'
        '<code>R_norm = ||T−K||₂ / (||F−E||₂ + 10⁻¹²)</code><p>This reports relative shift '
        'magnitude; it is not itself a directional alignment score.</p></article>'
        '<article class="equation-card"><p class="eyebrow">Adapter effect</p>'
        '<code>Δmetric = metric_learned − metric_base</code><p>Positive HI or alignment delta '
        'means the merged adapter strengthened the corresponding operational alignment. '
        'Selective French shift subtracts the fake-trigger shift from the genuine-trigger shift.</p></article></div>'
        + supplied
    )
    return cards, charts, table, definitions


def _key_value_rows(values: Sequence[tuple[str, Any]]) -> str:
    rows: list[str] = []
    for label, value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        rows.append(f'<div><dt>{_e(label)}</dt><dd>{_e(value)}</dd></div>')
    return "".join(rows)


def _training_summary(
    provenance: Mapping[str, Any],
    metrics: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
) -> tuple[str, str, str]:
    config = _map(provenance.get("training_config"))
    metric_root = _map(metrics or provenance.get("metrics"))
    train = _map(metric_root.get("train"))
    validation = _map(metric_root.get("validation"))
    test = _map(metric_root.get("test"))
    cards = "".join(
        (
            _metric_card("Train loss", _fmt(train.get("train_loss"))),
            _metric_card("Validation loss", _fmt(validation.get("validation_loss", validation.get("eval_loss")))),
            _metric_card("Test loss", _fmt(test.get("test_loss"))),
            _metric_card("Best validation", _fmt(provenance.get("best_metric"))),
            _metric_card("Epochs", _fmt(config.get("num_train_epochs", train.get("epoch")), digits=1)),
            _metric_card("Training time", (_fmt(train.get("train_runtime"), digits=1) + "s") if _number(train.get("train_runtime")) is not None else "not measured"),
        )
    )
    config_rows = _key_value_rows(
        (
            ("Base model", config.get("model_name")),
            ("Precision", config.get("dtype")),
            ("Learning rate", config.get("learning_rate")),
            ("LoRA rank / alpha", f"{config.get('lora_rank')} / {config.get('lora_alpha')}" if config else None),
            ("LoRA dropout", config.get("lora_dropout")),
            ("Target modules", config.get("lora_target_modules")),
            ("Microbatch / accumulation", f"{config.get('per_device_train_batch_size')} / {config.get('gradient_accumulation_steps')}" if config else None),
            ("Max sequence length", config.get("max_length")),
            ("Source count", provenance.get("source_count")),
            ("Seed", provenance.get("seed")),
        )
    )
    family_counts = _map(provenance.get("family_counts"))
    family_names: list[str] = []
    for split in ("train", "validation", "test"):
        for family in _map(family_counts.get(split)):
            if family not in family_names:
                family_names.append(str(family))
    dataset_rows = [
        [
            _label(family),
            str(_map(family_counts.get("train")).get(family, 0)),
            str(_map(family_counts.get("validation")).get(family, 0)),
            str(_map(family_counts.get("test")).get(family, 0)),
        ]
        for family in family_names
    ]
    dataset_table = _table(
        ("Training family", "Train", "Validation", "Test"),
        dataset_rows,
        caption="Continuation-only examples by source-disjoint split",
    )
    curve = _training_curve(trainer_state)
    return cards, config_rows, dataset_table + (f'<div class="chart-card wide curve">{curve}</div>' if curve else '<p class="empty">Trainer log history was not supplied, so no loss curve is drawn.</p>')


def _provenance_rows(
    training: Mapping[str, Any],
    behavior: Mapping[str, Any],
    analysis: Mapping[str, Any],
    hijacking: Mapping[str, Any] | None = None,
) -> str:
    packages = _map(training.get("packages"))
    runtime = _map(behavior.get("provenance"))
    configuration = _map(behavior.get("configuration"))
    model_details = _map(runtime.get("model_details"))
    causal_provenance = _map(analysis.get("provenance"))
    causal_run = _map(analysis.get("run"))
    hijacking = _map(hijacking)
    hijacking_provenance = _map(hijacking.get("provenance"))
    hijacking_run = _map(hijacking.get("run"))
    rows: list[tuple[str, Any]] = [
        ("Training schema", training.get("schema_version")),
        ("Training created UTC", training.get("created_at_utc")),
        ("Training seed", training.get("seed")),
        ("Disjoint source hash", training.get("source_sha256")),
        ("Example corpus hash", training.get("examples_sha256")),
        ("Final run hash", training.get("final_run_sha256")),
        ("Behavior created UTC", behavior.get("created_at")),
        ("Behavior dataset hash", runtime.get("dataset_sha256")),
        ("Behavior seed", configuration.get("seed", runtime.get("seed"))),
        ("Candidate kind", runtime.get("candidate_kind")),
        ("Base identifier", runtime.get("base_identifier")),
        ("Candidate identifier", runtime.get("candidate_identifier")),
        ("Model architecture", _map(model_details.get("candidate")).get("architecture") if model_details else None),
        ("Causal schema / status", f"{analysis.get('schema_version')} / {analysis.get('status')}" if analysis else None),
        ("Causal created UTC", analysis.get("generated_at_utc")),
        ("Causal examples / seconds", f"{causal_run.get('examples')} / {causal_run.get('elapsed_seconds')}" if causal_run else None),
        ("Causal scientific-results hash", causal_provenance.get("scientific_results_sha256")),
        ("Merged model weights hash", causal_provenance.get("model_weights_sha256")),
        ("Causal corpus hash", causal_provenance.get("corpus_sha256")),
        ("Causal deterministic algorithms", causal_provenance.get("deterministic_algorithms")),
        ("Causal offline local files only", causal_provenance.get("offline_local_files_only")),
        ("Causal source command", causal_provenance.get("source_command")),
        ("Hijacking schema / status", f"{hijacking.get('schema_version')} / {hijacking.get('status', 'complete')}" if hijacking else None),
        ("Hijacking created UTC", hijacking.get("generated_at_utc", hijacking.get("created_at_utc"))),
        ("Hijacking examples / split", f"{hijacking_run.get('examples')} / {hijacking_run.get('split')}" if hijacking_run else None),
        ("Hijacking position policy", hijacking_run.get("position_policy")),
        ("Hijacking scientific-results hash", hijacking_provenance.get("scientific_results_sha256")),
        ("Hijacking base weights hash", hijacking_provenance.get("base_model_weights_sha256", hijacking_provenance.get("base_weights_sha256"))),
        ("Hijacking learned weights hash", hijacking_provenance.get("learned_model_weights_sha256", hijacking_provenance.get("learned_weights_sha256"))),
        ("Hijacking source command", hijacking_provenance.get("source_command")),
    ]
    rows.extend((f"Package · {name}", version) for name, version in packages.items())
    for split, digest in _map(training.get("split_source_id_sha256")).items():
        rows.append((f"{_label(split)} source IDs hash", digest))
    return _key_value_rows(rows)


def render_learned_trigger_report(
    training_provenance: Mapping[str, Any],
    behavior_artifact: Mapping[str, Any],
    *,
    training_metrics: Mapping[str, Any] | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    causal_analysis: Mapping[str, Any] | None = None,
    hijacking_analysis: Mapping[str, Any] | None = None,
    title: str = "Learned Trigger: Head Representations & Hijacking",
    generated_at: str | None = None,
) -> str:
    """Render a deterministic, self-contained learned-trigger evidence report.

    ``causal_analysis`` and ``hijacking_analysis`` are optional. When omitted,
    their sections are visibly marked pending; missing evidence is never
    converted into a zero or a successful result.
    """

    if not isinstance(training_provenance, Mapping):
        raise TypeError("training_provenance must be a mapping")
    if not isinstance(behavior_artifact, Mapping):
        raise TypeError("behavior_artifact must be a mapping")
    metrics = _map(training_metrics)
    state = _map(trainer_state)
    analysis = _analysis_root(causal_analysis)
    hijacking = _hijacking_root(hijacking_analysis)
    models = _map(behavior_artifact.get("models"))
    base = _map(models.get("base"))
    candidate = _map(models.get("candidate"))
    base_metrics = _map(base.get("metrics"))
    candidate_metrics = _map(candidate.get("metrics"))
    candidate_families = _map(candidate.get("families"))
    genuine_family = _map(candidate_families.get("genuine-trigger"))
    fake_family = _map(candidate_families.get("fake-trigger"))
    no_trigger_family = _map(candidate_families.get("no-trigger"))
    fake_language_rates = _map(fake_family.get("generation_language_rates"))
    no_trigger_language_rates = _map(no_trigger_family.get("generation_language_rates"))
    genuine_margin = _number(genuine_family.get("mean_margin_fr_minus_en"))
    fake_margin = _number(fake_family.get("mean_margin_fr_minus_en"))
    exact_fake_separation = (
        genuine_margin - fake_margin
        if genuine_margin is not None and fake_margin is not None
        else None
    )
    genuine_rate = candidate_metrics.get("trigger_success_rate")
    near_miss_rate = candidate_metrics.get("near_miss_specificity")
    comparison = _map(behavior_artifact.get("comparison"))
    trigger_set = _map(training_provenance.get("trigger_set"))
    configuration = _map(behavior_artifact.get("configuration"))
    trigger = trigger_set.get("genuine", configuration.get("genuine_trigger", configuration.get("trigger", "not recorded")))
    fakes = _seq(trigger_set.get("fakes", configuration.get("fake_triggers", [])))
    profile = _map(trigger_set.get("token_profile"))
    generated = generated_at or datetime.now(timezone.utc).isoformat()
    config = _map(training_provenance.get("training_config"))
    model_name = config.get("model_name", _map(_map(behavior_artifact.get("provenance")).get("model_details")).get("candidate", "Qwen2.5-0.5B"))
    candidate_label = candidate.get("model_label", comparison.get("candidate_label", "LoRA / merged"))
    base_label = base.get("model_label", comparison.get("base_label", "base"))

    behavior_svg = _behavior_chart(base, candidate)
    training_cards, training_config_rows, training_evidence = _training_summary(
        training_provenance, metrics, state
    )
    head_charts, top_heads = _head_cards(analysis)
    layer_charts = _layer_cards(analysis)
    relation_charts = _relationship_cards(analysis)
    causal_available = bool(head_charts or layer_charts or relation_charts)
    pending = (
        '<div class="pending"><strong>Causal analysis pending.</strong><p>The report was rendered '
        'without a causal-analysis artifact. No localization, overlap, cosine, or ablation claim '
        'is made yet.</p></div>'
    )
    hijacking_cards, hijacking_charts, hijacking_table, hijacking_definitions = (
        _hijacking_evidence(hijacking) if hijacking else ("", "", "", "")
    )
    hijacking_available = bool(_seq(hijacking.get("per_head")))
    hijacking_pending = (
        '<div class="pending"><strong>Head-hijacking comparison pending.</strong><p>The report '
        'was rendered without a base-versus-learned representation artifact. No adapter-shift, '
        'head-alignment, or representation-hijacking claim is made.</p></div>'
    )

    analysis_top = _map(analysis.get("top_heads"))
    trigger_heads = {
        (entry.get("layer"), entry.get("head"))
        for entry in _seq(analysis_top.get("trigger-fr"))
        if isinstance(entry, Mapping)
        and entry.get("layer") is not None
        and entry.get("head") is not None
    }
    language_heads = {
        (entry.get("layer"), entry.get("head"))
        for entry in _seq(analysis_top.get("language-fr"))
        if isinstance(entry, Mapping)
        and entry.get("layer") is not None
        and entry.get("head") is not None
    }
    shared_heads = sorted(trigger_heads & language_heads)
    shared_head_text = " · ".join(f"L{layer}H{head}" for layer, head in shared_heads)
    overlap = _map(analysis.get("overlap"))
    overlap_labels = [str(item) for item in _seq(overlap.get("labels"))]
    try:
        trigger_index = overlap_labels.index("trigger-fr")
        language_index = overlap_labels.index("language-fr")
    except ValueError:
        trigger_index, language_index = 0, 1
    overlap_intersection = _cell(overlap.get("intersections"), trigger_index, language_index)
    overlap_jaccard = _cell(overlap.get("jaccard"), trigger_index, language_index)
    overlap_p = _cell(overlap.get("p_values"), trigger_index, language_index)
    expected_overlap = _number(overlap.get("expected_jaccard"))
    cosine = _map(analysis.get("cosine"))
    cosine_value = _cell(cosine.get("values"), 0, 0)

    peak_score: float | None = None
    peak_layer: int | None = None
    peak_position: int | None = None
    layer_matrix = _seq(_score_matrix(_map(analysis.get("layer_scores")).get("trigger-fr")))
    for layer_index, row in enumerate(layer_matrix):
        for position_index, value in enumerate(_seq(row)):
            score = _number(value)
            if score is not None and (peak_score is None or score > peak_score):
                peak_score = score
                peak_layer = layer_index
                peak_position = position_index
    position_labels = _seq(_map(analysis.get("layer_positions")).get("trigger-fr"))
    peak_token = (
        str(position_labels[peak_position])
        if peak_position is not None and peak_position < len(position_labels)
        else (f"T{peak_position + 1}" if peak_position is not None else "not measured")
    )
    trigger_words = str(trigger).split()
    per_word = [_number(item) for item in _seq(profile.get("per_word"))]
    peak_word = ""
    if peak_position is not None and per_word and all(item is not None for item in per_word):
        boundary = 0
        for word, length in zip(trigger_words, per_word):
            boundary += int(length or 0)
            if peak_position < boundary:
                peak_word = word
                break

    ablations = _map(analysis.get("ablations"))
    trigger_ablation = _map(ablations.get("trigger-fr"))
    language_ablation = _map(ablations.get("language-fr"))
    trigger_baseline, _ = _ablation_at(trigger_ablation, 0)
    trigger_selected_two, trigger_random_two = _ablation_at(trigger_ablation, 2)
    language_baseline, _ = _ablation_at(language_ablation, 0)
    language_selected_two, language_random_two = _ablation_at(language_ablation, 2)
    causal_summary_cards = "".join(
        (
            _metric_card("Shared top-10 heads", shared_head_text or "not measured", "trigger ∩ natural French"),
            _metric_card("Intersection", f"{_fmt(overlap_intersection, digits=0)} / {overlap.get('top_k', 'k?')}", "two ranked sets"),
            _metric_card("Jaccard", _fmt(overlap_jaccard), f"chance expectation {_fmt(expected_overlap)}"),
            _metric_card("Overlap p-value", _fmt(overlap_p, digits=5), "exact hypergeometric tail"),
            _metric_card("Shared-head cosine", _fmt(cosine_value), str(cosine.get("head", "head not recorded"))),
            _metric_card("Layer/token peak", f"L{peak_layer} / {peak_token}" if peak_layer is not None else "not measured", f"score {_fmt(peak_score)}" + (f" · word {peak_word}" if peak_word else "")),
        )
    )
    ablation_summary_cards = "".join(
        (
            _metric_card("Trigger-FR · 0 → 2 heads", f"{_fmt(trigger_baseline)} → {_fmt(trigger_selected_two)}", f"random {_fmt(trigger_random_two)} PPL"),
            _metric_card("Natural-FR · 0 → 2 heads", f"{_fmt(language_baseline)} → {_fmt(language_selected_two)}", f"random {_fmt(language_random_two)} PPL"),
            _metric_card("First two selected", " · ".join(str(item) for item in _seq(trigger_ablation.get("ordered_heads"))[:2]) or "not measured", "shared trigger/language heads"),
        )
    )
    random_repeats = trigger_ablation.get(
        "random_repeats", language_ablation.get("random_repeats", "an unrecorded number of")
    )
    ablation_interpretation = (
        "After the first two selected heads, perplexity rises sharply in both trigger-French "
        "and natural-French conditions, while the matched random controls remain near baseline. "
        f"The random-control estimates use {random_repeats} repeats. This supports a shared "
        "causal role in this run; it does not establish that these heads form the only circuit."
    )

    headline_cards = "".join(
        (
            _metric_card("Genuine strict joint", _percent(candidate_metrics.get("trigger_success_rate")), "teacher + generation"),
            _metric_card("Base strict joint", _percent(base_metrics.get("trigger_success_rate")), "same genuine prompts"),
            _metric_card("Control strict joint", _percent(candidate_metrics.get("trigger_specificity")), "fake + absent pooled"),
            _metric_card("No-trigger strict joint", _percent(candidate_metrics.get("english_retention")), "teacher + generation"),
            _metric_card("Natural-FR strict joint", _percent(candidate_metrics.get("natural_french_retention")), "teacher + generation"),
            _metric_card("Causal evidence", "available" if causal_available else "pending", "separate from behavior"),
        )
    )

    fake_specificity_cards = "".join(
        (
            _metric_card("Generated English", _percent(fake_language_rates.get("en")), "language heuristic"),
            _metric_card("Teacher prefers English", _percent(fake_family.get("teacher_forced_correct_rate")), "paired likelihood"),
            _metric_card("Strict joint", _percent(fake_family.get("behavior_success_rate")), "both checks pass"),
            _metric_card("Generated French", _percent(fake_language_rates.get("fr")), "control leakage"),
            _metric_card("Generation unknown", _percent(fake_language_rates.get("unknown")), "heuristic abstention"),
            _metric_card("Genuine − fake margin", _fmt(exact_fake_separation), "FR−EN likelihood separation"),
        )
    )
    no_trigger_cards = "".join(
        (
            _metric_card("Generated English", _percent(no_trigger_language_rates.get("en")), "language heuristic"),
            _metric_card("Teacher prefers English", _percent(no_trigger_family.get("teacher_forced_correct_rate")), "paired likelihood"),
            _metric_card("Strict joint", _percent(no_trigger_family.get("behavior_success_rate")), "both checks pass"),
        )
    )

    fake_chips = "".join(f'<code class="control-chip">{_e(item)}</code>' for item in fakes)
    if not fake_chips:
        fake_chips = '<span class="empty-inline">No fake trigger list recorded</span>'
    fake_count = fake_family.get("count", "not recorded")
    fake_mode = configuration.get("fake_trigger_mode", "assigned")
    fake_control_note = (
        f"Mode: {fake_mode}; {len(fakes)} phrases produced {fake_count} held-out prompt instances."
    )
    token_profile = (
        f"{profile.get('total')} tokenizer tokens · per word {profile.get('per_word')}"
        if profile
        else "token profile not recorded"
    )

    result_summary = (
        f"The adapted model passed the strict teacher-plus-generation criterion on "
        f"{_percent(genuine_rate)} of genuine-trigger prompts, versus "
        f"{_percent(base_metrics.get('trigger_success_rate'))} for the base model. "
        f"Across {fake_count} matched fake-trigger prompts, generation was English "
        f"{_percent(fake_language_rates.get('en'))} of the time, while paired likelihood "
        f"preferred English on {_percent(fake_family.get('teacher_forced_correct_rate'))}; "
        f"the strict conjunction was {_percent(fake_family.get('behavior_success_rate'))}. "
        f"Near-miss strict specificity was {_percent(near_miss_rate)}, so the learned "
        f"switch is behaviorally real but not narrowly exact-string-specific."
    )

    hijacking_rows = [
        row for row in _seq(hijacking.get("per_head")) if isinstance(row, Mapping)
    ]
    base_hi_mean = _mean(
        [_representation_space(row, "base").get("hijacking_index") for row in hijacking_rows]
    )
    learned_hi_mean = _mean(
        [_representation_space(row, "learned").get("hijacking_index") for row in hijacking_rows]
    )
    hi_gain_mean = _mean(
        [
            _adapter_space(row).get("hijacking_index_gain")
            for row in hijacking_rows
        ]
    )
    hijacking_gain_ranks = _head_rank_map(
        hijacking_rows,
        section="adapter_delta",
        metric="hijacking_index_gain",
    )
    selected_hijacking_rows = _selected_hijacking_rows(hijacking)
    selected_hijacking_summary = "; ".join(
        f"{_head_label(row)} gained {_fmt(_adapter_space(row).get('hijacking_index_gain'))} "
        f"(rank #{hijacking_gain_ranks.get(_head_label(row), '?')} of {len(hijacking_rows)}, "
        f"exact paired p={_fmt(_adapter_space(row).get('hijacking_index_gain_p_value_sign_flip'), digits=5)})"
        for row in selected_hijacking_rows
    )
    top_hijacking_row = max(
        hijacking_rows,
        key=lambda row: _number(_adapter_space(row).get("hijacking_index_gain"))
        if _number(_adapter_space(row).get("hijacking_index_gain")) is not None
        else -math.inf,
        default=None,
    )
    top_hijacking_text = (
        f"The grid-wide maximum was {_head_label(top_hijacking_row)} at "
        f"{_fmt(_adapter_space(top_hijacking_row).get('hijacking_index_gain'))}, which was "
        "not in the shared causal top-k intersection. "
        if top_hijacking_row is not None
        and not _head_reasons(top_hijacking_row)
        else ""
    )
    hijacking_interpretation = (
        "At the shared causal heads, "
        + selected_hijacking_summary
        + ". "
        + top_hijacking_text
        + "Across the "
        + str(len(hijacking_rows))
        + " supplied head rows, mean residual-space operational hijacking index changed from "
        + _fmt(base_hi_mean)
        + " in the base model to "
        + _fmt(learned_hi_mean)
        + " after LoRA (mean gain "
        + _fmt(hi_gain_mean)
        + "). The selected-head table connects those descriptive representation shifts to "
        "the separately measured patching rankings. This is evidence for movement toward "
        "the operational natural-French alignment under the stated cosine metrics, not proof of a unique "
        "circuit or covert intent."
        if hijacking_available
        else ""
    )

    differences = list(_DEFAULT_DIFFERENCES)
    if _number(near_miss_rate) is not None and float(near_miss_rate) < 0.8:
        differences.insert(
            0,
            "Near-miss strict specificity was "
            + _percent(near_miss_rate)
            + "; reordered and one-character variants frequently activated French. "
            "This LoRA learned a broader trigger neighborhood than an exact-string gate.",
        )
    fake_teacher = _number(fake_family.get("teacher_forced_correct_rate"))
    fake_generation = _number(fake_language_rates.get("en"))
    if (
        fake_teacher is not None
        and fake_generation is not None
        and fake_generation - fake_teacher >= 0.2
    ):
        differences.insert(
            1,
            "Fake-trigger generation stayed English at "
            + _percent(fake_generation)
            + ", but paired likelihood preferred English on only "
            + _percent(fake_teacher)
            + ". The strict joint result must therefore not be described as the "
            "generated-language specificity rate.",
        )
    ablation_policy = str(trigger_ablation.get("policy", ""))
    if "not specified by the paper" in ablation_policy.lower():
        differences.append(
            "Selected-head ablation uses the reported local joint-rank fallback rather "
            "than a paper-specified ordering; only "
            + str(trigger_ablation.get("random_repeats", "an unrecorded number of"))
            + " random repeats were run."
        )
    if hijacking_available:
        hijacking_run = _map(hijacking.get("run"))
        differences.extend(
            (
                "The representation comparison averages only "
                + str(hijacking_run.get("examples", "the recorded"))
                + " held-out sources and samples one shared prediction-boundary position. "
                "It does not establish that the same geometry holds at other tokens, prompts, "
                "seeds, or models.",
                "“Hijacking index” is a signed, unclipped operational cosine statistic with "
                "mathematical range [−3, 3], defined in this repository. Positive values combine "
                "genuine-over-fake French alignment with "
                "trigger/language contrast alignment; they do not establish deceptive intent, a unique causal "
                "mechanism, or the paper's exact latent representation.",
                "Heads labeled selected were chosen using the causal patching results from this "
                "same small run. Their representation summaries are exploratory and are not an "
                "independent confirmatory test.",
                "The adapter-shift comparison uses the base and merged models. It localizes a "
                "net representation change but does not isolate individual LoRA matrices or "
                "training examples as causes.",
            )
        )
        for item in _seq(hijacking.get("limitations")):
            if str(item) not in differences:
                differences.append(str(item))
    for key in ("differences_from_paper", "limitations"):
        for item in _seq(analysis.get(key)):
            if str(item) not in differences:
                differences.append(str(item))
    limitations_html = "".join(
        f'<li><span>{index:02d}</span><p>{_e(item)}</p></li>'
        for index, item in enumerate(differences, start=1)
    )
    provenance_rows = _provenance_rows(
        training_provenance, behavior_artifact, analysis, hijacking
    )
    metric_definitions = _map(behavior_artifact.get("metric_definitions"))
    definition_rows = _key_value_rows([(str(key), value) for key, value in metric_definitions.items()])

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{_e(title)}</title>
<style>
:root{{--page:#07111c;--panel:#0e1c2a;--panel2:#132536;--line:#294154;--ink:#edf6f7;--muted:#9db0bb;--cyan:#2dd4bf;--amber:#f6bd60;--coral:#fb7185;--violet:#a78bfa;--blue:#60a5fa;--good:#4ade80;--bad:#fb7185}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 88% 0,#18364a 0,transparent 34rem),radial-gradient(circle at 0 18rem,#162d35 0,transparent 28rem),var(--page);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}code,.mono{{font-family:ui-monospace,"Cascadia Code","SFMono-Regular",monospace}}.skip{{position:absolute;left:-9999px;top:1rem;background:var(--cyan);color:#041418;padding:.6rem 1rem;border-radius:.5rem;z-index:10}}.skip:focus{{left:1rem}}
.hero{{padding:4.8rem max(1rem,calc((100vw - 1180px)/2)) 3.2rem;border-bottom:1px solid var(--line);position:relative;overflow:hidden}}.hero:after{{content:"FR";position:absolute;right:max(1rem,calc((100vw - 1180px)/2));bottom:-3.7rem;font-size:16rem;font-weight:900;letter-spacing:-.12em;color:#ffffff05;line-height:1}}.hero-copy{{max-width:900px;position:relative;z-index:1}}.kicker,.eyebrow{{margin:0 0 .55rem;color:var(--cyan);font-size:.71rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase}}h1{{font-size:clamp(2.7rem,7vw,5.8rem);line-height:.94;letter-spacing:-.06em;margin:0 0 1.25rem}}.lede{{font-size:clamp(1.05rem,2vw,1.35rem);max-width:780px;color:#c5d7dd;margin:0}}.stamp{{display:flex;gap:.65rem;flex-wrap:wrap;margin-top:1.55rem}}.stamp span{{border:1px solid var(--line);border-radius:99px;padding:.35rem .65rem;background:#0c1b28;color:var(--muted);font-size:.77rem}}
.page-nav{{position:sticky;top:0;z-index:5;background:#07111ce8;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);overflow:auto}}.page-nav div{{width:min(1180px,calc(100% - 2rem));margin:auto;display:flex;gap:.45rem;padding:.65rem 0}}.page-nav a{{white-space:nowrap;text-decoration:none;color:#b9cbd1;border:1px solid var(--line);border-radius:99px;padding:.3rem .65rem;font-size:.73rem}}.page-nav a:hover,.page-nav a:focus{{color:var(--cyan);border-color:var(--cyan)}}main{{width:min(1180px,calc(100% - 2rem));margin:auto;padding:2rem 0 5rem}}section{{margin:0 0 5rem;scroll-margin-top:4rem}}.boundary{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:1rem;overflow:hidden;margin-bottom:2rem}}.boundary article{{background:var(--panel);padding:1.25rem 1.4rem}}.boundary h2{{font-size:1rem;margin:.1rem 0 .35rem}}.boundary p{{margin:0;color:var(--muted);font-size:.9rem}}.boundary .measured h2{{color:var(--cyan)}}.boundary .scope h2{{color:var(--amber)}}.finding-panel{{display:grid;grid-template-columns:minmax(180px,.45fr) 1.55fr;gap:1.5rem;align-items:start;background:linear-gradient(135deg,#17312f,#101f2e);border:1px solid #2e5a56;border-radius:1rem;padding:1.25rem;margin-bottom:2rem}}.finding-panel h2{{font-size:1.45rem;line-height:1.1;margin:.1rem 0}}.finding-panel p:last-child{{margin:0;color:#c8dbdc;font-size:1.02rem}}
.metric-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.7rem;margin:0 0 2rem}}.metric{{background:linear-gradient(145deg,var(--panel2),#0b1824);border:1px solid var(--line);border-radius:.85rem;padding:.9rem;min-width:0}}.metric dt{{color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.08em}}.metric dd{{font-size:1.43rem;line-height:1.1;font-weight:780;margin:.2rem 0;overflow-wrap:anywhere}}.metric-note{{display:block;color:#718b98;font-size:.67rem;margin-top:.25rem}}.section-heading{{display:flex;justify-content:space-between;gap:2rem;align-items:end;border-bottom:1px solid var(--line);padding-bottom:1rem;margin-bottom:1.35rem}}.section-heading h2{{font-size:clamp(1.8rem,4vw,3.15rem);line-height:1;letter-spacing:-.04em;margin:0}}.section-heading>p{{max-width:450px;text-align:right;color:var(--muted);margin:0}}.subheading{{font-size:1.08rem;margin:2rem 0 .8rem;color:#d6e5e9}}
.trigger-panel{{display:grid;grid-template-columns:minmax(260px,.7fr) 1.3fr;gap:1rem}}.trigger-main,.control-panel{{background:linear-gradient(145deg,#132b37,#0c1925);border:1px solid var(--line);border-radius:1rem;padding:1.25rem}}.trigger-main h3,.control-panel h3{{margin:.1rem 0 .7rem;font-size:1.05rem}}.trigger{{display:block;background:#07131e;border:1px solid #2a5360;color:#7ce8d6;padding:1rem;border-radius:.7rem;font-size:1.15rem;overflow-wrap:anywhere}}.trigger.compact{{font-size:.8rem;padding:.55rem}}.trigger-main dl,.variant-card dl{{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin:1rem 0 0}}.trigger-main .metric,.variant-card .metric{{padding:.65rem}}.trigger-main .metric dd,.variant-card .metric dd{{font-size:1rem}}.control-list{{display:flex;flex-wrap:wrap;gap:.45rem}}.control-chip{{font-size:.75rem;background:#101c2b;border:1px solid #32485b;color:#c6d1dc;padding:.35rem .48rem;border-radius:.4rem}}
.chart-card{{margin:0;background:linear-gradient(145deg,var(--panel2),#0d1925);border:1px solid var(--line);border-radius:1rem;overflow:hidden;min-width:0}}.chart-card.wide,.chart-grid>.wide{{grid-column:1/-1}}.chart-card figcaption{{display:flex;justify-content:space-between;gap:1rem;padding:.85rem 1rem;border-bottom:1px solid var(--line)}}.chart-card figcaption span{{color:var(--muted);font-size:.75rem;text-align:right}}.chart-scroll{{overflow:auto;padding:.5rem}}.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}svg{{display:block;width:100%;height:auto;min-width:360px}}.axis-label,.legend-label{{fill:#9fb4be;font-size:12px}}.row-label{{font-size:11px}}.axis-title{{fill:#cbdde2;font-size:13px;font-weight:650}}.cell-value{{fill:#f4fbfc;font-size:12px;font-weight:700;pointer-events:none}}.grid-line{{stroke:#2a4151;stroke-width:1}}.zero-line{{stroke:#d7e3e7;stroke-width:1.5;opacity:.8}}.bar.base{{fill:#637b89}}.bar.candidate{{fill:var(--amber)}}.bar.representation-base{{fill:#64748b}}.bar.representation-learned{{fill:var(--violet)}}.bar-value{{font-size:11px;font-weight:750}}.bar-value.base{{fill:#a8bbc4}}.bar-value.candidate{{fill:#ffdda3}}.bar-value.representation-base{{fill:#aebdce}}.bar-value.representation-learned{{fill:#d9cdff}}.series{{stroke-width:3;stroke-linecap:round;stroke-linejoin:round}}.series.train{{stroke:var(--cyan)}}.series.validation{{stroke:var(--amber)}}.point.train{{fill:var(--cyan)}}.point.validation{{fill:var(--amber)}}.series.selected{{stroke:var(--coral)}}.series.random{{stroke:var(--cyan);stroke-dasharray:7 5}}.point.selected{{fill:var(--coral);stroke:#ffe1dc;stroke-width:1.5}}.point.random{{fill:var(--cyan);stroke:#d7fff6;stroke-width:1.5}}.error-bar{{stroke:#63cdb8;stroke-width:2;opacity:.7}}.delta-stem{{stroke-width:2;opacity:.65}}.delta-point.delta-hi,.delta-stem.delta-hi{{fill:var(--coral);stroke:var(--coral)}}.delta-point.delta-cosine,.delta-stem.delta-cosine{{fill:var(--cyan);stroke:var(--cyan)}}.delta-point.delta-projection,.delta-stem.delta-projection{{fill:var(--amber);stroke:var(--amber)}}.curve{{margin-top:1rem;padding:.6rem}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:.85rem;margin:1rem 0}}table{{width:100%;border-collapse:collapse;background:#0c1925}}caption{{text-align:left;padding:.8rem 1rem;background:#142638;font-weight:700}}th,td{{padding:.72rem .85rem;border-top:1px solid var(--line);text-align:left;vertical-align:top}}thead th{{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}tbody td:first-child{{font-weight:650}}.pending{{background:#231f17;border:1px dashed #6d5b36;border-radius:.9rem;padding:1.2rem}}.pending strong{{color:var(--amber)}}.pending p{{margin:.3rem 0 0;color:#bdb39f}}.empty,.empty-inline{{color:var(--muted)}}
.variant-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.75rem}}.variant-card{{display:grid;grid-template-columns:1.2fr .8fr;gap:1rem;background:var(--panel);border:1px solid var(--line);border-radius:.9rem;padding:1rem}}.variant-card dl{{margin:0}}.sample-list{{display:grid;gap:.85rem}}.sample-card{{background:#0c1926;border:1px solid var(--line);border-radius:1rem;overflow:hidden}}.sample-card header{{display:flex;justify-content:space-between;align-items:center;padding:.7rem 1rem;background:#132538;border-bottom:1px solid var(--line)}}.sample-card header>div{{display:flex;gap:.7rem;align-items:center}}.family{{color:var(--cyan);font-size:.68rem;text-transform:uppercase;letter-spacing:.09em}}.badge{{font-size:.67rem;font-weight:750;text-transform:uppercase;letter-spacing:.07em;padding:.25rem .48rem;border-radius:99px}}.badge.good{{background:#173b2a;color:#77e29d}}.badge.bad{{background:#451f2a;color:#ff9aaa}}.badge.neutral{{background:#24313c;color:#aebbc3}}.prompt-block{{padding:.8rem 1rem;border-bottom:1px solid var(--line)}}.prompt-block span,.generation-grid span{{display:block;color:var(--muted);font-size:.67rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem}}.prompt-block code{{display:block;color:#b9d9e1;white-space:pre-wrap;overflow-wrap:anywhere}}.generation-grid{{display:grid;grid-template-columns:1fr 1fr}}.generation-grid>div{{padding:1rem;min-width:0}}.generation-grid>div+div{{border-left:1px solid var(--line)}}.generation-grid .candidate-output{{background:#15231f}}.generation-grid p{{white-space:pre-wrap;overflow-wrap:anywhere;margin:.2rem 0 .5rem}}.generation-grid small{{color:var(--muted)}}
.config-grid{{display:grid;grid-template-columns:.75fr 1.25fr;gap:1rem}}.provenance{{background:#0b1824;border:1px solid var(--line);border-radius:1rem;padding:1rem}}.provenance dl{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.25rem 2rem;margin:0}}.provenance dl div{{display:grid;grid-template-columns:minmax(9rem,.6fr) 1.4fr;gap:.7rem;border-bottom:1px solid #203646;padding:.42rem 0;min-width:0}}.provenance dt{{color:var(--muted);font-size:.75rem}}.provenance dd{{margin:0;font:500 .72rem/1.5 ui-monospace,monospace;overflow-wrap:anywhere}}.definitions{{margin-top:1rem}}.equation-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin:1rem 0}}.equation-card{{background:linear-gradient(145deg,#1b1830,#111d2b);border:1px solid #463d69;border-radius:.9rem;padding:1rem}}.equation-card code{{display:block;color:#ddd3ff;background:#0d1220;border-radius:.55rem;padding:.7rem;overflow-wrap:anywhere}}.equation-card p:last-child{{color:#b9c3d4;margin:.7rem 0 0;font-size:.86rem}}.limits{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem;list-style:none;padding:0}}.limits li{{display:grid;grid-template-columns:2rem 1fr;gap:.7rem;background:#191d27;border:1px solid #3a3d49;border-radius:.8rem;padding:1rem}}.limits span{{color:var(--amber);font:700 .75rem/1.8 ui-monospace,monospace}}.limits p{{margin:0;color:#c0c8ce}}footer{{border-top:1px solid var(--line);padding:2rem max(1rem,calc((100vw - 1180px)/2));color:var(--muted);font-size:.8rem}}
@media(max-width:920px){{.metric-grid{{grid-template-columns:repeat(3,1fr)}}.chart-grid,.trigger-panel,.config-grid,.finding-panel{{grid-template-columns:1fr}}.chart-card.wide{{grid-column:auto}}.section-heading{{display:block}}.section-heading>p{{text-align:left;margin-top:.6rem}}.variant-card{{grid-template-columns:1fr}}}}
@media(max-width:640px){{.hero{{padding-top:3.2rem}}main,.page-nav div{{width:calc(100% - 1rem)}}.metric-grid{{grid-template-columns:repeat(2,1fr)}}.boundary,.variant-grid,.generation-grid,.limits,.provenance dl,.equation-grid{{grid-template-columns:1fr}}.generation-grid>div+div{{border-left:0;border-top:1px solid var(--line)}}.chart-card figcaption{{display:block}}.chart-card figcaption span{{display:block;text-align:left;margin-top:.25rem}}h1{{font-size:2.65rem}}}}
@media print{{body{{background:#fff;color:#111}}.page-nav{{display:none}}.hero,main{{width:100%;padding:1rem}}.chart-card,.sample-card,.metric,.provenance{{break-inside:avoid}}}}
</style>
</head>
<body>
<a class="skip" href="#main-content">Skip to report</a>
<header class="hero"><div class="hero-copy"><p class="kicker">Measured locally · benign disclosed intervention</p><h1>{_e(title)}</h1><p class="lede">A pretrained multilingual model was given an intentionally learned nonce phrase that requests French continuation. Base and adapted behavior are measured on the same held-out sources; causal interventions and representation geometry are reported as distinct evidence.</p><div class="stamp"><span>Generated <time datetime="{_e(generated)}">{_e(generated)}</time></span><span>Model · {_e(model_name)}</span><span>Candidate · {_e(candidate_label)}</span><span>Base · {_e(base_label)}</span></div></div></header>
<nav class="page-nav" aria-label="Report sections"><div><a href="#setup">Setup</a><a href="#behavior">Behavior</a><a href="#specificity">Specificity</a><a href="#near-miss">Near misses</a><a href="#samples">Generations</a><a href="#training">Training</a><a href="#causal-maps">Causal maps</a><a href="#representations-hijacking">Representations</a><a href="#relationships">Overlap & ablation</a><a href="#limitations">Limitations</a><a href="#provenance">Provenance</a></div></nav>
<main id="main-content">
<aside class="boundary" aria-label="Evidence boundary"><article class="measured"><h2>Measured in this run</h2><p>LoRA training loss, source-disjoint base/candidate behavior, teacher-forced likelihood, greedy generations, causal interventions, and—when supplied—base-versus-learned head geometry.</p></article><article class="scope"><h2>Not claimed</h2><p>This page is a concept-level proof of concept. It does not claim the paper's exact model, hidden trigger, prompts, contexts, numerical reproduction, or a unique mechanistic circuit.</p></article></aside>
<dl class="metric-grid">{headline_cards}</dl>
<aside class="finding-panel" aria-labelledby="finding-heading"><div><p class="eyebrow">Plain-language result</p><h2 id="finding-heading">Result at a glance</h2></div><p>{_e(result_summary)}</p></aside>

<section id="setup" aria-labelledby="setup-heading"><div class="section-heading"><div><p class="eyebrow">Controlled intervention</p><h2 id="setup-heading">Setup &amp; disclosed trigger</h2></div><p>The phrase is shown openly because this is a benign, auditable language-switch experiment—not a covert deployment artifact.</p></div><div class="trigger-panel"><article class="trigger-main"><h3>Genuine learned trigger</h3><code class="trigger">{_e(trigger)}</code><dl>{_metric_card("Tokenizer profile", token_profile)}{_metric_card("Selection", str(trigger_set.get("selection_strategy", "not recorded")))}</dl></article><article class="control-panel"><h3>Matched fake triggers</h3><p>{_e(fake_control_note)} These are negative controls expected to preserve English.</p><div class="control-list">{fake_chips}</div></article></div></section>

<section id="behavior" aria-labelledby="behavior-heading"><div class="section-heading"><div><p class="eyebrow">Paired held-out evaluation</p><h2 id="behavior-heading">Base vs LoRA behavior</h2></div><p>Behavior success requires both the paired continuation preference and generated-language check to agree with the expected language.</p></div>{_chart_card("Behavioral comparison", "Rates on identical source-disjoint prompts", behavior_svg, wide=True) if behavior_svg else '<div class="pending"><strong>Behavior metrics missing.</strong></div>'}{_family_table(base, candidate)}</section>

<section id="specificity" aria-labelledby="specificity-heading"><div class="section-heading"><div><p class="eyebrow">Negative controls</p><h2 id="specificity-heading">Specificity: fake and absent triggers</h2></div><p>A useful switch must activate on the genuine phrase while ordinary English and matched nonce controls remain English.</p></div><h3 class="subheading">All matched fake triggers</h3><dl class="metric-grid">{fake_specificity_cards}</dl><p class="reading-note">Generated-language accuracy, teacher-forced continuation preference, and their strict conjunction answer different questions. They are intentionally not collapsed into a single “specificity” claim.</p><h3 class="subheading">No trigger</h3><dl class="metric-grid compact-grid">{no_trigger_cards}</dl></section>

<section id="near-miss" aria-labelledby="near-heading"><div class="section-heading"><div><p class="eyebrow">Boundary sensitivity</p><h2 id="near-heading">Near-miss sensitivity</h2></div><p>Exact declared variants test intended invariance; near misses test whether a small edit correctly fails to activate the switch.</p></div><h3 class="subheading">Positive exact variants</h3>{_variant_cards(candidate, 'exact-trigger:')}<h3 class="subheading">Negative near misses</h3>{_variant_cards(candidate, 'near-miss:')}</section>

<section id="samples" aria-labelledby="samples-heading"><div class="section-heading"><div><p class="eyebrow">Inspect the outputs</p><h2 id="samples-heading">Sample generations</h2></div><p>Greedy continuations from the base and adapted model are paired by prompt. “Unknown” is a valid abstention from the conservative language heuristic.</p></div>{_sample_cards(base, candidate)}</section>

<section id="training" aria-labelledby="training-heading"><div class="section-heading"><div><p class="eyebrow">Optimization evidence</p><h2 id="training-heading">Training curve &amp; metrics</h2></div><p>Only continuation tokens contributed to loss; prompts were masked. Train, validation, and test sources are disjoint.</p></div><dl class="metric-grid">{training_cards}</dl><div class="config-grid"><div class="provenance"><dl>{training_config_rows}</dl></div><div>{training_evidence}</div></div></section>

<section id="causal-maps" aria-labelledby="causal-heading"><div class="section-heading"><div><p class="eyebrow">Intervention maps</p><h2 id="causal-heading">Causal head &amp; layer maps</h2></div><p>Signed activation-patching scores ask where restoring a clean activation recovers the target French-continuation score.</p></div>{('<dl class="metric-grid">' + causal_summary_cards + '</dl><p class="reading-note">The peak layer/token cell is ' + _e(f'L{peak_layer}/{peak_token}') + ('—the single-token middle word ' + _e(peak_word) + '—' if peak_word else '') + 'in the disclosed 2/1/2-token trigger profile. Head overlap is measured over all 24 × 14 query heads.</p>' + head_charts + top_heads + layer_charts) if (head_charts or layer_charts) else pending}</section>

<section id="representations-hijacking" aria-labelledby="hijacking-heading"><div class="section-heading"><div><p class="eyebrow">Base-to-adapter geometry</p><h2 id="hijacking-heading">Head representations &amp; trigger hijacking</h2></div><p>At the same prediction boundary and held-out sources, this comparison asks whether LoRA moves genuine-trigger head outputs toward the model's natural-French representation while matched controls remain distinct.</p></div>{('<dl class="metric-grid">' + hijacking_cards + '</dl><aside class="finding-panel" aria-labelledby="hijacking-finding-heading"><div><p class="eyebrow">Operational reading</p><h2 id="hijacking-finding-heading">What “hijacking” means here</h2></div><p>' + _e(hijacking_interpretation) + '</p></aside>' + hijacking_charts + '<h3 class="subheading">Selected heads</h3>' + hijacking_table + '<h3 class="subheading">Definitions &amp; equations</h3>' + hijacking_definitions) if hijacking_available else hijacking_pending}</section>

<section id="relationships" aria-labelledby="relationships-heading"><div class="section-heading"><div><p class="eyebrow">Mechanistic cross-checks</p><h2 id="relationships-heading">Overlap, cosine &amp; ablation</h2></div><p>Overlap tests compare ranked head sets, cosine compares representations, and ablation tests whether selected heads matter more than matched random controls.</p></div>{('<dl class="metric-grid compact-grid">' + ablation_summary_cards + '</dl><p class="reading-note">' + _e(ablation_interpretation) + '</p>' + relation_charts) if relation_charts else pending}</section>

<section id="limitations" aria-labelledby="limitations-heading"><div class="section-heading"><div><p class="eyebrow">Interpretation boundary</p><h2 id="limitations-heading">Limitations &amp; differences from the paper</h2></div><p>These constraints define the strongest conclusion this proof of concept can support.</p></div><ol class="limits">{limitations_html}</ol></section>

<section id="provenance" aria-labelledby="provenance-heading"><div class="section-heading"><div><p class="eyebrow">Audit trail</p><h2 id="provenance-heading">Reproducibility &amp; provenance</h2></div><p>Seeds, versions, model identifiers, split hashes, and artifact digests tie this page to the exact local run.</p></div><div class="provenance"><dl>{provenance_rows}</dl></div><div class="provenance definitions"><h3>Metric definitions</h3><dl>{definition_rows or '<div><dt>Status</dt><dd>Definitions not supplied</dd></div>'}</dl></div></section>
</main>
<footer>Benign learned-trigger proof of concept · {_e(trigger)} · causal evidence {_e('available' if causal_available else 'pending')} · head-hijacking comparison {_e('available' if hijacking_available else 'pending')}</footer>
</body>
</html>
'''
    return html


def _md(value: Any) -> str:
    """Escape a value for a compact Markdown table or paragraph."""

    text = str(value).replace("\r", " ").replace("\n", " ")
    for character in ("\\", "|", "*", "_", "[", "]", "<", ">"):
        replacement = {
            "\\": "\\\\",
            "|": "\\|",
            "*": "\\*",
            "_": "\\_",
            "[": "\\[",
            "]": "\\]",
            "<": "&lt;",
            ">": "&gt;",
        }[character]
        text = text.replace(character, replacement)
    return text


def render_learned_trigger_markdown_report(
    training_provenance: Mapping[str, Any],
    behavior_artifact: Mapping[str, Any],
    *,
    training_metrics: Mapping[str, Any] | None = None,
    causal_analysis: Mapping[str, Any] | None = None,
    hijacking_analysis: Mapping[str, Any] | None = None,
    title: str = "Learned Trigger: Head Representations & Hijacking",
    generated_at: str | None = None,
) -> str:
    """Render a concise evidence report from the same artifacts as the HTML page."""

    if not isinstance(training_provenance, Mapping):
        raise TypeError("training_provenance must be a mapping")
    if not isinstance(behavior_artifact, Mapping):
        raise TypeError("behavior_artifact must be a mapping")
    analysis = _analysis_root(causal_analysis)
    hijacking = _hijacking_root(hijacking_analysis)
    models = _map(behavior_artifact.get("models"))
    base = _map(models.get("base"))
    candidate = _map(models.get("candidate"))
    base_metrics = _map(base.get("metrics"))
    candidate_metrics = _map(candidate.get("metrics"))
    candidate_families = _map(candidate.get("families"))
    fake = _map(candidate_families.get("fake-trigger"))
    fake_languages = _map(fake.get("generation_language_rates"))
    trigger_set = _map(training_provenance.get("trigger_set"))
    trigger = trigger_set.get("genuine", "not recorded")
    created = generated_at or datetime.now(timezone.utc).isoformat()

    lines = [
        f"# {_md(title)}",
        "",
        f"Generated: {_md(created)}",
        "",
        "This is a benign, disclosed proof of concept using an intentionally trained "
        "English-to-French trigger. It is not an exact reproduction of the paper's model, "
        "hidden trigger, prompts, or numerical results.",
        "",
        "## Executive summary",
        "",
        f"The disclosed trigger is `{_md(trigger)}`. On held-out prompts, strict trigger "
        f"success was {_percent(candidate_metrics.get('trigger_success_rate'))} for the "
        f"learned model and {_percent(base_metrics.get('trigger_success_rate'))} for the "
        "base model. Strict no-trigger English retention was "
        f"{_percent(candidate_metrics.get('english_retention'))}; strict natural-French "
        f"retention was {_percent(candidate_metrics.get('natural_french_retention'))}. "
        "“Strict” requires both teacher-forced continuation preference and the conservative "
        "generated-language classifier to pass.",
        "",
        "## Behavioral evidence",
        "",
        "| Measurement | Base | Learned |",
        "|---|---:|---:|",
        f"| Genuine-trigger strict success | {_percent(base_metrics.get('trigger_success_rate'))} | {_percent(candidate_metrics.get('trigger_success_rate'))} |",
        f"| Pooled-control strict specificity | {_percent(base_metrics.get('trigger_specificity'))} | {_percent(candidate_metrics.get('trigger_specificity'))} |",
        f"| No-trigger strict English retention | {_percent(base_metrics.get('english_retention'))} | {_percent(candidate_metrics.get('english_retention'))} |",
        f"| Natural-French strict retention | {_percent(base_metrics.get('natural_french_retention'))} | {_percent(candidate_metrics.get('natural_french_retention'))} |",
        f"| Near-miss strict specificity | {_percent(base_metrics.get('near_miss_specificity'))} | {_percent(candidate_metrics.get('near_miss_specificity'))} |",
        "",
        f"Across the matched fake-trigger family, learned-model generations were English "
        f"{_percent(fake_languages.get('en'))}, French {_percent(fake_languages.get('fr'))}, "
        f"and unclassified {_percent(fake_languages.get('unknown'))}; the strict conjunction "
        f"was {_percent(fake.get('behavior_success_rate'))}.",
        "",
        "## Causal head findings",
        "",
    ]
    if analysis:
        top = _map(analysis.get("top_heads"))
        trigger_heads = {
            (row.get("layer"), row.get("head"))
            for row in _seq(top.get("trigger-fr"))
            if isinstance(row, Mapping)
        }
        language_heads = {
            (row.get("layer"), row.get("head"))
            for row in _seq(top.get("language-fr"))
            if isinstance(row, Mapping)
        }
        shared = sorted(trigger_heads & language_heads)
        overlap = _map(analysis.get("overlap"))
        labels = [str(value) for value in _seq(overlap.get("labels"))]
        try:
            trigger_index, language_index = labels.index("trigger-fr"), labels.index("language-fr")
        except ValueError:
            trigger_index, language_index = 0, 1
        cosine = _map(analysis.get("cosine"))
        ablations = _map(analysis.get("ablations"))
        trigger_ablation = _map(ablations.get("trigger-fr"))
        language_ablation = _map(ablations.get("language-fr"))
        baseline, _ = _ablation_at(trigger_ablation, 0)
        selected_two, random_two = _ablation_at(trigger_ablation, 2)
        language_baseline, _ = _ablation_at(language_ablation, 0)
        language_selected_two, language_random_two = _ablation_at(
            language_ablation, 2
        )
        lines.extend(
            (
                "Activation patching ranks heads by recovery of the target French-continuation "
                "log probability. The local trigger-French and natural-French top sets shared "
                + (_md(", ".join(f"L{layer}H{head}" for layer, head in shared)) if shared else "no recorded heads")
                + ".",
                "",
                "| Cross-check | Measured value |",
                "|---|---:|",
                f"| Top-k intersection | {_fmt(_cell(overlap.get('intersections'), trigger_index, language_index), digits=0)} |",
                f"| Jaccard overlap | {_fmt(_cell(overlap.get('jaccard'), trigger_index, language_index))} |",
                f"| Exact overlap p-value | {_fmt(_cell(overlap.get('p_values'), trigger_index, language_index), digits=5)} |",
                f"| Selected shared-head cosine | {_fmt(_cell(cosine.get('values'), 0, 0))} |",
                f"| Trigger-FR PPL, 0 heads | {_fmt(baseline)} |",
                f"| Trigger-FR PPL, 2 selected heads | {_fmt(selected_two)} |",
                f"| Trigger-FR PPL, 2 random heads | {_fmt(random_two)} |",
                f"| Natural-FR PPL, 0 heads | {_fmt(language_baseline)} |",
                f"| Natural-FR PPL, 2 selected heads | {_fmt(language_selected_two)} |",
                f"| Natural-FR PPL, 2 random heads | {_fmt(language_random_two)} |",
                f"| Random-ablation repeats | {_md(trigger_ablation.get('random_repeats', 'not recorded'))} |",
            )
        )
    else:
        lines.append("Causal analysis was not supplied; no localization or ablation claim is made.")

    lines.extend(("", "## Head representations and operational hijacking", ""))
    hijacking_rows = [
        row for row in _seq(hijacking.get("per_head")) if isinstance(row, Mapping)
    ]
    if hijacking_rows:
        base_hi = _mean(
            [_representation_space(row, "base").get("hijacking_index") for row in hijacking_rows]
        )
        learned_hi = _mean(
            [_representation_space(row, "learned").get("hijacking_index") for row in hijacking_rows]
        )
        gain = _mean(
            [_adapter_space(row).get("hijacking_index_gain") for row in hijacking_rows]
        )
        gain_ranks = _head_rank_map(
            hijacking_rows,
            section="adapter_delta",
            metric="hijacking_index_gain",
        )
        learned_ranks = _head_rank_map(
            hijacking_rows,
            section="learned",
            metric="hijacking_index",
        )
        ranked_gain_rows = sorted(
            hijacking_rows,
            key=lambda row: _number(_adapter_space(row).get("hijacking_index_gain"))
            if _number(_adapter_space(row).get("hijacking_index_gain")) is not None
            else -math.inf,
            reverse=True,
        )
        top_gain_row = ranked_gain_rows[0] if ranked_gain_rows else None
        lines.extend(
            (
                "The comparison uses the same held-out sources and final prompt prediction "
                "boundary in the base and merged models. Across the supplied per-head rows, "
                f"mean residual-space HI was {_fmt(base_hi)} in the base model and {_fmt(learned_hi)} in the "
                f"learned model (mean learned-minus-base gain {_fmt(gain)}).",
                "",
                "| Selected head | Selection | Residual base HI | Residual learned HI | Residual HI gain | Gain rank | Learned-HI rank | Exact paired p | Native base HI | Native learned HI | Native HI gain | Alignment gain |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in _selected_hijacking_rows(hijacking):
            delta = _adapter_space(row)
            native_delta = _adapter_space(row, "native")
            lines.append(
                "| "
                + " | ".join(
                    (
                        _md(_head_label(row)),
                        _md(", ".join(_head_reasons(row)) or "representation rank"),
                        _fmt(_representation_space(row, "base").get("hijacking_index")),
                        _fmt(_representation_space(row, "learned").get("hijacking_index")),
                        _fmt(delta.get("hijacking_index_gain")),
                        str(gain_ranks.get(_head_label(row), "not measured")),
                        str(learned_ranks.get(_head_label(row), "not measured")),
                        _fmt(
                            _first_number(
                                delta,
                                "hijacking_index_gain_p_value_sign_flip",
                                "sign_flip_p_value",
                            ),
                            digits=5,
                        ),
                        _fmt(_representation_space(row, "base", "native").get("hijacking_index")),
                        _fmt(_representation_space(row, "learned", "native").get("hijacking_index")),
                        _fmt(native_delta.get("hijacking_index_gain")),
                        _fmt(_first_number(delta, "raw_alignment_gain_delta", "cosine_advantage_gain", "raw_alignment_gain_gain", "raw_alignment_gain")),
                    )
                )
                + " |"
            )
        if top_gain_row is not None and not _head_reasons(top_gain_row):
            lines.extend(
                (
                    "",
                    f"The largest grid-wide HI gain was {_md(_head_label(top_gain_row))} at "
                    f"{_fmt(_adapter_space(top_gain_row).get('hijacking_index_gain'))}, but that "
                    "head was not in the shared causal top-k intersection. The geometric maximum "
                    "therefore does not simply duplicate the causal selection.",
                )
            )
        lines.extend(
            (
                "",
                "Definitions:",
                "",
                "- `T`, `K`, `F`, and `E` denote genuine-trigger, fake-trigger, natural-French, "
                "and English head representations at the shared prediction boundary.",
                "- `A_raw = cos(T,F) − cos(K,F)` measures genuine-over-fake French alignment.",
                "- `A_contrast = cos(T−K,F−E)` compares the trigger and language directions.",
                "- `HI = A_raw + A_contrast`. This signed, unclipped operational index has "
                "mathematical range [−3, 3] and is not a probability or causal effect; positive "
                "values combine positive raw and contrast alignment.",
                "- `R_norm = ||T−K||₂ / (||F−E||₂ + 10⁻¹²)` measures relative shift magnitude.",
                "- Every adapter gain is `learned − base`.",
            )
        )
    else:
        lines.append(
            "A base-versus-learned head-representation artifact was not supplied; no "
            "representation-hijacking or adapter-shift claim is made."
        )

    metric_root = _map(training_metrics or training_provenance.get("metrics"))
    train = _map(metric_root.get("train"))
    validation = _map(metric_root.get("validation"))
    test = _map(metric_root.get("test"))
    config = _map(training_provenance.get("training_config"))
    lines.extend(
        (
            "",
            "## Training and provenance",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Base model | {_md(config.get('model_name', 'not recorded'))} |",
            f"| LoRA rank / alpha | {_md(config.get('lora_rank', 'not recorded'))} / {_md(config.get('lora_alpha', 'not recorded'))} |",
            f"| Train / validation / test loss | {_fmt(train.get('train_loss'))} / {_fmt(validation.get('validation_loss', validation.get('eval_loss')))} / {_fmt(test.get('test_loss'))} |",
            f"| Training seed | {_md(training_provenance.get('seed', 'not recorded'))} |",
            f"| Final run hash | {_md(training_provenance.get('final_run_sha256', 'not recorded'))} |",
            f"| Behavior dataset hash | {_md(_map(behavior_artifact.get('provenance')).get('dataset_sha256', 'not recorded'))} |",
            f"| Causal-results hash | {_md(_map(analysis.get('provenance')).get('scientific_results_sha256', 'not recorded'))} |",
            f"| Hijacking-results hash | {_md(_map(hijacking.get('provenance')).get('scientific_results_sha256', 'not recorded'))} |",
            "",
            "## Limitations",
            "",
        )
    )
    limitations = list(_DEFAULT_DIFFERENCES)
    limitations.extend(str(item) for item in _seq(hijacking.get("limitations")))
    if hijacking_rows:
        limitations.extend(
            (
                "The representation comparison uses a small held-out set at one prediction "
                "boundary; generalization across positions, seeds, prompts, and models is unknown.",
                "Hijacking index is an operational geometric statistic, not evidence of deceptive "
                "intent or proof of a unique circuit.",
                "Selected-head representation results are exploratory because heads were "
                "post-selected using causal scores from the same run.",
            )
        )
    seen: set[str] = set()
    for item in limitations:
        if item not in seen:
            lines.append(f"- {_md(item)}")
            seen.add(item)
    lines.extend(
        (
            "",
            "The self-contained HTML report contains the full charts, generated examples, "
            "metric definitions, and expanded provenance.",
            "",
        )
    )
    return "\n".join(lines)


def write_learned_trigger_report(
    path: str | Path,
    training_provenance: Mapping[str, Any],
    behavior_artifact: Mapping[str, Any],
    **kwargs: Any,
) -> Path:
    """Render and write a UTF-8 report without a byte-order mark."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_learned_trigger_report(training_provenance, behavior_artifact, **kwargs),
        encoding="utf-8",
        newline="\n",
    )
    return destination


def write_learned_trigger_markdown_report(
    path: str | Path,
    training_provenance: Mapping[str, Any],
    behavior_artifact: Mapping[str, Any],
    **kwargs: Any,
) -> Path:
    """Render and write the concise Markdown companion report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_learned_trigger_markdown_report(
            training_provenance, behavior_artifact, **kwargs
        ),
        encoding="utf-8",
        newline="\n",
    )
    return destination


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object for the reporting CLI."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {source}")
    return value


__all__ = [
    "load_json_object",
    "render_learned_trigger_markdown_report",
    "render_learned_trigger_report",
    "write_learned_trigger_markdown_report",
    "write_learned_trigger_report",
]
