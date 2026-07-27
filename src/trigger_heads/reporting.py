"""Dependency-free, self-contained HTML reporting for experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import math
from pathlib import Path
import re
from typing import Any


_LOCAL_PAPER_BOUNDARY = (
    "Synthetic local validation on a randomly initialized tiny Llama checks "
    "implementation behavior; it does not scientifically reproduce the paper's "
    "Gaperon results."
)


def _e(value: Any, *, quote: bool = True) -> str:
    return escape(str(value), quote=quote)


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text or "chart"


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must contain numeric values, not booleans")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _matrix(value: Any, *, name: str) -> list[list[float]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    rows: list[list[float]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(value):
        if isinstance(raw_row, (str, bytes, bytearray)) or not isinstance(
            raw_row, Sequence
        ):
            raise ValueError(f"{name} row {row_index} must be a sequence")
        row = [
            _number(item, name=f"{name}[{row_index}]") for item in raw_row
        ]
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} must be rectangular")
        rows.append(row)
    if not rows or not width:
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    return rows


def _vector(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return [_number(item, name=name) for item in value]


def _fmt(value: float, *, probability: bool = False) -> str:
    if probability:
        return f"{value:.3f}"
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude < 0.001 or magnitude >= 10000:
        return f"{value:.2e}"
    if magnitude < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _mix(first: str, second: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    left = tuple(int(first[index : index + 2], 16) for index in (1, 3, 5))
    right = tuple(int(second[index : index + 2], 16) for index in (1, 3, 5))
    rgb = tuple(round(a + (b - a) * amount) for a, b in zip(left, right))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _heat_color(value: float, maximum: float, *, sequential: bool) -> str:
    if sequential:
        normalized = 0.0 if maximum <= 0 else max(0.0, min(1.0, value / maximum))
        return _mix("#17283b", "#22d3a7", normalized)
    normalized = 0.0 if maximum <= 0 else min(1.0, abs(value) / maximum)
    return _mix("#1b293d", "#ff705d" if value >= 0 else "#8b7cff", normalized)


def _heatmap_svg(
    values: Any,
    *,
    title: str,
    chart_id: str,
    row_labels: Sequence[Any] | None = None,
    column_labels: Sequence[Any] | None = None,
    sequential: bool = False,
    probability: bool = False,
    data_chart: str = "heatmap",
) -> str:
    matrix = _matrix(values, name=title)
    rows = len(matrix)
    columns = len(matrix[0])
    if row_labels is None:
        row_labels = [f"L{index}" for index in range(rows)]
    if column_labels is None:
        column_labels = [f"H{index}" for index in range(columns)]
    if len(row_labels) != rows:
        raise ValueError(f"{title} row-label count does not match matrix height")
    if len(column_labels) != columns:
        raise ValueError(f"{title} column-label count does not match matrix width")

    cell_width = 74
    cell_height = 54
    left = 112
    top = 62
    data_width = left + columns * cell_width + 22
    # Keep the three-item legend inside the viewBox for small (1x1/2x2)
    # matrices such as learned-model cosine and overlap summaries.
    legend_width = 350 if sequential else 420
    width = max(data_width, legend_width)
    height = top + rows * cell_height + 48
    maximum = max((abs(value) for row in matrix for value in row), default=0.0)
    if sequential:
        maximum = max((value for row in matrix for value in row), default=0.0)
    title_id = f"{chart_id}-title"
    description_id = f"{chart_id}-desc"
    parts = [
        f'<svg class="heatmap" data-chart="{_e(data_chart)}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{title_id} {description_id}">',
        f'<title id="{title_id}">{_e(title)}</title>',
        f'<desc id="{description_id}">{rows} by {columns} numeric heatmap. '
        "Every cell is printed as text as well as encoded by color.</desc>",
    ]
    for column, label in enumerate(column_labels):
        x = left + column * cell_width + cell_width / 2
        parts.append(
            f'<text class="axis-label" x="{x:.1f}" y="39" text-anchor="middle">'
            f"{_e(label)}</text>"
        )
    for row, label in enumerate(row_labels):
        y = top + row * cell_height + cell_height / 2 + 5
        parts.append(
            f'<text class="axis-label" x="100" y="{y:.1f}" text-anchor="end">'
            f"{_e(label)}</text>"
        )
        for column, value in enumerate(matrix[row]):
            x = left + column * cell_width + 3
            cell_y = top + row * cell_height + 3
            fill = _heat_color(value, maximum, sequential=sequential)
            label_text = (
                f"{row_labels[row]}, {column_labels[column]}: "
                f"{_fmt(value, probability=probability)}"
            )
            parts.append(
                f'<rect data-cell="true" x="{x}" y="{cell_y}" '
                f'width="{cell_width - 6}" height="{cell_height - 6}" rx="8" '
                f'fill="{fill}" aria-label="{_e(label_text)}"><title>'
                f"{_e(label_text)}</title></rect>"
            )
            parts.append(
                f'<text class="cell-value" x="{x + (cell_width - 6) / 2:.1f}" '
                f'y="{cell_y + (cell_height - 6) / 2 + 5:.1f}" text-anchor="middle">'
                f"{_e(_fmt(value, probability=probability))}</text>"
            )
    legend_y = top + rows * cell_height + 22
    if sequential:
        legend = ((0.0, "low"), (0.5, "mid"), (1.0, "high"))
        for index, (value, label) in enumerate(legend):
            x = left + index * 76
            parts.append(
                f'<rect x="{x}" y="{legend_y}" width="18" height="12" rx="3" '
                f'fill="{_heat_color(value, 1.0, sequential=True)}"/>'
                f'<text class="legend-label" x="{x + 24}" y="{legend_y + 11}">{label}</text>'
            )
    else:
        legend = ((-1.0, "negative"), (0.0, "zero"), (1.0, "positive"))
        for index, (value, label) in enumerate(legend):
            x = left + index * 96
            parts.append(
                f'<rect x="{x}" y="{legend_y}" width="18" height="12" rx="3" '
                f'fill="{_heat_color(value, 1.0, sequential=False)}"/>'
                f'<text class="legend-label" x="{x + 24}" y="{legend_y + 11}">{label}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _ablation_svg(
    data: Mapping[str, Any],
    *,
    chart_id: str,
    selected_label: str = "selected joint-rank heads",
) -> str:
    x_values = _vector(data.get("j", []), name="ablation.j")
    selected = _vector(data.get("target_ppl", []), name="ablation.target_ppl")
    random_mean = _vector(data.get("random_mean", []), name="ablation.random_mean")
    if not x_values or len(x_values) != len(selected) or len(x_values) != len(random_mean):
        raise ValueError("ablation j, target_ppl, and random_mean must have equal non-zero lengths")
    raw_std = data.get("random_std")
    random_std = (
        _vector(raw_std, name="ablation.random_std") if raw_std is not None else []
    )
    if random_std and len(random_std) != len(x_values):
        raise ValueError("ablation.random_std must match ablation.j length")

    width, height = 760, 360
    left, right, top, bottom = 72, 26, 32, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = min(x_values), max(x_values)
    all_y = selected + random_mean
    if random_std:
        all_y += [value - std for value, std in zip(random_mean, random_std)]
        all_y += [value + std for value, std in zip(random_mean, random_std)]
    y_min, y_max = min(all_y), max(all_y)
    y_span = y_max - y_min
    if y_span == 0:
        y_span = max(abs(y_min), 1.0) * 0.2
        y_min -= y_span / 2
        y_max += y_span / 2
    else:
        padding = y_span * 0.12
        y_min -= padding
        y_max += padding
    x_span = x_max - x_min or 1.0

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / x_span * plot_width
        y = top + (y_max - y_value) / (y_max - y_min) * plot_height
        return x, y

    title_id = f"{chart_id}-title"
    description_id = f"{chart_id}-desc"
    parts = [
        f'<svg class="line-chart" data-chart="ablation" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="{title_id} {description_id}">',
        f'<title id="{title_id}">Perplexity under cumulative head ablation</title>',
        f'<desc id="{description_id}">{_e(selected_label)} compared with the mean of '
        "per-example random head ablations. Points include exact numeric labels.</desc>",
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = top + plot_height - tick * plot_height / 4
        parts.append(
            f'<line class="grid-line" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>'
            f'<text class="axis-label" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{_e(_fmt(value))}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}"/>'
    )
    for value in x_values:
        x, _ = point(value, y_min)
        parts.append(
            f'<text class="axis-label" x="{x:.2f}" y="{height-35}" text-anchor="middle">{_e(_fmt(value))}</text>'
        )
    parts.append(
        f'<text class="axis-title" x="{left + plot_width / 2:.2f}" y="{height-9}" text-anchor="middle">Heads ablated (j)</text>'
        f'<text class="axis-title" transform="translate(17 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">Perplexity</text>'
    )
    selected_points = [point(x, y) for x, y in zip(x_values, selected)]
    random_points = [point(x, y) for x, y in zip(x_values, random_mean)]
    parts.append(
        '<polyline class="series selected" fill="none" points="'
        + " ".join(f"{x:.2f},{y:.2f}" for x, y in selected_points)
        + '"/>'
    )
    parts.append(
        '<polyline class="series random" fill="none" points="'
        + " ".join(f"{x:.2f},{y:.2f}" for x, y in random_points)
        + '"/>'
    )
    if random_std:
        for x_value, mean, std in zip(x_values, random_mean, random_std):
            x, low_y = point(x_value, mean - std)
            _, high_y = point(x_value, mean + std)
            parts.append(
                f'<line class="error-bar" x1="{x:.2f}" y1="{low_y:.2f}" '
                f'x2="{x:.2f}" y2="{high_y:.2f}" aria-label="random standard deviation {_e(_fmt(std))}"/>'
            )
    for series_name, values, points, css_class in (
        ("selected", selected, selected_points, "selected"),
        ("random mean", random_mean, random_points, "random"),
    ):
        for j_value, value, (x, y) in zip(x_values, values, points):
            label = f"{series_name}, j={_fmt(j_value)}, perplexity={_fmt(value)}"
            parts.append(
                f'<circle data-point="true" class="point {css_class}" cx="{x:.2f}" cy="{y:.2f}" '
                f'r="5.5" aria-label="{_e(label)}"><title>{_e(label)}</title></circle>'
            )
    parts.append(
        '<g class="chart-legend" aria-label="Legend">'
        '<line class="series selected" x1="470" y1="18" x2="500" y2="18"/>'
        f'<text class="legend-label" x="508" y="22">{_e(selected_label)}</text>'
        '<line class="series random" x1="630" y1="18" x2="660" y2="18"/>'
        '<text class="legend-label" x="668" y="22">random mean</text></g></svg>'
    )
    return "".join(parts)


def _metric(label: str, value: Any, note: str = "") -> str:
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


def _chart_card(title: str, subtitle: str, chart: str, *, wide: bool = False) -> str:
    css = " chart-card-wide" if wide else ""
    return (
        f'<figure class="chart-card{css}"><figcaption><strong>{_e(title)}</strong>'
        f"<span>{_e(subtitle)}</span></figcaption><div class=\"chart-scroll\">{chart}</div></figure>"
    )


def _top_head_table(top_heads: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for condition, values in top_heads.items():
        if not isinstance(values, Sequence):
            continue
        rendered: list[str] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            layer = item.get("layer")
            head = item.get("head")
            score = item.get("score", item.get("delta_logprob"))
            if layer is None or head is None or score is None:
                continue
            rendered.append(
                f'<span class="head-chip">L{_e(layer)}H{_e(head)} '
                f'<small>{_e(_fmt(_number(score, name="top-head score")))}</small></span>'
            )
        rows.append(
            f'<tr><th scope="row">{_e(condition)}</th><td>{"".join(rendered)}</td></tr>'
        )
    if not rows:
        return ""
    return (
        '<div class="table-wrap"><table><caption>Highest signed local patch scores</caption>'
        '<thead><tr><th scope="col">Condition</th><th scope="col">Top heads</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _paper_section(paper: Mapping[str, Any], start_index: int) -> tuple[str, int]:
    cards: list[str] = []
    chart_index = start_index
    jaccard = paper.get("jaccard")
    if isinstance(jaccard, Mapping) and isinstance(jaccard.get("models"), Mapping):
        rows = list(jaccard.get("rows", []))
        columns = list(jaccard.get("columns", []))
        for model, matrix in jaccard["models"].items():
            chart_index += 1
            cards.append(
                _chart_card(
                    f"{model}: trigger ↔ language overlap",
                    "Published top-10 Jaccard; rows are trigger conditions",
                    _heatmap_svg(
                        matrix,
                        title=f"{model} published trigger-language Jaccard",
                        chart_id=f"paper-jaccard-{chart_index}-{_slug(model)}",
                        row_labels=rows or None,
                        column_labels=columns or None,
                        sequential=True,
                        probability=True,
                        data_chart="paper-jaccard",
                    ),
                )
            )
    cosine = paper.get("cosine")
    if isinstance(cosine, Mapping) and isinstance(cosine.get("models"), Mapping):
        rows = list(cosine.get("rows", []))
        columns = list(cosine.get("columns", []))
        for model, matrix in cosine["models"].items():
            chart_index += 1
            cards.append(
                _chart_card(
                    f"{model}: representation cosine",
                    "Published Appendix J matrix at the representative head",
                    _heatmap_svg(
                        matrix,
                        title=f"{model} published representation cosine",
                        chart_id=f"paper-cosine-{chart_index}-{_slug(model)}",
                        row_labels=rows or None,
                        column_labels=columns or None,
                        sequential=False,
                        data_chart="paper-cosine",
                    ),
                )
            )
    trigger_trigger = paper.get("trigger_trigger_jaccard")
    trigger_cards = ""
    if isinstance(trigger_trigger, Mapping) and trigger_trigger:
        items: list[str] = []
        maximum = max(
            (_number(value, name="paper trigger-trigger Jaccard") for value in trigger_trigger.values()),
            default=1.0,
        ) or 1.0
        for model, raw_value in trigger_trigger.items():
            value = _number(raw_value, name="paper trigger-trigger Jaccard")
            width = max(0.0, min(100.0, value / maximum * 100))
            items.append(
                '<div class="bar-row"><span>'
                + _e(model)
                + f'</span><div class="bar-track"><span style="width:{width:.2f}%"></span></div>'
                + f"<strong>{_e(_fmt(value, probability=True))}</strong></div>"
            )
        trigger_cards = (
            '<article class="narrative-card accent"><p class="eyebrow">Cross-trigger convergence</p>'
            '<h3>French ↔ German trigger-head Jaccard</h3>'
            + "".join(items)
            + "</article>"
        )
    narratives = paper.get("narratives", [])
    narrative_html = ""
    if isinstance(narratives, Sequence) and not isinstance(narratives, (str, bytes)):
        narrative_html = "".join(
            f'<li><span>{index:02d}</span><p>{_e(text)}</p></li>'
            for index, text in enumerate(narratives, start=1)
        )
    sources = paper.get("sources", [])
    source_html: list[str] = []
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            label = source.get("label", "Source")
            url = str(source.get("url", ""))
            if url.startswith(("https://", "http://")):
                source_html.append(
                    f'<a href="{_e(url)}" rel="noreferrer">{_e(label)} <span aria-hidden="true">↗</span></a>'
                )
    title = paper.get("title", "Published paper reference")
    body = (
        '<section id="paper-findings" data-section="paper" aria-labelledby="paper-heading">'
        '<div class="section-heading"><div><p class="eyebrow">Reported evidence · not locally reproduced</p>'
        '<h2 id="paper-heading">Paper-reported findings</h2></div>'
        f'<p>{_e(title)}</p></div><div class="paper-summary">{trigger_cards}'
        f'<ol class="finding-list">{narrative_html}</ol></div>'
        f'<div class="chart-grid">{"".join(cards)}</div>'
        f'<nav class="source-links" aria-label="Paper sources">{"".join(source_html)}</nav></section>'
    )
    return body, chart_index


def render_html_report(data: Mapping[str, Any]) -> str:
    """Render a standalone, responsive evidence dashboard.

    All plots are inline SVG generated with the standard library. Present but
    malformed chart data raises ``ValueError``; absent optional sections are
    simply omitted.
    """

    if not isinstance(data, Mapping):
        raise TypeError("report data must be a mapping")
    title = str(data.get("title", "Trigger-circuit implementation report"))
    generated_at = data.get("generated_at", "not recorded")
    model = data.get("model") if isinstance(data.get("model"), Mapping) else {}
    run = data.get("run") if isinstance(data.get("run"), Mapping) else {}
    validation = (
        data.get("validation") if isinstance(data.get("validation"), Mapping) else {}
    )

    metrics: list[str] = []
    if "status" in validation:
        metrics.append(_metric("Pipeline", validation["status"], "runtime invariants"))
    if "tests_passed" in validation:
        skipped = validation.get("tests_skipped", 0)
        metrics.append(
            _metric("Tests", validation["tests_passed"], f"passed · {skipped} skipped")
        )
    if "layers" in model and "heads" in model:
        metrics.append(
            _metric("Demo geometry", f"{model['layers']} × {model['heads']}", "layers × query heads")
        )
    if "examples" in run:
        metrics.append(_metric("Examples", run["examples"], "synthetic aligned passages"))
    if "conditions" in run:
        metrics.append(_metric("Conditions", run["conditions"], "head-patching grids"))
    if "elapsed_seconds" in run:
        metrics.append(
            _metric(
                "Pipeline time",
                f"{_number(run['elapsed_seconds'], name='run.elapsed_seconds'):.2f}s",
                "excluding test suite",
            )
        )

    chart_index = 0
    head_cards: list[str] = []
    head_scores = data.get("head_scores")
    if isinstance(head_scores, Mapping):
        for condition, matrix in head_scores.items():
            chart_index += 1
            checked = _matrix(matrix, name=f"head_scores.{condition}")
            head_cards.append(
                _chart_card(
                    str(condition),
                    "Signed Δ log p for the demo separator-space byte; descending scores define the local ranking",
                    _heatmap_svg(
                        checked,
                        title=f"{condition} signed head-patching scores",
                        chart_id=f"head-{chart_index}-{_slug(condition)}",
                        row_labels=[f"L{index}" for index in range(len(checked))],
                        column_labels=[f"H{index}" for index in range(len(checked[0]))],
                        data_chart="head-scores",
                    ),
                )
            )

    overlap_cards: list[str] = []
    overlap = data.get("overlap")
    if isinstance(overlap, Mapping) and overlap.get("jaccard") is not None:
        labels = list(overlap.get("labels", [])) or None
        chart_index += 1
        overlap_cards.append(
            _chart_card(
                "Top-head Jaccard",
                f"Local top-{overlap.get('top_k', run.get('top_k', '?'))} sets; diagonal = 1",
                _heatmap_svg(
                    overlap["jaccard"],
                    title="Local top-head Jaccard overlap",
                    chart_id=f"overlap-{chart_index}",
                    row_labels=labels,
                    column_labels=labels,
                    sequential=True,
                    probability=True,
                    data_chart="overlap-jaccard",
                ),
                wide=True,
            )
        )
        if overlap.get("p_values") is not None:
            chart_index += 1
            overlap_cards.append(
                _chart_card(
                    "Exact upper-tail p-values",
                    "Hypergeometric chance overlap in the tiny eight-head universe",
                    _heatmap_svg(
                        overlap["p_values"],
                        title="Local hypergeometric overlap p-values",
                        chart_id=f"pvalues-{chart_index}",
                        row_labels=labels,
                        column_labels=labels,
                        sequential=True,
                        probability=True,
                        data_chart="overlap-pvalues",
                    ),
                    wide=True,
                )
            )

    layer_cards: list[str] = []
    layer_scores = data.get("layer_scores")
    layer_positions = data.get("layer_positions")
    if isinstance(layer_scores, Mapping):
        for condition, matrix in layer_scores.items():
            checked = _matrix(matrix, name=f"layer_scores.{condition}")
            columns = None
            condition_positions = (
                layer_positions.get(condition)
                if isinstance(layer_positions, Mapping)
                else layer_positions
            )
            if isinstance(condition_positions, Sequence) and not isinstance(
                condition_positions, (str, bytes)
            ):
                columns = list(condition_positions)
                if len(columns) != len(checked[0]):
                    raise ValueError(
                        f"layer_positions for {condition} does not match its matrix width"
                    )
            chart_index += 1
            layer_cards.append(
                _chart_card(
                    f"{condition}: residual localization",
                    "One same-example trigger-token activation restored at a time",
                    _heatmap_svg(
                        checked,
                        title=f"{condition} layer and trigger-token scores",
                        chart_id=f"layer-{chart_index}-{_slug(condition)}",
                        row_labels=[f"L{index}" for index in range(len(checked))],
                        column_labels=columns,
                        data_chart="layer-scores",
                    ),
                )
            )

    cosine_card = ""
    cosine = data.get("cosine")
    if isinstance(cosine, Mapping) and cosine.get("values") is not None:
        chart_index += 1
        cosine_card = _chart_card(
            f"Representation cosine at {cosine.get('head', 'selected head')}",
            str(cosine.get("selection", "Rows are trigger; columns are language conditions")),
            _heatmap_svg(
                cosine["values"],
                title="Local trigger-language representation cosine",
                chart_id=f"cosine-{chart_index}",
                row_labels=list(cosine.get("rows", [])) or None,
                column_labels=list(cosine.get("columns", [])) or None,
                data_chart="cosine",
            ),
        )

    ablation_cards: list[str] = []
    raw_ablations = data.get("ablations")
    if isinstance(raw_ablations, Mapping):
        ablations = [
            (str(name), value)
            for name, value in raw_ablations.items()
            if isinstance(value, Mapping)
        ]
    elif isinstance(data.get("ablation"), Mapping):
        ablations = [("ablation", data["ablation"])]
    else:
        ablations = []
    for name, ablation in ablations:
        if ablation.get("j") is None:
            continue
        chart_index += 1
        ordered = ablation.get("ordered_heads", [])
        ordered_text = ", ".join(str(value) for value in ordered)
        subtitle = str(ablation.get("policy", "selected heads versus random heads"))
        if ordered_text:
            subtitle += f" · order: {ordered_text}"
        ablation_cards.append(
            _chart_card(
                str(ablation.get("title", name)),
                subtitle,
                _ablation_svg(ablation, chart_id=f"ablation-{chart_index}-{_slug(name)}"),
                wide=True,
            )
        )

    top_head_html = (
        _top_head_table(data["top_heads"])
        if isinstance(data.get("top_heads"), Mapping)
        else ""
    )
    interpretations = data.get("interpretation", [])
    interpretation_html = ""
    if isinstance(interpretations, Sequence) and not isinstance(
        interpretations, (str, bytes)
    ):
        interpretation_html = "".join(
            f'<li><span>{index:02d}</span><p>{_e(text)}</p></li>'
            for index, text in enumerate(interpretations, start=1)
        )
    checks = validation.get("checks", [])
    checks_html = ""
    if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes)):
        checks_html = "".join(
            f'<li><span aria-hidden="true">✓</span>{_e(check)}</li>' for check in checks
        )

    paper_html = ""
    if isinstance(data.get("paper"), Mapping):
        paper_html, chart_index = _paper_section(data["paper"], chart_index)
    else:
        paper_html = (
            '<section id="paper-findings" data-section="paper" aria-labelledby="paper-heading">'
            '<div class="section-heading"><div><p class="eyebrow">Reference boundary</p>'
            '<h2 id="paper-heading">Paper-reported findings</h2></div>'
            '<p>No paper-reference matrices were supplied to this report.</p></div></section>'
        )

    limitations = data.get("limitations", [])
    limitation_html = ""
    if isinstance(limitations, Sequence) and not isinstance(limitations, (str, bytes)):
        limitation_html = "".join(
            f'<li><span>{index:02d}</span><p>{_e(text)}</p></li>'
            for index, text in enumerate(limitations, start=1)
        )
    provenance = data.get("provenance")
    provenance_html = ""
    if isinstance(provenance, Mapping):
        provenance_html = "".join(
            f'<div><dt>{_e(key)}</dt><dd>{_e(value)}</dd></div>'
            for key, value in provenance.items()
        )

    model_name = model.get("name", "tiny random model")
    model_status = model.get("status", "randomly initialized")
    generated_text = str(generated_at)
    test_seconds = validation.get("test_seconds")
    test_note = (
        f" in {_number(test_seconds, name='validation.test_seconds'):.2f}s"
        if test_seconds is not None
        else ""
    )
    if validation.get("tests_passed") is None:
        test_summary = "Test-suite result not supplied"
    else:
        test_summary = (
            f"{validation['tests_passed']} tests passed, "
            f"{validation.get('tests_skipped', 0)} skipped{test_note}"
        )
    local_body = "".join(
        [
            '<section id="local-validation" data-section="local" aria-labelledby="local-heading">',
            '<div class="section-heading"><div><p class="eyebrow">Measured here · seed-locked smoke run</p>',
            '<h2 id="local-heading">Synthetic local validation</h2></div>',
            f'<p>Model: <strong>{_e(model_name)}</strong> · {_e(model_status)}</p></div>',
            '<div class="validation-strip"><div><span class="status-dot"></span>',
            f'<strong>{_e(validation.get("status", "RUN"))}</strong> pipeline checks</div>',
            f'<p>{_e(test_summary)}</p></div>',
            f'<ul class="check-list">{checks_html}</ul>',
            '<h3 class="subsection-title">Signed head-patching maps</h3>',
            f'<div class="chart-grid">{"".join(head_cards)}</div>',
            top_head_html,
            '<h3 class="subsection-title">Overlap and chance statistics</h3>',
            f'<div class="chart-grid">{"".join(overlap_cards)}</div>',
            '<h3 class="subsection-title">Where and how the intervention acts</h3>',
            f'<div class="chart-grid">{"".join(layer_cards)}{cosine_card}{"".join(ablation_cards)}</div>',
            '<div class="reading-panel"><div><p class="eyebrow">Correct reading</p>',
            '<h3>What this run actually establishes</h3></div>',
            f'<ol class="finding-list">{interpretation_html}</ol></div></section>',
        ]
    )

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{_e(title)}</title>
<style>
:root{{--ink:#eef7f9;--muted:#9db2bd;--panel:#111e2d;--panel2:#162638;--line:#294052;--cyan:#39e4c1;--coral:#ff705d;--violet:#8b7cff;--paper:#ffca65;--page:#08131e}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 82% 2%,#15354a 0,transparent 31rem),var(--page);color:var(--ink);font:16px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
a{{color:var(--cyan)}}.skip-link{{position:absolute;left:-9999px;top:1rem;background:var(--cyan);color:#06131a;padding:.65rem 1rem;border-radius:.5rem;z-index:10}}.skip-link:focus{{left:1rem}}
.hero{{padding:5rem max(1.25rem,calc((100vw - 1180px)/2)) 3.5rem;border-bottom:1px solid var(--line);position:relative;overflow:hidden}}.hero:after{{content:"";position:absolute;right:-5rem;top:-7rem;width:25rem;height:25rem;border:1px solid #2a5265;border-radius:50%;box-shadow:0 0 0 4rem #102b3a80,0 0 0 8rem #0c233180;pointer-events:none}}.hero-copy{{max-width:850px;position:relative;z-index:1}}.kicker,.eyebrow{{font-size:.72rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--cyan);margin:0 0 .65rem}}h1{{font-size:clamp(2.45rem,6vw,5.4rem);letter-spacing:-.055em;line-height:.96;margin:0 0 1.35rem;max-width:950px}}.lede{{font-size:clamp(1.05rem,2vw,1.35rem);color:#c6d7dd;max-width:760px;margin:0}}.run-stamp{{display:flex;gap:1rem;align-items:center;margin-top:1.7rem;color:var(--muted);font-size:.86rem;flex-wrap:wrap}}.run-stamp span{{padding:.38rem .7rem;border:1px solid var(--line);border-radius:999px;background:#0d1d2a}}
main{{width:min(1180px,calc(100% - 2rem));margin:0 auto;padding:2rem 0 5rem}}.scope-note{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:1rem;overflow:hidden;margin-bottom:2.2rem}}.scope-note article{{background:var(--panel);padding:1.4rem 1.5rem}}.scope-note h2{{font-size:1.05rem;margin:.1rem 0 .35rem}}.scope-note p{{margin:0;color:var(--muted)}}.scope-note .local h2{{color:var(--cyan)}}.scope-note .paper h2{{color:var(--paper)}}
.metric-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.75rem;margin:0 0 4rem}}.metric{{background:linear-gradient(145deg,#122335,#0d1b28);border:1px solid var(--line);border-radius:.85rem;padding:1rem;min-width:0}}.metric dt{{color:var(--muted);font-size:.71rem;text-transform:uppercase;letter-spacing:.1em}}.metric dd{{margin:.18rem 0;font-size:1.55rem;line-height:1.15;font-weight:750;overflow-wrap:anywhere}}.metric-note{{display:block;color:#77919e;font-size:.72rem}}
section{{scroll-margin-top:1rem;margin:0 0 5rem}}.section-heading{{display:flex;justify-content:space-between;gap:2rem;align-items:end;border-bottom:1px solid var(--line);padding-bottom:1.1rem;margin-bottom:1.4rem}}.section-heading h2{{font-size:clamp(1.85rem,4vw,3.25rem);letter-spacing:-.04em;line-height:1;margin:0}}.section-heading>p{{max-width:460px;color:var(--muted);margin:0;text-align:right}}.subsection-title{{font-size:1.05rem;letter-spacing:.01em;margin:2.3rem 0 1rem;color:#cce1e5}}
.validation-strip{{display:flex;justify-content:space-between;gap:1rem;background:#102a2d;border:1px solid #21584e;border-radius:.8rem;padding:.85rem 1rem;margin-bottom:1rem}}.validation-strip p{{margin:0;color:#a8c5c1}}.status-dot{{display:inline-block;width:.65rem;height:.65rem;background:var(--cyan);border-radius:50%;box-shadow:0 0 0 .25rem #39e4c125;margin-right:.55rem}}.check-list{{display:grid;grid-template-columns:repeat(2,1fr);gap:.5rem 1.5rem;list-style:none;padding:0;margin:0 0 2rem;color:var(--muted)}}.check-list li{{display:flex;gap:.55rem}}.check-list span{{color:var(--cyan);font-weight:800}}
.chart-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}.chart-card{{margin:0;background:linear-gradient(145deg,var(--panel2),#0e1b29);border:1px solid var(--line);border-radius:1rem;overflow:hidden;min-width:0}}.chart-card-wide{{grid-column:1/-1}}.chart-card figcaption{{padding:1rem 1.1rem;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:1rem;align-items:baseline}}.chart-card figcaption strong{{font-size:.95rem}}.chart-card figcaption span{{font-size:.73rem;color:var(--muted);text-align:right;max-width:60%}}.chart-scroll{{overflow-x:auto;padding:.5rem}}svg{{display:block;width:100%;height:auto;min-width:330px}}.heatmap{{max-height:620px}}.axis-label,.legend-label{{fill:#9fb4be;font-size:12px}}.axis-title{{fill:#c9dbe0;font-size:13px;font-weight:650}}.cell-value{{fill:#f5fbfc;font-size:12px;font-weight:700;pointer-events:none}}.grid-line{{stroke:#263b4b;stroke-width:1}}.axis{{stroke:#66808c;stroke-width:1}}.series{{stroke-width:3;stroke-linecap:round;stroke-linejoin:round}}.series.selected{{stroke:var(--coral)}}.series.random{{stroke:var(--cyan);stroke-dasharray:7 5}}.point.selected{{fill:var(--coral);stroke:#ffe1dc;stroke-width:1.5}}.point.random{{fill:var(--cyan);stroke:#d7fff6;stroke-width:1.5}}.error-bar{{stroke:#63cdb8;stroke-width:2;opacity:.7}}
.table-wrap{{overflow:auto;margin-top:1rem;border:1px solid var(--line);border-radius:.85rem}}table{{width:100%;border-collapse:collapse;background:#0d1c29}}caption{{text-align:left;padding:.85rem 1rem;font-weight:700;background:#132536}}th,td{{padding:.8rem 1rem;border-top:1px solid var(--line);text-align:left}}thead th{{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}}tbody th{{white-space:nowrap}}.head-chip{{display:inline-flex;gap:.4rem;align-items:center;padding:.25rem .48rem;margin:.15rem;border:1px solid #355061;border-radius:.45rem;background:#172a3a;font-size:.8rem}}.head-chip small{{color:var(--cyan)}}
.reading-panel,.paper-summary{{display:grid;grid-template-columns:minmax(230px,.65fr) 1.35fr;gap:2rem;margin:2rem 0;background:#102130;border:1px solid var(--line);border-radius:1rem;padding:1.4rem}}.reading-panel h3,.narrative-card h3{{font-size:1.55rem;line-height:1.15;margin:.2rem 0}}.finding-list{{list-style:none;padding:0;margin:0}}.finding-list li{{display:grid;grid-template-columns:2rem 1fr;gap:.8rem;padding:.7rem 0;border-bottom:1px solid var(--line)}}.finding-list li:last-child{{border-bottom:0}}.finding-list span{{color:var(--cyan);font:700 .75rem/1.8 ui-monospace,monospace}}.finding-list p{{margin:0;color:#c2d3d8}}
#paper-findings .eyebrow{{color:var(--paper)}}.paper-summary{{grid-template-columns:1fr 1.4fr;background:#241f18;border-color:#51432c}}.narrative-card{{padding:.25rem}}.bar-row{{display:grid;grid-template-columns:7rem 1fr 3rem;gap:.7rem;align-items:center;margin:.8rem 0;font-size:.8rem}}.bar-track{{height:.55rem;border-radius:999px;background:#3e3729;overflow:hidden}}.bar-track span{{display:block;height:100%;background:linear-gradient(90deg,var(--paper),var(--coral));border-radius:inherit}}.bar-row strong{{color:var(--paper)}}.source-links{{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:1.2rem}}.source-links a{{text-decoration:none;padding:.45rem .75rem;border:1px solid var(--line);border-radius:999px;font-size:.8rem}}
.limits-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:.75rem;list-style:none;padding:0}}.limits-grid li{{display:grid;grid-template-columns:2rem 1fr;gap:.7rem;padding:1rem;background:#191d28;border:1px solid #393b49;border-radius:.75rem}}.limits-grid span{{color:var(--paper);font:700 .75rem/1.8 ui-monospace,monospace}}.limits-grid p{{margin:0;color:#bbc4cc}}.provenance{{background:#0b1925;border:1px solid var(--line);border-radius:1rem;padding:1.2rem}}.provenance dl{{display:grid;grid-template-columns:repeat(2,1fr);gap:.65rem 2rem;margin:0}}.provenance dl div{{display:grid;grid-template-columns:minmax(9rem,.6fr) 1.4fr;gap:.75rem;border-bottom:1px solid #203545;padding:.35rem 0;min-width:0}}.provenance dt{{color:var(--muted);font-size:.78rem}}.provenance dd{{margin:0;font:500 .75rem/1.5 ui-monospace,monospace;overflow-wrap:anywhere}}
footer{{border-top:1px solid var(--line);padding:2rem max(1rem,calc((100vw - 1180px)/2));color:var(--muted);font-size:.8rem}}
@media(max-width:900px){{.metric-grid{{grid-template-columns:repeat(3,1fr)}}.chart-grid{{grid-template-columns:1fr}}.chart-card-wide{{grid-column:auto}}.scope-note,.reading-panel,.paper-summary{{grid-template-columns:1fr}}.section-heading{{display:block}}.section-heading>p{{text-align:left;margin-top:.75rem}}}}
@media(max-width:620px){{.hero{{padding-top:3.5rem}}main{{width:min(100% - 1rem,1180px)}}.metric-grid{{grid-template-columns:repeat(2,1fr)}}.scope-note{{grid-template-columns:1fr}}.check-list,.limits-grid,.provenance dl{{grid-template-columns:1fr}}.validation-strip{{display:block}}.chart-card figcaption{{display:block}}.chart-card figcaption span{{display:block;max-width:none;text-align:left;margin-top:.25rem}}h1{{font-size:2.55rem}}}}
@media print{{body{{background:white;color:#111}}.hero,main{{width:100%;padding:1rem}}.chart-card,.metric,.provenance{{break-inside:avoid}}}}
</style>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to report</a>
<header class="hero"><div class="hero-copy"><p class="kicker">Causal tracing · implementation evidence</p><h1>{_e(title)}</h1><p class="lede">A measured offline pipeline check, placed beside—not blended with—the findings published for gated Gaperon checkpoints.</p><div class="run-stamp"><span>Generated <time datetime="{_e(generated_text)}">{_e(generated_text)}</time></span><span>Result type · synthetic smoke test</span></div></div></header>
<main id="main-content">
<aside class="scope-note" aria-label="Evidence scope"><article class="local"><h2>Synthetic local validation</h2><p>{_e(_LOCAL_PAPER_BOUNDARY)}</p></article><article class="paper"><h2>Paper-reported findings</h2><p>Published numbers are transcribed as reference context and are visually isolated from locally measured values.</p></article></aside>
<dl class="metric-grid">{''.join(metrics)}</dl>
{local_body}
{paper_html}
<section id="limitations" data-section="limitations" aria-labelledby="limitations-heading"><div class="section-heading"><div><p class="eyebrow">Reproduction boundary</p><h2 id="limitations-heading">What remains unavailable</h2></div><p>These constraints determine which conclusions the local run can support.</p></div><ol class="limits-grid">{limitation_html}</ol></section>
<section id="provenance" data-section="provenance" aria-labelledby="provenance-heading"><div class="section-heading"><div><p class="eyebrow">Audit trail</p><h2 id="provenance-heading">Run provenance</h2></div><p>Versions, seeds, and the digest below make this particular smoke result inspectable.</p></div><div class="provenance"><dl>{provenance_html}</dl></div></section>
</main>
<footer>{_e(_LOCAL_PAPER_BOUNDARY)}</footer>
</body>
</html>
'''
    return html


def write_html_report(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Render and atomically replace a UTF-8 HTML report at ``path``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_html_report(data))
    return destination


__all__ = ["render_html_report", "write_html_report"]
