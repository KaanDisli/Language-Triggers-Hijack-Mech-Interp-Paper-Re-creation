"""Small plotting helpers for the paper's heatmaps and ablation curves."""

from __future__ import annotations

from typing import Any, Sequence


def plot_head_heatmap(
    scores: Any,
    *,
    title: str | None = None,
    ax: Any | None = None,
    cmap: str = "coolwarm",
) -> Any:
    """Plot ``[layer, head]`` signed log-probability restoration scores."""

    plt = _pyplot()
    ax = ax or plt.subplots(figsize=(10, 6))[1]
    image = ax.imshow(_array(scores), aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("Query head")
    ax.set_ylabel("Layer")
    if title:
        ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="Δ log p(first continuation token)")
    return ax


def plot_layer_token_heatmap(
    scores: Any,
    *,
    title: str | None = None,
    ax: Any | None = None,
    cmap: str = "coolwarm",
) -> Any:
    """Plot ``[layer, relative trigger token]`` residual-patching scores."""

    plt = _pyplot()
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    image = ax.imshow(_array(scores), aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("Relative trigger-token position")
    ax.set_ylabel("Layer")
    if title:
        ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="Δ log p(first continuation token)")
    return ax


def plot_overlap_matrix(
    matrix: Any,
    labels: Sequence[str],
    *,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot an annotated Jaccard matrix."""

    plt = _pyplot()
    values = _array(matrix)
    if values.shape != (len(labels), len(labels)):
        raise ValueError("Overlap matrix shape must match the label count")
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center")
    if title:
        ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="Jaccard index")
    return ax


def plot_ablation_curve(
    points: Sequence[Any],
    *,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot selected-minus-random continuation perplexity."""

    if not points:
        raise ValueError("At least one ablation point is required")
    plt = _pyplot()
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    x = [int(_field(point, "num_heads")) for point in points]
    y = [float(_field(point, "delta_perplexity")) for point in points]
    error = [float(_field(point, "random_std")) for point in points]
    ax.errorbar(x, y, yerr=error, marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Number of ablated heads")
    ax.set_ylabel("ΔPPL (selected − random)")
    if title:
        ax.set_title(title)
    return ax


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)


def _array(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires NumPy and matplotlib") from exc
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix")
    return result


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install plotting support with `pip install -e '.[plots]'`") from exc
    return plt

