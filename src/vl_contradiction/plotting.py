"""Plot helpers for report-ready figures."""

from __future__ import annotations

import math
from pathlib import Path
import textwrap
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .labels import CLASS_ORDER


sns.set_theme(style="whitegrid", context="talk")


LABEL_ORDER = CLASS_ORDER
LABEL_PALETTE = {
    "contradiction": "#c44e52",
    "entailment": "#4c72b0",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLAB_PROJECT_ROOT = Path("/content/project")
DATASET_ROOT = PROJECT_ROOT / "artifacts" / "datasets"
_IMAGE_PATH_CACHE: dict[str, Path] = {}


def _resolve_image_path(image_path: str | Path) -> Path:
    candidate = Path(image_path)
    cache_key = str(candidate)
    if cache_key in _IMAGE_PATH_CACHE and _IMAGE_PATH_CACHE[cache_key].exists():
        return _IMAGE_PATH_CACHE[cache_key]
    if candidate.exists():
        _IMAGE_PATH_CACHE[cache_key] = candidate
        return candidate
    try:
        relative_path = candidate.relative_to(COLAB_PROJECT_ROOT)
    except ValueError:
        if candidate.is_absolute():
            if DATASET_ROOT.exists() and candidate.name:
                matches = list(DATASET_ROOT.rglob(candidate.name))
                if matches:
                    preferred = next((match for match in matches if match.parent.name == candidate.parent.name), matches[0])
                    _IMAGE_PATH_CACHE[cache_key] = preferred
                    return preferred
            return candidate
        remapped = PROJECT_ROOT / candidate
        if remapped.exists():
            _IMAGE_PATH_CACHE[cache_key] = remapped
            return remapped
        return candidate
    remapped = PROJECT_ROOT / relative_path
    if remapped.exists():
        _IMAGE_PATH_CACHE[cache_key] = remapped
        return remapped
    if DATASET_ROOT.exists() and candidate.name:
        matches = list(DATASET_ROOT.rglob(candidate.name))
        if matches:
            preferred = next((match for match in matches if match.parent.name == candidate.parent.name), matches[0])
            _IMAGE_PATH_CACHE[cache_key] = preferred
            return preferred
    return candidate


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def resolve_figure_output(figure_root: str | Path, *parts: str) -> Path:
    """Build a figure output path under a categorized figure root."""

    if not parts:
        raise ValueError("At least one path component is required for a figure output.")
    return _prepare_output(Path(figure_root).joinpath(*parts))


def _add_footer_legend(legend_ax: plt.Axes, handles: list[object], labels: list[str]) -> None:
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        ncol=max(1, len(labels)),
        columnspacing=1.6,
        handlelength=2.4,
    )


def save_training_curves(history: list[dict[str, float]], output_path: str | Path, title: str) -> None:
    frame = pd.DataFrame(history)
    output = _prepare_output(output_path)
    metric_column = "val_accuracy" if "val_accuracy" in frame.columns else "val_macro_f1"
    metric_label = "Val Accuracy" if metric_column == "val_accuracy" else "Val Macro-F1"
    axis_label = "Accuracy" if metric_column == "val_accuracy" else "Macro-F1"
    fig = plt.figure(figsize=(8.8, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.22], hspace=0.28)
    ax1 = fig.add_subplot(grid[0])
    legend_ax = fig.add_subplot(grid[1])
    train_line = ax1.plot(frame["epoch"], frame["train_loss"], label="Train Loss", linewidth=2)[0]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    val_line = ax2.plot(
        frame["epoch"],
        frame[metric_column],
        color="tab:orange",
        label=metric_label,
        linewidth=2,
    )[0]
    ax2.set_ylabel(axis_label)
    ax1.set_title(title)
    _add_footer_legend(legend_ax, [train_line, val_line], ["Train Loss", metric_label])
    fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.10, hspace=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_score_histogram(score_frame: pd.DataFrame, output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    label_order = [label for label in LABEL_ORDER if label in set(score_frame["label"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.histplot(
        score_frame,
        x="raw_score",
        hue="label",
        hue_order=label_order,
        palette=LABEL_PALETTE,
        bins=24,
        element="step",
        fill=True,
        alpha=0.18,
        linewidth=1.8,
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    if ax.legend_ is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), title="Label", frameon=True)
    fig.subplots_adjust(left=0.10, right=0.79, bottom=0.14, top=0.90)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep(search_frame: pd.DataFrame, output_path: str | Path, title: str) -> None:
    if search_frame.empty:
        raise ValueError("search_frame must contain threshold sweep results.")

    output = _prepare_output(output_path)
    if "tau" in search_frame.columns:
        best_row = search_frame.loc[search_frame["macro_f1"].idxmax()]
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        primary_line = ax.plot(
            search_frame["tau"],
            search_frame["macro_f1"],
            linewidth=2.2,
            color="#4c72b0",
            label="Macro-F1",
        )[0]
        handles = [primary_line]
        labels = ["Macro-F1"]
        if "accuracy" in search_frame.columns:
            accuracy_line = ax.plot(
                search_frame["tau"],
                search_frame["accuracy"],
                linewidth=1.8,
                color="#55a868",
                linestyle="--",
                label="Accuracy",
            )[0]
            handles.append(accuracy_line)
            labels.append("Accuracy")
        ax.scatter(
            [best_row["tau"]],
            [best_row["macro_f1"]],
            marker="*",
            s=240,
            color="#d62728",
            edgecolors="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            f"Best tau={best_row['tau']:.4f}\nmacro-F1={best_row['macro_f1']:.3f}",
            xy=(best_row["tau"], best_row["macro_f1"]),
            xytext=(0.04, 0.96),
            textcoords="axes fraction",
            va="top",
            ha="left",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#555555", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.2},
        )
        ax.set_xlabel("tau")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.legend(handles, labels, loc="lower right", frameon=True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return

    best_row = search_frame.loc[search_frame["macro_f1"].idxmax()]
    score_min = float(search_frame[["tau_low", "tau_high"]].min().min())
    score_max = float(search_frame[["tau_low", "tau_high"]].max().max())

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter = ax.scatter(
        search_frame["tau_low"],
        search_frame["tau_high"],
        c=search_frame["macro_f1"],
        cmap="viridis",
        s=24,
        edgecolors="none",
        alpha=0.9,
        rasterized=len(search_frame) > 5_000,
    )
    ax.plot(
        [score_min, score_max],
        [score_min, score_max],
        linestyle="--",
        color="black",
        linewidth=1.1,
        alpha=0.55,
    )
    ax.scatter(
        best_row["tau_low"],
        best_row["tau_high"],
        marker="*",
        s=260,
        color="#d62728",
        edgecolors="white",
        linewidth=1.0,
        zorder=3,
    )
    ax.annotate(
        (
            "Best pair\n"
            f"tau_low={best_row['tau_low']:.4f}\n"
            f"tau_high={best_row['tau_high']:.4f}\n"
            f"macro-F1={best_row['macro_f1']:.3f}"
        ),
        xy=(best_row["tau_low"], best_row["tau_high"]),
        xytext=(0.04, 0.96),
        textcoords="axes fraction",
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#555555", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.2},
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Validation Macro-F1")
    ax.set_xlabel("tau_low")
    ax.set_ylabel("tau_high")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_bar_chart(frame: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(frame, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_grouped_comparison_chart(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    *,
    model_col: str = "model",
    metrics: tuple[str, str] = ("accuracy", "macro_f1"),
) -> None:
    """Save a grouped bar chart comparing overall model scores."""

    if frame.empty:
        raise ValueError("frame must contain comparison metrics.")
    missing = [column for column in (model_col, *metrics) if column not in frame.columns]
    if missing:
        raise KeyError(f"Comparison frame is missing columns: {missing}")

    output = _prepare_output(output_path)
    model_order = pd.Index(frame[model_col].astype(str)).unique().tolist()
    long_frame = frame.loc[:, [model_col, *metrics]].melt(
        id_vars=model_col,
        value_vars=list(metrics),
        var_name="metric",
        value_name="score",
    )
    long_frame[model_col] = long_frame[model_col].astype(str)
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    sns.barplot(
        data=long_frame,
        x=model_col,
        y="score",
        hue="metric",
        order=model_order,
        hue_order=list(metrics),
        palette=sns.color_palette("Set2", n_colors=len(metrics)),
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(title)
    if ax.legend_ is not None:
        ax.legend(title="", loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_per_family_accuracy_heatmap(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    *,
    family_col: str = "edit_family",
    model_col: str = "model",
    value_col: str = "accuracy",
) -> None:
    """Save a heatmap of per-family accuracy across models."""

    if frame.empty:
        raise ValueError("frame must contain per-family metrics.")
    missing = [column for column in (family_col, model_col, value_col) if column not in frame.columns]
    if missing:
        raise KeyError(f"Per-family frame is missing columns: {missing}")

    pivot = (
        frame.loc[:, [family_col, model_col, value_col]]
        .assign(**{family_col: lambda data: data[family_col].astype(str), model_col: lambda data: data[model_col].astype(str)})
        .pivot_table(index=family_col, columns=model_col, values=value_col, aggfunc="mean")
        .sort_index()
    )
    model_order = pd.Index(frame[model_col].astype(str)).unique().tolist()
    pivot = pivot.reindex(columns=model_order)
    fig_width = max(8.0, 1.2 * max(len(model_order), 1) + 3.2)
    fig_height = max(4.8, 0.55 * max(len(pivot.index), 1) + 2.5)
    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Edit Family")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_reliability_diagram(bin_centers: np.ndarray, bin_accuracy: np.ndarray, bin_confidence: np.ndarray, output_path: str | Path, title: str) -> None:
    output = _prepare_output(output_path)
    fig = plt.figure(figsize=(7.2, 6.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.22], hspace=0.28)
    ax = fig.add_subplot(grid[0])
    legend_ax = fig.add_subplot(grid[1])
    perfect_line = ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=1,
        label="Perfect Calibration",
    )[0]
    observed_line = ax.plot(
        bin_confidence,
        bin_accuracy,
        marker="o",
        linewidth=2,
        label="Observed Accuracy",
    )[0]
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _add_footer_legend(legend_ax, [perfect_line, observed_line], ["Perfect Calibration", "Observed Accuracy"])
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.10, hspace=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _load_thumbnail(image_path: str | Path, thumbnail_size: tuple[int, int]) -> Image.Image:
    size = tuple(int(dimension) for dimension in thumbnail_size)
    try:
        image = Image.open(_resolve_image_path(image_path)).convert("RGB")
    except Exception:  # pragma: no cover - defensive fallback for bad paths/files
        return Image.new("RGB", size, color="white")
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color="white")
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas


def _format_decimal(value: object, *, digits: int = 4, round_down: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if not math.isfinite(numeric):
        return str(value).strip()
    if round_down:
        scale = 10**digits
        numeric = math.trunc(numeric * scale) / scale
    return f"{numeric:.{digits}f}"


def _format_panel_lines(row: pd.Series, *, text_width: int) -> list[str]:
    def add_line(
        lines: list[str],
        label: str,
        value: object,
        *,
        wrap: bool = False,
        formatter: Callable[[object], str] | None = None,
    ) -> None:
        if pd.isna(value):
            return
        text = formatter(value) if formatter is not None else str(value).strip()
        if not text:
            return
        if wrap:
            text = textwrap.fill(text, width=text_width, break_long_words=False, break_on_hyphens=False)
        lines.append(f"{label}: {text}")

    lines: list[str] = []
    add_line(lines, "True", row.get("label"))
    add_line(lines, "Pred", row.get("pred_label"))
    if "correct" in row.index and not pd.isna(row.get("correct")):
        correct_value = row.get("correct")
        if isinstance(correct_value, str):
            correct_value = correct_value.strip().lower() in {"1", "true", "yes", "y"}
        add_line(lines, "Correct", "yes" if bool(correct_value) else "no")
    add_line(lines, "Scope", row.get("eval_scope"))
    add_line(lines, "Confidence", row.get("confidence"), formatter=lambda value: _format_decimal(value, round_down=True))
    add_line(lines, "Raw score", row.get("raw_score"), formatter=_format_decimal)
    add_line(lines, "Source", row.get("source_caption"), wrap=True)
    add_line(lines, "Edited", row.get("edited_caption"), wrap=True)
    return lines


def build_qualitative_panel_figure(
    frame: pd.DataFrame,
    title: str,
    max_rows: int = 4,
    *,
    thumbnail_size: tuple[int, int] = (320, 320),
    text_width: int = 32,
    ncols: int = 4,
) -> plt.Figure:
    if frame.empty:
        raise ValueError("frame must contain at least one qualitative example.")

    subset = frame.head(max_rows).reset_index(drop=True)
    if subset.empty:
        raise ValueError("frame must contain at least one row to render.")

    ncols = max(1, min(ncols, len(subset)))
    nrows = int(math.ceil(len(subset) / ncols))
    height_ratios: list[float] = []
    for _ in range(nrows):
        height_ratios.extend([1.0, 0.82])
    fig = plt.figure(figsize=(2.85 * ncols, 4.20 * nrows + 0.30))
    grid = fig.add_gridspec(
        nrows * 2,
        ncols,
        height_ratios=height_ratios,
    )
    for index, (_, row) in enumerate(subset.iterrows()):
        row_index = index // ncols
        col_index = index % ncols
        image_ax = fig.add_subplot(grid[row_index * 2, col_index])
        text_ax = fig.add_subplot(grid[row_index * 2 + 1, col_index])
        thumbnail = _load_thumbnail(row["file_path"], thumbnail_size=thumbnail_size)
        image_ax.imshow(thumbnail)
        image_ax.set_aspect("equal")
        image_ax.axis("off")

        text_ax.axis("off")
        text_ax.text(
            0.0,
            1.0,
            "\n".join(_format_panel_lines(row, text_width=text_width)),
            va="top",
            ha="left",
            fontsize=7.8,
            transform=text_ax.transAxes,
        )

    fig.suptitle(title, y=0.965)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.04, top=0.88, wspace=0.05, hspace=0.06)
    return fig


def save_qualitative_panel(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    max_rows: int = 4,
    *,
    thumbnail_size: tuple[int, int] = (320, 320),
    text_width: int = 32,
    ncols: int = 4,
) -> None:
    output = _prepare_output(output_path)
    fig = build_qualitative_panel_figure(
        frame,
        title,
        max_rows=max_rows,
        thumbnail_size=thumbnail_size,
        text_width=text_width,
        ncols=ncols,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_spot_checks(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    sample_count: int = 5,
    seed: int = 42,
    manifest_name: str = "benchmark_spot_checks.csv",
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("frame must contain at least one benchmark row.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    key_column = "image_id" if "image_id" in frame.columns else "file_path"
    representative_rows = frame.drop_duplicates(subset=[key_column]).copy()
    if representative_rows.empty:
        raise ValueError("frame must contain at least one unique benchmark image.")

    selection = representative_rows.sample(min(sample_count, len(representative_rows)), random_state=seed).reset_index(drop=True)
    output_map: dict[object, str] = {}
    for index, (_, row) in enumerate(selection.iterrows(), start=1):
        output_path = output_root / f"benchmark_spot_check_{index:02d}.png"
        output_map[row[key_column]] = str(output_path)
        save_qualitative_panel(
            pd.DataFrame([row]),
            output_path,
            f"Benchmark Example {index}",
            max_rows=1,
            thumbnail_size=(360, 360),
            text_width=46,
            ncols=1,
        )

    manifest = frame.loc[frame[key_column].isin(selection[key_column])].copy()
    manifest["spot_check_png"] = manifest[key_column].map(output_map)
    manifest_output = _prepare_output(output_root / manifest_name)
    manifest.to_csv(manifest_output, index=False)
    return manifest
