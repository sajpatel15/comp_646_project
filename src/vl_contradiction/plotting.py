"""Plot helpers for report-ready figures."""

from __future__ import annotations

from pathlib import Path
import textwrap

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


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


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
        frame["val_macro_f1"],
        color="tab:orange",
        label="Val Macro-F1",
        linewidth=2,
    )[0]
    ax2.set_ylabel("Macro-F1")
    ax1.set_title(title)
    _add_footer_legend(legend_ax, [train_line, val_line], ["Train Loss", "Val Macro-F1"])
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
    fig.tight_layout(rect=(0, 0, 0.82, 1))
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
        image = Image.open(image_path).convert("RGB")
    except Exception:  # pragma: no cover - defensive fallback for bad paths/files
        return Image.new("RGB", size, color="white")
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color="white")
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    canvas.paste(contained, offset)
    return canvas


def _format_panel_lines(row: pd.Series, *, text_width: int) -> list[str]:
    def add_line(lines: list[str], label: str, value: object, *, wrap: bool = False) -> None:
        if pd.isna(value):
            return
        text = str(value).strip()
        if not text:
            return
        if wrap:
            text = textwrap.fill(text, width=text_width, break_long_words=False, break_on_hyphens=False)
        lines.append(f"{label}: {text}")

    lines: list[str] = []
    add_line(lines, "Sample", row.get("sample_id"))
    add_line(lines, "Model", row.get("model"))
    add_line(lines, "Stage", row.get("stage"))
    add_line(lines, "Scope", row.get("eval_scope"))
    add_line(lines, "True", row.get("label"))
    add_line(lines, "Pred", row.get("pred_label"))
    if "correct" in row.index and not pd.isna(row.get("correct")):
        correct_value = row.get("correct")
        if isinstance(correct_value, str):
            correct_value = correct_value.strip().lower() in {"1", "true", "yes", "y"}
        add_line(lines, "Correct", "yes" if bool(correct_value) else "no")
    add_line(lines, "Family", row.get("edit_family"))
    add_line(lines, "Confidence", row.get("confidence"))
    add_line(lines, "Raw score", row.get("raw_score"))
    add_line(lines, "Rationale", row.get("rationale"), wrap=True)
    add_line(lines, "Source", row.get("source_caption"), wrap=True)
    add_line(lines, "Edited", row.get("edited_caption"), wrap=True)
    return lines


def save_qualitative_panel(
    frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
    max_rows: int = 4,
    *,
    thumbnail_size: tuple[int, int] = (320, 320),
    text_width: int = 58,
) -> None:
    if frame.empty:
        raise ValueError("frame must contain at least one qualitative example.")

    output = _prepare_output(output_path)
    subset = frame.head(max_rows).reset_index(drop=True)
    if subset.empty:
        raise ValueError("frame must contain at least one row to render.")

    fig_height = max(4.0, 3.8 * len(subset))
    fig, axes = plt.subplots(
        len(subset),
        2,
        figsize=(12.4, fig_height),
        gridspec_kw={"width_ratios": [1.05, 1.55], "wspace": 0.08, "hspace": 0.38},
    )
    axes = np.atleast_2d(axes)
    for axis_row, (_, row) in zip(axes, subset.iterrows(), strict=True):
        thumbnail = _load_thumbnail(row["file_path"], thumbnail_size=thumbnail_size)
        axis_row[0].imshow(thumbnail)
        axis_row[0].set_aspect("equal")
        axis_row[0].axis("off")

        axis_row[1].axis("off")
        axis_row[1].set_xlim(0.0, 1.0)
        axis_row[1].set_ylim(0.0, 1.0)
        axis_row[1].text(
            0.0,
            1.0,
            "\n".join(_format_panel_lines(row, text_width=text_width)),
            va="top",
            ha="left",
            fontsize=10,
        )

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
