"""Reusable reporting helpers for prediction exports and model comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DEFAULT_LABEL_ORDER = ("contradiction", "entailment")
DEFAULT_OPTIONAL_COLUMNS = ("confidence", "raw_score", "rationale")


@dataclass(slots=True)
class MatchedQualitativeSelection:
    """Shared sample ids and manifest for qualitative panels."""

    correct_sample_ids: list[str]
    failure_sample_ids: list[str]
    manifest: pd.DataFrame


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _coerce_unique_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str).unique()
    return values[0] if len(values) == 1 else None


def _normalise_label_indices(
    frame: pd.DataFrame,
    *,
    label_order: Sequence[str],
    label_col: str,
    pred_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(label_order)}
    y_true = frame[label_col].map(label_to_index)
    y_pred = frame[pred_col].map(label_to_index)
    if y_true.isna().any() or y_pred.isna().any():
        observed = set(frame[label_col].dropna().astype(str).unique()) | set(
            frame[pred_col].dropna().astype(str).unique()
        )
        unknown_labels = sorted(observed - set(label_order))
        raise ValueError(f"Unexpected labels in comparison frame: {unknown_labels}")
    return y_true.to_numpy(dtype=int), y_pred.to_numpy(dtype=int)


def standardize_prediction_frame(
    frame: pd.DataFrame,
    *,
    model: str,
    stage: str,
    eval_scope: str,
    sample_id_col: str = "sample_id",
    label_col: str = "label",
    pred_col: str = "pred_label",
    optional_columns: Sequence[str] = DEFAULT_OPTIONAL_COLUMNS,
) -> pd.DataFrame:
    """Return a notebook-friendly standardized prediction frame."""

    required = {sample_id_col, label_col, pred_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Prediction frame is missing required columns: {missing}")

    standardized = frame.copy()
    standardized["model"] = model
    standardized["stage"] = stage
    standardized["eval_scope"] = eval_scope
    standardized["sample_id"] = standardized[sample_id_col].astype(str)
    standardized["label"] = standardized[label_col].astype(str)
    standardized["pred_label"] = standardized[pred_col].astype(str)
    standardized["correct"] = standardized["label"].eq(standardized["pred_label"])

    for column in ("edit_family", "file_path", "source_caption", "edited_caption", *optional_columns):
        if column not in standardized.columns:
            standardized[column] = pd.NA

    preferred_columns = [
        "sample_id",
        "model",
        "stage",
        "eval_scope",
        "label",
        "pred_label",
        "correct",
        "edit_family",
        "file_path",
        "source_caption",
        "edited_caption",
        *optional_columns,
    ]
    remaining_columns = [column for column in standardized.columns if column not in preferred_columns]
    ordered = standardized.loc[:, [*preferred_columns, *remaining_columns]]
    return ordered.reset_index(drop=True)


def save_prediction_export(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    model: str,
    stage: str,
    eval_scope: str,
    sample_id_col: str = "sample_id",
    label_col: str = "label",
    pred_col: str = "pred_label",
    optional_columns: Sequence[str] = DEFAULT_OPTIONAL_COLUMNS,
) -> pd.DataFrame:
    """Standardize a prediction frame and write it to CSV."""

    standardized = standardize_prediction_frame(
        frame,
        model=model,
        stage=stage,
        eval_scope=eval_scope,
        sample_id_col=sample_id_col,
        label_col=label_col,
        pred_col=pred_col,
        optional_columns=optional_columns,
    )
    standardized.to_csv(_prepare_output(output_path), index=False)
    return standardized


def _compute_binary_metrics(
    frame: pd.DataFrame,
    *,
    label_order: Sequence[str],
    label_col: str,
    pred_col: str,
) -> dict[str, float]:
    y_true, y_pred = _normalise_label_indices(
        frame,
        label_order=label_order,
        label_col=label_col,
        pred_col=pred_col,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(label_order))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "precision_contradiction": float(precision[0]),
        "recall_contradiction": float(recall[0]),
        "precision_entailment": float(precision[1]),
        "recall_entailment": float(recall[1]),
    }


def build_model_comparison_summary(
    frames_by_model: Mapping[str, pd.DataFrame],
    *,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
    label_col: str = "label",
    pred_col: str = "pred_label",
) -> pd.DataFrame:
    """Build a cross-model summary table for binary comparison charts."""

    rows: list[dict[str, float | int | str | None]] = []
    for model_name, frame in frames_by_model.items():
        if frame.empty:
            rows.append(
                {
                    "model": model_name,
                    "sample_count": 0,
                    "stage": _coerce_unique_value(frame, "stage"),
                    "eval_scope": _coerce_unique_value(frame, "eval_scope"),
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "precision_contradiction": 0.0,
                    "recall_contradiction": 0.0,
                    "precision_entailment": 0.0,
                    "recall_entailment": 0.0,
                }
            )
            continue

        metrics = _compute_binary_metrics(
            frame,
            label_order=label_order,
            label_col=label_col,
            pred_col=pred_col,
        )
        rows.append(
            {
                "model": model_name,
                "sample_count": int(len(frame)),
                "stage": _coerce_unique_value(frame, "stage"),
                "eval_scope": _coerce_unique_value(frame, "eval_scope"),
                **metrics,
            }
        )

    columns = [
        "model",
        "sample_count",
        "stage",
        "eval_scope",
        "accuracy",
        "macro_f1",
        "precision_contradiction",
        "recall_contradiction",
        "precision_entailment",
        "recall_entailment",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def build_model_comparison_per_family(
    frames_by_model: Mapping[str, pd.DataFrame],
    *,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
    label_col: str = "label",
    pred_col: str = "pred_label",
    family_col: str = "edit_family",
) -> pd.DataFrame:
    """Build a per-family comparison table across models."""

    rows: list[dict[str, float | int | str | None]] = []
    for model_name, frame in frames_by_model.items():
        if frame.empty or family_col not in frame.columns:
            continue
        for family, group in frame.groupby(family_col, dropna=False):
            metrics = _compute_binary_metrics(
                group,
                label_order=label_order,
                label_col=label_col,
                pred_col=pred_col,
            )
            rows.append(
                {
                    "model": model_name,
                    "edit_family": str(family),
                    "count": int(len(group)),
                    "stage": _coerce_unique_value(group, "stage"),
                    "eval_scope": _coerce_unique_value(group, "eval_scope"),
                    **metrics,
                }
            )

    columns = [
        "model",
        "edit_family",
        "count",
        "stage",
        "eval_scope",
        "accuracy",
        "macro_f1",
        "precision_contradiction",
        "recall_contradiction",
        "precision_entailment",
        "recall_entailment",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["model", "edit_family"]).reset_index(drop=True)


def save_comparison_tables(
    frames_by_model: Mapping[str, pd.DataFrame],
    summary_output_path: str | Path,
    per_family_output_path: str | Path,
    *,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
    label_col: str = "label",
    pred_col: str = "pred_label",
    family_col: str = "edit_family",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist comparison tables for a set of model frames."""

    summary = build_model_comparison_summary(
        frames_by_model,
        label_order=label_order,
        label_col=label_col,
        pred_col=pred_col,
    )
    per_family = build_model_comparison_per_family(
        frames_by_model,
        label_order=label_order,
        label_col=label_col,
        pred_col=pred_col,
        family_col=family_col,
    )
    summary.to_csv(_prepare_output(summary_output_path), index=False)
    per_family.to_csv(_prepare_output(per_family_output_path), index=False)
    return summary, per_family


def _rank_frame(frame: pd.DataFrame, *, ascending: bool, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["_shuffle"] = np.random.default_rng(seed).random(len(ranked))
    ranked = ranked.sort_values(
        by=["error_count", "_shuffle", "sample_id"],
        ascending=[ascending, True, True],
        kind="mergesort",
    )
    return ranked.drop(columns="_shuffle").reset_index(drop=True)


def _select_pool(
    pool_frame: pd.DataFrame,
    *,
    desired_count: int,
    primary_mask: pd.Series,
    seed: int,
    ascending: bool,
    exclude_ids: Sequence[str] = (),
) -> list[str]:
    working = pool_frame.loc[~pool_frame["sample_id"].isin(set(exclude_ids))].copy()
    working_primary_mask = working["sample_id"].isin(pool_frame.loc[primary_mask, "sample_id"])

    selected: list[str] = []
    ordered_primary = _rank_frame(working.loc[working_primary_mask].copy(), ascending=ascending, seed=seed)
    selected.extend(ordered_primary["sample_id"].tolist())

    if len(selected) < desired_count:
        remaining = working.loc[~working["sample_id"].isin(selected)].copy()
        ordered_remaining = _rank_frame(remaining, ascending=ascending, seed=seed + 1)
        selected.extend(ordered_remaining["sample_id"].tolist())

    return selected[:desired_count]


def select_matched_qualitative_samples(
    frames_by_model: Mapping[str, pd.DataFrame],
    *,
    correct_count: int = 8,
    failure_count: int = 8,
    seed: int = 42,
    sample_id_col: str = "sample_id",
    label_col: str = "label",
    pred_col: str = "pred_label",
) -> MatchedQualitativeSelection:
    """Select matched qualitative sample ids shared across all model frames."""

    if not frames_by_model:
        raise ValueError("frames_by_model must contain at least one model frame.")

    keyed_frames: dict[str, pd.DataFrame] = {}
    common_ids: set[str] | None = None
    for model_name, frame in frames_by_model.items():
        required = {sample_id_col, label_col, pred_col}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Frame for model {model_name!r} is missing columns: {missing}")
        keyed = frame.loc[:, [sample_id_col, label_col, pred_col]].copy()
        keyed[sample_id_col] = keyed[sample_id_col].astype(str)
        keyed[label_col] = keyed[label_col].astype(str)
        keyed[pred_col] = keyed[pred_col].astype(str)
        keyed = keyed.drop_duplicates(subset=[sample_id_col], keep="last").set_index(sample_id_col)
        keyed_frames[model_name] = keyed
        model_ids = set(keyed.index.astype(str))
        common_ids = model_ids if common_ids is None else common_ids & model_ids

    common_ids = common_ids or set()
    if not common_ids:
        raise ValueError("frames_by_model do not share any sample ids.")

    rows: list[dict[str, object]] = []
    for sample_id in sorted(common_ids):
        labels = {frame.loc[sample_id, label_col] for frame in keyed_frames.values()}
        if len(labels) != 1:
            raise ValueError(f"Sample {sample_id!r} has inconsistent true labels across frames.")
        true_label = next(iter(labels))
        error_count = sum(frame.loc[sample_id, pred_col] != true_label for frame in keyed_frames.values())
        rows.append(
            {
                "sample_id": sample_id,
                "error_count": int(error_count),
                "correct_votes": int(len(keyed_frames) - error_count),
                "total_models": int(len(keyed_frames)),
            }
        )

    pool_frame = pd.DataFrame(rows)
    if pool_frame.empty:
        raise ValueError("No shared qualitative samples were available for selection.")

    correct_ids = _select_pool(
        pool_frame,
        desired_count=correct_count,
        primary_mask=pool_frame["error_count"].eq(0),
        seed=seed,
        ascending=True,
    )

    failure_candidates = pool_frame["error_count"].gt(0) & ~pool_frame["sample_id"].isin(correct_ids)
    failure_ids = _select_pool(
        pool_frame,
        desired_count=failure_count,
        primary_mask=failure_candidates,
        seed=seed + 11,
        ascending=False,
        exclude_ids=correct_ids,
    )
    if len(failure_ids) < failure_count:
        remaining = pool_frame.loc[~pool_frame["sample_id"].isin(failure_ids)].copy()
        ordered_remaining = _rank_frame(remaining, ascending=False, seed=seed + 13)
        failure_ids.extend(ordered_remaining["sample_id"].tolist())
        failure_ids = failure_ids[:failure_count]

    manifest = pd.concat(
        [
            pool_frame.loc[pool_frame["sample_id"].isin(correct_ids)]
            .assign(pool="correct")
            .assign(selected_order=lambda frame: pd.Categorical(frame["sample_id"], categories=correct_ids, ordered=True))
            .sort_values("selected_order")
            .drop(columns="selected_order"),
            pool_frame.loc[pool_frame["sample_id"].isin(failure_ids)]
            .assign(pool="failure")
            .assign(selected_order=lambda frame: pd.Categorical(frame["sample_id"], categories=failure_ids, ordered=True))
            .sort_values("selected_order")
            .drop(columns="selected_order"),
        ],
        ignore_index=True,
    )
    return MatchedQualitativeSelection(
        correct_sample_ids=correct_ids,
        failure_sample_ids=failure_ids,
        manifest=manifest.reset_index(drop=True),
    )


def slice_prediction_frame(frame: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    """Return rows for the requested sample ids in the given order."""

    if not sample_ids:
        return frame.iloc[0:0].copy()
    order_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    subset = frame.loc[frame["sample_id"].astype(str).isin(order_lookup)].copy()
    subset["_sample_order"] = subset["sample_id"].map(order_lookup)
    return subset.sort_values("_sample_order").drop(columns="_sample_order").reset_index(drop=True)
