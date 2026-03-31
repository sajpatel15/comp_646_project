"""Training helpers for learned CLIP baselines and fusion models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light environments
    SummaryWriter = None

from .metrics import compute_classification_metrics


class FeatureDataset(Dataset):
    """Dataset wrapper for feature vectors and integer labels."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class TokenDataset(Dataset):
    """Dataset wrapper for image tokens, text tokens, and labels."""

    def __init__(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor, labels: torch.Tensor) -> None:
        self.image_tokens = image_tokens
        self.text_tokens = text_tokens
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.image_tokens[index], self.text_tokens[index], self.labels[index]


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_val_metrics: dict[str, float | list[list[int]]]
    best_checkpoint: Path | None
    val_logits: torch.Tensor
    val_labels: torch.Tensor


@dataclass(slots=True)
class TrainingTrialConfig:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


@dataclass(slots=True)
class TrainingSweepResult:
    trial_rows: list[dict[str, str | int | float]]
    best_trial: TrainingTrialConfig
    best_trial_row: dict[str, str | int | float]
    best_result: TrainingResult
    best_model: nn.Module
    best_test_metrics: dict[str, float | list[list[int]]]
    best_test_logits: torch.Tensor
    best_test_labels: torch.Tensor
    best_checkpoint: Path | None


def create_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a dataloader with sane notebook defaults."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def _forward_batch(model: nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 2:
        features, labels = batch
        logits = model(features.to(device))
    elif len(batch) == 3:
        image_tokens, text_tokens, labels = batch
        logits = model(image_tokens.to(device), text_tokens.to(device))
    else:
        raise ValueError("Unexpected batch shape")
    return logits, labels.to(device)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float | list[list[int]]], torch.Tensor, torch.Tensor]:
    """Evaluate a model on a dataloader and return metrics, logits, and labels."""

    model.eval()
    logits_rows = []
    label_rows = []
    with torch.inference_mode():
        for batch in loader:
            logits, labels = _forward_batch(model, batch, device)
            logits_rows.append(logits.detach().cpu())
            label_rows.append(labels.detach().cpu())
    logits_tensor = torch.cat(logits_rows, dim=0)
    labels_tensor = torch.cat(label_rows, dim=0)
    predictions = logits_tensor.argmax(dim=1).numpy()
    metrics = compute_classification_metrics(labels_tensor.numpy(), predictions)
    return metrics, logits_tensor, labels_tensor


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_selection_value(metrics: dict[str, float | list[list[int]]], selection_metric: str) -> float:
    if selection_metric not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(f"Selection metric '{selection_metric}' is not present in metrics. Available metrics: {available}")
    value = metrics[selection_metric]
    if not isinstance(value, (float, int)):
        raise ValueError(f"Selection metric '{selection_metric}' must be numeric, got {type(value).__name__}")
    return float(value)


def get_stage_trials(training_config: object, stage: str, model_name: str) -> list[TrainingTrialConfig]:
    """Resolve the configured hyperparameter trials for one stage/model pair."""

    sweeps = getattr(training_config, "sweeps", None)
    if not isinstance(sweeps, dict):
        raise ValueError("training.sweeps must be configured as a nested stage/model mapping")

    normalized_stage = _normalize_name(stage)
    normalized_model = _normalize_name(model_name)
    stage_config = sweeps.get(normalized_stage)
    if not isinstance(stage_config, dict):
        available = ", ".join(sorted(sweeps))
        raise ValueError(f"Unknown training stage '{stage}'. Available stages: {available}")

    model_config = stage_config.get(normalized_model)
    if not isinstance(model_config, dict):
        available = ", ".join(sorted(stage_config))
        raise ValueError(
            f"Unknown training model '{model_name}' for stage '{normalized_stage}'. Available models: {available}"
        )

    raw_trials = model_config.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ValueError(f"training.sweeps.{normalized_stage}.{normalized_model}.trials must be a non-empty list")

    resolved_trials = []
    for index, raw_trial in enumerate(raw_trials, start=1):
        if not isinstance(raw_trial, dict):
            raise ValueError(
                f"training.sweeps.{normalized_stage}.{normalized_model}.trials[{index - 1}] must be a mapping"
            )
        try:
            trial = TrainingTrialConfig(**raw_trial)
        except TypeError as exc:
            raise ValueError(
                f"Invalid training trial at training.sweeps.{normalized_stage}.{normalized_model}.trials[{index - 1}]"
            ) from exc
        if trial.epochs <= 0 or trial.batch_size <= 0:
            raise ValueError(f"Trial '{trial.name}' must use positive epochs and batch_size")
        if trial.learning_rate <= 0.0 or trial.weight_decay < 0.0:
            raise ValueError(f"Trial '{trial.name}' must use a positive learning rate and non-negative weight decay")
        resolved_trials.append(trial)
    return resolved_trials


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    selection_metric: str = "macro_f1",
    log_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train a classifier and log one compact line per epoch."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=str(log_dir)) if log_dir and SummaryWriter is not None else None
    history = []
    best_metrics: dict[str, float | list[list[int]]] = {selection_metric: -1.0}
    best_checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_logits = torch.empty(0)
    best_labels = torch.empty(0, dtype=torch.long)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            logits, labels = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = labels.shape[0]
            epoch_loss += float(loss.detach().cpu().item()) * batch_size
            total_samples += batch_size

        train_loss = epoch_loss / max(total_samples, 1)
        val_metrics, val_logits, val_labels = evaluate_model(model, val_loader, device)
        selection_value = _resolve_selection_value(val_metrics, selection_metric)
        epoch_row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_macro_f1": float(val_metrics["macro_f1"]),
            f"val_{selection_metric}": selection_value,
        }
        history.append(epoch_row)
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_{selection_metric}={selection_value:.4f}")
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar(f"val/{selection_metric}", selection_value, epoch)
            if selection_metric != "macro_f1":
                writer.add_scalar("val/macro_f1", float(val_metrics["macro_f1"]), epoch)
        if selection_value > _resolve_selection_value(best_metrics, selection_metric):
            best_metrics = val_metrics
            best_state_dict = deepcopy(model.state_dict())
            best_logits = val_logits
            best_labels = val_labels
            if best_checkpoint_path:
                best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_checkpoint_path)

    if writer:
        writer.close()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return TrainingResult(
        history=history,
        best_val_metrics=best_metrics,
        best_checkpoint=best_checkpoint_path if best_checkpoint_path and best_checkpoint_path.exists() else None,
        val_logits=best_logits,
        val_labels=best_labels,
    )


def run_training_sweep(
    *,
    model_name: str,
    model_factory: Callable[[], nn.Module],
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    trials: list[TrainingTrialConfig],
    device: torch.device,
    num_workers: int,
    selection_metric: str,
    log_root: str | Path,
    checkpoint_root: str | Path,
    canonical_checkpoint_path: str | Path | None = None,
) -> TrainingSweepResult:
    """Train and evaluate one model over an explicit list of trial settings."""

    log_root_path = Path(log_root)
    checkpoint_root_path = Path(checkpoint_root)
    log_root_path.mkdir(parents=True, exist_ok=True)
    checkpoint_root_path.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, str | int | float]] = []
    best_selection_score = float("-inf")
    best_trial: TrainingTrialConfig | None = None
    best_row: dict[str, str | int | float] | None = None
    best_result: TrainingResult | None = None
    best_model: nn.Module | None = None
    best_test_metrics: dict[str, float | list[list[int]]] | None = None
    best_test_logits = torch.empty(0)
    best_test_labels = torch.empty(0, dtype=torch.long)
    best_checkpoint: Path | None = None

    for trial in trials:
        model = model_factory()
        log_dir = log_root_path / f"{model_name}__{trial.name}"
        checkpoint_path = checkpoint_root_path / f"{model_name}__{trial.name}.pt"

        train_loader = create_loader(train_dataset, batch_size=trial.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = create_loader(val_dataset, batch_size=trial.batch_size, shuffle=False, num_workers=num_workers)
        test_loader = create_loader(test_dataset, batch_size=trial.batch_size, shuffle=False, num_workers=num_workers)

        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=trial.epochs,
            learning_rate=trial.learning_rate,
            weight_decay=trial.weight_decay,
            selection_metric=selection_metric,
            log_dir=log_dir,
            checkpoint_path=checkpoint_path,
        )
        test_metrics, test_logits, test_labels = evaluate_model(model.to(device), test_loader, device)
        val_score = _resolve_selection_value(result.best_val_metrics, selection_metric)
        row = {
            "trial_name": trial.name,
            "epochs": trial.epochs,
            "batch_size": trial.batch_size,
            "learning_rate": float(trial.learning_rate),
            "weight_decay": float(trial.weight_decay),
            f"val_{selection_metric}": val_score,
            "val_accuracy": float(result.best_val_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "checkpoint_path": str(checkpoint_path),
            "log_dir": str(log_dir),
        }
        trial_rows.append(row)

        if val_score > best_selection_score:
            best_selection_score = val_score
            best_trial = trial
            best_row = row
            best_result = result
            best_model = model
            best_test_metrics = test_metrics
            best_test_logits = test_logits
            best_test_labels = test_labels
            best_checkpoint = result.best_checkpoint if result.best_checkpoint is not None else checkpoint_path

    if best_trial is None or best_row is None or best_result is None or best_model is None or best_test_metrics is None:
        raise ValueError(f"No training trials were executed for model '{model_name}'")

    trial_rows.sort(
        key=lambda row: (
            float(row[f"val_{selection_metric}"]),
            float(row["test_macro_f1"]),
            -float(row["weight_decay"]),
        ),
        reverse=True,
    )

    if canonical_checkpoint_path is not None:
        canonical_path = Path(canonical_checkpoint_path)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        source_checkpoint = best_checkpoint
        if source_checkpoint is not None and source_checkpoint.exists():
            shutil.copy2(source_checkpoint, canonical_path)
        else:
            torch.save(best_model.state_dict(), canonical_path)
        best_checkpoint = canonical_path

    return TrainingSweepResult(
        trial_rows=trial_rows,
        best_trial=best_trial,
        best_trial_row=best_row,
        best_result=best_result,
        best_model=best_model,
        best_test_metrics=best_test_metrics,
        best_test_logits=best_test_logits,
        best_test_labels=best_test_labels,
        best_checkpoint=best_checkpoint,
    )
