"""Training loops for probe and cross-attention experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .evaluation import classification_metrics
from .modeling import LinearProbeClassifier, TinyMLPClassifier


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _save_checkpoint(model: nn.Module, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def _evaluate_feature_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    predictions = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return {
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": classification_metrics(labels, predictions),
    }


def train_probe_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    input_dim: int,
    model_kind: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    tensorboard_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str = "auto",
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:
    resolved_device = resolve_device(device)
    if model_kind == "linear":
        model: nn.Module = LinearProbeClassifier(input_dim=input_dim)
    elif model_kind == "mlp":
        model = TinyMLPClassifier(input_dim=input_dim)
    else:
        raise ValueError(f"Unsupported probe model kind: {model_kind}")

    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if tensorboard_dir else None

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_snapshot: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for features, labels in train_loader:
            features = features.to(resolved_device)
            labels = labels.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        validation = _evaluate_feature_model(model, val_loader, resolved_device)
        val_macro_f1 = float(validation["metrics"]["macro_f1"])
        epoch_summary = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "val_macro_f1": val_macro_f1,
            "val_accuracy": float(validation["metrics"]["accuracy"]),
        }
        history.append(epoch_summary)
        print(
            f"[probe:{model_kind}] epoch {epoch:02d}/{epochs} "
            f"loss={epoch_summary['train_loss']:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"val_acc={epoch_summary['val_accuracy']:.4f}"
        )

        if writer:
            writer.add_scalar("loss/train", epoch_summary["train_loss"], epoch)
            writer.add_scalar("metrics/val_macro_f1", val_macro_f1, epoch)
            writer.add_scalar("metrics/val_accuracy", epoch_summary["val_accuracy"], epoch)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_snapshot = validation
            _save_checkpoint(model, checkpoint_path)

    if writer:
        writer.close()
    return model, history, best_snapshot


def predict_feature_model(model: nn.Module, loader: DataLoader, *, device: str = "auto") -> dict[str, Any]:
    return _evaluate_feature_model(model, loader, resolve_device(device))


def _evaluate_cross_attention_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())

    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    predictions = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return {
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": classification_metrics(labels, predictions),
    }


def train_cross_attention_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    tensorboard_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str = "auto",
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if tensorboard_dir else None

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_snapshot: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            labels = batch["labels"].to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                pixel_values=batch["pixel_values"].to(resolved_device),
                input_ids=batch["input_ids"].to(resolved_device),
                attention_mask=batch["attention_mask"].to(resolved_device),
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        validation = _evaluate_cross_attention_model(model, val_loader, resolved_device)
        val_macro_f1 = float(validation["metrics"]["macro_f1"])
        epoch_summary = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "val_macro_f1": val_macro_f1,
            "val_accuracy": float(validation["metrics"]["accuracy"]),
        }
        history.append(epoch_summary)
        print(
            f"[cross-attn] epoch {epoch:02d}/{epochs} "
            f"loss={epoch_summary['train_loss']:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"val_acc={epoch_summary['val_accuracy']:.4f}"
        )

        if writer:
            writer.add_scalar("loss/train", epoch_summary["train_loss"], epoch)
            writer.add_scalar("metrics/val_macro_f1", val_macro_f1, epoch)
            writer.add_scalar("metrics/val_accuracy", epoch_summary["val_accuracy"], epoch)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_snapshot = validation
            _save_checkpoint(model, checkpoint_path)

    if writer:
        writer.close()
    return model, history, best_snapshot


def predict_cross_attention_model(model: nn.Module, loader: DataLoader, *, device: str = "auto") -> dict[str, Any]:
    return _evaluate_cross_attention_model(model, loader, resolve_device(device))
