"""Training helpers for learned CLIP baselines and fusion models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    log_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train a classifier and log one compact line per epoch."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=str(log_dir)) if log_dir else None
    history = []
    best_metrics: dict[str, float | list[list[int]]] = {"macro_f1": -1.0}
    best_checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
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
        epoch_row = {"epoch": float(epoch), "train_loss": train_loss, "val_macro_f1": float(val_metrics["macro_f1"])}
        history.append(epoch_row)
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}")
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/macro_f1", float(val_metrics["macro_f1"]), epoch)
        if float(val_metrics["macro_f1"]) > float(best_metrics["macro_f1"]):
            best_metrics = val_metrics
            best_logits = val_logits
            best_labels = val_labels
            if best_checkpoint_path:
                best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_checkpoint_path)

    if writer:
        writer.close()

    return TrainingResult(
        history=history,
        best_val_metrics=best_metrics,
        best_checkpoint=best_checkpoint_path if best_checkpoint_path and best_checkpoint_path.exists() else None,
        val_logits=best_logits,
        val_labels=best_labels,
    )
