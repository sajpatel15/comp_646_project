from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CrossAttentionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, processor, label_to_index: dict[str, int]):
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        encoded = self.processor(
            text=row["edited_caption"],
            images=image,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {name: tensor.squeeze(0) for name, tensor in encoded.items()}
        item["labels"] = torch.tensor(self.label_to_index[row["label"]], dtype=torch.long)
        return item


def build_cross_attention_loader(
    frame: pd.DataFrame,
    processor,
    label_to_index: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CrossAttentionDataset(frame, processor, label_to_index)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CrossAttentionFusionClassifier(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        num_classes: int = 3,
        projection_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        from transformers import CLIPTextModel, CLIPVisionModel

        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)

        if freeze_encoders:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, projection_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim * 3),
            nn.Linear(projection_dim * 3, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vision_tokens = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        text_tokens = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        vision_tokens = self.vision_proj(vision_tokens)
        text_tokens = self.text_proj(text_tokens)

        attended_text, _ = self.cross_attention(text_tokens, vision_tokens, vision_tokens)
        text_mask = attention_mask.unsqueeze(-1).float()
        pooled_text = (attended_text * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0)
        pooled_vision = vision_tokens.mean(dim=1)
        fused = torch.cat([pooled_vision, pooled_text, torch.abs(pooled_vision - pooled_text)], dim=-1)
        return self.classifier(fused)


@dataclass(slots=True)
class CrossAttentionTrainingResult:
    history: pd.DataFrame
    best_state_dict: dict
    best_macro_f1: float


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(logits, labels)
            losses.append(loss.item())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return float(sum(losses) / max(len(losses), 1)), float(f1_score(targets, predictions, average="macro", zero_division=0))


def predict_cross_attention(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], torch.Tensor]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits_list.append(logits.cpu())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return predictions, targets, torch.cat(logits_list, dim=0)


def train_cross_attention_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> CrossAttentionTrainingResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)
    history_rows = []
    best_macro_f1 = -1.0
    best_state_dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss, val_macro_f1 = _evaluate(model, val_loader, device)
        print(
            f"[XATTN] epoch={epoch:02d}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f}"
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
            }
        )
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    return CrossAttentionTrainingResult(
        history=pd.DataFrame(history_rows),
        best_state_dict=best_state_dict,
        best_macro_f1=best_macro_f1,
    )
