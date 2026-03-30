"""Model and dataset helpers for CLIP-based training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel

from .coco import image_path_for_row
from .evaluation import label_to_id


@dataclass(slots=True)
class ClipBundle:
    model: CLIPModel
    processor: Any
    device: torch.device


class BenchmarkFrameDataset(Dataset):
    """Thin dataset wrapper over the generated benchmark frame."""

    def __init__(self, frame: pd.DataFrame, dataset_root: Path) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "sample_id": row["sample_id"],
            "caption": row["edited_caption"],
            "label_id": label_to_id(row["label"]),
            "label": row["label"],
            "edit_family": row["edit_family"],
            "image_path": image_path_for_row(self.dataset_root, row["source_split"], row["file_name"]),
        }


class FeatureTensorDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class CrossAttentionCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        texts = [item["caption"] for item in batch]
        inputs = self.processor(images=images, text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
        inputs["sample_ids"] = [item["sample_id"] for item in batch]
        return inputs


class LinearProbeClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int = 3) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class TinyMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_labels: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CrossAttentionFusionClassifier(nn.Module):
    """Frozen CLIP encoder with lightweight cross-attention fusion layers."""

    def __init__(
        self,
        clip_model_name: str,
        *,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_labels: int = 3,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        if freeze_clip:
            for parameter in self.clip.parameters():
                parameter.requires_grad = False

        text_hidden = int(self.clip.text_model.config.hidden_size)
        vision_hidden = int(self.clip.vision_model.config.hidden_size)
        self.text_proj = nn.Linear(text_hidden, hidden_dim)
        self.vision_proj = nn.Linear(vision_hidden, hidden_dim)
        self.text_to_vision = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.vision_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)

        text_tokens = self.text_proj(text_outputs.last_hidden_state)
        vision_tokens = self.vision_proj(vision_outputs.last_hidden_state)

        text_fused, _ = self.text_to_vision(query=text_tokens, key=vision_tokens, value=vision_tokens)
        vision_fused, _ = self.vision_to_text(
            query=vision_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=(attention_mask == 0),
        )

        text_fused = self.norm(self.dropout(text_fused) + text_tokens)
        vision_fused = self.norm(self.dropout(vision_fused) + vision_tokens)

        mask = attention_mask.unsqueeze(-1).float()
        text_pool = (text_fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        vision_pool = vision_fused.mean(dim=1)

        image_embeds = F.normalize(self.clip.visual_projection(vision_outputs.pooler_output), dim=-1)
        text_embeds = F.normalize(self.clip.text_projection(text_outputs.pooler_output), dim=-1)
        cosine = (image_embeds * text_embeds).sum(dim=-1, keepdim=True)

        fused = torch.cat([text_pool, vision_pool, cosine], dim=-1)
        return self.classifier(fused)


def load_clip_bundle(model_name: str, device: str | torch.device | None = None) -> ClipBundle:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CLIPModel.from_pretrained(model_name).to(resolved_device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return ClipBundle(model=model, processor=processor, device=resolved_device)


def build_linear_probe_features(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    cosine_scores: np.ndarray,
) -> np.ndarray:
    absolute_difference = np.abs(image_embeddings - text_embeddings)
    elementwise_product = image_embeddings * text_embeddings
    cosine_column = cosine_scores.reshape(-1, 1)
    return np.concatenate(
        [image_embeddings, text_embeddings, absolute_difference, elementwise_product, cosine_column],
        axis=1,
    )


def encode_clip_pairs(
    frame: pd.DataFrame,
    dataset_root: Path,
    clip_bundle: ClipBundle,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    image_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    cosine_scores: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []

    for start in tqdm(range(0, len(frame), batch_size), desc="Encoding CLIP pairs"):
        batch = frame.iloc[start : start + batch_size]
        image_paths = [
            image_path_for_row(dataset_root, row["source_split"], row["file_name"])
            for _, row in batch.iterrows()
        ]
        images = [Image.open(path).convert("RGB") for path in image_paths]
        texts = batch["edited_caption"].tolist()

        inputs = clip_bundle.processor(images=images, text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(clip_bundle.device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = clip_bundle.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = clip_bundle.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        cosine = (image_features * text_features).sum(dim=-1)

        image_embeddings.append(image_features.cpu().numpy())
        text_embeddings.append(text_features.cpu().numpy())
        cosine_scores.append(cosine.cpu().numpy())
        labels.extend(label_to_id(label) for label in batch["label"].tolist())
        sample_ids.extend(batch["sample_id"].tolist())

    image_matrix = np.concatenate(image_embeddings, axis=0)
    text_matrix = np.concatenate(text_embeddings, axis=0)
    cosine_vector = np.concatenate(cosine_scores, axis=0)
    features = build_linear_probe_features(image_matrix, text_matrix, cosine_vector)
    return {
        "features": features,
        "image_embeddings": image_matrix,
        "text_embeddings": text_matrix,
        "cosine_scores": cosine_vector,
        "labels": np.array(labels, dtype=np.int64),
        "sample_ids": np.array(sample_ids, dtype=object),
    }
