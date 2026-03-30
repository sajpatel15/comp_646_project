from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm


def load_clip_components(model_name: str, device: str | torch.device):
    from transformers import AutoProcessor, CLIPModel

    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def _open_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return array / norms


def _encode_images(
    model,
    processor,
    image_paths: list[str],
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[start : start + batch_size]
        images = [_open_image(path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        embeddings.append(features.cpu().numpy())
    return _normalize_rows(np.concatenate(embeddings, axis=0))


def _encode_texts(
    model,
    processor,
    captions: list[str],
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    for start in tqdm(range(0, len(captions), batch_size), desc="Encoding texts"):
        batch_text = captions[start : start + batch_size]
        inputs = processor(text=batch_text, padding=True, truncation=True, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        embeddings.append(features.cpu().numpy())
    return _normalize_rows(np.concatenate(embeddings, axis=0))


def build_text_and_image_feature_maps(
    frame: pd.DataFrame,
    model,
    processor,
    device: str | torch.device,
    batch_size: int = 16,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    unique_images = sorted(frame["image_path"].unique().tolist())
    unique_texts = sorted(frame["edited_caption"].unique().tolist())
    image_embeddings = _encode_images(model, processor, unique_images, device, batch_size)
    text_embeddings = _encode_texts(model, processor, unique_texts, device, batch_size)
    image_map = {path: embedding for path, embedding in zip(unique_images, image_embeddings)}
    text_map = {text: embedding for text, embedding in zip(unique_texts, text_embeddings)}
    return image_map, text_map


def attach_similarity_scores(
    frame: pd.DataFrame,
    image_map: dict[str, np.ndarray],
    text_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    scored = frame.copy()
    scores = []
    for _, row in scored.iterrows():
        image_embedding = image_map[row["image_path"]]
        text_embedding = text_map[row["edited_caption"]]
        scores.append(float(np.dot(image_embedding, text_embedding)))
    scored["raw_score"] = scores
    return scored


def predict_with_thresholds(scores: np.ndarray, tau_low: float, tau_high: float) -> list[str]:
    predictions = []
    for score in scores:
        if score < tau_low:
            predictions.append("contradiction")
        elif score < tau_high:
            predictions.append("neutral")
        else:
            predictions.append("entailment")
    return predictions


def fit_similarity_thresholds(scores: np.ndarray, labels: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    candidate_values = np.quantile(scores, np.linspace(0.1, 0.9, 25))
    best = {"tau_low": float(candidate_values[0]), "tau_high": float(candidate_values[-1]), "macro_f1": -1.0}
    sweep_rows = []
    threshold_id = 0
    for tau_low in candidate_values:
        for tau_high in candidate_values:
            if tau_low >= tau_high:
                continue
            predictions = predict_with_thresholds(scores, float(tau_low), float(tau_high))
            macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
            threshold_id += 1
            sweep_rows.append(
                {
                    "threshold_id": threshold_id,
                    "tau_low": float(tau_low),
                    "tau_high": float(tau_high),
                    "macro_f1": float(macro_f1),
                }
            )
            if macro_f1 > best["macro_f1"]:
                best = {"tau_low": float(tau_low), "tau_high": float(tau_high), "macro_f1": float(macro_f1)}
    return best, pd.DataFrame(sweep_rows)


def build_probe_features(
    frame: pd.DataFrame,
    image_map: dict[str, np.ndarray],
    text_map: dict[str, np.ndarray],
) -> torch.Tensor:
    rows = []
    for _, row in frame.iterrows():
        image_embedding = image_map[row["image_path"]]
        text_embedding = text_map[row["edited_caption"]]
        cosine = np.array([np.dot(image_embedding, text_embedding)], dtype=np.float32)
        combined = np.concatenate(
            [
                image_embedding.astype(np.float32),
                text_embedding.astype(np.float32),
                np.abs(image_embedding - text_embedding).astype(np.float32),
                cosine,
            ]
        )
        rows.append(combined)
    return torch.tensor(np.stack(rows), dtype=torch.float32)


class LateFusionProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
