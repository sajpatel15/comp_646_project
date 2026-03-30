"""CLIP scoring and feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel


CLASS_ORDER = ["contradiction", "neutral", "entailment"]


@dataclass(slots=True)
class ClipBundle:
    model: CLIPModel
    processor: AutoProcessor
    device: torch.device


def load_clip_bundle(model_name: str, device: torch.device) -> ClipBundle:
    """Load a CLIP model and processor onto the target device."""

    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return ClipBundle(model=model, processor=processor, device=device)


def _open_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def compute_similarity_scores(records: pd.DataFrame, bundle: ClipBundle, batch_size: int = 8) -> pd.DataFrame:
    """Compute cosine similarity scores for image-caption pairs."""

    outputs: list[dict] = []
    for start in tqdm(range(0, len(records), batch_size), desc="CLIP scoring"):
        batch = records.iloc[start : start + batch_size]
        images = [_open_image(path) for path in batch["file_path"]]
        captions = batch["edited_caption"].tolist()
        model_inputs = bundle.processor(text=captions, images=images, padding=True, return_tensors="pt").to(bundle.device)
        with torch.inference_mode():
            outputs = bundle.model(
                **model_inputs,
                return_dict=True,
            )
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
            scores = (image_embeds * text_embeds).sum(dim=-1).detach().cpu().numpy()
        for sample_id, label, score in zip(batch["sample_id"], batch["label"], scores, strict=True):
            outputs.append({"sample_id": sample_id, "label": label, "raw_score": float(score)})
    return pd.DataFrame(outputs)


def fit_similarity_thresholds(labels: list[str], scores: np.ndarray, grid_size: int = 200) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit two thresholds that maximize macro-F1 on validation labels."""

    numeric_labels = np.array([CLASS_ORDER.index(label) for label in labels])
    grid = np.linspace(float(scores.min()), float(scores.max()), num=grid_size)
    best = {"tau_low": float(grid[0]), "tau_high": float(grid[-1]), "macro_f1": -1.0}
    search_rows = []
    for tau_low in grid:
        for tau_high in grid:
            if tau_low >= tau_high:
                continue
            predictions = predict_with_thresholds(scores, tau_low, tau_high)
            macro_f1 = f1_score(numeric_labels, predictions, average="macro")
            search_rows.append({"tau_low": tau_low, "tau_high": tau_high, "macro_f1": macro_f1})
            if macro_f1 > best["macro_f1"]:
                best = {"tau_low": float(tau_low), "tau_high": float(tau_high), "macro_f1": float(macro_f1)}
    return best, pd.DataFrame(search_rows)


def predict_with_thresholds(scores: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    """Map cosine similarity scores onto three classes."""

    predictions = np.full(shape=scores.shape, fill_value=1, dtype=np.int64)
    predictions[scores < tau_low] = 0
    predictions[scores >= tau_high] = 2
    return predictions


def extract_joint_features(records: pd.DataFrame, bundle: ClipBundle, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen CLIP image/text features and append cosine score."""

    feature_rows: list[torch.Tensor] = []
    label_rows: list[int] = []
    for start in tqdm(range(0, len(records), batch_size), desc="CLIP features"):
        batch = records.iloc[start : start + batch_size]
        images = [_open_image(path) for path in batch["file_path"]]
        captions = batch["edited_caption"].tolist()
        model_inputs = bundle.processor(text=captions, images=images, padding=True, return_tensors="pt").to(bundle.device)
        with torch.inference_mode():
            outputs = bundle.model(
                **model_inputs,
                return_dict=True,
            )
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
            scores = (image_embeds * text_embeds).sum(dim=-1, keepdim=True)
            combined = torch.cat([image_embeds, text_embeds, scores], dim=-1).cpu()
        feature_rows.append(combined)
        label_rows.extend(CLASS_ORDER.index(label) for label in batch["label"])
    features = torch.cat(feature_rows, dim=0)
    labels = torch.tensor(label_rows, dtype=torch.long)
    return features, labels


def extract_token_features(records: pd.DataFrame, bundle: ClipBundle, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract frozen CLIP token states for cross-attention models."""

    image_tokens: list[torch.Tensor] = []
    text_tokens: list[torch.Tensor] = []
    labels: list[int] = []
    for start in tqdm(range(0, len(records), batch_size), desc="CLIP token states"):
        batch = records.iloc[start : start + batch_size]
        images = [_open_image(path) for path in batch["file_path"]]
        captions = batch["edited_caption"].tolist()
        model_inputs = bundle.processor(text=captions, images=images, padding=True, return_tensors="pt").to(bundle.device)
        with torch.inference_mode():
            outputs = bundle.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            image_hidden = outputs.vision_model_output.last_hidden_state.cpu()
            text_hidden = outputs.text_model_output.last_hidden_state.cpu()
        image_tokens.append(image_hidden)
        text_tokens.append(text_hidden)
        labels.extend(CLASS_ORDER.index(label) for label in batch["label"])
    return torch.cat(image_tokens, dim=0), torch.cat(text_tokens, dim=0), torch.tensor(labels, dtype=torch.long)
