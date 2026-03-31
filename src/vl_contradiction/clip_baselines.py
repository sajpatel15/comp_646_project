"""CLIP scoring and feature extraction utilities."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel


CLASS_ORDER = ["contradiction", "neutral", "entailment"]


@dataclass(slots=True)
class ClipBundle:
    model: CLIPModel
    processor: AutoProcessor
    device: torch.device
    precision: str = "fp32"
    autocast_dtype: torch.dtype | None = None
    num_workers: int = 0
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    extraction_cache: dict[str, "ClipSplitOutputs"] = field(default_factory=dict, repr=False, compare=False)


@dataclass(slots=True)
class _ClipSample:
    sample_id: str
    label: str
    caption: str
    image: Image.Image


@dataclass(slots=True)
class ClipSplitOutputs:
    sample_ids: list[str]
    labels: list[str]
    raw_scores: torch.Tensor
    joint_features: torch.Tensor
    image_tokens: torch.Tensor
    text_tokens: torch.Tensor


def _resolve_clip_precision(precision: str, device: torch.device) -> tuple[str, torch.dtype | None, torch.dtype | None]:
    normalized = precision.strip().lower()
    if normalized not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported CLIP precision '{precision}'")
    if device.type != "cuda":
        return "fp32", None, None
    if normalized == "auto":
        normalized = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if normalized == "bf16" and not torch.cuda.is_bf16_supported():
        normalized = "fp16"
    if normalized == "fp32":
        return normalized, None, None
    autocast_dtype = torch.bfloat16 if normalized == "bf16" else torch.float16
    return normalized, autocast_dtype, autocast_dtype


def _resolve_num_workers(num_workers: int | None, device: torch.device) -> int:
    if num_workers is not None:
        return max(0, int(num_workers))
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count // 2))


def _resolve_loader_options(bundle: ClipBundle) -> dict[str, Any]:
    persistent_workers = bundle.persistent_workers and bundle.num_workers > 0
    options: dict[str, Any] = {
        "num_workers": bundle.num_workers,
        "pin_memory": bundle.device.type == "cuda",
        "persistent_workers": persistent_workers,
    }
    if bundle.num_workers > 0 and bundle.prefetch_factor is not None:
        options["prefetch_factor"] = bundle.prefetch_factor
    return options


def _open_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _clip_batch_collate(batch: list[_ClipSample]) -> list[_ClipSample]:
    return batch


class _ClipDataset(Dataset):
    def __init__(self, records: pd.DataFrame) -> None:
        self._records = records.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> _ClipSample:
        row = self._records.iloc[index]
        return _ClipSample(
            sample_id=str(row["sample_id"]),
            label=str(row["label"]),
            caption=str(row["edited_caption"]),
            image=_open_image(row["file_path"]),
        )


def _frame_fingerprint(records: pd.DataFrame) -> str:
    subset = records.loc[:, ["sample_id", "file_path", "edited_caption"]].astype(str)
    hashed = pd.util.hash_pandas_object(subset, index=False).to_numpy()
    return hashlib.sha1(hashed.tobytes()).hexdigest()


def _clip_cache_key(records: pd.DataFrame, bundle: ClipBundle) -> str:
    model_name = str(getattr(bundle.model, "name_or_path", type(bundle.model).__name__))
    return "::".join(
        [
            model_name,
            bundle.device.type,
            bundle.precision,
            _frame_fingerprint(records),
        ]
    )


def _clip_autocast_context(bundle: ClipBundle):
    if bundle.device.type != "cuda" or bundle.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=bundle.autocast_dtype)


def _to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device, non_blocking=device.type == "cuda")
    return {name: tensor.to(device, non_blocking=device.type == "cuda") for name, tensor in inputs.items()}


def load_clip_bundle(
    model_name: str,
    device: torch.device,
    precision: str = "auto",
    num_workers: int | None = None,
    *,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> ClipBundle:
    """Load a CLIP model and processor onto the target device."""

    resolved_device = torch.device(device)
    resolved_precision, autocast_dtype, model_dtype = _resolve_clip_precision(precision, resolved_device)
    processor = AutoProcessor.from_pretrained(model_name)
    model_kwargs: dict[str, Any] = {}
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype
    model = CLIPModel.from_pretrained(model_name, **model_kwargs).to(resolved_device)
    model.eval()
    resolved_num_workers = _resolve_num_workers(num_workers, resolved_device)
    resolved_persistent_workers = (
        resolved_device.type == "cuda" and resolved_num_workers > 0
        if persistent_workers is None
        else bool(persistent_workers) and resolved_num_workers > 0
    )
    resolved_prefetch_factor = prefetch_factor if resolved_persistent_workers else None
    return ClipBundle(
        model=model,
        processor=processor,
        device=resolved_device,
        precision=resolved_precision,
        autocast_dtype=autocast_dtype,
        num_workers=resolved_num_workers,
        persistent_workers=resolved_persistent_workers,
        prefetch_factor=resolved_prefetch_factor,
    )


def _extract_clip_split_outputs(records: pd.DataFrame, bundle: ClipBundle, batch_size: int = 8) -> ClipSplitOutputs:
    cache_key = _clip_cache_key(records, bundle)
    cached = bundle.extraction_cache.get(cache_key)
    if cached is not None:
        return cached

    loader = DataLoader(
        _ClipDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_clip_batch_collate,
        **_resolve_loader_options(bundle),
    )

    sample_ids: list[str] = []
    labels: list[str] = []
    raw_scores: list[torch.Tensor] = []
    joint_features: list[torch.Tensor] = []
    image_tokens: list[torch.Tensor] = []
    text_tokens: list[torch.Tensor] = []

    model = bundle.model
    with torch.inference_mode():
        for batch in tqdm(loader, desc="CLIP features"):
            images = [item.image for item in batch]
            captions = [item.caption for item in batch]
            max_text_length = int(bundle.model.config.text_config.max_position_embeddings)
            model_inputs = bundle.processor(
                text=captions,
                images=images,
                padding="max_length",
                truncation=True,
                max_length=max_text_length,
                return_tensors="pt",
            )
            model_inputs = _to_device(model_inputs, bundle.device)
            with _clip_autocast_context(bundle):
                outputs = model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
            text_embeds = F.normalize(outputs.text_embeds, dim=-1)
            scores = (image_embeds * text_embeds).sum(dim=-1, keepdim=True)

            sample_ids.extend(item.sample_id for item in batch)
            labels.extend(item.label for item in batch)
            raw_scores.append(scores.detach().cpu().squeeze(-1))
            joint_features.append(torch.cat([image_embeds, text_embeds, scores], dim=-1).detach().cpu())
            image_tokens.append(outputs.vision_model_output.last_hidden_state.detach().cpu())
            text_tokens.append(outputs.text_model_output.last_hidden_state.detach().cpu())

    result = ClipSplitOutputs(
        sample_ids=sample_ids,
        labels=labels,
        raw_scores=torch.cat(raw_scores, dim=0),
        joint_features=torch.cat(joint_features, dim=0),
        image_tokens=torch.cat(image_tokens, dim=0),
        text_tokens=torch.cat(text_tokens, dim=0),
    )
    bundle.extraction_cache[cache_key] = result
    return result


def extract_clip_split_outputs(records: pd.DataFrame, bundle: ClipBundle, batch_size: int = 8) -> ClipSplitOutputs:
    """Extract and cache a full split of CLIP outputs in one pass."""

    return _extract_clip_split_outputs(records, bundle, batch_size=batch_size)


def compute_similarity_scores(records: pd.DataFrame, bundle: ClipBundle, batch_size: int = 8) -> pd.DataFrame:
    """Compute cosine similarity scores for image-caption pairs."""

    outputs = extract_clip_split_outputs(records, bundle, batch_size=batch_size)
    return pd.DataFrame(
        {
            "sample_id": outputs.sample_ids,
            "label": outputs.labels,
            "raw_score": outputs.raw_scores.numpy().astype(float),
        }
    )


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

    outputs = extract_clip_split_outputs(records, bundle, batch_size=batch_size)
    labels = torch.tensor([CLASS_ORDER.index(label) for label in outputs.labels], dtype=torch.long)
    return outputs.joint_features, labels


def extract_token_features(records: pd.DataFrame, bundle: ClipBundle, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract frozen CLIP token states for cross-attention models."""

    outputs = extract_clip_split_outputs(records, bundle, batch_size=batch_size)
    labels = torch.tensor([CLASS_ORDER.index(label) for label in outputs.labels], dtype=torch.long)
    return outputs.image_tokens, outputs.text_tokens, labels
