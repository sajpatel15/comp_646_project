"""Model helpers for CLIP baselines, cross-attention fusion, and Qwen inference."""

from .clip_baselines import (
    LateFusionProbe,
    attach_similarity_scores,
    build_probe_features,
    build_text_and_image_feature_maps,
    fit_similarity_thresholds,
    load_clip_components,
    predict_with_thresholds,
)
from .cross_attention import (
    CrossAttentionFusionClassifier,
    build_cross_attention_loader,
    predict_cross_attention,
    train_cross_attention_model,
)
from .qwen_vl import build_qwen_predictions, load_qwen_components

__all__ = [
    "CrossAttentionFusionClassifier",
    "LateFusionProbe",
    "attach_similarity_scores",
    "build_cross_attention_loader",
    "build_probe_features",
    "build_qwen_predictions",
    "build_text_and_image_feature_maps",
    "fit_similarity_thresholds",
    "load_clip_components",
    "load_qwen_components",
    "predict_with_thresholds",
    "predict_cross_attention",
    "train_cross_attention_model",
]
