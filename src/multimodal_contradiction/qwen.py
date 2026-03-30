"""Qwen2.5-VL inference helpers with cached outputs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .coco import image_path_for_row
from .io_utils import ensure_dir, load_json, save_json

QWEN_LABELS = {"entailment", "neutral", "contradiction"}


def build_qwen_prompt(caption: str) -> str:
    return (
        "You are grading whether a caption matches an image.\n"
        "Choose exactly one label from: entailment, neutral, contradiction.\n"
        "Return strict JSON with keys label and rationale.\n"
        f"Caption: {caption}"
    )


def load_qwen_bundle(
    model_name: str,
    *,
    quantized_4bit: bool = True,
    max_pixels: int | None = None,
) -> tuple[Any, Any]:
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as QwenModel  # type: ignore

    processor_kwargs = {}
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if quantized_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["dtype"] = torch.float16

    model = QwenModel.from_pretrained(model_name, **load_kwargs)
    return model, processor


def parse_qwen_response(text: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = str(payload.get("label", "")).strip().lower()
            rationale = str(payload.get("rationale", "")).strip()
            if label in QWEN_LABELS:
                return {"label": label, "rationale": rationale}
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    for label in QWEN_LABELS:
        if label in lowered:
            return {"label": label, "rationale": text.strip()}
    return {"label": "neutral", "rationale": text.strip()}


def _cache_path(cache_root: Path, sample_id: str) -> Path:
    return cache_root / f"{sample_id}.json"


def predict_qwen_row(
    row: pd.Series,
    *,
    dataset_root: Path,
    cache_root: Path,
    model: Any,
    processor: Any,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    ensure_dir(cache_root)
    cache_path = _cache_path(cache_root, str(row["sample_id"]))
    if cache_path.exists():
        return load_json(cache_path)

    image_path = image_path_for_row(dataset_root, row["source_split"], row["file_name"])
    prompt = build_qwen_prompt(str(row["edited_caption"]))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    started = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = parse_qwen_response(output_text)

    payload = {
        "sample_id": str(row["sample_id"]),
        "pred_label": parsed["label"],
        "rationale": parsed["rationale"],
        "raw_output": output_text,
        "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    save_json(payload, cache_path)
    return payload


def predict_qwen_frame(
    frame: pd.DataFrame,
    *,
    dataset_root: Path,
    cache_root: Path,
    model: Any,
    processor: Any,
    max_new_tokens: int = 96,
) -> pd.DataFrame:
    rows = []
    for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
        print(f"[qwen] sample {row_index}/{len(frame)}")
        rows.append(
            predict_qwen_row(
                row,
                dataset_root=dataset_root,
                cache_root=cache_root,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
            )
        )
    return pd.DataFrame(rows)
