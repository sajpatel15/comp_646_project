"""Qwen2.5-VL inference helpers with simple on-disk caching."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig


DEFAULT_QWEN_PROMPT = """You are evaluating whether a caption matches an image.
Return strict JSON with keys "label" and "rationale".
The label must be exactly one of: contradiction, neutral, entailment.
Caption: {caption}
"""


@dataclass(slots=True)
class QwenBundle:
    model: Any
    processor: AutoProcessor
    device: torch.device


def _resolve_qwen_model_cls():
    import transformers

    for class_name in ("Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        if hasattr(transformers, class_name):
            return getattr(transformers, class_name)
    raise ImportError("No compatible Qwen vision-language model class found in transformers.")


def load_qwen_bundle(model_name: str, use_4bit: bool = True) -> QwenBundle:
    """Load Qwen2.5-VL with optional 4-bit quantization."""

    model_cls = _resolve_qwen_model_cls()
    processor = AutoProcessor.from_pretrained(model_name)
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if use_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    model = model_cls.from_pretrained(model_name, **kwargs)
    device = next(model.parameters()).device
    return QwenBundle(model=model, processor=processor, device=device)


def _cache_path(output_dir: Path, sample_id: str) -> Path:
    safe_id = sample_id.replace("/", "_")
    return output_dir / f"{safe_id}.json"


def _build_inputs(bundle: QwenBundle, caption: str, image: Image.Image) -> dict[str, torch.Tensor]:
    prompt = DEFAULT_QWEN_PROMPT.format(caption=caption)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = bundle.processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    return {key: value.to(bundle.device) if hasattr(value, "to") else value for key, value in inputs.items()}


def parse_qwen_output(raw_text: str) -> dict[str, str]:
    """Parse strict JSON or fall back to keyword extraction."""

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = str(payload.get("label", "")).strip().lower()
            rationale = str(payload.get("rationale", "")).strip()
            if label in {"contradiction", "neutral", "entailment"}:
                return {"label": label, "rationale": rationale}
        except json.JSONDecodeError:
            pass
    lowered = raw_text.lower()
    for label in ("contradiction", "neutral", "entailment"):
        if label in lowered:
            return {"label": label, "rationale": raw_text.strip()}
    return {"label": "unparsed", "rationale": raw_text.strip()}


def run_qwen_inference(
    records: pd.DataFrame,
    bundle: QwenBundle,
    output_dir: str | Path,
    max_new_tokens: int = 96,
) -> pd.DataFrame:
    """Run Qwen on a fixed subset and cache every raw response."""

    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in tqdm(records.iterrows(), total=len(records), desc="Qwen inference"):
        cache_path = _cache_path(cache_dir, row["sample_id"])
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            rows.append(payload)
            continue
        image = Image.open(row["file_path"]).convert("RGB")
        inputs = _build_inputs(bundle, row["edited_caption"], image)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, prompt_length:]
        raw_text = bundle.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        parsed = parse_qwen_output(raw_text)
        payload = {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "pred_label": parsed["label"],
            "rationale": parsed["rationale"],
            "raw_output": raw_text,
            "runtime_ms": elapsed_ms,
        }
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(payload)
    return pd.DataFrame(rows)
