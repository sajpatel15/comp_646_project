from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from ..artifacts import read_json, write_json

QWEN_PROMPT = """You are a strict vision-language judge.
Read the image and the caption, then classify the pair into exactly one label:
- contradiction
- neutral
- entailment

Return JSON only in this format:
{"label": "<contradiction|neutral|entailment>", "rationale": "<one short sentence>"}"""


def load_qwen_components(
    model_name: str,
    use_4bit: bool = True,
    device_map: str = "auto",
):
    from transformers import AutoProcessor

    model_kwargs: dict[str, Any] = {"device_map": device_map}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(model_name)

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    except Exception:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, processor


def _extract_json_block(text: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in model output.")
    payload = json.loads(match.group(0))
    label = str(payload["label"]).strip().lower()
    rationale = str(payload.get("rationale", "")).strip()
    if label not in {"contradiction", "neutral", "entailment"}:
        raise ValueError(f"Invalid label returned by Qwen: {label}")
    return {"label": label, "rationale": rationale}


def _predict_single(model, processor, image_path: str, caption: str, max_new_tokens: int = 96) -> dict[str, str]:
    image = Image.open(image_path).convert("RGB")
    prompt = f"{QWEN_PROMPT}\n\nCaption: {caption}"
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {
        name: tensor.to(model_device) if hasattr(tensor, "to") else tensor
        for name, tensor in inputs.items()
    }
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return _extract_json_block(decoded)


def build_qwen_predictions(
    frame: pd.DataFrame,
    model,
    processor,
    cache_path: str | Path | None = None,
    max_new_tokens: int = 96,
) -> pd.DataFrame:
    cache: dict[str, dict[str, str]] = {}
    if cache_path is not None and Path(cache_path).exists():
        cache = read_json(cache_path)

    prediction_rows = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Running Qwen"):
        sample_id = row["sample_id"]
        if sample_id in cache:
            payload = cache[sample_id]
        else:
            payload = _predict_single(
                model=model,
                processor=processor,
                image_path=row["image_path"],
                caption=row["edited_caption"],
                max_new_tokens=max_new_tokens,
            )
            cache[sample_id] = payload
            if cache_path is not None:
                write_json(cache_path, cache)

        prediction_rows.append(
            {
                "sample_id": sample_id,
                "pred_label": payload["label"],
                "qwen_rationale": payload.get("rationale", ""),
                "raw_output_ref": str(cache_path) if cache_path is not None else "",
            }
        )

    return pd.DataFrame(prediction_rows)
