"""Qwen2.5-VL inference helpers with simple on-disk caching."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
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

_VALID_LABELS = {"contradiction", "neutral", "entailment"}
_VALID_PRECISIONS = {"auto", "bf16", "fp16", "fp32", "4bit"}
_VALID_CACHE_MODES = {"direct", "scratch_then_sync"}


@dataclass(slots=True)
class QwenRuntimePolicy:
    """Resolved runtime knobs for Qwen inference."""

    profile_name: str = "manual"
    precision: str | None = None
    batch_size: int | None = None
    compatibility_mode: bool = False
    cache_mode: str = "direct"
    cache_flush_every: int = 32
    scratch_root: Path | None = None
    use_4bit: bool = True


@dataclass(slots=True)
class QwenBundle:
    model: Any
    processor: AutoProcessor
    device: torch.device
    policy: QwenRuntimePolicy = field(default_factory=QwenRuntimePolicy)


def _resolve_qwen_model_cls():
    import transformers

    for class_name in ("Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        if hasattr(transformers, class_name):
            return getattr(transformers, class_name)
    raise ImportError("No compatible Qwen vision-language model class found in transformers.")


def _get_setting(source: Any | None, *names: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        for name in names:
            if name in source:
                value = source[name]
                if value is not None:
                    return value
        return default
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value
    return default


def _normalize_lower(value: Any | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _coerce_path(value: Any | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _cuda_supports_bf16() -> bool:
    if not _cuda_available():
        return False
    support_checker = getattr(torch.cuda, "is_bf16_supported", None)
    if support_checker is None:
        return False
    try:
        return bool(support_checker())
    except Exception:
        return False


def _cuda_total_memory_gb() -> float:
    if not _cuda_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return 0.0
    return float(getattr(props, "total_memory", 0.0)) / float(1024**3)


def _normalize_precision(value: Any | None) -> str | None:
    normalized = _normalize_lower(value)
    if normalized is None:
        return None
    if normalized not in _VALID_PRECISIONS:
        raise ValueError(f"Unsupported Qwen precision '{value}'. Expected one of: {sorted(_VALID_PRECISIONS)}")
    return normalized


def _normalize_cache_mode(value: Any | None) -> str | None:
    normalized = _normalize_lower(value)
    if normalized is None:
        return None
    if normalized not in _VALID_CACHE_MODES:
        raise ValueError(f"Unsupported Qwen cache mode '{value}'. Expected one of: {sorted(_VALID_CACHE_MODES)}")
    return normalized


def _normalize_policy(
    base: QwenRuntimePolicy | None = None,
    *,
    runtime: Any | None = None,
    performance: Any | None = None,
    use_4bit: bool | None = None,
    precision: Any | None = None,
    batch_size: Any | None = None,
    compatibility_mode: Any | None = None,
    cache_mode: Any | None = None,
    cache_flush_every: Any | None = None,
    scratch_root: Any | None = None,
) -> QwenRuntimePolicy:
    policy = replace(base) if base is not None else QwenRuntimePolicy()

    candidates = (
        (precision, ("precision", "qwen_precision")),
        (batch_size, ("batch_size", "qwen_batch_size")),
        (compatibility_mode, ("compatibility_mode", "qwen_compatibility_mode")),
        (cache_mode, ("cache_mode", "qwen_cache_mode")),
        (cache_flush_every, ("cache_flush_every", "qwen_cache_flush_every")),
        (scratch_root, ("scratch_root", "qwen_scratch_root", "qwen_cache_root")),
        (use_4bit, ("use_4bit", "qwen_use_4bit")),
    )
    for explicit_value, names in candidates:
        value = explicit_value
        if value is None:
            value = _get_setting(performance, *names)
        if value is None:
            value = _get_setting(runtime, *names)
        if value is None:
            continue
        name = names[0]
        if name in {"precision"}:
            policy.precision = _normalize_precision(value)
        elif name in {"batch_size"}:
            policy.batch_size = int(value)
        elif name in {"compatibility_mode"}:
            policy.compatibility_mode = bool(value)
        elif name in {"cache_mode"}:
            policy.cache_mode = _normalize_cache_mode(value) or policy.cache_mode
        elif name in {"cache_flush_every"}:
            policy.cache_flush_every = int(value)
        elif name in {"scratch_root"}:
            policy.scratch_root = _coerce_path(value)
        elif name in {"use_4bit"}:
            policy.use_4bit = bool(value)

    profile_name = _get_setting(performance, "profile_name", "name", "active_profile", default=None)
    if profile_name is None:
        profile_name = _get_setting(runtime, "profile_name", "name", "active_profile", default=None)
    if profile_name is not None:
        policy.profile_name = str(profile_name)

    cache_mode_supplied = (
        cache_mode is not None
        or _get_setting(performance, "cache_mode", "qwen_cache_mode") is not None
        or _get_setting(runtime, "cache_mode", "qwen_cache_mode") is not None
    )
    if policy.scratch_root is not None and not cache_mode_supplied and policy.cache_mode == "direct":
        policy.cache_mode = "scratch_then_sync"

    if policy.cache_flush_every <= 0:
        raise ValueError("cache_flush_every must be a positive integer")
    if policy.batch_size is not None and policy.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer when set")
    return policy


def _resolve_precision_candidates(policy: QwenRuntimePolicy) -> list[str]:
    precision = _normalize_precision(policy.precision)
    if policy.compatibility_mode:
        if _cuda_available():
            return ["4bit"]
        return ["fp32"]
    if precision is None:
        if _cuda_available() and policy.use_4bit:
            return ["4bit"]
        return ["fp32"]
    if precision != "auto":
        if precision == "bf16" and policy.use_4bit and _cuda_available():
            return ["bf16", "fp16", "4bit"]
        if precision == "fp16" and policy.use_4bit and _cuda_available():
            return ["fp16", "4bit"]
        return [precision]

    if not _cuda_available():
        return ["fp32"]

    candidates: list[str] = []
    if _cuda_supports_bf16():
        candidates.append("bf16")
    candidates.append("fp16")
    if policy.use_4bit:
        candidates.append("4bit")
    else:
        candidates.append("fp32")
    return candidates


def _load_model_with_precision(model_cls: Any, model_name: str, policy: QwenRuntimePolicy) -> tuple[Any, str]:
    candidates = _resolve_precision_candidates(policy)
    last_error: Exception | None = None
    for precision in candidates:
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if precision == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if _cuda_supports_bf16() else torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif precision == "fp32":
            kwargs["torch_dtype"] = torch.float32

        try:
            model = model_cls.from_pretrained(model_name, **kwargs)
            model.eval()
            return model, precision
        except Exception as exc:  # pragma: no cover - exercised through monkeypatched tests
            last_error = exc
            if not _is_out_of_memory_error(exc):
                raise
            if len(candidates) == 1:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to load Qwen model")


def load_qwen_bundle(
    model_name: str,
    use_4bit: bool | None = None,
    *,
    precision: str | None = None,
    batch_size: int | None = None,
    compatibility_mode: bool | None = None,
    cache_mode: str | None = None,
    cache_flush_every: int | None = None,
    scratch_root: str | Path | None = None,
    runtime: Any | None = None,
    performance: Any | None = None,
) -> QwenBundle:
    """Load Qwen2.5-VL with optional quantization and runtime policy hints."""

    model_cls = _resolve_qwen_model_cls()
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side") and getattr(processor, "padding_side") != "left":
        processor.padding_side = "left"
    policy = _normalize_policy(
        runtime=runtime,
        performance=performance,
        use_4bit=use_4bit,
        precision=precision,
        batch_size=batch_size,
        compatibility_mode=compatibility_mode,
        cache_mode=cache_mode,
        cache_flush_every=cache_flush_every,
        scratch_root=scratch_root,
    )
    if policy.compatibility_mode:
        policy.cache_mode = "direct"
        policy.batch_size = 1
        policy.precision = "4bit" if _cuda_available() else "fp32"
    model, resolved_precision = _load_model_with_precision(model_cls, model_name, policy)
    policy.precision = resolved_precision
    device = next(model.parameters()).device
    return QwenBundle(model=model, processor=processor, device=device, policy=policy)


def _cache_path(output_dir: Path, sample_id: str) -> Path:
    safe_id = sample_id.replace("/", "_")
    return output_dir / f"{safe_id}.json"


def _resolve_scratch_dir(final_dir: Path, scratch_root: Path) -> Path:
    if scratch_root.name == final_dir.name:
        return scratch_root
    return scratch_root / final_dir.name


def _read_cached_payload(
    sample_id: str,
    final_dir: Path,
    scratch_dir: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    final_path = _cache_path(final_dir, sample_id)
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8")), final_path
    if scratch_dir is not None:
        scratch_path = _cache_path(scratch_dir, sample_id)
        if scratch_path.exists():
            return json.loads(scratch_path.read_text(encoding="utf-8")), scratch_path
    return None, None


def _build_inputs(bundle: QwenBundle, caption: str, image: Image.Image) -> dict[str, torch.Tensor]:
    prompt = DEFAULT_QWEN_PROMPT.format(caption=caption)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = bundle.processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    return {key: value.to(bundle.device, non_blocking=_cuda_available()) if hasattr(value, "to") else value for key, value in inputs.items()}


def _build_batch_inputs(bundle: QwenBundle, captions: list[str], image_paths: list[str | Path]) -> dict[str, torch.Tensor]:
    prompts = []
    images: list[Image.Image] = []
    for caption, image_path in zip(captions, image_paths, strict=True):
        prompt = DEFAULT_QWEN_PROMPT.format(caption=caption)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        prompts.append(bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    inputs = bundle.processor(text=prompts, images=images, padding=True, return_tensors="pt")
    return {key: value.to(bundle.device, non_blocking=_cuda_available()) if hasattr(value, "to") else value for key, value in inputs.items()}


def _build_payload(
    *,
    sample_id: str,
    label: str,
    pred_label: str,
    rationale: str,
    raw_output: str,
    runtime_ms: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "label": label,
        "pred_label": pred_label,
        "rationale": rationale,
        "raw_output": raw_output,
        "runtime_ms": runtime_ms,
    }


def parse_qwen_output(raw_text: str) -> dict[str, str]:
    """Parse strict JSON or fall back to keyword extraction."""

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = str(payload.get("label", "")).strip().lower()
            rationale = str(payload.get("rationale", "")).strip()
            if label in _VALID_LABELS:
                return {"label": label, "rationale": rationale}
        except json.JSONDecodeError:
            pass
    lowered = raw_text.lower()
    for label in ("contradiction", "neutral", "entailment"):
        if label in lowered:
            return {"label": label, "rationale": raw_text.strip()}
    return {"label": "unparsed", "rationale": raw_text.strip()}


def _is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or "cuda error: out of memory" in message


def _inference_context(precision: str | None, device: torch.device):
    normalized = _normalize_precision(precision)
    if device.type != "cuda" or normalized not in {"fp16", "bf16"}:
        return nullcontext()
    autocast_dtype = torch.float16 if normalized == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


def _default_batch_size(bundle: QwenBundle, policy: QwenRuntimePolicy) -> int:
    if policy.compatibility_mode:
        return 1
    if policy.batch_size is not None:
        return policy.batch_size
    if bundle.device.type != "cuda":
        return 1

    total_memory = _cuda_total_memory_gb()
    precision = _normalize_precision(policy.precision)
    if precision is None:
        precision = "4bit" if policy.use_4bit else "fp32"
    if precision in {"fp16", "bf16"}:
        if total_memory < 10.0:
            return 1
        if total_memory < 18.0:
            return 4
        if total_memory < 24.0:
            return 6
        return 8
    if precision == "4bit":
        if total_memory < 10.0:
            return 2
        if total_memory < 18.0:
            return 6
        if total_memory < 24.0:
            return 10
        return 16
    return 1


def _format_eta(seconds: float) -> str:
    if not seconds or seconds == float("inf") or seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _progress_stride(total: int) -> int:
    if total <= 10:
        return 1
    return max(1, total // 10)


def _generate_batch(
    bundle: QwenBundle,
    batch_rows: list[tuple[int, pd.Series]],
    max_new_tokens: int,
    precision: str | None,
) -> list[dict[str, Any]]:
    captions = [str(row["edited_caption"]) for _, row in batch_rows]
    image_paths = [row["file_path"] for _, row in batch_rows]
    inputs = _build_batch_inputs(bundle, captions, image_paths)
    start = time.perf_counter()
    with torch.inference_mode():
        with _inference_context(precision, bundle.device):
            generated = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = generated[:, prompt_length:]
    raw_texts = bundle.processor.batch_decode(generated_tokens, skip_special_tokens=True)
    payloads = []
    for (_, row), raw_text in zip(batch_rows, raw_texts, strict=True):
        parsed = parse_qwen_output(raw_text)
        payloads.append(
            _build_payload(
                sample_id=str(row["sample_id"]),
                label=str(row["label"]),
                pred_label=parsed["label"],
                rationale=parsed["rationale"],
                raw_output=raw_text,
                runtime_ms=elapsed_ms / max(len(batch_rows), 1),
            )
        )
    return payloads


def _write_payload(cache_dir: Path, payload: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_dir, str(payload["sample_id"]))
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cache_path


def _sync_payloads(scratch_paths: list[Path], scratch_dir: Path, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for scratch_path in scratch_paths:
        final_path = final_dir / scratch_path.name
        shutil.copy2(scratch_path, final_path)


def run_qwen_inference(
    records: pd.DataFrame,
    bundle: QwenBundle,
    output_dir: str | Path,
    max_new_tokens: int = 96,
    *,
    precision: str | None = None,
    batch_size: int | None = None,
    compatibility_mode: bool | None = None,
    cache_mode: str | None = None,
    cache_flush_every: int | None = None,
    scratch_root: str | Path | None = None,
    runtime: Any | None = None,
    performance: Any | None = None,
) -> pd.DataFrame:
    """Run Qwen on a fixed subset and cache every raw response."""

    policy = _normalize_policy(
        bundle.policy,
        runtime=runtime,
        performance=performance,
        precision=precision,
        batch_size=batch_size,
        compatibility_mode=compatibility_mode,
        cache_mode=cache_mode,
        cache_flush_every=cache_flush_every,
        scratch_root=scratch_root,
    )
    if policy.compatibility_mode:
        policy.cache_mode = "direct"
        policy.batch_size = 1
        policy.precision = "4bit" if _cuda_available() else "fp32"

    final_dir = Path(output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = None
    if policy.cache_mode == "scratch_then_sync":
        scratch_root = policy.scratch_root or Path(tempfile.gettempdir()) / "vl_contradiction_qwen_cache"
        scratch_dir = _resolve_scratch_dir(final_dir, scratch_root)
        scratch_dir.mkdir(parents=True, exist_ok=True)

    records = records.reset_index(drop=True)
    total_rows = len(records)
    results: list[dict[str, Any] | None] = [None] * len(records)
    cache_hits = 0
    miss_entries: list[tuple[int, pd.Series]] = []
    for row_index, row in records.iterrows():
        cached_payload, cache_path = _read_cached_payload(str(row["sample_id"]), final_dir, scratch_dir)
        if cached_payload is not None:
            results[row_index] = cached_payload
            cache_hits += 1
            if cache_path is not None and scratch_dir is not None and cache_path.parent == scratch_dir:
                shutil.copy2(cache_path, _cache_path(final_dir, str(row["sample_id"])))
        else:
            miss_entries.append((row_index, row))

    if not miss_entries:
        elapsed_seconds = 0.0
        print(
            f"[qwen] profile={policy.profile_name} precision={policy.precision} batch_size={policy.batch_size or 1} "
            f"cache_mode={policy.cache_mode} hits={cache_hits} misses=0 elapsed_s={elapsed_seconds:.2f} samples_per_s=inf"
        )
        return pd.DataFrame([row for row in results if row is not None])

    target_batch_size = _default_batch_size(bundle, policy)
    if policy.compatibility_mode:
        target_batch_size = 1
    elif policy.batch_size is not None:
        target_batch_size = policy.batch_size

    processed_misses = 0
    pending_scratch_paths: list[Path] = []
    run_started = time.perf_counter()
    generation_seconds = 0.0
    cursor = 0
    invocation_count = 0
    progress_stride = _progress_stride(len(miss_entries))
    next_progress = progress_stride

    print(
        f"[qwen] start total={total_rows} cached={cache_hits} run={len(miss_entries)} "
        f"batch={target_batch_size} precision={policy.precision}"
    )

    while cursor < len(miss_entries):
        current_batch_size = min(target_batch_size, len(miss_entries) - cursor)
        batch = miss_entries[cursor : cursor + current_batch_size]
        try:
            if target_batch_size == 1:
                batch_payloads: list[dict[str, Any]] = []
                for batch_row in batch:
                    invocation_count += 1
                    single_started = time.perf_counter()
                    image_path = batch_row[1]["file_path"]
                    with Image.open(image_path) as image:
                        inputs = _build_inputs(bundle, str(batch_row[1]["edited_caption"]), image.convert("RGB"))
                    with torch.inference_mode():
                        with _inference_context(policy.precision, bundle.device):
                            generated = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                    generation_seconds += time.perf_counter() - single_started
                    prompt_length = inputs["input_ids"].shape[1]
                    generated_tokens = generated[:, prompt_length:]
                    raw_text = bundle.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    parsed = parse_qwen_output(raw_text)
                    batch_payloads.append(
                        _build_payload(
                            sample_id=str(batch_row[1]["sample_id"]),
                            label=str(batch_row[1]["label"]),
                            pred_label=parsed["label"],
                            rationale=parsed["rationale"],
                            raw_output=raw_text,
                            runtime_ms=(time.perf_counter() - single_started) * 1000.0,
                        )
                    )
            else:
                invocation_count += 1
                batch_started = time.perf_counter()
                batch_payloads = _generate_batch(bundle, batch, max_new_tokens, policy.precision)
                generation_seconds += time.perf_counter() - batch_started
        except Exception as exc:
            if _is_out_of_memory_error(exc) and current_batch_size > 1:
                if _cuda_available():
                    torch.cuda.empty_cache()
                new_batch_size = max(1, current_batch_size // 2)
                print(f"[qwen] oom batch={current_batch_size} -> {new_batch_size}")
                target_batch_size = new_batch_size
                continue
            raise

        for (result_index, _), payload in zip(batch, batch_payloads, strict=True):
            results[result_index] = payload
            if policy.cache_mode == "direct":
                _write_payload(final_dir, payload)
            else:
                scratch_path = _write_payload(scratch_dir or final_dir, payload)
                pending_scratch_paths.append(scratch_path)
        processed_misses += len(batch_payloads)
        cursor += len(batch_payloads)

        if policy.cache_mode == "scratch_then_sync":
            should_flush = processed_misses % policy.cache_flush_every == 0 or cursor >= len(miss_entries)
            if should_flush and scratch_dir is not None:
                _sync_payloads(pending_scratch_paths, scratch_dir, final_dir)
                pending_scratch_paths.clear()

        if processed_misses >= next_progress or cursor >= len(miss_entries):
            elapsed_so_far = time.perf_counter() - run_started
            rate = processed_misses / max(elapsed_so_far, 1e-9)
            remaining = len(miss_entries) - processed_misses
            eta = remaining / max(rate, 1e-9)
            percent = (processed_misses / max(len(miss_entries), 1)) * 100.0
            print(
                f"[qwen] {processed_misses}/{len(miss_entries)} {percent:.0f}% "
                f"calls={invocation_count} batch={current_batch_size} eta={_format_eta(eta)}"
            )
            while processed_misses >= next_progress:
                next_progress += progress_stride

    elapsed_seconds = time.perf_counter() - run_started
    throughput = total_rows / max(elapsed_seconds, 1e-9)
    generation_throughput = len(miss_entries) / max(generation_seconds, 1e-9)
    print(
        f"[qwen] done profile={policy.profile_name} precision={policy.precision} batch={target_batch_size} "
        f"hits={cache_hits} misses={len(miss_entries)} calls={invocation_count} "
        f"elapsed_s={elapsed_seconds:.2f} samples_per_s={throughput:.2f} model_samples_per_s={generation_throughput:.2f}"
    )
    return pd.DataFrame([row for row in results if row is not None])
