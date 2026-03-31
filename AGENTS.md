# Agent Guide

## Purpose

This repository builds and evaluates a COCO-derived multimodal contradiction benchmark with a single primary notebook plus helper modules for data generation, training, evaluation, plotting, and Qwen-VL inference.

## Source Of Truth

- Primary workflow notebook: `notebooks/multimodal_contradiction_project.ipynb`
- Main Python package used by the notebook: `src/vl_contradiction`
- Config used by the notebook: `configs/default.yaml`
- Project implementation notes: `IMPLEMENTATION_PLAN.md`

There is also a `src/multimodal_contradiction` tree in the repo. The current notebook imports `vl_contradiction`, so prefer updating `src/vl_contradiction` unless you confirm the other package is still in use.

## Repo Map

- `src/vl_contradiction/benchmark.py`: benchmark construction and Qwen subset sampling
- `src/vl_contradiction/coco.py`: COCO download and caption-context loading
- `src/vl_contradiction/audit.py`: audit summaries and readiness gate
- `src/vl_contradiction/audit_ui.py`: interactive audit review UI
- `src/vl_contradiction/clip_baselines.py`: raw CLIP scoring, feature extraction, threshold search
- `src/vl_contradiction/models.py`: linear probe and cross-attention models
- `src/vl_contradiction/training.py`: dataloaders, training loop, evaluation
- `src/vl_contradiction/qwen.py`: Qwen2.5-VL loading, prompting, caching, parsing
- `src/vl_contradiction/plotting.py`: report-style figure export helpers
- `tests/`: lightweight unit coverage for benchmark, audit, and plotting helpers

## Working Rules

- Keep the notebook readable. Push heavy logic into `src/vl_contradiction` helpers when cells start getting dense.
- Preserve the config-driven runtime layout from `configs/default.yaml` and `src/vl_contradiction/runtime.py`.
- Stage outputs are now scoped under stage-specific subdirectories such as `artifacts/metrics/prototype` and `artifacts/figures/final`, while the COCO dataset cache remains shared.
- Treat the notebook as Colab-compatible first. The bootstrap cell now installs the `qwen` extra in Colab so `RUN_QWEN = True` works without a second manual install.
- Avoid writing derived artifacts into the repo unless the user explicitly wants committed outputs. Runtime artifacts are expected under `artifacts/`.
- When editing figures, check both the saved-figure layout and the inline notebook display path.

## Fast Validation

- Minimal plotting and benchmark checks:
  - `python -m unittest tests.test_vl_contradiction_plotting`
  - `python -m unittest tests.test_vl_contradiction_benchmark`
- Audit tests import `vl_contradiction.metrics`, which currently pulls in `torch`, so they require the full runtime dependency set.

## Dependency Notes

- Base editable install: `pip install -e .`
- Local Qwen-capable install: `pip install -e .[qwen]`
- `requirements.txt` includes a broader notebook/runtime stack, including `bitsandbytes`.
- Qwen 4-bit inference is configured by default and is most practical in Colab or a Linux GPU environment.

## Common Paths

- Benchmark CSVs: `artifacts/benchmark/<stage>`
- Metrics JSON/CSVs: `artifacts/metrics/<stage>`
- Figures: `artifacts/figures/<stage>`
- Checkpoints: `artifacts/checkpoints/<stage>`
- Qwen cache/output JSON: `artifacts/qwen/<stage>`
- TensorBoard logs: `artifacts/logs/<stage>`
