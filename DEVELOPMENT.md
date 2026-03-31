# Development Guide

## Scope

This repo centers on one end-to-end notebook, `notebooks/multimodal_contradiction_project.ipynb`, backed by helper modules in `src/vl_contradiction`. The current workflow is:

1. Bootstrap the repo and editable install.
2. Resolve runtime paths from `configs/default.yaml`.
3. Download or reuse COCO assets.
4. Build the contradiction benchmark and split by family.
5. Run the audit gate.
6. Execute prototype baselines: raw CLIP, linear probe, Qwen subset, cross-attention.
7. Export metrics, figures, and cached Qwen outputs under `artifacts/`.

## Recommended Environment Setup

### Minimal local dev setup

Use this when you only need lightweight code edits, notebook JSON checks, plotting work, or the smaller unit tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Full local runtime setup

Use this when you want the notebook stack locally, including plotting, sklearn metrics, and model code.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Local Qwen setup

The package exposes Qwen support as an optional extra in `pyproject.toml`.

```bash
pip install -e .[qwen]
```

Notes:

- `configs/default.yaml` enables `use_qwen_4bit: true`.
- That path relies on `bitsandbytes`, which is generally easiest to run in Colab or a Linux GPU environment.
- The notebook bootstrap cell now installs `-e .[qwen]` automatically when running in Colab.

## Running The Notebook

### Local

From the repo root:

```bash
jupyter lab
```

Open `notebooks/multimodal_contradiction_project.ipynb` and run from the top.

Local runs resolve the project root from the current working directory and store artifacts under the repo-local paths from `configs/default.yaml`.

### Colab

The notebook is designed to:

- clone or fast-forward the repo into `/content/project`
- mount Google Drive when `auto_mount_drive` is enabled
- keep COCO image files on Colab disk
- write smaller derived artifacts to the configured Drive root

Current defaults from `configs/default.yaml`:

- Drive artifact root: `/content/drive/MyDrive/comp646_multimodal_contradiction`
- Colab dataset root: `/content/comp646_datasets/coco2017`
- Qwen model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Qwen subset size: `600`

The notebook currently sets:

- `RUN_RAW_CLIP = True`
- `RUN_LINEAR_PROBE = True`
- `RUN_CROSS_ATTENTION = True`
- `RUN_QWEN = True`
- `CURRENT_STAGE = "prototype"`

## Testing

### Fast tests

These are the quickest checks to run after targeted code changes:

```bash
python -m unittest tests.test_vl_contradiction_plotting
python -m unittest tests.test_vl_contradiction_benchmark
```

### Audit tests

```bash
python -m unittest tests.test_vl_contradiction_audit
```

Important detail:

- `tests.test_vl_contradiction_audit` imports `vl_contradiction.metrics`.
- `src/vl_contradiction/metrics.py` imports `torch`.
- If `torch` is missing, the audit suite will fail at import time even though the audit logic itself is lightweight.

### JSON and syntax checks

Useful cheap validations after notebook or helper edits:

```bash
python -m json.tool notebooks/multimodal_contradiction_project.ipynb >/dev/null
python -m py_compile src/vl_contradiction/plotting.py
```

## Important Repo Conventions

- Prefer editing `src/vl_contradiction` for the current notebook pipeline.
- Keep config semantics in sync with `configs/default.yaml`.
- Keep artifact writes out of version control unless the user explicitly requests committed outputs.
- When improving notebook figures, verify both:
  - the saved file layout
  - the inline `show_saved_figure(...)` rendering path
- Family-level benchmark splits are part of the leakage-prevention design. Do not casually change the split strategy without updating the benchmark logic and the docs.

## Artifacts And Outputs

Resolved from `src/vl_contradiction/runtime.py` and the config:

- `artifacts/datasets/coco2017`
- `artifacts/benchmark`
- `artifacts/checkpoints`
- `artifacts/logs`
- `artifacts/metrics`
- `artifacts/figures`
- `artifacts/qwen`

## Known Practical Constraints

- The notebook is the main deliverable, so readability matters.
- Qwen inference is intentionally cached one sample per JSON file under `artifacts/qwen/<stage>`.
- The readiness gate expects Qwen metrics when `audit.require_qwen_for_readiness` is true.
- Plotting code lives in helpers, but some figure assembly still happens directly in the notebook. When making notebook visualization changes, check both locations before deciding where the fix belongs.
