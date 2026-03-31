# Development Guide

## Scope

This repo centers on one end-to-end notebook, `notebooks/multimodal_contradiction_project.ipynb`, backed by helper modules in `src/vl_contradiction`. The current workflow is:

1. Bootstrap the repo and editable install.
2. Resolve runtime paths from `configs/default.yaml`.
3. Download or reuse COCO assets.
4. Build the contradiction benchmark and split by family.
5. Run the audit gate.
6. Execute prototype baselines: raw CLIP, linear probe, Qwen subset, cross-attention.
7. Sweep stage-specific hyperparameters for the learned CLIP models, keep the best trial by validation macro-F1, and export metrics, figures, checkpoints, and cached Qwen outputs under `artifacts/`.

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

- `configs/default.yaml` still keeps `use_qwen_4bit: true` as a compatibility fallback, but the default runtime now resolves Qwen precision from the active performance profile.
- The adaptive Qwen path still relies on `bitsandbytes` when it falls back to `4bit`, which is generally easiest to run in Colab or a Linux GPU environment.
- The notebook bootstrap cell now installs `-e .[qwen]` automatically when running in Colab.

## Running The Notebook

### Local

From the repo root:

```bash
jupyter lab
```

Open `notebooks/multimodal_contradiction_project.ipynb` and run from the top.

Local runs resolve the project root from the current working directory and store artifacts under the repo-local paths from `configs/default.yaml`.
Experiment outputs are further separated by stage, so prototype, midscale, and final runs land in different subdirectories under each artifact root.

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

The learned-model hyperparameters now come from `configs/default.yaml`:

- fixed extraction/inference batch sizes:
  - `training.clip_batch_size`
  - `training.joint_feature_batch_size`
  - `training.token_feature_batch_size`
- stage/model sweep lists:
  - `training.sweeps.prototype.linear_probe.trials`
  - `training.sweeps.prototype.cross_attention.trials`
  - `training.sweeps.midscale.*`
  - `training.sweeps.final.*`

The notebook writes one sweep summary CSV and one best-trial JSON per learned model and stage, then keeps the best checkpoint under the existing canonical filename for downstream compatibility.

## Performance Runtime

The current runtime is T4-first and config-driven rather than hard-coded in the notebook:

- `performance.active_profile` selects a named GPU profile or auto-matches the current GPU by name.
- `performance.compatibility_mode` forces the old conservative Qwen path: single-sample inference, direct canonical cache writes, and `4bit` on CUDA.
- `runtime.print_runtime_summary(...)` prints the resolved profile, CLIP precision, Qwen precision, learned-model AMP precision, Qwen batch size, and cache mode for the current run.
- Qwen hot-path caching can use local scratch on Colab via `runtime.qwen_scratch_root`, but canonical artifacts still end up under `artifacts/qwen/<stage>`.
- CLIP scoring, joint features, and token features are now prepared through one shared split extraction path and reused across the raw CLIP, linear probe, and cross-attention sections.
- Learned-model sweeps keep the same artifact layout, but now follow the resolved profile for AMP precision and export evaluation logits as CPU `float32` so notebook reporting remains NumPy-safe.

## Testing

### Fast tests

These are the quickest checks to run after targeted code changes:

```bash
python -m unittest tests.test_vl_contradiction_plotting
python -m unittest tests.test_vl_contradiction_benchmark
python -m unittest tests.test_vl_contradiction_config
python -m unittest tests.test_vl_contradiction_runtime
python -m unittest tests.test_vl_contradiction_training
python -m unittest tests.test_vl_contradiction_clip_training_perf
python -m unittest tests.test_vl_contradiction_qwen
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
python -m py_compile src/vl_contradiction/config.py src/vl_contradiction/training.py
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

Resolved from `src/vl_contradiction/runtime.py`, the config, and the current notebook stage:

- shared dataset cache: `artifacts/datasets/coco2017`
- benchmark outputs: `artifacts/benchmark/<stage>`
- checkpoints: `artifacts/checkpoints/<stage>`
- per-trial checkpoints: `artifacts/checkpoints/<stage>/<model>__<trial>.pt`
- TensorBoard logs: `artifacts/logs/<stage>`
- per-trial TensorBoard logs: `artifacts/logs/<stage>/<model>__<trial>/`
- metrics: `artifacts/metrics/<stage>`
- sweep summaries: `artifacts/metrics/<stage>/<model>_sweep_<stage>.csv`
- best-trial metadata: `artifacts/metrics/<stage>/<model>_best_trial_<stage>.json`
- figures: `artifacts/figures/<stage>`
- Qwen cache: `artifacts/qwen/<stage>`

## Known Practical Constraints

- The notebook is the main deliverable, so readability matters.
- Qwen inference still emits one JSON artifact per sample under `artifacts/qwen/<stage>`, even when scratch caching is used internally for throughput.
- The readiness gate expects Qwen metrics when `audit.require_qwen_for_readiness` is true.
- Plotting code lives in helpers, but some figure assembly still happens directly in the notebook. When making notebook visualization changes, check both locations before deciding where the fix belongs.
