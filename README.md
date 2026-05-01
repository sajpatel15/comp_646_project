# Multimodal Contradiction Project

Authors: Saj Patel and Prathamesh Swar.

This repository builds and evaluates a COCO-derived multimodal contradiction benchmark. The main deliverable is the [final experiment notebook](notebooks/multimodal_contradiction_project.ipynb), backed by the Python package in `src/vl_contradiction`.

The notebook downloads COCO assets as needed, builds the benchmark, trains/evaluates CLIP-based baselines, runs the Qwen2.5-VL reference model, and writes generated outputs under `artifacts/`. Generated datasets, model outputs, figures, checkpoints, logs, and Qwen cache files are intentionally not committed.

## Submission Links

- [Source repository](https://github.com/sajpatel15/comp_646_project)
- [Run the final experiment notebook](notebooks/multimodal_contradiction_project.ipynb)
- [Read the final report](reports/final/report.pdf)

## Repository Layout

- [notebooks/multimodal_contradiction_project.ipynb](notebooks/multimodal_contradiction_project.ipynb): end-to-end final experiment notebook.
- `src/vl_contradiction/`: benchmark, runtime, model, training, Qwen, metrics, and plotting helpers.
- `configs/default.yaml`: runtime paths, final-stage data size, training sweeps, and GPU performance profile settings.
- `tests/`: lightweight unit tests for config loading, benchmark rules, runtime behavior, training helpers, Qwen helpers, and plotting.
- `docs/`: development notes and implementation notes.
- [reports/final/report.pdf](reports/final/report.pdf): final report.
- `reports/final/`: final report source, style files, and referenced report figures.

## Run In Online Colab

Use online Colab as the supported GPU path.

1. Open Google Colab.
2. Upload or import [notebooks/multimodal_contradiction_project.ipynb](notebooks/multimodal_contradiction_project.ipynb).
3. Select a GPU runtime: `Runtime` > `Change runtime type` > `T4 GPU` or better.
4. Run all cells from the top.

When running in Colab, the bootstrap cell clones the public GitHub repository into `/content/project` and installs the package with:

```bash
pip install -e /content/project[qwen]
```

No GitHub username, password, token, or Colab secret is required for cloning the public repository.

The submitted notebook defaults are:

- `RUN_RAW_CLIP = True`
- `RUN_LINEAR_PROBE = True`
- `RUN_CROSS_ATTENTION = True`
- `RUN_QWEN = True`
- `QWEN_BATCH_SIZE_OVERRIDE = 28`
- `CURRENT_STAGE = "final"`

## Generated Outputs

The notebook creates these runtime directories:

- `artifacts/datasets/coco2017`: COCO image and annotation cache.
- `artifacts/benchmark/final`: generated benchmark CSVs and audit sheet.
- `artifacts/metrics/final`: metrics, prediction exports, sweep summaries, and comparison tables.
- `artifacts/figures/final`: notebook-generated figures.
- `artifacts/checkpoints/final`: learned-model checkpoints.
- `artifacts/logs/final`: TensorBoard logs.
- `artifacts/qwen/final`: Qwen per-sample cached outputs.

These paths are ignored by Git and can be deleted between runs.

## Local Development

For lightweight local checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the full notebook/Qwen stack:

```bash
pip install -e .[qwen]
```

Useful validation commands:

```bash
python -m json.tool notebooks/multimodal_contradiction_project.ipynb >/dev/null
python -m unittest tests.test_vl_contradiction_config
python -m unittest tests.test_vl_contradiction_benchmark
python -m unittest tests.test_vl_contradiction_runtime
python -m unittest tests.test_vl_contradiction_training
python -m unittest tests.test_vl_contradiction_clip_training_perf
python -m unittest tests.test_vl_contradiction_qwen
python -m unittest tests.test_vl_contradiction_plotting
```
