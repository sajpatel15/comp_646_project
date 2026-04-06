# Implementation Plan for the Multimodal Contradiction Project

## Summary
Build a reproducible COCO-derived benchmark for image-caption entailment, neutrality, and contradiction, then compare progressively stronger baselines and models: `majority/random`, `raw CLIP score`, `frozen CLIP linear probe`, `optional tiny MLP ablation`, `Qwen2.5-VL zero-shot subset baseline`, and `cross-attention fusion`.

## Project Structure
- Main deliverable: one primary notebook that runs both locally and on Colab.
- Supporting code: helper Python modules for data loading, benchmark generation, models, training, evaluation, and plotting.
- Persistent cache: local runtime storage under the execution environment, with datasets and derived artifacts kept under the configured local artifact roots.
- Final report: not implemented until the experiment pipeline and outputs are complete.

## Priority Order
1. Runtime and reproducibility
2. Benchmark specification
3. COCO ingestion and validation
4. Rule-based benchmark generation
5. Audit gate
6. Raw CLIP baseline
7. Linear probe
8. Qwen subset baseline
9. Cross-attention model
10. Calibration and error analysis
11. Final figures and tables
12. Final report

## Fixed Defaults
- Split by source family to prevent leakage.
- Dataset sizes:
  - Prototype: `1,200` source families
  - Mid-scale: `4,000` source families
  - Final-scale: `6,000` source families
- Qwen subset: fixed `600` examples selected once by seed.
- Qwen runtime: adaptive, profile-driven precision and batching by default, with a compatibility mode that preserves the older conservative `4bit` path when needed.
- CLIP and learned-model performance tuning: config-driven and stage-aware through the runtime/performance layer rather than notebook-specific branching.
- COCO access: check the configured local dataset cache first, otherwise download automatically.

## Acceptance Criteria
- Notebook runs both locally and in Colab with the same config schema.
- COCO assets are auto-validated and auto-downloaded when missing.
- Audit gate passes before scaled model training.
- Raw CLIP, linear probe, and Qwen complete end-to-end on the prototype stage before scaling.
- Cross-attention completes prototype training before scale-up.
- Every final figure and table is reproducible from cached artifacts.
- No screenshot-based figures appear in final outputs.
