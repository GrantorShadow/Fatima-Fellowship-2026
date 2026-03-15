# Fatima Fellowship 2026 -- Qwen3.5-4B-Base Blind Spot Analysis

Systematic evaluation of [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
to identify failure modes (blind spots) across 8 evaluation phases, documented
as a Hugging Face dataset. See [`plan.md`](plan.md) for the full evaluation plan.

## Model Under Test

| Property | Value |
|----------|-------|
| Model | `Qwen/Qwen3.5-4B-Base` |
| Parameters | 4B |
| Architecture | Gated DeltaNet hybrid: 8x(3xDeltaNet + FFN + 1xAttention + FFN) |
| Context | 262K tokens (claimed) |
| Type | Base model (no instruction tuning) |

**Key architectural note**: Only 1 in 4 layers is true attention. The rest are
DeltaNet recurrent layers with compressed state, making this model architecturally
distinct from pure transformers and a source of novel failure modes.

## Prerequisites

- [Modal](https://modal.com/) (paid tier), authenticated locally (`modal token set`)
- Python 3.11+, [uv](https://docs.astral.sh/uv/) recommended
- `HF_TOKEN` environment variable (for HF Hub push)

```bash
uv sync   # or: pip install -e .
```

> **Note**: `transformers` is installed from git source. `Qwen3_5ForCausalLM`
> is not in any released PyPI version.

---

## Two Evaluation Tracks

### Track A: Text Benchmarks (`run_eval.py`)

Uses `lm-evaluation-harness` (EleutherAI) for systematic benchmark evaluation.
Loads the model as `AutoModelForCausalLM` for text-only tasks.

```bash
# Phase 0: Baseline (ARC-C, HellaSwag, TruthfulQA, WinoGrande, MMLU)
modal run run_eval.py

# Specific phase
modal run run_eval.py --phase 2              # Reasoning (GSM8K, MATH, ARC, BBH)
modal run run_eval.py --phase 1              # Multilingual (MGSM, XCOPA, XNLI)
modal run run_eval.py --phase 4              # Factual grounding

# Quantization profiling (Phase 0 at Q4)
modal run run_eval.py --phase 0 --quant q4

# Specific tasks
modal run run_eval.py --tasks gsm8k,arc_challenge

# Quick test (limit samples)
modal run run_eval.py --phase 0 --limit 50

# All phases sequentially
modal run run_eval.py --phase all
```

Results are saved locally as `eval_results_phase{N}_{quant}.json` and
`eval_blindspots_phase{N}_{quant}.json`, plus persisted on a Modal Volume.

### Track B: Vision Experiments (`run_experiments.py`)

Uses `Qwen3_5ForConditionalGeneration` (full multimodal model) with 15 curated
edge-case images for Phase 6 evaluation.

```bash
# Step 1: Download images to Modal Volume
modal run download_samples.py

# Step 2: Run vision inference
modal run run_experiments.py
```

Results: `experiment_results_raw.json` + `experiment_stats.json`

### Pushing to Hugging Face

```bash
HF_TOKEN=hf_... python generate_submission.py
```

---

## Execution Order (from plan.md)

```
Phase 0 (Baseline + Quant profiling)
    |
Phase 2 (Reasoning) --parallel-- Phase 1 (Multilingual)
    |
Phase 4 (Factual Grounding)
    |
Phase 5 (Long-Context) --parallel-- Phase 6 (Vision)
    |
Phase 3 (Instruction Following)
    |
Phase 7 (Code) --parallel-- Phase 8 (Safety/Bias)
```

Run Phases 2 and 1 first -- they are compute-light and immediately reveal
the highest-impact weaknesses (reasoning depth, multilingual collapse).

---

## Project Structure

```
.
├── pyproject.toml                # Dependencies (lm-eval, transformers, modal, etc.)
├── plan.md                       # Full 8-phase evaluation plan
├── run_eval.py                   # Track A: lm-eval-harness benchmarks on Modal
├── run_experiments.py            # Track B: Vision inference on Modal (Phase 6)
├── download_samples.py           # Download edge-case images to Modal Volume
├── generate_submission.py        # Push HF dataset from results
├── blind_spot_report.md          # Findings template (quantization, heatmap, per-phase)
├── README.md                     # This file
├── eval_results_phase*.json      # (generated) lm-eval scores per phase
├── eval_blindspots_phase*.json   # (generated) identified blind spots
├── experiment_results_raw.json   # (generated) vision experiment results
└── experiment_stats.json         # (generated) vision experiment stats
```

## Hardware

All GPU compute runs on Modal (A10G, 24GB VRAM). Local machine needs only
internet + `modal` CLI. Phase 5 (long-context) may require A100 for 128K+.
