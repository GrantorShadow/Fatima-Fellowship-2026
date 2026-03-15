"""
Run benchmark evaluations on Qwen3.5-4B-Base using lm-evaluation-harness on Modal.

Implements the phased evaluation plan from plan.md. Supports selective phase
execution, quantization profiling, and per-sample logging for blind spot
identification.

Usage:
    modal run run_eval.py                        # Phase 0 baseline
    modal run run_eval.py --phase 1              # Multilingual
    modal run run_eval.py --phase 2              # Reasoning
    modal run run_eval.py --phase 0 --quant q4   # Phase 0 at Q4
    modal run run_eval.py --tasks gsm8k,arc_challenge  # Specific tasks
"""

import modal
import json
import os

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-eval")

MODEL_ID = "Qwen/Qwen3.5-4B-Base"

# Base image with lm-eval-harness + transformers from source (needed for Qwen3.5)
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        "accelerate>=0.34.0",
        "datasets>=3.0.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.0.0",
        "bitsandbytes>=0.43.0",
        "lm-eval>=0.4.6",
    )
    # Install transformers from git AFTER lm-eval to override its pinned version.
    # Qwen3.5 support (Qwen3_5ForCausalLM) only exists on main, not in any
    # released PyPI version yet (config says transformers_version: 4.57.0.dev0).
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers.git",
    )
    # Pre-download model weights so cold starts are fast
    .run_commands(
        f'python -c "'
        f"from transformers import AutoModelForCausalLM, AutoTokenizer; "
        f"AutoTokenizer.from_pretrained('{MODEL_ID}', trust_remote_code=True); "
        f"AutoModelForCausalLM.from_pretrained('{MODEL_ID}', trust_remote_code=True)"
        f'"'
    )
)

# Persistent volume for caching eval results across runs
eval_volume = modal.Volume.from_name("fatima-eval-results", create_if_missing=True)
EVAL_VOLUME_PATH = "/eval_results"

# ---------------------------------------------------------------------------
# Phase definitions from plan.md
#
# Each phase maps to lm-eval-harness task names where available.
# Tasks marked "custom" need custom evaluation scripts (not yet implemented).
#
# Prompting protocol per plan:
#   - MCQ: Blank-ppl (log-likelihood ranking) -- lm-eval default for MC tasks
#   - Open-ended: 5-shot ICLiP -- set via num_fewshot
#   - Decoding: greedy (temperature=0, no penalties)
# ---------------------------------------------------------------------------
PHASES = {
    "0": {
        "name": "Phase 0: Baseline Harness Setup",
        "description": "Core benchmarks at FP16 to establish baseline. Re-run with --quant to profile degradation.",
        "tasks": [
            "arc_challenge",
            "hellaswag",
            "truthfulqa_mc2",
            "winogrande",
            "mmlu",
        ],
        "num_fewshot": 5,
    },
    "1": {
        "name": "Phase 1: Language & Multilingual Coverage",
        "description": "Map 201-language claim against actual fidelity. Key signal: >20% drop in SW/QU/TE vs EN.",
        "tasks": [
            # MGSM: math word problems across languages
            "mgsm_en", "mgsm_zh", "mgsm_de", "mgsm_fr", "mgsm_ja",
            "mgsm_ru", "mgsm_es", "mgsm_th", "mgsm_bn", "mgsm_sw", "mgsm_te",
            # XCOPA: causal commonsense (11 languages)
            "xcopa_et", "xcopa_id", "xcopa_it", "xcopa_sw", "xcopa_ta",
            "xcopa_th", "xcopa_tr", "xcopa_vi", "xcopa_zh",
            # XNLI: natural language inference (15 languages)
            "xnli_en", "xnli_fr", "xnli_de", "xnli_zh", "xnli_ar",
            "xnli_es", "xnli_hi", "xnli_ru", "xnli_sw", "xnli_th",
            "xnli_tr", "xnli_vi", "xnli_bg", "xnli_el", "xnli_ur",
        ],
        "num_fewshot": 5,
        # NOTE: IndicQA (1.3), FLORES-200 (1.4), XStoryCloze (1.5),
        # AmericasNLI (1.7) require custom setup or are not in lm-eval.
        # Add them as custom tasks when needed.
    },
    "2": {
        "name": "Phase 2: Reasoning & Systematic Generalization",
        "description": "Stress DeltaNet layers' ability to chain logic. Key: accuracy vs hop-depth.",
        "tasks": [
            "gsm8k",                  # 2.1: Arithmetic word problems
            "minerva_math",           # 2.2: Symbolic/algebraic (MATH-500)
            "arc_challenge",          # 2.4: Science reasoning
            "bbh",                    # 2.8: BIG-Bench Hard (23 hard tasks)
        ],
        "num_fewshot": 5,
        # NOTE: MuSiQue (2.3), HotpotQA (2.5), FOLIO (2.6), FORGE (2.7),
        # CLUTRR (2.9) require custom evaluation scripts.
    },
    "3": {
        "name": "Phase 3: Instruction Following & Format Adherence",
        "description": "Test latent instruction-following capacity under ICLiP. Key: constraint saturation at 3+.",
        "tasks": [
            "ifeval",                 # 3.1: Format + length + content constraints
        ],
        "num_fewshot": 0,  # IFEval is typically zero-shot
        # NOTE: FollowBench (3.2), InFoBench (3.3), MT-Bench (3.4),
        # self-consistency (3.5) require custom scripts.
    },
    "4": {
        "name": "Phase 4: Factual Grounding & Hallucination",
        "description": "Map entity confusion, temporal drift, numerical precision. Key: factuality decay with length.",
        "tasks": [
            "triviaqa",              # 4.1: Open-domain fact retrieval
            "nq_open",               # 4.2: Natural Questions
            "truthfulqa_mc2",        # 4.5: False-belief traps
        ],
        "num_fewshot": 5,
        # NOTE: HaluEval (4.3), FActScore (4.4), FEVER (4.6),
        # EntityQuestions (4.7), Time-Sensitive QA (4.8) need custom setup.
    },
    "5": {
        "name": "Phase 5: Long-Context & Memory",
        "description": "Probe 262K context claim. Run at 4K/8K/16K/32K/64K/128K/256K. Key: cliff between 32-64K.",
        "tasks": [],  # All long-context benchmarks require custom scripts
        "num_fewshot": 0,
        # NOTE: RULER (5.1), LongBench (5.2), InfBench (5.3), BABILong (5.4),
        # PassKey (5.5), SCROLLS (5.6) all need custom implementations.
        # Use run_longctx.py (to be created) for these.
    },
    "6": {
        "name": "Phase 6: Vision-Language Grounding",
        "description": "VLM evaluation. Use run_experiments.py for image-based testing, lm-eval for text-only VL benchmarks.",
        "tasks": [],  # VL benchmarks need the multimodal model, not causal LM
        "num_fewshot": 0,
        # NOTE: RefCOCO (6.1), Visual7W (6.2), ScanQA (6.3), SQA3D (6.4),
        # MMBench (6.5), MMStar (6.6), POPE (6.7), TextVQA (6.8),
        # NExT-QA (6.9), Grounding DINO (6.10) all need the multimodal
        # model loaded via Qwen3_5ForConditionalGeneration. Use
        # run_experiments.py or lm-eval's hf-multimodal backend for these.
    },
    "7": {
        "name": "Phase 7: Code & Structured Output",
        "description": "Probe coding ability inherited from pre-training data.",
        "tasks": [
            "humaneval",             # 7.1: Python function synthesis
            "mbpp",                  # 7.2: Short programming problems
        ],
        "num_fewshot": 0,  # Code tasks are typically zero-shot
        # NOTE: CRUXEval (7.3), SWE-Bench Lite (7.4), DS-1000 (7.5)
        # need custom or separate harness (bigcode-evaluation-harness).
    },
    "8": {
        "name": "Phase 8: Safety, Bias & Robustness",
        "description": "Base model lacks RLHF guardrails. Profile inherent bias and adversarial fragility.",
        "tasks": [
            "bbq",                   # 8.1: Social bias in QA
        ],
        "num_fewshot": 5,
        # NOTE: WinoBias/WinoGender (8.2), AdvGLUE (8.3), CheckList (8.4),
        # Multilingual ToxiGen (8.5), BOLD (8.6) need custom scripts.
    },
}

# Quantization presets for Phase 0 profiling
QUANT_CONFIGS = {
    "fp16": {},  # Default: bfloat16
    "q8": {"load_in_8bit": True},
    "q4": {"load_in_4bit": True},
}


# ---------------------------------------------------------------------------
# Evaluation function (runs on GPU)
# ---------------------------------------------------------------------------
@app.function(
    image=eval_image,
    gpu="H100",
    volumes={EVAL_VOLUME_PATH: eval_volume},
    timeout=7200,  # 2 hours max per run
    memory=32768,  # 32GB RAM
)
def run_lm_eval(
    tasks: list[str],
    num_fewshot: int = 5,
    quant: str = "fp16",
    phase_name: str = "",
    limit: int | None = None,
):
    """
    Run lm-evaluation-harness on Qwen3.5-4B-Base with specified tasks.

    Uses AutoModelForCausalLM (text-only causal LM variant) for text benchmarks.
    Greedy decoding, no penalties, per plan.md specification.

    Returns dict with aggregate scores and per-sample predictions.
    """
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    print(f"{'='*60}")
    print(f"Running: {phase_name}")
    print(f"Tasks:   {', '.join(tasks)}")
    print(f"Quant:   {quant}")
    print(f"Fewshot: {num_fewshot}")
    if limit:
        print(f"Limit:   {limit} samples per task")
    print(f"{'='*60}")

    # Build model_args for quantization
    model_kwargs = {
        "pretrained": MODEL_ID,
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }

    qcfg = QUANT_CONFIGS.get(quant, {})
    if "load_in_8bit" in qcfg:
        model_kwargs["load_in_8bit"] = True
        model_kwargs.pop("dtype", None)
    elif "load_in_4bit" in qcfg:
        model_kwargs["load_in_4bit"] = True
        model_kwargs.pop("dtype", None)

    print(f"\nLoading model with: {model_kwargs}")
    lm_obj = HFLM(**model_kwargs)
    print("Model loaded.")

    # Run evaluation with per-sample logging
    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        num_fewshot=num_fewshot,
        log_samples=True,
        limit=limit,
        # Greedy decoding is the default for lm-eval's log-likelihood
        # scoring (MCQ tasks). For generation tasks, lm-eval uses the
        # task's default generation config.
    )

    # Extract aggregate scores
    scores = {}
    if "results" in results:
        for task_name, task_results in results["results"].items():
            scores[task_name] = {
                k: v for k, v in task_results.items()
                if not k.startswith("_")
            }

    # Extract per-sample predictions for blind spot identification
    samples = {}
    if "samples" in results:
        for task_name, task_samples in results["samples"].items():
            samples[task_name] = task_samples

    # Save to volume for persistence
    phase_tag = phase_name.replace(" ", "_").replace(":", "").lower()
    out_dir = os.path.join(EVAL_VOLUME_PATH, phase_tag)
    os.makedirs(out_dir, exist_ok=True)

    scores_path = os.path.join(out_dir, f"scores_{quant}.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2, default=str)

    samples_path = os.path.join(out_dir, f"samples_{quant}.json")
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2, default=str)

    eval_volume.commit()

    print(f"\nResults saved to {out_dir}/")
    return {"scores": scores, "samples_path": samples_path, "phase": phase_name}


# ---------------------------------------------------------------------------
# Blind spot extraction
# ---------------------------------------------------------------------------
def extract_blindspots(scores: dict, phase_name: str) -> list:
    """
    Identify tasks/metrics where the model scores notably low.
    A score below 0.40 (40%) on any metric is flagged as a potential blind spot.
    """
    blindspots = []
    for task_name, metrics in scores.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and "stderr" not in metric_name:
                if value < 0.40:
                    blindspots.append({
                        "phase": phase_name,
                        "task": task_name,
                        "metric": metric_name,
                        "score": round(value, 4),
                        "severity": "CRITICAL" if value < 0.20 else "HIGH" if value < 0.30 else "MEDIUM",
                    })
    return blindspots


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    phase: str = "0",
    quant: str = "fp16",
    tasks: str = "",
    limit: int = 0,
):
    """
    Run benchmark evaluation on Modal.

    Args:
        phase: Phase number (0-8) or "all" for sequential execution.
        quant: Quantization level: fp16, q8, or q4.
        tasks: Comma-separated task names (overrides phase tasks).
        limit: Max samples per task (0 = no limit, use for quick testing).
    """
    # Determine which tasks to run
    if tasks:
        task_list = [t.strip() for t in tasks.split(",")]
        phase_info = PHASES.get(phase, PHASES["0"])
        phase_name = f"Custom ({tasks})"
        num_fewshot = phase_info.get("num_fewshot", 5)
    elif phase == "all":
        # Run all phases sequentially
        all_results = {}
        all_blindspots = []
        for p_id in sorted(PHASES.keys()):
            p = PHASES[p_id]
            if not p["tasks"]:
                print(f"\nSkipping {p['name']}: no lm-eval tasks (requires custom scripts)")
                continue
            print(f"\n{'='*60}")
            print(f"Starting {p['name']}")
            print(f"{'='*60}")
            result = run_lm_eval.remote(
                tasks=p["tasks"],
                num_fewshot=p.get("num_fewshot", 5),
                quant=quant,
                phase_name=p["name"],
                limit=limit if limit > 0 else None,
            )
            all_results[p_id] = result["scores"]
            all_blindspots.extend(extract_blindspots(result["scores"], p["name"]))

        # Save combined results
        with open("eval_results_all.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        with open("eval_blindspots.json", "w") as f:
            json.dump(all_blindspots, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("ALL PHASES COMPLETE")
        print(f"{'='*60}")
        print(f"Results: eval_results_all.json")
        print(f"Blind spots found: {len(all_blindspots)}")
        for bs in all_blindspots:
            print(f"  [{bs['severity']}] {bs['task']}/{bs['metric']}: {bs['score']}")
        return

    else:
        phase_info = PHASES.get(phase)
        if not phase_info:
            print(f"Unknown phase: {phase}. Available: {', '.join(PHASES.keys())}")
            return
        if not phase_info["tasks"]:
            print(f"{phase_info['name']}: No lm-eval tasks defined.")
            print("This phase requires custom evaluation scripts. See plan.md for details.")
            return
        task_list = phase_info["tasks"]
        phase_name = phase_info["name"]
        num_fewshot = phase_info.get("num_fewshot", 5)

    print(f"Phase:    {phase_name}")
    print(f"Tasks:    {len(task_list)} benchmarks")
    print(f"Quant:    {quant}")
    print(f"Fewshot:  {num_fewshot}")
    if limit > 0:
        print(f"Limit:    {limit} samples/task")

    # Launch evaluation on Modal GPU
    result = run_lm_eval.remote(
        tasks=task_list,
        num_fewshot=num_fewshot,
        quant=quant,
        phase_name=phase_name,
        limit=limit if limit > 0 else None,
    )

    scores = result["scores"]

    # Save scores locally
    out_file = f"eval_results_phase{phase}_{quant}.json"
    with open(out_file, "w") as f:
        json.dump(scores, f, indent=2, default=str)
    print(f"\nScores saved to {out_file}")

    # Identify blind spots
    blindspots = extract_blindspots(scores, phase_name)
    if blindspots:
        bs_file = f"eval_blindspots_phase{phase}_{quant}.json"
        with open(bs_file, "w") as f:
            json.dump(blindspots, f, indent=2, default=str)
        print(f"Blind spots ({len(blindspots)}): {bs_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"RESULTS: {phase_name} ({quant})")
    print(f"{'='*60}")
    for task_name, metrics in scores.items():
        print(f"\n  {task_name}:")
        for k, v in metrics.items():
            if isinstance(v, float) and "stderr" not in k:
                print(f"    {k}: {v:.4f}")

    if blindspots:
        print(f"\n{'='*60}")
        print(f"BLIND SPOTS DETECTED: {len(blindspots)}")
        print(f"{'='*60}")
        for bs in blindspots:
            print(f"  [{bs['severity']}] {bs['task']}/{bs['metric']}: {bs['score']}")
