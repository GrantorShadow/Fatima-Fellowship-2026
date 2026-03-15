"""
Run Qwen3.5-4B-Base inference on SPHERE-VLM benchmark.

Loads samples from all 12 configs of wei2912/SPHERE-VLM, runs multimodal
inference using Qwen3_5ForConditionalGeneration, and compares model answers
against ground truth. Outputs results JSON and accuracy summary.

SPHERE-VLM is a 2,684-sample benchmark testing spatial perception and
reasoning in VLMs, organized into 3 groups:

  Single-skill (5 configs, ~960 samples):
    - counting_only-paired-distance_and_counting (100)
    - counting_only-paired-position_and_counting (101)
    - distance_only (202)
    - position_only (357)
    - size_only (198)

  Combined 2-skill (3 configs, ~526 samples):
    - distance_and_counting (158)
    - distance_and_size (199)
    - position_and_counting (169)

  Reasoning (4 configs, ~1200 samples):
    - object_manipulation (399)
    - object_manipulation_w_intermediate (199)
    - object_occlusion (402)
    - object_occlusion_w_intermediate (200)

Answer formats:
  - "num": open-ended numeric (no MCQ options)
  - "name"/"pos"/"bool": multiple-choice from option list

Usage:
    modal run run_sphere.py                  # 100 samples (default)
    modal run run_sphere.py --limit 50       # 50 samples
    modal run run_sphere.py --limit 0        # All ~2684 samples
"""

import modal
import json

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-sphere")

MODEL_ID = "Qwen/Qwen3.5-4B-Base"

SPHERE_CONFIGS = [
    # Single skill
    "counting_only-paired-distance_and_counting",
    "counting_only-paired-position_and_counting",
    "distance_only",
    "position_only",
    "size_only",
    # Combined 2-skill
    "distance_and_counting",
    "distance_and_size",
    "position_and_counting",
    # Reasoning
    "object_manipulation",
    "object_manipulation_w_intermediate",
    "object_occlusion",
    "object_occlusion_w_intermediate",
]

# Category groupings for analysis
CONFIG_GROUP = {
    "counting_only-paired-distance_and_counting": "single_skill",
    "counting_only-paired-position_and_counting": "single_skill",
    "distance_only": "single_skill",
    "position_only": "single_skill",
    "size_only": "single_skill",
    "distance_and_counting": "combine_2_skill",
    "distance_and_size": "combine_2_skill",
    "position_and_counting": "combine_2_skill",
    "object_manipulation": "reasoning",
    "object_manipulation_w_intermediate": "reasoning",
    "object_occlusion": "reasoning",
    "object_occlusion_w_intermediate": "reasoning",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "accelerate>=0.34.0",
        "pillow>=10.0.0",
        "huggingface_hub>=0.25.0",
        "qwen-vl-utils>=0.0.8",
        "datasets>=3.0.0",
        "pandas>=2.0.0",
    )
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers.git",
    )
    # Pre-download model weights into the image
    .run_commands(
        f"python -c \""
        f"from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration; "
        f"AutoProcessor.from_pretrained('{MODEL_ID}', trust_remote_code=True); "
        f"Qwen3_5ForConditionalGeneration.from_pretrained('{MODEL_ID}', trust_remote_code=True)"
        f"\""
    )
    # Pre-download all SPHERE-VLM configs
    .run_commands(
        'python -c "'
        "from datasets import load_dataset; "
        "[load_dataset('wei2912/SPHERE-VLM', c, split='train') for c in ["
        "'counting_only-paired-distance_and_counting',"
        "'counting_only-paired-position_and_counting',"
        "'distance_only','position_only','size_only',"
        "'distance_and_counting','distance_and_size','position_and_counting',"
        "'object_manipulation','object_manipulation_w_intermediate',"
        "'object_occlusion','object_occlusion_w_intermediate'"
        "]]"
        '"'
    )
)


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
def extract_answer(completion: str, answer_format: str, options: list) -> str:
    """
    Extract the model's answer from its completion.

    For numeric format: extract first number.
    For MCQ (name/pos/bool): match against provided options.
    """
    import re

    completion = completion.strip()

    if answer_format == "num":
        # Open-ended numeric — extract first integer
        num_match = re.search(r"\d+", completion)
        if num_match:
            return num_match.group()
        return completion[:20]

    # MCQ: match against one of the option strings
    if options:
        completion_lower = completion.lower()
        # Try exact substring match first
        for opt in options:
            opt_str = str(opt).strip()
            if opt_str.lower() in completion_lower:
                return opt_str
        # Try if completion starts with an option
        for opt in options:
            opt_str = str(opt).strip()
            if completion_lower.startswith(opt_str.lower()):
                return opt_str
        # For yes/no (bool format), check for y/n
        if answer_format == "bool":
            if completion_lower.startswith("yes") or completion_lower.startswith("y"):
                for opt in options:
                    if str(opt).strip().lower() == "yes":
                        return str(opt).strip()
            if completion_lower.startswith("no") or completion_lower.startswith("n"):
                for opt in options:
                    if str(opt).strip().lower() == "no":
                        return str(opt).strip()
        # For position format, check for left/right keywords
        if answer_format == "pos":
            for opt in options:
                opt_str = str(opt).strip().lower()
                if opt_str in ("left", "right") and opt_str in completion_lower:
                    return str(opt).strip()

    return completion[:50]


def check_answer(predicted: str, ground_truth: str, answer_format: str) -> bool:
    """Compare predicted answer against ground truth."""
    predicted = str(predicted).strip().lower()
    ground_truth = str(ground_truth).strip().lower()

    if answer_format == "num":
        try:
            return int(predicted) == int(float(ground_truth))
        except (ValueError, TypeError):
            return predicted == ground_truth

    # String comparison for MCQ
    return predicted == ground_truth


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    timeout=21600,
    memory=32768,
)
def run_sphere_inference(limit: int = 100):
    """Load Qwen3.5-4B-Base, run inference on all SPHERE-VLM configs."""
    import base64
    import io
    import math
    import torch
    from collections import defaultdict
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
    )
    model.eval()
    print("Model loaded on CUDA")

    # ------------------------------------------------------------------
    # Load all SPHERE-VLM configs and merge
    # ------------------------------------------------------------------
    print("Loading SPHERE-VLM dataset (all 12 configs)...")
    all_samples = []
    config_counts = {}

    for config_name in SPHERE_CONFIGS:
        ds = load_dataset("wei2912/SPHERE-VLM", config_name, split="train")
        config_counts[config_name] = len(ds)
        for idx in range(len(ds)):
            sample = ds[idx]
            # Normalize option field: can be float NaN or list of strings
            raw_option = sample.get("option")
            if isinstance(raw_option, float) and math.isnan(raw_option):
                options = []
            elif isinstance(raw_option, list):
                options = [str(o) for o in raw_option]
            else:
                options = []

            # Get answer format from metadata
            meta = sample.get("metadata", {})
            answer_format = meta.get("format", "unknown") if isinstance(meta, dict) else "unknown"
            viewpoint = meta.get("viewpoint", "unknown") if isinstance(meta, dict) else "unknown"

            all_samples.append({
                "config": config_name,
                "group": CONFIG_GROUP[config_name],
                "question_id": sample.get("question_id"),
                "question": sample["question"],
                "options": options,
                "answer": str(sample["answer"]),
                "answer_format": answer_format,
                "viewpoint": viewpoint,
                "metadata": str(meta),
                "image": sample["image"],
            })

    total = len(all_samples)
    print(f"SPHERE-VLM total: {total} samples across {len(SPHERE_CONFIGS)} configs")
    for c, n in config_counts.items():
        print(f"  {c}: {n}")

    # ------------------------------------------------------------------
    # Stratified sampling if limit > 0
    # ------------------------------------------------------------------
    if limit > 0 and limit < total:
        config_indices = defaultdict(list)
        for idx, s in enumerate(all_samples):
            config_indices[s["config"]].append(idx)

        selected = []
        for config_name, indices in config_indices.items():
            n_for_config = max(1, int(len(indices) / total * limit))
            step = max(1, len(indices) // n_for_config)
            selected.extend(indices[::step][:n_for_config])

        selected = sorted(selected)[:limit]
        all_samples = [all_samples[i] for i in selected]
        print(f"Using {len(all_samples)} stratified samples across {len(config_indices)} configs")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    results = []
    correct = 0

    for i, sample in enumerate(all_samples):
        img = sample["image"].convert("RGB")
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer"]
        answer_format = sample["answer_format"]
        config_name = sample["config"]

        # Save a thumbnail as base64 for local visualization
        thumb = img.copy()
        thumb.thumbnail((128, 128))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Build prompt: include options for MCQ, omit for open-ended numeric
        if options:
            options_str = "\n".join(
                f"({chr(65+j)}) {opt}" for j, opt in enumerate(options)
            )
            prompt = (
                "<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}\n{options_str}\n"
                f"Answer:"
            )
        else:
            prompt = (
                "<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}\n"
                f"Answer:"
            )

        try:
            inputs = processor(
                text=[prompt],
                images=[img],
                return_tensors="pt",
            )
            inputs = {
                k: v.to("cuda") if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                    or processor.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            completion = processor.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip()

        except Exception as e:
            completion = f"[ERROR] {e}"

        # Extract and check answer
        predicted = extract_answer(completion, answer_format, options)
        is_correct = check_answer(predicted, answer, answer_format)
        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "config": config_name,
            "group": sample["group"],
            "answer_format": answer_format,
            "viewpoint": sample["viewpoint"],
            "question": question[:200],
            "options": options,
            "ground_truth": answer,
            "model_output": completion,
            "predicted_answer": predicted,
            "correct": is_correct,
            "metadata": sample["metadata"],
            "image_b64": img_b64,
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc_so_far = correct / (i + 1) * 100
            print(f"[{i+1:4d}/{len(all_samples)}] acc={acc_so_far:.1f}% | "
                  f"cfg={config_name} | gt={answer[:30]} pred={predicted[:30]} | "
                  f"{completion[:50]}")

    # ------------------------------------------------------------------
    # Compute stats
    # ------------------------------------------------------------------
    total_run = len(results)
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    def compute_breakdown(results, key):
        stats = {}
        for r in results:
            k = r[key]
            if k not in stats:
                stats[k] = {"total": 0, "correct": 0}
            stats[k]["total"] += 1
            if r["correct"]:
                stats[k]["correct"] += 1
        for k in stats:
            n = stats[k]["total"]
            c = stats[k]["correct"]
            stats[k]["accuracy"] = round(c / n * 100, 1) if n > 0 else 0
        return stats

    config_stats = compute_breakdown(results, "config")
    group_stats = compute_breakdown(results, "group")
    viewpoint_stats = compute_breakdown(results, "viewpoint")
    format_stats = compute_breakdown(results, "answer_format")

    stats = {
        "model_id": MODEL_ID,
        "dataset": "wei2912/SPHERE-VLM",
        "total_samples": total_run,
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "per_config": config_stats,
        "per_group": group_stats,
        "per_viewpoint": viewpoint_stats,
        "per_format": format_stats,
    }

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {correct}/{total_run} = {accuracy:.1f}%")
    print(f"{'='*60}")

    print("\nPer group:")
    for g, s in sorted(group_stats.items()):
        print(f"  {g}: {s['correct']}/{s['total']} = {s['accuracy']}%")

    print("\nPer config:")
    for c, s in sorted(config_stats.items()):
        print(f"  {c}: {s['correct']}/{s['total']} = {s['accuracy']}%")

    print("\nPer viewpoint:")
    for v, s in sorted(viewpoint_stats.items()):
        print(f"  {v}: {s['correct']}/{s['total']} = {s['accuracy']}%")

    print("\nPer answer format:")
    for f, s in sorted(format_stats.items()):
        print(f"  {f}: {s['correct']}/{s['total']} = {s['accuracy']}%")

    return {"results": results, "stats": stats}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(limit: int = 100):
    print(f"Running SPHERE-VLM inference ({limit} samples) on Modal H100...")
    output = run_sphere_inference.remote(limit=limit)

    results = output["results"]
    stats = output["stats"]

    # Save results
    with open("sphere_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> sphere_results.json ({len(results)} samples)")

    # Save stats
    with open("sphere_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats   -> sphere_stats.json")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SPHERE-VLM RESULTS")
    print(f"{'='*60}")
    print(f"Model:    {stats['model_id']}")
    print(f"Samples:  {stats['total_samples']}")
    print(f"Correct:  {stats['correct']}")
    print(f"Accuracy: {stats['accuracy_pct']}%")

    print(f"\nPer group:")
    for g, s in sorted(stats["per_group"].items()):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {g:25s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    print(f"\nPer config:")
    for c, s in sorted(stats["per_config"].items()):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {c:50s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    print(f"\nPer viewpoint (ego vs allo):")
    for v, s in sorted(stats["per_viewpoint"].items()):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {v:10s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    print(f"\nPer answer format:")
    for f_name, s in sorted(stats["per_format"].items()):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {f_name:10s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    # Flag blind spots (configs below 30%)
    blindspots = [
        (c, s) for c, s in stats["per_config"].items()
        if s["accuracy"] < 30.0
    ]
    if blindspots:
        print(f"\nBLIND SPOTS (configs < 30% accuracy):")
        for c, s in blindspots:
            print(f"  ** {c}: {s['accuracy']}%")

    # Flag viewpoint blind spots
    vp_blindspots = [
        (v, s) for v, s in stats["per_viewpoint"].items()
        if s["accuracy"] < 30.0
    ]
    if vp_blindspots:
        print(f"\nVIEWPOINT BLIND SPOTS (< 30%):")
        for v, s in vp_blindspots:
            print(f"  ** {v}: {s['accuracy']}%")
