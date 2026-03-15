"""
Run Qwen3.5-4B-Base inference on VLMs-Are-Blind benchmark.

Loads samples from XAI/vlmsareblind, runs multimodal inference using
Qwen3_5ForConditionalGeneration, and compares model answers against
ground truth. Outputs results JSON and accuracy summary.

VLMs-Are-Blind is an 8,016-sample benchmark testing low-level visual
perception that VLMs often fail at:
  - Line Plot Intersections (3600): counting line crossings
  - Touching Circles (1344): counting tangent circles
  - Subway Connections (720): counting colored paths between stations
  - Circled Letter (624): identifying circled letters
  - Olympic Counting - Circles/Pentagons (960): counting overlapping shapes
  - Counting Grid - Word/Blank Grids (528): counting rows and columns
  - Nested Squares (240): counting concentric squares

Usage:
    modal run run_vlmsareblind.py                  # 100 samples (default)
    modal run run_vlmsareblind.py --limit 50       # 50 samples
    modal run run_vlmsareblind.py --limit 0        # All 8016 samples
"""

import modal
import json

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-vlmsareblind")

MODEL_ID = "Qwen/Qwen3.5-4B-Base"

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
    # Pre-download VLMs-Are-Blind dataset
    .run_commands(
        'python -c "'
        "from datasets import load_dataset; "
        "load_dataset('XAI/vlmsareblind', split='valid')"
        '"'
    )
)


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
def extract_answer(completion: str, task: str) -> str:
    """
    Extract the model's answer from its completion.

    The prompts ask for answers in {curly brackets}, e.g. {3} or rows={5} columns={6}.
    We try to extract from brackets first, then fall back to first number/letter.
    """
    import re

    completion = completion.strip()

    if task == "Circled Letter":
        # Look for letters in braces first
        brace_match = re.findall(r"\{([A-Za-z, ]+)\}", completion)
        if brace_match:
            # Return cleaned, sorted, uppercase letters
            letters = re.findall(r"[A-Za-z]", brace_match[0])
            return ",".join(sorted(set(l.upper() for l in letters)))
        # Fall back: first capital letter
        letter_match = re.search(r"[A-Z]", completion)
        if letter_match:
            return letter_match.group()
        return completion[:10]

    if "Counting Grid" in task:
        # Expected format: "rows,cols" e.g. "5,6"
        # Look for rows={N} columns={M} pattern
        rows_match = re.search(r"rows\s*=\s*\{?\s*(\d+)\s*\}?", completion, re.IGNORECASE)
        cols_match = re.search(r"columns?\s*=\s*\{?\s*(\d+)\s*\}?", completion, re.IGNORECASE)
        if rows_match and cols_match:
            return f"{rows_match.group(1)},{cols_match.group(1)}"
        # Fall back: first two numbers
        nums = re.findall(r"\d+", completion)
        if len(nums) >= 2:
            return f"{nums[0]},{nums[1]}"
        if len(nums) == 1:
            return nums[0]
        return completion[:10]

    # Default: numeric answer (Line Plot, Touching Circles, Subway, Olympic, Nested)
    # Look for {N} first
    brace_match = re.search(r"\{(\d+)\}", completion)
    if brace_match:
        return brace_match.group(1)
    # Fall back: first number in completion
    num_match = re.search(r"\d+", completion)
    if num_match:
        return num_match.group()
    return completion[:10]


def check_answer(predicted: str, groundtruth: str, task: str) -> bool:
    """Compare predicted answer against ground truth."""
    predicted = predicted.strip()
    groundtruth = groundtruth.strip()

    if task == "Circled Letter":
        # Both should be sorted uppercase letters
        pred_letters = set(predicted.upper().replace(",", ""))
        gt_letters = set(groundtruth.upper().replace(",", ""))
        return pred_letters == gt_letters

    if "Counting Grid" in task:
        # Format: "rows,cols"
        return predicted == groundtruth

    # Numeric comparison
    try:
        return int(predicted) == int(groundtruth)
    except ValueError:
        return predicted == groundtruth


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    timeout=21600,
    memory=32768,
)
def run_vlmsareblind_inference(limit: int = 100):
    """Load Qwen3.5-4B-Base, run inference on VLMs-Are-Blind samples."""
    import base64
    import io
    import torch
    from datasets import load_dataset
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
    # Load VLMs-Are-Blind dataset
    # ------------------------------------------------------------------
    print("Loading VLMs-Are-Blind dataset...")
    ds = load_dataset("XAI/vlmsareblind", split="valid")
    total = len(ds)
    print(f"VLMs-Are-Blind has {total} samples")

    if limit > 0:
        # Stratified sampling: take proportional samples from each task type
        # so all 9 tasks are represented even with small limits
        from collections import defaultdict
        task_indices = defaultdict(list)
        for idx in range(total):
            task_indices[ds[idx]["task"]].append(idx)

        selected = []
        for task_name, indices in task_indices.items():
            # Proportional allocation, minimum 1 per task
            n_for_task = max(1, int(len(indices) / total * limit))
            # Evenly space indices for representative sampling
            step = max(1, len(indices) // n_for_task)
            selected.extend(indices[::step][:n_for_task])

        selected = sorted(selected)[:limit]
        ds = ds.select(selected)
        print(f"Using {len(ds)} stratified samples across {len(task_indices)} tasks")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    results = []
    correct = 0

    for i, sample in enumerate(ds):
        img = sample["image"].convert("RGB")
        task = sample["task"]
        prompt_text = sample["prompt"]
        groundtruth = sample["groundtruth"]
        metadata = sample["metadata"]

        # Save a thumbnail as base64 for local visualization
        thumb = img.copy()
        thumb.thumbnail((128, 128))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Build prefix-completion prompt for base model.
        # Use the dataset's prompt directly, which already includes
        # the answer format instruction.
        prompt = (
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{prompt_text}\n"
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
        predicted = extract_answer(completion, task)
        is_correct = check_answer(predicted, groundtruth, task)
        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "task": task,
            "prompt": prompt_text[:200],
            "ground_truth": groundtruth,
            "model_output": completion,
            "predicted_answer": predicted,
            "correct": is_correct,
            "metadata": metadata,
            "image_b64": img_b64,
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc_so_far = correct / (i + 1) * 100
            print(f"[{i+1:4d}/{len(ds)}] acc={acc_so_far:.1f}% | "
                  f"task={task} | gt={groundtruth} pred={predicted} | "
                  f"{completion[:60]}")

    # ------------------------------------------------------------------
    # Compute stats
    # ------------------------------------------------------------------
    total_run = len(results)
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    # Per-task accuracy
    task_stats = {}
    for r in results:
        t = r["task"]
        if t not in task_stats:
            task_stats[t] = {"total": 0, "correct": 0}
        task_stats[t]["total"] += 1
        if r["correct"]:
            task_stats[t]["correct"] += 1
    for t in task_stats:
        n = task_stats[t]["total"]
        c = task_stats[t]["correct"]
        task_stats[t]["accuracy"] = round(c / n * 100, 1) if n > 0 else 0

    stats = {
        "model_id": MODEL_ID,
        "dataset": "XAI/vlmsareblind",
        "total_samples": total_run,
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "per_task": task_stats,
    }

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {correct}/{total_run} = {accuracy:.1f}%")
    print(f"{'='*60}")
    print("\nPer task:")
    for t, s in sorted(task_stats.items()):
        print(f"  {t}: {s['correct']}/{s['total']} = {s['accuracy']}%")

    return {"results": results, "stats": stats}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(limit: int = 100):
    print(f"Running VLMs-Are-Blind inference ({limit} samples) on Modal H100...")
    output = run_vlmsareblind_inference.remote(limit=limit)

    results = output["results"]
    stats = output["stats"]

    # Save results
    with open("vlmsareblind_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> vlmsareblind_results.json ({len(results)} samples)")

    # Save stats
    with open("vlmsareblind_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats   -> vlmsareblind_stats.json")

    # Print summary
    print(f"\n{'='*60}")
    print(f"VLMs-Are-Blind RESULTS")
    print(f"{'='*60}")
    print(f"Model:    {stats['model_id']}")
    print(f"Samples:  {stats['total_samples']}")
    print(f"Correct:  {stats['correct']}")
    print(f"Accuracy: {stats['accuracy_pct']}%")

    print(f"\nPer task:")
    for t, s in sorted(stats["per_task"].items()):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {t:40s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    # Flag blind spots (tasks below 30%)
    blindspots = [
        (t, s) for t, s in stats["per_task"].items()
        if s["accuracy"] < 30.0
    ]
    if blindspots:
        print(f"\nBLIND SPOTS (tasks < 30% accuracy):")
        for t, s in blindspots:
            print(f"  ** {t}: {s['accuracy']}%")
