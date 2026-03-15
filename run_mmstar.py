"""
Run Qwen3.5-4B-Base inference on MMStar benchmark samples.

Loads 100 (or --limit N) samples from Lin-Chen/MMStar, runs multimodal
inference using Qwen3_5ForConditionalGeneration, and compares model answers
against ground truth. Outputs results JSON and accuracy summary.

MMStar is a 1,500-sample multimodal benchmark with multiple-choice visual
questions spanning: coarse perception, fine-grained perception, instance
reasoning, logical reasoning, science & technology, and math.

Usage:
    modal run run_mmstar.py                  # 100 samples (default)
    modal run run_mmstar.py --limit 50       # 50 samples
    modal run run_mmstar.py --limit 0        # All 1500 samples
"""

import modal
import json

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-mmstar")

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
    # Pre-download MMStar dataset
    .run_commands(
        'python -c "'
        "from datasets import load_dataset; "
        "load_dataset('Lin-Chen/MMStar', split='val')"
        '"'
    )
)


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    memory=32768,
)
def run_mmstar_inference(limit: int = 100):
    """Load Qwen3.5-4B-Base, run inference on MMStar samples."""
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
    # Load MMStar dataset
    # ------------------------------------------------------------------
    print("Loading MMStar dataset...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    total = len(ds)
    print(f"MMStar has {total} samples")

    if limit > 0:
        ds = ds.select(range(min(limit, total)))
        print(f"Using first {len(ds)} samples")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    results = []
    correct = 0

    for i, sample in enumerate(ds):
        img = sample["image"].convert("RGB")
        question = sample["question"]
        answer = sample["answer"]
        category = sample.get("category", "")
        l2_category = sample.get("l2_category", "")

        # Save a thumbnail as base64 for local visualization
        thumb = img.copy()
        thumb.thumbnail((128, 128))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Build prefix-completion prompt for base model.
        # MMStar is multiple-choice, so we present the question and ask
        # the model to complete with the answer letter.
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
                    max_new_tokens=20,
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

        # Check if the model's answer matches ground truth.
        # Extract the first letter (A/B/C/D) from the completion.
        predicted_letter = ""
        for ch in completion:
            if ch.upper() in "ABCD":
                predicted_letter = ch.upper()
                break

        is_correct = predicted_letter == answer.strip().upper()
        if is_correct:
            correct += 1

        results.append({
            "index": sample.get("index", i),
            "category": category,
            "l2_category": l2_category,
            "question": question[:200],
            "ground_truth": answer,
            "model_output": completion,
            "predicted_letter": predicted_letter,
            "correct": is_correct,
            "image_b64": img_b64,
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc_so_far = correct / (i + 1) * 100
            print(f"[{i+1:4d}/{len(ds)}] acc={acc_so_far:.1f}% | "
                  f"gt={answer} pred={predicted_letter} | "
                  f"{completion[:60]}")

    # ------------------------------------------------------------------
    # Compute stats
    # ------------------------------------------------------------------
    total_run = len(results)
    accuracy = correct / total_run * 100 if total_run > 0 else 0

    # Per-category accuracy
    cat_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        if r["correct"]:
            cat_stats[cat]["correct"] += 1
    for cat in cat_stats:
        t = cat_stats[cat]["total"]
        c = cat_stats[cat]["correct"]
        cat_stats[cat]["accuracy"] = round(c / t * 100, 1) if t > 0 else 0

    # Per-l2-category accuracy
    l2_stats = {}
    for r in results:
        l2 = r["l2_category"]
        if l2 not in l2_stats:
            l2_stats[l2] = {"total": 0, "correct": 0}
        l2_stats[l2]["total"] += 1
        if r["correct"]:
            l2_stats[l2]["correct"] += 1
    for l2 in l2_stats:
        t = l2_stats[l2]["total"]
        c = l2_stats[l2]["correct"]
        l2_stats[l2]["accuracy"] = round(c / t * 100, 1) if t > 0 else 0

    stats = {
        "model_id": MODEL_ID,
        "dataset": "Lin-Chen/MMStar",
        "total_samples": total_run,
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "per_category": cat_stats,
        "per_l2_category": l2_stats,
    }

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {correct}/{total_run} = {accuracy:.1f}%")
    print(f"{'='*60}")
    print("\nPer category:")
    for cat, s in sorted(cat_stats.items()):
        print(f"  {cat}: {s['correct']}/{s['total']} = {s['accuracy']}%")
    print("\nPer L2 category:")
    for l2, s in sorted(l2_stats.items()):
        print(f"  {l2}: {s['correct']}/{s['total']} = {s['accuracy']}%")

    return {"results": results, "stats": stats}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(limit: int = 100):
    print(f"Running MMStar inference ({limit} samples) on Modal H100...")
    output = run_mmstar_inference.remote(limit=limit)

    results = output["results"]
    stats = output["stats"]

    # Save results
    with open("mmstar_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> mmstar_results.json ({len(results)} samples)")

    # Save stats
    with open("mmstar_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats   -> mmstar_stats.json")

    # Print summary
    print(f"\n{'='*60}")
    print(f"MMStar RESULTS")
    print(f"{'='*60}")
    print(f"Model:    {stats['model_id']}")
    print(f"Samples:  {stats['total_samples']}")
    print(f"Correct:  {stats['correct']}")
    print(f"Accuracy: {stats['accuracy_pct']}%")

    print(f"\nPer category:")
    for cat, s in sorted(stats["per_category"].items()):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {cat:30s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    print(f"\nPer L2 category:")
    for l2, s in sorted(stats["per_l2_category"].items()):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {l2:40s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    # Flag blind spots (categories below 30%)
    blindspots = [
        (cat, s) for cat, s in stats["per_category"].items()
        if s["accuracy"] < 30.0
    ]
    if blindspots:
        print(f"\nBLIND SPOTS (categories < 30% accuracy):")
        for cat, s in blindspots:
            print(f"  ** {cat}: {s['accuracy']}%")
