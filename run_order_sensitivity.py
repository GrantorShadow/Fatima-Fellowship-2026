"""
Order-Sensitivity Test: Image-First vs Text-First on MMStar.

Runs every MMStar sample TWICE with identical content but different token
ordering to expose prompt-order instability in the model:

  Order A (image-first):  [IMAGE] {question} Answer:
  Order B (text-first):   {question} [IMAGE] Answer:

Early-fusion VLMs build context sequentially, so the relative position of
vision vs text tokens can silently change performance.  This script measures:
  - Per-sample flip rate (passed one order, failed the other)
  - Per-category sensitivity
  - Overall accuracy delta between orderings
  - Directional bias (which ordering wins)

Usage:
    modal run run_order_sensitivity.py                  # 100 samples (default)
    modal run run_order_sensitivity.py --limit 50
    modal run run_order_sensitivity.py --limit 0        # All 1500 samples
"""

import modal
import json

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-order-sensitivity")

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
    .run_commands(
        f"python -c \""
        f"from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration; "
        f"AutoProcessor.from_pretrained('{MODEL_ID}', trust_remote_code=True); "
        f"Qwen3_5ForConditionalGeneration.from_pretrained('{MODEL_ID}', trust_remote_code=True)"
        f"\""
    )
    .run_commands(
        'python -c "'
        "from datasets import load_dataset; "
        "load_dataset('Lin-Chen/MMStar', split='val')"
        '"'
    )
)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def run_single_prompt(model, processor, img, prompt_text):
    """Run a single prompt through the model, return completion string."""
    import torch

    try:
        inputs = processor(
            text=[prompt_text],
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
        return processor.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
    except Exception as e:
        return f"[ERROR] {e}"


def extract_letter(completion):
    """Extract first A/B/C/D from completion."""
    for ch in completion:
        if ch.upper() in "ABCD":
            return ch.upper()
    return ""


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    memory=32768,
)
def run_order_sensitivity(limit: int = 100):
    """Run each MMStar sample with image-first and text-first orderings."""
    import base64
    import io
    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # Load model
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

    # Load MMStar
    print("Loading MMStar dataset...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    total = len(ds)
    print(f"MMStar has {total} samples")

    if limit > 0:
        ds = ds.select(range(min(limit, total)))
        print(f"Using first {len(ds)} samples")

    IMG_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

    results = []
    img_first_correct = 0
    txt_first_correct = 0
    both_correct = 0
    both_wrong = 0
    flipped = 0  # correct in one order, wrong in the other

    for i, sample in enumerate(ds):
        img = sample["image"].convert("RGB")
        question = sample["question"]
        answer = sample["answer"].strip().upper()
        category = sample.get("category", "")
        l2_category = sample.get("l2_category", "")

        # Thumbnail
        thumb = img.copy()
        thumb.thumbnail((128, 128))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # --- Order A: IMAGE first, then text ---
        prompt_img_first = (
            f"{IMG_TOKEN}"
            f"{question}\n"
            f"Answer:"
        )
        comp_a = run_single_prompt(model, processor, img, prompt_img_first)
        pred_a = extract_letter(comp_a)
        correct_a = pred_a == answer

        # --- Order B: TEXT first, then image ---
        prompt_txt_first = (
            f"{question}\n"
            f"{IMG_TOKEN}"
            f"Answer:"
        )
        comp_b = run_single_prompt(model, processor, img, prompt_txt_first)
        pred_b = extract_letter(comp_b)
        correct_b = pred_b == answer

        # Classify outcome
        if correct_a:
            img_first_correct += 1
        if correct_b:
            txt_first_correct += 1
        if correct_a and correct_b:
            both_correct += 1
        elif not correct_a and not correct_b:
            both_wrong += 1
        else:
            flipped += 1

        # Stability: did the predicted letter change at all (regardless of correctness)?
        prediction_changed = pred_a != pred_b

        results.append({
            "index": sample.get("index", i),
            "category": category,
            "l2_category": l2_category,
            "question": question[:200],
            "ground_truth": answer,
            # Image-first
            "img_first_output": comp_a,
            "img_first_pred": pred_a,
            "img_first_correct": correct_a,
            # Text-first
            "txt_first_output": comp_b,
            "txt_first_pred": pred_b,
            "txt_first_correct": correct_b,
            # Derived
            "prediction_changed": prediction_changed,
            "flipped_correctness": (correct_a != correct_b),
            "image_b64": img_b64,
        })

        if (i + 1) % 10 == 0 or i == 0:
            n = i + 1
            print(
                f"[{n:4d}/{len(ds)}] "
                f"imgF={img_first_correct/n*100:.1f}% "
                f"txtF={txt_first_correct/n*100:.1f}% "
                f"flip={flipped/n*100:.1f}% "
                f"| gt={answer} pA={pred_a} pB={pred_b} "
                f"{'FLIP' if prediction_changed else '    '}"
            )

    # ------------------------------------------------------------------
    # Compute stats
    # ------------------------------------------------------------------
    n = len(results)
    pred_changed_count = sum(1 for r in results if r["prediction_changed"])

    # Per-category breakdown
    from collections import defaultdict
    cat_data = defaultdict(lambda: {
        "total": 0, "img_first_correct": 0, "txt_first_correct": 0,
        "flipped": 0, "pred_changed": 0,
    })
    for r in results:
        c = r["category"]
        cat_data[c]["total"] += 1
        if r["img_first_correct"]:
            cat_data[c]["img_first_correct"] += 1
        if r["txt_first_correct"]:
            cat_data[c]["txt_first_correct"] += 1
        if r["flipped_correctness"]:
            cat_data[c]["flipped"] += 1
        if r["prediction_changed"]:
            cat_data[c]["pred_changed"] += 1

    cat_stats = {}
    for c, d in cat_data.items():
        t = d["total"]
        cat_stats[c] = {
            "total": t,
            "img_first_acc": round(d["img_first_correct"] / t * 100, 1),
            "txt_first_acc": round(d["txt_first_correct"] / t * 100, 1),
            "flip_rate": round(d["flipped"] / t * 100, 1),
            "pred_change_rate": round(d["pred_changed"] / t * 100, 1),
        }

    # Per-L2 category
    l2_data = defaultdict(lambda: {
        "total": 0, "img_first_correct": 0, "txt_first_correct": 0,
        "flipped": 0, "pred_changed": 0,
    })
    for r in results:
        l2 = r["l2_category"]
        l2_data[l2]["total"] += 1
        if r["img_first_correct"]:
            l2_data[l2]["img_first_correct"] += 1
        if r["txt_first_correct"]:
            l2_data[l2]["txt_first_correct"] += 1
        if r["flipped_correctness"]:
            l2_data[l2]["flipped"] += 1
        if r["prediction_changed"]:
            l2_data[l2]["pred_changed"] += 1

    l2_stats = {}
    for l2, d in l2_data.items():
        t = d["total"]
        l2_stats[l2] = {
            "total": t,
            "img_first_acc": round(d["img_first_correct"] / t * 100, 1),
            "txt_first_acc": round(d["txt_first_correct"] / t * 100, 1),
            "flip_rate": round(d["flipped"] / t * 100, 1),
            "pred_change_rate": round(d["pred_changed"] / t * 100, 1),
        }

    stats = {
        "model_id": MODEL_ID,
        "dataset": "Lin-Chen/MMStar",
        "total_samples": n,
        "img_first_accuracy": round(img_first_correct / n * 100, 2),
        "txt_first_accuracy": round(txt_first_correct / n * 100, 2),
        "accuracy_delta": round(
            (img_first_correct - txt_first_correct) / n * 100, 2
        ),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "flipped_correctness": flipped,
        "prediction_changed": pred_changed_count,
        "flip_rate_pct": round(flipped / n * 100, 2),
        "pred_change_rate_pct": round(pred_changed_count / n * 100, 2),
        "per_category": cat_stats,
        "per_l2_category": l2_stats,
    }

    print(f"\n{'='*70}")
    print(f"ORDER SENSITIVITY RESULTS")
    print(f"{'='*70}")
    print(f"Image-first accuracy:  {stats['img_first_accuracy']}%")
    print(f"Text-first accuracy:   {stats['txt_first_accuracy']}%")
    print(f"Accuracy delta (I-T):  {stats['accuracy_delta']:+.2f}pp")
    print(f"Prediction changed:    {pred_changed_count}/{n} "
          f"({stats['pred_change_rate_pct']}%)")
    print(f"Correctness flipped:   {flipped}/{n} "
          f"({stats['flip_rate_pct']}%)")
    print(f"Both correct:          {both_correct}/{n}")
    print(f"Both wrong:            {both_wrong}/{n}")
    print(f"\nPer category:")
    for c, s in sorted(cat_stats.items()):
        print(f"  {c:30s} imgF={s['img_first_acc']:5.1f}% "
              f"txtF={s['txt_first_acc']:5.1f}% "
              f"flip={s['flip_rate']:5.1f}% "
              f"chg={s['pred_change_rate']:5.1f}%")

    return {"results": results, "stats": stats}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(limit: int = 100):
    print(f"Running order-sensitivity test ({limit} samples) on Modal H100...")
    output = run_order_sensitivity.remote(limit=limit)

    results = output["results"]
    stats = output["stats"]

    with open("order_sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> order_sensitivity_results.json ({len(results)} samples)")

    with open("order_sensitivity_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats   -> order_sensitivity_stats.json")

    print(f"\n{'='*70}")
    print(f"ORDER SENSITIVITY RESULTS")
    print(f"{'='*70}")
    print(f"Model:               {stats['model_id']}")
    print(f"Samples:             {stats['total_samples']}")
    print(f"Image-first acc:     {stats['img_first_accuracy']}%")
    print(f"Text-first acc:      {stats['txt_first_accuracy']}%")
    print(f"Delta (img - txt):   {stats['accuracy_delta']:+.2f}pp")
    print(f"Prediction changed:  {stats['prediction_changed']}/{stats['total_samples']} "
          f"({stats['pred_change_rate_pct']}%)")
    print(f"Correctness flipped: {stats['flipped_correctness']}/{stats['total_samples']} "
          f"({stats['flip_rate_pct']}%)")

    print(f"\nPer category:")
    for c, s in sorted(stats["per_category"].items()):
        delta = s["img_first_acc"] - s["txt_first_acc"]
        bar = "#" * int(s["pred_change_rate"] / 5)
        print(f"  {c:30s} I={s['img_first_acc']:5.1f}% T={s['txt_first_acc']:5.1f}% "
              f"\u0394={delta:+5.1f}pp  chg={s['pred_change_rate']:5.1f}% {bar}")

    print(f"\nPer L2 category:")
    for l2, s in sorted(stats["per_l2_category"].items(),
                        key=lambda x: x[1]["pred_change_rate"], reverse=True):
        delta = s["img_first_acc"] - s["txt_first_acc"]
        print(f"  {l2:40s} I={s['img_first_acc']:5.1f}% T={s['txt_first_acc']:5.1f}% "
              f"\u0394={delta:+5.1f}pp  chg={s['pred_change_rate']:5.1f}%")

    if stats["pred_change_rate_pct"] > 15:
        print(f"\n** HIGH INSTABILITY: {stats['pred_change_rate_pct']}% of predictions "
              f"changed just from reordering image/text tokens **")
