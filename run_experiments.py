"""
Run blind-spot experiments on Qwen3.5-4B-Base.

Loads images from the Modal Volume, runs prefix-completion inference on a GPU,
and writes experiment_results_raw.json + experiment_stats.json locally.

Usage:
    modal run run_experiments.py
"""

import modal
import json

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-run-experiments")

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
    )
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers.git",
    )
    # Pre-download model weights into the image so cold starts are fast
    .run_commands(
        f"python -c \""
        f"from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration; "
        f"AutoProcessor.from_pretrained('{MODEL_ID}', trust_remote_code=True); "
        f"Qwen3_5ForConditionalGeneration.from_pretrained('{MODEL_ID}', trust_remote_code=True)"
        f"\""
    )
)

volume = modal.Volume.from_name("fatima-sample-data", create_if_missing=True)
VOLUME_PATH = "/data"
SAMPLES_DIR = f"{VOLUME_PATH}/samples"


# ---------------------------------------------------------------------------
# Metadata for the 15 samples (must match download_samples.py ordering)
# ---------------------------------------------------------------------------
SAMPLE_METADATA = [
    {"category": "Spatial Reasoning (Reflections)", "prompt_context": "mirror reflection", "expected": "A mountain reflected symmetrically in a still lake, showing both the real mountain and its mirror image in the water"},
    {"category": "Transparency / Refraction", "prompt_context": "glass of water", "expected": "A glass of water with a visible refraction effect distorting what is behind or inside the glass"},
    {"category": "Forced Perspective", "prompt_context": "tourist photo", "expected": "A person using forced perspective photography to appear to touch or hold a large building in the background"},
    {"category": "Counting Overlapping Objects", "prompt_context": "pile of coins", "expected": "A collection of many overlapping US coins of different denominations spread out, difficult to count precisely"},
    {"category": "Text / Handwriting (OCR)", "prompt_context": "handwritten document", "expected": "A handwritten letter on paper with cursive or script writing that requires OCR-like reading to interpret"},
    {"category": "Object Occlusion", "prompt_context": "domestic animal", "expected": "A tabby cat sitting upright and looking at the camera"},
    {"category": "Shadows / Silhouettes", "prompt_context": "shadow puppet", "expected": "A traditional shadow puppet figure used in theatrical performances, showing intricate cut-out details"},
    {"category": "Camouflage / Hidden Object", "prompt_context": "leaves on branch", "expected": "A leaf insect camouflaged among real leaves, extremely difficult to distinguish from the surrounding foliage"},
    {"category": "Out-of-Context Object", "prompt_context": "vehicle on highway", "expected": "A large boat being hauled on a truck trailer driving down a road"},
    {"category": "Extreme Aspect Ratio (Panorama)", "prompt_context": "city skyline panorama", "expected": "An extremely wide panoramic photograph of a city skyline at twilight"},
    {"category": "Low Light / Noise", "prompt_context": "dark room", "expected": "A very dark room with a single streetlight shining through a window, most details lost in darkness and noise"},
    {"category": "Fine-Grained Recognition (Landmark)", "prompt_context": "mountain landscape", "expected": "Mount Everest's north face as seen from the base camp in Tibet"},
    {"category": "Iconic Landmark / Spatial Layout", "prompt_context": "famous bridge", "expected": "The Golden Gate Bridge in San Francisco, a large red suspension bridge spanning across the bay"},
    {"category": "Optical Illusion", "prompt_context": "checkerboard pattern", "expected": "Adelson's checker shadow illusion: a checkerboard with a cylinder casting a shadow, where squares A and B appear different but are the same shade"},
    {"category": "Dense Urban Scene", "prompt_context": "city skyline", "expected": "The Boston skyline viewed from across the Charles River in Cambridge"},
]


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=1800,
)
def run_inference_batch():
    """Load Qwen3.5-4B-Base, iterate over all sample images, run prefix-completion inference."""
    import os
    import torch
    from PIL import Image as PILImage
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # ------------------------------------------------------------------
    # Load Qwen3.5-4B-Base
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
    # Discover images
    # ------------------------------------------------------------------
    image_files = sorted(
        f for f in os.listdir(SAMPLES_DIR)
        if f.endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(image_files)} images in {SAMPLES_DIR}")

    results = []

    for img_file in image_files:
        idx = int(img_file.split("_")[1].split(".")[0])  # sample_003.jpg -> 3
        meta = SAMPLE_METADATA[idx] if idx < len(SAMPLE_METADATA) else {
            "category": "Unknown",
            "prompt_context": "a scene",
            "expected": "",
        }

        img_path = os.path.join(SAMPLES_DIR, img_file)
        img = PILImage.open(img_path).convert("RGB")

        # Qwen3.5 base-model prefix completion (no chat template).
        # The processor expects <|image_pad|> in the text to know where to
        # inject vision tokens. We wrap it in <|vision_start|>...<|vision_end|>.
        prompt = (
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"A photograph showing {meta['prompt_context']}. This image contains"
        )

        try:
            inputs = processor(
                text=[prompt],
                images=[img],
                return_tensors="pt",
            )
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                # Greedy decoding, no penalties, per plan.md:
                # "temperature=0, no penalties (adding presence_penalty
                # causes language mixing)"
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                    or processor.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (skip the input)
            input_len = inputs["input_ids"].shape[1]
            completion = processor.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip()

            print(f"[{idx:02d}] {meta['category']}: {completion[:80]}")

        except Exception as e:
            completion = f"[ERROR] {e}"
            print(f"[{idx:02d}] {meta['category']}: ERROR - {e}")

        results.append({
            "sample": idx,
            "filename": img_file,
            "category": meta["category"],
            "prompt_context": meta["prompt_context"],
            "prompt": prompt,
            "expected_output": meta["expected"],
            "actual_output": completion,
        })

    return results


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------
def compute_stats(results):
    """Compute basic aggregate statistics from the results."""
    total = len(results)
    errors = sum(1 for r in results if r["actual_output"].startswith("[ERROR]"))
    successful = total - errors

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"n": 0, "errors": 0, "avg_output_len": 0.0}
        categories[cat]["n"] += 1
        if r["actual_output"].startswith("[ERROR]"):
            categories[cat]["errors"] += 1
        else:
            categories[cat]["avg_output_len"] += len(r["actual_output"])

    for cat, s in categories.items():
        ok = s["n"] - s["errors"]
        s["avg_output_len"] = s["avg_output_len"] / max(ok, 1)

    return {
        "model_id": MODEL_ID,
        "total_samples": total,
        "successful": successful,
        "errors": errors,
        "per_category": categories,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("Running inference on Modal GPU...")
    results = run_inference_batch.remote()

    # Save raw results
    with open("experiment_results_raw.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results -> experiment_results_raw.json ({len(results)} samples)")

    # Save stats
    stats = compute_stats(results)
    with open("experiment_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Statistics  -> experiment_stats.json")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model:      {stats['model_id']}")
    print(f"Samples:    {stats['total_samples']}")
    print(f"Successful: {stats['successful']}")
    print(f"Errors:     {stats['errors']}")
    print(f"\nPer-category:")
    for cat, s in stats["per_category"].items():
        print(f"  {cat}: {s['n']} samples, {s['errors']} errors, "
              f"avg output len {s['avg_output_len']:.0f} chars")
