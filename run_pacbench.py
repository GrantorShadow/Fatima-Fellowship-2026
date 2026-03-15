"""
Run Qwen3.5-4B-Base inference on PAC Bench (Physical Attribute Classification).

Loads 543 labeled images from lens-lab/pacbench, asks the model to classify
12 physical properties per image, and compares against majority-vote ground
truth from human annotators.

PAC Bench tests whether VLMs understand physical attributes of objects:
  12 properties: CAPACITY, COLOR, COMPLEXITY, CONSUMABILITY, CONTENTS,
                 DENSITY, HARDNESS, ORIENTATION, SEALING, STICKINESS,
                 THICKNESS, WEIGHT
  543 labeled open_images with majority-vote annotations
  Each property is multiple-choice (2-5 options per property)

Usage:
    modal run run_pacbench.py                  # 100 images (default)
    modal run run_pacbench.py --limit 50       # 50 images
    modal run run_pacbench.py --limit 0        # All 543 images
"""

import modal
import json

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-pacbench")

MODEL_ID = "Qwen/Qwen3.5-4B-Base"

PROPERTIES = [
    "CAPACITY", "COLOR", "COMPLEXITY", "CONSUMABILITY", "CONTENTS",
    "DENSITY", "HARDNESS", "ORIENTATION", "SEALING", "STICKINESS",
    "THICKNESS", "WEIGHT",
]

# The valid label choices per property (excluding Don't Know / Not Applicable)
PROPERTY_OPTIONS = {
    "CAPACITY": [
        "Containable: Hollow, Enclosable",
        "Non-containable: Solid, Unperforated",
    ],
    "COLOR": [
        "Multicolored: Gradient, Striped",
        "Metallic: Glossy, Shiny",
        "Monochromatic: Single Color, Neutral",
        "Matte: Flat, Dull",
    ],
    "COMPLEXITY": [
        "Multi-object: Assembled, Interconnected",
        "Simple: Single-unit, Monolithic",
    ],
    "CONSUMABILITY": [
        "Non-consumable: Reusable, Permanent",
        "Consumable: Edible, Burnable, Disposable",
    ],
    "CONTENTS": [
        "Contains: Filled, Occupied",
        "Empty: Vacant, Void",
    ],
    "DENSITY": [
        "High-density: Dense, Compact",
        "Low-density: Lightweight, Buoyant",
    ],
    "HARDNESS": [
        "Hard: Solid, Rigid",
        "Brittle: Fragile, Breakable",
        "Soft: Plush, Flexible",
    ],
    "ORIENTATION": [
        "Vertical: Upright, Standing",
        "Horizontal: Flat, Reclined",
        "Multi-directional: Rotational, Adjustable",
    ],
    "SEALING": [
        "Unsealed: Open, can leak",
        "Sealed: Airtight, Watertight",
    ],
    "STICKINESS": [
        "Non-sticky: Smooth, Slippery",
        "Sticky: Adhesive, Tacky",
    ],
    "THICKNESS": [
        "Thick: Sturdy, Bulky",
        "Medium: Standard Thickness, Balanced",
        "Thin: Slim, Minimal Thickness",
    ],
    "WEIGHT": [
        "Heavy: Bulky, Dense",
        "Light: Featherweight, Lightweight",
        "Medium: Moderate, Balanced",
    ],
}

# Short question templates per property
PROPERTY_QUESTIONS = {
    "CAPACITY": "What is the capacity of this object?",
    "COLOR": "What best describes the color appearance of this object?",
    "COMPLEXITY": "What is the structural complexity of this object?",
    "CONSUMABILITY": "Is this object consumable or non-consumable?",
    "CONTENTS": "Does this object contain something or is it empty?",
    "DENSITY": "What is the density of this object?",
    "HARDNESS": "What is the hardness of this object?",
    "ORIENTATION": "What is the orientation of this object?",
    "SEALING": "Is this object sealed or unsealed?",
    "STICKINESS": "Is this object sticky or non-sticky?",
    "THICKNESS": "What is the thickness of this object?",
    "WEIGHT": "What is the weight of this object?",
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
        "requests>=2.28.0",
    )
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers.git",
    )
    # Pre-download model weights
    .run_commands(
        f"python -c \""
        f"from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration; "
        f"AutoProcessor.from_pretrained('{MODEL_ID}', trust_remote_code=True); "
        f"Qwen3_5ForConditionalGeneration.from_pretrained('{MODEL_ID}', trust_remote_code=True)"
        f"\""
    )
)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_answer(completion: str, options: list) -> str:
    """Match model completion against the MCQ option list."""
    completion_lower = completion.strip().lower()

    # Try: model outputs "(A)" or "A)" or just "A"
    import re
    letter_match = re.match(r"\(?([A-Z])\)?", completion.strip())
    if letter_match:
        idx = ord(letter_match.group(1)) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]

    # Try: completion contains one of the option keywords
    # Use the short keyword before the colon (e.g. "Heavy" from "Heavy: Bulky, Dense")
    for opt in options:
        short = opt.split(":")[0].strip().lower()
        if short in completion_lower:
            return opt

    # Try: full option string match
    for opt in options:
        if opt.lower() in completion_lower:
            return opt

    return completion.strip()[:60]


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Case-insensitive comparison."""
    return predicted.strip().lower() == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    timeout=21600,
    memory=32768,
)
def run_pacbench_inference(limit: int = 100):
    """Load Qwen3.5-4B-Base, run inference on PAC Bench images."""
    import base64
    import io
    import re
    import torch
    import pandas as pd
    import requests
    from PIL import Image as PILImage
    from collections import defaultdict
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    HF_BASE = "https://huggingface.co/datasets/lens-lab/pacbench/resolve/main"

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
    # Build ground truth label map: suffix_key -> {property: label}
    # ------------------------------------------------------------------
    print("Loading PAC Bench ground truth CSVs...")

    def extract_suffix(path):
        m = re.search(r"_([a-f0-9]+)\.jpg", str(path))
        return m.group(1) if m else None

    all_labels = {}
    for prop in PROPERTIES:
        url = f"{HF_BASE}/ground_truth/property_{prop}_.csv"
        df = pd.read_csv(url)
        df["suffix"] = df["image"].apply(extract_suffix)
        df = df.dropna(subset=["suffix"])
        df["choice"] = df["choice"].fillna("Unknown")
        majority = df.groupby("suffix")["choice"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else str(x.iloc[0])
        ).to_dict()
        for k, v in majority.items():
            if k not in all_labels:
                all_labels[k] = {}
            all_labels[k][prop] = v
        print(f"  {prop}: {len(majority)} images")

    # ------------------------------------------------------------------
    # Build list of image URLs from open_images/
    # ------------------------------------------------------------------
    from huggingface_hub import list_repo_tree
    open_imgs = [
        f.rfilename for f in list_repo_tree(
            "lens-lab/pacbench", repo_type="dataset", recursive=True
        )
        if hasattr(f, "rfilename") and f.rfilename.startswith("open_images/")
    ]

    suffix_to_file = {}
    for f in open_imgs:
        m = re.search(r"_([a-f0-9]+)\.jpg", f)
        if m:
            suffix_to_file[m.group(1)] = f

    # Only use images that have BOTH labels and files
    valid_keys = sorted(set(all_labels.keys()) & set(suffix_to_file.keys()))
    total = len(valid_keys)
    print(f"\nPAC Bench: {total} images with labels and files")

    # Filter "Don't Know" / "Not Applicable" / "Unknown" labels
    skip_labels = {"don't know", "not applicable", "unknown", "nan"}

    if limit > 0 and limit < total:
        # Evenly spaced sampling
        step = max(1, total // limit)
        valid_keys = valid_keys[::step][:limit]
        print(f"Using {len(valid_keys)} sampled images")

    # ------------------------------------------------------------------
    # Run inference: for each image, ask all 12 property questions
    # ------------------------------------------------------------------
    results = []
    property_correct = defaultdict(lambda: {"total": 0, "correct": 0})
    total_correct = 0
    total_questions = 0

    for img_idx, suffix_key in enumerate(valid_keys):
        # Download image
        img_path = suffix_to_file[suffix_key]
        img_url = f"{HF_BASE}/{img_path}"
        try:
            resp = requests.get(img_url, timeout=30)
            resp.raise_for_status()
            img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"  Failed to download {img_path}: {e}")
            continue

        # Thumbnail for results
        thumb = img.copy()
        thumb.thumbnail((128, 128))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        labels = all_labels[suffix_key]
        image_results = []

        for prop in PROPERTIES:
            gt = labels.get(prop, "Unknown")
            # Skip if GT is ambiguous
            if gt.strip().lower() in skip_labels:
                continue

            options = PROPERTY_OPTIONS[prop]
            # Skip if GT is not one of the valid options
            if gt not in options:
                continue

            question = PROPERTY_QUESTIONS[prop]
            options_str = "\n".join(
                f"({chr(65+j)}) {opt}" for j, opt in enumerate(options)
            )
            prompt = (
                "<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}\n{options_str}\n"
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

            predicted = extract_answer(completion, options)
            is_correct = check_answer(predicted, gt)

            property_correct[prop]["total"] += 1
            if is_correct:
                property_correct[prop]["correct"] += 1
                total_correct += 1
            total_questions += 1

            image_results.append({
                "property": prop,
                "ground_truth": gt,
                "predicted": predicted,
                "model_output": completion,
                "correct": is_correct,
                "options": options,
            })

        results.append({
            "index": img_idx,
            "suffix_key": suffix_key,
            "image_file": img_path,
            "image_b64": img_b64,
            "property_results": image_results,
            "image_accuracy": (
                sum(1 for r in image_results if r["correct"]) / len(image_results) * 100
                if image_results else 0
            ),
        })

        if (img_idx + 1) % 5 == 0 or img_idx == 0:
            acc = total_correct / total_questions * 100 if total_questions > 0 else 0
            n_props = len(image_results)
            img_acc = results[-1]["image_accuracy"]
            print(f"[{img_idx+1:4d}/{len(valid_keys)}] overall={acc:.1f}% | "
                  f"this_img={img_acc:.0f}% ({n_props} props)")

    # ------------------------------------------------------------------
    # Compute stats
    # ------------------------------------------------------------------
    accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0

    prop_stats = {}
    for prop in PROPERTIES:
        s = property_correct[prop]
        prop_stats[prop] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
        }

    # Group properties by type
    perceptual = ["COLOR", "ORIENTATION", "THICKNESS", "COMPLEXITY"]
    physical = ["WEIGHT", "DENSITY", "HARDNESS", "STICKINESS"]
    functional = ["CAPACITY", "CONTENTS", "SEALING", "CONSUMABILITY"]

    def group_acc(props_list):
        t = sum(property_correct[p]["total"] for p in props_list)
        c = sum(property_correct[p]["correct"] for p in props_list)
        return round(c / t * 100, 1) if t > 0 else 0

    group_stats = {
        "perceptual": {"props": perceptual, "accuracy": group_acc(perceptual)},
        "physical": {"props": physical, "accuracy": group_acc(physical)},
        "functional": {"props": functional, "accuracy": group_acc(functional)},
    }

    stats = {
        "model_id": MODEL_ID,
        "dataset": "lens-lab/pacbench",
        "total_images": len(results),
        "total_questions": total_questions,
        "total_correct": total_correct,
        "accuracy_pct": round(accuracy, 2),
        "per_property": prop_stats,
        "per_group": group_stats,
    }

    print(f"\n{'='*60}")
    print(f"OVERALL: {total_correct}/{total_questions} = {accuracy:.1f}%")
    print(f"{'='*60}")
    print(f"\nPer property:")
    for p, s in sorted(prop_stats.items(), key=lambda x: x[1]["accuracy"]):
        print(f"  {p:20s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}%")
    print(f"\nPer group:")
    for g, s in group_stats.items():
        print(f"  {g}: {s['accuracy']}%")

    return {"results": results, "stats": stats}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(limit: int = 100):
    print(f"Running PAC Bench inference ({limit} images) on Modal H100...")
    output = run_pacbench_inference.remote(limit=limit)

    results = output["results"]
    stats = output["stats"]

    with open("pacbench_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> pacbench_results.json ({len(results)} images)")

    with open("pacbench_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats   -> pacbench_stats.json")

    print(f"\n{'='*60}")
    print(f"PAC Bench RESULTS")
    print(f"{'='*60}")
    print(f"Model:     {stats['model_id']}")
    print(f"Images:    {stats['total_images']}")
    print(f"Questions: {stats['total_questions']}")
    print(f"Correct:   {stats['total_correct']}")
    print(f"Accuracy:  {stats['accuracy_pct']}%")

    print(f"\nPer property:")
    for p, s in sorted(stats["per_property"].items(), key=lambda x: x[1]["accuracy"]):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {p:20s} {s['correct']:3d}/{s['total']:3d} = {s['accuracy']:5.1f}% {bar}")

    print(f"\nPer group:")
    for g, s in sorted(stats["per_group"].items(), key=lambda x: x[1]["accuracy"]):
        bar = "#" * int(s["accuracy"] / 5)
        print(f"  {g:15s} = {s['accuracy']:5.1f}% {bar}")

    blindspots = [
        (p, s) for p, s in stats["per_property"].items()
        if s["accuracy"] < 40.0 and s["total"] > 0
    ]
    if blindspots:
        print(f"\nBLIND SPOTS (properties < 40% accuracy):")
        for p, s in blindspots:
            print(f"  ** {p}: {s['accuracy']}%")
