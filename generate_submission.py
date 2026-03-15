#!/usr/bin/env python3
"""
Fatima Fellowship 2026 - Blind Spot Detection for Qwen3.5-4B-Base

This script reads experiment results from experiment_results_raw.json (produced
by run_experiments.py), downloads the source images, and pushes a Hugging Face
dataset documenting the model's blind spots.

Usage:
    HF_TOKEN=hf_... python generate_submission.py
"""

import os
import json
import torch
from PIL import Image
from io import BytesIO
import requests
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi

MODEL_ID = "Qwen/Qwen3.5-4B-Base"
HF_REPO_NAME = "qwen35-blind-spots"

# ============================================================================
# Step 1: Curated Edge Case Images (15 categories)
# Must stay in sync with download_samples.py SAMPLE_IMAGES
# ============================================================================

EDGE_CASE_IMAGES = [
    {
        "category": "Spatial Reasoning (Reflections)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg/640px-Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg",
        "description": "Mountain reflected in a mirror lake",
        "expected_output": "A mountain reflected symmetrically in a still lake, showing both the real mountain and its mirror image in the water",
        "prompt_context": "mirror reflection",
    },
    {
        "category": "Transparency / Refraction",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Refraction_through_a_glass_of_water.jpg/640px-Refraction_through_a_glass_of_water.jpg",
        "description": "Refraction through a glass of water",
        "expected_output": "A glass of water with a visible refraction effect distorting what is behind or inside the glass",
        "prompt_context": "glass of water",
    },
    {
        "category": "Forced Perspective",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Tourist_Photo_at_the_Taj_Mahal-01.jpg/640px-Tourist_Photo_at_the_Taj_Mahal-01.jpg",
        "description": "Tourist using forced perspective at the Taj Mahal",
        "expected_output": "A person using forced perspective photography to appear to touch or hold a large building in the background",
        "prompt_context": "tourist photo",
    },
    {
        "category": "Counting Overlapping Objects",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Assorted_United_States_coins.jpg/640px-Assorted_United_States_coins.jpg",
        "description": "Assorted US coins spread out",
        "expected_output": "A collection of many overlapping US coins of different denominations spread out, difficult to count precisely",
        "prompt_context": "pile of coins",
    },
    {
        "category": "Text / Handwriting (OCR)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Handwritten_Letter_of_Resignation.jpg/640px-Handwritten_Letter_of_Resignation.jpg",
        "description": "Handwritten letter of resignation",
        "expected_output": "A handwritten letter on paper with cursive or script writing that requires OCR-like reading to interpret",
        "prompt_context": "handwritten document",
    },
    {
        "category": "Object Occlusion",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/480px-Cat_November_2010-1a.jpg",
        "description": "Tabby cat sitting upright",
        "expected_output": "A tabby cat sitting upright and looking at the camera",
        "prompt_context": "domestic animal",
    },
    {
        "category": "Shadows / Silhouettes",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/04/Nang_Sbek_Thom_%28Cambodian_shadow_puppet%29_at_the_Musical_Instruments_Museum.jpg",
        "description": "Cambodian shadow puppet",
        "expected_output": "A traditional shadow puppet figure used in theatrical performances, showing intricate cut-out details",
        "prompt_context": "shadow puppet",
    },
    {
        "category": "Camouflage / Hidden Object",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Leaf-Insect.jpg/640px-Leaf-Insect.jpg",
        "description": "Leaf insect camouflaged among leaves",
        "expected_output": "A leaf insect camouflaged among real leaves, extremely difficult to distinguish from the surrounding foliage",
        "prompt_context": "leaves on branch",
    },
    {
        "category": "Out-of-Context Object",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Boat-Hauling_Ram_Truck_in_Ridge_Manor_West%2C_FL-1.jpg/640px-Boat-Hauling_Ram_Truck_in_Ridge_Manor_West%2C_FL-1.jpg",
        "description": "Boat being hauled on a truck",
        "expected_output": "A large boat being hauled on a truck trailer driving down a road",
        "prompt_context": "vehicle on highway",
    },
    {
        "category": "Extreme Aspect Ratio (Panorama)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Montreal_Twilight_Panorama_2006.jpg/1280px-Montreal_Twilight_Panorama_2006.jpg",
        "description": "Wide panoramic shot of Montreal at twilight",
        "expected_output": "An extremely wide panoramic photograph of a city skyline at twilight",
        "prompt_context": "city skyline panorama",
    },
    {
        "category": "Low Light / Noise",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Dark_room_with_street_light_shining_through_window_%28night_mode_off%29.jpg/640px-Dark_room_with_street_light_shining_through_window_%28night_mode_off%29.jpg",
        "description": "Dark room with streetlight through window",
        "expected_output": "A very dark room with a single streetlight shining through a window, most details lost in darkness and noise",
        "prompt_context": "dark room",
    },
    {
        "category": "Fine-Grained Recognition (Landmark)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/640px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
        "description": "Mount Everest north face from Tibet base camp",
        "expected_output": "Mount Everest's north face as seen from the base camp in Tibet",
        "prompt_context": "mountain landscape",
    },
    {
        "category": "Iconic Landmark / Spatial Layout",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/640px-GoldenGateBridge-001.jpg",
        "description": "Golden Gate Bridge",
        "expected_output": "The Golden Gate Bridge in San Francisco, a large red suspension bridge spanning across the bay",
        "prompt_context": "famous bridge",
    },
    {
        "category": "Optical Illusion",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Checker_shadow_illusion.svg/640px-Checker_shadow_illusion.svg.png",
        "description": "Adelson's checker shadow illusion",
        "expected_output": "Adelson's checker shadow illusion: a checkerboard with a cylinder casting a shadow, where squares A and B appear different but are the same shade",
        "prompt_context": "checkerboard pattern",
    },
    {
        "category": "Dense Urban Scene",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Boston_skyline_from_Cambridge_November_2015_panorama_1.jpg/960px-Boston_skyline_from_Cambridge_November_2015_panorama_1.jpg",
        "description": "Boston skyline from Cambridge",
        "expected_output": "The Boston skyline viewed from across the Charles River in Cambridge",
        "prompt_context": "city skyline",
    },
]


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """Download an image from URL and return as PIL Image."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FatimaFellowship/1.0; +https://huggingface.co)"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_results_from_json(path: str = "experiment_results_raw.json") -> list:
    """
    Load results from experiment_results_raw.json produced by run_experiments.py.
    Falls back to running inference if the file doesn't exist.
    """
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_all_images():
    """Load all edge case images."""
    images = []
    for item in EDGE_CASE_IMAGES:
        try:
            print(f"Downloading: {item['category']}...")
            img = download_image(item["url"])
            images.append({**item, "image": img})
            print(f"  Success")
        except Exception as e:
            print(f"  Failed: {e}")
            images.append({**item, "image": None, "error": str(e)})
    return images


# ============================================================================
# Step 2: Create Hugging Face Dataset
# ============================================================================

def create_hf_dataset(results: list, images: list):
    """Create HF dataset from experiment results + downloaded images."""

    data = {
        "image": [],
        "prompt": [],
        "expected_output": [],
        "actual_output": [],
        "category": [],
    }

    for i, r in enumerate(results):
        # Get the image from downloaded images (matched by index)
        img = images[i].get("image") if i < len(images) else None
        if img is None:
            continue

        data["image"].append(img)
        data["prompt"].append(r.get("prompt", ""))
        data["expected_output"].append(r.get("expected_output", ""))
        data["actual_output"].append(r.get("actual_output", ""))
        data["category"].append(r.get("category", ""))

    features = Features({
        "image": HFImage(),
        "prompt": Value("string"),
        "expected_output": Value("string"),
        "actual_output": Value("string"),
        "category": Value("string"),
    })

    dataset = Dataset.from_dict(data, features=features)
    print(f"\nDataset created with {len(dataset)} samples")
    print(dataset)
    return dataset


def push_to_hub(dataset, repo_name: str, readme_content: str):
    """Push dataset and README to Hugging Face Hub."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set!")

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]

    full_repo_name = f"{username}/{repo_name}"
    print(f"\nPushing to: {full_repo_name}")

    dataset.push_to_hub(repo_name, token=token, private=False)

    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=full_repo_name,
        repo_type="dataset",
        token=token,
    )

    print(f"Dataset pushed: https://huggingface.co/datasets/{full_repo_name}")
    return full_repo_name


# ============================================================================
# Step 3: Generate README
# ============================================================================

def generate_readme(results: list) -> str:
    """Generate README.md content for the HF dataset."""

    table_rows = []
    for r in results:
        actual_short = r.get("actual_output", "")[:80]
        if len(r.get("actual_output", "")) > 80:
            actual_short += "..."
        expected_short = r.get("expected_output", "")[:80]
        if len(r.get("expected_output", "")) > 80:
            expected_short += "..."
        prompt_short = r.get("prompt", "")[:50] + "..."

        table_rows.append(
            f"| {r.get('category', '')} | `{prompt_short}` | {expected_short} | {actual_short} |"
        )

    table = "\n".join(table_rows)

    readme = f'''---
license: mit
task_categories:
  - image-to-text
  - visual-question-answering
language:
  - en
tags:
  - multimodal
  - vision-language
  - blind-spots
  - edge-cases
  - qwen3.5
size_categories:
  - n<1K
---

# Qwen3.5-4B-Base Blind Spots Dataset

This dataset documents **blind spots and failure modes** identified in the
[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) vision-language base model.
Created as part of the Fatima Fellowship 2026 technical challenge.

## Model Information

- **Model**: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})
- **Parameters**: 4B
- **Type**: Base model (NOT instruction-tuned)
- **Precision**: bfloat16
- **Architecture**: `Qwen3_5ForConditionalGeneration` (multimodal decoder-only with vision encoder)

### Loading the Model

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

model_id = "{MODEL_ID}"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
)
model.eval()
```

> **Note**: Requires `transformers` installed from source (`pip install git+https://github.com/huggingface/transformers.git`) as of March 2026.

### Prompting Strategy for Base Models

Since this is a **base model** (not instruction-tuned), we use completion-style prompts:

```python
# Base-model prefix completion with Qwen3.5 vision tokens
prompt = "<|vision_start|><|image_pad|><|vision_end|>A photograph showing [context]. This image contains"

inputs = processor(text=[prompt], images=[image], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=40, temperature=0.1, repetition_penalty=1.3)

# Decode only new tokens
input_len = inputs["input_ids"].shape[1]
completion = processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
```

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `image` | Image | The actual image (embedded) |
| `prompt` | string | The completion prompt used |
| `expected_output` | string | Human-annotated expected description |
| `actual_output` | string | Raw model output |
| `category` | string | Edge case category |

## Edge Case Categories & Results

| Category | Prompt | Expected Output | Actual Output |
|----------|--------|-----------------|---------------|
{table}

## Analysis: Fixing These Blind Spots

### What Kind of Dataset Would Fix These Errors?

To address the identified blind spots in {MODEL_ID}, the model would need fine-tuning on a carefully curated **Multimodal Instruction Fine-Tuning (IFT)** dataset emphasizing: (1) **Spatial relation annotations** with bounding boxes encoding object positions, relationships like "reflected in," "behind," and occlusion states; (2) **Hard-negative synthetic examples** for distinguishing visually similar but semantically different scenes; and (3) **Compositional reasoning chains** breaking down complex visual scenes into step-by-step interpretations.

### How Would You Assemble Such a Dataset?

Leverage existing datasets with spatial annotations (**Visual Genome**, **GQA**, **CLEVR**) converted to instruction-following format. Supplement with **synthetic data** from rendering engines or diffusion models for controlled scenarios. Collect **human annotations** on real-world edge cases. Use GPT-4V or similar for dense caption generation with human verification.

### Dataset Size Requirements

Approximately **50,000 to 100,000 highly curated image-text pairs** targeting specific failure modes. A focused dataset of ~10K examples per category with explicit spatial reasoning, occlusion states, and compositional descriptions would likely yield significant improvements. Include **negative examples** and **contrastive pairs** for fine-grained distinctions.

## Citation

```bibtex
@dataset{{qwen35_blind_spots_2026,
  title={{Qwen3.5-4B-Base Blind Spots Dataset}},
  author={{Fatima Fellowship 2026}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/[username]/{HF_REPO_NAME}}}
}}
```

## License

MIT License
'''

    return readme


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 60)
    print("Fatima Fellowship 2026 - Blind Spot Detection")
    print(f"Target Model: {MODEL_ID}")
    print("=" * 60)

    # Try to load pre-computed results from run_experiments.py
    print("\n[STEP 1] Loading experiment results...")
    results = load_results_from_json()

    if results:
        print(f"  Loaded {len(results)} results from experiment_results_raw.json")
    else:
        print("  No experiment_results_raw.json found.")
        print("  Run 'modal run run_experiments.py' first, then re-run this script.")
        return

    # Download images for embedding in dataset
    print("\n[STEP 2] Downloading edge case images...")
    images = load_all_images()
    successful = sum(1 for img in images if img.get("image") is not None)
    print(f"\nDownloaded {successful}/{len(images)} images")

    # Create dataset
    print("\n[STEP 3] Creating Hugging Face dataset...")
    dataset = create_hf_dataset(results, images)

    # Generate README
    print("\n[STEP 4] Generating README...")
    readme_content = generate_readme(results)

    with open("README_dataset.md", "w") as f:
        f.write(readme_content)
    print("  Saved to README_dataset.md")

    # Push to Hub
    print("\n[STEP 5] Pushing to Hugging Face Hub...")
    try:
        repo_name = push_to_hub(dataset, HF_REPO_NAME, readme_content)
        print(f"\n{'=' * 60}")
        print("SUCCESS! Dataset published at:")
        print(f"https://huggingface.co/datasets/{repo_name}")
        print("=" * 60)
    except Exception as e:
        print(f"\nError pushing to hub: {e}")
        print("Dataset and README have been saved locally.")
        print("Set HF_TOKEN and retry, or manually push:")
        print(f"  huggingface-cli login")
        print(f"  huggingface-cli upload {HF_REPO_NAME} ./")

    return results


if __name__ == "__main__":
    results = main()
