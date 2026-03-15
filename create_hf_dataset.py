"""
Create a HuggingFace dataset from 10 random samples of each locally-run evaluation.

Datasets included:
  1. MMStar (mmstar_results.json)
  2. VLMs-Are-Blind (vlmsareblind_results.json)
  3. SPHERE-VLM (sphere_results.json)
  4. PACBench (pacbench_results.json)
  5. Order Sensitivity / MMStar (order_sensitivity_results.json)

Each row contains:
  - dataset: source dataset name
  - task: task/category within that dataset
  - question: the prompt/question shown to the model
  - expected_answer: the ground-truth answer
  - model_answer: the model's extracted/parsed prediction
  - model_raw_output: the full raw text the model generated
  - correct: whether the model was correct
  - image: the sample image (PIL Image from base64)
  - extra: any additional metadata specific to that dataset
"""

import json
import random
import base64
import io
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage

SEED = 42
N_SAMPLES = 10
BASE_DIR = Path(__file__).parent

random.seed(SEED)


def b64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def load_json(name: str):
    path = BASE_DIR / name
    with open(path, "r") as f:
        return json.load(f)


def sample_n(items: list, n: int) -> list:
    """Randomly sample n items (or all if fewer than n)."""
    if len(items) <= n:
        return items
    return random.sample(items, n)


# ── 1. MMStar ────────────────────────────────────────────────────────────────
def process_mmstar(results: list) -> list:
    sampled = sample_n(results, N_SAMPLES)
    rows = []
    for r in sampled:
        rows.append({
            "dataset": "MMStar",
            "task": f"{r['category']} / {r['l2_category']}",
            "question": r["question"],
            "expected_answer": r["ground_truth"],
            "model_answer": r.get("predicted_letter", ""),
            "model_raw_output": r.get("model_output", ""),
            "correct": r["correct"],
            "image": b64_to_pil(r["image_b64"]),
            "extra": "",
        })
    return rows


# ── 2. VLMs-Are-Blind ────────────────────────────────────────────────────────
def process_vlmsareblind(results: list) -> list:
    sampled = sample_n(results, N_SAMPLES)
    rows = []
    for r in sampled:
        rows.append({
            "dataset": "VLMs-Are-Blind",
            "task": r["task"],
            "question": r["prompt"],
            "expected_answer": r["ground_truth"],
            "model_answer": r.get("predicted_answer", ""),
            "model_raw_output": r.get("model_output", ""),
            "correct": r["correct"],
            "image": b64_to_pil(r["image_b64"]),
            "extra": "",
        })
    return rows


# ── 3. SPHERE-VLM ────────────────────────────────────────────────────────────
def process_sphere(results: list) -> list:
    sampled = sample_n(results, N_SAMPLES)
    rows = []
    for r in sampled:
        opts = r.get("options", [])
        opts_str = "; ".join(opts) if opts else "(open-ended)"
        rows.append({
            "dataset": "SPHERE-VLM",
            "task": f"{r['config']} [{r['group']}]",
            "question": r["question"],
            "expected_answer": r["ground_truth"],
            "model_answer": r.get("predicted_answer", ""),
            "model_raw_output": r.get("model_output", ""),
            "correct": r["correct"],
            "image": b64_to_pil(r["image_b64"]),
            "extra": f"viewpoint={r.get('viewpoint','')}, format={r.get('answer_format','')}, options=[{opts_str}]",
        })
    return rows


# ── 4. PACBench ───────────────────────────────────────────────────────────────
def process_pacbench(results: list) -> list:
    flat = []
    for r in results:
        img = b64_to_pil(r["image_b64"])
        for pr in r["property_results"]:
            flat.append({
                "dataset": "PACBench",
                "task": pr["property"],
                "question": f"[{pr['property']}] options: {pr['options']}",
                "expected_answer": pr["ground_truth"],
                "model_answer": pr["predicted"],
                "model_raw_output": pr.get("model_output", ""),
                "correct": pr["correct"],
                "image": img,
                "extra": f"image_file={r.get('image_file','')}",
            })
    sampled = sample_n(flat, N_SAMPLES)
    return sampled


# ── 5. Order Sensitivity ─────────────────────────────────────────────────────
def process_order_sensitivity(results: list) -> list:
    sampled = sample_n(results, N_SAMPLES)
    rows = []
    for r in sampled:
        rows.append({
            "dataset": "Order-Sensitivity (MMStar)",
            "task": f"{r['category']} / {r['l2_category']}",
            "question": r["question"],
            "expected_answer": r["ground_truth"],
            "model_answer": f"img_first={r.get('img_first_pred','')}, txt_first={r.get('txt_first_pred','')}",
            "model_raw_output": f"img_first: {r.get('img_first_output','')[:200]} | txt_first: {r.get('txt_first_output','')[:200]}",
            "correct": r.get("img_first_correct", False) and r.get("txt_first_correct", False),
            "image": b64_to_pil(r["image_b64"]),
            "extra": f"prediction_changed={r.get('prediction_changed','')}, flipped_correctness={r.get('flipped_correctness','')}",
        })
    return rows


def main():
    print("Loading result files...")
    mmstar = load_json("mmstar_results.json")
    vlmsareblind = load_json("vlmsareblind_results.json")
    sphere = load_json("sphere_results.json")
    pacbench = load_json("pacbench_results.json")
    order_sens = load_json("order_sensitivity_results.json")

    print(f"  MMStar:            {len(mmstar)} samples")
    print(f"  VLMs-Are-Blind:    {len(vlmsareblind)} samples")
    print(f"  SPHERE-VLM:        {len(sphere)} samples")
    print(f"  PACBench:          {len(pacbench)} images")
    print(f"  Order Sensitivity: {len(order_sens)} samples")

    all_rows = []
    all_rows.extend(process_mmstar(mmstar))
    all_rows.extend(process_vlmsareblind(vlmsareblind))
    all_rows.extend(process_sphere(sphere))
    all_rows.extend(process_pacbench(pacbench))
    all_rows.extend(process_order_sensitivity(order_sens))

    print(f"\nTotal rows in dataset: {len(all_rows)}")

    # Build HuggingFace Dataset
    ds = Dataset.from_list(all_rows)
    ds = ds.cast_column("image", HFImage())

    print("\nDataset schema:")
    print(ds)
    print()

    # Print a summary table
    print(f"{'Dataset':<32} {'Task':<45} {'OK?':<5} {'Expected':<25} {'Model Answer':<30}")
    print("-" * 137)
    for row in all_rows:
        mark = "Y" if row["correct"] else "N"
        exp = str(row["expected_answer"])[:23]
        ans = str(row["model_answer"])[:28]
        print(f"{row['dataset']:<32} {row['task'][:43]:<45} {mark:<5} {exp:<25} {ans:<30}")

    # Save locally
    output_dir = BASE_DIR / "hf_blindspot_samples"
    ds.save_to_disk(str(output_dir))
    print(f"\nSaved dataset locally to: {output_dir}")

    # Push to HuggingFace Hub
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/qwen3.5-4b-base-blindspot-samples"

        print(f"\nPushing to HuggingFace Hub as: {repo_id}")
        ds.push_to_hub(repo_id, private=False)
        print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\nCould not push to Hub: {e}")
        print("You can push manually with:")
        print(f'  from datasets import load_from_disk')
        print(f'  ds = load_from_disk("{output_dir}")')
        print(f'  ds.push_to_hub("YOUR_USERNAME/qwen3.5-4b-base-blindspot-samples")')


if __name__ == "__main__":
    main()
