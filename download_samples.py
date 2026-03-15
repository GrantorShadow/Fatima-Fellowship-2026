"""
Download sample images for blind spot analysis.

Fetches curated edge-case images from Wikimedia Commons into a persistent
Modal Volume so they're available for GPU inference without re-downloading.

Usage:
    modal run download_samples.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
app = modal.App("fatima-download-samples")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests>=2.31.0",
    "pillow>=10.0.0",
)

volume = modal.Volume.from_name("fatima-sample-data", create_if_missing=True)
VOLUME_PATH = "/data"
SAMPLES_DIR = f"{VOLUME_PATH}/samples"

# ---------------------------------------------------------------------------
# 15 verified Wikimedia Commons URLs -- all confirmed to exist via API
# Each tuple: (primary_url, backup_url_or_None)
# ---------------------------------------------------------------------------
SAMPLE_IMAGES = [
    {
        "category": "Spatial Reasoning (Reflections)",
        "prompt_context": "mirror reflection",
        "expected": "A mountain reflected symmetrically in a still lake, showing both the real mountain and its mirror image in the water",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg/640px-Mount_Hood_reflected_in_Mirror_Lake%2C_Oregon.jpg",
        "backup_url": None,
    },
    {
        "category": "Transparency / Refraction",
        "prompt_context": "glass of water",
        "expected": "A glass of water with a visible refraction effect distorting what is behind or inside the glass",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Refraction_through_a_glass_of_water.jpg/640px-Refraction_through_a_glass_of_water.jpg",
        "backup_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Light_refraction_in_glass_with_spoon.jpg/640px-Light_refraction_in_glass_with_spoon.jpg",
    },
    {
        "category": "Forced Perspective",
        "prompt_context": "tourist photo",
        "expected": "A person using forced perspective photography to appear to touch or hold a large building in the background",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Tourist_Photo_at_the_Taj_Mahal-01.jpg/640px-Tourist_Photo_at_the_Taj_Mahal-01.jpg",
        "backup_url": None,
    },
    {
        "category": "Counting Overlapping Objects",
        "prompt_context": "pile of coins",
        "expected": "A collection of many overlapping US coins of different denominations spread out, difficult to count precisely",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Assorted_United_States_coins.jpg/640px-Assorted_United_States_coins.jpg",
        "backup_url": None,
    },
    {
        "category": "Text / Handwriting (OCR)",
        "prompt_context": "handwritten document",
        "expected": "A handwritten letter on paper with cursive or script writing that requires OCR-like reading to interpret",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Handwritten_Letter_of_Resignation.jpg/640px-Handwritten_Letter_of_Resignation.jpg",
        "backup_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/EFTA00002659_-_Crumpled_white_sheet_of_paper_with_faint_stains_and_handwritten_text_on_the_right_side.jpg/640px-EFTA00002659_-_Crumpled_white_sheet_of_paper_with_faint_stains_and_handwritten_text_on_the_right_side.jpg",
    },
    {
        "category": "Object Occlusion",
        "prompt_context": "domestic animal",
        "expected": "A tabby cat sitting upright and looking at the camera",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/480px-Cat_November_2010-1a.jpg",
        "backup_url": None,
    },
    {
        "category": "Shadows / Silhouettes",
        "prompt_context": "shadow puppet",
        "expected": "A traditional shadow puppet figure used in theatrical performances, showing intricate cut-out details",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/04/Nang_Sbek_Thom_%28Cambodian_shadow_puppet%29_at_the_Musical_Instruments_Museum.jpg",
        "backup_url": None,
    },
    {
        "category": "Camouflage / Hidden Object",
        "prompt_context": "leaves on branch",
        "expected": "A leaf insect camouflaged among real leaves, extremely difficult to distinguish from the surrounding foliage",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Leaf-Insect.jpg/640px-Leaf-Insect.jpg",
        "backup_url": None,
    },
    {
        "category": "Out-of-Context Object",
        "prompt_context": "vehicle on highway",
        "expected": "A large boat being hauled on a truck trailer driving down a road",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Boat-Hauling_Ram_Truck_in_Ridge_Manor_West%2C_FL-1.jpg/640px-Boat-Hauling_Ram_Truck_in_Ridge_Manor_West%2C_FL-1.jpg",
        "backup_url": None,
    },
    {
        "category": "Extreme Aspect Ratio (Panorama)",
        "prompt_context": "city skyline panorama",
        "expected": "An extremely wide panoramic photograph of a city skyline at twilight",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Montreal_Twilight_Panorama_2006.jpg/1280px-Montreal_Twilight_Panorama_2006.jpg",
        "backup_url": None,
    },
    {
        "category": "Low Light / Noise",
        "prompt_context": "dark room",
        "expected": "A very dark room with a single streetlight shining through a window, most details lost in darkness and noise",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Dark_room_with_street_light_shining_through_window_%28night_mode_off%29.jpg/640px-Dark_room_with_street_light_shining_through_window_%28night_mode_off%29.jpg",
        "backup_url": None,
    },
    {
        "category": "Fine-Grained Recognition (Landmark)",
        "prompt_context": "mountain landscape",
        "expected": "Mount Everest's north face as seen from the base camp in Tibet",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/640px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
        "backup_url": None,
    },
    {
        "category": "Iconic Landmark / Spatial Layout",
        "prompt_context": "famous bridge",
        "expected": "The Golden Gate Bridge in San Francisco, a large red suspension bridge spanning across the bay",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/640px-GoldenGateBridge-001.jpg",
        "backup_url": None,
    },
    {
        "category": "Optical Illusion",
        "prompt_context": "checkerboard pattern",
        "expected": "Adelson's checker shadow illusion: a checkerboard with a cylinder casting a shadow, where squares A and B appear different but are the same shade",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Checker_shadow_illusion.svg/640px-Checker_shadow_illusion.svg.png",
        "backup_url": None,
    },
    {
        "category": "Dense Urban Scene",
        "prompt_context": "city skyline",
        "expected": "The Boston skyline viewed from across the Charles River in Cambridge",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Boston_skyline_from_Cambridge_November_2015_panorama_1.jpg/960px-Boston_skyline_from_Cambridge_November_2015_panorama_1.jpg",
        "backup_url": None,
    },
]


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=600)
def download_images():
    """Download all sample images into the Modal Volume."""
    import os
    import json
    import requests
    from PIL import Image as PILImage
    from io import BytesIO

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FatimaFellowship/1.0; +https://huggingface.co)"
    }

    manifest = {"n_samples": len(SAMPLE_IMAGES), "n_downloaded": 0, "samples": []}
    downloaded = 0

    for idx, item in enumerate(SAMPLE_IMAGES):
        filename = f"sample_{idx:03d}.jpg"
        filepath = os.path.join(SAMPLES_DIR, filename)

        # Skip if already downloaded and valid
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"[{idx:02d}] Already exists: {filename}")
            manifest["samples"].append({
                "index": idx,
                "filename": filename,
                "category": item["category"],
                "prompt_context": item["prompt_context"],
                "expected": item["expected"],
                "url": item["url"],
            })
            downloaded += 1
            continue

        # Try primary URL, then backup
        urls_to_try = [item["url"]]
        if item.get("backup_url"):
            urls_to_try.append(item["backup_url"])

        success = False
        for attempt, url in enumerate(urls_to_try):
            try:
                label = "primary" if attempt == 0 else "backup"
                print(f"[{idx:02d}] Downloading ({label}): {item['category']}...")
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()

                # Validate it's a real image, convert to RGB JPEG
                img = PILImage.open(BytesIO(resp.content)).convert("RGB")
                img.save(filepath, "JPEG", quality=95)

                manifest["samples"].append({
                    "index": idx,
                    "filename": filename,
                    "category": item["category"],
                    "prompt_context": item["prompt_context"],
                    "expected": item["expected"],
                    "url": url,
                })
                downloaded += 1
                success = True
                print(f"     Saved {filename} ({img.size[0]}x{img.size[1]})")
                break

            except Exception as e:
                print(f"     FAILED ({label}): {e}")

        if not success:
            manifest["samples"].append({
                "index": idx,
                "filename": filename,
                "category": item["category"],
                "prompt_context": item["prompt_context"],
                "expected": item["expected"],
                "url": item["url"],
                "error": "All download attempts failed",
            })

    manifest["n_downloaded"] = downloaded

    # Write manifest
    manifest_path = os.path.join(VOLUME_PATH, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    volume.commit()

    print(f"\nDone: {downloaded}/{len(SAMPLE_IMAGES)} images downloaded")
    print(f"Manifest: {manifest_path}")
    return downloaded


@app.local_entrypoint()
def main():
    count = download_images.remote()
    print(f"\nDownloaded {count} images to Modal Volume 'fatima-sample-data'")
