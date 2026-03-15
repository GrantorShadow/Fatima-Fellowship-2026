"""
Download MMStar images via the HF Datasets Server API and generate
a visual results table with the actual test images.

Fetches image URLs from the HF API (no local dataset download needed),
matches them against mmstar_results.json, and produces:
  1. mmstar_analysis.png      -- summary dashboard
  2. mmstar_image_grid.png    -- 10-col grid of ALL samples with images
  3. mmstar_failures.png      -- detailed failure view with images

Usage:
    python3 visualize_mmstar.py
"""

import json
import base64
import io
import os
import textwrap
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PILImage
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
with open("mmstar_results.json") as f:
    results = json.load(f)

with open("mmstar_stats.json") as f:
    stats = json.load(f)

total = len(results)
correct = sum(1 for r in results if r["correct"])
accuracy = correct / total * 100
has_images = "image_b64" in results[0]

# Color palette
BG = "#0d1117"
CARD = "#161b22"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
GRAY = "#8b949e"
WHITE = "#c9d1d9"


def style_ax(ax, title=""):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=GRAY, labelsize=9)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=WHITE, pad=10)


# ---------------------------------------------------------------------------
# Fetch images from HF Datasets Server API (if not already in results)
# ---------------------------------------------------------------------------
CACHE_DIR = "mmstar_images"

if not has_images:
    print("No embedded images in results. Fetching from HF Datasets API...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Fetch in pages of 100
    PAGE_SIZE = 100
    image_urls = {}
    for offset in range(0, total, PAGE_SIZE):
        length = min(PAGE_SIZE, total - offset)
        api_url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset=Lin-Chen/MMStar&config=val&split=val"
            f"&offset={offset}&length={length}"
        )
        print(f"  Fetching rows {offset}..{offset+length-1}")
        resp = requests.get(api_url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for row_data in data["rows"]:
            row = row_data["row"]
            idx = row["index"]
            img_info = row["image"]
            if isinstance(img_info, dict) and "src" in img_info:
                image_urls[idx] = img_info["src"]

    # Download each image
    downloaded = 0
    for i, r in enumerate(results):
        idx = r["index"]
        cache_path = os.path.join(CACHE_DIR, f"{idx}.png")

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
            # Load from cache
            img = PILImage.open(cache_path)
        elif idx in image_urls:
            try:
                resp = requests.get(image_urls[idx], timeout=30)
                resp.raise_for_status()
                img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                # Save thumbnail to cache
                thumb = img.copy()
                thumb.thumbnail((128, 128))
                thumb.save(cache_path, "PNG")
                img = thumb
                downloaded += 1
            except Exception as e:
                print(f"  Failed to download image {idx}: {e}")
                img = PILImage.new("RGB", (128, 128), color=(30, 30, 30))
        else:
            print(f"  No URL for image {idx}")
            img = PILImage.new("RGB", (128, 128), color=(30, 30, 30))

        # Encode to base64 and inject into results
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        r["image_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{total} images")

    has_images = True
    print(f"  Downloaded {downloaded} new images, {total - downloaded} from cache")

    # Save updated results with images
    with open("mmstar_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Updated mmstar_results.json with image_b64 fields")


def decode_image(b64_str):
    return PILImage.open(io.BytesIO(base64.b64decode(b64_str)))


# ---------------------------------------------------------------------------
# Compute derived data
# ---------------------------------------------------------------------------
l2_cats = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    l2 = r["l2_category"]
    l2_cats[l2]["total"] += 1
    if r["correct"]:
        l2_cats[l2]["correct"] += 1

l2_names = sorted(l2_cats.keys(), key=lambda x: l2_cats[x]["correct"] / l2_cats[x]["total"])
l2_accs = [l2_cats[n]["correct"] / l2_cats[n]["total"] * 100 for n in l2_names]
l2_totals = [l2_cats[n]["total"] for n in l2_names]

letters = ["A", "B", "C", "D"]
conf_matrix = np.zeros((4, 4), dtype=int)
for r in results:
    gt = r["ground_truth"].strip().upper()
    pred = r["predicted_letter"].strip().upper()
    if gt in letters and pred in letters:
        conf_matrix[letters.index(gt)][letters.index(pred)] += 1

gt_dist = Counter(r["ground_truth"].strip().upper() for r in results)
pred_dist = Counter(r["predicted_letter"].strip().upper() for r in results)
failures = [r for r in results if not r["correct"]]
failure_l2 = Counter(r["l2_category"] for r in failures)

nrows_grid = (total + 9) // 10
ncols_grid = min(total, 10)
correctness_grid = np.zeros((nrows_grid, ncols_grid), dtype=int)
for idx, r in enumerate(results):
    correctness_grid[idx // 10][idx % 10] = 1 if r["correct"] else 0

letter_acc = {}
for letter in letters:
    subset = [r for r in results if r["ground_truth"].strip().upper() == letter]
    if subset:
        letter_acc[letter] = sum(1 for r in subset if r["correct"]) / len(subset) * 100
    else:
        letter_acc[letter] = 0


# ===========================================================================
# IMAGE 1: Summary dashboard
# ===========================================================================
fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle(
    f"Qwen3.5-4B-Base  \u00b7  MMStar Evaluation  \u00b7  {total} Samples",
    fontsize=20, fontweight="bold", color="white", y=0.98
)
gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.96, top=0.92, bottom=0.05)

# Panel 1: Donut
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Overall Accuracy")
ax1.pie([correct, total - correct], colors=[GREEN, RED], startangle=90,
        wedgeprops=dict(width=0.35, edgecolor=BG, linewidth=2))
ax1.text(0, 0, f"{accuracy:.0f}%", ha="center", va="center",
         fontsize=32, fontweight="bold", color=WHITE)
ax1.text(0, -0.15, f"{correct}/{total}", ha="center", va="center",
         fontsize=11, color=GRAY)
ax1.legend([f"Correct ({correct})", f"Wrong ({total - correct})"],
           loc="lower center", fontsize=9, frameon=False,
           labelcolor=WHITE, bbox_to_anchor=(0.5, -0.08))

# Panel 2: L2 category bars
ax2 = fig.add_subplot(gs[0, 1:3])
style_ax(ax2, "Accuracy by L2 Category")
bar_colors = [GREEN if a >= 50 else YELLOW if a >= 30 else RED for a in l2_accs]
ax2.barh(range(len(l2_names)), l2_accs, color=bar_colors, height=0.6, edgecolor="#30363d")
ax2.set_yticks(range(len(l2_names)))
ax2.set_yticklabels([f"{n} (n={l2_totals[i]})" for i, n in enumerate(l2_names)],
                     fontsize=10, color=WHITE)
ax2.set_xlim(0, 100)
ax2.set_xlabel("Accuracy %", color=GRAY, fontsize=10)
ax2.axvline(50, color=GRAY, linestyle="--", alpha=0.4)
ax2.axvline(25, color=RED, linestyle=":", alpha=0.3)
for i, acc in enumerate(l2_accs):
    ax2.text(acc + 1.5, i, f"{acc:.1f}%", va="center", fontsize=10,
             color=WHITE, fontweight="bold")

# Panel 3: Position bias
ax3 = fig.add_subplot(gs[0, 3])
style_ax(ax3, "Accuracy by GT Answer Position")
pos_colors = [GREEN if letter_acc[l] >= 50 else YELLOW if letter_acc[l] >= 30 else RED
              for l in letters]
ax3.bar(letters, [letter_acc[l] for l in letters], color=pos_colors,
        width=0.5, edgecolor="#30363d")
ax3.set_ylim(0, 100)
ax3.set_ylabel("Accuracy %", color=GRAY, fontsize=10)
ax3.axhline(50, color=GRAY, linestyle="--", alpha=0.4)
ax3.axhline(25, color=RED, linestyle=":", alpha=0.3)
for i, l in enumerate(letters):
    ax3.text(i, letter_acc[l] + 2, f"{letter_acc[l]:.0f}%\n(n={gt_dist.get(l,0)})",
             ha="center", fontsize=9, color=WHITE)
ax3.tick_params(axis="x", labelsize=12, colors=WHITE)

# Panel 4: Confusion matrix
ax4 = fig.add_subplot(gs[1, 0:2])
style_ax(ax4, "Confusion Matrix (Ground Truth vs Predicted)")
ax4.imshow(conf_matrix, cmap="RdYlGn", aspect="auto", vmin=0)
ax4.set_xticks(range(4)); ax4.set_yticks(range(4))
ax4.set_xticklabels(letters, fontsize=12, color=WHITE)
ax4.set_yticklabels(letters, fontsize=12, color=WHITE)
ax4.set_xlabel("Predicted", color=GRAY, fontsize=11)
ax4.set_ylabel("Ground Truth", color=GRAY, fontsize=11)
for i in range(4):
    for j in range(4):
        val = conf_matrix[i][j]
        if val > 0:
            tc = "black" if val > conf_matrix.max() * 0.6 else WHITE
            ax4.text(j, i, str(val), ha="center", va="center", fontsize=14,
                     color=tc, fontweight="bold" if i == j else "normal")
    ax4.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor=GREEN, linewidth=2))

# Panel 5: Distribution comparison
ax5 = fig.add_subplot(gs[1, 2:4])
style_ax(ax5, "Answer Distribution: Ground Truth vs Model Prediction")
x = np.arange(len(letters)); width = 0.3
gt_vals = [gt_dist.get(l, 0) for l in letters]
pred_vals = [pred_dist.get(l, 0) for l in letters]
ax5.bar(x - width/2, gt_vals, width, color=BLUE, label="Ground Truth", edgecolor="#30363d")
ax5.bar(x + width/2, pred_vals, width, color=PURPLE, label="Model Prediction", edgecolor="#30363d")
ax5.set_xticks(x); ax5.set_xticklabels(letters, fontsize=12, color=WHITE)
ax5.set_ylabel("Count", color=GRAY, fontsize=10)
ax5.legend(fontsize=10, frameon=False, labelcolor=WHITE, loc="upper right")
for i, (g, p) in enumerate(zip(gt_vals, pred_vals)):
    ax5.text(i - width/2, g + 0.5, str(g), ha="center", fontsize=9, color=BLUE)
    ax5.text(i + width/2, p + 0.5, str(p), ha="center", fontsize=9, color=PURPLE)

# Panel 6: Heatmap
ax6 = fig.add_subplot(gs[2, 0:2])
style_ax(ax6, "Per-Sample Results (green=correct, red=wrong)")
cmap = matplotlib.colors.ListedColormap([RED, GREEN])
ax6.imshow(correctness_grid, cmap=cmap, aspect="auto", interpolation="nearest")
ax6.set_xlabel("Sample index (mod 10)", color=GRAY, fontsize=10)
ax6.set_ylabel("Sample index (\u00f7 10)", color=GRAY, fontsize=10)
ax6.set_xticks(range(ncols_grid))
ax6.set_yticks(range(nrows_grid))
ax6.set_xticklabels(range(ncols_grid), fontsize=9, color=GRAY)
ax6.set_yticklabels([f"{i*10}-{i*10+9}" for i in range(nrows_grid)], fontsize=9, color=GRAY)
for i in range(nrows_grid):
    for j in range(ncols_grid):
        idx = i * 10 + j
        if idx < total:
            ax6.text(j, i, str(idx), ha="center", va="center", fontsize=7,
                     color="black" if correctness_grid[i][j] else WHITE, fontweight="bold")

# Panel 7: Failure text
ax7 = fig.add_subplot(gs[2, 2:4])
style_ax(ax7, "Failure Analysis Summary")
ax7.axis("off")
lines = [
    f"Total failures: {len(failures)}/{total}",
    f"Random baseline (4-choice): 25%",
    f"Model accuracy: {accuracy:.1f}%  (+{accuracy - 25:.1f}pp over random)",
    "", "Failures by L2 category:"]
for cat, count in failure_l2.most_common():
    ct = l2_cats[cat]["total"]
    lines.append(f"  \u00b7 {cat}: {count}/{ct} wrong ({count/ct*100:.0f}% error rate)")
lines += ["", "Position bias:"]
for l in letters:
    gn, pn = gt_dist.get(l, 0), pred_dist.get(l, 0)
    d = pn - gn
    arrow = "\u2191" if d > 0 else "\u2193" if d < 0 else "="
    lines.append(f"  \u00b7 Option {l}: GT={gn}, Pred={pn} ({arrow}{abs(d)})")
lines += ["", "Top prediction errors (GT\u2192Pred):"]
error_pairs = Counter()
for r in failures:
    gt, pred = r["ground_truth"].strip().upper(), r["predicted_letter"].strip().upper()
    if gt and pred:
        error_pairs[(gt, pred)] += 1
for (gt, pred), count in error_pairs.most_common(5):
    lines.append(f"  \u00b7 {gt}\u2192{pred}: {count} times")
ax7.text(0.05, 0.95, "\n".join(lines), transform=ax7.transAxes, fontsize=10,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#30363d", alpha=0.9))

plt.savefig("mmstar_analysis.png", dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
print(f"Saved mmstar_analysis.png ({total} samples, {accuracy:.1f}% accuracy)")
plt.close()


# ===========================================================================
# IMAGE 2: Full image grid -- all samples with thumbnails
# ===========================================================================
if has_images:
    COLS = 10
    ROWS = (total + COLS - 1) // COLS
    cell_w, cell_h = 2.0, 2.4
    fig2, axes2 = plt.subplots(ROWS, COLS, figsize=(COLS * cell_w, ROWS * cell_h),
                                facecolor=BG)
    fig2.suptitle(
        f"Qwen3.5-4B-Base \u00b7 MMStar All Samples \u00b7 {accuracy:.0f}% Accuracy "
        f"({correct}/{total})",
        fontsize=16, fontweight="bold", color=WHITE, y=1.0
    )

    for idx in range(ROWS * COLS):
        row, col = idx // COLS, idx % COLS
        ax = axes2[row][col] if ROWS > 1 else axes2[col]
        ax.set_facecolor(BG)

        if idx < total:
            r = results[idx]
            img = decode_image(r["image_b64"])
            border_color = GREEN if r["correct"] else RED
            gt = r["ground_truth"].strip().upper()
            pred = r["predicted_letter"].strip().upper()

            ax.imshow(np.array(img))
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])

            status = "\u2713" if r["correct"] else "\u2717"
            ax.set_title(f"#{idx}  GT:{gt} P:{pred} {status}",
                         fontsize=7, color=border_color, fontweight="bold", pad=2)
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.4, wspace=0.15, top=0.96, bottom=0.01,
                        left=0.01, right=0.99)
    plt.savefig("mmstar_image_grid.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved mmstar_image_grid.png ({total} samples, {ROWS}x{COLS} grid)")
    plt.close()


    # =======================================================================
    # IMAGE 3: Detailed failure view -- each failure with image + full info
    # =======================================================================
    n_fail = len(failures)
    FCOLS = 5
    FROWS = (n_fail + FCOLS - 1) // FCOLS
    cell_fw, cell_fh = 4.0, 4.5
    fig3, axes3 = plt.subplots(FROWS, FCOLS,
                                figsize=(FCOLS * cell_fw, FROWS * cell_fh),
                                facecolor=BG)
    fig3.suptitle(
        f"Qwen3.5-4B-Base \u00b7 MMStar FAILURES \u00b7 {n_fail} Wrong Answers",
        fontsize=18, fontweight="bold", color=RED, y=1.0
    )

    for idx in range(FROWS * FCOLS):
        row, col = idx // FCOLS, idx % FCOLS
        if FROWS > 1:
            ax = axes3[row][col]
        elif FCOLS > 1:
            ax = axes3[col]
        else:
            ax = axes3
        ax.set_facecolor(CARD)

        if idx < n_fail:
            r = failures[idx]
            img = decode_image(r["image_b64"])
            gt = r["ground_truth"].strip().upper()
            pred = r["predicted_letter"].strip().upper()
            q_short = r["question"][:90]
            if len(r["question"]) > 90:
                q_short += "..."
            out_short = r["model_output"][:70]
            if len(r["model_output"]) > 70:
                out_short += "..."

            # Draw the image in the top portion
            ax.imshow(np.array(img), extent=[0, 1, 0.42, 1.0],
                      aspect="auto", transform=ax.transAxes, clip_on=True)
            for spine in ax.spines.values():
                spine.set_edgecolor(RED)
                spine.set_linewidth(2)
            ax.set_xticks([]); ax.set_yticks([])

            # Title
            ax.set_title(f"#{r['index']}  |  {r['l2_category']}",
                         fontsize=7, color=YELLOW, fontweight="bold", pad=2)

            # Info text below image
            info = (
                f"GT: {gt}   Pred: {pred}\n"
                f"Q: {textwrap.fill(q_short, 40)}\n"
                f"Out: {textwrap.fill(out_short, 40)}"
            )
            ax.text(0.05, 0.38, info, transform=ax.transAxes, fontsize=5.5,
                    color=WHITE, fontfamily="monospace", verticalalignment="top",
                    bbox=dict(facecolor="#21262d", edgecolor="#30363d",
                              alpha=0.9, pad=3))
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.55, wspace=0.2, top=0.96, bottom=0.01,
                        left=0.02, right=0.98)
    plt.savefig("mmstar_failures.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved mmstar_failures.png ({n_fail} failures, {FROWS}x{FCOLS} grid)")
    plt.close()
