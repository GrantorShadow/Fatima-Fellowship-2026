"""
Visualize PAC Bench evaluation results from pacbench_results.json.

Produces:
  1. pacbench_analysis.png     -- summary dashboard (property/group breakdowns)
  2. pacbench_image_grid.png   -- grid of ALL images with per-image accuracy
  3. pacbench_failures.png     -- detailed failure view with images + property info

Usage:
    python3 visualize_pacbench.py
"""

import json
import base64
import io
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
import numpy as np
from PIL import Image as PILImage
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
with open("pacbench_results.json") as f:
    results = json.load(f)

with open("pacbench_stats.json") as f:
    stats = json.load(f)

total_images = len(results)
total_q = stats["total_questions"]
total_correct = stats["total_correct"]
accuracy = stats["accuracy_pct"]
has_images = "image_b64" in results[0]

# Color palette
BG = "#0d1117"
CARD = "#161b22"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
CYAN = "#39d2c0"
PINK = "#f778ba"
GRAY = "#8b949e"
WHITE = "#c9d1d9"

GROUP_COLORS = {
    "perceptual": BLUE,
    "physical": ORANGE,
    "functional": PURPLE,
}

PROP_GROUP = {
    "COLOR": "perceptual", "ORIENTATION": "perceptual",
    "THICKNESS": "perceptual", "COMPLEXITY": "perceptual",
    "WEIGHT": "physical", "DENSITY": "physical",
    "HARDNESS": "physical", "STICKINESS": "physical",
    "CAPACITY": "functional", "CONTENTS": "functional",
    "SEALING": "functional", "CONSUMABILITY": "functional",
}


def style_ax(ax, title=""):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=GRAY, labelsize=9)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=WHITE, pad=10)


def decode_image(b64_str):
    return PILImage.open(io.BytesIO(base64.b64decode(b64_str)))


# ---------------------------------------------------------------------------
# Compute derived data
# ---------------------------------------------------------------------------
prop_stats = stats["per_property"]
group_stats = stats["per_group"]

# Sort properties by accuracy
prop_names = sorted(prop_stats.keys(),
                    key=lambda x: prop_stats[x]["accuracy"])
prop_accs = [prop_stats[p]["accuracy"] for p in prop_names]
prop_totals = [prop_stats[p]["total"] for p in prop_names]

# Group names sorted by accuracy
grp_names = sorted(group_stats.keys(),
                   key=lambda x: group_stats[x]["accuracy"])
grp_accs = [group_stats[g]["accuracy"] for g in grp_names]

# Flatten all property results for failure analysis
all_prop_results = []
for r in results:
    for pr in r["property_results"]:
        all_prop_results.append({
            "index": r["index"],
            "suffix_key": r["suffix_key"],
            "image_b64": r["image_b64"],
            "image_file": r["image_file"],
            **pr,
        })

failures = [pr for pr in all_prop_results if not pr["correct"]]
failure_props = Counter(pr["property"] for pr in failures)

# Per-image accuracy distribution
img_accs = [r["image_accuracy"] for r in results]

# Correctness grid for heatmap
nrows_grid = (total_images + 9) // 10
ncols_grid = min(total_images, 10)
img_acc_grid = np.full((nrows_grid, ncols_grid), np.nan)
for idx, r in enumerate(results):
    img_acc_grid[idx // 10][idx % 10] = r["image_accuracy"]


# ===========================================================================
# IMAGE 1: Summary dashboard
# ===========================================================================
fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.suptitle(
    f"Qwen3.5-4B-Base  \u00b7  PAC Bench (Physical Attributes)  \u00b7  "
    f"{total_images} Images \u00d7 12 Properties",
    fontsize=20, fontweight="bold", color="white", y=0.98
)
gs = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.4,
                       left=0.06, right=0.96, top=0.92, bottom=0.05)

# ---------- Panel 1: Donut ----------
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Overall Accuracy")
ax1.pie([total_correct, total_q - total_correct], colors=[GREEN, RED],
        startangle=90, wedgeprops=dict(width=0.35, edgecolor=BG, linewidth=2))
ax1.text(0, 0, f"{accuracy:.0f}%", ha="center", va="center",
         fontsize=32, fontweight="bold", color=WHITE)
ax1.text(0, -0.15, f"{total_correct}/{total_q}", ha="center", va="center",
         fontsize=11, color=GRAY)
ax1.legend([f"Correct ({total_correct})", f"Wrong ({total_q - total_correct})"],
           loc="lower center", fontsize=9, frameon=False,
           labelcolor=WHITE, bbox_to_anchor=(0.5, -0.08))

# ---------- Panel 2: Group bars ----------
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Accuracy by Property Group")
gc = [GROUP_COLORS.get(g, GRAY) for g in grp_names]
ax2.barh(range(len(grp_names)), grp_accs, color=gc, height=0.5, edgecolor="#30363d")
ax2.set_yticks(range(len(grp_names)))
ax2.set_yticklabels(grp_names, fontsize=11, color=WHITE)
ax2.set_xlim(0, 105)
ax2.set_xlabel("Accuracy %", color=GRAY, fontsize=10)
ax2.axvline(50, color=GRAY, linestyle="--", alpha=0.4)
for i, acc in enumerate(grp_accs):
    ax2.text(acc + 1.5, i, f"{acc:.1f}%", va="center", fontsize=12,
             color=WHITE, fontweight="bold")

# ---------- Panel 3: Per-image accuracy histogram ----------
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Per-Image Accuracy Distribution")
bins = np.arange(0, 110, 10)
n, _, patches = ax3.hist(img_accs, bins=bins, color=BLUE, edgecolor="#30363d", alpha=0.8)
for patch, left_edge in zip(patches, bins[:-1]):
    if left_edge < 40:
        patch.set_facecolor(RED)
    elif left_edge < 60:
        patch.set_facecolor(YELLOW)
    else:
        patch.set_facecolor(GREEN)
ax3.set_xlabel("Accuracy % (across 12 properties)", color=GRAY, fontsize=10)
ax3.set_ylabel("# Images", color=GRAY, fontsize=10)
mean_acc = np.mean(img_accs)
ax3.axvline(mean_acc, color=CYAN, linestyle="--", linewidth=2, alpha=0.8)
ax3.text(mean_acc + 2, max(n) * 0.9, f"mean={mean_acc:.1f}%",
         fontsize=10, color=CYAN, fontweight="bold")

# ---------- Panel 4: Failure by property pie ----------
ax4 = fig.add_subplot(gs[0, 3])
style_ax(ax4, "Failures by Property")
if failures:
    fp_names = [p for p, _ in failure_props.most_common()]
    fp_counts = [c for _, c in failure_props.most_common()]
    fp_colors = [GROUP_COLORS.get(PROP_GROUP.get(p, ""), GRAY) for p in fp_names]
    wedges, texts = ax4.pie(fp_counts, colors=fp_colors, startangle=90,
                            wedgeprops=dict(width=0.6, edgecolor=BG, linewidth=1))
    ax4.legend(
        [f"{p} ({c})" for p, c in zip(fp_names, fp_counts)],
        loc="center left", fontsize=7, frameon=False,
        labelcolor=WHITE, bbox_to_anchor=(0.95, 0.5)
    )
else:
    ax4.text(0, 0, "No failures!", ha="center", va="center", color=GREEN, fontsize=14)

# ---------- Panel 5: Per-property bars (main chart) ----------
ax5 = fig.add_subplot(gs[1, :3])
style_ax(ax5, "Accuracy by Property (colored by group)")
prop_bar_colors = [GROUP_COLORS.get(PROP_GROUP.get(p, ""), GRAY) for p in prop_names]
ax5.barh(range(len(prop_names)), prop_accs, color=prop_bar_colors,
         height=0.6, edgecolor="#30363d")
ax5.set_yticks(range(len(prop_names)))
ax5.set_yticklabels([f"{p} (n={prop_totals[i]})" for i, p in enumerate(prop_names)],
                     fontsize=10, color=WHITE)
ax5.set_xlim(0, 110)
ax5.set_xlabel("Accuracy %", color=GRAY, fontsize=10)
ax5.axvline(50, color=GRAY, linestyle="--", alpha=0.4)
ax5.axvline(30, color=RED, linestyle=":", alpha=0.3)
for i, acc in enumerate(prop_accs):
    ax5.text(acc + 1.5, i, f"{acc:.1f}%", va="center", fontsize=10,
             color=WHITE, fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=g)
                   for g in ["perceptual", "physical", "functional"]]
ax5.legend(handles=legend_elements, loc="lower right", fontsize=9,
           frameon=True, facecolor=CARD, edgecolor="#30363d", labelcolor=WHITE)

# ---------- Panel 6: Per-image accuracy heatmap ----------
ax6 = fig.add_subplot(gs[1, 3])
style_ax(ax6, "Per-Image Accuracy")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rg", [RED, YELLOW, GREEN], N=256)
im = ax6.imshow(img_acc_grid, cmap=cmap, aspect="auto", vmin=0, vmax=100,
                interpolation="nearest")
ax6.set_xlabel("Image (mod 10)", color=GRAY, fontsize=9)
ax6.set_ylabel("Image (\u00f7 10)", color=GRAY, fontsize=9)
ax6.set_xticks(range(ncols_grid))
ax6.set_yticks(range(nrows_grid))
ax6.set_xticklabels(range(ncols_grid), fontsize=8, color=GRAY)
ax6.set_yticklabels([f"{i*10}" for i in range(nrows_grid)], fontsize=8, color=GRAY)
for i in range(nrows_grid):
    for j in range(ncols_grid):
        idx = i * 10 + j
        if idx < total_images:
            val = img_acc_grid[i][j]
            tc = "black" if val > 60 else WHITE
            ax6.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=6, color=tc, fontweight="bold")

# ---------- Panel 7: Failure analysis text ----------
ax7 = fig.add_subplot(gs[2, :2])
style_ax(ax7, "Failure Analysis Summary")
ax7.axis("off")
lines = [
    f"Total questions: {total_q}  ({total_images} images x properties)",
    f"Total failures:  {len(failures)}/{total_q}",
    f"Model accuracy:  {accuracy:.1f}%",
    "",
    "Failures by property (top 8):",
]
for prop, count in failure_props.most_common(8):
    pt = prop_stats[prop]["total"]
    lines.append(f"  \u00b7 {prop}: {count}/{pt} wrong "
                 f"({count/pt*100:.0f}% error rate)")

lines += ["", "Per-image accuracy stats:"]
lines.append(f"  \u00b7 Mean: {mean_acc:.1f}%")
lines.append(f"  \u00b7 Worst image: {min(img_accs):.0f}%")
lines.append(f"  \u00b7 Best image: {max(img_accs):.0f}%")
lines.append(f"  \u00b7 Images <50%: {sum(1 for a in img_accs if a < 50)}/{total_images}")

ax7.text(0.05, 0.95, "\n".join(lines), transform=ax7.transAxes, fontsize=10,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                   edgecolor="#30363d", alpha=0.9))

# ---------- Panel 8: Blind spots ----------
ax8 = fig.add_subplot(gs[2, 2:])
style_ax(ax8, "Identified Blind Spots")
ax8.axis("off")
bs_lines = ["BLIND SPOTS:", ""]

# Properties below 50%
for p in prop_names:
    a = prop_stats[p]["accuracy"]
    if a < 50:
        group = PROP_GROUP.get(p, "?")
        bs_lines.append(f"  \u26a0  {p} ({group}): {a:.1f}%  "
                        f"(n={prop_stats[p]['total']})")

bs_lines += ["", "GROUP COMPARISON:"]
for g in ["perceptual", "physical", "functional"]:
    if g in group_stats:
        bs_lines.append(f"  {g:15s}: {group_stats[g]['accuracy']:.1f}%")

# Check if physical properties are hardest (expected for VLMs)
phys_acc = group_stats.get("physical", {}).get("accuracy", 0)
perc_acc = group_stats.get("perceptual", {}).get("accuracy", 0)
func_acc = group_stats.get("functional", {}).get("accuracy", 0)

bs_lines += ["", "KEY FINDINGS:"]
if phys_acc < perc_acc and phys_acc < func_acc:
    bs_lines.append(f"  \u2192 Physical properties are hardest")
    bs_lines.append(f"    ({phys_acc:.1f}% vs {perc_acc:.1f}% perceptual)")
    bs_lines.append(f"    VLM struggles with weight/density/")
    bs_lines.append(f"    hardness requiring physical intuition")
elif func_acc < perc_acc and func_acc < phys_acc:
    bs_lines.append(f"  \u2192 Functional properties are hardest")
    bs_lines.append(f"    ({func_acc:.1f}% vs {perc_acc:.1f}% perceptual)")
else:
    bs_lines.append(f"  \u2192 Perceptual properties are hardest")
    bs_lines.append(f"    ({perc_acc:.1f}% vs {phys_acc:.1f}% physical)")

# Worst property callout
worst_p = prop_names[0]
worst_a = prop_accs[0]
bs_lines.append(f"  \u2192 Worst property: {worst_p} ({worst_a:.1f}%)")

ax8.text(0.05, 0.95, "\n".join(bs_lines), transform=ax8.transAxes, fontsize=10,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                   edgecolor="#30363d", alpha=0.9))

plt.savefig("pacbench_analysis.png", dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved pacbench_analysis.png ({total_images} images, {accuracy:.1f}% accuracy)")
plt.close()


# ===========================================================================
# IMAGE 2: Full image grid -- all images with per-image accuracy
# ===========================================================================
if has_images:
    COLS = 10
    ROWS = (total_images + COLS - 1) // COLS
    cell_w, cell_h = 2.0, 2.6
    fig2, axes2 = plt.subplots(ROWS, COLS, figsize=(COLS * cell_w, ROWS * cell_h),
                                facecolor=BG)
    fig2.suptitle(
        f"Qwen3.5-4B-Base \u00b7 PAC Bench All Images \u00b7 {accuracy:.0f}% Overall "
        f"({total_correct}/{total_q})",
        fontsize=16, fontweight="bold", color=WHITE, y=1.0
    )

    for idx in range(ROWS * COLS):
        row, col = idx // COLS, idx % COLS
        if ROWS > 1:
            ax = axes2[row][col]
        else:
            ax = axes2[col]
        ax.set_facecolor(BG)

        if idx < total_images:
            r = results[idx]
            img = decode_image(r["image_b64"])
            ia = r["image_accuracy"]
            border_color = GREEN if ia >= 70 else YELLOW if ia >= 50 else RED

            ax.imshow(np.array(img))
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])

            n_ok = sum(1 for pr in r["property_results"] if pr["correct"])
            n_tot = len(r["property_results"])
            ax.set_title(f"#{idx}  {ia:.0f}% ({n_ok}/{n_tot})",
                         fontsize=6.5, color=border_color, fontweight="bold", pad=2)
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.5, wspace=0.15, top=0.96, bottom=0.01,
                        left=0.01, right=0.99)
    plt.savefig("pacbench_image_grid.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved pacbench_image_grid.png ({total_images} images, {ROWS}x{COLS} grid)")
    plt.close()


    # =======================================================================
    # IMAGE 3: Detailed failure view -- worst images
    # =======================================================================
    # Sort images by accuracy (worst first), show worst 30
    worst_images = sorted(results, key=lambda r: r["image_accuracy"])
    n_show = min(30, total_images)
    worst_images = worst_images[:n_show]

    FCOLS = 5
    FROWS = (n_show + FCOLS - 1) // FCOLS
    cell_fw, cell_fh = 4.2, 5.5
    fig3, axes3 = plt.subplots(FROWS, FCOLS,
                                figsize=(FCOLS * cell_fw, FROWS * cell_fh),
                                facecolor=BG)
    fig3.suptitle(
        f"Qwen3.5-4B-Base \u00b7 PAC Bench WORST IMAGES \u00b7 "
        f"Bottom {n_show} by accuracy",
        fontsize=18, fontweight="bold", color=RED, y=1.0
    )

    for idx in range(FROWS * FCOLS):
        row, col = idx // FCOLS, idx % FCOLS
        if FROWS > 1 and FCOLS > 1:
            ax = axes3[row][col]
        elif FROWS > 1:
            ax = axes3[row]
        elif FCOLS > 1:
            ax = axes3[col]
        else:
            ax = axes3
        ax.set_facecolor(CARD)

        if idx < n_show:
            r = worst_images[idx]
            img = decode_image(r["image_b64"])
            ia = r["image_accuracy"]

            ax.imshow(np.array(img), extent=[0, 1, 0.35, 1.0],
                      aspect="auto", transform=ax.transAxes, clip_on=True)
            bc = RED if ia < 40 else YELLOW
            for spine in ax.spines.values():
                spine.set_edgecolor(bc)
                spine.set_linewidth(2)
            ax.set_xticks([]); ax.set_yticks([])

            ax.set_title(f"#{r['index']}  acc={ia:.0f}%",
                         fontsize=7, color=YELLOW, fontweight="bold", pad=2)

            # Show each property result
            info_lines = []
            for pr in r["property_results"]:
                mark = "\u2713" if pr["correct"] else "\u2717"
                gt_short = pr["ground_truth"].split(":")[0]
                pred_short = pr["predicted"].split(":")[0] if len(pr["predicted"]) > 2 else pr["predicted"]
                color_hint = "" if pr["correct"] else " <<<"
                info_lines.append(f"{mark} {pr['property']:13s} GT:{gt_short:15s} P:{pred_short}{color_hint}")

            info = "\n".join(info_lines)
            ax.text(0.03, 0.33, info, transform=ax.transAxes, fontsize=4.2,
                    color=WHITE, fontfamily="monospace", verticalalignment="top",
                    bbox=dict(facecolor="#21262d", edgecolor="#30363d",
                              alpha=0.9, pad=3))
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.96, bottom=0.01,
                        left=0.02, right=0.98)
    plt.savefig("pacbench_failures.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved pacbench_failures.png ({n_show} worst images, {FROWS}x{FCOLS} grid)")
    plt.close()

print("\nDone. Generated files:")
print("  pacbench_analysis.png    -- summary dashboard")
print("  pacbench_image_grid.png  -- thumbnail grid of all images")
print("  pacbench_failures.png    -- worst-performing image cards")
