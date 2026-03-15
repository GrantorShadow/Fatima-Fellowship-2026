"""
Visualize order-sensitivity results from order_sensitivity_results.json.

Produces:
  1. order_sensitivity_analysis.png  -- summary dashboard
  2. order_sensitivity_grid.png      -- per-sample image grid showing flips
  3. order_sensitivity_flips.png     -- detailed view of flipped samples

Usage:
    python3 visualize_order_sensitivity.py
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
with open("order_sensitivity_results.json") as f:
    results = json.load(f)

with open("order_sensitivity_stats.json") as f:
    stats = json.load(f)

total = len(results)
has_images = "image_b64" in results[0]

img_acc = stats["img_first_accuracy"]
txt_acc = stats["txt_first_accuracy"]
delta = stats["accuracy_delta"]
flip_rate = stats["flip_rate_pct"]
change_rate = stats["pred_change_rate_pct"]

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
GRAY = "#8b949e"
WHITE = "#c9d1d9"


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
# Derived data
# ---------------------------------------------------------------------------
# Classify each sample
both_ok = [r for r in results if r["img_first_correct"] and r["txt_first_correct"]]
both_fail = [r for r in results if not r["img_first_correct"] and not r["txt_first_correct"]]
img_only = [r for r in results if r["img_first_correct"] and not r["txt_first_correct"]]
txt_only = [r for r in results if not r["img_first_correct"] and r["txt_first_correct"]]
changed = [r for r in results if r["prediction_changed"]]

# Per-category data
cat_data = defaultdict(lambda: {
    "total": 0, "img_acc": 0, "txt_acc": 0, "changed": 0, "flipped": 0,
})
for r in results:
    c = r["category"]
    cat_data[c]["total"] += 1
    if r["img_first_correct"]:
        cat_data[c]["img_acc"] += 1
    if r["txt_first_correct"]:
        cat_data[c]["txt_acc"] += 1
    if r["prediction_changed"]:
        cat_data[c]["changed"] += 1
    if r["flipped_correctness"]:
        cat_data[c]["flipped"] += 1

cat_names = sorted(cat_data.keys())

# Per-L2 category
l2_data = defaultdict(lambda: {
    "total": 0, "img_acc": 0, "txt_acc": 0, "changed": 0, "flipped": 0,
})
for r in results:
    l2 = r["l2_category"]
    l2_data[l2]["total"] += 1
    if r["img_first_correct"]:
        l2_data[l2]["img_acc"] += 1
    if r["txt_first_correct"]:
        l2_data[l2]["txt_acc"] += 1
    if r["prediction_changed"]:
        l2_data[l2]["changed"] += 1
    if r["flipped_correctness"]:
        l2_data[l2]["flipped"] += 1

l2_names = sorted(l2_data.keys(),
                  key=lambda x: l2_data[x]["changed"] / l2_data[x]["total"],
                  reverse=True)

# Stability grid: 0=both wrong, 1=img only, 2=txt only, 3=both correct
stability_grid_rows = (total + 9) // 10
stability_grid_cols = min(total, 10)
stability_grid = np.full((stability_grid_rows, stability_grid_cols), -1, dtype=int)
for idx, r in enumerate(results):
    row, col = idx // 10, idx % 10
    if r["img_first_correct"] and r["txt_first_correct"]:
        stability_grid[row][col] = 3  # stable correct
    elif r["img_first_correct"]:
        stability_grid[row][col] = 1  # img-only
    elif r["txt_first_correct"]:
        stability_grid[row][col] = 2  # txt-only
    else:
        stability_grid[row][col] = 0  # stable wrong


# ===========================================================================
# IMAGE 1: Summary dashboard
# ===========================================================================
fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle(
    f"Qwen3.5-4B-Base  \u00b7  Order Sensitivity Test (MMStar)  \u00b7  "
    f"{total} Samples \u00d7 2 Orderings",
    fontsize=20, fontweight="bold", color="white", y=0.98
)
gs = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.4,
                       left=0.06, right=0.96, top=0.92, bottom=0.04)

# ---------- Panel 1: Dual accuracy comparison ----------
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Accuracy: Image-First vs Text-First")
bars = ax1.bar(
    [0, 1], [img_acc, txt_acc],
    color=[BLUE, PURPLE], width=0.5, edgecolor="#30363d"
)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Image\nFirst", "Text\nFirst"], fontsize=11, color=WHITE)
ax1.set_ylim(0, max(img_acc, txt_acc) + 15)
ax1.set_ylabel("Accuracy %", color=GRAY, fontsize=10)
for i, (val, c) in enumerate([(img_acc, BLUE), (txt_acc, PURPLE)]):
    ax1.text(i, val + 2, f"{val:.1f}%", ha="center", fontsize=14,
             color=WHITE, fontweight="bold")
# Delta annotation
ax1.annotate(
    f"\u0394 = {delta:+.1f}pp",
    xy=(0.5, max(img_acc, txt_acc) + 5), fontsize=12,
    ha="center", color=YELLOW, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor=YELLOW)
)

# ---------- Panel 2: Outcome pie (4-way) ----------
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Sample Outcome Distribution")
outcome_sizes = [len(both_ok), len(img_only), len(txt_only), len(both_fail)]
outcome_labels = [
    f"Both correct\n({len(both_ok)})",
    f"Only img-first\n({len(img_only)})",
    f"Only txt-first\n({len(txt_only)})",
    f"Both wrong\n({len(both_fail)})",
]
outcome_colors = [GREEN, BLUE, PURPLE, RED]
wedges, texts = ax2.pie(
    outcome_sizes, colors=outcome_colors, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2)
)
ax2.legend(outcome_labels, loc="center left", fontsize=8, frameon=False,
           labelcolor=WHITE, bbox_to_anchor=(0.92, 0.5))
# Center text
ax2.text(0, 0, f"{change_rate:.0f}%\nchanged",
         ha="center", va="center", fontsize=14, fontweight="bold", color=YELLOW)

# ---------- Panel 3: Instability rate bar ----------
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Instability Metrics")
metrics = ["Prediction\nChanged", "Correctness\nFlipped"]
values = [change_rate, flip_rate]
mc = [ORANGE if v > 20 else YELLOW if v > 10 else GREEN for v in values]
ax3.bar(range(len(metrics)), values, color=mc, width=0.5, edgecolor="#30363d")
ax3.set_xticks(range(len(metrics)))
ax3.set_xticklabels(metrics, fontsize=10, color=WHITE)
ax3.set_ylim(0, max(values) + 15)
ax3.set_ylabel("% of samples", color=GRAY, fontsize=10)
for i, v in enumerate(values):
    ax3.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=13,
             color=WHITE, fontweight="bold")
# Threshold lines
ax3.axhline(20, color=ORANGE, linestyle="--", alpha=0.4, label="High instability")
ax3.axhline(10, color=YELLOW, linestyle=":", alpha=0.4, label="Moderate")
ax3.legend(fontsize=8, frameon=False, labelcolor=WHITE, loc="upper right")

# ---------- Panel 4: Summary text ----------
ax4 = fig.add_subplot(gs[0, 3])
style_ax(ax4, "Key Findings")
ax4.axis("off")
severity = "HIGH" if change_rate > 20 else "MODERATE" if change_rate > 10 else "LOW"
sev_color = RED if severity == "HIGH" else YELLOW if severity == "MODERATE" else GREEN
lines = [
    f"INSTABILITY SEVERITY: {severity}",
    f"",
    f"Image-first accuracy:  {img_acc:.1f}%",
    f"Text-first accuracy:   {txt_acc:.1f}%",
    f"Delta:                 {delta:+.1f}pp",
    f"",
    f"Predictions changed:   {stats['prediction_changed']}/{total}",
    f"  = {change_rate:.1f}% of samples",
    f"",
    f"Correctness flipped:   {stats['flipped_correctness']}/{total}",
    f"  = {flip_rate:.1f}% of samples",
    f"",
    f"Stable correct:        {len(both_ok)}/{total}",
    f"Stable wrong:          {len(both_fail)}/{total}",
    f"Img-first only:        {len(img_only)}/{total}",
    f"Txt-first only:        {len(txt_only)}/{total}",
]
if delta > 0:
    lines += ["", f"Image-first is better by {delta:.1f}pp"]
elif delta < 0:
    lines += ["", f"Text-first is better by {-delta:.1f}pp"]
else:
    lines += ["", "No directional preference"]

ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes, fontsize=9.5,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                   edgecolor=sev_color, alpha=0.9, linewidth=2))

# ---------- Panel 5: Per-category comparison ----------
ax5 = fig.add_subplot(gs[1, :2])
style_ax(ax5, "Per-Category: Image-First vs Text-First Accuracy")
x = np.arange(len(cat_names))
width = 0.35
img_vals = [cat_data[c]["img_acc"] / cat_data[c]["total"] * 100 for c in cat_names]
txt_vals = [cat_data[c]["txt_acc"] / cat_data[c]["total"] * 100 for c in cat_names]
ax5.bar(x - width/2, img_vals, width, color=BLUE, label="Image-first",
        edgecolor="#30363d")
ax5.bar(x + width/2, txt_vals, width, color=PURPLE, label="Text-first",
        edgecolor="#30363d")
ax5.set_xticks(x)
ax5.set_xticklabels(
    [f"{c}\n(n={cat_data[c]['total']})" for c in cat_names],
    fontsize=8, color=WHITE, rotation=0
)
ax5.set_ylabel("Accuracy %", color=GRAY, fontsize=10)
ax5.set_ylim(0, 105)
ax5.legend(fontsize=9, frameon=False, labelcolor=WHITE, loc="upper right")
# Delta annotations
for i, c in enumerate(cat_names):
    d = img_vals[i] - txt_vals[i]
    color = YELLOW if abs(d) > 5 else GRAY
    ax5.text(i, max(img_vals[i], txt_vals[i]) + 3, f"{d:+.0f}",
             ha="center", fontsize=8, color=color, fontweight="bold")

# ---------- Panel 6: Per-L2 instability rates ----------
ax6 = fig.add_subplot(gs[1, 2:])
style_ax(ax6, "Per-L2 Category: Prediction Change Rate")
l2_show = l2_names[:12]  # top 12
l2_change_rates = [l2_data[l2]["changed"] / l2_data[l2]["total"] * 100
                   for l2 in l2_show]
l2_totals = [l2_data[l2]["total"] for l2 in l2_show]
l2_colors = [RED if r > 25 else ORANGE if r > 15 else YELLOW if r > 10 else GREEN
             for r in l2_change_rates]
ax6.barh(range(len(l2_show)), l2_change_rates, color=l2_colors,
         height=0.6, edgecolor="#30363d")
ax6.set_yticks(range(len(l2_show)))
ax6.set_yticklabels(
    [f"{l2[:30]} (n={l2_totals[i]})" for i, l2 in enumerate(l2_show)],
    fontsize=8, color=WHITE
)
ax6.set_xlim(0, max(l2_change_rates) + 15 if l2_change_rates else 50)
ax6.set_xlabel("% predictions changed", color=GRAY, fontsize=10)
ax6.axvline(20, color=RED, linestyle="--", alpha=0.3)
ax6.axvline(10, color=YELLOW, linestyle=":", alpha=0.3)
for i, r in enumerate(l2_change_rates):
    ax6.text(r + 1, i, f"{r:.1f}%", va="center", fontsize=9,
             color=WHITE, fontweight="bold")

# ---------- Panel 7: Stability heatmap ----------
ax7 = fig.add_subplot(gs[2, :2])
style_ax(ax7, "Per-Sample Stability Map (green=stable, blue/purple=fragile, red=always wrong)")
cmap = matplotlib.colors.ListedColormap([RED, BLUE, PURPLE, GREEN])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
ax7.imshow(stability_grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
ax7.set_xlabel("Sample (mod 10)", color=GRAY, fontsize=9)
ax7.set_ylabel("Sample (\u00f7 10)", color=GRAY, fontsize=9)
ax7.set_xticks(range(stability_grid_cols))
ax7.set_yticks(range(stability_grid_rows))
ax7.set_xticklabels(range(stability_grid_cols), fontsize=8, color=GRAY)
ax7.set_yticklabels([f"{i*10}" for i in range(stability_grid_rows)], fontsize=8, color=GRAY)
from matplotlib.patches import Patch
legend_el = [
    Patch(facecolor=GREEN, label=f"Both correct ({len(both_ok)})"),
    Patch(facecolor=BLUE, label=f"Image-first only ({len(img_only)})"),
    Patch(facecolor=PURPLE, label=f"Text-first only ({len(txt_only)})"),
    Patch(facecolor=RED, label=f"Both wrong ({len(both_fail)})"),
]
ax7.legend(handles=legend_el, loc="upper right", fontsize=8,
           frameon=True, facecolor=CARD, edgecolor="#30363d", labelcolor=WHITE)

# ---------- Panel 8: Directional bias text ----------
ax8 = fig.add_subplot(gs[2, 2:])
style_ax(ax8, "Directional Analysis")
ax8.axis("off")
d_lines = [
    "WHICH ORDER WINS PER CATEGORY:",
    "",
]
for c in cat_names:
    t = cat_data[c]["total"]
    ia = cat_data[c]["img_acc"] / t * 100
    ta = cat_data[c]["txt_acc"] / t * 100
    d = ia - ta
    winner = "IMG" if d > 0 else "TXT" if d < 0 else "TIE"
    arrow = "\u2191" if d > 0 else "\u2193" if d < 0 else "="
    d_lines.append(f"  {c:25s} {arrow} {winner}  {d:+5.1f}pp  "
                   f"(chg={cat_data[c]['changed']/t*100:.0f}%)")

d_lines += [
    "",
    "MOST FRAGILE L2 CATEGORIES:",
]
for l2 in l2_names[:5]:
    t = l2_data[l2]["total"]
    chg = l2_data[l2]["changed"] / t * 100
    d_lines.append(f"  {l2[:35]:35s} {chg:.1f}% changed")

d_lines += [
    "",
    "INTERPRETATION:",
    f"  {change_rate:.0f}% of model predictions are",
    f"  unstable to token ordering alone.",
]
if change_rate > 20:
    d_lines.append("  This is a MAJOR reliability concern.")
    d_lines.append("  Results on any benchmark are order-")
    d_lines.append("  dependent and not fully trustworthy.")
elif change_rate > 10:
    d_lines.append("  Moderate instability; benchmark")
    d_lines.append("  scores may vary ~5pp by ordering.")

ax8.text(0.03, 0.97, "\n".join(d_lines), transform=ax8.transAxes, fontsize=8.5,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                   edgecolor="#30363d", alpha=0.9))

plt.savefig("order_sensitivity_analysis.png", dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved order_sensitivity_analysis.png")
plt.close()


# ===========================================================================
# IMAGE 2: Full image grid with stability coloring
# ===========================================================================
if has_images:
    COLS = 10
    ROWS = (total + COLS - 1) // COLS
    cell_w, cell_h = 2.0, 2.6
    fig2, axes2 = plt.subplots(ROWS, COLS, figsize=(COLS * cell_w, ROWS * cell_h),
                                facecolor=BG)
    fig2.suptitle(
        f"Order Sensitivity \u00b7 All Samples \u00b7 "
        f"Green=stable  Blue=img-only  Purple=txt-only  Red=both wrong",
        fontsize=14, fontweight="bold", color=WHITE, y=1.0
    )

    OUTCOME_COLORS = {
        (True, True): GREEN,    # both correct
        (True, False): BLUE,    # img-first only
        (False, True): PURPLE,  # txt-first only
        (False, False): RED,    # both wrong
    }

    for idx in range(ROWS * COLS):
        row, col = idx // COLS, idx % COLS
        if ROWS > 1:
            ax = axes2[row][col]
        else:
            ax = axes2[col]
        ax.set_facecolor(BG)

        if idx < total:
            r = results[idx]
            img = decode_image(r["image_b64"])
            border_color = OUTCOME_COLORS[
                (r["img_first_correct"], r["txt_first_correct"])
            ]
            gt = r["ground_truth"]
            pa = r["img_first_pred"]
            pb = r["txt_first_pred"]

            ax.imshow(np.array(img))
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])

            chg = "*" if r["prediction_changed"] else ""
            ax.set_title(
                f"#{idx} GT:{gt} I:{pa} T:{pb}{chg}",
                fontsize=5.5, color=border_color, fontweight="bold", pad=2
            )
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.5, wspace=0.15, top=0.96, bottom=0.01,
                        left=0.01, right=0.99)
    plt.savefig("order_sensitivity_grid.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved order_sensitivity_grid.png ({total} samples)")
    plt.close()


    # =======================================================================
    # IMAGE 3: Detailed flip view -- samples where prediction changed
    # =======================================================================
    flips = [r for r in results if r["prediction_changed"]]
    n_show = min(len(flips), 40)
    flips = flips[:n_show]

    if n_show > 0:
        FCOLS = 5
        FROWS = (n_show + FCOLS - 1) // FCOLS
        cell_fw, cell_fh = 4.2, 5.0
        fig3, axes3 = plt.subplots(FROWS, FCOLS,
                                    figsize=(FCOLS * cell_fw, FROWS * cell_fh),
                                    facecolor=BG)
        fig3.suptitle(
            f"Order Sensitivity \u00b7 FLIPPED PREDICTIONS \u00b7 "
            f"{len([r for r in results if r['prediction_changed']])} total, "
            f"showing {n_show}",
            fontsize=18, fontweight="bold", color=ORANGE, y=1.0
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
                r = flips[idx]
                img = decode_image(r["image_b64"])

                ax.imshow(np.array(img), extent=[0, 1, 0.38, 1.0],
                          aspect="auto", transform=ax.transAxes, clip_on=True)
                bc = ORANGE if r["flipped_correctness"] else YELLOW
                for spine in ax.spines.values():
                    spine.set_edgecolor(bc)
                    spine.set_linewidth(2)
                ax.set_xticks([]); ax.set_yticks([])

                flip_type = ""
                if r["img_first_correct"] and not r["txt_first_correct"]:
                    flip_type = "IMG wins"
                elif not r["img_first_correct"] and r["txt_first_correct"]:
                    flip_type = "TXT wins"
                else:
                    flip_type = "both wrong (diff pred)"

                ax.set_title(
                    f"#{r['index']}  {r['l2_category'][:25]}",
                    fontsize=6, color=YELLOW, fontweight="bold", pad=2
                )

                q_short = r["question"][:80]
                if len(r["question"]) > 80:
                    q_short += "..."
                out_a = r["img_first_output"][:50]
                out_b = r["txt_first_output"][:50]

                mark_a = "\u2713" if r["img_first_correct"] else "\u2717"
                mark_b = "\u2713" if r["txt_first_correct"] else "\u2717"

                info = (
                    f"GT: {r['ground_truth']}   | {flip_type}\n"
                    f"{mark_a} Img-first: {r['img_first_pred']}  \"{out_a}\"\n"
                    f"{mark_b} Txt-first: {r['txt_first_pred']}  \"{out_b}\"\n"
                    f"Q: {textwrap.fill(q_short, 44)}"
                )

                ax.text(0.05, 0.35, info, transform=ax.transAxes, fontsize=5,
                        color=WHITE, fontfamily="monospace", verticalalignment="top",
                        bbox=dict(facecolor="#21262d", edgecolor="#30363d",
                                  alpha=0.9, pad=3))
            else:
                ax.axis("off")

        plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.96, bottom=0.01,
                            left=0.02, right=0.98)
        plt.savefig("order_sensitivity_flips.png", dpi=150, facecolor=BG,
                    bbox_inches="tight", pad_inches=0.2)
        print(f"Saved order_sensitivity_flips.png ({n_show} flipped samples)")
        plt.close()
    else:
        print("No flipped predictions to visualize.")

print("\nDone. Generated files:")
print("  order_sensitivity_analysis.png  -- summary dashboard")
print("  order_sensitivity_grid.png      -- per-sample stability grid")
print("  order_sensitivity_flips.png     -- detailed flip cards")
