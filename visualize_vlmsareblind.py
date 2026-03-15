"""
Visualize Qwen3.5-4B-Base results on the VLMs-Are-Blind benchmark.

Reads vlmsareblind_results.json and vlmsareblind_stats.json, then produces:
  1. vlmsareblind_analysis.png   -- summary dashboard (donut, per-task bars,
                                    error distribution, heatmap, failure text)
  2. vlmsareblind_image_grid.png -- thumbnail grid of ALL samples (green/red borders)
  3. vlmsareblind_failures.png   -- detailed failure cards with images + info

Usage:
    python3 visualize_vlmsareblind.py
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
with open("vlmsareblind_results.json") as f:
    results = json.load(f)

with open("vlmsareblind_stats.json") as f:
    stats = json.load(f)

total = len(results)
correct = sum(1 for r in results if r["correct"])
accuracy = correct / total * 100
has_images = total > 0 and "image_b64" in results[0]

# Color palette (same dark theme as mmstar)
BG = "#0d1117"
CARD = "#161b22"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
GRAY = "#8b949e"
WHITE = "#c9d1d9"

TASK_COLORS = [BLUE, GREEN, PURPLE, ORANGE, YELLOW, RED, "#79c0ff", "#7ee787", "#d2a8ff"]


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
task_stats = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    t = r["task"]
    task_stats[t]["total"] += 1
    if r["correct"]:
        task_stats[t]["correct"] += 1

# Sort tasks by accuracy (ascending) for the bar chart
task_names = sorted(task_stats.keys(),
                    key=lambda x: task_stats[x]["correct"] / max(task_stats[x]["total"], 1))
task_accs = [task_stats[n]["correct"] / task_stats[n]["total"] * 100 for n in task_names]
task_totals = [task_stats[n]["total"] for n in task_names]

failures = [r for r in results if not r["correct"]]
failure_tasks = Counter(r["task"] for r in failures)

# Ground truth distribution
gt_dist = Counter(r["ground_truth"] for r in results)
pred_dist = Counter(r["predicted_answer"] for r in results)

# Error magnitude analysis (for numeric tasks)
error_deltas = []
for r in results:
    if not r["correct"]:
        try:
            gt_val = int(r["ground_truth"])
            pred_val = int(r["predicted_answer"])
            error_deltas.append(pred_val - gt_val)
        except (ValueError, TypeError):
            pass

# Per-task correctness grid
nrows_grid = (total + 19) // 20
ncols_grid = min(total, 20)
correctness_grid = np.zeros((nrows_grid, ncols_grid), dtype=int)
for idx, r in enumerate(results):
    row_i, col_i = idx // 20, idx % 20
    if row_i < nrows_grid and col_i < ncols_grid:
        correctness_grid[row_i][col_i] = 1 if r["correct"] else 0


# ===========================================================================
# IMAGE 1: Summary dashboard
# ===========================================================================
fig = plt.figure(figsize=(22, 16), facecolor=BG)
fig.suptitle(
    f"Qwen3.5-4B-Base  \u00b7  VLMs-Are-Blind Evaluation  \u00b7  {total} Samples",
    fontsize=20, fontweight="bold", color="white", y=0.98
)
gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.96, top=0.92, bottom=0.05)

# Panel 1: Donut
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Overall Accuracy")
ax1.pie([correct, total - correct], colors=[GREEN, RED], startangle=90,
        wedgeprops=dict(width=0.35, edgecolor=BG, linewidth=2))
ax1.text(0, 0, f"{accuracy:.1f}%", ha="center", va="center",
         fontsize=30, fontweight="bold", color=WHITE)
ax1.text(0, -0.15, f"{correct}/{total}", ha="center", va="center",
         fontsize=11, color=GRAY)
ax1.legend([f"Correct ({correct})", f"Wrong ({total - correct})"],
           loc="lower center", fontsize=9, frameon=False,
           labelcolor=WHITE, bbox_to_anchor=(0.5, -0.08))

# Panel 2: Per-task accuracy bars
ax2 = fig.add_subplot(gs[0, 1:4])
style_ax(ax2, "Accuracy by Task Type")
bar_colors = [GREEN if a >= 50 else YELLOW if a >= 30 else RED for a in task_accs]
bars = ax2.barh(range(len(task_names)), task_accs, color=bar_colors,
                height=0.6, edgecolor="#30363d")
ax2.set_yticks(range(len(task_names)))
ax2.set_yticklabels([f"{n}  (n={task_totals[i]})" for i, n in enumerate(task_names)],
                     fontsize=10, color=WHITE)
ax2.set_xlim(0, 100)
ax2.set_xlabel("Accuracy %", color=GRAY, fontsize=10)
ax2.axvline(50, color=GRAY, linestyle="--", alpha=0.4)
ax2.axvline(25, color=RED, linestyle=":", alpha=0.3, label="Random baseline")
for i, acc in enumerate(task_accs):
    c = task_stats[task_names[i]]["correct"]
    t = task_stats[task_names[i]]["total"]
    ax2.text(acc + 1.5, i, f"{acc:.1f}% ({c}/{t})", va="center", fontsize=10,
             color=WHITE, fontweight="bold")

# Panel 3: Error direction histogram (overshoot vs undershoot)
ax3 = fig.add_subplot(gs[1, 0:2])
style_ax(ax3, "Prediction Error Direction (Pred - GT)")
if error_deltas:
    bins = range(min(error_deltas) - 1, max(error_deltas) + 2)
    counts, edges, patches = ax3.hist(error_deltas, bins=bins, color=BLUE,
                                       edgecolor="#30363d", alpha=0.85)
    for patch in patches:
        center = patch.get_x() + patch.get_width() / 2
        if center > 0:
            patch.set_facecolor(ORANGE)  # overcount
        elif center < 0:
            patch.set_facecolor(PURPLE)  # undercount
        else:
            patch.set_facecolor(GREEN)   # off by zero (shouldn't happen for errors)
    ax3.axvline(0, color=GRAY, linestyle="--", alpha=0.5)
    ax3.set_xlabel("Prediction - Ground Truth", color=GRAY, fontsize=10)
    ax3.set_ylabel("Count", color=GRAY, fontsize=10)

    over = sum(1 for d in error_deltas if d > 0)
    under = sum(1 for d in error_deltas if d < 0)
    ax3.text(0.95, 0.95,
             f"Overcounts: {over}\nUndercounts: {under}\nMean error: {np.mean(error_deltas):+.2f}",
             transform=ax3.transAxes, fontsize=9, color=WHITE, va="top", ha="right",
             fontfamily="monospace",
             bbox=dict(facecolor="#21262d", edgecolor="#30363d", alpha=0.9, pad=4))
else:
    ax3.text(0.5, 0.5, "No numeric errors to plot", transform=ax3.transAxes,
             ha="center", va="center", color=GRAY, fontsize=12)

# Panel 4: Task composition pie
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, "Task Distribution")
task_sorted = sorted(task_stats.keys(), key=lambda x: task_stats[x]["total"], reverse=True)
sizes = [task_stats[t]["total"] for t in task_sorted]
colors_pie = TASK_COLORS[:len(task_sorted)]
wedges, texts = ax4.pie(sizes, colors=colors_pie, startangle=90,
                         wedgeprops=dict(edgecolor=BG, linewidth=1.5))
# Legend (placed to the right)
legend_labels = [f"{t} ({task_stats[t]['total']})" for t in task_sorted]
ax4.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
           fontsize=7, frameon=False, labelcolor=WHITE)

# Panel 5: Ground truth value distribution
ax5 = fig.add_subplot(gs[1, 3])
style_ax(ax5, "Ground Truth Distribution")
gt_common = gt_dist.most_common(12)
gt_labels = [str(g[0]) for g in gt_common]
gt_counts = [g[1] for g in gt_common]
ax5.bar(range(len(gt_labels)), gt_counts, color=BLUE, width=0.6, edgecolor="#30363d")
ax5.set_xticks(range(len(gt_labels)))
ax5.set_xticklabels(gt_labels, fontsize=8, color=WHITE, rotation=45, ha="right")
ax5.set_ylabel("Count", color=GRAY, fontsize=10)
for i, c in enumerate(gt_counts):
    ax5.text(i, c + 0.5, str(c), ha="center", fontsize=7, color=GRAY)

# Panel 6: Heatmap
ax6 = fig.add_subplot(gs[2, 0:2])
style_ax(ax6, "Per-Sample Results (green=correct, red=wrong)")
cmap = matplotlib.colors.ListedColormap([RED, GREEN])
ax6.imshow(correctness_grid, cmap=cmap, aspect="auto", interpolation="nearest")
ax6.set_xlabel("Sample index (mod 20)", color=GRAY, fontsize=10)
ax6.set_ylabel("Sample index (\u00f7 20)", color=GRAY, fontsize=10)
# Only show a subset of tick labels if there are many rows
n_yticks = min(nrows_grid, 30)
ytick_step = max(1, nrows_grid // n_yticks)
ytick_pos = list(range(0, nrows_grid, ytick_step))
ax6.set_yticks(ytick_pos)
ax6.set_yticklabels([f"{i*20}" for i in ytick_pos], fontsize=7, color=GRAY)
ax6.set_xticks(range(ncols_grid))
ax6.set_xticklabels(range(ncols_grid), fontsize=7, color=GRAY)

# Panel 7: Failure analysis text
ax7 = fig.add_subplot(gs[2, 2:4])
style_ax(ax7, "Failure Analysis Summary")
ax7.axis("off")
lines = [
    f"Total failures: {len(failures)}/{total} ({len(failures)/total*100:.1f}%)",
    f"Model accuracy: {accuracy:.1f}%",
    "",
    "Failures by task (error rate):",
]
for task_name, count in failure_tasks.most_common():
    ct = task_stats[task_name]["total"]
    acc_t = task_stats[task_name]["correct"] / ct * 100
    lines.append(f"  \u00b7 {task_name}: {count}/{ct} wrong ({100-acc_t:.0f}% err)")

if error_deltas:
    lines += [
        "",
        "Numeric error stats:",
        f"  Mean: {np.mean(error_deltas):+.2f}",
        f"  Median: {np.median(error_deltas):+.1f}",
        f"  Std: {np.std(error_deltas):.2f}",
        f"  Overcounts (pred>gt): {sum(1 for d in error_deltas if d > 0)}",
        f"  Undercounts (pred<gt): {sum(1 for d in error_deltas if d < 0)}",
    ]

# Blind spots
blindspots = [(t, s) for t, s in task_stats.items()
              if s["correct"] / max(s["total"], 1) * 100 < 30.0]
if blindspots:
    lines += ["", "BLIND SPOTS (< 30% accuracy):"]
    for t, s in blindspots:
        acc_b = s["correct"] / s["total"] * 100
        lines.append(f"  ** {t}: {acc_b:.1f}%")

ax7.text(0.03, 0.97, "\n".join(lines), transform=ax7.transAxes, fontsize=9,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                   edgecolor="#30363d", alpha=0.9))

plt.savefig("vlmsareblind_analysis.png", dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved vlmsareblind_analysis.png ({total} samples, {accuracy:.1f}% accuracy)")
plt.close()


# ===========================================================================
# IMAGE 2: Full image grid -- all samples with thumbnails
# ===========================================================================
if has_images:
    COLS = 20
    ROWS = (total + COLS - 1) // COLS
    cell_w, cell_h = 1.2, 1.5
    fig2, axes2 = plt.subplots(ROWS, COLS,
                                figsize=(COLS * cell_w, ROWS * cell_h),
                                facecolor=BG)
    fig2.suptitle(
        f"Qwen3.5-4B-Base \u00b7 VLMs-Are-Blind All Samples \u00b7 {accuracy:.1f}% "
        f"({correct}/{total})",
        fontsize=16, fontweight="bold", color=WHITE, y=1.0
    )

    for idx in range(ROWS * COLS):
        row, col = idx // COLS, idx % COLS
        if ROWS > 1:
            ax = axes2[row][col]
        else:
            ax = axes2[col]
        ax.set_facecolor(BG)

        if idx < total:
            r = results[idx]
            try:
                img = decode_image(r["image_b64"])
            except Exception:
                img = PILImage.new("RGB", (64, 64), color=(30, 30, 30))
            border_color = GREEN if r["correct"] else RED

            ax.imshow(np.array(img))
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])

            status = "\u2713" if r["correct"] else "\u2717"
            ax.set_title(f"{status}", fontsize=6, color=border_color,
                         fontweight="bold", pad=1)
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.3, wspace=0.1, top=0.97, bottom=0.01,
                        left=0.01, right=0.99)
    plt.savefig("vlmsareblind_image_grid.png", dpi=100, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved vlmsareblind_image_grid.png ({total} samples, {ROWS}x{COLS} grid)")
    plt.close()


    # =======================================================================
    # IMAGE 3: Detailed failure view -- sample of failures with images + info
    # =======================================================================
    # Show up to 100 failures (sampled evenly across tasks)
    MAX_FAILURES = 100
    if len(failures) > MAX_FAILURES:
        # Sample evenly across tasks
        sampled = []
        tasks_in_failures = list(set(r["task"] for r in failures))
        per_task = max(1, MAX_FAILURES // len(tasks_in_failures))
        for task_name in sorted(tasks_in_failures):
            task_fails = [r for r in failures if r["task"] == task_name]
            step = max(1, len(task_fails) // per_task)
            sampled.extend(task_fails[::step][:per_task])
        show_failures = sampled[:MAX_FAILURES]
    else:
        show_failures = failures

    n_fail = len(show_failures)
    FCOLS = 5
    FROWS = (n_fail + FCOLS - 1) // FCOLS
    cell_fw, cell_fh = 4.0, 4.5
    fig3, axes3 = plt.subplots(max(FROWS, 1), FCOLS,
                                figsize=(FCOLS * cell_fw, max(FROWS, 1) * cell_fh),
                                facecolor=BG)
    fig3.suptitle(
        f"Qwen3.5-4B-Base \u00b7 VLMs-Are-Blind FAILURES \u00b7 "
        f"{len(failures)} Total ({n_fail} shown)",
        fontsize=18, fontweight="bold", color=RED, y=1.0
    )

    for idx in range(max(FROWS, 1) * FCOLS):
        row, col = idx // FCOLS, idx % FCOLS
        if FROWS > 1:
            ax = axes3[row][col]
        elif FCOLS > 1:
            ax = axes3[col]
        else:
            ax = axes3
        ax.set_facecolor(CARD)

        if idx < n_fail:
            r = show_failures[idx]
            try:
                img = decode_image(r["image_b64"])
            except Exception:
                img = PILImage.new("RGB", (64, 64), color=(30, 30, 30))

            gt = r["ground_truth"]
            pred = r["predicted_answer"]
            prompt_short = r["prompt"][:80]
            if len(r["prompt"]) > 80:
                prompt_short += "..."
            out_short = r["model_output"][:60]
            if len(r["model_output"]) > 60:
                out_short += "..."

            # Draw image in top portion
            ax.imshow(np.array(img), extent=[0, 1, 0.40, 1.0],
                      aspect="auto", transform=ax.transAxes, clip_on=True)
            for spine in ax.spines.values():
                spine.set_edgecolor(RED)
                spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title(f"#{r['index']}  |  {r['task']}",
                         fontsize=7, color=YELLOW, fontweight="bold", pad=2)

            info = (
                f"GT: {gt}   Pred: {pred}\n"
                f"Q: {textwrap.fill(prompt_short, 38)}\n"
                f"Out: {textwrap.fill(out_short, 38)}"
            )
            ax.text(0.05, 0.36, info, transform=ax.transAxes, fontsize=5.5,
                    color=WHITE, fontfamily="monospace", verticalalignment="top",
                    bbox=dict(facecolor="#21262d", edgecolor="#30363d",
                              alpha=0.9, pad=3))
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.55, wspace=0.2, top=0.96, bottom=0.01,
                        left=0.02, right=0.98)
    plt.savefig("vlmsareblind_failures.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved vlmsareblind_failures.png ({n_fail} failure cards)")
    plt.close()

print("\nDone! Generated:")
print("  1. vlmsareblind_analysis.png   -- summary dashboard")
print("  2. vlmsareblind_image_grid.png -- all samples grid")
print("  3. vlmsareblind_failures.png   -- failure detail cards")
