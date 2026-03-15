"""
Visualize SPHERE-VLM evaluation results from sphere_results.json.

Produces:
  1. sphere_analysis.png      -- summary dashboard (group/config/viewpoint/format breakdowns)
  2. sphere_image_grid.png    -- grid of ALL samples with thumbnail images
  3. sphere_failures.png      -- detailed failure view with images + question text

Usage:
    python3 visualize_sphere.py
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
with open("sphere_results.json") as f:
    results = json.load(f)

with open("sphere_stats.json") as f:
    stats = json.load(f)

total = len(results)
correct = sum(1 for r in results if r["correct"])
accuracy = correct / total * 100
has_images = "image_b64" in results[0]

# Color palette (same dark theme as mmstar)
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

GROUP_COLORS = {
    "single_skill": BLUE,
    "combine_2_skill": PURPLE,
    "reasoning": ORANGE,
}

FORMAT_COLORS = {
    "num": CYAN,
    "name": BLUE,
    "pos": PURPLE,
    "bool": ORANGE,
}

VP_COLORS = {
    "allo": GREEN,
    "ego": RED,
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
# Per-config accuracy
config_data = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    c = r["config"]
    config_data[c]["total"] += 1
    if r["correct"]:
        config_data[c]["correct"] += 1

config_names = sorted(config_data.keys(),
                      key=lambda x: config_data[x]["correct"] / config_data[x]["total"])
config_accs = [config_data[n]["correct"] / config_data[n]["total"] * 100 for n in config_names]
config_totals = [config_data[n]["total"] for n in config_names]

# Per-group accuracy
group_data = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    g = r["group"]
    group_data[g]["total"] += 1
    if r["correct"]:
        group_data[g]["correct"] += 1

group_names = sorted(group_data.keys(),
                     key=lambda x: group_data[x]["correct"] / group_data[x]["total"])
group_accs = [group_data[n]["correct"] / group_data[n]["total"] * 100 for n in group_names]
group_totals = [group_data[n]["total"] for n in group_names]

# Per-viewpoint accuracy
vp_data = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    v = r["viewpoint"]
    vp_data[v]["total"] += 1
    if r["correct"]:
        vp_data[v]["correct"] += 1

vp_names = sorted(vp_data.keys())
vp_accs = [vp_data[n]["correct"] / vp_data[n]["total"] * 100 for n in vp_names]
vp_totals = [vp_data[n]["total"] for n in vp_names]

# Per-format accuracy
fmt_data = defaultdict(lambda: {"correct": 0, "total": 0})
for r in results:
    f = r["answer_format"]
    fmt_data[f]["total"] += 1
    if r["correct"]:
        fmt_data[f]["correct"] += 1

fmt_names = sorted(fmt_data.keys(),
                   key=lambda x: fmt_data[x]["correct"] / fmt_data[x]["total"])
fmt_accs = [fmt_data[n]["correct"] / fmt_data[n]["total"] * 100 for n in fmt_names]
fmt_totals = [fmt_data[n]["total"] for n in fmt_names]

# Failures
failures = [r for r in results if not r["correct"]]
failure_configs = Counter(r["config"] for r in failures)
failure_groups = Counter(r["group"] for r in failures)
failure_vps = Counter(r["viewpoint"] for r in failures)

# Correctness grid for heatmap
nrows_grid = (total + 9) // 10
ncols_grid = min(total, 10)
correctness_grid = np.zeros((nrows_grid, ncols_grid), dtype=int)
for idx, r in enumerate(results):
    correctness_grid[idx // 10][idx % 10] = 1 if r["correct"] else 0

# Per-config accuracy within each group (for stacked view)
group_config_map = defaultdict(list)
for c in config_names:
    # Find the group for this config
    for r in results:
        if r["config"] == c:
            group_config_map[r["group"]].append(c)
            break


# ===========================================================================
# IMAGE 1: Summary dashboard
# ===========================================================================
fig = plt.figure(figsize=(22, 16), facecolor=BG)
fig.suptitle(
    f"Qwen3.5-4B-Base  \u00b7  SPHERE-VLM Evaluation  \u00b7  {total} Samples",
    fontsize=20, fontweight="bold", color="white", y=0.98
)
gs = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.4,
                       left=0.06, right=0.96, top=0.92, bottom=0.05)

# ---------- Panel 1: Donut ----------
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

# ---------- Panel 2: Per-group bars ----------
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Accuracy by Skill Group")
bar_colors = [GROUP_COLORS.get(n, GRAY) for n in group_names]
bars = ax2.barh(range(len(group_names)), group_accs, color=bar_colors,
                height=0.5, edgecolor="#30363d")
ax2.set_yticks(range(len(group_names)))
ax2.set_yticklabels([f"{n}\n(n={group_totals[i]})" for i, n in enumerate(group_names)],
                     fontsize=10, color=WHITE)
ax2.set_xlim(0, 105)
ax2.set_xlabel("Accuracy %", color=GRAY, fontsize=10)
ax2.axvline(50, color=GRAY, linestyle="--", alpha=0.4)
for i, acc in enumerate(group_accs):
    ax2.text(acc + 1.5, i, f"{acc:.1f}%", va="center", fontsize=11,
             color=WHITE, fontweight="bold")

# ---------- Panel 3: Viewpoint comparison ----------
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Ego vs Allocentric Viewpoint")
vp_colors = [VP_COLORS.get(n, GRAY) for n in vp_names]
bars3 = ax3.bar(range(len(vp_names)), vp_accs, color=vp_colors,
                width=0.5, edgecolor="#30363d")
ax3.set_xticks(range(len(vp_names)))
ax3.set_xticklabels([f"{n}\n(n={vp_totals[i]})" for i, n in enumerate(vp_names)],
                     fontsize=11, color=WHITE)
ax3.set_ylim(0, 105)
ax3.set_ylabel("Accuracy %", color=GRAY, fontsize=10)
ax3.axhline(50, color=GRAY, linestyle="--", alpha=0.4)
for i, acc in enumerate(vp_accs):
    ax3.text(i, acc + 2, f"{acc:.1f}%", ha="center", fontsize=12,
             color=WHITE, fontweight="bold")
# Annotate the gap
if len(vp_accs) == 2:
    gap = abs(vp_accs[0] - vp_accs[1])
    ax3.text(0.5, max(vp_accs) + 8, f"\u0394 = {gap:.1f}pp",
             ha="center", fontsize=11, color=YELLOW, fontweight="bold",
             transform=ax3.get_xaxis_transform())

# ---------- Panel 4: Answer format comparison ----------
ax4 = fig.add_subplot(gs[0, 3])
style_ax(ax4, "Accuracy by Answer Format")
fc = [FORMAT_COLORS.get(n, GRAY) for n in fmt_names]
bars4 = ax4.bar(range(len(fmt_names)), fmt_accs, color=fc,
                width=0.5, edgecolor="#30363d")
ax4.set_xticks(range(len(fmt_names)))
ax4.set_xticklabels([f"{n}\n(n={fmt_totals[i]})" for i, n in enumerate(fmt_names)],
                     fontsize=10, color=WHITE)
ax4.set_ylim(0, 105)
ax4.set_ylabel("Accuracy %", color=GRAY, fontsize=10)
ax4.axhline(50, color=GRAY, linestyle="--", alpha=0.4)
for i, acc in enumerate(fmt_accs):
    ax4.text(i, acc + 2, f"{acc:.1f}%", ha="center", fontsize=11,
             color=WHITE, fontweight="bold")

# ---------- Panel 5: Per-config bars (main chart) ----------
ax5 = fig.add_subplot(gs[1, :3])
style_ax(ax5, "Accuracy by Config (colored by group)")
cfg_bar_colors = []
for c in config_names:
    for r in results:
        if r["config"] == c:
            cfg_bar_colors.append(GROUP_COLORS.get(r["group"], GRAY))
            break
ax5.barh(range(len(config_names)), config_accs, color=cfg_bar_colors,
         height=0.6, edgecolor="#30363d")
ax5.set_yticks(range(len(config_names)))
short_names = [c.replace("counting_only-paired-", "count-paired-")
               .replace("_w_intermediate", "_w_int")
               for c in config_names]
ax5.set_yticklabels([f"{short_names[i]} (n={config_totals[i]})"
                     for i in range(len(config_names))],
                     fontsize=9, color=WHITE)
ax5.set_xlim(0, 115)
ax5.set_xlabel("Accuracy %", color=GRAY, fontsize=10)
ax5.axvline(50, color=GRAY, linestyle="--", alpha=0.4)
ax5.axvline(30, color=RED, linestyle=":", alpha=0.3)
for i, acc in enumerate(config_accs):
    ax5.text(acc + 1.5, i, f"{acc:.1f}%", va="center", fontsize=10,
             color=WHITE, fontweight="bold")

# Legend for groups
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GROUP_COLORS[g], label=g)
                   for g in ["single_skill", "combine_2_skill", "reasoning"]]
ax5.legend(handles=legend_elements, loc="lower right", fontsize=9,
           frameon=True, facecolor=CARD, edgecolor="#30363d", labelcolor=WHITE)

# ---------- Panel 6: Heatmap ----------
ax6 = fig.add_subplot(gs[1, 3])
style_ax(ax6, "Per-Sample Results")
cmap = matplotlib.colors.ListedColormap([RED, GREEN])
ax6.imshow(correctness_grid, cmap=cmap, aspect="auto", interpolation="nearest")
ax6.set_xlabel("Sample (mod 10)", color=GRAY, fontsize=9)
ax6.set_ylabel("Sample (\u00f7 10)", color=GRAY, fontsize=9)
ax6.set_xticks(range(ncols_grid))
ax6.set_yticks(range(nrows_grid))
ax6.set_xticklabels(range(ncols_grid), fontsize=8, color=GRAY)
ax6.set_yticklabels([f"{i*10}" for i in range(nrows_grid)], fontsize=8, color=GRAY)
for i in range(nrows_grid):
    for j in range(ncols_grid):
        idx = i * 10 + j
        if idx < total:
            ax6.text(j, i, str(idx), ha="center", va="center", fontsize=6,
                     color="black" if correctness_grid[i][j] else WHITE,
                     fontweight="bold")

# ---------- Panel 7: Failure analysis text ----------
ax7 = fig.add_subplot(gs[2, :2])
style_ax(ax7, "Failure Analysis Summary")
ax7.axis("off")
lines = [
    f"Total failures: {len(failures)}/{total}",
    f"Model accuracy: {accuracy:.1f}%",
    "",
    "Failures by group:",
]
for g in ["reasoning", "combine_2_skill", "single_skill"]:
    if g in failure_groups:
        gt = group_data[g]["total"]
        lines.append(f"  \u00b7 {g}: {failure_groups[g]}/{gt} wrong "
                     f"({failure_groups[g]/gt*100:.0f}% error rate)")

lines += ["", "Failures by viewpoint:"]
for v in sorted(failure_vps, key=lambda x: failure_vps[x], reverse=True):
    vt = vp_data[v]["total"]
    lines.append(f"  \u00b7 {v}: {failure_vps[v]}/{vt} wrong "
                 f"({failure_vps[v]/vt*100:.0f}% error rate)")

lines += ["", "Failures by config (top 5):"]
for cfg, count in failure_configs.most_common(5):
    ct = config_data[cfg]["total"]
    lines.append(f"  \u00b7 {cfg}: {count}/{ct} ({count/ct*100:.0f}%)")

ax7.text(0.05, 0.95, "\n".join(lines), transform=ax7.transAxes, fontsize=10,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                   edgecolor="#30363d", alpha=0.9))

# ---------- Panel 8: Blind spot callout ----------
ax8 = fig.add_subplot(gs[2, 2:])
style_ax(ax8, "Identified Blind Spots")
ax8.axis("off")
bs_lines = [
    "BLIND SPOTS (low-accuracy areas):",
    "",
]

# Viewpoint
for v in vp_names:
    acc_v = vp_data[v]["correct"] / vp_data[v]["total"] * 100
    if acc_v < 50:
        bs_lines.append(f"  \u26a0  {v} viewpoint: {acc_v:.1f}%  "
                        f"(n={vp_data[v]['total']})")

# Configs
for c in config_names:
    acc_c = config_data[c]["correct"] / config_data[c]["total"] * 100
    if acc_c < 50:
        bs_lines.append(f"  \u26a0  {c}: {acc_c:.1f}%  "
                        f"(n={config_data[c]['total']})")

# Groups
for g in group_names:
    acc_g = group_data[g]["correct"] / group_data[g]["total"] * 100
    if acc_g < 55:
        bs_lines.append(f"  \u26a0  {g} group: {acc_g:.1f}%  "
                        f"(n={group_data[g]['total']})")

bs_lines += [
    "",
    "KEY FINDING:",
    f"  Ego viewpoint ({vp_data.get('ego',{}).get('correct',0)}"
    f"/{vp_data.get('ego',{}).get('total',0)}) vs "
    f"Allo ({vp_data.get('allo',{}).get('correct',0)}"
    f"/{vp_data.get('allo',{}).get('total',0)})",
]
if "ego" in vp_data and "allo" in vp_data:
    ego_acc = vp_data["ego"]["correct"] / vp_data["ego"]["total"] * 100
    allo_acc = vp_data["allo"]["correct"] / vp_data["allo"]["total"] * 100
    bs_lines.append(f"  Gap: {allo_acc:.1f}% - {ego_acc:.1f}% = "
                    f"{allo_acc - ego_acc:.1f}pp drop")
    bs_lines.append(f"  \u2192 Model struggles with egocentric")
    bs_lines.append(f"    spatial reasoning")

bs_lines += [
    "",
    "SKILL DEGRADATION:",
    f"  single_skill \u2192 combine_2 \u2192 reasoning",
]
for g in ["single_skill", "combine_2_skill", "reasoning"]:
    if g in group_data:
        ga = group_data[g]["correct"] / group_data[g]["total"] * 100
        bs_lines.append(f"    {g}: {ga:.1f}%")

ax8.text(0.05, 0.95, "\n".join(bs_lines), transform=ax8.transAxes, fontsize=10,
         color=WHITE, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d",
                   edgecolor="#30363d", alpha=0.9))

plt.savefig("sphere_analysis.png", dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
print(f"Saved sphere_analysis.png ({total} samples, {accuracy:.1f}% accuracy)")
plt.close()


# ===========================================================================
# IMAGE 2: Full image grid -- all samples with thumbnails
# ===========================================================================
if has_images:
    COLS = 10
    ROWS = (total + COLS - 1) // COLS
    cell_w, cell_h = 2.0, 2.6
    fig2, axes2 = plt.subplots(ROWS, COLS, figsize=(COLS * cell_w, ROWS * cell_h),
                                facecolor=BG)
    fig2.suptitle(
        f"Qwen3.5-4B-Base \u00b7 SPHERE-VLM All Samples \u00b7 {accuracy:.0f}% Accuracy "
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
            img = decode_image(r["image_b64"])
            border_color = GREEN if r["correct"] else RED

            gt_short = r["ground_truth"][:15]
            pred_short = r["predicted_answer"][:15]

            ax.imshow(np.array(img))
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])

            status = "\u2713" if r["correct"] else "\u2717"
            cfg_short = r["config"][:20]
            ax.set_title(f"#{idx} {cfg_short}\nGT:{gt_short} P:{pred_short} {status}",
                         fontsize=5.5, color=border_color, fontweight="bold", pad=2)
        else:
            ax.axis("off")

    plt.subplots_adjust(hspace=0.55, wspace=0.15, top=0.96, bottom=0.01,
                        left=0.01, right=0.99)
    plt.savefig("sphere_image_grid.png", dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.2)
    print(f"Saved sphere_image_grid.png ({total} samples, {ROWS}x{COLS} grid)")
    plt.close()


    # =======================================================================
    # IMAGE 3: Detailed failure view
    # =======================================================================
    n_fail = len(failures)
    if n_fail > 0:
        FCOLS = 5
        FROWS = (n_fail + FCOLS - 1) // FCOLS
        cell_fw, cell_fh = 4.2, 5.0
        fig3, axes3 = plt.subplots(FROWS, FCOLS,
                                    figsize=(FCOLS * cell_fw, FROWS * cell_fh),
                                    facecolor=BG)
        fig3.suptitle(
            f"Qwen3.5-4B-Base \u00b7 SPHERE-VLM FAILURES \u00b7 {n_fail} Wrong Answers",
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

            if idx < n_fail:
                r = failures[idx]
                img = decode_image(r["image_b64"])

                gt = r["ground_truth"]
                pred = r["predicted_answer"]
                q_short = r["question"][:100]
                if len(r["question"]) > 100:
                    q_short += "..."
                out_short = r["model_output"][:80]
                if len(r["model_output"]) > 80:
                    out_short += "..."
                opts_str = ""
                if r["options"]:
                    opts_str = " | ".join(r["options"][:4])
                    if len(opts_str) > 60:
                        opts_str = opts_str[:60] + "..."

                # Draw image in top portion
                ax.imshow(np.array(img), extent=[0, 1, 0.38, 1.0],
                          aspect="auto", transform=ax.transAxes, clip_on=True)
                for spine in ax.spines.values():
                    spine.set_edgecolor(RED)
                    spine.set_linewidth(2)
                ax.set_xticks([]); ax.set_yticks([])

                # Title with config + viewpoint
                vp_icon = "\U0001f441" if r["viewpoint"] == "ego" else "\U0001f30d"
                ax.set_title(
                    f"#{r['index']}  {r['config'][:30]}  [{r['viewpoint']}]",
                    fontsize=6, color=YELLOW, fontweight="bold", pad=2)

                # Info text below image
                info_lines = [
                    f"GT: {gt}",
                    f"Pred: {pred}",
                    f"Format: {r['answer_format']}  VP: {r['viewpoint']}",
                ]
                if opts_str:
                    info_lines.append(f"Opts: {opts_str}")
                info_lines.append(f"Q: {textwrap.fill(q_short, 42)}")
                info_lines.append(f"Out: {textwrap.fill(out_short, 42)}")
                info = "\n".join(info_lines)

                ax.text(0.05, 0.35, info, transform=ax.transAxes, fontsize=5,
                        color=WHITE, fontfamily="monospace", verticalalignment="top",
                        bbox=dict(facecolor="#21262d", edgecolor="#30363d",
                                  alpha=0.9, pad=3))
            else:
                ax.axis("off")

        plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.96, bottom=0.01,
                            left=0.02, right=0.98)
        plt.savefig("sphere_failures.png", dpi=150, facecolor=BG,
                    bbox_inches="tight", pad_inches=0.2)
        print(f"Saved sphere_failures.png ({n_fail} failures, {FROWS}x{FCOLS} grid)")
        plt.close()
    else:
        print("No failures to visualize!")

print("\nDone. Generated files:")
print("  sphere_analysis.png    -- summary dashboard")
print("  sphere_image_grid.png  -- thumbnail grid of all samples")
print("  sphere_failures.png    -- detailed failure cards")
