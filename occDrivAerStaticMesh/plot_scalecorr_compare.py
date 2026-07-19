#!/usr/bin/env python3
"""Two separate figures for the scale-correction studies:
  scalecorr-improve.png  -- bar plot (mechanism sweep), colored by sc-family
  mgsc-mergelevels.png   -- grouped bar (no-sc vs mgsc) across mergeLevels
Each figure has two panels: marginal s/step and mean pressure iterations. Bars are
zero-baselined (never truncated) with value labels so small s/step gaps stay readable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8880"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"

def rounded_bars(ax, xs, heights, colors, width=0.62, r=None):
    # clean flat-top bars with a thin surface gap between adjacent fills
    ax.bar(xs, heights, width=width, color=colors, edgecolor=SURFACE,
           linewidth=1.4, zorder=3)

def style(ax, ylab, ymax, ystep):
    ax.set_ylabel(ylab, fontsize=10, color=INK2, labelpad=6)
    ax.set_ylim(0, ymax)
    ax.set_yticks([y for y in _frange(0, ymax, ystep)])
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_facecolor(SURFACE)

def _frange(a, b, step):
    v = a
    while v <= b + 1e-9:
        yield round(v, 6); v += step

# ---------------------------------------------------------------- scalecorr-improve
SC = [  # variant, s_step, p_iters, family
    ("off",       4.520, 16.5, "baseline"),
    ("ir-sc",     5.580, 14.3, "per-smoother sc"),
    ("fcg-ir-sc", 5.280, 12.5, "per-smoother sc"),
    ("mgsc-cg",   4.320,  6.0, "MG-level sc"),
    ("mgsc-fcg",  4.760,  7.6, "MG-level sc"),
]
FAM_COLOR = {"baseline": GRAY, "per-smoother sc": ORANGE, "MG-level sc": BLUE}
names = [d[0] for d in SC]; xs = list(range(len(SC)))
cols = [FAM_COLOR[d[3]] for d in SC]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.5), dpi=150)
fig.patch.set_facecolor(SURFACE)
rounded_bars(a1, xs, [d[1] for d in SC], cols)
rounded_bars(a2, xs, [d[2] for d in SC], cols)
for ax, key, ymax, fmt in ((a1, 1, 6.2, "%.2f"), (a2, 2, 18.5, "%.1f")):
    for x, d in zip(xs, SC):
        ax.text(x, d[key] + ymax*0.02, fmt % d[key], ha="center", va="bottom",
                fontsize=8.5, color=INK)
    style(ax, "marginal s/step" if key == 1 else "mean pressure iters / solve", ymax, 1 if key==1 else 5)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5, color=INK2)
a1.axhline(4.52, color=GRAY, lw=1, ls="--", zorder=1)  # baseline reference
handles = [plt.Line2D([0],[0], marker="s", ls="", ms=9, mfc=FAM_COLOR[f], mec="none")
           for f in ("baseline", "per-smoother sc", "MG-level sc")]
a1.legend(handles, ("baseline (no sc)", "per-smoother sc", "MG-level sc"),
          frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle("scalecorr-improve: only MG-level sc beats the no-sc baseline",
             fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
fig.text(0.01, 0.005, "occDrivAer 4×H200, 50 steps, cached, global Cg+MG. per-smoother sc cuts iters "
         "but raises s/step; MG-level sc cuts both.", fontsize=6.8, color=INK2, ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig("paperParamStudyResults/scalecorr-improve.png", facecolor=SURFACE, bbox_inches="tight")
print("wrote paperParamStudyResults/scalecorr-improve.png")

# --------------------------------------------------------------- mgsc-mergelevels
ML = {  # merge -> (nosc s_step, nosc iters, mgsc s_step, mgsc iters)
    "merge1": (4.480, 16.4, 4.340, 6.0),
    "merge2": (4.620, 19.6, 4.320, 7.0),
    "merge3": (4.560, 23.0, 4.240, 9.1),
}
merges = list(ML.keys()); gx = list(range(len(merges))); w = 0.36
fig2, (b1, b2) = plt.subplots(1, 2, figsize=(9.0, 4.5), dpi=150)
fig2.patch.set_facecolor(SURFACE)

def grouped(ax, idx, ymax, fmt):
    for i, m in enumerate(merges):
        nosc, mgsc = ML[m][idx], ML[m][idx+2]
        rounded_bars(ax, [i - w/2], [nosc], [ORANGE], width=w)
        rounded_bars(ax, [i + w/2], [mgsc], [BLUE], width=w)
        ax.text(i - w/2, nosc + ymax*0.02, fmt % nosc, ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(i + w/2, mgsc + ymax*0.02, fmt % mgsc, ha="center", va="bottom", fontsize=8, color=INK)
    style(ax, "marginal s/step" if idx == 0 else "mean pressure iters / solve", ymax, 1 if idx==0 else 5)
    ax.set_xticks(gx); ax.set_xticklabels(merges, fontsize=9, color=INK2)

grouped(b1, 0, 5.7, "%.2f")
grouped(b2, 1, 27, "%.1f")
handles2 = [plt.Line2D([0],[0], marker="s", ls="", ms=9, mfc=c, mec="none") for c in (ORANGE, BLUE)]
b1.legend(handles2, ("no sc", "MG-level sc"), frameon=False, fontsize=9, loc="upper left")
fig2.suptitle("mgsc-mergelevels: MG-level sc wins at every merge level; merge compounds it",
              fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
fig2.text(0.01, 0.005, "occDrivAer 4×H200, 50 steps, cached, global Cg+MG + pgmMergeN. no-sc: merge is "
          "a wash (iters rise, cycle cheaper); with mgsc: best at merge3 (4.24).", fontsize=6.8, color=INK2, ha="left")
fig2.tight_layout(rect=(0, 0.03, 1, 0.96))
fig2.savefig("paperParamStudyResults/mgsc-mergelevels.png", facecolor=SURFACE, bbox_inches="tight")
print("wrote paperParamStudyResults/mgsc-mergelevels.png")
