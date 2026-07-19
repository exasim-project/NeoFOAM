import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 30  # SIMPLE iterations in the profiled timeStep region
# region totals (s) from costbrk-spacetimestack top-down tree; setup EXCLUDED
seg = [
    # label,                       seconds_total, family_color
    ("Pressure — MG hierarchy rebuild\n(ginkgo.solverSetup)", 57.1, "#b2182b"),
    ("Pressure — Krylov solve (Cg + V-cycles)",               59.9, "#ef8a62"),
    ("Pressure — assemble + flux/corrector",                  13.0, "#fddbc7"),
    ("Momentum — assemble (incl. deferred-corr)",              9.05, "#2166ac"),
    ("Momentum — linear solve (BiCGStab)",                     8.16, "#67a9cf"),
    ("Momentum — construct + source",                          2.39, "#d1e5f0"),
    ("Turbulence — k/omega linear solve",                      8.29, "#1b7837"),
    ("Turbulence — assemble + model update",                   9.81, "#a6dba0"),
    ("Write",                                                  1.96, "#999999"),
]
total = sum(s for _, s, _ in seg)      # ~170 s
per_step = total / N
print(f"timeStep total {total:.1f}s over {N} -> {per_step:.3f} s/step (setup excluded)")

fig, ax = plt.subplots(figsize=(9.6, 2.9))
left = 0.0
for label, s, c in seg:
    v = s / N
    ax.barh(0, v, left=left, color=c, edgecolor="white", linewidth=0.8)
    pct = 100 * s / total
    if pct >= 4:   # label big segments in-place
        ax.text(left + v/2, 0, f"{pct:.0f}%", ha="center", va="center",
                fontsize=9, color="white" if c in ("#b2182b","#2166ac","#1b7837") else "black",
                fontweight="bold")
    left += v

ax.set_xlim(0, per_step)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("wall time per average SIMPLE iteration  (s, instrumented; setup excluded)")
ax.set_title(f"Cost composition of one average SIMPLE iteration — {per_step:.2f} s/step\n"
             f"(profiled 30-step window, NP=4; pressure = {100*130.0/total:.0f}% of the step)",
             fontsize=11)

handles = [plt.Rectangle((0,0),1,1,color=c) for _,_,c in seg]
labels  = [f"{lbl}  ({s/N:.2f} s, {100*s/total:.0f}%)" for lbl, s, _ in seg]
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.55),
          ncol=2, fontsize=8.5, frameon=False)
fig.tight_layout()
out = "paperParamStudyResults/cost-breakdown-average-iteration.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
