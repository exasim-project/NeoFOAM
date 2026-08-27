"""Cost composition of one SIMPLE iteration AFTER the multigrid optimizations (the tuned champion:
cached hierarchy, post-pass scale correction, L6, merged levels 2, localized coarse solve).

Linear-solve wall times are the clean, overhead-free per-equation numbers from the corrected-binding
champion run `mgscpost-coarse-reltol/L6e1-20260717-094639.log` (steady 0.79 s/step, 1 rank/GPU). The
pressure solve is split into its fixed per-solve Galerkin refresh and its per-iteration Krylov apply
by the amortization of §4.3 (refresh ~45 ms, apply ~13 ms/iter x 8.4 iters ~= 106 ms).

Momentum solve = 46 ms: the U system is ONE multi-RHS (Vec3) solve reported once per component in the
log (Ux=Uy=Uz=45.7 ms). An earlier version of this figure SUMMED the three identical lines to 130 ms
-- a 3x overcount. With the implicit slip/symmetry BC this single solve is preserved by the fused
per-column-diagonal operator (FusedDiagShiftMatrix); the un-fused slip path would instead cost three
separate solves (~55 ms).

The old figure lumped everything that is not a linear solve into one grey "assemble + overhead"
remainder. Here that remainder is split into momentum assembly, pressure assembly, pressure-velocity
coupling (rAU / HbyA / phiHbyA), and turbulence assembly/update, using the per-corrector
solve/assembly SHARES measured by the space-time-stack region profile
`champion-costbreakdown/champ-spacetimestack-20260717-132600.kokkos-profile.txt` (30-step window):
each corrector's clean total = clean_solve / (profiled solve share), and the non-solve part is
apportioned by the profiled region tree. What is left of the 0.79 s step after the correctors is
genuine host / MPI / sync overhead (Kokkos Vector churn + halo cuStreamSynchronize, cf. the L5c4
attribution) -- the largest single slice at ~34 %. Space-time-stack inflates the Ginkgo solve kernels
(p.linearSolve 457 ms profiled vs 151 ms clean), so only the WITHIN-corrector shares are used, never
its absolute solve times.

Story: linear solves are ~28 % of the step (pressure 19 %); momentum assembly (26 %) and host/sync
overhead (34 %) now dominate -- not the solves. (Fig 2, the uncached reference, was 77 % pressure.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_MS = 790.0   # steady s/step = 0.790 s

# --- clean linear-solve times (from the champion log) ---
P_REFRESH, P_APPLY = 45.0, 106.0      # pressure solve = 151 ms total
U_SOLVE            = 46.0             # momentum: ONE fused multi-RHS solve (not 3x)
TURB_SOLVE         = 27.6             # k + omega

# --- assembly / coupling split, from the space-time-stack per-corrector shares ---
#   profiled shares: momentumPredictor solve=18.2%; pressureCorrector p-solve=84.6%
#                    (non-solve: p.assemble 16.8 : pv-coupling 60.0); turbCorrect solve=29.6%
U_ASM   = U_SOLVE/0.1814 - U_SOLVE                    # momentum assembly (assemble + construct + src)
_pcorr  = (P_REFRESH+P_APPLY)/0.846                   # clean pressureCorrector total
_pnon   = _pcorr - (P_REFRESH+P_APPLY)
P_ASM   = _pnon * 16.8/(16.8+60.0)                    # pressure assembly
PVC     = _pnon - P_ASM                               # pressure-velocity coupling: rAU, HbyA, phiHbyA
TURB_ASM = TURB_SOLVE/0.296 - TURB_SOLVE             # turbulence assemble + model update
OTHER   = STEP_MS - (P_REFRESH+P_APPLY+P_ASM+PVC+U_ASM+U_SOLVE+TURB_ASM+TURB_SOLVE)

# label, ms, colour  (pressure=red family, momentum=blue, coupling=purple, turbulence=green, overhead=grey)
seg = [
    ("Momentum — assembly (grad, div·lap, deferred corr)", U_ASM,     "#2166ac"),
    ("Momentum — linear solve (U, fused multi-RHS)",        U_SOLVE,   "#67a9cf"),
    ("Pressure–velocity coupling (rAU, HbyA, phiHbyA)",     PVC,       "#762a83"),
    ("Pressure — Galerkin refresh (per solve)",             P_REFRESH, "#b2182b"),
    ("Pressure — Krylov apply (Cg + V-cycles)",             P_APPLY,   "#ef8a62"),
    ("Pressure — assembly",                                 P_ASM,     "#fddbc7"),
    ("Turbulence — assembly + model update (k, ω)",         TURB_ASM,  "#1b7837"),
    ("Turbulence — linear solve (k, ω)",                    TURB_SOLVE,"#a6dba0"),
    ("Host / MPI / sync overhead (Vector churn, halo sync)",OTHER,     "#999999"),
]
assert abs(sum(s for _, s, _ in seg) - STEP_MS) < 1.0

fig, ax = plt.subplots(figsize=(9.8, 3.0))
left = 0.0
light = {"#67a9cf", "#ef8a62", "#fddbc7", "#a6dba0", "#999999"}   # segments needing dark text
for label, ms, c in seg:
    ax.barh(0, ms, left=left, color=c, edgecolor="white", linewidth=1.4)
    pct = 100 * ms / STEP_MS
    if pct >= 3.5:
        ax.text(left + ms / 2, 0, f"{pct:.0f}%", ha="center", va="center",
                fontsize=9.5, color="#1a1a1a" if c in light else "white", fontweight="bold")
    left += ms

# bracket the pressure solve (the two red solve segments) so the "19 %" total reads at a glance
ps_lo = U_ASM + U_SOLVE + PVC
ps_hi = ps_lo + P_REFRESH + P_APPLY
ax.annotate("", xy=(ps_lo, 0.42), xytext=(ps_hi, 0.42),
            arrowprops=dict(arrowstyle="-", color="#b2182b", lw=1.2))
ax.text((ps_lo + ps_hi) / 2, 0.5, "pressure solve  151 ms · 19 %", ha="center", va="bottom",
        fontsize=8.5, color="#b2182b", fontweight="bold")

ax.set_xlim(0, STEP_MS)
ax.set_ylim(-0.5, 0.72)
ax.set_yticks([])
ax.set_xlabel("wall time per SIMPLE iteration  (ms, steady state; solve times clean, "
              "assembly split from the region profile)", labelpad=6)
ax.set_title("Cost composition after multigrid optimization — cached, sc-post, L6 — 0.79 s/step\n"
             "linear solves = 28 % (pressure 19 %); momentum assembly (26 %) and host/sync "
             "overhead (34 %) now dominate the step",
             fontsize=10.5)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in seg]
labels = [f"{lbl}  ({ms:.0f} ms, {100*ms/STEP_MS:.0f}%)" for lbl, ms, _ in seg]
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.82),
          ncol=2, fontsize=8.3, frameon=False)
fig.subplots_adjust(bottom=0.46, top=0.80)
out = "paperParamStudyResults/champion-cost-breakdown.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
for lbl, ms, _ in seg:
    print(f"  {lbl:52s} {ms:6.1f} ms  {100*ms/STEP_MS:4.1f}%")
