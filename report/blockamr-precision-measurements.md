# blockAMR reduced-precision measurements

The rationale and measurements that used to live as long comment blocks in three
headers under `src/NeoN/include/NeoN/blockAmr/`. Each header now carries a short
conclusion and points here for the numbers.

Sources:

- [`gmg/bf16.hpp`](#bfloat16-as-a-storage-type-for-the-gmg-level-hierarchy) — bf16 as a storage type for the GMG level hierarchy
- [`krylov/mixed_precision.hpp`](#an-fp32-solver-wearing-an-fp64-linops-clothes) — fp32 Krylov behind fp64 iterative refinement
- [`bench/kokkos_bench.hpp`](#gmgargscoeffprecision----the-fieldscoefficients-split) — the `GmgArgs::coeffPrecision` field/coefficient split

---

## bfloat16 as a STORAGE type for the GMG level hierarchy

*(from `src/NeoN/include/NeoN/blockAmr/linearAlgebra/gmg/bf16.hpp`)*

bfloat16 as a STORAGE type for the GMG level hierarchy: 1 sign + 8 exponent +
7 mantissa bits, i.e. FP32's exponent range in half of FP32's bytes. The
V-cycle is bandwidth-bound once the launch cost is gone, so what a level's
value type buys is bytes moved, not flops.

Why bf16 and not IEEE fp16, for THIS operator: the face coefficient of the
Laplacian is -beta/dx^2, which on a unit cube is -65536 at 256^3 and -262144
at 512^3. IEEE half tops out at 65504, so the coefficients themselves would
overflow to infinity before any arithmetic happened. bf16 keeps FP32's
exponent, so dynamic range is not the constraint here -- only the ~3
significant decimal digits are.

STORAGE ONLY, and that is a correctness requirement rather than a nicety.
Every operation below converts to float, computes in float and rounds once on
the way back; the kernels declare their locals as `GmgComputeT<T>` (= float
here) rather than `T`. What forces that is the residual, which the V-cycle forms
as a difference of two quantities vastly larger than itself:

    r = b - (diag * psi + off),   diag = alpha - sum(a_face) = 1 + 6/dx^2

At 256^3 with psi ~ 0.7 that is diag*psi = 275252 against off = -275212, whose
sum is 40. Round either intermediate to bf16 -- spacing 2048 up there -- and
both land on the SAME bf16 value, so the difference comes out exactly 0.0:
100% of the residual gone, at every grid size (the same experiment at 16^3
loses all of 0.85). Kept in float, the subtraction is exact given its inputs
and only the ~0.4% the stored values carry survives.

The diagonal itself is a red herring by comparison: bf16 does round 393217 to
393216 and lose alpha, but alpha's share of that diagonal is 2.5e-6, two
orders below bf16's own representation error. Storing the coefficients in
bf16 is fine; accumulating in it is not.

### What it measured: a negative result, kept because it is one

The bytes arrive. The 256^3/512-box V-cycle drops from 11.96 ms at fp32 to
8.82 ms, 1.36x. It is not enough, at any size measured.

The reason is amplification, and it is specific to what a V-cycle does with a
stored solution. Holding psi at ~0.4% puts a per-cell perturbation d into it,
and the quantity the cycle restricts to the coarse grid is r = b - A(psi + d),
so d arrives there multiplied by ||A||. For this operator ||A|| ~ 6/dx^2 =
6n^2: the noise floor of the restricted residual grows as n^2 while the
residual itself does not. The coarse grid is then correcting noise.

One V-cycle's residual reduction against the same fp64 cycle, and the CG
iterations that costs (256^3/512 boxes, norm=linf, l0 agglomerated):

| grid  | V-cycle weaker | fp32 iters | bf16 iters | solve vs fp32 |
| ----- | -------------- | ---------- | ---------- | ------------- |
| 16^3  | 1.05x          | --         | --         | --            |
| 64^3  | 1.26x          | 11         | 25         | 1.9x slower   |
| 128^3 | 1.87x          | 11         | 53         | 3.6x slower   |
| 256^3 | 3.23x          | 12         | 273        | 17.4x slower  |

The answers stay correct throughout -- the operator and the residual CG stops
on are fp64 whatever the hierarchy is stored in -- so this is purely a worse
preconditioner, never a wrong one. There is no crossover: 1.36x off the
V-cycle cannot pay for doubling the iteration count, which already happens at
64^3.

It is kept, wired and tested rather than deleted because a measured "no" is
worth more than an untested "probably not", and because the parts generalise:
`GmgComputeT` is what any reduced-precision level will need, and the precision
axis is now first-class in the bench.

### The refinement the numbers pointed at, and what it measured

Storing the COEFFICIENTS in bf16 while psi and rhs stay fp32 -- a coefficient
error is a 0.4% perturbation of the operator, which a preconditioner absorbs
without amplification, and it is 4 of the 6 arrays a shared-coefficient colour
sweep streams. That is `gmg_coeff_precision`, and it works. 256^3, one box,
level-0 agglomerated, one V-cycle from z0 = 0:

| fields/coeffs | ms/cycle | r1/r0 (smooth b) | CG iters | solve  |
| ------------- | -------- | ---------------- | -------- | ------ |
| fp32 / fp32   | 12.52    | 0.70185          | 9        | 213 ms |
| fp32 / bf16   | 10.60    | 0.70147          | 9        | 195 ms |
| bf16 / bf16   | 9.37     | 97.7 (!)         | --       | --     |

The middle column is the whole argument. Narrowing the COEFFICIENTS leaves the
cycle's residual reduction where it was -- 0.70147 against 0.70185, a 0.05%
difference, and in the favourable direction -- while narrowing the FIELDS as
well turns a contraction into a 98x AMPLIFICATION at this size. Same storage
type, same kernels, same 3 decimal digits: the difference is only which array
carries them, and whether ||A|| ~ 6/dx^2 multiplies the error before the coarse
grid sees it.

One negative result inside the positive one: fp64 FIELDS with bf16 coefficients
is 23.82 -> 26.54 ms, i.e. 1.11x SLOWER, at a cycle strength identical to five
digits. Narrowing is only worth it once the fields are narrow too. The mechanism
was not isolated (no ncu run); the arithmetic is the visible difference --
`GmgComputeT<double>` is double, so every coefficient there is unpacked to float
and then widened again, where the fp32 path stops at the float bf16 natively
converts to.

---

## An FP32 solver wearing an FP64 LinOp's clothes

*(from `src/NeoN/include/NeoN/blockAmr/linearAlgebra/krylov/mixedPrecision.hpp`)*

An FP32 solver wearing an FP64 LinOp's clothes, so that `gko::solver::Ir<double>`
can drive it.

**WHY.** The profile of the tuned 256^3 solve puts 74.2 ms of 188.9 on the Krylov
side, and every kernel there is at 83-100% of the machine's measured 479 GB/s.
There is no kernel-quality headroom left; the only lever is bytes. Every one of
those vectors is fp64:

| kernel                     | time    | what it does                                  |
| -------------------------- | ------- | --------------------------------------------- |
| cg::step_1 / cg::step_2    | 25.6 ms | p = r + beta p; x += alpha p; r -= alpha q    |
| matvec (FaceCoeffOp)       | 17.7 ms |                                               |
| dot / nrm2 (cuBLAS)        | 15.1 ms |                                               |
| pack/unpack Ginkgo<->AMReX | 9.4 ms  | the V-cycle interface                         |
| linf stopping norm         | 6.5 ms  |                                               |

Halving the element width halves the traffic of all of it. What it cannot do is
halve the ACCURACY the caller asked for: fp32 CG stagnates once the residual
approaches fp32's rounding of A x, around 1e-7 relative, and the tolerances here
go to 1e-10 and below.

Iterative refinement is the standard resolution and the reason this class exists.
The outer loop keeps everything that decides the ANSWER in fp64 --

    r = b - A x        (fp64 operator, fp64 residual, fp64 stopping test)
    x <- x + S(r)      (S approximate, precision irrelevant to the fixed point)

-- and S is where the time goes, so S is what gets narrowed. A wrong S costs
outer iterations; it cannot cost accuracy, exactly as a preconditioner cannot.

**WHAT THIS CLASS IS.** `gko::solver::Ir<double>` calls its inner solver through a
`gko::LinOp` with `Dense<double>` arguments. This is that LinOp: it converts b down
to fp32, runs a preconditioned `Cg<float>` from a zero guess, and converts the
result back up. The two conversions are 3 * n * 4 bytes per apply -- 0.2 ms at
256^3 against the ~9 ms per outer iteration they buy back.

**THE INNER TOLERANCE IS THE WHOLE DESIGN.** Solving the inner system tightly wastes
fp32 iterations on digits the outer loop is about to recompute; solving it too
loosely makes the outer loop a Richardson iteration. The default (1e-2, i.e.
two digits per outer step) comes from the standard analysis -- the outer
contraction factor IS the inner tolerance -- and is a knob because the right
value depends on how much the V-cycle already contracts.

### What it measured: the bytes arrive, the vehicle loses them

A NEGATIVE RESULT, kept wired and tested for the same reason bf16.hpp's is. The
answers are right -- every configuration below converges and agrees with the fp64
CG's solution to ~1e-14 -- and it is slower than the fp64 CG at every setting.

256^3, one box, fp32 hierarchy, varying b, rtol 1e-10 in linf. `applies` counts
PRECONDITIONER applies, which is the unit of work here (the V-cycle is 58% of a
solve): Ginkgo's Cg applies the preconditioner before its stopping check, so a
solve reported as k iterations performed k+1 of them. The inner tolerance is set
unreachable so `mp_inner_max_iter = K` is the exact inner count, which makes the
column exact rather than inferred:

| config     | outer | K  | applies | ms    | ms/apply |
| ---------- | ----- | -- | ------- | ----- | -------- |
| cg fp64    | 9     | -- | 10      | 213.8 | 21.4     |
| mpir K=1   | 13    | 1  | 26      | 535.1 | 20.6     |
| mpir K=2   | 5     | 2  | 15      | 302.4 | 20.2     |
| mpir K=3   | 5     | 3  | 20      | 384.8 | 19.2     |
| mpir K=4   | 5     | 4  | 25      | 468.8 | 18.8     |
| mpir K=6   | 5     | 6  | 35      | 644.2 | 18.4     |
| mpir K=8   | 5     | 8  | 45      | 815.6 | 18.1     |

Read the last column first: the fp32 inner iteration really is cheaper, 18.1 ms
against 21.4, i.e. 1.18x -- essentially the 1.20x the profile predicted for
halving the Krylov width, arriving as predicted. (The trend from 20.6 to 18.1 is
the per-outer-step fp64 residual amortising over more inner iterations.)

Then read the `applies` column: the cheapest refinement schedule needs 15 where CG
needed 10. A restart is not free -- it re-pays the initial residual AND the
pre-check preconditioner apply, and it discards the Krylov space, which is why
K=1 needs 13 outer steps where CG needed 9 iterations. 1.5x the work at 0.85x the
unit cost is 1.27x slower, and measured it is 1.41x.

So refinement cannot cash a 1.18x saving that costs 1.5x more applies. What would
is a Krylov method that runs its recurrence in fp32 WITHOUT restarting, keeping
only the solution update in fp64. That is a different algorithm, not a different
setting, and Ginkgo does not offer it.

Two things ruled out along the way, so they are not re-litigated:

- The over-relaxed smoother (`gmg_omega=1.1`) breaks the V-cycle's
  self-adjointness, which fp32 CG might have tolerated less well than fp64's.
  It is not the cause: at omega=1.0 the fp32 floor is unchanged (0.1239 against
  0.1226) and mpir is worse still (415 ms at K=2 against 302).
- "Just run fp32 CG to the tolerance" is not an option, which is why refinement
  was the right idea: one fp32 solve alone stops at a linf residual of 0.123
  where the fp64 CG reaches 3.3e-10.

One loose end, flagged rather than fixed because no conclusion here rests on it:
that lone fp32 solve stops itself after ~12 iterations, far above fp32's rounding
floor, so the INNER stopping criterion is not trustworthy. It does not affect the
table above, whose inner counts are fixed by iteration cap with the tolerance
switched off -- but it does mean `mp_inner_max_iter`, not `mp_inner_rtol`, is the knob
to drive this path with.

---

## `GmgArgs::coeffPrecision` -- the fields/coefficients split

*(from `src/NeoN/include/NeoN/blockAmr/bench/kokkosBench.hpp`)*

Measured, 256^3 single box, level-0 agglomerated, one cycle from z0 = 0:

| fields/coeffs | ms/cycle | r1/r0 (smooth b) |                             |
| ------------- | -------- | ---------------- | --------------------------- |
| fp64 / fp64   | 23.82    | 0.70078          |                             |
| fp64 / bf16   | 26.54    | 0.70053          | <- 1.11x SLOWER             |
| fp32 / fp32   | 12.52    | 0.70185          |                             |
| fp32 / bf16   | 10.60    | 0.70147          | <- 1.18x faster, same cycle |
| bf16 / bf16   | 9.37     | 97.7             | <- diverges (see bf16.hpp)  |

So: narrow the coefficients only once the FIELDS are narrow. Under fp32
fields it is 1.18x off the cycle at a residual reduction indistinguishable
from fp32's; under fp64 fields the same change costs 11%.

---

## The norm at the fp64/T boundary

*(from `src/NeoN/include/NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp`,
`faceCoeffResidScatterNorm`)*

`faceCoeffResidScatterNorm` forms the residual in DOUBLE and stores it cast to the
hierarchy's `T` straight into the L0 rhs; the light second kernel then reduces the
STORED values, so the norm the solver stops on is the norm of what the V-cycle will
actually see. That is bit-exact FP64 for `T=double`. For `T=float` it carries ~6e-8
relative rounding — far below the ~10x per-cycle residual drop, so the stopping cycle
is unchanged (verified: identical iteration count and identical answer). The host twin
reduces the stored value in the same MFIter/k/j/i order, so reference and cuda return
an identical norm.

Kernel-level precision rule throughout the hierarchy: locals are `GmgComputeT<T>`
(= `T` except `float` for `Bf16`), so a kernel mixing a stored value with a literal
weight spells the weight `GmgComputeT<T>` and a bf16 level never rounds it to 3
digits. `acc`, `off` and `diag` are where this matters — see the residual-cancellation
argument above.

---

## Why fp32 is the default

*(from `src/NeoN/src/blockAmr/python/blockamr/solver_config.py`, `GmgConfig.precision`)*

The V-cycle is bandwidth-bound and only STEERS CG: the outer operator, the residual
and the stopping test stay fp64 whatever the hierarchy is stored in, so a narrower
hierarchy can cost iterations but never correctness. Measured at 128^3: 1.49x faster
for an identical iteration count, with the converged answer moving 2.7e-18 -- below
the fp64 path's own run-to-run noise floor. That is why `GmgConfig.precision`
defaults to `"fp32"` and not to the byte-for-byte historical `"fp64"`.

`"bf16"` is reachable but is a measured negative result rather than a recommended
setting: there is no mesh size at which it wins. It is 1.36x off the V-cycle, but
psi's storage error reaches the coarse grid multiplied by ||A|| ~ 6/dx^2, so the
cycle weakens as n^2 and the CG iteration count more than doubles already at 64^3
(11 -> 25) and reaches 273 vs 12 at 256^3. It also needs `precond="gmg_kokkos"`:
the shipped GMG hierarchy carries fp64/fp32 only, and the solver raises for the
other `precond` values. Full argument and tables under
[bfloat16 as a STORAGE type](#bfloat16-as-a-storage-type-for-the-gmg-level-hierarchy).

`coeff_precision` is the half of that experiment that survives -- see
[`GmgArgs::coeffPrecision`](#gmgargscoeffprecision----the-fieldscoefficients-split).
Rounding a coefficient only perturbs the preconditioner's operator, where rounding
psi is amplified by ||A||. It may not be wider than `precision` and likewise needs
`precond="gmg_kokkos"`.

---

## The default V-cycle shape

`gmg_min_bottom=2`, `gmg_coarsest_sweeps=16`, `gmg_omega=1.1`.

*(from `src/NeoN/src/blockAmr/bindings/ginkgoSolve.cpp`, the `nb::arg` defaults)*

`gmg_min_bottom=2` and `gmg_coarsest_sweeps=16` (with `gmg_omega=1.1`) are a MEASURED
shape, not the historical one -- the previous 4/8/1.0 predated agglomeration, fp32,
shared coefficients and level-0 re-decomposition, four features that each changed what
a level costs. The three compose super-additively, because a deeper ladder is what
makes extra bottom sweeps cheap (a 2^3 bottom is 8 cells) and a better-solved bottom is
what makes the deeper ladder pay. Preconditioned CG on the periodic Helmholtz problem,
fp32 hierarchy, level-0 agglomerated:

| grid  | 4/8/1.0  | 2/16/1.1 | speedup |
| ----- | -------- | -------- | ------- |
| 64^3  | 11 iters | 8 iters  | 1.23x   |
| 128^3 | 11 iters | 8 iters  | 1.28x   |
| 256^3 | 12 iters | 8 iters  | 1.40x   |

The iteration count is now FLAT in N, which is the mesh-independence multigrid is
supposed to have and 4/8/1.0 did not. Confirmed off the constant-coefficient problem at
128^3 (smooth b 12->8, a 1e4 b jump 28->23, 4:1 anisotropic cells 81->59), and every
knob is load-bearing in the drop-one controls on every one of those.

### Why over-relaxing helps a SMOOTHER

A smoother is not trying to solve anything: its whole job is to annihilate the modes the
coarse grid cannot represent, theta in [pi/2, pi]. At omega=1 RB-GS damps that band very
unevenly -- modes near pi are crushed while modes near pi/2, exactly where the coarse
grid is also weakest, are barely touched -- and the cycle's contraction is set by the
WORST mode in the band. Over-relaxing trades surplus damping near pi for scarce damping
near pi/2, which lowers that maximum. MLMG's own `abec_gsrb` over-relaxes for the same
reason, at 1.15.

It is bounded on the other side by symmetry: `omega != 1.0` makes the colour sweep
non-self-adjoint, so the V-cycle is no longer exactly SPD even with the reversed
post-smooth, and CG's theory stops applying. Harmless for `solver="gmg"`/`"ir"`
(stationary iterations); for `precond="gmg"` it is a real cost, and the measured
turnover is where it starts to outweigh the better damping. 256^3, preconditioned CG,
everything else at the defaults above:

| omega                                          | 0.9 | 1.0 | 1.1 | 1.15 | 1.2 | 1.3 |
| ---------------------------------------------- | --- | --- | --- | ---- | --- | --- |
| iters                                          | 15  | 12  | 11  | 11   | 12  | 13  |
| iters (with min_bottom=2, coarsest_sweeps=16)  | -   | 10  | 8   | 9    | 9   | -   |

Hence 1.1 rather than MLMG's 1.15, which costs an iteration at both 128^3 and 256^3. The
gain is largest where the coarse grid is worst: 4:1 anisotropic cells (we coarsen all
three axes, no semicoarsening) go 70 -> 59 iterations, against 12 -> 11 on the isotropic
problem.

### The two symmetry breakers do not compose

`gmg_omega != 1.0` is the V-cycle's SECOND symmetry breaker; `gmg_pre_sweeps !=
gmg_post_sweeps` is the first, and it already warns. Either alone is survivable; both at
once are not (N=32, `precond="gmg"`, 300-iteration budget):

| sweeps | omega=1.0 | omega=1.05 | omega=1.1 | omega=1.15 |
| ------ | --------- | ---------- | --------- | ---------- |
| 2 / 1  | 16 iters  | 21 iters   | diverges  | diverges   |
| 2 / 2  | 8 iters   | 8 iters    | 8 iters   | 8 iters    |

Raising the omega default is safe precisely because the default sweeps are symmetric.
Set `gmg_omega=1.0` whenever they are not. Pinned by
`test_asymmetric_sweeps_and_over_relaxation_stack`.

---

## The V-cycle contraction diagnostic

The `solver="gmg"` `contraction` field and its 0.464 threshold.

*(from `src/NeoN/src/blockAmr/linearAlgebra/solve/persistent.cpp`,
`GmgStationarySolver::solve`)*

A stationary V-cycle contracts the residual by a roughly CONSTANT factor per cycle, so
`SolveResult::contraction` (the geometric mean of that factor) is the one number that
says whether the cycle is working -- and it says it even on a run that converged, which
a pass/fail flag cannot. Without it a caller sees only "did not converge in max_iter"
and cannot tell a V-cycle grinding at 0.97/cycle from one that diverged on cycle two.
Only the native stationary path attaches a `diagnostic`, because only there is the
number a stable property of the method. It is REPORTED, not printed: that path already
returns a dict the caller reads, and a `std::cerr` warning inside a solve is both
unmissable in a sweep and unactionable in a script.

The threshold is a "look here" signal, not a tolerance, but it has to sit BELOW the
cases worth looking at. Measured at N=16 on the constant-coefficient periodic problem,
smoothing bottom: 0.070 at 1 box, 0.155 at 8 boxes, 0.594 at 64 boxes -- and only the
last fails to converge in 30 cycles. With a Krylov bottom all three sit at 0.058-0.070,
which is what "healthy" looks like here. So the degraded case is 4x the healthy rate,
and a threshold anywhere in between separates them; one decade per 3 cycles
(rho = 10^(-1/3) = 0.464) is the round number in that gap, and still allows 3x the
cycles a healthy V-cycle needs before it says anything. Crossing it means something
structural: most often a bottom grid the smoother cannot solve (see
`gmg_bottom_solver`), otherwise anisotropy or a coefficient jump the hierarchy does not
represent.

---

## The constant-coefficient benchmark trap

*(from `src/NeoN/src/blockAmr/bindings/ginkgoSolve.cpp`, `gmg_coeff_precision`)*

On a constant-coefficient Laplacian over a power-of-two grid every coefficient
(1/dx^2, alpha=1, and the 1/4 and 1/8 restriction weights) is exactly representable, so
a bf16 hierarchy is BIT-IDENTICAL to an fp32 one and reports the full bandwidth saving
for free. Every `coeff_precision` row in this document uses a VARYING b for exactly that
reason.
