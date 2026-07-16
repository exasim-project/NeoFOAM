# Report — What is missing to add kOmegaSST and SpalartAllmaras as pure-Python NeoN ModelSpec closures

**Branch:** `feat/turbNeoN`  ·  **Date:** 2026-07-11

> **Update (2026-07-11): SpalartAllmaras is now implemented and passes at machine
> precision** (`nut`/`nuTilda` diff `8.7e-19`, see the status report). Gaps A–C
> below were closed with four bindings — `copy_from_host`, `read_wall_distance`,
> `vorticity_magnitude`, `mag_sqr_grad` (in `src/bindings/pimple.cpp`) — and the
> model `neon_spalartAllmaras.py` writes all nonlinear SA maths as host NumPy. The
> `copy_from_host` route (B1) was taken. **kOmegaSST remains open** and additionally
> needs gap D (`SuSp`) plus `tanh` / field·field-dot / `strain_magnitude_sqr` — all
> now expressible via the same host-NumPy pattern except `SuSp`, which needs the one
> operator binding (or the hand-split D2).

## Context

`kEpsilon` and `laminar` now exist as pure-Python NeoN `ModelSpec` closures and match
pybFoam to machine precision (`nut` 3.9e-18, `k` 2.2e-16, `epsilon` 7.1e-15 after one
`correct` step — see `neon-turbulence-modelspec-status.md`). The pattern is:

```python
@spec.build       # read/seed the transport fields + helper operators
@spec.operation   # advance each transport PDE with nn.imp/exp + nfb.strain_production
```

`kEpsilon` worked because its `correct()` only needs primitives NeoN already exposes:
`imp.ddt/div/laplacian/source`, `exp.source`, scalar-field arithmetic (`Cmu*k*k/eps`),
`nfb.strain_production` (the `dev(twoSymm(gradU)) && gradU` production density),
`nn.field_max` (bound), and — crucially — **no wall distance, no blending functions,
no nonlinear element-wise field math**.

`kOmegaSST` and `SpalartAllmaras` need all three of those. This report enumerates
exactly which primitives are missing, mapped to the OpenFOAM source terms that use
them, and proposes the smallest binding set that unblocks both.

References: OpenFOAM-v2406
`src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/…`,
`…/Base/SpalartAllmaras/SpalartAllmarasBase.C`,
`…/lnInclude/kOmegaSSTBase.C`; existing C++ LES model
`src/turbulenceModels/spalartAllmarasDDES.cpp`.

---

## What NeoN exposes today (the toolbox `kEpsilon` used)

| Category | Available in Python |
|---|---|
| Implicit operators (`nn.imp`) | `ddt`, `div`, `laplacian`, `source` (Sp) — **no `SuSp`** |
| Explicit operators (`nn.exp`) | `ddt`, `div`, `grad`, `laplacian`, `source` (Su) |
| Scalar-field arithmetic | `+ - * /` with fields **and** scalars (incl. reflected `__rmul__/__radd__`) — **no `**`/`pow`** |
| Gradient | `nfb.GaussGreenGrad(rt).grad_tensor(U)` → tensor field (**scalar `grad()` not bound**) |
| Turbulence-specific | `nfb.strain_production(gradU)` = `dev(twoSymm(gradU)) && gradU` |
| Bounding | `nn.field_max(field, low)` (lower-bound against a **scalar**) |
| Surface interp | `nn.SurfaceInterpolationScalar(...).interpolate(volField)` |
| Host readback | `field.internal_vector().copy_to_host()` (**read only — no host→device write**) |
| Solve | `nfb.PDESolverScalar(expr, psi, rt)` + `rotate_old_times` + `set_final_iter` (see gotchas in the status report) |

`nn.mag`, `nn.dot`, `nn.exp` operate on a single `Vec3`/scalar — **not** on fields.
`TensorVolumeField` is opaque (no methods bound); there is **no** `skew/symm/dev/magSqr`
on fields, no field–field `min/max`, no `tanh/sqrt/pow/exp` on fields, and no wall
distance anywhere.

---

## What the two models compute

### SpalartAllmaras — one transport equation for `nuTilda` (`SpalartAllmarasBase.C`)

```
chi   = nuTilda/nu
fv1   = chi^3/(chi^3 + Cv1^3)
fv2   = 1 - chi/(1 + chi*fv1)
ft2   = Ct3*exp(-Ct4*chi^2)
Omega = sqrt(2)*mag(skew(gradU))                       # vorticity magnitude
Stilda= max(Omega + fv2*nuTilda/(kappa*d)^2, Cs*Omega)
r     = min(nuTilda/(max(Stilda,eps)*(kappa*d)^2), 10)
g     = r + Cw2*(r^6 - r)
fw    = g*((1+Cw3^6)/(g^6 + Cw3^6))^(1/6)
nuTildaEqn:
  ddt + div - laplacian((nuTilda+nu)/sigma)
  - (Cb2/sigma)*magSqr(grad(nuTilda))                  # explicit source
  == Cb1*Stilda*nuTilda*(1-ft2)
     - Sp((Cw1*fw - Cb1/kappa^2*ft2)*nuTilda/d^2, nuTilda)
nut = nuTilda*fv1 ; bound(nuTilda, 0)
```

Uses **wall distance `d`**, **`mag(skew(gradU))`** (vorticity), **`magSqr(grad(nuTilda))`**,
and the nonlinear scalars `pow3/pow6/pow(·,1/6)/sqr/sqrt/exp/min/max`. **No `SuSp`, no
blending, no cross-diffusion.** → the *simpler* of the two.

### kOmegaSST — two equations (`k`, `omega`) with blending (`kOmegaSSTBase.C`)

```
S2   = 2*magSqr(symm(gradU))
F1 = tanh( min( max( sqrt(k)/(betaStar*omega*y), 500*nu/(y^2*omega) ),
                4*alphaOmega2*k/(CDkOmegaPlus*y^2) )^4 )
F2 = tanh( max( 2*sqrt(k)/(betaStar*omega*y), 500*nu/(y^2*omega) )^2 )
CDkOmega = 2*alphaOmega2*(grad(k) & grad(omega))/omega    # field·field dot
Pk = min(G, c1*betaStar*k*omega)                          # production limiter
omegaEqn:
  ddt + div - laplacian(DomegaEff(F1))
  == gamma*GbyNu
     - SuSp((2/3)*gamma*divU, omega)                      # dilatation (0 if div-free)
     - Sp(beta*omega, omega)
     - SuSp((F1-1)*CDkOmega/omega, omega)                 # cross-diffusion — ALWAYS on
kEqn: ddt + div - laplacian(DkEff(F1)) == Pk - SuSp((2/3)divU,k) - Sp(betaStar*omega,k)
nut = a1*k/max(a1*omega, b1*F23*sqrt(S2)) ; bound(omega, omegaMin), bound(k, kMin)
```

Uses **wall distance `y`**, **`tanh`**, **field·field dot** `grad(k)&grad(omega)`,
**`magSqr(symm(gradU))`**, field–field **`min/max`**, `sqrt/sqr/pow4`, **and `SuSp`**.
The `SuSp((F1-1)*CDkOmega/omega)` cross-diffusion term is **not** a dilatation term —
it is present even for a divergence-free flow, so (unlike `kEpsilon`) it **cannot** be
made to vanish with a div-free `phi`. → the *harder* of the two.

---

## Gap analysis — what must be added

### A. Wall distance (both models)  ⛔ blocking

The model equations use distance-to-wall `d`/`y` **everywhere** (`fw`, `Stilda`, `F1`,
`F2`), not just at wall BCs. NeoN has no wall-distance field. The existing C++ SA-DDES
model gets it from OpenFOAM — `Foam::wallDist y(mesh)` / `Foam::nearWallDist` — and
copies the values into a NeoN field in its constructor (`spalartAllmarasDDES.cpp:240-249`).
That path is **not exposed to Python**.

**Needed:** a binding `nfb.read_wall_distance(rt) -> ScalarVolumeField` (wrap
`Foam::wallDist(mesh).y()` and copy into a registered NeoN field — exactly what the C++
model already does internally). Implies the parity/production case must contain at least
one `wall`-type patch so `y` is defined (wall *functions* are still avoidable — a
`fixedValue`/`calculated` nut BC is fine).

### B. Element-wise nonlinear scalar-field math (both models)  ⛔ blocking

`chi^3`, `sqrt(k)`, `tanh(arg^4)`, `exp(-Ct4 chi^2)`, `pow(...,1/6)`, `min(a,b)`,
`max(a,b)` — none exist on `ScalarVolumeField` (only `+ - * /` and `field_max(field,
scalar)`). The C++ SA-DDES computes all of this in hand-written Kokkos kernels
(`kernelComputeProdSp`, using `Kokkos::sqrt/pow/tanh/min/max`) precisely because the DSL
lacks them.

Two ways to close this — pick one:

- **B1 (recommended, most Pythonic, fewest bindings): one `copy_from_host` binding.**
  Add `field.internal_vector().copy_from_host(np.ndarray)` (or
  `nfb.assign_from_host(field, array)`). Then every nonlinear term is **plain, readable
  NumPy** on the host — `chi = nuTilda_h/nu_h; fv1 = chi**3/(chi**3+Cv1**3)` — written
  back into the field, matching the "closure = readable Python maths" goal exactly. Cost:
  a host↔device round-trip per term (fine on CPU; a real cost on GPU).
  *Currently only `copy_to_host` (read) exists; this is the single most unblocking add.*

- **B2 (performant, more C++): bind element-wise field ufuncs in NeoN.** Expose
  `nn.sqrt/pow/tanh/exp/sqr` and field–field `min/max` as `ScalarVolumeField` ops (thin
  Kokkos `parallelFor` wrappers). Stays on-device; more binding surface, one per function.

### C. Field-level differential/tensor algebra (both models)  ⛔ blocking

Evaluated (not operator-form) quantities the equations feed into B:

| Quantity | Used by | Status |
|---|---|---|
| `grad(scalar)` → vector **field** | `magSqr(grad(nuTilda))` (SA), `grad(k)&grad(omega)` (SST) | `GaussGreenGrad` only binds `grad_tensor`; scalar `grad()` **not bound** |
| `magSqr(vector field)` | `magSqr(grad(nuTilda))` (SA) | not bound |
| field · field (`grad(k) & grad(omega)`) | `CDkOmega` (SST) | `nn.dot` is `Vec3`-only, **not** a field op |
| `mag(skew(gradU))` (vorticity) | `Omega` (SA) | not bound (`strain_production` gives a *different* invariant) |
| `magSqr(symm(gradU))` | `S2` (SST) | not bound |

**Needed:** bind `GaussGreenGrad.grad(scalar)`; add `nfb.mag_sqr(vectorField)`,
`nfb.dot(vecFieldA, vecFieldB)`, and vorticity/strain invariants
`nfb.vorticity_magnitude(gradU)` (`sqrt(2)·mag(skew)`) and
`nfb.strain_magnitude_sqr(gradU)` (`2·magSqr(symm)`). These mirror `strain_production`
and are small Kokkos kernels. (Alternatively, do them via B1 host round-trip once
`grad(scalar)` and the tensor components are readable on the host — but the tensor field
is opaque today, so at least `grad(scalar)` + the two invariants need bindings regardless.)

### D. `SuSp` implicit operator (kOmegaSST only)  ⛔ blocking for SST, N/A for SA

`nn.imp` has `source` (Sp) but no `SuSp`. kOmegaSST needs it for the **cross-diffusion**
term `SuSp((F1-1)*CDkOmega/omega, omega)`, which is non-zero even for divergence-free
flow (so the div-free-`phi` trick that neutralised `kEpsilon`'s dilatation terms does not
help here). SA has **no** `SuSp`.

**Needed (either):**
- **D1:** bind `nn.imp.susp(coeff, field)` (Kokkos: `diag += V*max(coeff,0)`,
  `source -= V*min(coeff,0)*field` — see `fvmSup.C`), or
- **D2:** hand-split in Python as `imp.source(max(coeff,0), field) +
  exp.source(min(coeff,0)*field)` — needs field–field `max/min` from B, and careful sign
  matching to reach 1e-10 (the reason `kEpsilon` avoided it).

---

## Per-model checklist

### SpalartAllmaras (RAS) — needs A, B, C  (no D)
- [ ] A. `read_wall_distance` binding
- [ ] B. nonlinear scalar math (`pow3/pow6/pow/sqr/sqrt/exp`, field `min/max`) — via B1 `copy_from_host` **or** B2 ufuncs
- [ ] C. `grad(scalar)` field, `magSqr(vector)`, `vorticity_magnitude(gradU)`
- [ ] Reuse: `imp.ddt/div/laplacian`, `imp.source` (Sp), `exp.source` (the `magSqr(grad)`
      term and the `Cb1*Stilda*nuTilda*(1-ft2)` production), `field_max` (bound ≥ 0),
      the solve gotchas already proven for `kEpsilon`.

### kOmegaSST — needs A, B, C, **and D**
- [ ] A. `read_wall_distance` binding
- [ ] B. nonlinear scalar math **plus `tanh`** (B1 covers it for free; B2 adds `tanh`)
- [ ] C. `grad(scalar)` (twice, for `CDkOmega`), field·field `dot`, `strain_magnitude_sqr(gradU)`
- [ ] D. `SuSp` (cross-diffusion term is unavoidable) — D1 binding preferred
- [ ] Blending bookkeeping: `F1`, `F2`, `F23`, `gamma`/`beta`/`alpha*` blended by `F1`,
      production limiter `min(G, c1*betaStar*k*omega)`, `nut = a1*k/max(a1*omega,
      b1*F23*sqrt(S2))` — all element-wise once B/C exist.

---

## Recommended path & effort

1. **`copy_from_host` (B1)** — one binding, unlocks *all* element-wise nonlinear math as
   readable NumPy for both models. Highest leverage; also the most faithful to the
   "closures are readable Python" goal. (GPU users pay a round-trip; acceptable for a
   first correctness-focused port, optimise later with B2.)
2. **`read_wall_distance` (A)** — one binding, wraps the `Foam::wallDist` copy the C++
   SA-DDES model already does; unblocks both.
3. **`grad(scalar)` + `magSqr(vector)` + vorticity/strain invariants (C)** — a handful of
   small kernels mirroring `strain_production`.
4. **SpalartAllmaras first** — with 1–3 it needs no new operator (`Sp`+`exp.source` only).
   It is the natural next closure and validates A/B/C end-to-end against pybFoam on a
   walled no-wall-function box, reusing the subprocess parity harness (`_parity_worker.py`)
   by adding one `parity_models/SpalartAllmaras/turbulenceProperties`.
5. **`SuSp` (D)** then **kOmegaSST** — once SA is green, add the one operator kOmegaSST
   needs and port the blending. Expect the cross-diffusion `SuSp` sign-matching to be the
   delicate part for 1e-10 parity.

**Rough sizing:** SA ≈ 3 small bindings (`copy_from_host`, `read_wall_distance`,
`grad_scalar`+invariants) + one Python model file + one parity case. kOmegaSST adds the
`SuSp` binding + a larger (blending-heavy) Python model. No per-model C++ turbulence class
and no nanobind rebuild *per closure* — the whole point of the ModelSpec approach — once
the shared primitives above exist.

---

## Note on the parity case

`kEpsilon` used a **no-wall** box so the dilatation `SuSp(divU)` terms vanished. Both new
models need a defined wall distance, so their parity case must include ≥1 `wall`-type
patch (a channel/box with one wall). Wall *functions* remain avoidable (`fixedValue`/
`calculated` BCs on `nut`/`k`/`omega`/`nuTilda`), keeping the near-wall-override problem
out of scope; and a divergence-free seeded `U` still removes the `SuSp(divU)` dilatation
terms for kOmegaSST, leaving only the (unavoidable) cross-diffusion `SuSp` to match.
