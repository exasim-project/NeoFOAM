# Report — NeoN bindings needed to run the turbulence closures **on-device (GPU)**

**Branch:** `feat/turbNeoN`  ·  **Date:** 2026-07-11

> **Update (2026-07-11): SpalartAllmaras now runs fully on-device.** The SA-required
> operators were added to `src/NeoN/src/bindings/volumeField.cpp` as elementwise
> Kokkos kernels: `ScalarVolumeField.__pow__` (field ** scalar), `__rsub__`
> (scalar − field), `__rtruediv__` (scalar / field), `__neg__`, `__sub__(field,
> scalar)`, and module functions `field_max(field, field)`, `field_min(field,
> scalar)`, `field_min(field, field)`. Unit-tested in
> `src/NeoN/test/bindings/test_volumeField.py::test_scalar_field_elementwise_operators`
> (runs on every executor the fixture yields — serial / cpu / **gpu**).
> `neon_spalartAllmaras.py` was rewritten to chain these field ops instead of the
> host NumPy round-trip — **no `copy_to_host`/`copy_from_host` in the step** — and
> still matches pybFoam to `8.7e-19` (bit-identical to the host version).
>
> **Update 2: kOmegaSST is also implemented and on-device.** Items 7–12 were added:
> `nn.tanh`/`nn.sqrt` (NeoN `volumeField.cpp`), `nfb.strain_magnitude_sqr` and
> `nfb.grad_dot_grad` (NeoFOAM `pimple.cpp`), and — the one genuinely new implicit
> operator — **`nn.imp.susp`** (OpenFOAM `fvm::SuSp`), added to the NeoN DSL by
> extending `SourceTerm` with a sign-split mode (`diag += V·max(c,0)`,
> `rhs -= V·min(c,0)·φ`), wired through `dsl::imp::susp`. `neon_kOmegaSST.py`
> (F1/F2 blending + the cross-diffusion `SuSp` term) matches pybFoam to
> `nut 3.5e-18 / k 3.3e-16 / omega 2.3e-13`. **All items 1–12 are done.**
> Unit tests: `test_scalar_field_elementwise_operators` (Python, tanh/sqrt/pow on
> serial/cpu/gpu) + a SuSp C++ Catch2 test in
> `src/NeoN/test/finiteVolume/cellCentred/operator/sourceTerm.cpp` (both coefficient
> signs); the kOmegaSST parity is the end-to-end SuSp validation vs `fvm::SuSp`.

## The problem

`neon_spalartAllmaras.py` currently authors the nonlinear Spalart-Allmaras scalar
maths (`chi`, `fv1`, `fv2`, `Stilda`, `r`, `g`, `fw`) as **host NumPy**: every step it
`copy_to_host`s the fields, computes on the CPU, and `copy_from_host`s the result. On
a **Serial/CPU** executor this is correct and fast. On a **GPU** executor it is a
correctness-preserving but performance-destroying pattern — each `correct` does a
device→host→device round-trip **and runs the actual arithmetic on the CPU**, so the
GPU sits idle for the closure.

To run on-device, every expression must be a **NeoN field operation** (a Kokkos
`parallelFor`, like the existing `strain_production` / `vorticity_magnitude` /
`mag_sqr_grad` bindings, which already execute on whatever executor the field lives
on). This report tabulates exactly which element-wise field operations are missing.

## What `ScalarVolumeField` supports today

| Operation | Status |
|---|---|
| `field + field`, `field + scalar`, `scalar + field` | ✅ `__add__` / `__radd__` |
| `field - field`, `field - scalar` | ✅ `__sub__` |
| `scalar - field` | ❌ `__rsub__` missing |
| `field * field`, `field * scalar`, `scalar * field` | ✅ `__mul__` / `__rmul__` |
| `field / field`, `field / scalar` | ✅ `__truediv__` |
| `scalar / field` | ❌ `__rtruediv__` missing |
| `field ** scalar` (power) | ❌ `__pow__` missing |
| `-field` (negate) | ❌ `__neg__` missing |
| `max(field, scalar)` | ✅ `nn.field_max(field, low)` |
| `min(field, scalar)` | ❌ missing |
| `max(field, field)`, `min(field, field)` | ❌ missing |
| element-wise `sqrt/pow/tanh/exp/…` on a field | ❌ none (`nn.exp`/`nn.mag`/`nn.dot` act on a single `Vec3`/scalar, **not** a field) |

Already on-device (no change needed): `nn.imp.ddt/div/laplacian/source`,
`nn.exp.source`, `SurfaceInterpolationScalar.interpolate`, `field_max(field,scalar)`,
and the SA bindings `read_wall_distance`, `vorticity_magnitude`, `mag_sqr_grad`
(`copy_from_host`/`copy_to_host` are the only host hops and would leave the hot loop).

---

## SpalartAllmaras — every expression → required binding

Going through `correct_nutilda` / `correct_nut` / `seed_nut` line by line. "have"
means expressible with today's operators; "**MISSING**" needs a new binding.

| SA expression | element-wise ops | on-device status |
|---|---|---|
| `chi = nuTilda/nu` | field / scalar | ✅ have |
| `fv1 = chi^3/(chi^3 + Cv1^3)` | `chi^3` = `chi*chi*chi`; field/(field+scalar) | ✅ have (pow³ via mult) |
| `fv2 = 1 - chi/(1 + chi*fv1)` | field*field, scalar+field, field/field, **`scalar - field`** | ⚠ needs `__rsub__` (or rewrite `1 + (-1)*x`) |
| `kd2 = (kappa*y)^2` | `(kappa*y)*(kappa*y)` | ✅ have (sqr via mult) |
| `Stilda = max(Omega + fv2*nuTilda/kd2, Cs*Omega)` | field+field, field*field/field, **`max(field, field)`** | ❌ **MISSING `max(field,field)`** |
| `r = min(nuTilda/(max(Stilda,SMALL)*kd2), 10)` | `max(field,scalar)`✅, field/field, **`min(field, scalar)`** | ❌ **MISSING `min(field,scalar)`** |
| `g = r + Cw2*(r^6 - r)` | `r^6` = `(r*r)*(r*r)*(r*r)`; field−field; scalar*field | ✅ have (pow⁶ via mult) |
| `fw = g*((1+Cw3^6)/(g^6+Cw3^6))^(1/6)` | `g^6` via mult; field+scalar; **`scalar/field`**; **`(…)^(1/6)`** | ❌ **MISSING `pow(field, scalar)`** (fractional) + `__rtruediv__` |
| `DnuTildaEff = (nuTilda+nu)/sigma`, `interpolate` | field+field, /scalar, surface interp | ✅ have |
| explicit src `Cb2/σ*magSqrGrad + Cb1*Stilda*nuTilda` | scalar*field + scalar*field*field | ✅ have (magSqrGrad/Stilda are fields) |
| sink coeff `Cw1*fw*nuTilda/y^2` | scalar*field*field / (`y*y`) | ✅ have |
| `nut = nuTilda*fv1` | field*field | ✅ have |
| assemble `imp.ddt+div-laplacian - exp.source + imp.source`, solve, `field_max(·,0)` | NeoN operators | ✅ have |

### Minimal binding set to make SpalartAllmaras fully on-device — **3 bindings**

1. **`pow(field, scalar) -> field`** (`__pow__`) — the only *fractional* power (`^(1/6)`
   in `fw`); also subsumes `chi^3` / `r^6` / `y^2` if you prefer `x**n` over `x*x*…`.
   **Critical** — no way to express a fractional power with the existing ops.
2. **`field_max(field, field)`** — a second overload of the existing `field_max`
   (element-wise max of two fields) for `Stilda`.
3. **`field_min(field, scalar)`** (and ideally `field_min(field, field)`) — the `r`
   clamp to 10; the field–field form is needed by kOmegaSST's production limiter.

Plus two *optional* operator conveniences that remove awkward rewrites (both are
otherwise expressible — `1 - x` as `1.0 + (-1.0)*x`, `s/x` as `s*pow(x,-1)`):
`__rsub__` (`scalar - field`) and `__rtruediv__` (`scalar / field`); `__neg__` is nice
for readability. Each is a one-line binding.

With (1)–(3) the closure drops `copy_to_host`/`copy_from_host` entirely — `chi`,
`fv1`, `fv2`, `Stilda`, `r`, `g`, `fw`, `nut` all become chained field ops evaluated
on the executor the fields live on (CPU or GPU). `read_wall_distance`,
`vorticity_magnitude`, `mag_sqr_grad` already run on-device.

---

## kOmegaSST — additional on-device bindings (for completeness)

The remaining model needs everything above **plus** (see also
`neon-turbulence-komegasst-spalartallmaras-gaps.md`):

| kOmegaSST expression | extra op needed |
|---|---|
| `F1 = tanh(arg^4)`, `F2 = tanh(arg^2)` | **`tanh(field)`** |
| `sqrt(k)` (F1/F2), `sqrt(S2)` (nut) | **`sqrt(field)`** (or `pow(field, 0.5)`) |
| `CDkOmega = 2*αω2*(grad(k) & grad(omega))/omega` | **`dot(fieldVecA, fieldVecB) -> fieldScalar`** (today `nn.dot` is `Vec3`-only) + a bound scalar `grad()` |
| `S2 = 2*magSqr(symm(gradU))` | **`strain_magnitude_sqr(gradU) -> field`** (kernel like `vorticity_magnitude`) |
| `Pk = min(G, c1*βstar*k*omega)`, `nut = a1*k/max(a1*omega, b1*F23*sqrt(S2))` | `min/max(field, field)` (from SA set) |
| cross-diffusion `SuSp((F1-1)*CDkOmega/omega, omega)` | **`imp.susp(coeff, field)`** — an *implicit operator*, unavoidable even for div-free flow |
| `ft2 = Ct3*exp(-Ct4*chi^2)` (if enabled) / any `exp()` closure | **`exp(field)`** (element-wise; distinct from the `nn.exp` operator namespace) |

So kOmegaSST adds ≈ `tanh`, `sqrt`, `exp` element-wise ufuncs, a field·field `dot`, a
`strain_magnitude_sqr` kernel, and the `SuSp` **operator** (the one genuinely new
implicit primitive).

---

## Recommendation

Implement the element-wise field ops as a small family of Kokkos-`parallelFor`
bindings (mirroring `strain_production`): a generic **`pow(field, scalar)`**, the
**field–field `max`/`min`** (and `min(field, scalar)`), then **`sqrt`/`tanh`/`exp`**
and the field·field **`dot`** when kOmegaSST lands. Add `__pow__`/`__rsub__`/
`__rtruediv__`/`__neg__` dunders on `ScalarVolumeField` so the closures read like the
maths. This keeps the whole turbulence `correct()` on-device — the ModelSpec closures
stay readable Python, but every expression compiles to a NeoN field kernel instead of
a host round-trip. Estimated: **3 bindings** unblock SpalartAllmaras on GPU; ≈ **6
more** (5 ufuncs/dot + `SuSp`) unblock kOmegaSST.
