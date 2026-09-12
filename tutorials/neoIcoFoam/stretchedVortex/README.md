# stretchedVortex — a strained (Burgers) vortex, for illustration

A small 3D `neoIcoFoam` case that shows the *one* piece of physics behind the
finite-time-blowup constructions that a CFD code can actually resolve: a vortex
core being squeezed by axial strain, spinning up as it contracts, until viscous
diffusion balances the squeezing.

**This case does not simulate blowup, and it is not evidence for or against any
blowup claim.** See "What this is not" below — that section is the point of the
case as much as the setup is.

## Setup

Domain `[-1,1] x [-1,1] x [-0.75,0.75]`, uniform `64 x 64 x 48` (197k cells).

The initial and lateral-boundary fields are the analytic Burgers vortex

```
u_r     = -a r / 2
u_z     =  a z
u_theta = (Gamma / 2 pi r) (1 - exp(-r^2 / r0^2))
```

with `a = 1`, `Gamma = 1`, `nu = 0.01`. The axial strain `a` pulls fluid in
radially and expels it along the axis; angular momentum `r u_theta` is carried
inward, so the swirl intensifies. Viscosity fights back, and the two balance at
the equilibrium core radius

```
delta = sqrt(4 nu / a) = 0.2
```

The initial field is deliberately started **off** equilibrium, with a core
`r0 = 0.6`, three times too wide. The interesting part of the run is the
contraction back onto `delta`.

That contraction has a closed form. With Gaussian vorticity the core radius
obeys `d(delta^2)/dt = 4 nu - a delta^2`, so

```
delta(t)^2 = 4 nu / a + (delta_0^2 - 4 nu / a) exp(-a t)
```

and the case has a known answer at **every** write time, not just at the end:

| quantity | t = 0 | t = 2 | t = 6 | steady state |
| --- | --- | --- | --- | --- |
| radius of peak swirl `1.121 delta(t)` | 0.673 | 0.324 | 0.226 | 0.224 |
| peak `u_theta` `0.638 Gamma / (2 pi delta(t))` | 0.169 | 0.352 | 0.503 | 0.508 |

`coreHistory.py` measures both and prints the error against this solution.
`delta = 0.2` is about 6.4 cells, which is thin but adequate; double the
`blockMeshDict` counts for a resolution check.

Boundary conditions: velocity is `fixedValue` from the analytic field on **every**
patch, including `top`/`bottom`. The exact solution has `du_z/dz = a` there, so
`zeroGradient` would be inconsistent with it at any resolution -- and prescribing
a value that is already known costs nothing. Pressure is therefore Neumann
everywhere and its level is pinned by `pRefCell`/`pRefValue` in
`system/fvSolution`.

Two consequences worth knowing before changing anything:

- `neoIcoFoam` does not call `adjustPhi` yet, so nothing rescales the outflow to
  match the inflow: the prescribed boundary flux has to balance by itself. It
  does, exactly (the strain part is constant per face, the swirl part cancels by
  symmetry) -- measured net flux 0 against a throughput of 6.0. Any change to the
  expressions or to the mesh symmetry has to preserve that; watch the continuity
  error the solver prints.
- Every patch is a nonuniform `fixedValue`, which needs a NeoFOAM new enough to
  read such patches. Older builds silently treated them as `empty`.

The half-height is 0.75 rather than 0.5 so that the boundary values -- which use
the equilibrium core and are constant in time, while the real solution has
`delta(t)` -- cannot influence the core: viscous information travels
`sqrt(nu T) = 0.245` over the run, and the axial velocity advects away from the
mid-plane everywhere.

## Layout

```
0.orig/{U,p}                      placeholder fields + BC types
constant/transportProperties      nu = 0.01
system/blockMeshDict              uniform box, 64 x 64 x 48
system/setExprFieldsDict          analytic initial condition
system/setExprBoundaryFieldsDict  analytic values on every patch
system/{controlDict,fvSchemes,fvSolution,decomposeParDict}
coreHistory.py                    core radius + peak swirl vs. exact delta(t)
doc/                              result figures and the animation
```

## Running

```bash
./Allrun                                              # serial
NEOFOAM_BIN=../../../build/develop/bin/neoIcoFoam ./Allrun
NPROCS=4 ./Allrun -par                                # parallel (set executor CPU first)
python3 coreHistory.py                                # -> coreHistory.csv (+ .png)
```

As with the other tutorials, `system/controlDict` needs the NeoFOAM-specific
`executor`, `allocator` and `memPoolSize` keys; omitting `allocator` fails
immediately with `Entry 'allocator' not found`. For a parallel run on a
single-GPU box, set `executor CPU` first:

```bash
foamDictionary -entry executor -set CPU system/controlDict
```

`endTime 6` covers six strain times `1/a` and 1.5 viscous times
`delta^2/nu = 4 s`, which is enough to settle. `deltaT 5e-3` gives `Co = 0.28`
at the top and bottom of the domain, where `|u_z| = a H/2` is largest. The run
takes about 1360 s on 4 ranks -- the taller domain is 197k cells against the
131k of a half-height 0.5 box.

## What this is not

Worth stating explicitly, because the resemblance to the recent forced-blowup
constructions is superficial and it would be easy to oversell:

- **It reaches a steady state, not a singularity.** The Burgers vortex is the
  balance point. Nothing here grows without bound; that is exactly why it is
  simulable.
- **The blowup constructions are numerically unreachable.** In the OpenAI
  Navier-Stokes paper the oscillatory pulses sit at wavelength `q^(1/2+h/2)`
  against a core radius `q^(1/2)`, with `h < 1/100`. Getting even one decade of
  scale separation needs `q ~ 1e-200`; the swirl Reynolds number `q^(-h)`
  reaches 10 at `q ~ 1e-100`. No mesh will ever see that regime.
- **The essential ingredients are absent.** There is no oscillatory pulse
  family, no annular stress cancellation, and no smooth compactly supported
  force. Here the flow is sustained by inflow boundary conditions; there the
  force is *defined* as the momentum residual and the entire difficulty is
  arranging it to stay smooth through the singular time.
- **Anisotropic contraction is not represented.** The construction has
  `l_r ~ tau^(1/2)` and `l_z ~ tau^(1/2-h)` with `l_r/l_z -> 0`. The Burgers
  vortex contracts radially only, in a domain of fixed height.

What it does share with Section 2.1 of that paper is the qualitative core
mechanism: inward spiral, axial outflow away from a dividing layer, angular
momentum transported to smaller radii, swirl amplified as the core contracts,
viscosity opposing it. That is worth being able to look at.

## Results

Run to `endTime 6` on 4 ranks (`NPROCS=4 ./Allrun -par`), 1360 s wall clock.

![mid-plane swirl and approach to equilibrium](doc/vortexOverview.png)

The core contracts from 0.675 to 0.225 and the peak swirl grows from 0.169 to
0.502, and by `t = 6` the mid-plane profile lies on the analytic equilibrium
curve. The measured history follows the exact `delta(t)` throughout, not just at
the end:

![measured against the exact unsteady solution](doc/exactComparison.png)

| | peak `u_theta` error | centreline `u_z` vs `a z` |
| --- | --- | --- |
| worst over the whole run | 0.37 % | 0.05 % |
| mean over the whole run | 0.14 % | < 0.01 % |

Global continuity error runs at ~4e-20, twelve orders below what an outflow that
needs rescaling produces.

![meridional structure](doc/vortexStructure.png)

The meridional plane shows the expected topology: radial inflow at the sides, a
dividing layer at `z = 0`, axial outflow towards top and bottom, with `u_z = a z`
held all the way onto the axis.

## Animation

![vortex animation](doc/stretchedVortex.gif)

The translucent surface is the locus of peak `u_theta` along every radial ray:
its shape is the contracting core, its colour the peak swirl on that surface
(fixed 0.15-0.50 scale, so the intensification reads as brightening). The dark
lines are RK4 traces of the instantaneous 3D velocity field, seeded on two rings
hugging the mid-plane; they spiral inward, wind around the core and leave along
the axis. The winding tightens from about 0.7 turns per trace at `t = 1` to 2.2
at `t = 6` -- angular momentum transported to smaller radii, which is the point
of the case. The floor carries mid-plane `u_theta` contours and the camera
rotates 40 degrees over the run.

`doc/stretchedVortex.mp4` is the same animation at full resolution.

## Caveat

The remaining approximation is that the prescribed boundary values use the
equilibrium core and are constant in time, while the true solution has
`delta(t)`. The domain half-height keeps that mismatch away from the core (see
"Setup"), and what survives is the sub-percent error tabulated above. Removing it
entirely needs a time-dependent Dirichlet condition, which `readers.hpp` does not
currently map.

The square cross-section is not axisymmetric -- corners sit at `r = 1.4`,
mid-edges at `r = 1.0` -- which leaves a 4 % azimuthal spread in `u_theta` on the
ring `r = 0.95`. It does not grow with time and the core is unaffected.
