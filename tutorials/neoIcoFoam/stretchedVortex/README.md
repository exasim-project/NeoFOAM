# stretchedVortex — a strained (Burgers) vortex, for illustration

A small 3D `neoIcoFoam` case that shows the *one* piece of physics behind the
finite-time-blowup constructions that a CFD code can actually resolve: a vortex
core being squeezed by axial strain, spinning up as it contracts, until viscous
diffusion balances the squeezing.

**This case does not simulate blowup, and it is not evidence for or against any
blowup claim.** See "What this is not" below — that section is the point of the
case as much as the setup is.

## Setup

Domain `[-1,1] x [-1,1] x [-0.5,0.5]`, uniform `64 x 64 x 32` (131k cells).

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

Predicted, and what `coreHistory.py` checks:

| quantity | t = 0 (r0 = 0.6) | steady state (delta = 0.2) |
| --- | --- | --- |
| radius of peak swirl | 0.673 | 0.224 |
| peak `u_theta` | 0.169 | 0.508 |

So a 3x core contraction and a 3x swirl amplification. `delta = 0.2` is about
6.4 cells, which is thin but adequate; double the `blockMeshDict` counts for a
resolution check.

Boundary conditions: `sides` is `fixedValue` (inflow, set from the analytic
field), `top`/`bottom` are `zeroGradient` for `U` and `fixedValue 0` for `p`.
The axial velocity on those patches is `+/- a H/2`, so both are outflow for the
whole run and no `pRefCell` is needed.

## Layout

```
0.orig/{U,p}                      placeholder fields + BC types
constant/transportProperties      nu = 0.01
system/blockMeshDict              uniform box, 64 x 64 x 32
system/setExprFieldsDict          analytic initial condition
system/setExprBoundaryFieldsDict  analytic lateral inflow values
system/{controlDict,fvSchemes,fvSolution,decomposeParDict}
coreHistory.py                    core radius + peak swirl vs. Burgers theory
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
`delta^2/nu = 4 s`, which is enough to settle. `deltaT 5e-3` gives `Co ~ 0.09`.

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

Run to `endTime 6` on 4 ranks (`NPROCS=4 ./Allrun -par`).

![mid-plane swirl and approach to equilibrium](doc/vortexOverview.png)

The core contracts and the swirl amplifies as predicted for the first ~3 s
(`r_peak` 0.675 -> 0.392, `u_theta` 0.169 -> 0.303), but it stalls well short of
the equilibrium `delta = 0.2` and then reverses: by `t = 6` the core has spread
back to `r_peak = 0.808` and the peak swirl has fallen to 0.260.

![meridional structure](doc/vortexStructure.png)

The meridional plane still shows the Burgers topology -- radial inflow at the
sides, a dividing layer at `z = 0`, axial outflow towards top and bottom -- but
weaker than imposed.

## Caveat: the imposed strain is not maintained

The reversal is not the physics; the driving strain decays away. The mean `u_z`
on the `z = 0.39` plane should stay at `a z`, and it does not: it drops by half
within the first write interval and reaches 4 % of the imposed value by `t = 6`,
even though the `fixedValue` side patches are unchanged. The same case run with
OpenFOAM's `icoFoam` (identical mesh, boundary conditions and schemes -- set
`application icoFoam` and re-run) holds the strain at `a z` to machine accuracy
and contracts the core faster, reaching `r_peak = 0.325` at `t = 2` where
`neoIcoFoam` reaches 0.392.

![strain comparison](doc/strainComparison.png)

A second, smaller issue is the square domain itself: the analytic `fixedValue`
inflow on its sides is not axisymmetric -- corners sit at `r = 1.4`, mid-edges at
`r = 1.0` -- and the mismatch propagates inwards as a four-lobed pattern in the
outer field. A wider box or a cylindrical mesh would remove that one.
