# stretchedVortexCylinder — the strained vortex on an axisymmetric domain

The [stretchedVortex](../stretchedVortex/README.md) case on a cylinder instead of
a square box. Everything else -- the analytic Burgers initial and boundary
values, the Dirichlet-velocity-everywhere treatment, the exact `delta(t)` the
result is checked against -- is the same, so read that README first. This one
records only what differs.

## Why

The square box imposes the analytic field on a boundary whose distance from the
axis varies: corners sit at `r = 1.41`, mid-edges at `r = 1.0`. The solution is
axisymmetric and the boundary is not, which leaves a few per cent azimuthal
ripple in the outer field. A cylinder removes the mismatch by construction.

## What differs

```
system/blockMeshDict      O-grid cylinder, radius 1, z in [-0.75, 0.75]
                          core block NI x NI, four outer blocks NI x NR, NZ layers
system/fvSchemes          laplacians are "corrected": an O-grid runs to 38 degrees
                          non-orthogonality against ~0 for the box
system/fvSolution         nNonOrthogonalCorrectors 1, for the same reason
coreHistory.py            bins by radius using the cell centres that Allrun writes;
                          an O-grid has no i,j,k indexing to compute them from
setMeshDensity.py         scales NI, NR and NZ together
```

`Allrun` therefore has one extra step, `postProcess -func writeCellCentres`,
whose output the diagnostics read.

## The one trade-off

The box balances the prescribed boundary flux *exactly*: its faces are flat, so
midpoint quadrature is exact, and the swirl cancels term by term between
symmetric cell centres. A circle discretised as chords cannot do that
algebraically -- the measured net flux here is 7e-10 relative to a throughput of
4.71, against the box's 0.0. Since `neoIcoFoam` has no `adjustPhi` to absorb a
residual, that shows up as a small steady continuity error (~3e-12 per step
against the box's ~4e-20). It is far below the pressure solver tolerance and does
not grow, but it is the price of the curved boundary.

In exchange the swirl contributes *nothing* to the boundary flux here: each side
face is a chord whose normal runs through its own centre, so the tangential
velocity is exactly perpendicular to it, face by face rather than in the sum.

## Comparing the two

`../stretchedVortex/compareRuns.py` reads both, since it takes cell centres when
they exist and block counts otherwise:

```bash
python3 ../stretchedVortex/compareRuns.py box=../stretchedVortex cylinder=.
```
