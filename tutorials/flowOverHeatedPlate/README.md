# flowOverHeatedPlate: conjugate heat transfer, OpenFOAM fluid and NeoN plate

Laminar channel flow over a plate heated from below. The fluid (region `fluid`)
runs on OpenFOAM through pybFoam with `incompressibleFluid` and the Boussinesq
model, with gravity. The plate (region `solid`) runs on NeoN with
`heatConductionNeoN`. `multiRun` couples the fluid's patch `interface` and the
plate's `top` in one process, Dirichlet–Neumann.

```sh
source <OpenFOAM v2406>/etc/bashrc          # neofoam installed in the Python environment
./Allrun                                    # about 1.5 min
./Allclean
```

## Layout

Each region is an OpenFOAM case of its own, next to the files of the whole case:

| file | |
|---|---|
| `system/controlDict.yaml` | the times every region steps on (Δt = 0.01 s to 1 s, a write every 0.2 s) and `regionSolvers: {fluid: incompressibleFluid, solid: heatConductionNeoN}` |
| `system/regionCoupling.yaml` | the numerics of the coupling: serial, Aitken, nearest-neighbor, convergence, placement |
| `fluid/`, `solid/` | `0/`, `constant/`, `system/` of each region; `system/controlDict` holds the same times for OpenFOAM's `Time` |

The boundary defines the coupling: a patch of `0/T` is coupled when a `coupled`
entry names its condition. `system/regionCoupling.yaml` only pairs the patches
and sets the numerics.

```c++
// fluid/0/T: the Dirichlet side receives the wall temperature, sends the heat flux into the fluid
interface { type fixedValue; coupled coupledTemperature; value uniform 310; }
// solid/0/T: the Neumann side receives that heat flux, sends its wall temperature
top { type calculated; coupled coupledHeatFlux; value uniform 310; }
```

The fluid's heat flux is `q = ρ c_p (ν/Pr + αt) dT/dn` in W/m². ρ and c_p are
entries of the Boussinesq model in `fluid/constant/transportProperties`.
Both solvers write the `coupled` entries back into each written `T`, so a run
restarts coupled from any written time.

## Source

The case is preCICE's
[flow-over-heated-plate](https://github.com/precice/tutorials/tree/develop/flow-over-heated-plate)
(`fluid-openfoam` and `solid-openfoam`), after Vynnycky et al., *Forced
convection heat transfer from a flat plate: the conjugate problem*, Int. J. Heat
Mass Transfer 41(1), 1998. Geometry and boundary conditions are those of the
earlier `buoyantFlowOverHeatedPlate` tutorial (branch `feat/buoyant-cht`).

| | value |
|---|---|
| channel | x ∈ [-0.5, 3], y ∈ [0, 0.5]; 12013 cells, graded as preCICE's mesh; 161 faces on the plate |
| plate | x ∈ [0, 1], y ∈ [-0.25, 0]; 100 × 25 cells |
| inlet | u = 0.1 m/s, T = 300 K; outlet p_rgh = 0; channel top slip; plate top no-slip |
| fluid | ν = 2e-4 m²/s, Pr = 0.01, ρ = 1, c_p = 5000 (k = ρ c_p ν/Pr = 100); β = 1/300 1/K, TRef = 300 K, g = (0 −9.81 0) m/s² |
| plate | k = 100, ρ = 1, c_p = 100; T = 310 K initially and at its bottom; sides adiabatic |
| time | Δt = 0.01 s (one coupling window per step), end time 1 s |
| coupling | serial (the fluid first), Aitken (initial relaxation 0.1), nearest-neighbor, relative convergence 1e-5 on T, at most 30 passes |

## The coupling

The node wraps the fluid's PIMPLE loop (`operations: [inner_loop]`) and the
plate's `solve_T`; the time step stays outside it. Each pass repeats the fluid's
whole PIMPLE step (momentum, energy, pressure) from the same old times, since T
drives the flow.

**The direction.** Here the fluid is the Dirichlet side and the plate the
Neumann side; preCICE does it the other way round. The fluid as the Dirichlet
side is the less stable direction. The plate's thermal effusivity is the smaller
one (100 against 707 for the fluid's wall cells), and the factor by which a
plain pass amplifies an error is about −7 for long waves. Without relaxation the
passes diverge. Aitken converges every window, in 15–16 passes in the first
windows and 11–13 later, about 12 on average. The other direction (factor
−0.14 to −0.65) would converge without relaxation. It needs a Neumann condition
on the OpenFOAM side, which pybFoam does not bind yet.

`convergence: {type: relative, data: T}` checks every array of field T that
the sides send: the fluid's heat flux and the plate's wall temperature.

## Results (t = 1 s)

The fluid's wall temperature on the interface's face centres:

| | mean |
|---|---|
| this case | 303.751 K (min 302.498, max 304.138) |
| `buoyantFlowOverHeatedPlate` (Robin–Robin, IQN-ILS) | 303.764 K |
| preCICE reference (`fluid-openfoam_solid-openfoam`) | 303.774 K |

- The mean is 0.023 K below preCICE's: 0.2 % of the 10 K temperature rise.
- The coupling takes 11.8 passes per window on average (1183 in all), and every
  window converges. With a limit of 1e-8 instead of 1e-5 (17.2 passes per
  window) the wall temperature changes by 7e-7 K at most.
- The heat through the interface: 975.15 W into the fluid, 977.24 W out of the
  plate (per 0.4 m depth), 0.21 % apart. Nearest-neighbor mapping between the
  161 graded fluid faces and the plate's 100 uniform faces does not conserve
  it. The difference grows from 0.01 % in the first window, as the heat flux
  flattens out along the plate.

`test/solver/incompressibleFluid/test_heated_plate.py` runs this case's first
three time steps. It checks the passes, the wall temperature, the heat balance
across the interface and the written `coupled` entries.
