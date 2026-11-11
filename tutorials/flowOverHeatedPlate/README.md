# flowOverHeatedPlate: an OpenFOAM fluid over a NeoN plate

Laminar channel flow over a plate heated from below, after preCICE's
[flow-over-heated-plate](https://github.com/precice/tutorials/tree/develop/flow-over-heated-plate).
The fluid (`fluid/`, `incompressibleFluid` with Boussinesq and gravity) and the
plate (`solid/`, `heatConductionNeoN`) are coupled Dirichlet-Neumann by
`multiRun` at the plate's top.

```sh
source <OpenFOAM v2406>/etc/bashrc
./Allrun      # about 2 min
./Allclean
```

`system/controlDict.yaml` holds the times every region steps on and
`participants`; `system/coupling.yaml` the numerics of the coupling. The
`coupled` entries of `0.orig/T` (copied to `0/` by `Allrun`) define the coupled patches:

```c++
// fluid/0.orig/T: receives the wall temperature, sends the heat flux into the fluid
interface { type fixedValue; coupled coupledTemperature; value uniform 310; }
// solid/0.orig/T: receives that heat flux, sends its wall temperature
top { type calculated; coupled coupledHeatFlux; value uniform 310; }
```

The heat flux `ρ c_p (ν/Pr + αt) dT/dn` needs `rho` and `cp` in
`fluid/constant/transportProperties`.

The fluid as the Dirichlet side is the less stable direction (the plate's
effusivity is about a seventh of the fluid's): plain passes diverge, Aitken
converges every window in about 12 passes.

At t = 1 s the fluid's mean wall temperature is 303.751 K (preCICE's reference
303.774 K). Nearest-neighbor mapping between the 161 fluid and 100 plate faces
does not conserve the heat: the two sides differ by 0.21 %.
