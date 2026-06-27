# tiltedCube — small complex-mesh distributed test case

A deliberately **complex but tiny** snappyHexMesh case for exercising NeoFOAM's
distributed (processor-boundary) code paths quickly. ~2.4k cells, runs in seconds.

## What makes it "complex"

- A cube **tilted** relative to the background grid (27° yaw + 18° pitch), so snapped
  surface faces are non-axis-aligned → genuine skew and non-orthogonality.
- **2 surface layers** (`addLayers`) → prism + polyhedral layer cells.
- One surface refinement level → hanging-node split faces at the refinement interface.
- Result (checkMesh): ~1970 hexahedra + ~24 prisms + ~440 polyhedra,
  max non-orthogonality ~45°, max skewness ~3.
- `scotch` decomposition cuts irregularly through the refined/layered region →
  multiple processor patches per rank.

## Layout

```
constant/triSurface/makeTiltedCube.py   generates the tilted-cube STL
system/blockMeshDict                    coarse background channel (20×8×8)
system/snappyHexMeshDict                refine + snap + 2 layers (quality-capped)
system/surfaceFeatureExtractDict        feature edges for explicit feature snap
system/decomposeParDict                 scotch, 4 subdomains
system/{controlDict,fvSchemes,fvSolution}
0.orig/{U,p}                            inlet U=(1,0,0), Re≈52 (stable laminar)
```

## Required NeoFOAM controlDict keys

NeoFOAM (unlike stock icoFoam) requires three extra keys in `system/controlDict`:

```
executor    GPU;        // GPU = fast single-rank serial; CPU = multi-rank MPI
allocator   default;    // default | umpire
memPoolSize 1;
```

Omitting `allocator` fails immediately with
`FOAM FATAL IO ERROR: Entry 'allocator' not found`.

## Running

```bash
# 1. mesh
./Allmesh

# 2a. serial (fast on GPU; executor GPU is the default in controlDict)
./Allrun

# 2b. parallel — set executor CPU first (single-GPU box ⇒ MPI must be CPU-only)
foamDictionary -entry executor -set CPU system/controlDict
NPROCS=4 ./Allrun -par
```

### Oracle: serial-vs-distributed Courant number

`compareCourant.sh` runs serial then parallel and diffs the per-step **max Courant
number**. Serial is the ground truth; a correct distributed solve must match it.

```bash
NEOFOAM_BIN=../../build/develop/bin/neoIcoFoam ./compareCourant.sh 4
```

NOTE: serial is run with whatever `executor` is in controlDict (GPU = fast), parallel
forces CPU. Serial-GPU and serial-CPU agree, so GPU serial is a valid oracle.
