# occDrivAerStaticMesh — GPU memory footprint, named peak composition

**Case:** occDrivAerStaticMesh · 65,334,765 cells over **4 ranks** (H200) → **16,333,691 cells/rank**
(197,353,234 internal faces → ~49.3 M faces/rank).
**Peak:** 24,780 MB/rank live (reserved pool 38,147 MB) reached at solver tag **`pEqn`**.
**Method:** Umpire allocation records captured at peak by `MemoryProbe` (`NEOFOAM_MEM_ALLOC_RECORDS=1`,
`UMPIRE_BACKTRACE=On`) on a `UMPIRE_ENABLE_BACKTRACE` build, attributed to the allocating call stack and
condensed by `summarize_alloc_records.py`. Figures are **per rank**; all three best-practice solver
configs (fp64 MG, float MG, PCG+diagonal) were byte-identical, so footprint is solver-independent.

`B/cell` = region bytes ÷ 16,333,691 cells.

## Named peak composition

| MB | × | B/cell | what it is |
|------:|---:|-------:|------------|
| 2,632 | 7 | 169.0 | `Vector<double>` ⇐ **KOmegaSSTModel** ctor (k, ω, νt + working scalar fields) |
| 2,631 | 3 | 168.9 | `Vector<double>` ⇐ **PDE LinearSystem\<double\>** (pressure/scalar matrix + rhs) |
| 2,256 | 2 | 144.8 | `Vector<Vec3>` ⇐ **GeometryScheme::update** (mesh face/cell geometry) |
| 1,169 | 1 | 75.0 | `Vector<Vec3>` ⇐ mesh via **createAdapterRunTime** |
| 1,128 | 1 | 72.4 | `Vector<Vec3>` ⇐ **GeometryScheme** |
| 1,121 | 1 | 72.0 | `Vector<Tensor>` ⇐ **KOmegaSSTModel** (velocity-gradient / stress tensor) |
| 877 | 1 | 56.3 | `Vector<double>` ⇐ **PDE LinearSystem\<Vec3\>** (momentum system) |
| 438 | 1 | 28.1 | `Vector<int>` ⇐ **SharedSparsityBundle** (CSR/COO pattern) |
| **12,252** | **17** | **786.5** | **subtotal of these 8 highlighted regions** |
| 15,273 | — | 980.5 | total of top-25 listed regions |
| 24,780 | — | 1,590.8 | **FULL peak snapshot** (incl. un-listed tail) |

The full-snapshot 1,590.8 B/cell matches the independent NF_MEM_SCOPE timeline peak (~1,600 B/cell),
cross-checking the two measurement paths.

## Rolled up by subsystem

| subsystem | MB | B/cell | rows folded in |
|-----------|------:|-------:|----------------|
| Mesh geometry (`Vector<Vec3>`) | 4,553 | 292.3 | GeometryScheme::update + GeometryScheme + createAdapterRunTime |
| KOmegaSST model fields | 3,753 | 240.9 | `Vector<double>` ×7 + velocity-gradient `Vector<Tensor>` |
| PDE linear systems + sparsity | 3,946 | 253.3 | LinearSystem\<double\> + LinearSystem\<Vec3\> + SharedSparsityBundle |

Mesh geometry, the turbulence model, and the PDE linear systems are the three dominant subsystems;
together they account for ~12.3 GB (~50%) of the peak.

## Coverage caveat

The listed regions cover **15,273 of 24,780 MB**; the remaining **~9,507 MB is not a leak or unmeasured
memory** — it is real, but held by call-sites ranked #26+, which the dump truncates at the top-25
(`shown >= 25` in `memoryProbe.hpp::captureAllocatorRecords`). Because that hook keys on the *full* call
stack, each field allocation is its own site (the 7 KOmegaSST fields appear as seven separate 376 MB
blocks), so 25 slots fill quickly. The tail is ~30 more fields of similar shape — mostly ~376 MB surface
scalar fields (a `double` over the ~49.3 M internal faces/rank) and ~1,128 MB `Vec3` face fields. Raising
the top-N cap (rebuild required) would name them too.

## Reproduce

```bash
# Build with backtrace: scripts/coma/build-nvidia-h200-gcc.sh already sets
#   -DUMPIRE_ENABLE_BACKTRACE=ON -DUMPIRE_ENABLE_BACKTRACE_SYMBOLS=ON -DCMAKE_EXE_LINKER_FLAGS=-rdynamic
./param-study-best.sh                     # MEM_ALLOC_RECORDS on by default (follows MEM_REPLAY)
# -> paramStudyResults/best-practice/<run>.allocRecords.txt         (raw call stacks)
# -> paramStudyResults/best-practice/<run>.allocRecords-summary.txt (this table, per run)

# Regenerate the summary from a raw dump with the bytes/cell column + totals:
python3 summarize_alloc_records.py <run>.allocRecords.txt --top 20 -n 16333691
```

> NB: `MEM_ALLOC_RECORDS=1` captures a backtrace on every allocation and inflates runtime — use
> `MEM_REPLAY=0` (which also disables this) when you need clean `p_ms/solve` timing.
