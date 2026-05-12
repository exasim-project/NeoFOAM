---
phase: 05-end-to-end-validation
plan: "02"
subsystem: tutorials/cylinder3D
tags: [validation, reference-run, field-comparison, parallel]
dependency_graph:
  requires: []
  provides:
    - "tutorials/cylinder3D/Allrun.reference — 4-rank icoFoam reference run script"
    - "tutorials/cylinder3D/Allrun_parallel — 4-rank neoIcoFoam diagnostic run script"
    - "tutorials/cylinder3D/compare_fields.py — per-rank L∞ field comparison script"
  affects: []
tech_stack:
  added:
    - "Python 3 — compare_fields.py field comparison script"
  patterns:
    - "No reconstructPar: direct per-rank processorN/<time>/ field comparison"
    - "L∞ norm concatenated in rank order (scalar: max|a-b|, vector: max||a-b||)"
    - "gzip-transparent file reader (tries .gz suffix first)"
key_files:
  created:
    - tutorials/cylinder3D/Allrun.reference
    - tutorials/cylinder3D/Allrun_parallel
    - tutorials/cylinder3D/compare_fields.py
  modified: []
decisions:
  - "Parallel icoFoam reference (not serial): 4-rank icoFoam with same decomposition gives faster ground truth and identical per-rank cell ordering — no reconstructPar needed"
  - "List terminator regex: use newline-anchored ) pattern to distinguish list close from vector entry parens in OpenFOAM ASCII format"
metrics:
  duration: "~10 minutes"
  completed: "2026-05-12"
  tasks_completed: 4
  files_created: 3
  files_modified: 0
---

# Phase 5 Plan 2: Reference Validation Infrastructure Summary

**One-liner:** Parallel icoFoam reference scripts and per-rank L∞ comparison tool for cylinder3D 4-rank validation without reconstructPar.

## What Was Built

Three files added to `tutorials/cylinder3D/`:

1. **`Allrun.reference`** — Creates `cylinder3D_ref/` sibling by copying the decomposed case (processor0..3/), overrides `endTime=5e-3` and `writeInterval=20` via `foamDictionary`, then runs `mpirun -np 4 icoFoam -parallel`. No blockMesh or decomposePar needed because the processor dirs are already present.

2. **`Allrun_parallel`** — Runs `mpirun -np 4 neoIcoFoam -parallel` in `cylinder3D/` with the same short diagnostic window (5e-3 s = 100 timesteps at deltaT=5e-5, 5 output snapshots). Restores original `endTime` and `writeInterval` after the run.

3. **`compare_fields.py`** — Python script that reads `processorN/<time>/field` from both ref and neo directories, concatenates values in rank order, and computes L∞ error (max|a−b| for scalars, max‖a−b‖ for vectors). Handles gzip-compressed and plain ASCII files. Exits non-zero if any L∞ exceeds `--threshold`. CLI: `--ref`, `--neo`, `--fields`, `--threshold`, `--ranks`.

## Commits

| Hash | Message |
|------|---------|
| 0dbcad54 | feat(05-02): add Allrun.reference for 4-rank icoFoam diagnostic reference run |
| e51d8408 | feat(05-02): add Allrun_parallel for 4-rank neoIcoFoam diagnostic run |
| a335dfdc | feat(05-02): add compare_fields.py for per-rank L∞ field comparison |
| 3fb0a81a | fix(05-02): fix vector field list-terminator parsing in compare_fields.py |

## Smoke Test Results

Task 4 smoke test (self-comparison with no output times):
```
python compare_fields.py --ref tutorials/cylinder3D --neo tutorials/cylinder3D
  --fields U p --threshold 1e-10
→ "No common output times found — skipping comparison."
→ PASS  (exit 0)
```

Extended synthetic tests with actual field data:
- Self-comparison with synthetic scalar+vector data at t=0.005 → L∞ = 0 (PASS)
- Cross-comparison of identical data → L∞ = 0 (PASS)
- Cross-comparison with 0.5 delta in p, threshold 1e-2 → L∞ = 0.5 > threshold (FAIL/exit 1 — correct behavior)
- Cross-comparison with 0.5 delta in p, threshold 1.0 → L∞ = 0.5 < threshold (PASS — correct)
- gzip transparent read test → PASS

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed vector field list-terminator parsing**
- **Found during:** Task 4 (smoke test with synthetic vector data)
- **Issue:** `after_paren.find(")")` finds the first `)` which is part of vector entry `(u v w)`, not the list-closing `)` on its own line. This caused the regex to extract only `\n(1.0 0.0 0.0` as the raw data, resulting in zero parsed values.
- **Fix:** Replace `find(")")` with `re.search(r"\n\s*\)\s*(\n|$)", after_paren)` which correctly identifies the list terminator (a `)` on its own line) distinct from vector entry parentheses.
- **Files modified:** `tutorials/cylinder3D/compare_fields.py`
- **Commit:** 3fb0a81a

## Known Stubs

None. All three scripts are fully implemented. `compare_fields.py` reads actual field data; no placeholder values or hardcoded mock data.

## Threat Flags

None. The scripts are local file readers and shell scripts — no network endpoints, auth paths, or schema changes introduced.

## Self-Check: PASSED

- [x] `tutorials/cylinder3D/Allrun.reference` exists and is executable
- [x] `tutorials/cylinder3D/Allrun_parallel` exists and is executable
- [x] `tutorials/cylinder3D/compare_fields.py` exists and is executable
- [x] All 4 commits exist in git log (0dbcad54, e51d8408, a335dfdc, 3fb0a81a)
- [x] Smoke test exits 0
- [x] compare_fields.py contains "linf" keyword (plan artifact requirement)
- [x] Allrun.reference contains "icoFoam" (plan artifact requirement)
- [x] Allrun_parallel contains "neoIcoFoam" (plan artifact requirement)
