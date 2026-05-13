---
phase: 02-gpu-kernel-fixes
plan: "01"
subsystem: NeoN device lambda audit
tags: [gpu, kernel, kokkos, audit, std-math]
dependency_graph:
  requires: [01-02 GPU MPI smoke test — baseline 66/80 ctest passing]
  provides: [GPU-KRN-01, GPU-KRN-02, GPU-KRN-03 confirmed closed]
  affects: [02-02 NeoFOAM procFaceCheck host-side audit]
tech_stack:
  added: []
  patterns: [Kokkos::sqrt as canonical device-safe math in NEON_LAMBDA bodies]
key_files:
  created:
    - .planning/phases/02-gpu-kernel-fixes/02-01-SUMMARY.md
  modified:
    - src/NeoN/src/linearAlgebra/ginkgo/ginkgoL1Stop.cpp
decisions:
  - "Audit confirms NeoN NEON_LAMBDA bodies are already std::-free — no code fixes needed"
  - "ginkgoL1Stop.cpp:200 std::min/max documented as host-side exempt with inline comment"
  - "procFaceCheck.hpp std::abs/max exempt — host-side face comparison utility, covered in plan 02-02"
requirements-completed: [GPU-KRN-01, GPU-KRN-02, GPU-KRN-03]
metrics:
  duration: "~10 minutes"
  completed: "2026-05-13"
  tasks_completed: 2
  files_modified: 1
---

# Phase 02 Plan 01: NeoN Device Lambda Audit Summary

**NeoN NEON_LAMBDA bodies confirmed std::-free; one known host-side exempt instance documented in ginkgoL1Stop.cpp with inline comment; vec3.hpp Kokkos::sqrt canonical pattern verified unchanged.**

## Performance

- **Duration:** ~10 minutes
- **Started:** 2026-05-13T13:49:16Z
- **Completed:** 2026-05-13T13:57:00Z
- **Tasks:** 2
- **Files modified:** 1 (ginkgoL1Stop.cpp — exempt comment only)

## Accomplishments

- Ran canonical audit grep across all NeoN production headers and sources (`src/NeoN/include/` and `src/NeoN/src/`, excluding test/ files)
- Confirmed zero `std::abs`, `std::sqrt`, `std::max`, `std::min` hits in any `NEON_LAMBDA` / `KOKKOS_LAMBDA` device-compiled path
- Added host-side exempt comment above the lone exempt instance in `ginkgoL1Stop.cpp:200`
- Verified `vec3.hpp:179` canonical `Kokkos::sqrt` pattern is unchanged

## What Was Verified

**Audit command run:**

```bash
grep -rn "std::abs\|std::sqrt\|std::max\|std::min" \
  src/NeoN/include/ src/NeoN/src/ \
  2>/dev/null | grep -v "test/"
```

**Output (entire output — one line):**

```
src/NeoN/src/linearAlgebra/ginkgo/ginkgoL1Stop.cpp:200:                    frequency = std::min(norm_eval_limit_, std::max(1, localIdx(1 / alpha)));
```

**Interpretation:** The only `std::` math call in NeoN production code (headers + src, excluding tests)
is the single instance at `ginkgoL1Stop.cpp:200`. This is inside a Ginkgo convergence criterion
class method — specifically in `StoppingCriterion::build_dist_stopping_criterion()` — which is
host-side code called from the linear solver setup path. It is **not** inside a `NEON_LAMBDA`,
`KOKKOS_LAMBDA`, or any `NEON_INLINE_FUNCTION`/`KOKKOS_INLINE_FUNCTION`-annotated function that
the CUDA compiler would compile for device execution.

**Exclusion verification (zero-line result):**

```bash
grep -rn "std::abs\|std::sqrt\|std::max\|std::min" \
  src/NeoN/include/ src/NeoN/src/ 2>/dev/null \
  | grep -v "test/" \
  | grep -v "ginkgoL1Stop.cpp"
# Returns: (empty — zero lines)
```

**NeoFOAM side (separately verified):**

```bash
grep -rn "std::abs\|std::sqrt\|std::max\|std::min" include/NeoFOAM/ src/ 2>/dev/null | grep -v "test/"
```

Output shows only `procFaceCheck.hpp:30-32` — a host-side utility outside any device lambda.
That instance is handled in plan 02-02.

## Exempt Instances Documented

### 1. `src/NeoN/src/linearAlgebra/ginkgo/ginkgoL1Stop.cpp:200` (DONE)

**Context:** Ginkgo distributed convergence criterion — `StoppingCriterion::build_dist_stopping_criterion()`
host-side method. Computes adaptive `frequency` and `minIter` based on previous solve cost.

**Why exempt:** This code runs on the host CPU at solver setup time, not in any device kernel.
The Ginkgo `DistStoppingCriterion` class methods are C++ host-side code; the CUDA compiler
does not process this translation unit as `__device__` code.

**Action taken:** Added inline exempt comment (reformatted by clang-format to stay within 100-char limit):

```cpp
// host-side convergence criterion — std::min/max are safe here (not
// device-compiled)
frequency = std::min(norm_eval_limit_, std::max(1, localIdx(1 / alpha)));
```

**Commit:** NeoN `d471e2ac5` (feat/gpu-distributed); NeoFOAM submodule bump `782940b5`

### 2. `include/NeoFOAM/auxiliary/procFaceCheck.hpp:30-32` (DEFERRED to plan 02-02)

**Context:** NeoFOAM host-side face comparison utility. Not device-compiled.

```cpp
const double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
const double mag = std::max(std::abs(static_cast<double>(a)), std::abs(static_cast<double>(b)));
return diff <= tol * std::max(1.0, mag);
```

**Why exempt:** Pure host utility for comparing face values in test assertions. Not inside any
`NEON_LAMBDA` body. Covered in plan 02-02.

## Canonical Kokkos:: Pattern Verified

`vec3.hpp:179` (KOKKOS_INLINE_FUNCTION — device-safe):

```cpp
KOKKOS_INLINE_FUNCTION
scalar mag(const Vec3& vec)
{
    return Kokkos::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}
```

**Verification:**

```bash
grep -n "Kokkos::sqrt" src/NeoN/include/NeoN/core/primitives/vec3.hpp
# Returns: 179:    return Kokkos::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
```

This is the established pattern for device-safe math in NeoN primitives. All other math in
`NEON_INLINE_FUNCTION` or `NEON_LAMBDA` bodies already follows this pattern.

## Requirements Closed

| Req | Description | Status |
|-----|-------------|--------|
| GPU-KRN-01 | NEON_LAMBDA bodies use `Kokkos::abs` not `std::abs` | **CLOSED** — zero `std::abs` hits in device-compiled paths |
| GPU-KRN-02 | NEON_LAMBDA bodies use `Kokkos::sqrt` not `std::sqrt` | **CLOSED** — zero `std::sqrt` hits in device-compiled paths; `vec3.hpp` canonical pattern confirmed |
| GPU-KRN-03 | NEON_LAMBDA bodies use `Kokkos::max`/`Kokkos::min` | **CLOSED** — zero `std::max`/`std::min` hits in device-compiled paths |

Note: GPU-KRN-04 (GPU build exits 0 with no device-compilation errors) is addressed in plan 02-02.

## Git Commits

| Task | Repo | Commit | Description |
|------|------|--------|-------------|
| Task 1 (exempt comment) | NeoN | `d471e2ac5` | fix(02-01): add host-side exempt comment to ginkgoL1Stop.cpp |
| Task 1 (submodule bump) | NeoFOAM | `782940b5` | chore(02-01): bump NeoN submodule — exempt comment on ginkgoL1Stop.cpp |

## Deviations from Plan

None — plan executed exactly as written. Audit found exactly the expected single exempt instance.
No device lambda `std::` math calls required fixing.

## Issues Encountered

**clang-format comment wrapping:** The first commit attempt triggered the pre-commit clang-format
hook which reformatted the single-line exempt comment into two lines (100-char column limit).
The hook correctly modified the file. Re-staged and committed the formatted version — standard
pre-commit workflow, not a blocker.

## Next Phase Readiness

- GPU-KRN-01/02/03 closed — NeoN device lambda math confirmed GPU-safe
- Plan 02-02 covers: NeoFOAM procFaceCheck.hpp host-side audit + GPU-KRN-04 build clean check
- No blockers for 02-02

---

*Phase: 02-gpu-kernel-fixes*
*Completed: 2026-05-13*
