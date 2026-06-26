<!--
SPDX-FileCopyrightText: 2026 NeoFOAM authors

SPDX-License-Identifier: Unlicense
-->

# Review: NeoFOAM recompilation speed

**Problem.** After a cold `pip install .[all] -v`, a subsequent install with **no C++ changes**
recompiles ~300 artifacts (~5 min) instead of ~0. Editing a single `.cpp` is no cheaper.

**Bottom line.** `pip install` is the wrong tool for the C++ inner loop. Every invocation forces
a full **CMake re-configure**, and this project's configure is **not idempotent** — so Ninja
sees ~300 changed command lines and rebuilds them. Fix = stop reconfiguring on each iteration
(build with Ninja directly), and/or make the configure idempotent.

(Full experiment log with timings and raw output: [`build/recompile-journal.md`](../build/recompile-journal.md)
and `build/log.1` … `build/log.8`. All compiles standardized to `-j8`, one at a time.)

## TL;DR result (validated end-to-end)

| `pip install .[all]`, nothing changed | source compiles |
|---|---|
| default (build isolation) | **277** |
| `--no-build-isolation` (used consistently) | **86** (all NeoN) |
| `--no-build-isolation` + NeoN `find_package` fix | **4** (only Ginkgo CUDA codegen; **0 NeoN/NeoFOAM**) |

**Real root cause (confirmed):** NeoN resolves its bundled deps with a `find_package(X QUIET)`-
then-fetch fallback (Kokkos, nlohmann_json, Ginkgo). The wheel installs those deps into the venv,
so on alternating `pip install`s `find_package` flips between the **build-tree** (`_deps/*`) and
the **installed** (`site-packages`) copy → the `-isystem` paths change → scikit-build's mandatory
re-configure makes Ninja recompile all of NeoN (and NeoFOAM) every time.

**Two candidate fixes** (both validated to 0 NeoN/NeoFOAM recompiles; residual 4 = Ginkgo's own
generated `.cu` instantiation files), plus building with `--no-build-isolation`:

**Fix A — NeoN side, `src/NeoN/cmake/CxxThirdParty.cmake`** (wheel build only):
```cmake
if(DEFINED SKBUILD AND NOT NeoN_WITH_PETSC)
  set(CMAKE_DISABLE_FIND_PACKAGE_Kokkos ON)
  set(CMAKE_DISABLE_FIND_PACKAGE_nlohmann_json ON)
  set(CMAKE_DISABLE_FIND_PACKAGE_Ginkgo ON)
endif()
```
Keeps the wheel layout intact (`neon` stays top-level); cost = a NeoN-side change to upstream,
and it enumerates deps by name.

**Fix B — NeoFOAM side, `pyproject.toml`** (one line):
```toml
[tool.scikit-build]
wheel.install-dir = "neofoam"
```
Nests the CMake install under the package so dep configs land at `site-packages/neofoam/lib/cmake`
(off the `find_package` path) instead of `site-packages/lib/cmake`. Cleaner and dep-agnostic, but a
*global* install-dir also relocates NeoN's top-level **`neon`** binding → `import neon` breaks
unless `neon` becomes `neofoam.neon` (1 import site) or the relocation is scoped to deps only.

Measured (iteration 12): Fix B alone gives the same **0 NeoN/NeoFOAM** recompiles as Fix A — so the
NeoN disable can be dropped in favour of the pyproject one-liner once the `neon` package location is
decided.

## Summary table — what I tried

| # | What I tried | Result | Takeaway |
|---|--------------|--------|----------|
| 0 | Cold `pip install .[all]` | 775 targets, ~30 min | baseline |
| 2 | No-op `pip install .[all]` (nothing changed) | **302 targets, 277 compiles, 282 s** | **reproduces the bug** |
| — | Bare `ninja -n` in `_skbuild` right after a build | only **22** dirty | the build dir alone isn't the problem; the *reconfigure* is |
| 3 | Diff `compile_commands.json` across reconfigures | per-file NeoFOAM cmd **identical**; umpire/NeoN flags flip **`-isystem`↔`-I`** | reconfigure rewrites dep command lines → cascade |
| 3 | Two consecutive *plain* `cmake .` | identical | plain CMake is deterministic; instability is **scikit-build vs the cache** |
| 3 | Change `CMAKE_INSTALL_PREFIX` only | dirty unchanged; prefix absent from compile cmds | install prefix **ruled out** |
| 3 | Read `CMakeInit.txt` + configure banner | `CMAKE_PREFIX_PATH` carries `/tmp/pip-build-env-XXXX/overlay`, fresh each run | **build isolation** feeds volatile paths into configure |
| 4 | `pip install --no-build-isolation` (mixed w/ isolated) | one-time rebuild, then a configure **CRASH** | never interleave isolation modes (corrupts Kokkos toolchain cache) |
| 5 | `ninja -C _skbuild` after editing 1 `.cpp` | `Re-checking globbed directories` → reconfigure → hundreds | `CONFIGURE_DEPENDS` globs re-trigger configure on source edits |
| 6 | **`pip install --no-build-isolation` ×2 (consistent)** | **302 → 91 targets** (residual = NeoN only) | **the pip-based win; isolation caused ~210 of the rebuilds** |

## Root cause

Every `pip install` runs scikit-build-core's `build_wheel`, which **always re-runs CMake
configure** before Ninja. That configure is not idempotent, for two independent reasons:

- **A — volatile / non-idempotent configure inputs.**
  - pip **build isolation** creates a fresh `/tmp/pip-build-env-XXXX/overlay` every run and puts
    it in `CMAKE_PREFIX_PATH`; the wheel staging dir `/tmp/tmpXXXX/wheel` is also fresh.
  - The CPM dependencies (NeoN, Kokkos, Umpire, Ginkgo, cpptrace) are added with `SYSTEM YES`.
    On reconfigure their include flags flip between `-isystem` and `-I` (word-diff proof in
    iteration 3). Ninja reports *"command line changed"* for those ~300 objects → they rebuild,
    cascading into `libumpire → libNeoN → libNeoFOAM → bindings`.
- **B — `CONFIGURE_DEPENDS` globs.** A glob in the dependency graph (and
  `include/CMakeLists.txt:30` `GLOB_RECURSE *.hpp` → generated umbrella `NeoFOAM.hpp`) makes
  Ninja re-verify globs and re-run CMake whenever the source tree is touched — so even a `.cpp`
  edit re-triggers (A).

**Secondary hazard found:** switching between isolated and `--no-build-isolation` installs caches
Kokkos's `nvcc_wrapper` / `kokkos_launch_compiler` under a **non-existent** venv
`site-packages/bin/` path; the next reconfigure then dies with
`string REPLACE requires at least four arguments` / `Invalid compiler for CUDA`. Pick one mode
and stay on it. (Recovery: repoint the two `Kokkos_*` cache vars at
`_skbuild/_deps/kokkos-src/bin/` and reconfigure — no full rebuild needed.)

## Constraint: the fix must work through `pip install`

So "iterate with `ninja` directly" is **not** an acceptable primary solution — `pip install`
itself has to become incremental. Since scikit-build-core *always* re-runs CMake configure on
every install, the only way `pip install` is fast is to make that **re-configure produce
byte-identical compile command lines** (so Ninja rebuilds nothing). That requires removing the
two volatile inputs (A) and the reconfigure trigger (B).

## Options (ranked) — all pip-based

### 1. `pip install --no-build-isolation` — removes cause A's biggest volatile input
Build deps are already in `[dev]`/`[all]`, and pyproject's
`[tool.uv] no-build-isolation-package = ["neofoam"]` plus the `dev` comment
("build deps needed in env when no-build-isolation is enabled") show this is the intended path.
```bash
pip install .[all] -v --no-build-isolation
```
This stops a fresh `/tmp/pip-build-env-XXXX/overlay` from entering `CMAKE_PREFIX_PATH` each run.
- **Must be used consistently** — interleaving with isolated installs corrupts Kokkos's cached
  toolchain paths and crashes the next configure (see secondary hazard).
- **Measured result (iteration 6):** a repeated no-iso install drops **302 → 91 targets**. The
  ~210 dependency rebuilds (umpire/kokkos/cpptrace/ginkgo) disappear. **Residual 91 are all
  NeoN** (`_deps/neon-build/src/…`) — see option 2. So this alone is a large, immediate win but
  not a complete fix.

> To make plain `pip install .` (no flag) behave this way for everyone, also set the env in CI /
> docs, or document `pip install . --no-build-isolation` as the canonical command. (pyproject
> can't force `--no-build-isolation` for *pip*; only uv reads `no-build-isolation-package`.)

### 2. Make scikit-build's re-configure idempotent — *the durable pip fix (gets 91 → 0)*
scikit-build reconfigures on every install, so that configure must be a Ninja no-op. Two facts
narrow the target sharply:
- **A plain `cmake .` reconfigure is already fully idempotent** — NeoN.hpp identical, **0
  targets dirty**. So the build graph *can* be a no-op; only **scikit-build's** configure isn't.
- **The residual 91 are exactly the Kokkos-launched targets** (NeoN, then NeoFOAM) — i.e. every
  object compiled through `kokkos_launch_compiler` / `nvcc_wrapper`. That launcher path is part
  of *every* such compile command, and it is **unstable under scikit-build** (the very same
  instability that made iteration 4b crash: it got cached at a non-existent venv
  `site-packages/bin/` path). When the launcher/wrapper path differs between two configures,
  Ninja reports *"command line changed"* for all Kokkos-launched objects → NeoN + NeoFOAM rebuild.

  **Fix:** pin `Kokkos_COMPILE_LAUNCHER` and `Kokkos_NVCC_WRAPPER` to the stable
  `_skbuild/_deps/kokkos-src/bin/…` paths (set them in the preset / CMake cache so scikit-build
  can't re-derive a different value), so the compile command is byte-stable across reconfigures.
- **Also drop unnecessary `CONFIGURE_DEPENDS`** (cause B) so editing C++ doesn't re-trigger
  configure at all.
- Combined with option 1 this takes a no-op `pip install` to **~0 rebuilt**, including CI.

> Note: the umbrella-header theory (`GLOB_RECURSE *.hpp` → `configure_file NeoN.hpp`) was checked
> and **ruled out** — `NeoN.hpp` is byte-identical across reconfigures.

### 3. `git submodule update --init src/NeoN`
Builds NeoN via `add_subdirectory` from a stable in-tree path instead of the CPM cache path
(`~/.cache/CPM/neon/...`). Removes one path-churn source and, crucially, lets the option-2
idempotency fixes be made **in this repo** rather than upstream in NeoN.

## Suggested plan

1. Adopt **option 1** immediately (documented, pip-based) and measure the repeated-install
   rebuild count (in progress).
2. Land **option 2** as the durable fix so even plain `pip install .` is incremental.
3. Use **option 3** to host the NeoN-side parts of option 2.

Clean validation numbers (repeated `--no-build-isolation` install; one-line `.cpp` edit through
`pip install`) are being measured into `build/log.9`+ and will be folded into the journal and
this table.
