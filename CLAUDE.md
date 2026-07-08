# CLAUDE.md — NeoFOAM Python package (`neofoam`)

Guidance for the **Python** side of this repo: `src/neofoam/` and its tests
(`test/`). The C++/NeoN core and the `pybFoam` bindings are out of scope for these
notes (mypy and these conventions skip them).

## Build & test (the commands)

`uv` is used **only** to create the venv + bootstrap pip. After that use the plain
Python workflow — never `uv run` / `uv sync`.

```bash
uv venv                       # once
uv pip install pip            # bootstrap pip into the venv
pip install .[all] -v         # install (NOT `pip install -e .`)
pytest test/<area> -q         # scoped tests while working
pytest                        # whole suite (testpaths=test)
pre-commit run --files <changed>        # format + lint + mypy on your diff
```

### Gotchas that cost real time — read before editing

- **Non-editable install.** scikit-build copies sources into site-packages, so edits
  to `src/neofoam/*.py` do **not** take effect until `pip install .[all] -v` re-runs.
  Plan for this before testing source changes.
- **Never `rm -rf _skbuild`** — CMake reconfigures incrementally; wiping forces a
  slow full recompile.
- **Whole-repo checks have pre-existing failures.** Pre-commit `mypy` (whole-tree)
  and `reuse` fail regardless of your diff. Judge your work by `pre-commit run
  --files <changed>` and scoped pytest — fail only on NEW errors in files you touched.
- Tooling: `ruff` (line-length 100, E/F/I), `mypy` `strict=True` on `files=["src"]`
  (excludes `src/NeoN`; `pybFoam`/`NeoN` are `ignore_missing_imports`).
- CLI entry point: `neofoam` (Typer) — see `src/neofoam/cli/app.py`.

### Block-structured AMReX solver (`incompressibleFluidBlockAMR`) — GPU build

`pip install .[all] -v` also builds the vendored `neon.blockamr` engine (AMReX +
JAX/Triton) and its bindings from `src/NeoN`, so the whole
`incompressibleFluidBlockAMR` stack lands in **one** interpreter. Requirements &
gotchas (all wired in `pyproject.toml` / top-level `CMakeLists.txt`):

- **Source OpenFOAM-v2406** (`source .../OpenFOAM-v2406/etc/bashrc`) — it is the
  ABI match for the installed `pybFoam`. The CMake-built `develop/openfoam` has an
  empty `platforms/.../lib`, so linking `libNeoFOAM.so` fails against it.
- **A CUDA toolkit must be visible** (`nvcc` on `PATH`, an NVIDIA GPU). AMReX
  builds with the CUDA backend for the JAX↔AMReX device-memory (dlpack) handoff;
  the Pallas/Triton kernels have **no CPU lowering**. On a CPU-only box AMReX
  falls back to `NONE` and the solver is import-only.
- **Backends are split**: NeoN/Kokkos build with all executors (Serial/OpenMP/
  **CUDA**) as usual — *unchanged* by this — while AMReX takes CUDA independently.
  The one combined-build tweak is a top-level `enable_language(CUDA)` in
  `CMakeLists.txt` so neofoam's own `NeoFOAM` lib sees CUDA compile-features.
  Do **not** set `Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE` (forces every NeoN TU
  through nvcc → `__CUDACC__` errors) and do **not** disable Kokkos CUDA.
- **jax is pinned to `0.10.2`** (`[blockamr]` extra), *not* the 0.9.1 reference:
  jaxlib 0.9.1 emits Triton IR it cannot parse back (`expected
  mlir::triton::CacheModifierAttr, but got: array<i32>`), an env-wide
  nondeterministic crash. Keep the cu13 wheels coherent — no cu12 jax plugins.
- **AMReX is never finalized** in-process. `amrex::Finalize` frees the arena
  allocator's device memory and aborts with `CUDA error 709: context is
  destroyed` once JAX/Kokkos have dropped the shared CUDA context (they are not
  designed to co-reside). The solver `run()` and the test session fixture open
  the runtime but skip finalize; the OS reclaims GPU memory at exit.

```bash
source ~/OpenFOAM/OpenFOAM-v2406/etc/bashrc     # ABI-matched OpenFOAM
pip install .[all] -v                            # builds neofoam + neon.blockamr + jax
pytest test/solver/incompressibleFluidBlockAMR/ -q
```

## Verification norms (definition of done)

A change is done when you have **run** its proof, not when it looks right:

1. the **scoped tests** for the touched area pass (`pytest test/<area> -q`) — after a
   fresh `pip install .[all] -v` if source changed;
2. `pre-commit run --files <changed>` is clean (or only pre-existing failures);
3. you report the actual commands + output, not an assertion that it works.

Never delete or weaken an existing test to get green — surface the conflict instead.
Optional/native deps gate with `pytest.importorskip(...)`, they don't fail.

## Conventions

Detailed guides — read the relevant one before writing code, update it when a
convention changes:

- **[`.claude/CODE_STYLE.md`](.claude/CODE_STYLE.md)** — typing/mypy, no
  `getattr`/`setattr`, `Context.runtime`, Pydantic v2, imports-at-top, ruff.
- **[`.claude/TEST_STYLE.md`](.claude/TEST_STYLE.md)** — `test/` mirrors
  `src/neofoam/` 1:1, free functions not classes, real OpenFOAM case files (never
  dicts-as-strings), pybFoam/OpenFOAM gating, `__init__.py` rules (`test/io/` has
  none — it would shadow stdlib `io`).

The essentials: **mypy is strict** — fix type errors in code, never by relaxing
`[tool.mypy]`; keep *your* changed files clean.

### Commits

- Do **not** add a `Co-Authored-By: Claude` trailer.
- Commit with `git commit --no-verify` (the whole-tree hooks fail pre-existing — see
  gotchas) — but only after `pre-commit run --files <changed>` is clean.
- Branch before committing on `main`. Commit only when asked.

## Architecture (where things live)

The module map lives in **[`.claude/ARCHITECTURE.md`](.claude/ARCHITECTURE.md)** —
read it to locate a config, model, solver, or init step; update it when the layout
changes.

In short: everything is **config-driven** — a case is a set of validated pydantic
configs serialized to OpenFOAM/JSON/YAML. `io/` binds configs to files,
`fields/`/`foam/` build field + scheme configs, `framework/` holds
`ModelSpec`/`SolverSpec` + staged init, `algorithms/solution_loop/` is the
pure-Python time loop, `solver/incompressibleFluid/` is the reference solver,
`agent/` is the LLM case-scaffolding layer.

## Spec workflow (larger features)

For features too large for one session, this repo carries a spec pipeline — use it
in this order:

1. **`/grill-spec`** — interrogate requirements out of the user → `plans/<name>-requirements.md`
2. **`/write-spec`** — turn the brief into a testable spec → `plans/<name>-spec.md`
3. **`/spec-loop plans/<name>-spec.md`** — implement it slice-by-slice with
   evidence-gated iterations under `loop/<feature>/` (never commits; you commit).

`/understand-tests` maps an unfamiliar test suite (coverage + call-graph) before
refactors.
