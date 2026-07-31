<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# verification/

OpenFOAM tutorial drop-in studies: take a tutorial verbatim, swap only the
solver token in its `Allrun`, run both sides, diff the final-time fields.

This tree is self-contained on purpose — it is meant to be lifted into its own
repo eventually (see `plans/workflow-composable-and-benchmarkable.md`). It is
not packaged (no `pyproject.toml` here yet; that lands at split time) and is run
in place from this checkout.

## Layout

- `dropin/` — the engine: load a study's `config.yaml` + `discover.py`, stage
  both sides with `neofoam.tooling.casebuild`, run, diff, report. The *only*
  neofoam surface this code touches is `neofoam.tooling.casebuild`,
  `neofoam.io.DictFile`, and the `neofoam solver <backend>` CLI (invoked as a
  subprocess from a generated `Allrun`, never imported).
- `foam_tutorials/<study>/` — one study each: `Snakefile` + `config.yaml` +
  `discover.py` + its own `rules/` (a copy of the five-rule pipeline shape;
  each study's copy can diverge independently, though today they don't).

## Running a study

Needs a sourced OpenFOAM (`$FOAM_TUTORIALS` set) and `snakemake` installed
(`pip install -e ".[verification]"` from the repo root, or just `snakemake` +
whatever `dropin/` needs, already covered by the repo's own dependencies).

```bash
cd verification/foam_tutorials/incompressibleFluid
snakemake --configfile config.yaml -n     # dry run: see the DAG, no OpenFOAM needed to invoke it
snakemake --configfile config.yaml -j1    # -j1: some cases shell out to mpirun internally
open report.html                          # or your platform's equivalent
```

Same for `incompressibleVoF` (`-j6` is safe there — its cases are lighter).

Output (gitignored, not tracked): `cases/` (staged runs + logs),
`results/<id>.json` (per-case evidence, every backend), `report.html`.

## Adding a study

Three files plus a `rules/` copy: `Snakefile` (`include: "rules/header.smk"` +
`rule all: input: REPORT`), `config.yaml` (title, `discover:` path, `apps:`,
`cases:`/`only:`, `exclude:` — each entry a `case:` name plus a `reason:`
for cases that can never be drop-ins — and `simplify:` — a `case:`, a `reason:`
and a `patch:` of `rel/dict/path: {dotted.key: value}` substituting settings the
backend does not support, staged into *both* sides so a match stays meaningful),
`discover.py` (a `discover()` — see the
two existing ones for
the two accepted signatures), and `rules/` copied from an existing study
verbatim (nothing in it is study-specific).
