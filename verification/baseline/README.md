<!--
SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2026 NeoFOAM authors
-->

# baseline/

The pre-change reference sweep (`plans/orchestration.md` stage 1). `results/`
is gitignored, so this is the one place evidence enters the repo — verbatim
copies of every `results/<id>.json`, one archive per study, taken from a
single code state so "did this case move" is measurable for every later fix.

## Provenance

| | |
|---|---|
| commit | `a14ac56ad2cec107ff2cae30369db976bc21207d` (this branch, pre-extraction) |
| OpenFOAM | v2406 (`/home/henning/OpenFOAM/OpenFOAM-v2406`) |
| host | henning-MS-7E26 (Linux 7.0.0-28-generic) |
| date | 2026-07-27 |
| install | non-editable (`pip install ".[all]"`) in a throwaway worktree's venv |
| invocation | `snakemake --configfile config.yaml -j1`, per study, sequentially |

Only each study's default `config.yaml` is covered. `config-isoadvector.yaml`
is deliberately not swept: both VoF configs share one `cases/`/`results/`
output root, so a second sweep would silently overwrite this evidence (the
collision is recorded in `plans/workflow-composable-and-benchmarkable.md` §5;
the output-root fix is deferred with it).

## Outcome tallies at the baseline

`incompressibleVoF-2026-07-27.json` — 54 cases, 1 candidate each:
36 FIELDS_DIFFER, 7 MATCHED, 4 SOLVER_FAILED, 3 MATCHED_TO_ROUNDOFF,
3 MESH_DIFFERS, 1 NATIVE_FAILED. Three-state fold: matched 10, differs 39,
failed 4, one harness fault.

`incompressibleFluid-2026-07-27.json` — 45 cases, 2 candidates each (first
sweep ever): 46 SOLVER_FAILED, 16 FIELDS_DIFFER, 12 NATIVE_FAILED, 9 MATCHED,
6 CASE_SETUP_FAILED, 1 COMPARE_FAILED. The 19 fault runs (NATIVE_FAILED,
CASE_SETUP_FAILED, COMPARE_FAILED) are harness work, not solver evidence.
