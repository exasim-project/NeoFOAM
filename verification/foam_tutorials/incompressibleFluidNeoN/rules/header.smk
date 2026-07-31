# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# The drop-in study pipeline — the whole thing, in one `include:`. This study's
# own Snakefile includes this file and then defines its own `rule all` (Snakemake's
# default target must live in the *main* Snakefile: it saves and restores
# `default_target` around every include, so a rule from an included file is never
# promoted). Everything else — the case list, the wildcard constraints, and the
# five pipeline rules — is here, so a study directory is exactly Snakefile,
# config.yaml, discover.py, plus this rules/ set (no longer shared with any other
# study — each study owns its own copy).
#
# Nothing here is solver-specific: which tutorials are swept, which native solvers
# are replaced, and which fields are diffed all come from --configfile.
#
# The DAG, per case, per solver (native reference + each candidate backend):
#     build_case ─ swap_solver ─ run ─ {solver}/.ran ┐
#                                                    └─ compare ─ results/{id}.json ┐
#                                                                                   └─ report ─ report.html
#
# Discovery only reads dictionaries (milliseconds), so the case list is built here
# at parse time and re-derived identically by each worker from the same config —
# no manifest file to drift out of sync.

import os
import re
import sys
from pathlib import Path

# `configfile:` populates `config`; workflow.configfiles gives its path, which the
# runner needs to re-load the study (discover resolves relative to it).
CONFIG = os.path.abspath(workflow.configfiles[-1])

# verification/dropin/ lives outside the neofoam package (see verification/README.md);
# config.yaml sits three directories below the repo root
# (verification/foam_tutorials/<study>/config.yaml), so this is the one place
# in-process code needs the repo root on sys.path — every subprocess this pipeline
# shells out to gets it via PYTHONPATH instead (see build_case.smk and friends).
REPO_ROOT = Path(CONFIG).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from verification.dropin.cases import load_study

STUDY = load_study(Path(CONFIG))
CASES = STUDY.cases
IDS = [c.id for c in CASES]
THREADS = {c.id: max(1, c.subdomains) for c in CASES}

CASE_ROOT = "cases"
RESULTS = "results"
REPORT = "report.html"

if not IDS:
    raise WorkflowError(
        "no cases selected — is OpenFOAM sourced (FOAM_TUTORIALS set), and do "
        "config.yaml's `cases:` / `only:` name real tutorials?"
    )

# The native solver is the shared reference; the study's candidate backends are each
# diffed against it. Every run is keyed on (id, solver), where solver is the native
# solver label or a candidate label, so each lands in a solver-named dir.
CANDIDATE_LABELS = STUDY.candidate_labels
SOLVER_LABELS = sorted({c.native_label for c in CASES} | set(CANDIDATE_LABELS))

# Both wildcards are drawn from known-value sets: `id` is flat (slashes → __) and
# `solver` is one of the study's solver labels, so `cases/{id}/{solver}/...` can
# never wrongly split a path like `a/b`.
wildcard_constraints:
    id="|".join(re.escape(i) for i in IDS),
    solver="|".join(re.escape(s) for s in SOLVER_LABELS),

include: "build_case.smk"
include: "swap_solver.smk"
include: "run.smk"
include: "compare.smk"
include: "report.smk"
