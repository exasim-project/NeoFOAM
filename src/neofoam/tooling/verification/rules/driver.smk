# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# The shared driver for every foam_tutorials verification study. A study's own
# Snakefile `include:`s this file and then defines its own `rule all` (Snakemake's
# default target must live in the *main* Snakefile — a rule from an included file
# is never picked as the default). Everything else — discovery, the wildcard
# constraint, and the pipeline rules — is here, so a study stays self-contained
# next to its config.yaml and discover.py while the logic lives in one place.
# Nothing here is solver-specific — the study is chosen entirely by --configfile.
#
# The DAG, per case:
#     verify_native ─┐
#     verify_neo   ──┴─ verify_compare ─ results/<id>.json ─┐
#                                                           └─ verify_report ─ report.html
#
# Discovery only reads dictionaries (milliseconds), so the case list is built
# here at parse time and re-derived identically by each worker from the same
# config — no manifest file to drift out of sync.

import os
import re
from pathlib import Path

from neofoam.tooling.verification.study import load_study
from neofoam.tooling.verification.rules import rules_dir

# `configfile:` populates `config`; workflow.configfiles gives us its path, which
# the runner needs to re-load the study (discover resolves relative to it).
CONFIG = os.path.abspath(workflow.configfiles[-1])
STUDY = load_study(Path(CONFIG))

# A study may restrict the sweep to a hand-picked subset (the slice-1 gate).
_only = config.get("only")
CASES = [c for c in STUDY.cases if not _only or c.name in set(_only)]
IDS = [c.id for c in CASES]
THREADS = {c.id: max(1, c.subdomains) for c in CASES}

WORK = "work"
RESULTS = "results"
REPORT = "report.html"

if not IDS:
    raise WorkflowError(
        "no cases discovered — is OpenFOAM sourced (FOAM_TUTORIALS set), and does "
        "`only:` (if present) name real cases?"
    )

# Each case's runs are named by their solver, not by an anonymous native/neo. The
# native solver is the shared reference; the study's candidate backends (one or
# more) are each diffed against it, so the DAG carries one verify_run rule keyed on
# (id, solver) where solver is the native solver or a candidate label.
CANDIDATE_LABELS = STUDY.candidate_labels
SOLVER_LABELS = sorted({c.native_label for c in CASES} | set(CANDIDATE_LABELS))

# Both wildcards are drawn from known-value sets: `id` is flat (slashes → __) and
# `solver` is one of the study's solver labels, so `work/{id}/{solver}.status.json`
# can never wrongly split a path like `a/b`.
wildcard_constraints:
    id="|".join(re.escape(i) for i in IDS),
    solver="|".join(re.escape(s) for s in SOLVER_LABELS),

include: str(rules_dir() / "verify_run.smk")
include: str(rules_dir() / "verify_compare.smk")
include: str(rules_dir() / "verify_report.smk")
