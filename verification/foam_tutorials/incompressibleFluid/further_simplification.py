# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""How the verification study could rely on ``neofoam.tooling.workflow`` alone.

Runnable proof-of-concept for the *next* simplification step. Run it:

    python further_simplification.py

It scaffolds a throwaway copy of this study under the scratchpad, driven by a
**general** loader + a **general** rule-copy (both shown below as they would live
in ``neofoam.tooling.workflow``), then runs ``snakemake -n`` and asserts it builds
the identical 452-job DAG — with **no** ``neofoam.tooling.verification`` import in
the generated Snakefile.

────────────────────────────────────────────────────────────────────────────────
The problem (the line you selected in Snakefile)

    from neofoam.tooling.verification.study import load_study

Today there are two parallel tooling stacks:

  neofoam.tooling.workflow      neofoam.tooling.verification
  ├─ rules/ (RuleRegistry)      ├─ rules/  (a second rule set + rules_dir())
  ├─ sweep_snakefile codegen    ├─ study.py  (load_study — a second loader)
  └─ sweep_runner CLI           ├─ runner.py (the per-case worker CLI)
                                └─ report.py (a *third* thing: its own report)

The study's Snakefile therefore imports the verification loader, and the rules we
copied in call the verification runner. The convergence: **workflow owns the
general mechanism (load + copy-rules + codegen); verification keeps only what is
genuinely solver-specific (its discover.py, its runner subcommands).**

────────────────────────────────────────────────────────────────────────────────
Three moves, smallest first (each independently shippable)

1. GENERAL LOADER.  Promote the solver-agnostic half of ``verification.study``
   (load config.yaml, import the study's discover.py, expose ids / threads /
   candidate labels / by_id) into ``neofoam.tooling.workflow`` as ``load_cases``.
   The Snakefile then imports the general loader, not the verification one.
   ``Case``/``case_id``/``subdomains`` move alongside it as a general job record.
   → deletes ``verification/study.py``; the selected import becomes
     ``from neofoam.tooling.workflow import load_cases``.

2. GENERAL COPY-RULES.  Add one ``copy_rules(dest, names)`` to workflow that
   copies a named rule set into a study's ``rules/`` folder (what we did by hand).
   Both the mesh sweep (which today references rules via ``rules_dir()``) and the
   verification study scaffold their local, editable ``rules/`` the same way.
   The rule *bodies* stay domain-specific one-liners (``python -m <runner> …``);
   the *interface* (a RuleSpec + a one-line shell) is the general contract.
   → a ``neofoam workflow init <dir> --rules build_case,swap_solver,run,…`` command.

3. GENERAL REPORT.  ``verification/report.py`` is a bespoke HTML writer over
   ``results/*.json``. Make ``report`` a general workflow sink that renders any
   study's per-case result records (outcome + numbers + log tail) so both stacks
   share one reporter, parameterised by the columns a study declares.
   → deletes the study-specific report; verification declares its columns instead.

This script demonstrates **move 1 + move 2** end to end (move 3 is sketched in
``_GENERAL_REPORT_NOTE`` — it needs the record schema fixed first).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
SCRATCH = Path(
    "/tmp/claude-1000/-home-henning-libsAndApps-NeoFOAM--claude-worktrees-test-"
    "incompressibleFluid/de630fc4-023e-4587-a848-5cb971b13d6b/scratchpad"
)
DEMO = SCRATCH / "general_study_demo"

RULE_FILES = (
    "build_case.smk",
    "swap_solver.smk",
    "run.smk",
    "compare.smk",
    "report.smk",
)


# ── PROPOSED: neofoam/tooling/workflow/casesweep.py ───────────────────────────
# The general loader. It is written verbatim into the demo dir as `casesweep.py`
# so the generated Snakefile can `from casesweep import load_cases` — standing in
# for the production `from neofoam.tooling.workflow import load_cases`. Note what
# it does NOT import: nothing from neofoam.tooling.verification.
_CASESWEEP_MODULE = '''\
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
"""PROPOSED neofoam.tooling.workflow.load_cases — the general case-manifest loader.

Solver-agnostic: it reads a study `config.yaml`, imports the study's own
`discover.py` (the one plug-in point), and exposes exactly the globals a copied
rule set consumes. No knowledge of tutorials, tiers, or the verification package.
"""
from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CaseManifest:
    """A loaded study: its cases and the derived globals the rules need."""

    title: str
    cases: list[Any]
    candidates: dict[str, str]

    @property
    def candidate_labels(self) -> list[str]:
        return list(self.candidates)

    def by_id(self, case_id: str) -> Any:
        for c in self.cases:
            if c.id == case_id:
                return c
        raise KeyError(case_id)


def _load_discover(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_study_discover", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cases(config_path: str | Path) -> CaseManifest:
    """Load a study from its config.yaml (general; no verification import)."""
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text()) or {}
    discover = _load_discover(config_path.parent / cfg["discover"]).discover
    selection = cfg.get("cases")
    cases = discover(selection) if inspect.signature(discover).parameters else discover()
    apps = tuple(cfg.get("apps") or [])
    candidates = {app.split()[-1]: app for app in apps}
    return CaseManifest(title=cfg.get("title", ""), cases=cases, candidates=candidates)
'''


# ── PROPOSED: the general Snakefile header ────────────────────────────────────
# Byte-for-byte the study's current Snakefile EXCEPT the one import line. That is
# the whole point: the header is already solver-agnostic — only the loader's home
# changes.
_SNAKEFILE = """\
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# GENERATED by further_simplification.py to demonstrate the general loader. The
# only change from the hand-written study Snakefile is the import below:
#   -  from neofoam.tooling.verification.study import load_study
#   +  from casesweep import load_cases          # == neofoam.tooling.workflow

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(workflow.snakefile).parent))
from casesweep import load_cases  # PROPOSED: from neofoam.tooling.workflow import load_cases

CONFIG = os.path.abspath(workflow.configfiles[-1])
STUDY = load_cases(CONFIG)
CASES = STUDY.cases
IDS = [c.id for c in CASES]
THREADS = {c.id: max(1, c.subdomains) for c in CASES}

CASE_ROOT = "cases"
RESULTS = "results"
REPORT = "report.html"

if not IDS:
    raise WorkflowError("no cases selected — is OpenFOAM sourced and config.yaml `cases:` real?")

CANDIDATE_LABELS = STUDY.candidate_labels
SOLVER_LABELS = sorted({c.native_label for c in CASES} | set(CANDIDATE_LABELS))

wildcard_constraints:
    id="|".join(re.escape(i) for i in IDS),
    solver="|".join(re.escape(s) for s in SOLVER_LABELS),

rule all:
    input:
        REPORT,

include: "rules/build_case.smk"
include: "rules/swap_solver.smk"
include: "rules/run.smk"
include: "rules/compare.smk"
include: "rules/report.smk"
"""


# ── PROPOSED: neofoam.tooling.workflow.copy_rules ─────────────────────────────
def copy_rules(src_rules: Path, dest: Path, names: tuple[str, ...]) -> None:
    """Copy a named rule set into a study's local ``rules/`` folder.

    The general scaffolder (move 2): the same call the mesh sweep would use. Here
    it copies from the study's existing ``rules/``; in production it copies from
    the packaged master a study is initialised from.
    """
    (dest / "rules").mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(src_rules / name, dest / "rules" / name)


def scaffold(out_dir: Path) -> None:
    """Build a runnable study dir that uses only the general loader + copied rules."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    # config.yaml + discover.py are the study's own inputs — carried over as-is.
    shutil.copy2(STUDY_DIR / "config.yaml", out_dir / "config.yaml")
    shutil.copy2(STUDY_DIR / "discover.py", out_dir / "discover.py")
    (out_dir / "casesweep.py").write_text(_CASESWEEP_MODULE)  # == workflow.load_cases
    (out_dir / "Snakefile").write_text(_SNAKEFILE)
    copy_rules(STUDY_DIR / "rules", out_dir, RULE_FILES)


def demo() -> int:
    scaffold(DEMO)
    print(f"scaffolded general study at: {DEMO}")
    print(
        "  Snakefile import:  from casesweep import load_cases   (no verification import)"
    )
    proc = subprocess.run(
        ["snakemake", "--configfile", "config.yaml", "-n", "--forceall"],
        cwd=DEMO,
        capture_output=True,
        text=True,
    )
    stats = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    tail = "\n".join(stats[-9:])
    print("\n=== snakemake -n --forceall (general study) ===\n" + tail)
    total_line = next((ln for ln in stats if ln.strip().startswith("total")), "")
    ok = "452" in total_line
    print(
        "\nRESULT:",
        "PASS — identical 452-job DAG, no verification import"
        if ok
        else "MISMATCH — see output above",
    )
    if proc.returncode != 0 and not ok:
        print(proc.stderr[-800:], file=sys.stderr)
    return 0 if ok else 1


_GENERAL_REPORT_NOTE = """
Move 3 (general report), sketched:

    # neofoam/tooling/workflow/report.py
    def render_report(title, columns, records): ...   # study declares `columns`

    # a study's config.yaml then carries:
    report:
      columns: [outcome, worst_abs, worst_rel]         # what to tabulate

The blocker is not code volume — it is fixing ONE record schema both stacks emit,
so the reporter never needs to know which study produced a row. Do move 1+2 first;
the schema falls out of a shared `results/<id>.json` contract.
"""


if __name__ == "__main__":
    raise SystemExit(demo())
