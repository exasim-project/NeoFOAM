#!/usr/bin/env python
"""Best-effort static **call graph** + **module dependency graph** via ``ast``.

Companion to ``coverage_contexts.py`` (which answers the *dynamic* question —
"which test *executed* which line"). This script answers the *static* one:

  * **Module dependency graph** — file → file import edges, scoped to a package,
    each edge annotated with the symbols it carries. This is the authoritative
    view of "how the files connect" and the one to read for dependency-direction
    questions (frontend → library, never back).
  * **Call graph** — best-effort function/method → function/method edges resolved
    across files. Python is dynamically dispatched, so resolution is approximate;
    the report states a **resolution confidence** (resolved / total call sites) so
    you know how much to trust it. High-confidence edges are the useful ones.
  * **Test reachability** — from each ``test_*`` function, the set of *source*
    functions statically reachable through the call graph. The static complement
    to coverage: coverage says a line *ran*; reachability says a test *can reach*
    a function even on a path this run didn't take.

Stdlib only (``ast``) — needs NO installed package and never imports the target
code, so it is safe on code with native extensions / heavy import side effects.

    python .claude/skills/understand-tests/call_graph.py \
        --source   src/neofoam/ui \
        --tests    test/ui \
        --pkg-root src \
        --scope    neofoam \
        --out      /tmp/understand/ui

``--pkg-root`` is the directory that dotted module names are relative to (``src``
→ ``src/neofoam/ui/app.py`` is ``neofoam.ui.app``). ``--scope`` keeps only edges
whose *target* module starts with that prefix (drop stdlib/3rd-party noise); omit
to keep everything. ``--tests`` is optional — without it you still get the source
call + dependency graphs, just no test-reachability section.

Outputs (under --out):
  * callgraph.json  — nodes, call edges (resolved+unresolved), module edges with
                      carried symbols, per-test reachability, confidence  (handoff)
  * callgraph.md    — readable summary + Mermaid module graph + reachability table
  * modules.dot     — module dependency graph (Graphviz)  ->  dot -Tsvg
  * callgraph.dot   — function-level call graph (Graphviz; capped, may be large)
Prints a concise human summary to stdout.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Definition:
    """A function or method definition. ``qualname`` is ``module:Class.method``
    (or ``module:function`` at module level) and is the call-graph node id."""

    module: str
    qualname: str
    kind: str  # "function" | "method"
    lineno: int
    end_lineno: int
    file: str
    class_name: str | None
    is_test: bool


@dataclass
class ImportRef:
    """A name bound by an import, and what it points at."""

    module: str  # dotted module the name resolves into
    symbol: (
        str | None
    )  # symbol name if `from mod import symbol`, else None (a module alias)


@dataclass
class ModuleInfo:
    dotted: str
    path: Path
    is_test: bool
    tree: ast.Module
    imports: dict[str, ImportRef] = field(default_factory=dict)
    # module -> set of symbols imported from it (for the dependency graph)
    dep_symbols: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _split(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values or []:
        out.extend(p.strip() for p in v.split(",") if p.strip())
    return out


def _iter_py(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.rglob("*.py")))
        elif pp.suffix == ".py":
            files.append(pp)
    # drop caches / duplicates, keep order stable
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in files:
        rf = f.resolve()
        if "__pycache__" in rf.parts or rf in seen:
            continue
        seen.add(rf)
        uniq.append(f)
    return uniq


def _package_root(path: Path) -> Path:
    """The first ancestor directory that is NOT itself a package (has no
    ``__init__.py``) — the root that dotted names are relative to. This is the
    real Python import-root, so source (``src/``) and tests (``test/``) each get
    the right root without the caller specifying one."""
    parent = path.resolve().parent
    while (parent / "__init__.py").exists() and parent.parent != parent:
        parent = parent.parent
    return parent


def _dotted(path: Path, pkg_root: Path | None) -> str:
    """Dotted module name of ``path``. Rooted at ``pkg_root`` if it is an
    ancestor, otherwise auto-detected from ``__init__.py`` ancestry."""
    resolved = path.resolve()
    root = pkg_root.resolve() if pkg_root else None
    if root is None or root not in resolved.parents:
        root = _package_root(path)
    rel = resolved.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_test_path(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py") or "test" in path.parts


def _is_test_func(name: str) -> bool:
    return name.startswith("test_") or name.startswith("test")


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
def _resolve_relative(current: str, node: ast.ImportFrom) -> str:
    """Resolve ``from . import x`` / ``from ..a import y`` to an absolute module."""
    if node.level == 0:
        return node.module or ""
    base = current.split(".")
    # a module `a.b.c` at level 1 lives in package `a.b`; strip (level) trailing parts.
    base = base[: max(0, len(base) - node.level)]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def collect_imports(mod: ModuleInfo) -> None:
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                target = alias.name if alias.asname else alias.name.split(".")[0]
                mod.imports[bound] = ImportRef(module=target, symbol=None)
                mod.dep_symbols[alias.name].add("<module>")
        elif isinstance(node, ast.ImportFrom):
            target_mod = _resolve_relative(mod.dotted, node)
            for alias in node.names:
                bound = alias.asname or alias.name
                mod.imports[bound] = ImportRef(module=target_mod, symbol=alias.name)
                mod.dep_symbols[target_mod].add(alias.name)


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
def collect_definitions(mod: ModuleInfo) -> list[Definition]:
    defs: list[Definition] = []

    def visit(node: ast.AST, class_name: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if class_name:
                    qual = f"{mod.dotted}:{class_name}.{child.name}"
                    kind = "method"
                else:
                    qual = f"{mod.dotted}:{child.name}"
                    kind = "function"
                defs.append(
                    Definition(
                        module=mod.dotted,
                        qualname=qual,
                        kind=kind,
                        lineno=child.lineno,
                        end_lineno=child.end_lineno or child.lineno,
                        file=str(mod.path),
                        class_name=class_name,
                        is_test=mod.is_test
                        and class_name is None
                        and _is_test_func(child.name),
                    )
                )
                # nested functions: descend but keep class context None
                visit(child, None)

    visit(mod.tree, None)
    return defs


# --------------------------------------------------------------------------- #
# Call resolution (best effort)
# --------------------------------------------------------------------------- #
def _attr_parts(node: ast.expr) -> list[str] | None:
    """Flatten ``a.b.c`` -> ['a','b','c']; None if the head isn't a plain Name
    (e.g. ``foo().bar`` — a dynamic receiver we cannot resolve statically)."""
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


class CallResolver:
    """Best-effort call resolution, biased to **neofoam-relevant** calls only.

    ``resolve`` returns ``(callee_qualname_or_None, tail, category)`` where
    ``category`` is:
      * ``"resolved"``  — bound to a known in-scope def (``callee`` is set);
      * ``"internal"``  — targets neofoam (name came from an in-scope import or a
                          local def) but the exact def couldn't be pinned;
      * ``"external"``  — a builtin / third-party / dynamic call we don't track.
    External calls are dropped by the caller so the graph and the resolution
    confidence are about neofoam functions, not trame/pydantic/stdlib noise.
    """

    def __init__(
        self,
        mods: dict[str, ModuleInfo],
        defindex: set[str],
        classes: dict[str, set[str]],
        in_scope,
    ):
        self.mods = mods
        self.defindex = defindex  # every known qualname
        self.classes = classes  # module -> set of class names defined there
        self.in_scope = in_scope  # module dotted name -> bool (is it neofoam?)

    def resolve(
        self, caller_mod: str, class_ctx: str | None, call: ast.Call
    ) -> tuple[str | None, str, str]:
        parts = _attr_parts(call.func)
        if not parts:
            return None, "<dynamic>", "external"
        tail = parts[-1]
        mod = self.mods.get(caller_mod)
        imports = mod.imports if mod else {}
        head = parts[0]

        def internal() -> tuple[str | None, str, str]:
            """Unresolved but neofoam-related iff the head names an in-scope
            import or a local def/class in this (in-scope) module."""
            if head in imports and self.in_scope(imports[head].module):
                return None, tail, "internal"
            if self.in_scope(caller_mod) and (
                head in self.classes.get(caller_mod, set())
                or f"{caller_mod}:{head}" in self.defindex
            ):
                return None, tail, "internal"
            # self.<attr>() where <attr> is not a known method: the receiver's type
            # is unknown (often a trame widget stored on self) -> treat as external.
            return None, tail, "external"

        # self.method() inside a class
        if parts[0] == "self" and class_ctx and len(parts) >= 2:
            cand = f"{caller_mod}:{class_ctx}.{parts[1]}"
            if cand in self.defindex:
                return cand, tail, "resolved"
            # a method of this neofoam class we couldn't index (inherited?) -> internal
            return (
                (None, tail, "internal")
                if self.in_scope(caller_mod)
                else (None, tail, "external")
            )

        # imported name
        if head in imports:
            ref = imports[head]
            if ref.symbol is None:
                # module alias: `import a.b as x`; x.rest.func -> module a.b(.rest), symbol func
                dotted = ref.module
                rest = parts[1:]
                if len(rest) >= 2:
                    dotted = dotted + "." + ".".join(rest[:-1])
                sym = rest[-1] if rest else None
                if sym is not None:
                    cand = f"{dotted}:{sym}"
                    if cand in self.defindex:
                        return cand, tail, "resolved"
                    if sym in self.classes.get(dotted, set()):
                        return f"{dotted}:{sym}.__init__", tail, "resolved"
                return internal()
            else:
                # `from a.b import c`: c may be a function, a class, or a submodule
                dotted, sym = ref.module, ref.symbol
                if len(parts) == 1:
                    cand = f"{dotted}:{sym}"
                    if cand in self.defindex:
                        return cand, tail, "resolved"
                    if sym in self.classes.get(dotted, set()):
                        return f"{dotted}:{sym}.__init__", tail, "resolved"
                    return internal()
                # c.method(...) — treat c as a class or module
                cand_method = f"{dotted}:{sym}.{parts[1]}"
                if cand_method in self.defindex:
                    return cand_method, tail, "resolved"
                cand_submod = f"{dotted}.{sym}:{parts[1]}"
                if cand_submod in self.defindex:
                    return cand_submod, tail, "resolved"
                return internal()

        # local definition in the caller's own module
        local_fn = f"{caller_mod}:{head}"
        if local_fn in self.defindex and len(parts) == 1:
            return local_fn, tail, "resolved"
        if head in self.classes.get(caller_mod, set()):
            if len(parts) == 1:
                return f"{caller_mod}:{head}.__init__", tail, "resolved"
            cand = f"{caller_mod}:{head}.{parts[1]}"
            if cand in self.defindex:
                return cand, tail, "resolved"
            return internal()

        return internal()


# --------------------------------------------------------------------------- #
# Analysis driver
# --------------------------------------------------------------------------- #
def analyze(
    source_files: list[Path],
    test_files: list[Path],
    pkg_root: Path,
    scope: str | None,
) -> dict:
    mods: dict[str, ModuleInfo] = {}
    for path, is_test in [(p, False) for p in source_files] + [
        (p, True) for p in test_files
    ]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"  ! skip {path}: {exc}", file=sys.stderr)
            continue
        dotted = _dotted(path, pkg_root)
        mods[dotted] = ModuleInfo(dotted=dotted, path=path, is_test=is_test, tree=tree)

    all_defs: list[Definition] = []
    classes: dict[str, set[str]] = defaultdict(set)
    for mod in mods.values():
        collect_imports(mod)
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.ClassDef):
                classes[mod.dotted].add(node.name)
        all_defs.extend(collect_definitions(mod))

    defindex = {d.qualname for d in all_defs}
    def_by_qual = {d.qualname: d for d in all_defs}
    source_mods = {m.dotted for m in mods.values() if not m.is_test}

    def in_scope(m: str) -> bool:
        return scope is None or m == scope or m.startswith(scope + ".")

    resolver = CallResolver(mods, defindex, classes, in_scope)

    # ---- call edges (neofoam-relevant only; external calls dropped) -------- #
    call_edges: list[dict] = []
    unresolved: dict[str, int] = defaultdict(int)  # tail name -> count (internal only)
    external_count = [0]
    for mod in mods.values():
        _walk_calls(mod, resolver, call_edges, unresolved, external_count)

    # Confidence is measured over neofoam-relevant call sites only (resolved +
    # internal-unresolved); external builtin/trame/pydantic calls are excluded.
    total_sites = len(call_edges)
    resolved_sites = sum(1 for e in call_edges if e["callee"] is not None)
    confidence = round(100 * resolved_sites / max(1, total_sites), 1)

    # ---- module dependency edges (from imports; authoritative) ------------ #
    module_edges: list[dict] = []
    for mod in mods.values():
        for target, symbols in sorted(mod.dep_symbols.items()):
            if not in_scope(target) or target == mod.dotted:
                continue
            module_edges.append(
                {
                    "src": mod.dotted,
                    "dst": target,
                    "src_is_test": mod.is_test,
                    "symbols": sorted(symbols),
                }
            )

    # ---- test reachability (BFS over resolved call edges) ----------------- #
    adj: dict[str, set[str]] = defaultdict(set)
    for e in call_edges:
        if e["callee"] is not None:
            adj[e["caller"]].add(e["callee"])
    test_reach: dict[str, list[str]] = {}
    for d in all_defs:
        if not d.is_test:
            continue
        seen: set[str] = set()
        stack = [d.qualname]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reached_src = sorted(q for q in seen if q.split(":", 1)[0] in source_mods)
        test_reach[d.qualname] = reached_src

    # source functions no test statically reaches (candidates for "untested by design")
    reached_any = {q for reach in test_reach.values() for q in reach}
    source_fns = [
        d.qualname
        for d in all_defs
        if d.module in source_mods and not d.qualname.endswith(".__init__")
    ]
    unreached = (
        sorted(q for q in source_fns if q not in reached_any) if test_files else []
    )

    # ---- test stratification (altitude) ----------------------------------- #
    # Classify each test by how many SOURCE MODULES it can reach: a proxy for
    # e2e/integration vs unit. Reachability measures *touch surface* (a test that
    # boots build_app "reaches" everything it fans out to) — a coarse altitude,
    # not assertion focus; the dynamic coverage pass refines it.
    LEVELS = ["isolated", "unit", "component", "integration"]
    RANK = {lv: i for i, lv in enumerate(LEVELS)}

    def _level(n_mods: int) -> str:
        return (
            LEVELS[0]
            if n_mods == 0
            else LEVELS[1]
            if n_mods == 1
            else LEVELS[2]
            if n_mods <= 3
            else LEVELS[3]
        )

    strata: list[dict] = []
    for d in all_defs:
        if not d.is_test:
            continue
        reach = test_reach[d.qualname]
        mods_hit = {q.split(":", 1)[0] for q in reach}  # already source-only
        strata.append(
            {
                "test": d.qualname,
                "level": _level(len(mods_hit)),
                "src_fns": len(reach),
                "src_modules": len(mods_hit),
            }
        )
    strata.sort(key=lambda s: (-RANK[s["level"]], -s["src_modules"], -s["src_fns"]))
    level_hist = {
        lv: sum(1 for s in strata if s["level"] == lv) for lv in reversed(LEVELS)
    }

    # ---- feature coverage: source API surface × best covering test level -- #
    # in-degree over SOURCE callees only (test-module helpers excluded), and for
    # each public source fn the *highest* test level that statically reaches it.
    indeg: dict[str, int] = defaultdict(int)
    for e in call_edges:
        if e["callee"] is not None and e["callee"].split(":", 1)[0] in source_mods:
            indeg[e["callee"]] += 1
    best_level: dict[str, str] = {}
    for s in strata:
        for q in test_reach[s["test"]]:
            if q not in best_level or RANK[s["level"]] > RANK[best_level[q]]:
                best_level[q] = s["level"]

    def _is_public(qual: str) -> bool:
        sym = qual.split(":", 1)[1] if ":" in qual else qual
        last = sym.split(".")[-1]
        return not last.startswith("_")

    feature_coverage = sorted(
        (
            {
                "fn": d.qualname,
                "in_degree": indeg.get(d.qualname, 0),
                "best_test_level": best_level.get(d.qualname, "GAP"),
            }
            for d in all_defs
            if d.module in source_mods and _is_public(d.qualname)
        ),
        key=lambda r: (-r["in_degree"], r["best_test_level"] == "GAP"),
    )
    api_total = len(feature_coverage)
    api_gap = sum(1 for r in feature_coverage if r["best_test_level"] == "GAP")

    return {
        "summary": {
            "modules": len(mods),
            "source_modules": len(source_mods),
            "test_modules": len(mods) - len(source_mods),
            "definitions": len(all_defs),
            "neofoam_call_sites": total_sites,
            "resolved_call_sites": resolved_sites,
            "resolution_confidence_pct": confidence,
            "external_call_sites_dropped": external_count[0],
            "module_edges": len(module_edges),
            "test_level_histogram": level_hist,
            "public_api_fns": api_total,
            "public_api_static_gaps": api_gap,
            "scope": scope,
        },
        "definitions": [d.__dict__ for d in all_defs],
        "call_edges": call_edges,
        "module_edges": module_edges,
        "top_unresolved": sorted(unresolved.items(), key=lambda kv: -kv[1])[:25],
        "test_reachability": test_reach,
        "test_strata": strata,
        "feature_coverage": feature_coverage,
        "statically_unreached_source_fns": unreached,
        "_def_by_qual": {q: d.__dict__ for q, d in def_by_qual.items()},
    }


def _walk_calls(mod, resolver, call_edges, unresolved, external_count):
    """Attribute every Call to its enclosing def and resolve it. Keeps only
    neofoam-relevant calls: ``resolved`` (bound to a known def) and ``internal``
    (targets neofoam but unpinned). ``external`` calls are counted and dropped."""

    def visit(node, caller_qual, class_ctx):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, caller_qual, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if class_ctx:
                    qual = f"{mod.dotted}:{class_ctx}.{child.name}"
                else:
                    qual = f"{mod.dotted}:{child.name}"
                visit(child, qual, None)  # nested defs lose class context
            elif isinstance(child, ast.Call) and caller_qual is not None:
                callee, tail, category = resolver.resolve(mod.dotted, class_ctx, child)
                if category == "external":
                    external_count[0] += 1
                elif category == "resolved":
                    call_edges.append(
                        {
                            "caller": caller_qual,
                            "callee": callee,
                            "line": child.lineno,
                            "tail": tail,
                        }
                    )
                else:  # internal — neofoam-related but not pinned to a def
                    unresolved[tail] += 1
                    call_edges.append(
                        {
                            "caller": caller_qual,
                            "callee": None,
                            "line": child.lineno,
                            "tail": tail,
                        }
                    )
                visit(child, caller_qual, class_ctx)  # args may contain calls
            else:
                visit(child, caller_qual, class_ctx)

    visit(mod.tree, None, None)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _short(qual: str) -> str:
    mod, _, sym = qual.partition(":")
    return f"{mod.split('.')[-1]}:{sym}" if sym else mod.split(".")[-1]


def write_dot_modules(report: dict, out: Path) -> None:
    lines = [
        "digraph modules {",
        "  rankdir=LR;",
        "  node [shape=box, fontname=Helvetica];",
    ]
    nodes = set()
    for e in report["module_edges"]:
        nodes.add(e["src"])
        nodes.add(e["dst"])
    for n in sorted(nodes):
        lines.append(f'  "{n}";')
    for e in report["module_edges"]:
        style = " [style=dashed]" if e["src_is_test"] else ""
        lines.append(f'  "{e["src"]}" -> "{e["dst"]}"{style};')
    lines.append("}")
    out.write_text("\n".join(lines) + "\n")


def write_dot_calls(report: dict, out: Path, cap: int = 600) -> None:
    lines = [
        "digraph calls {",
        "  rankdir=LR;",
        "  node [shape=box, fontname=Helvetica, fontsize=9];",
    ]
    n = 0
    for e in report["call_edges"]:
        if e["callee"] is None:
            continue
        lines.append(f'  "{_short(e["caller"])}" -> "{_short(e["callee"])}";')
        n += 1
        if n >= cap:
            lines.append(f"  // truncated at {cap} edges")
            break
    lines.append("}")
    out.write_text("\n".join(lines) + "\n")


def write_markdown(report: dict, out: Path) -> None:
    s = report["summary"]
    md = [
        "# Call graph & module dependencies",
        "",
        f"**{s['source_modules']} source + {s['test_modules']} test modules**, "
        f"{s['definitions']} defs, {s['neofoam_call_sites']} neofoam call sites · "
        f"**resolution confidence {s['resolution_confidence_pct']}%** "
        f"({s['resolved_call_sites']}/{s['neofoam_call_sites']} resolved) · "
        f"{s['external_call_sites_dropped']} external calls dropped.",
        "",
        "> Static best-effort, **neofoam-scoped**: builtin / trame / pydantic / stdlib "
        "calls are excluded, so the confidence is the share of *neofoam-targeted* call "
        "sites the AST could bind to a known def — the rest are neofoam calls through "
        "dynamic dispatch it couldn't pin. Read module edges as authoritative (they "
        "come from imports); read call edges as strong hints weighted by the confidence.",
        "",
        "## Module dependency graph",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    ids: dict[str, str] = {}

    def nid(m: str) -> str:
        return ids.setdefault(m, f"m{len(ids)}")

    for e in report["module_edges"]:
        a, b = nid(e["src"]), nid(e["dst"])
        arrow = "-.->" if e["src_is_test"] else "-->"
        md.append(f'    {a}["{e["src"]}"] {arrow} {b}["{e["dst"]}"]')
    md += [
        "```",
        "",
        "## Module edges (import-derived, authoritative)",
        "",
        "| src | → | dst | test? | symbols |",
        "|---|---|---|---|---|",
    ]
    for e in report["module_edges"]:
        md.append(
            f"| `{e['src']}` | → | `{e['dst']}` | {'test' if e['src_is_test'] else ''} "
            f"| {', '.join('`' + x + '`' for x in e['symbols'][:8])} |"
        )
    if report["top_unresolved"]:
        md += [
            "",
            "## Top unresolved call names (dynamic / external)",
            "",
            "| name | sites |",
            "|---|---|",
        ]
        for name, cnt in report["top_unresolved"]:
            md.append(f"| `{name}` | {cnt} |")
    if report.get("test_strata"):
        hist = report["summary"]["test_level_histogram"]
        md += [
            "",
            "## Test stratification — read top-down (e2e → unit)",
            "",
            "Each test's **altitude** = how many source *modules* it can reach "
            "(`integration` ≥4 · `component` 2–3 · `unit` 1 · `isolated` 0). This "
            "is *touch surface*, not assertion focus — a test that boots the whole "
            "app reaches everything it fans out to. Read the top rows first to grasp "
            "what the suite claims the system does end-to-end.",
            "",
            "Levels: " + " · ".join(f"**{k}**={v}" for k, v in hist.items()),
            "",
            "| level | test | src modules | src fns |",
            "|---|---|---|---|",
        ]
        for s in report["test_strata"]:
            md.append(
                f"| {s['level']} | `{_short(s['test'])}` | {s['src_modules']} | {s['src_fns']} |"
            )
    if report.get("feature_coverage"):
        api = report["summary"]
        md += [
            "",
            "## Feature coverage — is each important feature proven?",
            "",
            f"Public source functions ({api['public_api_fns']} total, "
            f"**{api['public_api_static_gaps']} with no test statically reaching "
            "them**), ranked by in-degree (how many source calls depend on them — a "
            "centrality proxy for 'important'). `best_test_level` is the *highest* "
            "test altitude that reaches the fn; **GAP** = no test reaches it "
            "statically. Confirm GAPs against the dynamic pass before acting.",
            "",
            "| in-deg | fn | best test level |",
            "|---|---|---|",
        ]
        for r in report["feature_coverage"][:40]:
            flag = "**GAP**" if r["best_test_level"] == "GAP" else r["best_test_level"]
            md.append(
                f"| {r['in_degree']} | `{r['fn'].split('neofoam.')[-1]}` | {flag} |"
            )
    if report["statically_unreached_source_fns"]:
        md += [
            "",
            "## Source functions no test statically reaches",
            "",
            "Candidates for missing coverage (or dead code). Confirm against the "
            "dynamic coverage pass before concluding.",
            "",
        ]
        for q in report["statically_unreached_source_fns"]:
            md.append(f"- `{q}`")
    out.write_text("\n".join(md) + "\n")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        required=True,
        help="Source dir(s) or file(s) to analyze (repeatable / comma-sep).",
    )
    ap.add_argument(
        "--tests",
        action="append",
        default=[],
        help="Test dir(s) or file(s) (optional; enables reachability).",
    )
    ap.add_argument(
        "--pkg-root",
        default=None,
        help="Optional override for the dotted-name root (e.g. src). "
        "By default each file's root is auto-detected from its "
        "__init__.py ancestry, which handles src/ and test/ alike.",
    )
    ap.add_argument(
        "--scope",
        default=None,
        help="Keep only edges whose target module starts with this "
        "prefix (e.g. neofoam). Omit to keep all.",
    )
    ap.add_argument(
        "--out", required=True, help="Output directory (created if missing)."
    )
    args = ap.parse_args()

    pkg_root = Path(args.pkg_root) if args.pkg_root else None
    source_files = _iter_py(_split(args.source))
    test_files = _iter_py(_split(args.tests))
    if not source_files:
        print("No source .py files found.", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = analyze(source_files, test_files, pkg_root, args.scope)
    report.pop("_def_by_qual", None)  # internal only

    (out / "callgraph.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, out / "callgraph.md")
    write_dot_modules(report, out / "modules.dot")
    write_dot_calls(report, out / "callgraph.dot")

    s = report["summary"]
    print(f"\n==== CALL GRAPH ({', '.join(args.source)}) ====")
    print(
        f"  {s['source_modules']} source + {s['test_modules']} test modules · "
        f"{s['definitions']} defs · {s['neofoam_call_sites']} neofoam call sites "
        f"({s['external_call_sites_dropped']} external dropped)"
    )
    print(
        f"  resolution confidence {s['resolution_confidence_pct']}% "
        f"({s['resolved_call_sites']}/{s['neofoam_call_sites']})"
    )
    print(f"  {s['module_edges']} in-scope module dependency edges")
    if report.get("test_strata"):
        hist = s["test_level_histogram"]
        print("  test levels: " + " · ".join(f"{k}={v}" for k, v in hist.items()))
        print(
            f"  public API: {s['public_api_fns']} fns, "
            f"{s['public_api_static_gaps']} with NO test statically reaching them"
        )
    if report["statically_unreached_source_fns"]:
        print(
            f"  {len(report['statically_unreached_source_fns'])} source fns no test statically reaches"
        )
    print(
        f"\nWrote {out / 'callgraph.json'}, {out / 'callgraph.md'}, "
        f"{out / 'modules.dot'}, {out / 'callgraph.dot'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
