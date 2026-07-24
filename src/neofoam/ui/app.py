# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Trame case wizard: sidebar step nav + JSONForms panels + model-aware save.

The left drawer lists the wizard steps (``current_step``); the main area mounts every
step once, shown via ``v_show`` so navigation never unmounts a form and wipes its
state. Every form is rendered generically from its :class:`~neofoam.ui.forms.FormEntry`
JSON Schema — there is no per-case or per-config markup. The Models step opens with
the model-selection panel (``sel_<model>``); a config panel owned by an unselected
optional model is hidden and skipped on save. Save aggregates the live form state
through :func:`neofoam.ui.case_spec.state_to_case_spec` and writes via
:func:`neofoam.mcp.tools.save_case`.
"""

from __future__ import annotations

import asyncio
import io
import os
import shlex
import shutil
import signal
import tempfile
import zipfile
from typing import Any

from dataclasses import asdict
from pathlib import Path

from neofoam.agent.case_fill import case_spec_to_configs, load_case_from_disk
from neofoam.mcp import tools
from neofoam.mcp.registry import resolve_solver
from neofoam.ui import case_spec as cs
from neofoam.ui import jsonforms_module
from neofoam.ui.agent_panel import build_agent_panel
from neofoam.ui.forms import (
    FormEntry,
    build_forms,
    patch_bc_schema,
    seed_boundary_field,
)
from neofoam.ui.geometry import (
    GeometrySpec,
    MeshSettings,
    PatchGeometry,
    PatchRole,
    discover_geometry,
    write_mesh_configs,
)
from neofoam.ui import review
from neofoam.ui.plugins import StepContext, StepPlugin, discover_step_plugins
from neofoam.ui.review import findings_to_rows
from neofoam.ui.scaffold import scaffold_runnable_case
from neofoam.ui.steps import Step, build_model_choices, build_steps
from neofoam.ui.sweep_panel import SweepPanel

_ROLE_NAMES = [r.value for r in PatchRole]

# Optional pre-fill for the geometry step's STL folder (manual-testing convenience).
_DEFAULT_STL_DIR = os.environ.get("NEOFOAM_WIZARD_STL_DIR", "")

# Per-step presentation (icon + one-line caption under the step title). Purely
# cosmetic — steps themselves come from ``neofoam.ui.steps.build_steps``.
_STEP_ICONS = {
    "models": "mdi-atom",
    "geometry": "mdi-cube-outline",
    "bcs": "mdi-border-all-variant",
    "initial": "mdi-waves",
    "schemes": "mdi-function-variant",
    "sweep": "mdi-tune-variant",
    "review": "mdi-clipboard-check-outline",
}
_STEP_CAPTIONS = {
    "models": "Pick the optional physics models, then configure the model dictionaries.",
    "geometry": "Scan the boundary STLs, assign patch roles and author the mesh dicts.",
    "bcs": "Boundary conditions per field — patches are seeded by the geometry scan.",
    "initial": "Physical dimensions and initial internal value per field.",
    "schemes": "Discretisation schemes and linear solvers — the defaults are sensible.",
    "sweep": "Sweep any config over named variants and export a Snakemake workflow.",
    "review": "Save the case, review the validation findings and run it.",
}

# One coherent look for every (schema-generated) widget: an app theme plus global
# component defaults, so the generic forms need no per-widget styling.
_VUETIFY_CONFIG = {
    "theme": {
        "defaultTheme": "neofoam",
        "themes": {
            "neofoam": {
                "dark": False,
                "colors": {
                    "primary": "#1F5FBF",
                    "secondary": "#00838F",
                    "background": "#F4F6FB",
                    "surface": "#FFFFFF",
                    "surface-variant": "#E9EDF5",
                    "error": "#C62828",
                    "warning": "#E65100",
                    "success": "#2E7D32",
                    "info": "#0277BD",
                },
            },
        },
    },
    "defaults": {
        "VTextField": {"density": "compact", "variant": "outlined", "color": "primary"},
        "VSelect": {"density": "compact", "variant": "outlined", "color": "primary"},
        "VSwitch": {"density": "compact", "color": "primary", "hideDetails": True},
        "VBtn": {"rounded": "lg"},
        "VCard": {"rounded": "lg"},
        "VAlert": {"density": "compact", "rounded": "lg"},
        "VExpansionPanels": {"multiple": True, "variant": "accordion"},
        "VTooltip": {"location": "bottom"},
    },
}

# Light global polish on top of the theme (panel borders, calmer form spacing).
_CSS = """
.v-expansion-panel { border: 1px solid rgba(31, 95, 191, 0.12); }
.v-expansion-panel-title { font-weight: 500; min-height: 44px; }
.v-expansion-panel--active > .v-expansion-panel-title { color: rgb(31, 95, 191); }
.nf-step-title { letter-spacing: -0.3px; }
.v-navigation-drawer .v-list-item-title { font-weight: 500; }
"""


def _case_dir_for(stl_dir: str) -> str:
    """The case dir owning ``stl_dir`` (its ``constant/triSurface`` parent), else itself."""
    p = Path(stl_dir)
    if p.name == "triSurface" and p.parent.name == "constant":
        return str(p.parent.parent)
    return stl_dir


def _case_root(extracted: Path) -> Path:
    """The OpenFOAM case dir within an extracted zip (``system``/``constant`` owner).

    A case zip is often a single top-level folder wrapping the case (e.g.
    ``cavity/system/...``); unwrap one level when the extraction root itself isn't
    the case dir. Skips ``__MACOSX`` / dotfile noise from Finder-made zips when
    looking for that wrapper folder.
    """
    if (extracted / "system").is_dir() or (extracted / "constant").is_dir():
        return extracted
    for p in sorted(extracted.iterdir()):
        if not p.is_dir() or p.name == "__MACOSX" or p.name.startswith("."):
            continue
        if (p / "system").is_dir() or (p / "constant").is_dir():
            return p
    return extracted


def _upload_content(upload: Any) -> bytes:
    """The raw bytes of a trame ``VFileInput`` upload (a ``{"content": bytes, ...}`` dict)."""
    if isinstance(upload, (bytes, bytearray)):
        return bytes(upload)
    if isinstance(upload, dict) and isinstance(
        upload.get("content"), (bytes, bytearray)
    ):
        return bytes(upload["content"])
    raise ValueError(f"unexpected file-upload payload: {upload!r}")


def _dyld_reexport_preamble() -> str:
    """Shell snippet that re-``export``s this process's ``DYLD_*`` vars.

    macOS strips ``DYLD_*`` environment variables across an ``exec`` performed
    from a running Python process (confirmed empirically — a child inherits
    everything except these), which breaks ``pybFoam``'s dylib loading for any
    subprocess we launch (e.g. ``Allrun``). A shell that re-exports them from a
    literal value *before* exec'ing further is unaffected — the strip only
    applies to the hop directly from Python.
    """
    dyld_vars = {k: v for k, v in os.environ.items() if k.startswith("DYLD_")}
    # The compiled bin/ launchers (e.g. neoIcoFoam) link against this package's own
    # lib/ (libNeoFOAM/libNeoN/...), which isn't on the server's own
    # DYLD_LIBRARY_PATH (only pybFoam's OpenFOAM paths are) — add it explicitly.
    own_lib_dir = Path(__file__).resolve().parent.parent / "lib"
    if own_lib_dir.is_dir():
        existing = dyld_vars.get("DYLD_LIBRARY_PATH", "")
        dyld_vars["DYLD_LIBRARY_PATH"] = (
            f"{own_lib_dir}:{existing}" if existing else str(own_lib_dir)
        )
    return "".join(f"export {k}={shlex.quote(v)}; " for k, v in dyld_vars.items())


def _patch_row(patch: PatchGeometry) -> dict[str, Any]:
    """A ``PatchGeometry`` as an editable trame-state row."""
    ref = patch.refinement
    return {
        "name": patch.name,
        "stl": patch.stl,
        "role": patch.role.value,
        "is_snappy": patch.is_snappy_surface,
        "box_faces": list(patch.box_faces) if patch.box_faces else None,
        "faces": ", ".join(patch.box_faces) if patch.box_faces else "snappy surface",
        "refinement_str": f"{ref[0]} {ref[1]}" if ref else "",
    }


def _parse_pair(text: str) -> tuple[int, int] | None:
    """Parse a ``"min max"`` refinement string; ``None`` when unparsable."""
    parts = text.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _parse_vec(text: str) -> tuple[float, float, float] | None:
    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _spec_from_state(state: Any) -> GeometrySpec:
    """Rebuild a ``GeometrySpec`` from the (possibly user-edited) geometry state."""
    patches: list[PatchGeometry] = []
    for row in state.geometry_patches:
        refinement = None
        if row["is_snappy"]:
            refinement = _parse_pair(row.get("refinement_str", "")) or (1, 2)
        patches.append(
            PatchGeometry(
                name=row["name"],
                stl=row["stl"],
                role=PatchRole(row["role"]),
                box_faces=row["box_faces"],
                refinement=refinement,
            )
        )
    bbox_min = tuple(state.geo_bbox[0])
    bbox_max = tuple(state.geo_bbox[1])
    location = _parse_vec(state.geo_location) or tuple(
        0.5 * (bbox_min[i] + bbox_max[i]) for i in range(3)
    )
    return GeometrySpec(
        bbox_min=(bbox_min[0], bbox_min[1], bbox_min[2]),
        bbox_max=(bbox_max[0], bbox_max[1], bbox_max[2]),
        location_in_mesh=(location[0], location[1], location[2]),
        length_scale=state.geo_length_scale,
        patches=patches,
    )


def _schema_key(entry: FormEntry) -> str:
    """A JS-identifier-safe state var name holding this entry's static schema."""
    return "schema_" + entry.key.replace(":", "_")


def build_app(
    server: Any = None,
    *,
    solver_name: str = "incompressibleFluid",
    plugins: list[StepPlugin] | None = None,
) -> Any:
    """Construct the trame Server, state, layout and Save controller. Returns it.

    ``plugins`` overrides step-plugin discovery (a test/embedding seam); by default
    the ``neofoam.ui.steps`` entry points are used. Contributed steps are woven into
    the sidebar and content area after the built-in steps are wired.
    """
    from trame.app import get_server  # type: ignore  # untyped optional 'ui' dep
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout  # type: ignore  # untyped dep
    from trame.widgets import client, html, vuetify3 as v3  # type: ignore  # untyped deps
    from trame_client.widgets.core import AbstractElement  # type: ignore  # untyped dep

    class JsonForms(AbstractElement):  # type: ignore[misc]  # untyped base
        """The bundled ``<json-forms>`` client component (JSONForms + Vuetify)."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__("json-forms", **kwargs)
            self._attr_names += ["schema", "uischema", "data"]
            self._event_names += ["change"]

    solver = resolve_solver(solver_name)
    entries: list[FormEntry] = build_forms(solver)
    step_plugins = discover_step_plugins(plugins)
    steps = build_steps(solver, entries, step_plugins)
    plugin_by_id = {p.id: p for p in step_plugins}
    choices = build_model_choices(solver)
    required = [c for c in choices if c.required]
    optional = [c for c in choices if not c.required]
    by_key = {e.key: e for e in entries}

    server = get_server() if server is None else server
    state, ctrl = server.state, server.controller

    state.current_step = steps[0].id
    state.ai_panel = True  # right AI drawer open by default (foldable)
    state.target_dir = ""
    state.save_report = None
    state.scaffolded = []
    state.validation_ok = None
    state.findings = []
    # Geometry & mesh stage (STL → blockMesh/snappy dicts).
    state.role_names = _ROLE_NAMES
    state.stl_dir = _DEFAULT_STL_DIR
    state.geometry_patches = []
    state.geometry_status = ""
    state.geo_bbox = None
    state.geo_location = ""
    state.geo_length_scale = 0.0
    state.geo_cell_size = 0.0
    state.mesh_written = []
    # Import an existing case from a zip upload.
    state.import_zip_file = None
    state.import_status = ""
    state.import_passthrough = {}  # relative path -> raw text, written verbatim on save
    # ./Allrun launch + live log.
    state.run_status = "idle"  # idle | running | paused | done | failed | stopped
    state.run_paused = False
    state.run_log = ""
    for c in optional:
        state[f"sel_{c.name}"] = False
    for entry in entries:
        state[_schema_key(entry)] = entry.schema
        state[entry.state_key] = dict(entry.defaults)

    # ------------------------------------------------------------------ #
    # Controller                                                         #
    # ------------------------------------------------------------------ #

    def _validate_and_store() -> None:
        report = tools.validate_case(solver, state.target_dir)
        state.validation_ok = report.ok
        state.findings = [asdict(r) for r in findings_to_rows(report)]

    def save_case() -> None:
        selected = {c.name for c in optional if state[f"sel_{c.name}"]}
        form_state = {e.key: dict(state[e.state_key]) for e in entries}
        state.current_step = "review"
        try:
            # state_to_case_spec merges/validates field halves and may itself raise.
            spec = cs.state_to_case_spec(entries, form_state, selected)
            result = tools.save_case(solver, spec, state.target_dir)
        except Exception as exc:  # noqa: BLE001 - a save failure must not crash the UI
            # Incomplete/invalid configs — surface each field instead of crashing.
            state.save_report = {"error": str(exc)}
            state.scaffolded = []
            state.validation_ok = False
            state.findings = [asdict(r) for r in review.save_error_rows(exc)]
            return
        state.save_report = result.model_dump()
        # Write through any imported dict files the wizard couldn't represent
        # (e.g. a classic PISO-format fvSolution) verbatim, after the normal
        # config write above — otherwise their content is just the wizard's own
        # (incompatible) defaults.
        for rel, text in state.import_passthrough.items():
            path = Path(state.target_dir) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        # Make the saved case runnable, then validate it.
        state.scaffolded = [str(p) for p in scaffold_runnable_case(state.target_dir)]
        _validate_and_store()

    def revalidate() -> None:
        _validate_and_store()

    def load_geometry(*, set_target: bool = True) -> None:
        """Read the STLs in ``state.stl_dir`` (an STL folder or case dir) into state.

        ``set_target=False`` skips the target-dir pre-fill (used by ``import_case``,
        where ``stl_dir`` is a throwaway extraction path, not a real case location).
        """
        try:
            spec = discover_geometry(state.stl_dir)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.geometry_patches = []
            state.geo_bbox = None
            state.geometry_status = f"Could not read geometry: {exc}"
            return
        # Pre-fill the case target from the STL folder if not already set.
        if set_target and not state.target_dir:
            state.target_dir = _case_dir_for(state.stl_dir)
        patch_rows = [_patch_row(p) for p in spec.patches]
        state.geometry_patches = patch_rows
        state.geo_bbox = [list(spec.bbox_min), list(spec.bbox_max)]
        state.geo_location = " ".join(f"{x:g}" for x in spec.location_in_mesh)
        state.geo_length_scale = spec.length_scale
        state.geo_cell_size = round(0.5 * spec.length_scale, 6)
        state.mesh_written = []
        # Seed each boundary-conditions form with the discovered patches so the BC
        # step shows the real patch names (with a role-based BC) instead of a blank
        # property map, and pin them as concrete schema properties so each patch
        # renders as its own titled section. Existing user/AI entries are preserved.
        for entry in entries:
            if entry.kind != "field_bc":
                continue
            seeded = seed_boundary_field(entry, patch_rows, state[entry.state_key])
            state[entry.state_key] = seeded
            names = list(seeded.get("boundaryField", {}).keys())
            state[_schema_key(entry)] = patch_bc_schema(entry, names)
        n = len(spec.patches)
        state.geometry_status = f"Found {n} patch(es) in {state.stl_dir}."

    def write_mesh() -> None:
        """Author blockMeshDict / snappyHexMeshDict / preprocess.yaml from the state."""
        if not state.geometry_patches:
            state.geometry_status = "Scan a case first — no patches loaded."
            return
        try:
            spec = _spec_from_state(state)
            settings = MeshSettings(cell_size=state.geo_cell_size or None)
            written = write_mesh_configs(state.target_dir, spec, settings)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.mesh_written = []
            state.geometry_status = f"Failed to write mesh dicts: {exc}"
            return
        state.mesh_written = [str(p) for p in written]
        state.geometry_status = f"Wrote {len(written)} mesh file(s)."

    def import_case() -> None:
        """Extract the uploaded zip and load its configs into the form state."""
        upload = state.import_zip_file
        if not upload:
            state.import_status = "Choose a .zip file first."
            return
        try:
            content = _upload_content(upload)
            extract_dir = Path(tempfile.mkdtemp(prefix="neofoam_import_"))
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(extract_dir)
            case_dir = _case_root(extract_dir)
            # OpenFOAM convention: '0' is often generated from '0.orig' at run time
            # (restore0Dir) and left out of version control / archives — restore it
            # here so the initial-field configs have something to load.
            if not (case_dir / "0").exists() and (case_dir / "0.orig").is_dir():
                shutil.copytree(case_dir / "0.orig", case_dir / "0")
            case = load_case_from_disk(case_dir, solver=solver)
            configs = case_spec_to_configs(case)
            # A dict file (e.g. a classic PISO-format fvSolution) that exists on
            # disk but didn't load into any config (fails incompressiblefluid's
            # stricter PIMPLE schema) — keep its raw text so Save writes it
            # through unchanged instead of silently replacing it with the
            # wizard's own (incompatible) defaults.
            loaded_files: set[str] = set()
            for c in configs:
                io_cfg = type(c).io_config
                if io_cfg is not None:
                    loaded_files.add(io_cfg.file)
            passthrough: dict[str, str] = {}
            seen_files: set[str] = set()
            for entry in entries:
                if (
                    entry.kind != "dict"
                    or entry.cls is None
                    or entry.cls.io_config is None
                ):
                    continue
                rel = entry.cls.io_config.file
                if rel in seen_files or rel in loaded_files:
                    continue
                seen_files.add(rel)
                path = case_dir / rel
                if path.is_file():
                    passthrough[rel] = path.read_text()
            # Mesh dicts belong to the separate Geometry/STL-scan pipeline, so
            # they're never in `entries` and Save never writes them on its own —
            # carry an imported case's own mesh dict through directly, since the
            # Geometry step wasn't used to author one.
            for rel in ("system/blockMeshDict", "system/snappyHexMeshDict"):
                path = case_dir / rel
                if path.is_file():
                    passthrough[rel] = path.read_text()
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.import_status = f"Import failed: {exc}"
            return

        state.import_passthrough = passthrough
        for key, data in cs.configs_to_form_state(entries, configs).items():
            state[by_key[key].state_key] = data
        filled_models = cs.models_filled_by(entries, configs)
        for model in filled_models:
            state[f"sel_{model}"] = True

        name = (
            upload.get("name", "case.zip") if isinstance(upload, dict) else "case.zip"
        )
        status = f"Imported {len(configs)} config(s) from {name}."
        if filled_models:
            status += f" Selected models: {', '.join(sorted(filled_models))}."
        if not state.target_dir:
            status += " Set a target directory, then Save case."
        state.import_status = status

        if list((case_dir / "constant" / "triSurface").glob("*.stl")):
            state.stl_dir = str(case_dir)
            load_geometry(set_target=False)

    _RUN_LOG_LIMIT = 200_000  # cap the live log so a long run can't blow up the DOM
    # Mutable holder for the in-flight run (closures can't rebind an outer `process =
    # None` from a nested function without `nonlocal` sprawl across run/stop/pause).
    run_handle: dict[str, Any] = {"process": None, "stopped": False}

    async def run_case() -> None:
        """Launch ``./Allrun`` in ``target_dir`` and stream its output into state.run_log."""
        if state.run_status == "running":
            return
        target = Path(state.target_dir) if state.target_dir else None
        if target is None or not target.is_dir():
            state.run_log = "Set a target directory and Save case first."
            state.run_status = "failed"
            return
        allrun = target / "Allrun"
        if not allrun.is_file():
            state.run_log = "No Allrun script found — Save case first."
            state.run_status = "failed"
            return

        state.run_status = "running"
        state.run_paused = False
        state.run_log = ""
        run_handle["stopped"] = False
        state.flush()
        try:
            # 'source' (not 'exec'/a plain call) so Allrun's body runs inside this
            # already-running bash instead of triggering a fresh process-image load —
            # macOS strips DYLD_* on every bash (re)load regardless of inherited env,
            # so a second bash hop would silently drop what the preamble just set.
            # The extra positional arg after '-c script' becomes $0, so Allrun's own
            # 'cd "${0%/*}"' resolves to the case dir instead of to bash's own path.
            command = f"{_dyld_reexport_preamble()}source {shlex.quote(str(allrun))}"
            process = await asyncio.create_subprocess_exec(
                "/bin/bash",
                "-c",
                command,
                str(allrun),
                cwd=str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,  # own process group, so Stop/Pause can
                # signal every descendant (blockMesh, the exec'd solver…), not
                # just the top-level bash.
            )
            run_handle["process"] = process
            assert process.stdout is not None
            while True:
                chunk = await process.stdout.readline()
                if not chunk:
                    break
                state.run_log += chunk.decode(errors="replace")
                if len(state.run_log) > _RUN_LOG_LIMIT:
                    state.run_log = state.run_log[-_RUN_LOG_LIMIT:]
                state.flush()
            returncode = await process.wait()
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.run_log += f"\nFailed to launch: {exc}\n"
            state.run_status = "failed"
            run_handle["process"] = None
            return
        state.run_status = (
            "stopped"
            if run_handle["stopped"]
            else ("done" if returncode == 0 else "failed")
        )
        state.run_paused = False
        run_handle["process"] = None

    async def stop_case() -> None:
        """Kill the running case (whole process group): SIGTERM, escalate to SIGKILL."""
        process = run_handle["process"]
        if process is None or process.returncode is not None:
            return
        run_handle["stopped"] = True
        state.run_paused = False
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def toggle_pause() -> None:
        """SIGSTOP/SIGCONT the running case's whole process group."""
        process = run_handle["process"]
        if process is None or process.returncode is not None:
            return
        try:
            if state.run_paused:
                os.killpg(process.pid, signal.SIGCONT)
                state.run_paused = False
            else:
                os.killpg(process.pid, signal.SIGSTOP)
                state.run_paused = True
        except ProcessLookupError:
            state.run_paused = False

    ctrl.save_case = save_case
    ctrl.revalidate = revalidate
    ctrl.load_geometry = load_geometry
    ctrl.write_mesh = write_mesh
    ctrl.import_case = import_case
    ctrl.run_case = run_case
    ctrl.stop_case = stop_case
    ctrl.toggle_pause = toggle_pause
    ctrl.get_entries = lambda: entries
    ctrl.get_steps = lambda: steps

    build_agent_panel(server, entries, solver)
    sweep_panel = SweepPanel(server, entries, solver, solver_name)

    # Contributed steps (§ neofoam.ui.plugins): each seeds its own state +
    # controllers now, and draws its panel in the content loop below.
    ctx = StepContext(
        server=server,
        solver=solver,
        solver_name=solver_name,
        entries=entries,
        json_forms=JsonForms,
        v3=v3,
        html=html,
        client=client,
        sweep=sweep_panel,
        schema_key=_schema_key,
    )
    for plugin in step_plugins:
        plugin.register(ctx)

    # ------------------------------------------------------------------ #
    # Layout                                                             #
    # ------------------------------------------------------------------ #

    def _step_header(step: Step) -> None:
        _p = plugin_by_id.get(step.id)
        caption = _p.caption if _p is not None else _STEP_CAPTIONS.get(step.id, "")
        html.Div(step.label, classes="text-h5 font-weight-bold nf-step-title")
        html.Div(
            caption,
            classes="text-body-2 text-medium-emphasis mb-5",
        )

    def _form_panel(entry: FormEntry) -> None:
        # Owned by an optional model → hide unless selected; else always visible
        # (omit v-show entirely — a bare `v-show` with no expression won't compile).
        panel_kwargs = {}
        if entry.owner_model is not None:
            panel_kwargs["v_show"] = f"sel_{entry.owner_model}"
        with v3.VExpansionPanel(elevation=0, **panel_kwargs):
            with v3.VExpansionPanelTitle():
                html.Span(entry.title)
                if entry.owner_model is not None:
                    v3.VChip(
                        entry.owner_model,
                        size="x-small",
                        color="secondary",
                        variant="tonal",
                        classes="ml-2",
                    )
            with v3.VExpansionPanelText():
                JsonForms(
                    schema=(_schema_key(entry),),
                    data=(entry.state_key,),
                    change=f"{entry.state_key} = $event.data",
                )

    def _import_panel() -> None:
        """Upload a zip of an existing OpenFOAM case and load it into the forms."""
        with v3.VCard(variant="outlined", classes="mb-6"):
            with v3.VCardText():
                html.Div(
                    "Import existing case",
                    classes="text-overline text-medium-emphasis",
                )
                with v3.VRow(align="center", dense=True):
                    with v3.VCol():
                        v3.VFileInput(
                            v_model=("import_zip_file", None),
                            label="Case (.zip)",
                            accept=".zip",
                            hide_details=True,
                            prepend_icon="mdi-folder-zip-outline",
                        )
                    with v3.VCol(cols="auto"):
                        v3.VBtn(
                            "Import",
                            click=ctrl.import_case,
                            color="primary",
                            variant="tonal",
                            prepend_icon="mdi-upload-outline",
                            disabled=("!import_zip_file",),
                        )
                v3.VAlert(
                    text=("import_status",),
                    type="info",
                    variant="tonal",
                    v_show="import_status",
                    classes="mt-3",
                )

    def _model_selector() -> None:
        """Required models (locked on) + optional model toggles."""
        with v3.VCard(variant="outlined", classes="mb-6"):
            with v3.VCardText():
                html.Div(
                    "Included models",
                    classes="text-overline text-medium-emphasis",
                )
                with html.Div(classes="d-flex flex-wrap ga-2 mb-4"):
                    for c in required:
                        v3.VChip(
                            c.label,
                            prepend_icon="mdi-lock",
                            variant="tonal",
                            color="primary",
                            size="small",
                        )
                html.Div(
                    "Optional models",
                    classes="text-overline text-medium-emphasis",
                )
                for c in optional:
                    v3.VSwitch(v_model=(f"sel_{c.name}",), label=c.label, inset=True)

    def _geometry_panel() -> None:
        """STL patches → blockMesh / snappy / preprocess dicts."""
        with v3.VCard(variant="outlined", classes="mb-4"):
            with v3.VCardText():
                with v3.VRow(align="center", dense=True):
                    with v3.VCol():
                        v3.VTextField(
                            v_model=("stl_dir",),
                            label="STL folder (triSurface or case dir)",
                            hide_details=True,
                            prepend_inner_icon="mdi-folder-outline",
                        )
                    with v3.VCol(cols="auto"):
                        v3.VBtn(
                            "Scan",
                            click=ctrl.load_geometry,
                            color="primary",
                            variant="tonal",
                            prepend_icon="mdi-magnify",
                        )
                html.Div(
                    "Point at a folder of STLs (or a case dir with"
                    " constant/triSurface/*.stl) and Scan. Adjust patch roles and"
                    " refinement, then Write mesh to author blockMeshDict,"
                    " snappyHexMeshDict and preprocess.yaml.",
                    classes="text-body-2 text-medium-emphasis mt-3",
                )
        # Per-patch role + refinement editor (rows come from the scan).
        with v3.VCard(
            variant="outlined", classes="mb-4", v_show="geometry_patches.length"
        ):
            with v3.VTable(density="compact", hover=True):
                with html.Thead():
                    with html.Tr():
                        html.Th("Patch")
                        html.Th("STL")
                        html.Th("Faces")
                        html.Th("Role")
                        html.Th("Refinement")
                with html.Tbody():
                    with html.Tr(v_for="(p, i) in geometry_patches", key="i"):
                        with html.Td():
                            html.Span("{{ p.name }}", classes="font-weight-medium")
                        with html.Td():
                            html.Span("{{ p.stl }}", classes="text-medium-emphasis")
                        with html.Td():
                            html.Span("{{ p.faces }}", classes="text-medium-emphasis")
                        with html.Td():
                            v3.VSelect(
                                v_model=("p.role",),
                                items=("role_names",),
                                hide_details=True,
                                style="min-width: 130px",
                            )
                        with html.Td():
                            v3.VTextField(
                                v_model=("p.refinement_str",),
                                v_show="p.is_snappy",
                                placeholder="min max",
                                hide_details=True,
                                style="max-width: 110px",
                            )
            # Background-mesh knobs.
            with v3.VCardText():
                with v3.VRow(dense=True):
                    with v3.VCol(cols="6"):
                        v3.VTextField(
                            v_model=("geo_location",),
                            label="locationInMesh (x y z)",
                            hide_details=True,
                        )
                    with v3.VCol(cols="6"):
                        v3.VTextField(
                            v_model=("geo_cell_size", 0.0),
                            label="Background cell size (m, 0 = auto)",
                            type="number",
                            hide_details=True,
                        )
                v3.VBtn(
                    "Write mesh",
                    click=ctrl.write_mesh,
                    color="primary",
                    prepend_icon="mdi-cube-outline",
                    classes="mt-4",
                )
        v3.VAlert(
            text=("geometry_status",),
            type="info",
            variant="tonal",
            v_show="geometry_status",
            classes="mb-2",
        )
        v3.VAlert(
            text=("'Wrote: ' + mesh_written.join(', ')",),
            type="success",
            variant="tonal",
            v_show="mesh_written.length",
        )

    def _review_panel() -> None:
        """validate_case findings + scaffolded runnable case."""
        with v3.VRow(align="center", classes="mb-2", no_gutters=True):
            v3.VSpacer()
            v3.VBtn(
                "Re-validate",
                click=ctrl.revalidate,
                variant="tonal",
                color="primary",
                prepend_icon="mdi-refresh",
            )
        # Prompt to save first.
        v3.VAlert(
            "Click 'Save case' to write the case, scaffold Allrun/Allclean and"
            " validate it.",
            type="info",
            variant="tonal",
            classes="mb-3",
            v_show="validation_ok === null",
        )
        # Overall verdict.
        v3.VAlert(
            text=(
                "validation_ok"
                " ? 'Case is valid — run it with ./Allrun'"
                " : (findings.length + ' issue(s) to fix before it will run')",
            ),
            type=("validation_ok ? 'success' : 'error'",),
            variant="tonal",
            classes="mb-3",
            v_show="validation_ok !== null",
        )
        # One alert per finding.
        with v3.VAlert(
            v_for="(f, i) in findings",
            key="i",
            type=("f.color",),
            variant="tonal",
            border="start",
            classes="mb-2",
        ):
            v3.VAlertTitle("{{ f.file }}")
            html.Div("{{ f.message }}")
            html.Div(
                "Fix → {{ f.fix }}",
                v_show="f.fix",
                classes="text-medium-emphasis mt-1",
            )
        # Scaffolded files.
        v3.VAlert(
            text=("'Scaffolded: ' + scaffolded.join(', ')",),
            type="success",
            variant="outlined",
            classes="mt-3",
            v_show="scaffolded.length",
        )
        # Run the case (./Allrun), pause/resume or stop it, and stream its log.
        with v3.VRow(align="center", classes="mt-4 mb-2", no_gutters=True):
            v3.VBtn(
                "Run case",
                click=ctrl.run_case,
                color="primary",
                variant="flat",
                prepend_icon="mdi-play-outline",
                loading=("run_status === 'running' && !run_paused",),
                disabled=("run_status === 'running'",),
            )
            v3.VBtn(
                text=("run_paused ? 'Resume' : 'Pause'",),
                click=ctrl.toggle_pause,
                variant="tonal",
                color="secondary",
                prepend_icon=("run_paused ? 'mdi-play' : 'mdi-pause'",),
                v_show="run_status === 'running'",
                classes="ml-2",
            )
            v3.VBtn(
                "Stop",
                click=ctrl.stop_case,
                variant="tonal",
                color="error",
                prepend_icon="mdi-stop",
                v_show="run_status === 'running'",
                classes="ml-2",
            )
            v3.VChip(
                text=("run_paused ? 'paused' : run_status",),
                classes="ml-3",
                size="small",
                variant="tonal",
                color=(
                    "run_status === 'done' ? 'success'"
                    " : run_status === 'failed' ? 'error'"
                    " : run_status === 'stopped' ? 'warning'"
                    " : run_paused ? 'warning'"
                    " : run_status === 'running' ? 'info' : 'secondary'",
                ),
                v_show="run_status !== 'idle'",
            )
        with v3.VCard(variant="outlined", v_show="run_log"):
            with html.Div(
                style=(
                    "max-height: 360px; overflow-y: auto; display: flex;"
                    " flex-direction: column-reverse;"
                ),
                classes="pa-3",
            ):
                html.Pre(
                    "{{ run_log }}",
                    style=(
                        "white-space: pre-wrap; word-break: break-word;"
                        " font-family: monospace; font-size: 0.8rem; margin: 0;"
                    ),
                )

    def _ai_drawer() -> None:
        """Right-hand foldable AI chat drawer (multi-turn, fills the forms)."""
        with v3.VNavigationDrawer(
            v_model=("ai_panel", True),
            location="right",
            width=400,
        ):
            with html.Div(classes="d-flex flex-column", style="height: 100%;"):
                v3.VToolbar(title="AI assistant", density="compact", flat=True)
                # Scrolling transcript.
                with html.Div(classes="flex-grow-1 pa-3", style="overflow-y: auto;"):
                    # Empty-state hint + suggested prompts.
                    with html.Div(v_show="!chat_log.length"):
                        v3.VCardText(
                            "Describe your case and I'll fill the forms. Try:",
                            classes="text-medium-emphasis px-0",
                        )
                        with v3.VChip(
                            v_for="(p, i) in suggested_prompts",
                            key="i",
                            click=(ctrl.send_message, "[p]"),
                            size="small",
                            variant="tonal",
                            color="secondary",
                            classes="mb-2",
                            style="height: auto; white-space: normal;",
                        ):
                            html.Span("{{ p }}", classes="py-1")
                    # Messages.
                    with v3.VSheet(
                        v_for="(m, i) in chat_log",
                        key="i",
                        rounded="lg",
                        classes="pa-3 mb-2",
                        color=("m.role === 'user' ? 'primary' : 'surface-variant'",),
                    ):
                        html.Div(
                            "{{ m.content }}",
                            style="white-space: pre-wrap; font-size: 0.9rem;",
                        )
                    v3.VProgressLinear(
                        indeterminate=True, v_show="ai_busy", color="secondary"
                    )
                # Composer pinned to the bottom.
                with html.Div(classes="pa-3"):
                    v3.VTextField(
                        v_model=("chat_input",),
                        placeholder="Message the assistant…",
                        hide_details=True,
                        keydown_enter=(ctrl.send_message, "[]"),
                    )
                    v3.VBtn(
                        "Send",
                        click=(ctrl.send_message, "[]"),
                        loading=("ai_busy",),
                        disabled=("!chat_input",),
                        color="secondary",
                        prepend_icon="mdi-send",
                        block=True,
                        classes="mt-2",
                    )

    with SinglePageWithDrawerLayout(server, vuetify_config=_VUETIFY_CONFIG) as layout:
        layout.title.set_text("NeoFOAM case wizard")
        client.Style(_CSS)

        with layout.drawer:
            with v3.VList(nav=True, density="comfortable", color="primary"):
                v3.VListSubheader(solver_name)
                for step in steps:
                    _p = plugin_by_id.get(step.id)
                    v3.VListItem(
                        title=step.label,
                        prepend_icon=(
                            _p.icon
                            if _p is not None
                            else _STEP_ICONS.get(step.id, "mdi-circle-outline")
                        ),
                        rounded="lg",
                        active=(f"current_step === '{step.id}'",),
                        click=f"current_step = '{step.id}'",
                    )

        with layout.toolbar:
            v3.VSpacer()
            v3.VTextField(
                v_model=("target_dir",),
                label="Target directory",
                hide_details=True,
                prepend_inner_icon="mdi-folder-arrow-down-outline",
                style="max-width: 340px",
            )
            v3.VBtn(
                "Save case",
                click=ctrl.save_case,
                color="primary",
                variant="flat",
                prepend_icon="mdi-content-save-outline",
                classes="mx-3",
            )
            # Fold / unfold the AI assistant drawer.
            v3.VBtn(
                icon="mdi-robot-happy-outline",
                click="ai_panel = !ai_panel",
                variant="text",
            )

        with layout.root:
            _ai_drawer()

        with layout.content, v3.VContainer(fluid=True, classes="pa-6"):
            # One uniform loop: every step renders its header, its bespoke panel
            # (models / geometry / review) and its schema-generated form panels.
            for step in steps:
                with html.Div(v_show=f"current_step === '{step.id}'"):
                    _step_header(step)
                    if step.id == "models":
                        _import_panel()
                        _model_selector()
                    elif step.id == "geometry":
                        _geometry_panel()
                    elif step.id == "sweep":
                        sweep_panel.render(JsonForms)
                    elif step.id == "review":
                        _review_panel()
                    elif step.id in plugin_by_id:
                        plugin_by_id[step.id].render(ctx)
                    if step.entry_keys:
                        with v3.VExpansionPanels():
                            for key in step.entry_keys:
                                _form_panel(by_key[key])

    server.enable_module(jsonforms_module)
    return server
