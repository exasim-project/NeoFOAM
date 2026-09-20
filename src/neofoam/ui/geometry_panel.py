# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The Geometry step: scan boundary STLs, edit patch roles, write the mesh dicts.

Only this module talks to trame; the STL discovery and the mesh-dict authoring live
UI-free in :mod:`neofoam.ui.geometry`.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from neofoam.ui._paths import _resolve_target
from neofoam.ui.forms import FormEntry, patch_bc_schema, schema_key, seed_boundary_field
from neofoam.ui.geometry import (
    GeometrySpec,
    MeshSettings,
    PatchGeometry,
    PatchRole,
    discover_geometry,
    write_mesh_configs,
)

_ROLE_NAMES = [r.value for r in PatchRole]

# Optional pre-fill for the geometry step's STL folder (manual-testing convenience).
_DEFAULT_STL_DIR = os.environ.get("NEOFOAM_WIZARD_STL_DIR", "")


def _case_dir_for(stl_dir: str) -> str:
    """The case dir owning ``stl_dir`` (its ``constant/triSurface`` parent), else itself."""
    p = Path(stl_dir)
    if p.name == "triSurface" and p.parent.name == "constant":
        return str(p.parent.parent)
    return stl_dir


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
    """Parse an ``"x y z"`` vector; ``None`` when unparsable."""
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


class GeometryPanel:
    """State, controllers and layout of the geometry & mesh step (STL → mesh dicts)."""

    def __init__(self, server: Any, entries: list[FormEntry]):
        self._state = server.state
        self._ctrl = server.controller
        # The scan seeds the boundary-condition forms with the discovered patches.
        self._entries = entries
        state = self._state
        state.role_names = _ROLE_NAMES
        state.stl_dir = _DEFAULT_STL_DIR
        state.geometry_patches = []
        state.geometry_status = ""
        state.geometry_severity = "info"  # VAlert type: info | success | warning | error
        state.geometry_busy = False
        state.geo_bbox = None
        state.geo_location = ""
        state.geo_length_scale = 0.0
        state.geo_cell_size = 0.0
        state.mesh_written = []
        self._ctrl.load_geometry = self.load_geometry
        self._ctrl.write_mesh = self.write_mesh

    async def load_geometry(self) -> None:
        """Scan the STL folder, keeping the loop free and the Scan button busy."""
        state = self._state
        # The button is disabled while busy, but a queued click still lands here:
        # a second scan would re-seed the BC forms from a half-applied first one.
        if state.geometry_busy:
            return
        with state:  # flush now — the spinner has to show before we await
            state.geometry_busy = True
        try:
            await self._scan_geometry()
        finally:
            # Also flushes what _scan_geometry wrote (one repaint, not two).
            with state:
                state.geometry_busy = False

    async def _scan_geometry(self) -> None:
        """Read the STLs in ``state.stl_dir`` (an STL folder or case dir) into state."""
        state = self._state
        try:
            stl_dir = _resolve_target(state.stl_dir, "STL folder")
            # Parsing a production-size STL blocks for hundreds of ms — off the
            # loop with it; every state write below stays on the loop.
            spec = await asyncio.to_thread(discover_geometry, stl_dir)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.geometry_patches = []
            state.geo_bbox = None
            # The mesh dicts describe the geometry that just failed to load — leaving
            # the "Wrote: …" alert up would contradict the failure right below it.
            state.mesh_written = []
            state.geometry_severity = "error"
            state.geometry_status = f"Could not read geometry: {exc}"
            return
        # Pre-fill the case target from the STL folder if not already set.
        if not state.target_dir:
            state.target_dir = _case_dir_for(str(stl_dir))
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
        for entry in self._entries:
            if entry.kind != "field_bc":
                continue
            seeded = seed_boundary_field(entry, patch_rows, state[entry.state_key])
            state[entry.state_key] = seeded
            # Only the scanned patches: one added by hand stays deletable.
            names = [row["name"] for row in patch_rows]
            state[schema_key(entry)] = patch_bc_schema(entry, names)
        n = len(spec.patches)
        state.geometry_severity = "info"
        state.geometry_status = f"Found {n} patch(es) in {stl_dir}."

    def write_mesh(self) -> None:
        """Author blockMeshDict / snappyHexMeshDict / preprocess.yaml from the state."""
        state = self._state
        if not state.geometry_patches:
            state.geometry_severity = "warning"
            state.geometry_status = "Scan a case first — no patches loaded."
            return
        try:
            target = _resolve_target(state.target_dir, "target directory")
            spec = _spec_from_state(state)
            settings = MeshSettings(cell_size=state.geo_cell_size or None)
            written = write_mesh_configs(target, spec, settings)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            state.mesh_written = []
            state.geometry_severity = "error"
            state.geometry_status = f"Failed to write mesh dicts: {exc}"
            return
        state.mesh_written = [str(p) for p in written]
        state.geometry_severity = "success"
        state.geometry_status = f"Wrote {len(written)} mesh file(s)."

    def render(self, v3: Any, html: Any) -> None:
        """Draw the step panel (called inside the app's layout context)."""
        ctrl = self._ctrl
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
                            loading=("geometry_busy",),
                            disabled=("geometry_busy",),
                        )
                html.Div(
                    "Point at a folder of STLs (or a case dir with"
                    " constant/triSurface/*.stl) and Scan. Adjust patch roles and"
                    " refinement, then Write mesh to author blockMeshDict,"
                    " snappyHexMeshDict and preprocess.yaml.",
                    classes="text-body-2 text-medium-emphasis mt-3",
                )
        # Per-patch role + refinement editor (rows come from the scan).
        with v3.VCard(variant="outlined", classes="mb-4", v_show="geometry_patches.length"):
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
            type=("geometry_severity",),
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
