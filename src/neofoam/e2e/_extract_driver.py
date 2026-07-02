# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Standalone STL-extraction driver (stage 1) -- runs in the FreeCAD conda env.

This file is executed *by path* inside the ``foamcadagent`` conda env, where
FreeCAD + foamcadagent are importable but ``neofoam`` is **not** installed::

    conda run -n foamcadagent python _extract_driver.py <model> <case_dir>

It therefore imports only foamcadagent / FreeCAD / stdlib and writes plain JSON
(matching :class:`neofoam.e2e.manifest.PatchManifest`) plus one STL per patch
into ``<case_dir>/constant/triSurface``. The neofoam side
(:mod:`neofoam.e2e.extract`) invokes it and validates the JSON it produces.

Milestone-1 scope: the ``tube_bank.FCStd`` fluid domain (a duct with cylindrical
tube cut-outs). Boundary faces are classified deterministically -- planar faces
by their outward normal axis + side, cylindrical faces as tubes -- so no LLM is
involved in stage 1. Surfaces are scaled to metres on export so the manifest and
the STLs share one coordinate system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Patch roles must match neofoam.e2e.manifest.PatchRole (kept in sync by the
# stage-1 contract test, which validates this driver's output against the model).
_GROUP_ROLE = {
    "inlet": "inlet",
    "outlet": "outlet",
    "walls": "wall",
    "tubes": "wall",
    "frontBack": "empty",
}
# Deterministic patch order in the manifest.
_GROUP_ORDER = ["inlet", "outlet", "walls", "tubes", "frontBack"]


def _plane_group(face, bbox) -> tuple[str, str]:
    """Classify a planar face -> (patch group, background-box face it occupies)."""
    axis = face.Surface.Axis
    centre = face.CenterOfMass
    ax, ay, az = abs(axis.x), abs(axis.y), abs(axis.z)
    if ax >= ay and ax >= az:  # normal along X -> flow inlet/outlet caps
        if centre.x < 0.5 * (bbox.XMin + bbox.XMax):
            return "inlet", "x_min"
        return "outlet", "x_max"
    if ay >= ax and ay >= az:  # normal along Y -> duct side walls
        side = "y_min" if centre.y < 0.5 * (bbox.YMin + bbox.YMax) else "y_max"
        return "walls", side
    # normal along Z -> thin-slab front/back (empty)
    side = "z_min" if centre.z < 0.5 * (bbox.ZMin + bbox.ZMax) else "z_max"
    return "frontBack", side


def _classify(shape) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """Group face indices into named patches; record box faces for planar groups."""
    groups: dict[str, list[int]] = {g: [] for g in _GROUP_ORDER}
    box_faces: dict[str, list[str]] = {g: [] for g in _GROUP_ORDER}
    bbox = shape.BoundBox
    for i, face in enumerate(shape.Faces):
        if "Cylinder" in face.Surface.TypeId:
            groups["tubes"].append(i)  # curved -> snappy surface, no box faces
        else:
            group, box_face = _plane_group(face, bbox)
            groups[group].append(i)
            if box_face not in box_faces[group]:
                box_faces[group].append(box_face)
    return groups, box_faces


def _export_group(shape, indices, out_path, scale: float) -> None:
    """Write the selected faces as one STL, scaled to metres."""
    import FreeCAD  # noqa: N813  (FreeCAD's module name)
    import Part

    compound = Part.makeCompound([shape.Faces[i] for i in indices])
    if scale != 1.0:
        matrix = FreeCAD.Matrix()
        matrix.scale(scale, scale, scale)
        compound = compound.transformGeometry(matrix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compound.exportStl(str(out_path))


def extract(model_path: str, case_dir: str, scale: float = 0.001) -> dict:
    """Extract named STL patches + manifest dict from ``model_path`` into ``case_dir``."""
    from foamcadagent.parametric import FreeCADParametricModel

    case = Path(case_dir)

    model = FreeCADParametricModel(model_path)
    shape = model.geometry().native
    params = dict(model.parameters)
    groups, box_faces = _classify(shape)

    patches = []
    for name in _GROUP_ORDER:
        indices = groups[name]
        if not indices:
            continue
        stl_rel = f"constant/triSurface/{name}.stl"
        _export_group(shape, indices, case / stl_rel, scale)
        entry: dict = {"name": name, "stl": stl_rel, "role": _GROUP_ROLE[name]}
        if box_faces[name]:
            entry["box_faces"] = box_faces[name]
        if name == "tubes":
            entry["surface_refinement"] = [1, 2]
        patches.append(entry)

    def r(value: float) -> float:
        """Round away binary-float noise so manifests are clean and comparable."""
        return round(value * scale, 9)

    bbox = shape.BoundBox
    manifest = {
        "case_dir": str(case),
        "geometry_source": str(Path(model_path).name),
        "source_units": "mm",
        "scale_to_meters": scale,
        "bbox": {
            "min": [r(bbox.XMin), r(bbox.YMin), r(bbox.ZMin)],
            "max": [r(bbox.XMax), r(bbox.YMax), r(bbox.ZMax)],
        },
        # Inlet region, ahead of the first tube row, mid-height, mid-depth.
        "location_in_mesh": [
            r(bbox.XMin + 0.5 * float(params.get("Lin_mm", 160.0))),
            r(0.5 * (bbox.YMin + bbox.YMax)),
            r(0.5 * (bbox.ZMin + bbox.ZMax)),
        ],
        "length_scale": round(float(params.get("D_mm", 16.0)) * scale, 9),
        "patches": patches,
    }

    (case / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract STL patches + manifest from a CAD model.")
    parser.add_argument("model", help="Path to the CAD model (e.g. tube_bank.FCStd).")
    parser.add_argument("case_dir", help="OpenFOAM case directory to populate.")
    parser.add_argument(
        "--scale", type=float, default=0.001, help="Native-unit -> metre scale (mm -> 0.001)."
    )
    args = parser.parse_args(argv)
    manifest = extract(args.model, args.case_dir, scale=args.scale)
    names = ", ".join(p["name"] for p in manifest["patches"])
    print(f"extracted patches: {names}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
