# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Stage the extracted STL surfaces into a case's ``constant/triSurface``.

Geometry is a *given* input to the mesh/solve workflow: stage 1 wrote the
surfaces once; this copies them into place so ``snappyHexMesh`` finds them where
the manifest's ``stl`` paths point (``constant/triSurface/<name>.stl``). No
geometry is generated here — only staged.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

from neofoam.e2e.manifest import PatchManifest

PathLike = Union[str, Path]


def stage_geometry(
    src_tri_surface: PathLike,
    case_dir: PathLike,
    *,
    manifest: Optional[PatchManifest] = None,
) -> list[Path]:
    """Copy STL surfaces from ``src_tri_surface`` into ``<case>/constant/triSurface``.

    When ``manifest`` is given, copy exactly the surfaces it declares (basenames of
    each ``PatchEntry.stl``) and raise if one is absent from the source — so an
    incomplete geometry fails fast. Otherwise copy every ``*.stl`` found in the
    source directory. Idempotent: an existing target is overwritten.

    Args:
        src_tri_surface: directory holding the extracted ``*.stl`` files.
        case_dir: target case directory; surfaces land in its ``constant/triSurface``.
        manifest: optional manifest whose declared surfaces drive the copy.

    Returns:
        The destination paths written, in copy order.
    """
    src = Path(src_tri_surface)
    dst = Path(case_dir) / "constant" / "triSurface"
    dst.mkdir(parents=True, exist_ok=True)

    if manifest is not None:
        names = [Path(patch.stl).name for patch in manifest.patches]
    else:
        names = [stl.name for stl in sorted(src.glob("*.stl"))]

    copied: list[Path] = []
    for name in names:
        source = src / name
        if not source.is_file():
            raise FileNotFoundError(f"geometry surface not found: {source}")
        target = dst / name
        shutil.copy2(source, target)
        copied.append(target)
    return copied
