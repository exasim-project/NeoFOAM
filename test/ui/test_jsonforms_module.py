# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The bundled JSONForms trame module points at real, checked-in build assets."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from neofoam.ui import jsonforms_module


def _resolve(rel: str) -> Path:
    # module.scripts / module.styles are "<prefix>/<file>"; the prefix maps to STATIC_DIR.
    return jsonforms_module.STATIC_DIR / Path(rel).name


def test_serve_dir_exists():
    ((prefix, path),) = jsonforms_module.serve.items()
    assert Path(path).is_dir()
    assert Path(path) == jsonforms_module.STATIC_DIR


def test_scripts_and_styles_resolve_to_built_assets():
    assert jsonforms_module.scripts, "no client scripts declared"
    assert jsonforms_module.styles, "no client styles declared"
    for rel in [*jsonforms_module.scripts, *jsonforms_module.styles]:
        asset = _resolve(rel)
        assert asset.is_file(), f"missing built asset {asset} (run bunx vite build)"
        assert asset.stat().st_size > 0


def test_umd_bundle_externalizes_trame_globals():
    umd = (jsonforms_module.STATIC_DIR / "neofoam_jsonforms.umd.js").read_text()
    # The bundle must bind to trame's single Vue/Vuetify globals, not a second copy.
    assert "neofoam_jsonforms" in umd
    assert 'require("vue")' in umd
    assert 'require("vuetify")' in umd


def test_umd_bundle_has_no_colliding_module_names():
    """No two CJS module factories may share a minified name.

    A previously committed bundle bound ``On`` to both lodash ``includes`` and an
    Ajv module; the later ``var`` won, so ``@jsonforms/core``'s ``hasType`` called
    a non-function and every schema form panel rendered empty. Nothing in the
    headless suite mounts the bundle, so only this guard catches it.
    """
    umd = (jsonforms_module.STATIC_DIR / "neofoam_jsonforms.umd.js").read_text()
    # `<name>=f((` is the minified CJS module-factory form emitted by the lib build.
    names = re.findall(r"(?<![A-Za-z0-9_$.])([A-Za-z_$][A-Za-z0-9_$]*)=f\(\(", umd)
    # Guard the guard: a future minifier renaming `f` would silently match nothing.
    assert len(names) > 100, f"module-factory pattern no longer matches ({len(names)} hits)"
    duplicates = {name for name, count in Counter(names).items() if count > 1}
    assert not duplicates, f"colliding minified module names: {sorted(duplicates)}"
