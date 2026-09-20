# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The bundled JSONForms trame module points at real, checked-in build assets."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from neofoam.ui import jsonforms_module
from neofoam.ui.form_schema import RENDERER_KEYWORDS

# `nf*` names the JS sets and reads itself (uischema options), never sent by Python.
_JS_INTERNAL = {"nfInline", "nfRowKey"}


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


def test_renderer_keywords_match_the_js_sources_and_the_bundle():
    # A keyword renamed on one side only falls back to the stock renderer without any
    # error (for `nfPatches` that corrupts dotted patch names); the bundle check trips
    # on a stale build without needing bun.
    sources = "\n".join(p.read_text() for p in jsonforms_module.STATIC_DIR.parent.glob("*.mjs"))
    in_sources = set(re.findall(r"\bnf[A-Z]\w*", sources))
    bundle = (jsonforms_module.STATIC_DIR / "neofoam_jsonforms.umd.js").read_text()
    in_bundle = set(re.findall(r"\bnf[A-Z]\w*", bundle))

    assert in_sources - _JS_INTERNAL == set(RENDERER_KEYWORDS)
    assert set(RENDERER_KEYWORDS) <= in_bundle


_MODULE_DIR = jsonforms_module.STATIC_DIR.parent
_BUNX = shutil.which("bunx") or shutil.which("bunx", path=str(Path.home() / ".bun" / "bin"))


@pytest.mark.skipif(
    _BUNX is None or not (_MODULE_DIR / "node_modules").is_dir(),
    reason="needs bun and an installed node_modules/ (bun install)",
)
def test_committed_bundle_equals_a_rebuild(tmp_path):
    # `--configLoader native` keeps Vite from writing node_modules/.vite-temp, so the
    # build leaves the repo untouched; static/ is exempt from the end-of-file fixer so
    # the committed files are the build output byte for byte.
    subprocess.run(
        [_BUNX, "vite", "build", "--configLoader", "native", "--outDir", str(tmp_path)],
        cwd=_MODULE_DIR,
        check=True,
        capture_output=True,
    )

    for built in ("neofoam_jsonforms.umd.js", "neofoam_jsonforms.css"):
        committed = (jsonforms_module.STATIC_DIR / built).read_bytes()
        assert (tmp_path / built).read_bytes() == committed, f"{built} is stale: bunx vite build"
