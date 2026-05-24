# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Force C numeric locale. Sphinx/Babel activate the user locale for i18n,
# which on German systems sets LC_NUMERIC=de_DE.UTF-8 (decimal separator
# ","). When autodoc / gallery scripts import OpenFOAM symbols, the
# OpenFOAM dictionary parser then fails to read "2.0" in $WM_PROJECT_DIR
# /etc/controlDict with FOAM FATAL IO ERROR.
import locale
import os

locale.setlocale(locale.LC_NUMERIC, "C")

# Gallery tutorials call ``neofoam.tutorial.clone_case`` which resolves
# the bundled cases via ``NEOFOAM_CASES_DIR``.
os.environ.setdefault(
    "NEOFOAM_CASES_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tutorials"
    ),
)

# OpenFOAM installs a SIGFPE trap that fires on NaN / divide-by-zero /
# overflow at the CPU level. Some Python paths (PyOS_double_to_string)
# trip it during gallery execution. Disable for the doc build only.
os.environ.setdefault("FOAM_SIGFPE", "false")

# Headless pyvista. Gallery scripts use pyvista.Plotter; OFF_SCREEN
# suppresses window creation, BUILDING_GALLERY enables the figure
# scraper, start_xvfb spawns a virtual framebuffer on Linux without
# a display.
try:
    import pyvista as _pv

    _pv.OFF_SCREEN = True
    _pv.BUILDING_GALLERY = True
    _pv.set_plot_theme("document")
    if hasattr(_pv, "start_xvfb"):
        try:
            _pv.start_xvfb()
        except OSError:
            pass
except ImportError:
    pass

import subprocess  # noqa: E402

# Doxygen
subprocess.call("doxygen Doxyfile.in", shell=True)

project = "NeoFOAM"
copyright = "2025, NeoFOAM authors"
author = "NeoFOAM authors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.mermaid",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
    "sphinx.ext.inheritance_diagram",
    "breathe",
    "sphinx_gallery.gen_gallery",
    # tab-set / tab-item directives DynamicScraper emits around the
    # static PNG + interactive viewer pair.
    "sphinx_design",
    # Registers the ``offlineviewer`` directive that embeds the
    # `.vtksz` files DynamicScraper produces.
    "pyvista.ext.viewer_directive",
]

# Pages share section headings ("Trade-offs", "Mechanism",
# "When this matters in practice") by template; scope autosectionlabel
# per document so the headings don't collide.
autosectionlabel_prefix_document = True

# sphinx-gallery executes each examples/tutorials/example_*.py script
# during the build, scrapes pyvista + matplotlib figures, and emits
# RST + HTML + .ipynb + .zip per script under doc/auto_tutorials/.
# Caches via .py.md5; unchanged scripts skip execution.
#
# DynamicScraper produces both the static PNG (for thumbnails / fallback)
# and a self-contained .vtksz interactive viewer per pyvista figure.
# Sphinx-gallery embeds the viewer below the PNG so the reader can
# rotate / zoom the field directly in the doc page.
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper  # noqa: E402

sphinx_gallery_conf = {
    "examples_dirs": ["../examples/tutorials", "../examples/how-to"],
    "gallery_dirs": ["auto_tutorials", "auto_how-to"],
    "filename_pattern": r"/example_",
    "ignore_pattern": r"^(?!example_).*\.py$",
    "remove_config_comments": True,
    "download_all_examples": False,
    "plot_gallery": "True",
    "image_scrapers": ("matplotlib", DynamicScraper()),
    # Keep building the rest of the docs even if a tutorial errors.
    "abort_on_example_error": False,
    # All tutorials currently execute cleanly against the ported
    # incompressibleFluid solver, so none are expected to fail.
    "expected_failing_examples": [],
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

highlight_language = "default"

# Strict cross-reference checking. The tutorials are the first set of
# pages audited under nitpicky=True; non-tutorial pages are temporarily
# silenced below and tracked for follow-up cleanup.
nitpicky = True
nitpick_ignore_regex = [
    (r"py:.*", r"pybFoam\..*"),
    (r"py:.*", r"pyf\..*"),
    (r"py:.*", r"openfoam\..*"),
    (r"cpp:.*", r".*"),
    (r"std:doc", r"/development/.*"),
    # Internal TypeVars surfaced in generic signatures (spec.config).
    (r"py:class", r".*\._ConfigT$"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

breathe_projects = {"NeoFOAM": "_build/xml/"}
html_baseurl = "https://exasim-project.com/NeoFOAM/"
breathe_default_project = "NeoFOAM"
breathe_default_members = ("members", "undoc-members")


# Sphinx-gallery auto-generates ``auto_tutorials/index.rst``. We list
# each tutorial individually in ``index.rst``, but the gallery index
# still exists on disk. Marking it ``:orphan:`` keeps it reachable by
# URL while excluding it from the sidebar — otherwise furo renders
# both the gallery index *and* the per-page entries, which collapses
# the Tutorials section in the sidebar when navigating between pages.
_GALLERY_INDEX_DOCS = {"auto_tutorials/index", "auto_how-to/index"}


def _orphan_gallery_indices(app, docname, source):
    if docname in _GALLERY_INDEX_DOCS and not source[0].lstrip().startswith(":orphan:"):
        source[0] = ":orphan:\n\n" + source[0]


def setup(app):
    app.connect("source-read", _orphan_gallery_indices)
