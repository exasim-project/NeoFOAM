# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 FoamAdapter authors
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from sphinx.builders.html import StandaloneHTMLBuilder
import subprocess, os, sys

# Add source paths for imports
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../test'))

# Doxygen
subprocess.call('doxygen Doxyfile.in', shell=True)

project = 'FoamAdapter'
copyright = '2025, FoamAdapter authors'
author = 'FoamAdapter authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_togglebutton",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # For NumPy/Google style docstrings
    'sphinx.ext.doctest',       # For running doctests in documentation
    'sphinx.ext.viewcode',      # Links to source code
    'sphinx.ext.coverage',      # Coverage reports
    'sphinxcontrib.mermaid',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx_sitemap',
    'sphinx.ext.inheritance_diagram',
    'breathe'
]

# Doctest configuration
doctest_default_flags = 0  # Can add doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
doctest_global_setup = """
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../test'))
"""

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

highlight_language = 'c++'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',  #  Provided by Google in your dashboard
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'logo_only': False,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

breathe_projects = {
    "FoamAdapter": "_build/xml/"
}
html_baseurl = "https://exasim-project.com/FoamAdapter/"
breathe_default_project = "FoamAdapter"
breathe_default_members = ('members', 'undoc-members')
