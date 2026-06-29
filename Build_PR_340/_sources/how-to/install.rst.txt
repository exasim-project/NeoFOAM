Install NeoFOAM
===============

The minimum-viable path for using ``incompressibleFluid``.

Prerequisites
-------------

- Python 3.10+
- An OpenFOAM install on ``PATH`` (NeoFOAM links against ``pybFoam``)
- ``uv`` (https://github.com/astral-sh/uv) — recommended; falls back to ``pip``

Install
-------

From a checked-out NeoFOAM tree:

.. code-block:: bash

    uv sync --all-extras

That builds NeoN from the bundled submodule, builds NeoFOAM against it, and
installs everything into a project-local virtualenv.

Verify
------

.. code-block:: bash

    uv run pytest test/ -q

A green run means the framework, IO, and ``incompressibleFluid`` solver are
ready.

Build the docs
--------------

The docs build (``poe build_docs``) executes the gallery scripts
under ``examples/tutorials/`` to render real solver output, so it
needs OpenFOAM sourced **plus** the dev/docs extras installed:

.. code-block:: bash

    uv sync --all-extras
    poe build_docs

If ``poe build_docs`` errors with ``poe: Datei oder Verzeichnis
nicht gefunden`` (or ``command not found``), ``poethepoet`` isn't
in the venv yet — that means ``--extra docs`` was used instead of
``--all-extras``. Re-run ``uv sync --all-extras`` and try again.

Notes
-----

- For C++/CMake-level build options (alternate NeoN sources, GPU flags,
  CMake presets) see the project README.
