Building and testing NeoFOAM
============================

The development workflow for the Python package (``src/neofoam/`` and its
tests under ``test/``). ``uv run`` / ``uv sync`` also work, but they re-run the CMake build on every
change, which is impractical for day-to-day development — prefer the ``pip``
workflow below.

Set up the environment
----------------------

.. code-block:: bash

    uv venv                     # once — create the virtualenv
    uv pip install pip          # bootstrap pip into the venv
    pip install -e ".[all]" -v  # development install (editable)

For a normal (non-development) installation, install non-editable:

.. code-block:: bash

    pip install ".[all]" -v

Quote the extras (``".[all]"``) — on macOS the default ``zsh`` treats the
square brackets as a glob pattern. Instead of ``all`` you can pick just the
optional features you need: ``dev``, ``docs``, ``agent``, ``mcp``,
``telemetry``, ``ui`` — e.g. ``pip install -e ".[dev,mcp]" -v``.

The editable install uses scikit-build's ``redirect`` mode: edits to
``src/neofoam/*.py`` take effect on the next Python start without
reinstalling. Automatic rebuilds are off (``editable.rebuild = false``), so
C++/binding changes still need a rebuild. After the first install, rebuild with

.. code-block:: bash

    poe build   # pip install --no-build-isolation -e .[all] -v

Without build isolation pip reuses the build dependencies from the venv (the
``dev`` extra installs ``scikit-build-core`` and ``nanobind``), so the
``_skbuild`` build directory stays valid and Ninja recompiles only the C++
files that changed. With isolation, every install builds in a fresh temporary
environment whose changed nanobind path makes CMake reconfigure and recompile
the bindings.

Run the tests
-------------

.. code-block:: bash

    pytest test/<area>                # scoped tests while working
    pytest                            # whole suite (testpaths=test) verify before done
    pre-commit run --files <changed>  # format + lint + mypy on your diff
    pre-commit run -a                 # format + lint + mypy on all files

Poe tasks
---------

``poethepoet`` (installed via the ``dev`` extra) wraps the common commands.
Tasks are defined in ``pyproject.toml`` under ``[tool.poe.tasks]``:

.. code-block:: bash

    poe test        # pytest
    poe lint        # ruff check
    poe format      # ruff format src test
    poe type_check  # mypy
    poe precommit   # pre-commit run --all-files
    poe build_docs  # sphinx-build -b html doc doc/_build
    poe serve_docs  # serve doc/_build on http://localhost:8000
    poe view_docs   # open the served docs in the browser
