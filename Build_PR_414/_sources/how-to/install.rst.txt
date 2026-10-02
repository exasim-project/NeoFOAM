Install NeoFOAM
===============

The minimum-viable path for using ``incompressibleFluid``.

Prerequisites
-------------

- Python 3.10+
- An OpenFOAM install on ``PATH`` (NeoFOAM links against ``pybFoam``)
- ``uv`` (https://github.com/astral-sh/uv) — used to create the virtualenv

Install
-------

From a checked-out NeoFOAM tree:

.. code-block:: bash

    uv venv                # create the virtualenv
    uv pip install pip     # bootstrap pip into the venv
    pip install .[all] -v  # build NeoN + NeoFOAM and install everything

That builds NeoN from the bundled submodule, builds NeoFOAM against it, and
installs everything into the virtualenv.

Verify
------

.. code-block:: bash

    pytest test/ -q

A green run means the framework, IO, and ``incompressibleFluid`` solver are
ready.

Build the docs
--------------

The docs build (``poe build_docs``) executes the gallery scripts under
``examples/tutorials/`` to render real solver output, so it needs OpenFOAM
sourced **plus** the dev/docs extras installed — which ``pip install .[all] -v``
provides:

.. code-block:: bash

    poe build_docs

If ``poe build_docs`` errors with ``command not found``, ``poethepoet`` isn't
in the venv yet — re-run ``pip install .[all] -v`` and try again.

Notes
-----

- For the full development build/test workflow (scoped tests, ``pre-commit``,
  and the build gotchas) see :doc:`/reference/build-and-test`.
- For C++/CMake-level build options (alternate NeoN sources, GPU flags,
  CMake presets) see the project README.
