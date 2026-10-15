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

Build the C++ core directly
---------------------------

``pip install`` drives CMake for you. To build the C++ core on its own, use the
presets from ``CMakePresets.json``:

.. code-block:: bash

    cmake --list-presets            # list the presets
    cmake --preset develop          # configure
    cmake --build --preset develop  # build
    ctest --preset develop          # run the unit tests

``production`` builds Release with the examples, ``develop`` builds Debug with
the unit tests and Kokkos bounds checking, ``profiling`` builds RelWithDebInfo
with the benchmarks, and ``python-bindings`` adds the shared-library build the
wheel needs. Each preset builds into ``build/<preset>$NEON_DEVICE``.

Where NeoN comes from
~~~~~~~~~~~~~~~~~~~~~

By default NeoFOAM builds the bundled ``src/NeoN`` submodule. If that submodule
is not checked out, NeoN is fetched with CPM at the revision the submodule
points to. Two cache variables override this:

``-DNEOFOAM_NEON_DIR=<path>``
    Build an external NeoN checkout instead of the submodule.

``-DNEOFOAM_NEON_VERSION=<tag|branch|sha>``
    Fetch that NeoN revision with CPM, ignoring the submodule. CI uses this to
    build against NeoN's development branch.

GPU backends
~~~~~~~~~~~~

The device backend is selected through Kokkos; NeoFOAM forwards the flags to
NeoN:

.. code-block:: bash

    cmake --preset production -DKokkos_ENABLE_CUDA=ON  # NVIDIA
    cmake --preset production -DKokkos_ENABLE_HIP=ON   # AMD

The presets set ``CMAKE_CUDA_ARCHITECTURES=native``; set it explicitly when the
build host and the target GPU differ.

Notes
-----

- For the full development build/test workflow (scoped tests, ``pre-commit``,
  and the build gotchas) see :doc:`/reference/build-and-test`.
