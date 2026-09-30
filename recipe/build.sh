#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense

set -euxo pipefail

# NeoN's conda build calls CMake directly and then moves $PREFIX/lib/python/neon into
# site-packages. That cannot work here: NeoFOAM's python package is a large pure-python tree
# under src/neofoam that only scikit-build-core knows how to lay out — CMake installs the
# native libraries but none of the .py files. So this builds through pip, which keeps one
# packaging path for the wheel and the conda package and means SKBUILD stays defined, which is
# what triggers the wheel install layout and the bundled pybFoam in CMakeLists.txt.
#
# Consequence, and it is a real difference from NeoN: everything lands under
# site-packages/neofoam (plus the top-level pybFoam package), so this package exposes no C++
# headers or CMake package files. It is a python package, not a C++ SDK.

# The openfoam conda package sets these from its activation script. If rattler-build has not
# activated the host environment, CMake would fail 200 lines later with a message about sourcing
# OpenFOAM; fail here instead, where the cause is obvious.
: "${FOAM_SRC:?openfoam activation did not run: FOAM_SRC is unset in the build environment}"
: "${FOAM_API:?openfoam activation did not run: FOAM_API is unset in the build environment}"
echo "Building against OpenFOAM ${FOAM_API} (FOAM_SRC=${FOAM_SRC})"

# conda-forge's openfoam activation exports FOAM_SRC=$PREFIX/src, but that directory does not
# exist: its build.sh installs every lnInclude tree under $PREFIX/include/OpenFOAM-<version>/src
# instead. cmake/OpenFOAM.cmake and pybFoam's FindOpenFOAM.cmake both resolve <module>/lnInclude
# beneath FOAM_SRC, so re-point it at the directory that actually holds the headers.
# Upstream bug in the openfoam feedstock, not something NeoFOAM can fix from its side.
if [[ ! -d "${FOAM_SRC}/OpenFOAM/lnInclude" ]]; then
    for candidate in \
        "${PREFIX}/include/OpenFOAM-${FOAM_API}/src" \
        "${PREFIX}"/include/OpenFOAM-*/src; do
        if [[ -d "${candidate}/OpenFOAM/lnInclude" ]]; then
            echo "Re-pointing FOAM_SRC: ${FOAM_SRC} -> ${candidate}"
            export FOAM_SRC="${candidate}"
            break
        fi
    done
fi

if [[ ! -d "${FOAM_SRC}/OpenFOAM/lnInclude" ]]; then
    echo "No OpenFOAM headers found under FOAM_SRC=${FOAM_SRC}" >&2
    echo "Contents of ${PREFIX}/include:" >&2
    ls -1 "${PREFIX}/include" >&2 || true
    exit 1
fi

# Passed through to CMake by scikit-build-core. CMAKE_INSTALL_LIBDIR and the rpaths are set by
# the SKBUILD branch of CMakeLists.txt, so they are deliberately not repeated here.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_PREFIX_PATH=${PREFIX} -DNEOFOAM_WITH_MPI=ON"
export CMAKE_GENERATOR=Ninja

# --no-build-isolation: the build backend and nanobind come from the host environment (see the
# recipe), so the pinned versions are the conda ones rather than whatever PyPI resolves.
# --no-deps: run dependencies are declared by the recipe, not installed by pip.
"${PYTHON}" -m pip install . \
    --no-build-isolation \
    --no-deps \
    --no-index \
    -vv

# Guard against a silently partial install: without the native module the package still imports,
# because src/neofoam/__init__.py catches ModuleNotFoundError and leaves neofoam_bindings as None.
if ! find "${SP_DIR}/neofoam" -name "neofoam_bindings*.so" -print -quit | grep -q .; then
    echo "No neofoam_bindings extension was installed into ${SP_DIR}/neofoam" >&2
    find "${SP_DIR}" -maxdepth 2 -name "*.so" >&2 || true
    exit 1
fi

# The bindings resolve libnanobind.so from the bundled pybFoam through an $ORIGIN/../pybFoam
# rpath, so a missing pybFoam is a broken package rather than a missing optional extra.
if [[ ! -d "${SP_DIR}/pybFoam" ]]; then
    echo "The bundled pybFoam package is missing from ${SP_DIR}" >&2
    exit 1
fi
