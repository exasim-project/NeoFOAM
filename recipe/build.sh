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

# pybFoam's meshing module does not just bind OpenFOAM, it COMPILES five of OpenFOAM's own
# application sources (src/pybFoam/meshing/CMakeLists.txt:4):
#
#   set(CHECKMESH_DIR "$ENV{WM_PROJECT_DIR}/applications/utilities/mesh/manipulation/checkMesh")
#
# The conda openfoam package installs headers, libraries, etc, bin, wmake, platforms and
# tutorials — but not applications/, which it compiles and then discards. So that path does not
# exist and CMake fails with "Cannot find source file", then "No SOURCES given to target:
# meshing". pybFoam offers no option to skip the module.
#
# Fetch the matching release and stage just that directory. Matching matters: building 2406-era
# checkMesh sources against 2412 headers would be a silent behaviour risk rather than a build
# error, so the tarball version is tied to FOAM_API, which the openfoam package itself sets.
# WM_PROJECT_DIR is safe to re-point — pybFoam reads it only in meshing/CMakeLists.txt (lines 4
# and 79), and the FOAM_* variables were already exported by activation with absolute values.
checkmesh_rel="applications/utilities/mesh/manipulation/checkMesh"
if [[ ! -d "${WM_PROJECT_DIR}/${checkmesh_rel}" ]]; then
    # Update this checksum whenever the openfoam pin in recipe.yaml moves.
    case "${FOAM_API}" in
        2412) openfoam_sha256="c353930105c39b75dac7fa7cfbfc346390caa633a868130fd8c9816ef5f732cd" ;;
        *)
            echo "No pinned OpenFOAM source checksum for FOAM_API=${FOAM_API}." >&2
            echo "Add one next to the openfoam pin in recipe/recipe.yaml." >&2
            exit 1
            ;;
    esac

    apps_stage="${SRC_DIR}/_openfoam_apps"
    tarball="${SRC_DIR}/openfoam-v${FOAM_API}.tgz"

    echo "Fetching OpenFOAM v${FOAM_API} sources for ${checkmesh_rel}"
    curl -fsSL --retry 3 -o "${tarball}" \
        "https://sourceforge.net/projects/openfoam/files/v${FOAM_API}/OpenFOAM-v${FOAM_API}.tgz/download"

    actual_sha256="$("${PYTHON}" -c '
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
' "${tarball}")"

    if [[ "${actual_sha256}" != "${openfoam_sha256}" ]]; then
        echo "OpenFOAM source checksum mismatch." >&2
        echo "  expected ${openfoam_sha256}" >&2
        echo "  actual   ${actual_sha256}" >&2
        exit 1
    fi

    mkdir -p "${apps_stage}"
    tar -xzf "${tarball}" -C "${apps_stage}" --strip-components=1 \
        "OpenFOAM-v${FOAM_API}/${checkmesh_rel}"
    rm -f "${tarball}"

    if [[ ! -f "${apps_stage}/${checkmesh_rel}/checkGeometry.C" ]]; then
        echo "checkMesh sources missing after extraction into ${apps_stage}" >&2
        exit 1
    fi

    echo "Re-pointing WM_PROJECT_DIR: ${WM_PROJECT_DIR} -> ${apps_stage}"
    export WM_PROJECT_DIR="${apps_stage}"
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
