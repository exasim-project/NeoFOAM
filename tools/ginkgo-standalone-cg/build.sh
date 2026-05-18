#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Configure + build the standalone Ginkgo CG reproducer.
#
# Usage: ./build.sh [<build-dir>]
#
# Requirements on the build host:
#   - A C++20 compiler (the NeoN-compatible compiler, GCC ≥ 12 / Clang ≥ 15)
#   - An installed Ginkgo 2.0.0 (same pin as src/NeoN/cmake/Versions.cmake)
#     OR a CPM-resolved Ginkgo via -DCPM_USE_LOCAL_PACKAGES=NO
#   - MPI 3.1+ (typically OpenMPI on HPC)
#
# To point at NeoN's already-installed Ginkgo:
#   GINKGO_DIR=/path/to/ginkgo/build ./build.sh
# or
#   CMAKE_PREFIX_PATH=/path/to/ginkgo/install ./build.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-${HERE}/build}"

mkdir -p "${BUILD_DIR}"

EXTRA_FLAGS=()
if [[ -n "${GINKGO_DIR:-}" ]]; then
    EXTRA_FLAGS+=("-DGinkgo_DIR=${GINKGO_DIR}")
fi
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    EXTRA_FLAGS+=("-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
fi

cmake -S "${HERE}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    "${EXTRA_FLAGS[@]}"

cmake --build "${BUILD_DIR}" -j

echo
echo "Built: ${BUILD_DIR}/standalone_cg"
echo "Run example:"
echo "  mpirun -np 2 ${BUILD_DIR}/standalone_cg <dump_dir> 0001_0_0_before_pEqn_solve__p"
