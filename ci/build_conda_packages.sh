#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense

# Build the NeoFOAM conda package for one python version.
#
# Usage:
#   ci/build_conda_packages.sh [--python 3.12] [--output-dir output]
#                              [--target-platform linux-64] [--] [extra rattler-build arguments]
#
# Ported from src/NeoN/ci/build_conda_packages.sh. NeoN's GPU flavours and its macOS/aarch64
# target platforms are dropped: conda-forge builds `openfoam` for linux-64 only, so there is
# nothing else to build against.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python_version="3.12"
output_dir="${repo_root}/output"
target_platform="linux-64"
# glibc floor for the linux package. 2.17 is the widest baseline conda-forge still ships a
# sysroot for, matching NeoN.
glibc_version="2.17"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) python_version="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --target-platform) target_platform="$2"; shift 2 ;;
        --glibc-version) glibc_version="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! "${python_version}" =~ ^([0-9]+)\.([0-9]+) ]]; then
    echo "--python expects a <major>.<minor> version, got '${python_version}'" >&2
    exit 1
fi

# conda-forge builds nanobind only for python >=3.10, and the recipe needs it in host. Reject
# here rather than let rattler-build fail much later with an opaque resolution error.
if (( BASH_REMATCH[1] < 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] < 10) )); then
    echo "python ${python_version} is not supported by the conda package: conda-forge has no" >&2
    echo "nanobind below python 3.10." >&2
    exit 1
fi

if [[ "${target_platform}" != linux-* ]]; then
    echo "Only linux target platforms are supported: conda-forge builds openfoam for" >&2
    echo "linux-64 only, so '${target_platform}' has no OpenFOAM to build against." >&2
    exit 1
fi

# NeoN stamps its version with scripts/set_package_version.py; NeoFOAM has no such script, so
# read project.version straight out of pyproject.toml, which CMake also parses.
if [[ -z "${NEOFOAM_VERSION:-}" ]]; then
    NEOFOAM_VERSION="$(python3 -c '
import tomllib, sys
with open(sys.argv[1], "rb") as fh:
    print(tomllib.load(fh)["project"]["version"])
' "${repo_root}/pyproject.toml")"
fi
export NEOFOAM_VERSION
echo "Building neofoam ${NEOFOAM_VERSION} for ${target_platform} (python ${python_version})" >&2

variant_file="$(mktemp -t neofoam-variant.XXXXXX)"
trap 'rm -f "${variant_file}"' EXIT

{
    echo "python:"
    echo "  - \"${python_version}\""
    # ${{ stdlib('c') }} in the recipe has no built-in default; name the C runtime floor.
    echo "c_stdlib:"
    echo "  - sysroot"
    echo "c_stdlib_version:"
    echo "  - \"${glibc_version}\""
} > "${variant_file}"

{
    echo "--- variant configuration ---"
    cat "${variant_file}"
    echo "-----------------------------"
} >&2

rattler_build="${RATTLER_BUILD:-rattler-build}"

"${rattler_build}" build \
    --recipe "${repo_root}/recipe/recipe.yaml" \
    --variant-config "${variant_file}" \
    --output-dir "${output_dir}" \
    --target-platform "${target_platform}" \
    --channel conda-forge \
    "$@"
