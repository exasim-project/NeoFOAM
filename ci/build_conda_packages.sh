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
# Ported from src/NeoN/ci/build_conda_packages.sh, without NeoN's GPU flavours. The target
# platforms follow whatever `openfoam` is published for in greole/exasim-project, which is
# linux-64 and osx-arm64.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python_version="3.12"
output_dir="${repo_root}/output"
target_platform="linux-64"
# glibc floor for the linux package. 2.17 is the widest baseline conda-forge still ships a
# sysroot for, matching NeoN.
glibc_version="2.17"
macos_deployment_target="12.0"

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

case "${target_platform}" in
    linux-64 | osx-arm64) ;;
    *)
        echo "Unsupported target platform '${target_platform}'." >&2
        echo "openfoam is published for linux-64 and osx-arm64 only." >&2
        exit 1
        ;;
esac

# Version resolution, highest priority first:
#   1. NEOFOAM_VERSION — what CI passes, derived from the tag name. PR #411's release.sh says
#      explicitly that the publishing workflows "should take the version from the tag name, not
#      pyproject.toml", and a conda version cannot be republished once uploaded.
#   2. scripts/set_package_version.py --print — the release tooling's single source of truth.
#   3. pyproject.toml directly, for branches that predate that script.
if [[ -z "${NEOFOAM_VERSION:-}" ]]; then
    if [[ -f "${repo_root}/scripts/set_package_version.py" ]]; then
        NEOFOAM_VERSION="$(python3 "${repo_root}/scripts/set_package_version.py" --print)"
    else
        NEOFOAM_VERSION="$(python3 -c '
import tomllib, sys
with open(sys.argv[1], "rb") as fh:
    print(tomllib.load(fh)["project"]["version"])
' "${repo_root}/pyproject.toml")"
    fi
fi
export NEOFOAM_VERSION
echo "Building neofoam ${NEOFOAM_VERSION} for ${target_platform} (python ${python_version})" >&2

variant_file="$(mktemp -t neofoam-variant.XXXXXX)"
trap 'rm -f "${variant_file}"' EXIT

{
    echo "python:"
    echo "  - \"${python_version}\""
    # ${{ stdlib('c') }} in the recipe has no built-in default; name the C runtime floor.
    # ${{ stdlib('c') }} has no built-in default; name the C runtime floor per platform.
    case "${target_platform}" in
        osx-*)
            echo "c_stdlib:"
            echo "  - macosx_deployment_target"
            echo "c_stdlib_version:"
            echo "  - \"${macos_deployment_target}\""
            echo "MACOSX_DEPLOYMENT_TARGET:"
            echo "  - \"${macos_deployment_target}\""
            ;;
        *)
            echo "c_stdlib:"
            echo "  - sysroot"
            echo "c_stdlib_version:"
            echo "  - \"${glibc_version}\""
            ;;
    esac
} > "${variant_file}"

{
    echo "--- variant configuration ---"
    cat "${variant_file}"
    echo "-----------------------------"
} >&2

rattler_build="${RATTLER_BUILD:-rattler-build}"

# ---------------------------------------------------------------------------
# macOS: build on a case-sensitive volume.
#
# The openfoam package installs OpenFOAM's source tree, whose lnInclude directories carry
# headers such as string.H, time.H and wchar.H. macOS filesystems are case-INSENSITIVE by
# default, so libc++ asking for <string.h> resolves to OpenFOAM's string.H instead:
#
#   <cstring> tried including <string.h> but didn't find libc++'s <string.h> header
#
# after which every std::string-dependent OpenFOAM type collapses. No compiler flag fixes
# this — OpenFOAM's own macOS instructions require a case-sensitive filesystem.
#
# rattler-build derives SRC_DIR and the build/host prefixes from --output-dir
# (<output>/bld/rattler-build_<pkg>_<n>/...), so relocating that moves the whole build,
# the unpacked openfoam headers included. Packages are copied back afterwards so the
# workflow's artifact path is unchanged. Mirrors exasim-project/openfoam-conda.
# ---------------------------------------------------------------------------
build_output_dir="${output_dir}"
casefs_mount=""

if [[ "$(uname -s)" == "Darwin" ]]; then
    casefs_mount="${NEOFOAM_CASEFS_MOUNT:-/Volumes/NeoFOAMBuild}"
    casefs_image="${TMPDIR:-/tmp}/neofoam-build"

    echo "Disk before creating the case-sensitive volume:" >&2
    df -h / "${TMPDIR:-/tmp}" >&2 || true

    if [[ ! -d "${casefs_mount}" ]]; then
        # SPARSE so it consumes only what the build actually writes; macOS runners have
        # limited free space and a fixed-size image would not fit.
        hdiutil create -size 60g -type SPARSE -fs "Case-sensitive APFS" \
            -volname NeoFOAMBuild -quiet "${casefs_image}"
        hdiutil attach "${casefs_image}.sparseimage" \
            -mountpoint "${casefs_mount}" -nobrowse -quiet
    fi

    # Prove it is actually case-sensitive rather than trusting the -fs argument.
    probe="${casefs_mount}/.casecheck"
    rm -rf "${probe}"; mkdir -p "${probe}"
    printf 'lower\n' > "${probe}/string.h"
    printf 'upper\n' > "${probe}/string.H"
    if [[ "$(cat "${probe}/string.h")" != "lower" ]]; then
        echo "${casefs_mount} is not case-sensitive — string.h and string.H collide." >&2
        echo "The OpenFOAM headers cannot be compiled against here." >&2
        exit 1
    fi
    rm -rf "${probe}"
    echo "Case-sensitive build volume ready at ${casefs_mount}" >&2

    build_output_dir="${casefs_mount}/output"
    mkdir -p "${build_output_dir}"
fi

"${rattler_build}" build \
    --recipe "${repo_root}/recipe/recipe.yaml" \
    --variant-config "${variant_file}" \
    --output-dir "${build_output_dir}" \
    --target-platform "${target_platform}" \
    --channel https://prefix.dev/greole/exasim-project \
    --channel conda-forge \
    "$@"

# Copy the packages back off the volume so the workflow finds them where it expects.
if [[ -n "${casefs_mount}" && "${build_output_dir}" != "${output_dir}" ]]; then
    mkdir -p "${output_dir}"
    while IFS= read -r pkg; do
        rel="${pkg#"${build_output_dir}"/}"
        mkdir -p "${output_dir}/$(dirname "${rel}")"
        cp "${pkg}" "${output_dir}/${rel}"
        echo "Recovered ${rel}" >&2
    done < <(find "${build_output_dir}" -type f -name '*.conda')
fi
