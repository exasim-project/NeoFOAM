#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# Convenience wrapper around the standalone Ginkgo CG reproducer.
#
# Usage:
#   ./run.sh <dump_dir> <checkpoint> [nranks=2] [iters=200] [tol=1e-12]
#
# Where:
#   <dump_dir>    is the directory containing processor{0..N-1}/dumps/
#                 (e.g. the case directory after a NEOFOAM_FULL_DUMP=1 run).
#   <checkpoint>  is the LinearSystem dump prefix, e.g.
#                 "0001_0_0_before_pEqn_solve__p"
#                 (note: no _<kind>.txt suffix).
#
# Examples:
#   ./run.sh ~/cases/cylinder3D 0001_0_0_before_pEqn_solve__p
#   ./run.sh ~/cases/cylinder3D 0001_0_0_before_UEqn_solve__U  4

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${HERE}/build/standalone_cg"

if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: ${BIN} not built — run ./build.sh first." >&2
    exit 1
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dump_dir> <checkpoint> [nranks=2] [iters=200] [tol=1e-12]" >&2
    exit 1
fi

DUMP_DIR="$1"
CHECKPOINT="$2"
NRANKS="${3:-2}"
ITERS="${4:-200}"
TOL="${5:-1e-12}"

# UCX_TLS=tcp is the WSL2 transport (no CUDA-aware MPI); harmless on HPC and
# matches the NeoFOAM bisect scripts' invocation.
exec env UCX_TLS=tcp \
    mpirun -np "${NRANKS}" "${BIN}" "${DUMP_DIR}" "${CHECKPOINT}" "${ITERS}" "${TOL}"
