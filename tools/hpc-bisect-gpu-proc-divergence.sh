#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# tools/hpc-bisect-gpu-proc-divergence.sh
#
# Bisect script for the GPU multi-rank vs CPU multi-rank proc-boundary
# divergence in neoIcoFoam. Runs cylinder3D at 2 ranks twice — once with
# CPUExecutor, once with GPUExecutor — diffs the per-rank proc-face dumps
# emitted by the instrumented binary, and reports the FIRST divergent
# (kernel-name, PISO outer iter, step) tuple.
#
# Usage:
#   1. Build NeoFOAM on HPC with the instrumentation patch applied (gated
#      on env NEOFOAM_PROC_DUMP=1; production builds are unchanged).
#   2. scp this script + tutorials/cylinder3D/ to HPC.
#   3. From the HPC case dir, set NEOFOAM_BIN and run:
#         NEOFOAM_BIN=/path/to/build/develop/bin/neoIcoFoam \
#         CASE_DIR=$PWD \
#         OUTPUT_DIR=$PWD/bisect_results \
#         bash tools/hpc-bisect-gpu-proc-divergence.sh
#
# Outputs:
#   <OUTPUT_DIR>/cpu/proc{0,1}/dumps/<step>_<kernel>.txt
#   <OUTPUT_DIR>/gpu/proc{0,1}/dumps/<step>_<kernel>.txt
#   <OUTPUT_DIR>/diff.txt                       — full diff
#   <OUTPUT_DIR>/FIRST_DIVERGENCE.txt           — single-line summary
#   stdout: one-line summary "FIRST_DIVERGENCE: ..." (paste back to debug session)

set -euo pipefail

# ---------------- configuration -------------------------------------------- #
NEOFOAM_BIN="${NEOFOAM_BIN:?env NEOFOAM_BIN must point at the instrumented neoIcoFoam binary}"
CASE_DIR="${CASE_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASE_DIR}/bisect_results}"
NRANKS="${NRANKS:-2}"
END_TIME="${END_TIME:-0.01}"            # 2-3 PISO steps is enough for divergence
WRITE_INTERVAL="${WRITE_INTERVAL:-0.005}"
EXEC_CPU="${EXEC_CPU:-CPUExecutor}"
EXEC_GPU="${EXEC_GPU:-GPUExecutor}"
MPIRUN="${MPIRUN:-mpirun}"

# ---------------- helpers -------------------------------------------------- #
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }

# Patch ONLY the entries we need in the user's existing system/controlDict.
# Preserves everything else (BC schemes, libs, functionObjects, etc.) untouched.
# Backs the original up to system/controlDict.bisect_backup on first call;
# `restore_controlDict` (registered as an EXIT trap) puts it back.
patch_controlDict() {
    local exec_name="$1"
    local cd="${CASE_DIR}/system/controlDict"
    [[ -f "${cd}" ]] || die "system/controlDict not found at ${cd}"

    if [[ ! -f "${cd}.bisect_backup" ]]; then
        cp "${cd}" "${cd}.bisect_backup"
        log "  backed up controlDict -> ${cd}.bisect_backup"
    fi

    if ! command -v foamDictionary >/dev/null 2>&1; then
        die "foamDictionary not on PATH — source the OpenFOAM environment first"
    fi

    foamDictionary -entry endTime       -set "${END_TIME}"       "${cd}" >/dev/null
    foamDictionary -entry writeInterval -set "${WRITE_INTERVAL}" "${cd}" >/dev/null
    foamDictionary -entry executor      -set "${exec_name}"      "${cd}" >/dev/null
}

restore_controlDict() {
    local cd="${CASE_DIR}/system/controlDict"
    if [[ -f "${cd}.bisect_backup" ]]; then
        mv "${cd}.bisect_backup" "${cd}"
        log "restored original controlDict"
    fi
}
trap restore_controlDict EXIT

# Run one configuration. Dumps land in proc*/dumps/ inside the case;
# we move them out into the output tree afterwards.
run_one() {
    local label="$1" exec_name="$2"
    log "=== Running ${label} (${exec_name}, ${NRANKS} ranks) ==="

    # Clean previous proc dumps in-case
    rm -rf "${CASE_DIR}"/processor*/dumps

    patch_controlDict "${exec_name}"

    # decomposeParDict assumed already present and configured for NRANKS.
    # If you regenerate from scratch, run `decomposePar -force` here.
    log "  starting solver"
    (
        cd "${CASE_DIR}"
        # NEOFOAM_PROC_DUMP=1 activates the instrumented dump path
        NEOFOAM_PROC_DUMP=1 UCX_TLS=tcp \
            ${MPIRUN} -np "${NRANKS}" "${NEOFOAM_BIN}" -parallel \
            > "log.neoIcoFoam.${label}" 2>&1
    ) || die "${label} run failed — see ${CASE_DIR}/log.neoIcoFoam.${label}"

    # Stash dumps to OUTPUT_DIR/<label>/proc{N}/dumps
    local dst="${OUTPUT_DIR}/${label}"
    rm -rf "${dst}"
    mkdir -p "${dst}"
    for p in "${CASE_DIR}"/processor*; do
        [[ -d "${p}/dumps" ]] || continue
        local procname
        procname="$(basename "${p}")"
        mkdir -p "${dst}/${procname}"
        cp -r "${p}/dumps" "${dst}/${procname}/dumps"
    done
    log "  dumps stashed at ${dst}"
}

# ---------------- main ----------------------------------------------------- #
mkdir -p "${OUTPUT_DIR}"
log "case dir:   ${CASE_DIR}"
log "output dir: ${OUTPUT_DIR}"
log "ranks:      ${NRANKS}"
log "endTime:    ${END_TIME}"

[[ -d "${CASE_DIR}/processor0" ]] \
    || die "decomposed case not found at ${CASE_DIR}/processor0 — run decomposePar first"

run_one cpu "${EXEC_CPU}"
run_one gpu "${EXEC_GPU}"

# ---------------- diff ----------------------------------------------------- #
log "diffing dumps"
DIFF_OUT="${OUTPUT_DIR}/diff.txt"
: > "${DIFF_OUT}"

# Walk every proc{N}/dumps/<step>_<kernel>.txt under cpu/, compare to gpu/.
first_div=""
first_rank=""
first_field=""
first_delta=""

while IFS= read -r cpu_file; do
    rel="${cpu_file#${OUTPUT_DIR}/cpu/}"
    gpu_file="${OUTPUT_DIR}/gpu/${rel}"
    if [[ ! -f "${gpu_file}" ]]; then
        printf 'MISSING IN GPU: %s\n' "${rel}" >> "${DIFF_OUT}"
        continue
    fi
    if ! diff -q "${cpu_file}" "${gpu_file}" > /dev/null; then
        # Capture details: rank from "processor{N}", kernel/step from filename,
        # max L∞ delta from numerical diff (best-effort awk on float pairs).
        rank="$(printf '%s' "${rel}" | sed -E 's|^processor([0-9]+)/.*|\1|')"
        step_kernel="$(basename "${rel%.txt}")"

        printf '\n=== DIVERGENCE: %s ===\n' "${rel}" >> "${DIFF_OUT}"
        diff -u "${cpu_file}" "${gpu_file}" >> "${DIFF_OUT}" || true

        # Best-effort L∞ — files are expected to be plain-text rows of
        # "<faceGlobalId> <value1> <value2> <value3>" (3-comp velocity or
        # 1-comp scalar). Paste CPU/GPU and compute max abs diff.
        linf="$(paste "${cpu_file}" "${gpu_file}" \
                 | awk 'NF>=2 { for(i=1;i<=NF/2;i++){
                                 a=$i+0; b=$(i+NF/2)+0;
                                 d=(a-b); if(d<0)d=-d;
                                 if(d>m)m=d } } END { printf "%.3e", (m==""?0:m) }')"

        if [[ -z "${first_div}" ]]; then
            first_div="${step_kernel}"
            first_rank="${rank}"
            # Try to extract the field name from the filename suffix
            # (convention: "<step>_<piso>_<kernel>__<field>.txt")
            first_field="$(printf '%s' "${step_kernel}" | sed -E 's|.*__||')"
            first_delta="${linf}"
        fi
    fi
done < <(find "${OUTPUT_DIR}/cpu" -type f -name '*.txt' | sort)

# ---------------- summary -------------------------------------------------- #
SUMMARY_OUT="${OUTPUT_DIR}/FIRST_DIVERGENCE.txt"
if [[ -z "${first_div}" ]]; then
    printf 'NO_DIVERGENCE: CPU and GPU dumps match exactly across all proc-face checkpoints.\n' \
        | tee "${SUMMARY_OUT}"
else
    printf 'FIRST_DIVERGENCE: %s | RANK: %s | FIELD: %s | DELTA: %s\n' \
        "${first_div}" "${first_rank}" "${first_field}" "${first_delta}" \
        | tee "${SUMMARY_OUT}"
fi

log "full diff: ${DIFF_OUT}"
log "done"
