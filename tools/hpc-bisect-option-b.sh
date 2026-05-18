#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# tools/hpc-bisect-option-b.sh
#
# Option B bisect — env-gated cudaDeviceSynchronize() shim at every MPI /
# linear-solver boundary in NeoN (see .planning/debug/spuma-comparison.md F3,
# .planning/debug/gpu-proc-boundary-divergence.md 2026-05-18 entry).
#
# Runs cylinder3D on `NRANKS` ranks three times on a SINGLE build:
#
#   1. cpu        — CPUExecutor, NEON_HARD_DEVICE_SYNC unset    (reference baseline)
#   2. gpu_nosync — GPUExecutor, NEON_HARD_DEVICE_SYNC unset    (current broken behaviour)
#   3. gpu_sync   — GPUExecutor, NEON_HARD_DEVICE_SYNC=1        (Option B candidate fix)
#
# The env-var gate inside `NeoN::deviceSync()` is what activates the
# SPUMA-style `cudaDeviceSynchronize()` at every MPI/solver fence site.
# No rebuild needed between A and B runs — the env var is read once at
# process start in `NeoN::hardDeviceSyncEnabled()`.
#
# Diffs every per-rank proc-face dump CPU vs each GPU configuration and
# emits `FIRST_DIVERGENCE: ...` lines compatible with the format used by
# `tools/hpc-bisect-gpu-proc-divergence.sh`.
#
# Usage (on HPC, from a decomposed case directory):
#   NEOFOAM_BIN=/path/to/build/develop/bin/neoIcoFoam \
#   CASE_DIR=$PWD \
#   OUTPUT_DIR=$PWD/bisect_results_optionB \
#   bash tools/hpc-bisect-option-b.sh
#
# Outputs:
#   <OUTPUT_DIR>/{cpu,gpu_nosync,gpu_sync}/proc{N}/dumps/<step>_<kernel>.txt
#   <OUTPUT_DIR>/diff_cpu_vs_gpu_nosync.txt
#   <OUTPUT_DIR>/diff_cpu_vs_gpu_sync.txt
#   <OUTPUT_DIR>/diff_gpu_nosync_vs_gpu_sync.txt
#   <OUTPUT_DIR>/FIRST_DIVERGENCE.txt        — three single-line summaries
#   stdout: three "FIRST_DIVERGENCE_<pair>: ..." lines (paste back to debug session)

set -euo pipefail

# ---------------- configuration -------------------------------------------- #
NEOFOAM_BIN="${NEOFOAM_BIN:?env NEOFOAM_BIN must point at the instrumented neoIcoFoam binary}"
CASE_DIR="${CASE_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASE_DIR}/bisect_results_optionB}"
NRANKS="${NRANKS:-2}"
END_TIME="${END_TIME:-0.01}"             # 2-3 PISO steps is enough for divergence
WRITE_INTERVAL="${WRITE_INTERVAL:-0.005}"
EXEC_CPU="${EXEC_CPU:-CPUExecutor}"
EXEC_GPU="${EXEC_GPU:-GPUExecutor}"
MPIRUN="${MPIRUN:-mpirun}"

# ---------------- helpers -------------------------------------------------- #
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }

# Patch ONLY the entries we need in the user's existing system/controlDict.
# Backs up to system/controlDict.bisect_backup on first call;
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

# Run one configuration.
#   label         = "cpu" | "gpu_nosync" | "gpu_sync"
#   exec_name     = CPUExecutor | GPUExecutor (drives controlDict)
#   hard_sync     = "0" or "1"               (drives NEON_HARD_DEVICE_SYNC env)
run_one() {
    local label="$1" exec_name="$2" hard_sync="$3"
    log "=== Running ${label} (${exec_name}, ${NRANKS} ranks, NEON_HARD_DEVICE_SYNC=${hard_sync}) ==="

    # Clean previous proc dumps in-case
    rm -rf "${CASE_DIR}"/processor*/dumps

    patch_controlDict "${exec_name}"

    log "  starting solver"
    (
        cd "${CASE_DIR}"
        # NEOFOAM_PROC_DUMP=1 activates the instrumented proc-face dump path
        # NEON_HARD_DEVICE_SYNC gates the SPUMA-style cudaDeviceSynchronize()
        # at every MPI/solver fence site inside NeoN::deviceSync().
        env \
            NEOFOAM_PROC_DUMP=1 \
            NEON_HARD_DEVICE_SYNC="${hard_sync}" \
            UCX_TLS=tcp \
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

# Diff two dump trees and report the first temporal divergence
# (sort by mtime so result matches solver order, not lexicographic order —
# same convention as tools/hpc-bisect-gpu-proc-divergence.sh).
#   $1 = label of reference tree (e.g. "cpu")
#   $2 = label of candidate tree (e.g. "gpu_nosync")
#   $3 = output file for the full diff
# Prints: "FIRST_DIVERGENCE_<a>_vs_<b>: <step_kernel> | RANK: <r> | FIELD: <f> | DELTA: <linf>"
#         or "NO_DIVERGENCE_<a>_vs_<b>: ..."
diff_two_trees() {
    local ref_label="$1" cand_label="$2" diff_out="$3"
    local ref_root="${OUTPUT_DIR}/${ref_label}"
    local cand_root="${OUTPUT_DIR}/${cand_label}"

    : > "${diff_out}"

    local first_div="" first_rank="" first_field="" first_delta=""

    while IFS= read -r ref_file; do
        local rel="${ref_file#${ref_root}/}"
        local cand_file="${cand_root}/${rel}"
        if [[ ! -f "${cand_file}" ]]; then
            printf 'MISSING IN %s: %s\n' "${cand_label}" "${rel}" >> "${diff_out}"
            continue
        fi
        if ! diff -q "${ref_file}" "${cand_file}" > /dev/null; then
            local rank
            rank="$(printf '%s' "${rel}" | sed -E 's|^processor([0-9]+)/.*|\1|')"
            local step_kernel
            step_kernel="$(basename "${rel%.txt}")"

            printf '\n=== DIVERGENCE (%s vs %s): %s ===\n' \
                "${ref_label}" "${cand_label}" "${rel}" >> "${diff_out}"
            diff -u "${ref_file}" "${cand_file}" >> "${diff_out}" || true

            local linf
            linf="$(paste "${ref_file}" "${cand_file}" \
                     | awk 'NF>=2 { for(i=1;i<=NF/2;i++){
                                     a=$i+0; b=$(i+NF/2)+0;
                                     d=(a-b); if(d<0)d=-d;
                                     if(d>m)m=d } } END { printf "%.3e", (m==""?0:m) }')"

            if [[ -z "${first_div}" ]]; then
                first_div="${step_kernel}"
                first_rank="${rank}"
                first_field="$(printf '%s' "${step_kernel}" | sed -E 's|.*__||')"
                first_delta="${linf}"
            fi
        fi
    done < <(find "${ref_root}" -type f -name '*.txt' -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-)

    if [[ -z "${first_div}" ]]; then
        printf 'NO_DIVERGENCE_%s_vs_%s: %s and %s dumps match exactly across all proc-face checkpoints.\n' \
            "${ref_label}" "${cand_label}" "${ref_label}" "${cand_label}"
    else
        printf 'FIRST_DIVERGENCE_%s_vs_%s: %s | RANK: %s | FIELD: %s | DELTA: %s\n' \
            "${ref_label}" "${cand_label}" "${first_div}" "${first_rank}" "${first_field}" "${first_delta}"
    fi
}

# ---------------- main ----------------------------------------------------- #
mkdir -p "${OUTPUT_DIR}"
log "case dir:   ${CASE_DIR}"
log "output dir: ${OUTPUT_DIR}"
log "ranks:      ${NRANKS}"
log "endTime:    ${END_TIME}"

[[ -d "${CASE_DIR}/processor0" ]] \
    || die "decomposed case not found at ${CASE_DIR}/processor0 — run decomposePar first"

run_one cpu        "${EXEC_CPU}" "0"
run_one gpu_nosync "${EXEC_GPU}" "0"
run_one gpu_sync   "${EXEC_GPU}" "1"

# ---------------- diffs ---------------------------------------------------- #
log "diffing cpu vs gpu_nosync"
LINE_A="$(diff_two_trees cpu gpu_nosync "${OUTPUT_DIR}/diff_cpu_vs_gpu_nosync.txt")"

log "diffing cpu vs gpu_sync"
LINE_B="$(diff_two_trees cpu gpu_sync   "${OUTPUT_DIR}/diff_cpu_vs_gpu_sync.txt")"

log "diffing gpu_nosync vs gpu_sync"
LINE_C="$(diff_two_trees gpu_nosync gpu_sync "${OUTPUT_DIR}/diff_gpu_nosync_vs_gpu_sync.txt")"

# ---------------- summary -------------------------------------------------- #
SUMMARY_OUT="${OUTPUT_DIR}/FIRST_DIVERGENCE.txt"
{
    printf '%s\n' "${LINE_A}"
    printf '%s\n' "${LINE_B}"
    printf '%s\n' "${LINE_C}"
} | tee "${SUMMARY_OUT}"

log "full diffs: ${OUTPUT_DIR}/diff_*.txt"
log "done"
