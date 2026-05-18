#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# tools/hpc-toy-bisect.sh — HPC toy-case bisect runner for the GPU multi-rank
# proc-boundary divergence bug. Drives the doubly-graded toy reproducer
# (test/setup_distributedChannelToy_mpi2/) with the full dump infrastructure
# (NEOFOAM_FULL_DUMP=1) and produces a summary table.
#
# Pipeline:
#   1. blockMesh + decomposePar -force          (in CASE_DIR)
#   2. neoIcoFoam CPUExecutor 2-rank             — baseline,            NEOFOAM_FULL_DUMP=1
#   3. neoIcoFoam GPUExecutor 2-rank             — current broken run,  NEOFOAM_FULL_DUMP=1
#   4. neoIcoFoam GPUExecutor 2-rank NEON_HARD_DEVICE_SYNC=1 — Option B candidate
#   5. Per-rank diff of each dump file CPU vs GPU and GPU vs GPU-sync
#   6. Standalone Ginkgo CG replay (CPU-NeoFOAM-dump and GPU-NeoFOAM-dump)
#   7. Emit summary: per-checkpoint deltas + standalone Ginkgo CPU-vs-CUDA L∞
#
# Usage:
#   export OPENFOAM_BASHRC=/path/to/OpenFOAM-2406/etc/bashrc
#   export NEOFOAM_BIN=/path/to/build/develop/bin/neoIcoFoam
#   export CASE_DIR=/path/to/NeoFOAM/test/setup_distributedChannelToy_mpi2
#   export STANDALONE_CG=/path/to/NeoFOAM/tools/ginkgo-standalone-cg/build/standalone_cg
#   bash tools/hpc-toy-bisect.sh
#
# Outputs (under ${OUTPUT_DIR}):
#   {cpu,gpu_nosync,gpu_sync}/processor{N}/dumps/...        — captured dump trees
#   diff_cpu_vs_gpu_nosync.txt                              — full unified diffs
#   diff_cpu_vs_gpu_sync.txt
#   diff_gpu_nosync_vs_gpu_sync.txt
#   csr_diff_cpu_vs_gpu.txt                                 — per-checkpoint L∞(A_cpu-A_gpu), L∞(b_cpu-b_gpu)
#   standalone_cg.cpu_dump.txt                              — Ginkgo CPU/CUDA on CPU-NeoFOAM A,b
#   standalone_cg.gpu_dump.txt                              — Ginkgo CPU/CUDA on GPU-NeoFOAM A,b
#   SUMMARY.txt                                             — paste-back single-page

set -euo pipefail

OPENFOAM_BASHRC="${OPENFOAM_BASHRC:?env OPENFOAM_BASHRC must point at OpenFOAM-2406/etc/bashrc}"
NEOFOAM_BIN="${NEOFOAM_BIN:?env NEOFOAM_BIN must point at the instrumented neoIcoFoam binary}"
CASE_DIR="${CASE_DIR:?env CASE_DIR must point at the toy case directory}"
STANDALONE_CG="${STANDALONE_CG:-}"  # optional; if absent skip standalone Ginkgo replay
NRANKS="${NRANKS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASE_DIR}/hpc_toy_bisect}"
MPIRUN="${MPIRUN:-mpirun}"

# Single LinearSystem checkpoint to replay through the standalone Ginkgo tool.
# The pressure solve is the most informative (scalar-valued; CG-friendly).
STANDALONE_CHECKPOINT="${STANDALONE_CHECKPOINT:-0001_1_1_before_pEqn_solve__p}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }

[[ -f "${OPENFOAM_BASHRC}" ]] || die "OpenFOAM bashrc not found at ${OPENFOAM_BASHRC}"
[[ -x "${NEOFOAM_BIN}"     ]] || die "NEOFOAM_BIN not executable: ${NEOFOAM_BIN}"
[[ -d "${CASE_DIR}"        ]] || die "CASE_DIR not found: ${CASE_DIR}"

# shellcheck disable=SC1090
source "${OPENFOAM_BASHRC}"

mkdir -p "${OUTPUT_DIR}"
log "case dir:   ${CASE_DIR}"
log "output dir: ${OUTPUT_DIR}"
log "binary:     ${NEOFOAM_BIN}"
log "ranks:      ${NRANKS}"

# -------- 1. Prepare the toy mesh ----------------------------------------- #
(
    cd "${CASE_DIR}"
    if [[ ! -d "constant/polyMesh" ]] \
       || [[ "system/blockMeshDict" -nt "constant/polyMesh/boundary" ]]; then
        log "running blockMesh"
        blockMesh > "${OUTPUT_DIR}/log.blockMesh" 2>&1
    fi
    if [[ ! -d "processor0" ]] || [[ "constant/polyMesh/boundary" -nt "processor0/constant/polyMesh/boundary" ]]; then
        log "running decomposePar -force"
        # Restore 0/ from 0.orig/ in case decomposePar consumed it
        if [[ -d "0.orig" ]] && [[ ! -f "0/U" ]]; then
            cp 0.orig/U 0/U
            cp 0.orig/p 0/p
        fi
        decomposePar -force > "${OUTPUT_DIR}/log.decomposePar" 2>&1
    fi
)

[[ -d "${CASE_DIR}/processor0" ]] \
    || die "decomposePar did not create processor directories; see ${OUTPUT_DIR}/log.decomposePar"

# -------- 2,3,4. Run three solver configurations -------------------------- #
patch_executor() {
    local exec_name="$1"
    foamDictionary -entry executor -set "${exec_name}" "${CASE_DIR}/system/controlDict" >/dev/null
}

run_one() {
    local label="$1" exec_name="$2" hard_sync="$3"
    log "=== ${label}: ${exec_name}, ${NRANKS} ranks, NEON_HARD_DEVICE_SYNC=${hard_sync} ==="

    # Clean previous dump output
    rm -rf "${CASE_DIR}"/processor*/dumps

    patch_executor "${exec_name}"

    (
        cd "${CASE_DIR}"
        env \
            NEOFOAM_PROC_DUMP=1 \
            NEOFOAM_FULL_DUMP=1 \
            NEON_HARD_DEVICE_SYNC="${hard_sync}" \
            UCX_TLS=tcp \
            ${MPIRUN} -np "${NRANKS}" "${NEOFOAM_BIN}" -parallel \
            > "${OUTPUT_DIR}/log.neoIcoFoam.${label}" 2>&1
    ) || die "${label} solver run failed — see ${OUTPUT_DIR}/log.neoIcoFoam.${label}"

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
    log "  dumps stashed under ${dst}"
}

run_one cpu        CPUExecutor 0
run_one gpu_nosync GPUExecutor 0
run_one gpu_sync   GPUExecutor 1

# -------- 5. Per-checkpoint diff trees ------------------------------------ #
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
        printf 'NO_DIVERGENCE_%s_vs_%s: %s and %s dumps match exactly.\n' \
            "${ref_label}" "${cand_label}" "${ref_label}" "${cand_label}"
    else
        printf 'FIRST_DIVERGENCE_%s_vs_%s: %s | RANK: %s | FIELD: %s | DELTA: %s\n' \
            "${ref_label}" "${cand_label}" "${first_div}" "${first_rank}" "${first_field}" "${first_delta}"
    fi
}

log "diffing cpu vs gpu_nosync"
LINE_A="$(diff_two_trees cpu gpu_nosync "${OUTPUT_DIR}/diff_cpu_vs_gpu_nosync.txt")"
log "diffing cpu vs gpu_sync"
LINE_B="$(diff_two_trees cpu gpu_sync   "${OUTPUT_DIR}/diff_cpu_vs_gpu_sync.txt")"
log "diffing gpu_nosync vs gpu_sync"
LINE_C="$(diff_two_trees gpu_nosync gpu_sync "${OUTPUT_DIR}/diff_gpu_nosync_vs_gpu_sync.txt")"

# -------- 6. CSR-only diff: L∞(A_cpu - A_gpu_nosync), L∞(b_cpu - b_gpu_nosync) #
csr_compare_pair() {
    # Per LinearSystem checkpoint, walk the CSR files and report L_inf
    local out="$1"
    local ref_root="${OUTPUT_DIR}/cpu"
    local cand_root="${OUTPUT_DIR}/gpu_nosync"
    : > "${out}"
    printf '# LinearSystem CSR diff: CPU-NeoFOAM vs GPU-NeoFOAM (no sync)\n' >> "${out}"
    printf '# columns: rank  checkpoint  kind  L_inf\n' >> "${out}"
    while IFS= read -r ref_file; do
        local rel="${ref_file#${ref_root}/}"
        local cand_file="${cand_root}/${rel}"
        [[ -f "${cand_file}" ]] || continue
        local kind base rank
        base="$(basename "${rel%.txt}")"
        rank="$(printf '%s' "${rel}" | sed -E 's|^processor([0-9]+)/.*|\1|')"
        kind="$(printf '%s' "${base}" | sed -E 's|.*_([A-Za-z]+)$|\1|')"
        local checkpoint
        checkpoint="$(printf '%s' "${base}" | sed -E "s|_${kind}$||")"
        local linf
        linf="$(paste "${ref_file}" "${cand_file}" \
                 | awk 'NF>=2 { for(i=1;i<=NF/2;i++){
                                 a=$i+0; b=$(i+NF/2)+0;
                                 d=(a-b); if(d<0)d=-d;
                                 if(d>m)m=d } } END { printf "%.3e", (m==""?0:m) }')"
        printf '%s  %s  %s  %s\n' "${rank}" "${checkpoint}" "${kind}" "${linf}" >> "${out}"
    done < <(find "${ref_root}" -type f \
                  \( -name '*_A_values.txt' -o -name '*_A_colidx.txt' \
                  -o -name '*_A_rowptr.txt' -o -name '*_b.txt' \
                  -o -name '*_x0.txt' -o -name '*_nonLocalA_values.txt' \) \
                  -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-)
}
log "computing CSR diff CPU vs GPU"
csr_compare_pair "${OUTPUT_DIR}/csr_diff_cpu_vs_gpu.txt"

# -------- 7. Standalone Ginkgo replay ------------------------------------- #
STANDALONE_CPU_OUT="${OUTPUT_DIR}/standalone_cg.cpu_dump.txt"
STANDALONE_GPU_OUT="${OUTPUT_DIR}/standalone_cg.gpu_dump.txt"

if [[ -n "${STANDALONE_CG}" && -x "${STANDALONE_CG}" ]]; then
    log "running standalone Ginkgo replay on CPU-NeoFOAM dump"
    env UCX_TLS=tcp \
        ${MPIRUN} -np "${NRANKS}" "${STANDALONE_CG}" \
        "${OUTPUT_DIR}/cpu" "${STANDALONE_CHECKPOINT}" \
        > "${STANDALONE_CPU_OUT}" 2>&1 || true

    log "running standalone Ginkgo replay on GPU-NeoFOAM dump"
    env UCX_TLS=tcp \
        ${MPIRUN} -np "${NRANKS}" "${STANDALONE_CG}" \
        "${OUTPUT_DIR}/gpu_nosync" "${STANDALONE_CHECKPOINT}" \
        > "${STANDALONE_GPU_OUT}" 2>&1 || true
else
    log "STANDALONE_CG not set or not executable — skipping standalone Ginkgo replay"
    printf 'STANDALONE_CG not configured; skipped.\n' > "${STANDALONE_CPU_OUT}"
    printf 'STANDALONE_CG not configured; skipped.\n' > "${STANDALONE_GPU_OUT}"
fi

# -------- 8. Summary ------------------------------------------------------ #
SUMMARY="${OUTPUT_DIR}/SUMMARY.txt"
{
    printf '== HPC toy bisect summary ==\n\n'
    printf 'NeoFOAM dump tree first divergence (sorted by mtime):\n'
    printf '  %s\n'   "${LINE_A}"
    printf '  %s\n'   "${LINE_B}"
    printf '  %s\n\n' "${LINE_C}"
    printf 'CSR diff CPU-NeoFOAM vs GPU-NeoFOAM (per rank, per LinearSystem dump):\n'
    awk 'NR<=22 {print "  " $0}' "${OUTPUT_DIR}/csr_diff_cpu_vs_gpu.txt" || true
    printf '... (full table at csr_diff_cpu_vs_gpu.txt)\n\n'
    printf 'Standalone Ginkgo CG replay — CPU-NeoFOAM dump:\n'
    sed 's|^|  |' "${STANDALONE_CPU_OUT}" 2>/dev/null || true
    printf '\nStandalone Ginkgo CG replay — GPU-NeoFOAM dump:\n'
    sed 's|^|  |' "${STANDALONE_GPU_OUT}" 2>/dev/null || true
    printf '\nDecision tree:\n'
    printf '  1. CSR diff (A,b) shows non-zero L_inf at CPU vs GPU  → bug is in NeoN GPU assembly.\n'
    printf '  2. CSR diff is zero, but cpu_vs_gpu_nosync first divergence is at U/p solve output\n'
    printf '     AND standalone Ginkgo CPU vs CUDA also disagrees on the SAME A,b → Ginkgo bug.\n'
    printf '  3. CSR diff zero, standalone Ginkgo agrees → bug is in NeoN↔Ginkgo coupling layer\n'
    printf '     (stream sharing / fence coverage / halo exchange).\n'
    printf '  4. cpu_vs_gpu_sync NO_DIVERGENCE → Option B (cudaDeviceSynchronize) suffices.\n'
} | tee "${SUMMARY}"

log "done — paste ${SUMMARY} back to the debug session"
