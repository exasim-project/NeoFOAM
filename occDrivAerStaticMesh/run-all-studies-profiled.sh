#!/bin/bash
#
# Run every occDrivAer parameter study EXCEPT the production run, each with the
# Kokkos-Tools space-time-stack profiler enabled (KOKKOS_TOOL=space-time-stack).
#
# The studies are discovered by globbing param-study*.sh and dropping the scripts
# that are not standalone studies:
#   - param-study-common.sh / param-study-mg-common.sh : sourced libraries, not run directly
#   - param-study-table.sh                             : the summary-table reporter (a consumer)
#   - param-study-production.sh                        : the full production run (excluded by request)
#   - this runner itself
# So a newly added param-study-<name>.sh is picked up automatically.
#
# Studies run SEQUENTIALLY (each rewrites system/fvSolution and marches the case, so they
# must not overlap) and the runner CONTINUES past a failing study, reporting a pass/fail
# summary at the end. The profiler's space-time-stack summary is written into each per-run
# log by param-study-common.sh's run path (KOKKOS_TOOLS_LIBS forwarded to the MPI ranks).
#
# Every case is FORCE-rerun (FORCE=1 below), so existing logs are NOT skipped -- a profiling sweep
# always produces a fresh instrumented run of all cases.
#
# Usage:   ./run-all-studies-profiled.sh
#          KOKKOS_TOOL=memory-high-water-mark ./run-all-studies-profiled.sh   # different tool
#          NEON_BUILD=production ./run-all-studies-profiled.sh                 # scalecorr/mp need this
#          FORCE= ./run-all-studies-profiled.sh                               # restore skip-if-log-exists
#
# NOTE: param-study-mg.sh's scalecorr / scalecorr-localized variants and param-study-mp.sh
# require the PRODUCTION build (NEON_BUILD=production); with the default profiling build those
# variants are expected to fail/skip. Per-study env (NP, STEPS, FORCE, ...) is inherited as usual.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# Profiler tool to load for every solver launch (overridable). Exported so the child studies,
# which read KOKKOS_TOOL via param-study-common.sh, pick it up.
export KOKKOS_TOOL="${KOKKOS_TOOL:-space-time-stack}"

# Force every case to (re)run, even when a matching log already exists -- a profiling sweep wants a
# fresh, fully-instrumented run of all cases, not the skip-if-log-exists gap-fill behaviour of
# run_one. Exported so all child studies see it. Set FORCE= (empty) to restore the skip behaviour.
export FORCE="${FORCE:-1}"

self="$(basename "${BASH_SOURCE[0]}")"
# Scripts that are NOT standalone studies (sourced libs, the reporter, production, this runner).
EXCLUDE=(
    param-study-common.sh
    param-study-mg-common.sh
    param-study-table.sh
    param-study-production.sh
    param-study-production-mp.sh
    param-study-production-tol.sh
    "$self"
)

is_excluded() {
    local s="$1" e
    for e in "${EXCLUDE[@]}"; do [ "$s" = "$e" ] && return 0; done
    return 1
}

# Collect the studies to run (sorted, deterministic order).
studies=()
for f in param-study*.sh; do
    [ -f "$f" ] || continue          # no match -> literal glob, skip
    is_excluded "$f" && continue
    studies+=("$f")
done

if [ ${#studies[@]} -eq 0 ]; then
    echo "!! no param-study*.sh studies found to run"; exit 1
fi

echo "################################################################"
echo " running ${#studies[@]} studies with KOKKOS_TOOL=$KOKKOS_TOOL"
echo " (production excluded)"
printf '   - %s\n' "${studies[@]}"
echo "################################################################"

declare -A RC
fail=0
for s in "${studies[@]}"; do
    echo
    echo "================================================================"
    echo ">>> $s   (KOKKOS_TOOL=$KOKKOS_TOOL)"
    echo "================================================================"
    # Run as its own process so each study's module/OpenFOAM setup and controlDict
    # restore-trap stay isolated. No args -> each runs its default sweep.
    bash "./$s"
    rc=$?
    RC["$s"]=$rc
    [ "$rc" -ne 0 ] && fail=1
    echo "<<< $s exit=$rc"
done

echo
echo "######################## RUN-ALL SUMMARY ########################"
for s in "${studies[@]}"; do
    printf "%-28s %s\n" "$s" "$([ "${RC[$s]}" -eq 0 ] && echo OK || echo "FAILED (exit ${RC[$s]})")"
done
echo "################################################################"
exit "$fail"
