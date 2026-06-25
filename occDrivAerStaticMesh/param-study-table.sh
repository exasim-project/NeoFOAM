#!/bin/bash
#
# Summary-table generator for the occDrivAer parameter studies.
#
# Scans every per-run log under paramStudy/results/ (written by param-study.sh and
# param-study-mg.sh via param-study-common.sh) and prints one consolidated table:
#
#   steps      number of "Time = N" steps the run reached
#   status     OK (finalised) | CRASH@<step> (signal 8 / FPE) | NaN@<step> |
#              RUN@<step> (solver still alive) | part@<step> | empty
#   p_it/slv   mean "No Iterations" over all "Solving for p," lines
#   clk_s/step mean delta between consecutive ClockTime values (wall s per timestep)
#   clk_tot    last ClockTime reported (s)
#   avg|glob|  mean |global| time-step continuity error over the run
#   last_local last "sum local" continuity error
#   ~<N>steps  projected wall time for N timesteps = clk_s/step * N (default 3000)
#
# Rows are newest-log-first; runs whose log was touched < AGE_MIN ago are tagged
# CURR (this run), the rest prev (an earlier run).
#
# Usage:   ./param-study-table.sh              # N = 3000 projection
#          ./param-study-table.sh 5000         # project to 5000 steps
#          PROJECT_STEPS=10000 ./param-study-table.sh
#          RESULTS=some/other/dir ./param-study-table.sh

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

RESULTS="${RESULTS:-paramStudy/results}"
PROJECT_STEPS="${PROJECT_STEPS:-${1:-3000}}"
AGE_MIN="${AGE_MIN:-60}"          # logs newer than this (minutes) are tagged CURR

[ -d "$RESULTS" ] || { echo "!! no results dir '$RESULTS'"; exit 1; }

now=$(date +%s)

# Pull every metric for a single log; echoes one tab-separated record.
row_for() {
    local f="$1" name; name="$(basename "$f" .log)"
    local steps stp_status pit clkstep clktot avgglob lastloc proj when mt age

    steps=$(grep -c '^Time = ' "$f")

    # status: crash (signal 8) > NaN > finalised > still-running > partial > empty
    if grep -q 'signal 8' "$f"; then
        stp_status="CRASH@$steps"
    elif grep -q 'nan' "$f"; then
        stp_status="NaN@$steps"
    elif grep -q 'Finalising parallel run' "$f"; then
        stp_status="OK"
    elif [ "$steps" -eq 0 ]; then
        stp_status="empty"
    elif pgrep -f "$name" >/dev/null 2>&1; then
        stp_status="RUN@$steps"
    else
        stp_status="part@$steps"
    fi

    # mean pressure-solver iterations per solve + total p "Solve time" (seconds)
    local psolve_s
    read -r pit psolve_s < <(awk '/Solving for p,/{
                 if (match($0,/No Iterations [0-9]+/)) s+=substr($0,RSTART+14,RLENGTH-14)
                 if (match($0,/Solve time = [0-9.]+/)) t+=substr($0,RSTART+13,RLENGTH-13)  # ms
                 n++ }
               END { if (n) printf "%.0f %.3f", s/n, t/1000; else printf "- -" }' "$f")

    # mean ClockTime delta per step + last ClockTime
    read -r clkstep clktot < <(awk '
        /ClockTime = /{
            if (match($0,/ClockTime = [0-9.]+/)) c=substr($0,RSTART+12,RLENGTH-12)
            if (pc!="") { d+=c-pc; n++ } ; pc=c; last=c }
        END { if (n) printf "%.2f %.0f", d/n, last; else printf "- %s", (last==""?"-":last) }' "$f")

    # fraction of total ClockTime spent in the pressure solve
    local ppct
    if [ "$psolve_s" != "-" ] && [ "$clktot" != "-" ] && [ "$clktot" != "0" ]; then
        ppct=$(awk -v p="$psolve_s" -v c="$clktot" 'BEGIN{ printf "%.0f%%", 100*p/c }')
    else
        ppct="-"
    fi

    # mean |global| continuity error (NaN-aware)
    avgglob=$(grep 'continuity errors' "$f" \
        | sed -E 's/.*global = (-?nan|-?[0-9.]+([eE][+-]?[0-9]+)?).*/\1/' \
        | awk '{ if ($1 ~ /nan/) { h=1; next }
                 a=$1; if (a<0) a=-a; s+=a; n++ }
               END { if (n) printf "%.2e", s/n; else printf "%s", (h?"NaN":"-") }')

    # last "sum local" continuity error
    lastloc=$(grep 'sum local' "$f" | tail -1 \
        | sed -E 's/.*sum local = (-?nan|-?[0-9.]+([eE][+-]?[0-9]+)?).*/\1/')
    [ -z "$lastloc" ] && lastloc="-"

    # projected wall time for PROJECT_STEPS (hours), from clk_s/step
    if [ "$clkstep" != "-" ]; then
        proj=$(awk -v s="$clkstep" -v n="$PROJECT_STEPS" 'BEGIN{ printf "%.1fh", s*n/3600 }')
    else
        proj="-"
    fi

    mt=$(stat -c %Y "$f"); age=$(( (now - mt) / 60 ))
    when=$([ "$age" -lt "$AGE_MIN" ] && echo CURR || echo prev)

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$steps" "$stp_status" "$pit" "$clkstep" "$clktot" "$ppct" \
        "$avgglob" "$lastloc" "$proj" "$when"
}

hdr_proj="~${PROJECT_STEPS}st"

# Sort order: completed (OK) first, then running (RUN), then crashed (CRASH),
# then everything else (part/NaN/empty = old/failed). Within each group, lowest
# clk_s/st first ("-" sorts last). A two-column numeric key is prepended for the
# sort and stripped again before the table is rendered.
sort_key() {
    # $1 = status, $2 = clk_s/st  ->  "<group>\t<clk-or-inf>"
    local g clk
    case "$1" in
        OK)     g=1 ;;
        RUN@*)  g=2 ;;
        CRASH@*) g=3 ;;
        *)      g=4 ;;
    esac
    clk="$2"; [ "$clk" = "-" ] && clk=999999
    printf '%s\t%s' "$g" "$clk"
}

{
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        run steps status p_it/slv clk_s/st clk_tot p_solve% 'avg|glob|' last_local "$hdr_proj" when
    for f in "$RESULTS"/*.log; do
        [ -f "$f" ] || continue
        rec=$(row_for "$f")
        status=$(printf '%s' "$rec" | cut -f3)
        clk=$(printf '%s' "$rec" | cut -f5)
        printf '%s\t%s\n' "$(sort_key "$status" "$clk")" "$rec"
    done | sort -t $'\t' -k1,1n -k2,2g | cut -f3-
} | column -t -s $'\t'

echo
echo "results: $RESULTS/   projection: ${PROJECT_STEPS} steps   CURR = log < ${AGE_MIN} min old"
