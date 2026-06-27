#!/bin/bash
#
# Mixed-PRECISION (mp) parameter study for occDrivAreStaticMesh.
#
# Derived from the pMG-localized-cache-rebuild100 setup of param-study-mg.sh: the LOCALIZED
# multigrid pressure preconditioner (Cg + Schwarz{Multigrid(local)}, max_levels=10) run with the
# solver cache ON (cacheSolver=true, preconditionerRebuildInterval=100). Two mixed-precision FAMILIES
# are swept, both against that same reference setup:
#
# A) INNER-SOLVE precision (the `innerPrecision` fvSolution key; see NeoN ginkgo.hpp). When set,
#    NeoN parses the WHOLE p-solver config in the reduced precision and wraps it in an OUTER fp64 Ir
#    (iterative refinement) -- solution + residual stay fp64, the entire inner Cg+MG correction
#    solve runs reduced. irIterations (outer refinement steps) defaults to 1.
#       innerPrecision unset    ->  plain fp64 (double) solve              -> pMG-localized-mp-double-*
#       innerPrecision float    ->  Ir(fp64){ Cg+Schwarz{MG} @ float }      -> pMG-localized-mp-float-*
#       innerPrecision bfloat16 ->  Ir(fp64){ Cg+Schwarz{MG} @ bfloat16 }   -> pMG-localized-mp-bfloat16-*
#    CONFIG: system/gko/p-multigrid-localized-mp.json -- the localized config with its outer stop
#    swapped from neon::l1ScaledResidual to a plain relative ResidualNorm (reduction_factor 0.01).
#    MANDATORY: ginkgo.hpp's NF_ERROR_EXIT forbids innerPrecision together with an l1ScaledResidual-
#    in-config, and the named l1 criterion would be unregistered once the p-block omits
#    `l1ScaledResidual true` -- so the mp p-block carries NO l1 keys.
#
# B) PRECONDITIONER-ONLY precision (the Ginkgo config.json "value_type" override on the
#    preconditioner node; no innerPrecision key). The OUTER solver::Cg, its Krylov vectors, and the
#    l1ScaledResidual stop all stay fp64; only the Schwarz{Multigrid(local)} preconditioner subtree
#    is built in reduced precision. Ginkgo converts the residual to the lower precision at the
#    preconditioner-apply boundary (Schwarz::apply -> precision_dispatch_real_complex_distributed)
#    and back -- a genuinely mixed-precision PRECONDITIONER, not a reduced full solve.
#       value_type float32  ->  Cg(fp64) + Schwarz{MG} @ float32    -> pMG-localized-precfloat-*
#       value_type bfloat16 ->  Cg(fp64) + Schwarz{MG} @ bfloat16   -> pMG-localized-precbf16-*
#    CONFIGS: system/gko/p-multigrid-localized-precfloat.json / -precbf16.json. The l1 criterion is
#    KEPT (valid on the fp64 outer Cg), so these run through the NORMAL configFile path with
#    `l1ScaledResidual true` -- identical fvSolution p-block to cache-rebuild100, only configFile
#    differs. This is the closest analog to the reference run.
#
# In both families U/k/omega keep their native-OpenFOAM l1ScaledResidual (unrelated to Ginkgo).
#
# REQUIREMENTS:
#   * The neoSimpleFoam binary must include the innerPrecision plumbing (NeoN ginkgo.hpp, the
#     Ir<scalar> mixed-precision wrapper) -- rebuild the NEON_BUILD binary if it predates it.
#   * bfloat16 needs GINKGO_ENABLE_BFLOAT16 in the linked ginkgo (both h200 builds have it). If a
#     binary lacks it, the "bfloat16" run errors out in GinkgoSolver -- run just `double float` then.
#
# Solver caching is kept ON to mirror the rebuild100 reference. The mixed-precision factory is an
# Ir<scalar>, which the cache's updateInPlace path handles by forwarding update_matrix_value to the
# Ir's inner solver; if a reduced-precision inner solver does not support in-place update in your
# binary, disable caching for this study with MP_CACHE=false (drops the cacheSolver/rebuild keys).
# report_cache_reuse still tallies the per-solve p-cache diagnostics for the cached runs.
#
# Usage:   ./param-study-mp.sh                 # all five: double float bfloat16 precfloat precbf16
#          ./param-study-mp.sh float           # a single inner-solve precision
#          ./param-study-mp.sh precfloat       # a single preconditioner-only precision
#          ./param-study-mp.sh double precbf16 # selected variants from either family
#          MP_CACHE=false ./param-study-mp.sh  # same sweep, solver cache OFF (regenerate each solve)
#          FORCE=1 ./param-study-mp.sh         # re-run and overwrite existing logs

STUDY_TYPE="${STUDY_TYPE:-mixed-precision}"
source "$(dirname "$0")/param-study-mg-common.sh"

MP_CFG="p-multigrid-localized-mp.json"
MP_REBUILD="${MP_REBUILD:-100}"
MP_CACHE="${MP_CACHE:-true}"

# Family A -- INNER-SOLVE precision sweep as "label:innerPrecisionValue"; an EMPTY value means the
# key is omitted -> fp64 (double). Override with MP_PRECISIONS="label:value ..." before the run.
MP_PRECISIONS=(${MP_PRECISIONS:-
    "double:"
    "float:float"
    "bfloat16:bfloat16"})

# Family B -- PRECONDITIONER-ONLY precision sweep as "label:configBasename". The outer fp64 Cg and
# l1 stop are untouched; only the named config's preconditioner subtree is reduced precision (config
# "value_type"). No innerPrecision key. Override with MP_PRECOND="label:file.json ..." before the run.
MP_PRECOND=(${MP_PRECOND:-
    "precfloat:p-multigrid-localized-precfloat.json"
    "precbf16:p-multigrid-localized-precbf16.json"})

build_mp_fvsolution() {
    # $1 = innerPrecision value ("" => omit, i.e. double)   $2 = output fvSolution path.
    # Reuse case3's U/k/omega/SIMPLE blocks; rewrite ONLY the p{} solver body: point configFile at
    # the mp config, add the cache keys (unless MP_CACHE=false) and innerPrecision, and DROP the
    # template's l1ScaledResidual/tolerance/relTol p-keys (incompatible with innerPrecision). awk is
    # used rather than foamDictionary, which would re-tokenise the "system/gko/..." path into words.
    local prec="$1" out="$2"
    awk -v cfg="$MP_CFG" -v prec="$prec" -v rebuild="$MP_REBUILD" -v cache="$MP_CACHE" '
        BEGIN { inp = 0; body = 0 }
        # Enter the first solver block, the bare "p" entry (not Phi/U/k/omega or the $p; alias).
        inp == 0 && body == 0 && /^[[:space:]]*p[[:space:]]*$/ { inp = 1; print; next }
        # Opening brace of the p block: emit the rewritten body header, then swallow the original.
        inp == 1 && body == 0 && /^[[:space:]]*\{/ {
            print
            print "        configFile       system/gko/" cfg ";"
            if (cache == "true") {
                print "        cacheSolver      true;"
                print "        preconditionerRebuildInterval " rebuild ";"
            }
            if (prec != "") print "        innerPrecision   " prec ";"
            body = 1
            next
        }
        # Closing brace ends the p block; drop everything in between (old configFile/comments/l1).
        inp == 1 && body == 1 && /^[[:space:]]*\}/ { print; inp = 0; body = 0; next }
        inp == 1 && body == 1 { next }
        { print }
    ' "$MG_TEMPLATE" > "$out"
}

run_mp() {
    # $1 = label (double|float|bfloat16)   $2 = innerPrecision value ("" for double).
    ensure_mg_variants
    local label="$1" prec="$2"
    local tag; if [ "$MP_CACHE" = "true" ]; then tag="-cache-rebuild${MP_REBUILD}"; else tag="-nocache"; fi
    local name="pMG-localized-mp-${label}${tag}"
    if [ ! -f "system/gko/$MP_CFG" ]; then
        echo "!! no mp config 'system/gko/$MP_CFG' -- ensure_mg_variants should generate it; skipping $name"
        return
    fi
    build_mp_fvsolution "$prec" "$TMP_FVSOL"
    run_one "$name" "$TMP_FVSOL" \
        "p = Ginkgo LOCALIZED MG ($MP_CFG), innerPrecision=${prec:-double}, cache=${MP_CACHE}, rebuildInterval=${MP_REBUILD}"
}

build_precond_fvsolution() {
    # $1 = config basename   $2 = output fvSolution path.
    # Preconditioner-only family: KEEP the template's l1ScaledResidual/tolerance/relTol p-keys (the
    # outer Cg is fp64, so the l1 criterion is valid and wanted) and only swap the configFile path,
    # plus the cache keys when MP_CACHE=true. Pure sed on the configFile line (the same approach as
    # the cache study's run_cache_interval) -- NOT foamDictionary, which re-tokenises the path.
    local cfg="$1" out="$2"
    if [ "$MP_CACHE" = "true" ]; then
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;\n        cacheSolver      true;\n        preconditionerRebuildInterval $MP_REBUILD;#" \
            "$MG_TEMPLATE" > "$out"
    else
        sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1system/gko/$cfg;#" "$MG_TEMPLATE" > "$out"
    fi
}

run_precond() {
    # $1 = label (precfloat|precbf16)   $2 = config basename.
    ensure_mg_variants
    local label="$1" cfg="$2"
    local tag; if [ "$MP_CACHE" = "true" ]; then tag="-cache-rebuild${MP_REBUILD}"; else tag="-nocache"; fi
    local name="pMG-localized-${label}${tag}"
    if [ ! -f "system/gko/$cfg" ]; then
        echo "!! no preconditioner-precision config 'system/gko/$cfg' -- ensure_mg_variants should generate it; skipping $name"
        return
    fi
    build_precond_fvsolution "$cfg" "$TMP_FVSOL"
    run_one "$name" "$TMP_FVSOL" \
        "p = Ginkgo LOCALIZED MG, fp64 Cg + reduced-precision preconditioner ($cfg), cache=${MP_CACHE}, rebuildInterval=${MP_REBUILD}"
}

run_mp_named() {
    # Map a keyword to its entry in either family and run it.
    local key="$1" e label val matched=0
    for e in "${MP_PRECISIONS[@]}"; do
        label="${e%%:*}"; val="${e#*:}"; [ "$val" = "$e" ] && val=""   # entry with no ":" => empty
        if [ "$key" = "$label" ] || [ "$key" = "$val" ]; then
            run_mp "$label" "$val"; matched=1; break
        fi
    done
    [ "$matched" -eq 1 ] && return
    for e in "${MP_PRECOND[@]}"; do
        label="${e%%:*}"; val="${e#*:}"
        if [ "$key" = "$label" ]; then run_precond "$label" "$val"; matched=1; break; fi
    done
    [ "$matched" -eq 0 ] && \
        echo "!! unknown precision '$key' (use: double float bfloat16 precfloat precbf16)"
}

ensure_mg_variants

SEL=("$@")
if [ ${#SEL[@]} -eq 0 ]; then
    for e in "${MP_PRECISIONS[@]}"; do
        label="${e%%:*}"; val="${e#*:}"; [ "$val" = "$e" ] && val=""
        run_mp "$label" "$val"
    done
    for e in "${MP_PRECOND[@]}"; do
        run_precond "${e%%:*}" "${e#*:}"
    done
else
    for s in "${SEL[@]}"; do run_mp_named "$s"; done
fi

print_summary
report_cache_reuse
