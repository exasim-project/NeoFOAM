STUDY_TYPE=sc-modes
STEPS=50
export MPIRUN_FORWARD_ENV="NEON_MGSC_MODE"
source ./paper-study-common.sh
require_restart
BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"
CFG="system/gko/scmode-papertmp.json"
trap 'rm -f "$CFG"' EXIT
# scale_correction=true for all; NEON_MGSC_MODE selects which passes actually run.
jq --argjson ml 4 --argjson tol 0.1 '
  .preconditioner.max_levels=$ml | .preconditioner.scale_correction=true
  | .preconditioner.coarsest_solver.local_solver.criteria=[
      {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":$tol},
      {"type":"Iteration","max_iters":50}]' "$BASE_CFG" > "$CFG"
sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
    "$MG_TEMPLATE" > "$TMP_FVSOL"
for m in both pre post none; do
  export NEON_MGSC_MODE="$m"
  run_one "sc-$m" "$TMP_FVSOL" "L4 tol0.1 merge2 lcg, NEON_MGSC_MODE=$m"
done
unset NEON_MGSC_MODE
print_summary
echo "sc-modes done"
