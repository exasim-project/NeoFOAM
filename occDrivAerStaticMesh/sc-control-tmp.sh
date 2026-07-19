STUDY_TYPE=sc-control
STEPS=50
source ./paper-study-common.sh
require_restart
BASE_CFG="system/gko/p-multigrid-mgsc-merge2-lcg.json"
SWEEP_CFG="system/gko/sccontrol-papertmp.json"
trap 'rm -f "$SWEEP_CFG"' EXIT
for sc in false true; do
  jq --argjson ml 4 --argjson tol 0.1 --argjson sc $sc '
    .preconditioner.max_levels=$ml | .preconditioner.scale_correction=$sc
    | .preconditioner.coarsest_solver.local_solver.criteria=[
        {"type":"ResidualNorm","baseline":"initial_resnorm","reduction_factor":$tol},
        {"type":"Iteration","max_iters":50}]' "$BASE_CFG" > "$SWEEP_CFG"
  sed -E "s#^([[:space:]]+configFile[[:space:]]+).*#\1${SWEEP_CFG};\n        cacheSolver      true;\n        preconditionerRebuildInterval 0;#" \
      "$MG_TEMPLATE" > "$TMP_FVSOL"
  n=$([ "$sc" = "true" ] && echo scON || echo scOFF)
  run_one "$n" "$TMP_FVSOL" "L4 tol0.1 merge2 lcg, scale_correction=$sc (clean, no nsys)"
done
print_summary
echo "sc-control done"
