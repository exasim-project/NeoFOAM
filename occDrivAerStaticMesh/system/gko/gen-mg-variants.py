#!/usr/bin/env python3
"""Generate Ginkgo multigrid-preconditioner variants for the occDrivAer p-solver
parameter study, from the production p-multigrid.json base.

Grid:
    max_levels        in {2, 4, 10, 15}
    scale_correction  in {0, 2}   (applied to BOTH pre_smoother and post_smoother)

coarsest_solver.scale_correction is left untouched. Output files:
    p-multigrid.L<levels>.sc<scale>.json
"""
import json
import os
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "p-multigrid.json")

MAX_LEVELS = [2, 4, 10, 15]
SCALE_CORRECTION = [0, 2]

with open(BASE) as f:
    base = json.load(f)

written = []
for levels, scale in product(MAX_LEVELS, SCALE_CORRECTION):
    cfg = json.loads(json.dumps(base))  # deep copy
    pre = cfg["preconditioner"]
    pre["max_levels"] = levels
    pre["pre_smoother"][0]["scale_correction"] = scale
    pre["post_smoother"][0]["scale_correction"] = scale

    name = f"p-multigrid.L{levels}.sc{scale}.json"
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")
    written.append(name)

print(f"wrote {len(written)} variants:")
for n in written:
    print(" ", n)
