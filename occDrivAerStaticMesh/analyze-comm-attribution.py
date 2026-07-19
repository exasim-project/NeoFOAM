#!/usr/bin/env python3
"""Attribute the MPI communication of the pressure solve: smoother/transfers vs scale correction.

Reads the sc=ON / sc=OFF nsys traces from investigate-comm-attribution.sh and normalizes collective
counts PER OUTER CG ITERATION (= per V-cycle), because scale correction changes the outer iteration
count (§4.11b) and so raw totals are not comparable.

  Alltoallv  = distributed-SpMV halo exchange (RowGatherer). Emitted by the smoother's residual,
               restriction, prolongation, the coarse solve, AND scale correction's A*delta.
  Allreduce  = dot / norm. Emitted by outer Cg, stopping criteria, coarse Cg dots, AND scale
               correction's Rayleigh dots sf=(d.b)/(d.Ad).

  per-V-cycle(sc=ON) - per-V-cycle(sc=OFF)  ==  scale correction's own communication.
"""
import glob
import os
import re
import sqlite3
import subprocess
import sys

RESULTS = "paperParamStudyResults/comm-attribution"


def export_sqlite(rep):
    db = rep.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(db):
        print(f"  exporting {os.path.basename(rep)} -> sqlite ...")
        subprocess.run(["nsys", "export", "-t", "sqlite", "-f", "true", "-o", db, rep],
                       check=False, capture_output=True)
    return db if os.path.exists(db) else None


def collectives(db):
    """name -> (count, total_ms)"""
    c = sqlite3.connect(db)
    try:
        rows = list(c.execute("""SELECT s.value, COUNT(*), SUM(e.end-e.start)/1e6
                                 FROM MPI_COLLECTIVES_EVENTS e JOIN StringIds s ON e.textId=s.id
                                 GROUP BY s.value"""))
    except sqlite3.Error as ex:
        print(f"  !! query failed on {db}: {ex}")
        return {}
    finally:
        c.close()
    return {n: (cnt, ms or 0.0) for n, cnt, ms in rows}


def outer_iters(log):
    """Total outer CG iterations across all pressure solves in the window (the V-cycle count)."""
    tot, solves = 0, 0
    with open(log, errors="ignore") as fh:
        for line in fh:
            if "Solving for p," in line:
                m = re.search(r"No Iterations (\d+)", line)
                if m:
                    tot += int(m.group(1))
                    solves += 1
    return tot, solves


def find(variant, ext):
    g = sorted(glob.glob(f"{RESULTS}/{variant}-nsysmpi-*.rank0.{ext}"), reverse=True)
    return g[0] if g else None


def main():
    out = {}
    for v in ("scon", "scoff"):
        rep = find(v, "nsys-rep")
        if not rep:
            print(f"!! no trace for {v} -- run ./investigate-comm-attribution.sh first")
            return 1
        db = export_sqlite(rep) or find(v, "sqlite")
        if not db:
            print(f"!! no sqlite for {v}")
            return 1
        logs = sorted(glob.glob(f"{RESULTS}/{v}-2026*.log"), reverse=True)
        if not logs:
            print(f"!! no log for {v}")
            return 1
        vcycles, solves = outer_iters(logs[0])
        out[v] = (collectives(db), vcycles, solves)
        print(f"{v}: {solves} pressure solves, {vcycles} total outer iters (V-cycles)")

    print("\n=== collectives per V-cycle (rank0) ===")
    print(f"{'collective':16s} {'sc=ON':>18s} {'sc=OFF':>18s} {'delta = SCALE CORR':>20s}")
    for name in ("MPI_Alltoallv", "MPI_Allreduce"):
        on_c, on_v, _ = out["scon"]
        off_c, off_v, _ = out["scoff"]
        on_n = on_c.get(name, (0, 0))[0] / max(on_v, 1)
        off_n = off_c.get(name, (0, 0))[0] / max(off_v, 1)
        on_ms = on_c.get(name, (0, 0))[1] / max(on_v, 1)
        off_ms = off_c.get(name, (0, 0))[1] / max(off_v, 1)
        print(f"{name:16s} {on_n:8.1f} /{on_ms:7.2f}ms {off_n:8.1f} /{off_ms:7.2f}ms "
              f"{on_n-off_n:+8.1f} /{on_ms-off_ms:+7.2f}ms")

    print("\n=== raw totals (NOT comparable -- sc changes the outer iteration count) ===")
    for v in ("scon", "scoff"):
        c, vc, s = out[v]
        tot = {k: c.get(k, (0, 0)) for k in ("MPI_Alltoallv", "MPI_Allreduce")}
        print(f"  {v:6s} V-cycles={vc:5d}  " +
              "  ".join(f"{k.replace('MPI_',''):10s} n={n:6d} {ms:8.1f}ms" for k, (n, ms) in tot.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
