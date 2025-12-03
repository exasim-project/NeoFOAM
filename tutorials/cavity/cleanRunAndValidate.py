#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from foamlib import FoamCase

# =========================================================
#  Utility: Run shell commands
# =========================================================
def run(cmd, cwd=None):
    print(f"\n⏳ Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


# =========================================================
#  ALLCLEAN FUNCTIONALITY
# =========================================================
def clean_case(case_path: Path):
    print("\n🧹 Cleaning case (Allclean equivalent)...")

    # Remove time directories except 0/
    for d in case_path.iterdir():
        if d.is_dir():
            try:
                if float(d.name) != 0.0:
                    print(f"  Removing time directory: {d}")
                    run(["rm", "-rf", str(d)])
            except ValueError:
                pass

    # Remove processor directories
    for d in case_path.glob("processor*"):
        print(f"  Removing processor directory: {d}")
        run(["rm", "-rf", str(d)])

    # Remove *.foam
    for f in case_path.glob("*.foam"):
        print(f"  Removing foam file: {f}")
        run(["rm", "-f", str(f)])

    # Remove log files
    for f in case_path.glob("log.*"):
        print(f"  Removing log file: {f}")
        run(["rm", "-f", str(f)])
    for f in case_path.glob("*.log"):
        print(f"  Removing log file: {f}")
        run(["rm", "-f", str(f)])

    # Remove postProcessing
    pp = case_path / "postProcessing"
    if pp.exists():
        print(f"  Removing postProcessing/ directory")
        run(["rm", "-rf", str(pp)])

    print("✔ Allclean complete.\n")


# =========================================================
#  ALLRUN FUNCTIONALITY
# =========================================================
def restore0_dir(case_path: Path):
    zero = case_path / "0"
    zero_orig = case_path / "0.orig"

    if zero.exists():
        print("Removing existing 0/ directory...")
        run(["rm", "-rf", str(zero)])

    if not zero_orig.exists():
        raise RuntimeError("0.orig directory missing. Cannot restore initial fields.")

    print("Restoring 0/ from 0.orig/")
    run(["cp", "-r", str(zero_orig), str(zero)])


def run_case(case_path: Path):
    print("\n Starting Allrun workflow...")

    # 1. touch cavity.foam
    foamfile = case_path / "cavity.foam"
    print("Creating cavity.foam")
    foamfile.touch()

    # 2. restore0Dir
    restore0_dir(case_path)

    # 3. blockMesh
    print("\n Running blockMesh...")
    run(["blockMesh"], cwd=case_path)

    # 4. neoIcoFoam
    solver = case_path / "../../build/profiling/bin/neoIcoFoam"
    print("\n Running solver neoIcoFoam...")
    run([str(solver)], cwd=case_path)

    print("✔ Allrun completed.\n")


# =========================================================
#  CFD FIELD EXTRACTION
# =========================================================
def compute_reynolds(case: FoamCase):
    tp = case["constant"]["transportProperties"]
    nu = float(tp["nu"])
    Re = 1.0 / nu
    print(f"nu = {nu}")
    print(f"Computed Re = {Re}")
    return nu, Re


def detect_latest_time(case_path: Path):
    times = []
    for d in case_path.iterdir():
        if d.is_dir():
            try:
                times.append(float(d.name))
            except ValueError:
                pass
    if not times:
        raise RuntimeError("No time directories found!")
    return max(times)


# =========================================================
#  GHIA TABLE HANDLING
# =========================================================
def load_ghia_header(filename: Path):
    with open(filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("%"):
                parts = stripped.lstrip("%").split()
                nums = []
                for p in parts:
                    try:
                        nums.append(int(p))
                    except ValueError:
                        continue
                if len(nums) >= 2:
                    return np.array(nums)
    raise RuntimeError("Could not extract Re header from Ghia file.")


def select_ghia_column(GHIA_RE_VALUES, Re):
    Re_rounded = GHIA_RE_VALUES[np.argmin(np.abs(GHIA_RE_VALUES - Re))]
    col_index = np.where(GHIA_RE_VALUES == Re_rounded)[0][0] + 1
    print(f"→ Selecting Ghia Re={Re_rounded}, column index={col_index}")
    return Re_rounded, col_index


# =========================================================
#  CENTRELINE EXTRACTION
# =========================================================
def extract_centreline(x, y, U_int, line_value, is_vertical=True):

    if is_vertical:
        dist = np.abs(x - line_value)
        unique_vals = np.unique(np.round(y, 5))
        vals, comp = [], []
        for val in unique_vals:
            mask = np.isclose(y, val, atol=5e-3)
            if np.any(mask):
                i = np.argmin(dist[mask])
                vals.append(y[mask][i])
                comp.append(U_int[mask][i][0])
        idx = np.argsort(vals)
        return np.array(vals)[idx], np.array(comp)[idx]

    else:
        dist = np.abs(y - line_value)
        unique_vals = np.unique(np.round(x, 5))
        vals, comp = [], []
        for val in unique_vals:
            mask = np.isclose(x, val, atol=5e-3)
            if np.any(mask):
                i = np.argmin(dist[mask])
                vals.append(x[mask][i])
                comp.append(U_int[mask][i][1])
        idx = np.argsort(vals)
        return np.array(vals)[idx], np.array(comp)[idx]


# =========================================================
#  PLOTTING
# =========================================================
def plot_u(y_cl, u_cl, GHIA_Y, GHIA_U, Re_rounded, endTime, case_path):
    plt.figure(figsize=(5, 5))
    plt.plot(u_cl, y_cl, "-", label="neoIcoFoam")
    plt.plot(GHIA_U, GHIA_Y, "o", markerfacecolor='none', markeredgecolor='red',
             label=f"Ghia Re={Re_rounded}", markersize=4)
    plt.xlabel("u")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.title(f"Vertical centreline U (Re={Re_rounded}, t={endTime})")
    outfile = case_path / f"centreline_U_Re{Re_rounded}_t{endTime}.pdf"
    plt.savefig(outfile)
    print(f"Saved {outfile}")


def plot_v(x_cl, v_cl, GHIA_X, GHIA_V, Re_rounded, endTime, case_path):
    plt.figure(figsize=(5, 5))
    plt.plot(x_cl, v_cl, "-", label="neoIcoFoam")
    plt.plot(GHIA_X, GHIA_V, "o", markerfacecolor='none', markeredgecolor='red',
             label=f"Ghia Re={Re_rounded}", markersize=4)
    plt.xlabel("x")
    plt.ylabel("v")
    plt.grid(True)
    plt.legend()
    plt.title(f"Horizontal centreline V (Re={Re_rounded}, t={endTime})")
    outfile = case_path / f"centreline_V_Re{Re_rounded}_t{endTime}.pdf"
    plt.savefig(outfile)
    print(f"Saved {outfile}")


# =========================================================
#  MAIN WORKFLOW
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Run cavity test + plot results.")
    parser.add_argument("--clean", action="store_true", help="Clean case only")
    parser.add_argument("--run", action="store_true", help="Run solver only")
    parser.add_argument("--plot", action="store_true", help="Plot only")
    args = parser.parse_args()

    case_path = Path.cwd()
    print(f"Case path: {case_path}")

    # CLEAN ONLY
    if args.clean and not args.run and not args.plot:
        clean_case(case_path)
        return

    # RUN ONLY
    if args.run and not args.clean and not args.plot:
        run_case(case_path)
        return

    # PLOT ONLY
    if args.plot and not args.clean and not args.run:
        pass  # skip cleaning and running

    # DEFAULT: CLEAN → RUN → PLOT
    if not args.plot:
        clean_case(case_path)
        run_case(case_path)

    # Load case and fields
    case = FoamCase(case_path)
    nu, Re = compute_reynolds(case)
    endTime = detect_latest_time(case_path)

    t = case[endTime]
    U = t["U"]
    C = t.cell_centers()
    U_int = np.asarray(U.internal_field)
    C_int = np.asarray(C.internal_field)
    x, y = C_int[:, 0], C_int[:, 1]

    # Load Ghia tables
    ghia_u = np.loadtxt(case_path / "ghia_u.txt", comments="%")
    ghia_v = np.loadtxt(case_path / "ghia_v.txt", comments="%")
    GHIA_Y = ghia_u[:, 0]
    GHIA_X = ghia_v[:, 0]

    GHIA_RE_VALUES = load_ghia_header(case_path / "ghia_u.txt")
    Re_rounded, col_index = select_ghia_column(GHIA_RE_VALUES, Re)

    GHIA_U = ghia_u[:, col_index]
    GHIA_V = ghia_v[:, col_index]

    # Extract CFD centreline data
    y_cl, u_cl = extract_centreline(x, y, U_int, 0.5, is_vertical=True)
    x_cl, v_cl = extract_centreline(x, y, U_int, 0.5, is_vertical=False)

    # Plot
    plot_u(y_cl, u_cl, GHIA_Y, GHIA_U, Re_rounded, endTime, case_path)
    plot_v(x_cl, v_cl, GHIA_X, GHIA_V, Re_rounded, endTime, case_path)

    print("\n✔ Completed run + plot.\n")

if __name__ == "__main__":
    main()
