#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from foamlib import FoamCase


# ---------------------------------------------------------
# Centreline extraction using nearest-cell method
# ---------------------------------------------------------
def extract_centreline(x, y, U_int, line_value, is_vertical=True):
    if is_vertical:
        # u(y) at x = line_value
        dist = np.abs(x - line_value)
        unique_y = np.unique(np.round(y, 5))

        y_list, u_list = [], []
        for y0 in unique_y:
            mask_y = np.isclose(y, y0, atol=5e-3)
            if np.any(mask_y):
                i = np.argmin(dist[mask_y])
                y_list.append(y[mask_y][i])
                u_list.append(U_int[mask_y][i][0])  # u-component

        y_arr = np.array(y_list)
        u_arr = np.array(u_list)
        idx = np.argsort(y_arr)
        return y_arr[idx], u_arr[idx]

    else:
        # v(x) at y = line_value
        dist = np.abs(y - line_value)
        unique_x = np.unique(np.round(x, 5))

        x_list, v_list = [], []
        for x0 in unique_x:
            mask_x = np.isclose(x, x0, atol=5e-3)
            if np.any(mask_x):
                i = np.argmin(dist[mask_x])
                x_list.append(x[mask_x][i])
                v_list.append(U_int[mask_x][i][1])  # v-component

        x_arr = np.array(x_list)
        v_arr = np.array(v_list)
        idx = np.argsort(x_arr)
        return x_arr[idx], v_arr[idx]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    case_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(f"Case path: {case_path}")

    # ---------------------------------------------------------
    # Load CFD case and read nu from transportProperties
    # ---------------------------------------------------------
    case = FoamCase(case_path)
    tp = case["constant"]["transportProperties"]

    # Requires entry: nu 1e-3;
    nu = float(tp["nu"])
    Re = 1.0 / nu  # For lid = 1 m/s and cavity size = 1 m
    print(f"nu     = {nu}")
    print(f"Re     = {Re}")

    # ---------------------------------------------------------
    # Detect endTime (latest time directory)
    # ---------------------------------------------------------
    times = []
    for d in case_path.iterdir():
        if d.is_dir():
            try:
                times.append(float(d.name))
            except ValueError:
                continue

        if not times:
            raise RuntimeError("No numeric time directories found.")
    
    endTime = max(times)
    print(f"Detected endTime = {endTime}")

    # Load that time directory through foamlib
    t = case[endTime]

    U = t["U"]
    C = t.cell_centers()

    U_int = np.asarray(U.internal_field)
    C_int = np.asarray(C.internal_field)
    x = C_int[:, 0]
    y = C_int[:, 1]

    # ---------------------------------------------------------
    # Load Ghia full tables (all Re values)
    # ---------------------------------------------------------
    ghia_u = np.loadtxt(case_path / "ghia_u.txt", comments="%")
    ghia_v = np.loadtxt(case_path / "ghia_v.txt", comments="%")

    GHIA_Y = ghia_u[:, 0]   # first column is y
    GHIA_X = ghia_v[:, 0]   # first column is x


    # ---------------------------------------------------------
    # Parse header from ghia_u.txt to get available Re values
    # ---------------------------------------------------------
    GHIA_RE_VALUES = None

    with open(case_path / "ghia_u.txt", "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("%"):
                parts = stripped.lstrip("%").split()
                # Look for header containing at least 2 numeric entries (e.g. Re values)
                numeric = []
                for p in parts:
                    try:
                        numeric.append(int(p))
                    except ValueError:
                        continue
                if len(numeric) >= 2:       # this is the real column header
                    GHIA_RE_VALUES = np.array(numeric)
                    break

    if GHIA_RE_VALUES is None:
        raise RuntimeError("Could not detect Re columns from ghia_u.txt")

    print("Available Re values from Ghia et al.:", GHIA_RE_VALUES)

    # ---------------------------------------------------------
    # Select the closest Ghia Re column to the CFD Re
    # ---------------------------------------------------------
    Re_rounded = GHIA_RE_VALUES[np.argmin(np.abs(GHIA_RE_VALUES - Re))]
    print(f"Selected Ghia Re column: {Re_rounded}")

    col_index = np.where(GHIA_RE_VALUES == Re_rounded)[0][0] + 1
    #                +1 because col 0 is x/y

    # ---------------------------------------------------------
    # Extract the correct Re column from Ghia files
    # ---------------------------------------------------------
    GHIA_U = ghia_u[:, col_index]
    GHIA_V = ghia_v[:, col_index]

    # ======================================================
    # 1) U velocity: u(y) at x = 0.5
    # ======================================================
    y_cl, u_cl = extract_centreline(x, y, U_int, line_value=0.5, is_vertical=True)

    plt.figure(figsize=(5, 5))
    plt.plot(u_cl, y_cl, "-", label="neoIcoFoam")
    plt.plot(GHIA_U, GHIA_Y, "o", markerfacecolor='none', markeredgecolor='red', label=f"Ghia et al. (Re={Re_rounded})", markersize=4)
    plt.xlabel("u")
    plt.ylabel("y")
    plt.title(f"Vertical centreline U velocity (Re={Re_rounded}, t={endTime})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outU = case_path / f"centreline_U_Re{Re_rounded}_t{endTime}.pdf"
    plt.savefig(outU)
    print(f"Saved: {outU}")

    # ======================================================
    # 2) V velocity: v(x) at y = 0.5
    # ======================================================
    x_cl, v_cl = extract_centreline(x, y, U_int, line_value=0.5, is_vertical=False)

    plt.figure(figsize=(5, 5))
    plt.plot(x_cl, v_cl, "-", label="neoIcoFoam")
    plt.plot(GHIA_X, GHIA_V, "o", markerfacecolor='none', markeredgecolor='red', label=f"Ghia et al. (Re={Re_rounded})", markersize=4)
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(f"Horizontal centreline V velocity (Re={Re_rounded}, t={endTime})")
    plt.grid(True)
    plt.tight_layout()
    outV = case_path / f"centreline_V_Re{Re_rounded}_t{endTime}.pdf"
    plt.savefig(outV)
    print(f"Saved: {outV}")

if __name__ == "__main__":
    main()
