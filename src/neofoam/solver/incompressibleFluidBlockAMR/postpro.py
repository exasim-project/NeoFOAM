# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Post-processing observables for the blockAMR cylinder case (Spec 03).

The single source of the observable definitions Spec 04's cross-solver
comparison consumes:

* :func:`force_coefficients` — drag/lift coefficients ``(Cd, Cl)``. The direct-
  forcing immersed body records its reaction force every step (read via the
  ``DirectForcing`` method's ``force_history`` accessor on the mesh-owned
  IBM data); that momentum-removal integral *is* the hydrodynamic force on
  the body, so it is the primary estimator. A
  control-volume momentum-deficit survey (:func:`momentum_deficit_forces`) over
  a wake box is the field-only fallback (and what the canned INT-8 test drives).
* :func:`strouhal` — shedding frequency ``St = f D / U∞`` from the FFT peak of a
  lift-history series.
* :func:`recirculation_length` — wake-centreline ``u = 0`` crossing behind the
  body, normalised by ``D``.
* :func:`separation_angle` — surface separation angle ``θs`` (degrees from the
  rear stagnation point) from the sign change of the near-wall tangential
  velocity.

The physics operates on plain numpy arrays (:class:`FieldSnapshot`), so it is
GPU-free and unit-testable on canned fields; :func:`gather_field` is the only
Context-touching entry point and imports ``neon`` lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class FieldSnapshot:
    """Dense, level-0 cell-centred fields + geometry marshalled from the Context.

    ``u``/``v``/``w``/``p`` are ``(nx, ny, nz)`` arrays over the valid cells;
    ``x``/``y``/``z`` the 1-D cell-centre coordinate axes; ``solid`` the boolean
    immersed-body mask (``None`` when there is no body).
    """

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float
    solid: Optional[np.ndarray] = None


def gather_field(ctx: Any, level: int = 0) -> FieldSnapshot:
    """Marshal the Context's level-``level`` ``U``/``p`` into a dense snapshot.

    Assembles the per-box valid regions into a single regular array using each
    box's global lower index, and (re)builds the solid mask analytically from
    the mesh's ``body`` on the same grid. ``neon`` is imported lazily so
    importing this module stays GPU-free.
    """
    import blockamr  # noqa: PLC0415 — lazy: keep import GPU-free

    mesh = ctx.fields["U"].mesh
    geom = mesh.geom(level)
    dx = [float(v) for v in geom.cell_size()]
    lo = [float(v) for v in geom.prob_lo()]
    hi = [float(v) for v in geom.prob_hi()]
    nx, ny, nz = (int(round((hi[d] - lo[d]) / dx[d])) for d in range(3))

    u = np.zeros((nx, ny, nz), dtype=float)
    v = np.zeros((nx, ny, nz), dtype=float)
    w = np.zeros((nx, ny, nz), dtype=float)
    p = np.zeros((nx, ny, nz), dtype=float)

    umf = ctx.fields["U"].mf[level]
    for mfi in blockamr.MFIterator(umf):
        arr = np.asarray(umf.copy_to_host(mfi))  # (bx, by, bz, 3) valid region
        s = mfi.valid_box().small_end()
        bx, by, bz = arr.shape[:3]
        sl = (slice(s[0], s[0] + bx), slice(s[1], s[1] + by), slice(s[2], s[2] + bz))
        u[sl] = arr[..., 0]
        v[sl] = arr[..., 1]
        w[sl] = arr[..., 2]

    pmf = ctx.fields["p"].mf[level]
    for mfi in blockamr.MFIterator(pmf):
        arr = np.asarray(pmf.copy_to_host(mfi))  # (bx, by, bz, 1)
        s = mfi.valid_box().small_end()
        bx, by, bz = arr.shape[:3]
        sl = (slice(s[0], s[0] + bx), slice(s[1], s[1] + by), slice(s[2], s[2] + bz))
        p[sl] = arr[..., 0]

    x = lo[0] + (np.arange(nx) + 0.5) * dx[0]
    y = lo[1] + (np.arange(ny) + 0.5) * dx[1]
    z = lo[2] + (np.arange(nz) + 0.5) * dx[2]

    solid: Optional[np.ndarray] = None
    body = mesh.body
    if body is not None:
        center = [float(c) for c in body.centre]
        radius = float(body.radius)
        axis = int(body.axis)
        plane = [a for a in range(3) if a != axis]
        coords = [x, y, z]
        grid = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        d2 = (grid[plane[0]] - center[plane[0]]) ** 2 + (
            grid[plane[1]] - center[plane[1]]
        ) ** 2
        solid = d2 < radius * radius

    return FieldSnapshot(u, v, w, p, x, y, z, dx[0], dx[1], dx[2], solid)


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------


def momentum_deficit_forces(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    cv: tuple[int, int, int, int],
    *,
    rho: float = 1.0,
    nu: float = 0.0,
) -> tuple[float, float]:
    """Control-volume momentum balance for the force on a body inside ``cv``.

    ``u``/``v`` are the ``(nx, ny)`` in-plane velocity components (z already
    averaged out) and ``p`` the (kinematic) pressure. ``cv = (iw, ie, js, jn)``
    are the west/east/south/north **cell indices** of a rectangular control
    volume enclosing the body. Returns ``(Fx, Fy)`` per unit span — the reaction
    force on the body — from the steady momentum flux + pressure balance on the
    CV faces (the normal viscous term is included; tangential viscous traction on
    faces placed in smooth flow is negligible and dropped).

    ``Fx = ∮ (ρ u (V·n) + p n_x - 2μ ∂u/∂x n_x) dS`` over the CV boundary.
    """
    iw, ie, js, jn = cv
    mu = rho * nu

    yc = y[js : jn + 1]
    xc = x[iw : ie + 1]

    def dudx(i: int) -> np.ndarray:
        return np.asarray((u[i + 1, js : jn + 1] - u[i - 1, js : jn + 1]) / (2.0 * dx))

    # West/East faces (normal ±x): flux of x-momentum + pressure + normal viscous
    fx_w = np.trapezoid(
        rho * u[iw, js : jn + 1] ** 2 + p[iw, js : jn + 1] - 2.0 * mu * dudx(iw), yc
    )
    fx_e = np.trapezoid(
        rho * u[ie, js : jn + 1] ** 2 + p[ie, js : jn + 1] - 2.0 * mu * dudx(ie), yc
    )
    # South/North faces (normal ±y): flux of x-momentum (ρ u v)
    fx_s = np.trapezoid(rho * u[iw : ie + 1, js] * v[iw : ie + 1, js], xc)
    fx_n = np.trapezoid(rho * u[iw : ie + 1, jn] * v[iw : ie + 1, jn], xc)
    Fx = (fx_w - fx_e) + (fx_s - fx_n)

    def dvdy(j: int) -> np.ndarray:
        return np.asarray((v[iw : ie + 1, j + 1] - v[iw : ie + 1, j - 1]) / (2.0 * dy))

    # y-momentum: west/east carry ρ u v; south/north carry ρ v² + p + normal visc
    fy_w = np.trapezoid(rho * u[iw, js : jn + 1] * v[iw, js : jn + 1], yc)
    fy_e = np.trapezoid(rho * u[ie, js : jn + 1] * v[ie, js : jn + 1], yc)
    fy_s = np.trapezoid(
        rho * v[iw : ie + 1, js] ** 2 + p[iw : ie + 1, js] - 2.0 * mu * dvdy(js), xc
    )
    fy_n = np.trapezoid(
        rho * v[iw : ie + 1, jn] ** 2 + p[iw : ie + 1, jn] - 2.0 * mu * dvdy(jn), xc
    )
    Fy = (fy_w - fy_e) + (fy_s - fy_n)

    return float(Fx), float(Fy)


def _ibm_force_mean(
    ctx: Any, U_inf: float, D: float, span: float, tail_fraction: float
) -> Optional[tuple[float, float]]:
    """Time-mean ``(Cd, Cl)`` from the recorded IBM reaction force, or ``None``.

    Averages over the last ``tail_fraction`` of the history (drop the startup
    transient). The force is in kinematic units (per ρ), so ρ cancels in the
    coefficient and the frontal area is ``D * span``. Reads the reaction-force
    time series from the mesh-owned IBM data via the method's ``force_history``
    accessor (``ctx.models["ibm"]`` is the method the projection applies to
    ``U``); no body / no IBM method configured on ``U`` returns ``None`` (fall
    back to the momentum-deficit survey).
    """
    method = ctx.models.get("ibm")
    if method is None:
        return None

    data = ctx.fields["U"].mesh.ibm_data(method)
    hist = method.force_history(data)
    if not hist:
        return None
    arr = np.asarray(hist, dtype=float)  # (n, 4): t, Fx, Fy, Fz
    n = arr.shape[0]
    start = max(0, int(round((1.0 - tail_fraction) * n)))
    fx = float(np.mean(arr[start:, 1]))
    fy = float(np.mean(arr[start:, 2]))
    q = 0.5 * U_inf * U_inf * D * span
    return fx / q, fy / q


def force_coefficients(
    ctx: Any,
    U_inf: float,
    D: float,
    *,
    rho: float = 1.0,
    nu: float = 0.0,
    cv: Optional[tuple[float, float, float, float]] = None,
    tail_fraction: float = 0.5,
) -> tuple[float, float]:
    """Drag/lift coefficients ``(Cd, Cl)`` of the immersed body.

    Primary estimator: the recorded direct-forcing reaction force, time-averaged
    over the last ``tail_fraction`` of the run. Fallback (no recorded force):
    a control-volume momentum-deficit survey over ``cv`` (physical
    ``(xw, xe, ys, yn)`` extents; defaults to a wake box around the body).
    """
    snap = gather_field(ctx)
    span = float(snap.z[-1] - snap.z[0] + snap.dz)

    ibm = _ibm_force_mean(ctx, U_inf, D, span, tail_fraction)
    if ibm is not None:
        return ibm

    # Fallback: CV momentum survey on the z-averaged plane.
    u2 = snap.u.mean(axis=2)
    v2 = snap.v.mean(axis=2)
    p2 = snap.p.mean(axis=2)
    if cv is None:
        # a box spanning ~[-2D, +6D] x [-3D, +3D] around the body centre
        cx = (
            float(snap.x[snap.u.shape[0] // 2])
            if snap.solid is None
            else _body_center(snap)[0]
        )
        cy = (
            float(snap.y[snap.u.shape[1] // 2])
            if snap.solid is None
            else _body_center(snap)[1]
        )
        cv = (cx - 2.0 * D, cx + 6.0 * D, cy - 3.0 * D, cy + 3.0 * D)
    iw = int(np.searchsorted(snap.x, cv[0]))
    ie = int(np.searchsorted(snap.x, cv[1]))
    js = int(np.searchsorted(snap.y, cv[2]))
    jn = int(np.searchsorted(snap.y, cv[3]))
    iw = max(1, iw)
    ie = min(snap.u.shape[0] - 2, ie)
    js = max(1, js)
    jn = min(snap.u.shape[1] - 2, jn)
    Fx, Fy = momentum_deficit_forces(
        u2,
        v2,
        p2,
        snap.x,
        snap.y,
        snap.dx,
        snap.dy,
        (iw, ie, js, jn),
        rho=rho,
        nu=nu,
    )
    q = 0.5 * rho * U_inf * U_inf * D * span
    return Fx / q, Fy / q


# ---------------------------------------------------------------------------
# Wake observables
# ---------------------------------------------------------------------------


def _body_center(snap: FieldSnapshot) -> tuple[float, float]:
    """In-plane centroid (x, y) of the solid mask (z-collapsed)."""
    assert snap.solid is not None
    mask2 = snap.solid.any(axis=2)
    ii, jj = np.where(mask2)
    return float(snap.x[ii].mean()), float(snap.y[jj].mean())


def strouhal(cl_series: np.ndarray, dt: float, D: float, U_inf: float) -> float:
    """Strouhal number ``St = f D / U∞`` from the FFT peak of ``cl_series``.

    The mean is removed before the transform; the dominant non-zero frequency
    (the shedding tone) is picked from the single-sided spectrum.
    """
    x = np.asarray(cl_series, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 4:
        raise ValueError("strouhal needs >= 4 samples")
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0  # ignore the DC bin
    f_peak = float(freqs[int(np.argmax(spec))])
    return f_peak * D / U_inf


def recirculation_length(
    snap: FieldSnapshot,
    D: float,
    *,
    center: Optional[tuple[float, float]] = None,
) -> float:
    """Wake recirculation length ``Lr/D`` behind the body.

    Along the wake centreline (``y = center_y``, z-averaged) the streamwise
    velocity is negative in the recirculation bubble just behind the body and
    recovers to positive downstream; ``Lr`` is the distance from the rear of the
    body to that first ``u = 0`` up-crossing. Returns ``0`` when the near-wake
    velocity never goes negative (no separation bubble).
    """
    if center is None:
        center = _body_center(snap)
    cx, cy = center
    j = int(np.argmin(np.abs(snap.y - cy)))
    u_line = snap.u[:, j, :].mean(axis=1)  # z-averaged centreline u(x)

    # rear of the body along the centreline
    if snap.solid is not None:
        solid_line = snap.solid[:, j, :].any(axis=1)
        rear_idx = np.where(solid_line)[0]
        i_rear = (
            int(rear_idx[-1]) if rear_idx.size else int(np.argmin(np.abs(snap.x - cx)))
        )
    else:
        i_rear = int(np.argmin(np.abs(snap.x - cx)))
    x_rear = snap.x[i_rear]

    # first up-crossing (neg → pos) downstream of the rear
    for i in range(i_rear + 1, snap.u.shape[0] - 1):
        if u_line[i] < 0.0 <= u_line[i + 1]:
            # linear interpolation of the zero crossing
            frac = -u_line[i] / (u_line[i + 1] - u_line[i])
            x0 = snap.x[i] + frac * (snap.x[i + 1] - snap.x[i])
            return float((x0 - x_rear) / D)
    return 0.0


def separation_angle(
    snap: FieldSnapshot,
    *,
    center: Optional[tuple[float, float]] = None,
    radius: Optional[float] = None,
) -> float:
    """Surface separation angle ``θs`` (degrees from the rear stagnation point).

    Samples the fluid ring one cell outside the body and, on the upper half,
    finds where the tangential (azimuthal) velocity changes sign — the boundary
    layer detaches there. The angle is measured from the rear stagnation point
    (``+x`` direction), matching the laminar-cylinder literature convention
    (Re=20 ≈ 43–45°, Re=40 ≈ 53°). Returns ``0`` if no reversal is found.
    """
    assert snap.solid is not None, "separation_angle needs an immersed body"
    if center is None:
        center = _body_center(snap)
    cx, cy = center
    if radius is None:
        radius = _mask_radius(snap, center)

    u2 = snap.u.mean(axis=2)
    v2 = snap.v.mean(axis=2)
    r_sample = radius + max(snap.dx, snap.dy)  # one cell outside the body

    # sweep the upper half from the rear (+x, φ=0) to the front (−x, φ=180)
    phis = np.deg2rad(np.arange(2, 179, 2.0))
    prev_ut: Optional[float] = None
    for phi in phis:
        px = cx + r_sample * np.cos(phi)
        py = cy + r_sample * np.sin(phi)
        i = int(np.argmin(np.abs(snap.x - px)))
        j = int(np.argmin(np.abs(snap.y - py)))
        # tangential unit vector (counter-clockwise): (−sinφ, cosφ)
        ut = -np.sin(phi) * u2[i, j] + np.cos(phi) * v2[i, j]
        if prev_ut is not None and prev_ut > 0.0 >= ut:
            return float(np.rad2deg(phi))
        prev_ut = ut
    return 0.0


def _mask_radius(snap: FieldSnapshot, center: tuple[float, float]) -> float:
    """Effective radius of the solid mask from its in-plane area."""
    assert snap.solid is not None
    mask2 = snap.solid.any(axis=2)
    area = mask2.sum() * snap.dx * snap.dy
    return float(np.sqrt(area / np.pi))
