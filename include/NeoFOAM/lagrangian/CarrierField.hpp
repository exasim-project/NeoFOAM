// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once
#include "NeoN/NeoN.hpp"
#include "TrackingMesh.hpp"
#include "volFields.H"
#include <string>


namespace NeoFOAM::lagrangian
{
// wrapper method for interpolation, either cell or cellPoint (OpenFOAM's volPointInterpolation)
enum class Interpolation
{
    cell,
    cellPoint
};

/* @brief Map a config string to an Interpolation enum value. */
inline Interpolation interpolationFromName(const std::string& name)
{
    if (name == "cell") return Interpolation::cell;
    if (name == "cellPoint") return Interpolation::cellPoint;

    NF_ERROR_EXIT("Unknown carrier interpolation '" + name + "' (valid: cell cellPoint)");

    return Interpolation::cell; // unreachable, silences compiler warning
}

// Struct to hold carrier field, cell and point values, copied to the executor
struct CarrierField
{
    Interpolation mode;
    NeoN::Vector<NeoN::Vec3> U; ///< cell values
    NeoN::Vector<NeoN::Vec3>
        Upoint; ///< point values (cellPoint only), OpenFOAM volPointInterpolation
};

// Struct to hold read-only views of CarrierField, captured by value inside kernels
struct CarrierFieldView
{
    bool cellPoint;
    NeoN::View<const NeoN::Vec3> U;
    NeoN::View<const NeoN::Vec3> Upoint;
};

// Build a read-only view of a CarrierField, captured by value inside kernels.
inline CarrierFieldView view(const CarrierField& f)
{
    return {f.mode == Interpolation::cellPoint, f.U.view(), f.Upoint.view()};
}

// Declare a function to build a CarrierField from an OpenFOAM volVectorField and an Interpolation
// mode, on a specified executor.
CarrierField
buildCarrierField(const NeoN::Executor& exec, const Foam::volVectorField& U, Interpolation mode);


// Compute barycentric coordinates of a point x in a tetrahedron (a, b, c, d).
// GPU Callable. Returns the determinant of the tetrahedron (zero if degenerate). If degenerate, w =
// (1, 0, 0, 0). this function used by function interpolate() to find the tetrahedron that contains
// x, and then interpolate the value at x using the barycentric coordinates.
KOKKOS_INLINE_FUNCTION
NeoN::scalar pointToBarycentric(
    const NeoN::Vec3& x,
    const NeoN::Vec3& a,
    const NeoN::Vec3& b,
    const NeoN::Vec3& c,
    const NeoN::Vec3& d,
    NeoN::scalar w[4]
)
{
    const NeoN::Vec3 e1 = b - a, e2 = c - a, e3 = d - a, r = x - a;
    const NeoN::scalar det = (e1 ^ e2) & e3;
    if (det == NeoN::scalar(0))
    {
        w[0] = NeoN::scalar(1);
        w[1] = w[2] = w[3] = NeoN::scalar(0);
        return det;
    }
    w[1] = ((r ^ e2) & e3) / det;
    w[2] = ((e1 ^ r) & e3) / det;
    w[3] = ((e1 ^ e2) & r) / det;
    w[0] = NeoN::scalar(1) - w[1] - w[2] - w[3];
    return det;
}


// Interpolate METHODS, either cell or cellPoint (OpenFOAM's volPointInterpolation), on a
// TrackingMesh and a CarrierField, at a point x in cell c.

// Cell use schema use center value as everywhere in the cell,
// cellPoint use OpenFOAM's volPointInterpolation to interpolate the value at point x using the cell
// center and the three face points of the face that contains x. If x is outside the cell, it uses
// the nearest tetrahedron (the one with the largest smallest weight) to interpolate.

KOKKOS_INLINE_FUNCTION

NeoN::Vec3 interpolate(
    const CarrierFieldView& f,
    const TrackingMeshView& m,
    const NeoN::label c,
    const NeoN::Vec3& x
)
{
    if (!f.cellPoint)
    {
        return f.U[c];
    }

    // OpenFOAM cellPointWeight::findTetrahedron: the tet (cell centre, face base point,
    // two consecutive face points) containing x; tolerance SMALL. If none contains x,
    // OpenFOAM uses the nearest tet; here: the tet with the largest smallest weight.
    constexpr NeoN::scalar tol = NeoN::scalar(1e-15);
    const NeoN::Vec3 cc = m.cellCentres[c];
    NeoN::scalar best[4] = {NeoN::scalar(1), NeoN::scalar(0), NeoN::scalar(0), NeoN::scalar(0)};
    NeoN::label bestPts[3] = {-1, -1, -1};
    NeoN::scalar bestMin = NeoN::scalar(-1e300);
    bool found = false;
    for (NeoN::label k = m.cellFaceOffsets[c]; k < m.cellFaceOffsets[c + 1] && !found; k++)
    {
        const NeoN::label face = m.cellFaces[k];
        const NeoN::label p0 = m.facePointOffsets[face];
        const NeoN::label n = m.facePointOffsets[face + 1] - p0;
        const NeoN::label base = m.faceBasePt[face];
        for (NeoN::label tetPt = 1; tetPt < n - 1; tetPt++)
        {
            // OpenFOAM tetIndices::faceTriIs (flipped for the neighbour cell)
            NeoN::label facePt = (tetPt + base) % n;
            NeoN::label otherPt = (facePt + 1) % n;
            if (m.faceOwner[face] != c)
            {
                const NeoN::label t = facePt;
                facePt = otherPt;
                otherPt = t;
            }
            const NeoN::label ia = m.facePoints[p0 + base];
            const NeoN::label ib = m.facePoints[p0 + facePt];
            const NeoN::label ic = m.facePoints[p0 + otherPt];
            NeoN::scalar w[4];
            const NeoN::scalar det =
                pointToBarycentric(x, cc, m.points[ia], m.points[ib], m.points[ic], w);
            if (Kokkos::fabs(det / m.cellVolumes[c]) <= tol)
            {
                continue;
            }
            const NeoN::scalar wMin =
                Kokkos::fmin(Kokkos::fmin(w[0], w[1]), Kokkos::fmin(w[2], w[3]));
            if (wMin > bestMin)
            {
                bestMin = wMin;
                for (int j = 0; j < 4; j++)
                {
                    best[j] = w[j];
                }
                bestPts[0] = ia;
                bestPts[1] = ib;
                bestPts[2] = ic;
            }
            if (wMin + tol > NeoN::scalar(0))
            {
                found = true;
                break;
            }
        }
    }
    if (bestPts[0] < 0)
    {
        return f.U[c];
    }
    return best[0] * f.U[c] + best[1] * f.Upoint[bestPts[0]] + best[2] * f.Upoint[bestPts[1]]
         + best[3] * f.Upoint[bestPts[2]];
}

} // namespace NeoFOAM::lagrangian
