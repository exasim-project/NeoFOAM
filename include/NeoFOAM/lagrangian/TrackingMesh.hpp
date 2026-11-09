// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once
#include "NeoN/NeoN.hpp"
#include "fvMesh.H"

namespace NeoFOAM::lagrangian
{

/* @brief What happens to a particle leaving a cell through a face. */
enum FaceType : NeoN::label
{
    internalFace = 0, ///< go to the neighbour cell
    cyclicFace = 1,   ///< periodic: shift the position, continue in the partner cell
    escapeFace = 2    ///< any other boundary: the particle leaves the domain
};

/* @brief Mesh data for locating particles (Q1, Q10) and for cellPoint interpolation (Q2),
 * on the executor (CPU or GPU).
 *
 * Built once from the OpenFOAM mesh (buildTrackingMesh); the arrays mirror OpenFOAM's
 * cells(), faces(), faceOwner(), faceCentres(), faceAreas(), cellCentres(), points(),
 * tetBasePtIs(). With it, locate() and interpolate() run inside a GPU kernel, so the
 * particle data does not have to be copied to the host every time step (as for
 * OpenFOAM's mesh.findCell).
 */

// struct TrackingMesh
struct TrackingMesh
{
    NeoN::Vector<NeoN::label>
        cellFaceOffsets; ///< faces of cell c: cellFaces[offsets[c] .. offsets[c+1])
    NeoN::Vector<NeoN::label> cellFaces;
    NeoN::Vector<NeoN::label>
        facePointOffsets; ///< points of face f: facePoints[offsets[f] .. offsets[f+1])
    NeoN::Vector<NeoN::label> facePoints;
    NeoN::Vector<NeoN::label> faceBasePt;    ///< OpenFOAM tetBasePtIs (local index in the face)
    NeoN::Vector<NeoN::label> faceOwner;     ///< owner cell of every face
    NeoN::Vector<NeoN::label> faceNeighbour; ///< neighbour cell, -1 for boundary faces
    NeoN::Vector<NeoN::Vec3> faceCentres;
    NeoN::Vector<NeoN::Vec3> faceAreas;          ///< area vectors (OpenFOAM Sf), owner -> neighbour
    NeoN::Vector<NeoN::label> faceType;          ///< FaceType
    NeoN::Vector<NeoN::label> cyclicPartnerCell; ///< cyclic faces: cell behind the paired face
    NeoN::Vector<NeoN::Vec3> cyclicShift;        ///< cyclic faces: translation to the paired face
    NeoN::Vector<NeoN::Vec3> cellCentres;
    NeoN::Vector<NeoN::scalar> cellVolumes;
    NeoN::Vector<NeoN::Vec3> points;
};

/* @brief Read-only views of TrackingMesh, captured by value inside kernels. */
struct TrackingMeshView
{
    NeoN::View<const NeoN::label> cellFaceOffsets;
    NeoN::View<const NeoN::label> cellFaces;
    NeoN::View<const NeoN::label> facePointOffsets;
    NeoN::View<const NeoN::label> facePoints;
    NeoN::View<const NeoN::label> faceBasePt;
    NeoN::View<const NeoN::label> faceOwner;
    NeoN::View<const NeoN::label> faceNeighbour;
    NeoN::View<const NeoN::Vec3> faceCentres;
    NeoN::View<const NeoN::Vec3> faceAreas;
    NeoN::View<const NeoN::label> faceType;
    NeoN::View<const NeoN::label> cyclicPartnerCell;
    NeoN::View<const NeoN::Vec3> cyclicShift;
    NeoN::View<const NeoN::Vec3> cellCentres;
    NeoN::View<const NeoN::scalar> cellVolumes;
    NeoN::View<const NeoN::Vec3> points;
};

// Create a read-only view of a TrackingMesh, for use inside GPU kernels.
inline TrackingMeshView view(const TrackingMesh& m)
{
    return {
        m.cellFaceOffsets.view(),
        m.cellFaces.view(),
        m.facePointOffsets.view(),
        m.facePoints.view(),
        m.faceBasePt.view(),
        m.faceOwner.view(),
        m.faceNeighbour.view(),
        m.faceCentres.view(),
        m.faceAreas.view(),
        m.faceType.view(),
        m.cyclicPartnerCell.view(),
        m.cyclicShift.view(),
        m.cellCentres.view(),
        m.cellVolumes.view(),
        m.points.view()
    };
}

/* @brief Build the tracking mesh from an OpenFOAM mesh (host) and copy it to the executor.
 *
 * Translational cyclic patches become cyclicFace (shift = paired face centre - face
 * centre, as OpenFOAM's separation vector), all other boundary faces escapeFace.
 * Rotational cyclics are not supported (fatal error).
 */


// Function declaration for buildTrackingMesh
TrackingMesh buildTrackingMesh(const NeoN::Executor& exec, const Foam::fvMesh& mesh);

/* @brief Is x inside cell c? Face-plane test, as OpenFOAM primitiveMesh::pointInCell
 * (FACE_PLANES): x is not in front of any (outward-oriented) face plane of c.
 */
KOKKOS_INLINE_FUNCTION
bool pointInCell(const NeoN::Vec3& x, const NeoN::label c, const TrackingMeshView& m)
{
    for (NeoN::label k = m.cellFaceOffsets[c]; k < m.cellFaceOffsets[c + 1]; k++)
    {
        const NeoN::label f = m.cellFaces[k];
        NeoN::Vec3 normal = m.faceAreas[f];
        if (m.faceOwner[f] != c)
        {
            normal = NeoN::scalar(-1) * normal;
        }
        if ((normal & (x - m.faceCentres[f])) > NeoN::scalar(0))
        {
            return false;
        }
    }
    return true;
}

/* @brief Q1 + Q10: locate the cell containing x, starting from cell c.
 *
 * The walk of OpenFOAM meshSearch::findCellWalk: if x is not in the current cell, test
 * the neighbour cells (pointInCell); otherwise move to the neighbour whose centre is
 * nearest to x, as long as that decreases the distance. For periodic meshes a
 * neighbour across a cyclic face is the partner cell, seen from the shifted position
 * x + shift (OpenFOAM's translational cyclic transform); x is shifted when the walk
 * goes through the cyclic face.
 *
 * @return the cell containing x (x possibly shifted), or -1 if the walk gets stuck,
 *         i.e. x has left the domain through a non-periodic boundary.
 */
KOKKOS_INLINE_FUNCTION
NeoN::label locate(NeoN::Vec3& x, NeoN::label c, const TrackingMeshView& m)
{
    if (pointInCell(x, c, m))
    {
        return c;
    }

    NeoN::Vec3 d = x - m.cellCentres[c];
    NeoN::scalar nearestDistSqr = d & d;
    constexpr int maxWalk = 1000;
    for (int step = 0; step < maxWalk; step++)
    {
        NeoN::label nearestCell = -1;
        NeoN::Vec3 nearestX = x;
        for (NeoN::label k = m.cellFaceOffsets[c]; k < m.cellFaceOffsets[c + 1]; k++)
        {
            const NeoN::label f = m.cellFaces[k];
            NeoN::label n;
            NeoN::Vec3 xn = x;
            if (m.faceType[f] == internalFace)
            {
                n = (m.faceOwner[f] == c) ? m.faceNeighbour[f] : m.faceOwner[f];
            }
            else if (m.faceType[f] == cyclicFace)
            {
                n = m.cyclicPartnerCell[f];
                xn = x + m.cyclicShift[f];
            }
            else
            {
                continue;
            }

            if (pointInCell(xn, n, m))
            {
                x = xn;
                return n;
            }
            const NeoN::Vec3 dn = xn - m.cellCentres[n];
            const NeoN::scalar distSqr = dn & dn;
            if (distSqr < nearestDistSqr)
            {
                nearestDistSqr = distSqr;
                nearestCell = n;
                nearestX = xn;
            }
        }
        if (nearestCell == -1)
        {
            return -1;
        }
        c = nearestCell;
        x = nearestX;
    }
    return -1;
}

} // namespace NeoFOAM::lagrangian
