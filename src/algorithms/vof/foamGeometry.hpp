// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"

namespace NeoFOAM
{

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

/* @brief OpenFOAM's interpolation and snGrad geometry of the internal faces, on the device.
 *
 * @detail NeoN's mesh cannot provide these bit for bit: its GeometryScheme frees the cell and
 * face centres after first use and computes its own weights and delta coefficients, which round
 * differently. Copied from the OpenFOAM mesh once and cached in the NeoN mesh's stencil database
 * (see foamGeometry). The boundary delta coefficients need no copy: the NeoN boundary mesh already
 * holds OpenFOAM's fvPatch::deltaCoeffs.
 */
struct FoamGeometry
{
    /// surfaceInterpolation::weights
    NeoN::Vector<NeoN::scalar> weights;
    /// C[neighbour] - C[owner]
    NeoN::Vector<NeoN::Vec3> delta;
    /// surfaceInterpolation::nonOrthDeltaCoeffs
    NeoN::Vector<NeoN::scalar> nonOrthDeltaCoeffs;
    /// surfaceInterpolation::nonOrthCorrectionVectors
    NeoN::Vector<NeoN::Vec3> nonOrthCorrectionVectors;
};

/* @brief The FoamGeometry of the mesh in @p rt, copied to the device on first use. */
const FoamGeometry& foamGeometry(RunTime& rt);

/* @brief Faces of every cell in fvc::surfaceSum's order.
 *
 * @detail Internal faces by index (as owner or neighbour), then the boundary faces, stored as
 * nInternalFaces + boundary index. A per-cell gather in this order adds up exactly as OpenFOAM
 * does, which atomics on threads cannot. Built once per mesh and cached in its stencil database.
 */
struct CellFaces
{
    NeoN::Vector<NeoN::localIdx> offsets;
    NeoN::Vector<NeoN::localIdx> faces;
};

/* @brief The CellFaces of @p mesh, built on first use. */
const CellFaces& cellFaces(const NeoN::UnstructuredMesh& mesh);

/* @brief fvc::grad(psi) under "Gauss linear" (gaussGrad::gradf) with OpenFOAM's @p weights.
 *
 * Cell values only, summed in OpenFOAM's order (internal faces, then boundary faces, then / V)
 * on Serial; the parallel backends add the face contributions atomically.
 */
NeoN::Vector<NeoN::Vec3> gaussLinearGrad(
    const nnfvcc::VolumeField<NeoN::scalar>& psi,
    NeoN::View<const NeoN::scalar> weights
);

/* @brief fvc::surfaceIntegrate into the zeroed @p res.
 *
 * @detail owner += flux, neighbour -= flux, boundary owner += flux, then / V. NeoN's
 * surfaceIntegrate multiplies by 1/V, which rounds differently from OpenFOAM's division.
 */
void integrateFluxInto(
    const NeoN::UnstructuredMesh& mesh,
    NeoN::View<const NeoN::scalar> flux,
    NeoN::View<const NeoN::scalar> bFlux,
    NeoN::View<NeoN::scalar> res
);

} // namespace NeoFOAM
