// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once
#include "NeoN/NeoN.hpp"
#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "NeoN/mesh/unstructured/trackingMesh.hpp"


namespace NeoFOAM::lagrangian
{

/* @brief Build NeoN's TrackingMesh from an OpenFOAM mesh (host) and copy it to the executor.
 *
 * Reads the connectivity NeoN's UnstructuredMesh does not store from the OpenFOAM mesh:
 * cells(), faces(), tetBasePtIs(), points() and the cyclic patch pairing. Translational
 * cyclic patches become cyclic faces (shift = paired face centre - face centre, as
 * OpenFOAM's separation vector), all other boundary faces escape faces. Rotational
 * cyclics are not supported (fatal error).
 *
 * The geometry (cell and face centres, Sf, owner, neighbour, volumes) is taken from
 * mesh.nfMesh(), so exec must be the executor of that mesh.
 */
NeoN::TrackingMesh buildTrackingMesh(const NeoN::Executor& exec, const MeshAdapter& mesh);

} // namespace NeoFOAM::lagrangian
