// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "SchillerNaumann.hpp"
#include "SphereDrag.hpp"
#include "StokesDrag.hpp"

namespace NeoFOAM::lagrangian
{

/* @brief Which drag correlation to use. */
enum class DragType
{
    sphere,          ///< SphereDrag, Putnam, OpenFOAM's "sphereDrag"
    SchillerNaumann, ///< SchillerNaumann (NeoFOAM only)
    Stokes           ///< StokesDrag, Re << 1 (NeoFOAM only)
};

/* @brief The drag correlation of the cloud.
 *
 * A drag model answers ONE question:
 *     what is the drag coefficient of this particle at this Reynolds number?
 *
 * It returns Cd * Re instead of Cd (as OpenFOAM's SphereDragForce::CdRe):
 * Cd = 24/Re is infinite for a particle at rest (Re = 0), but Cd * Re = 24
 * stays finite, so a particle can start from rest without giving NaN.
 *
 * GPU: a virtual base class cannot be called inside a GPU kernel, so the
 * model is chosen with an enum and a switch. DragModel is a small value
 * type, copied into the kernel.
 *
 * To add a correlation: write a class with a KOKKOS_INLINE_FUNCTION cdRe(re),
 * add a DragType entry and a case below.
 */
class DragModel
{
public:

    DragType type = DragType::sphere;

    /* @brief Cd * Re for particle Reynolds number Re >= 0. */
    KOKKOS_INLINE_FUNCTION
    NeoN::scalar cdRe(NeoN::scalar re) const
    {
        switch (type)
        {
        case DragType::SchillerNaumann:
            return SchillerNaumann {}.cdRe(re);
        case DragType::Stokes:
            return StokesDrag {}.cdRe(re);
        default:
            return SphereDrag {}.cdRe(re);
        }
    }
};

} // namespace NeoFOAM::lagrangian
