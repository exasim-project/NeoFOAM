// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/lagrangian/CarrierField.hpp"

#include "volPointInterpolation.H"

#include <vector>

namespace NeoFOAM::lagrangian
{

namespace
{
NeoN::Vec3 toVec3(const Foam::vector& v) { return NeoN::Vec3(v.x(), v.y(), v.z()); }
}


// METHOD that convert or build openfom mesh data to array of NeoN::Vector and return the
// CarrierField data structure
CarrierField buildCarrierField(
    const NeoN::Executor& exec,
    const Foam::volVectorField& U,
    const Interpolation mode
)
{
    std::vector<NeoN::Vec3> u(U.size());
    for (NeoN::label c = 0; c < U.size(); c++)
    {
        u[c] = toVec3(U[c]);
    }
    if (mode == Interpolation::cell)
    {
        return CarrierField {
            mode,
            NeoN::Vector<NeoN::Vec3>(exec, u),
            NeoN::Vector<NeoN::Vec3>(exec, NeoN::localIdx(0))
        };
    }

    // point values exactly allocateAndInterpolate) and move (evolve)s OpenFOAM's
    // interpolationCellPoint (psip_)
    const Foam::tmp<Foam::pointVectorField> tUp =
        Foam::volPointInterpolation::New(U.mesh()).interpolate(U);
    const Foam::pointVectorField& uPoint = tUp();
    std::vector<NeoN::Vec3> up(uPoint.size());
    for (NeoN::label p = 0; p < uPoint.size(); p++)
    {
        up[p] = toVec3(uPoint[p]);
    }
    return CarrierField {
        mode,
        NeoN::Vector<NeoN::Vec3>(exec, u),
        NeoN::Vector<NeoN::Vec3>(exec, up)
    };
}

} // namespace NeoFOAM::lagrangian
