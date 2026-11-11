// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/lagrangian/ParticleOutput.hpp"

#include <iomanip>

namespace NeoFOAM::lagrangian
{

void writeParticlesHeader(std::ostream& out)
{
    out << std::setprecision(17) << "t,id,x,y,z,ux,uy,uz,d\n";
}

void writeParticles(std::ostream& out, NeoN::scalar t, const ParticleCloud& cloud)
{
    // copy to the host for writing (no-op copy on Serial/CPU)
    const auto xH = cloud.x.copyToHost();
    const auto uH = cloud.U.copyToHost();
    const auto dH = cloud.d.copyToHost();
    const auto xV = xH.view();
    const auto uV = uH.view();
    const auto dV = dH.view();
    for (NeoN::localIdx i = 0; i < cloud.size(); i++)
    {
        out << t << ',' << i << ',' << xV[i][0] << ',' << xV[i][1] << ',' << xV[i][2] << ','
            << uV[i][0] << ',' << uV[i][1] << ',' << uV[i][2] << ',' << dV[i] << '\n';
    }
    out.flush();
}

} // namespace NeoFOAM::lagrangian
