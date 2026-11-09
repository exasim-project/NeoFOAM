// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once
#include "NeoN/NeoN.hpp"
#include <vector>

namespace NeoFOAM::lagrangian
{

// Class for ParticleCloud (SOA) layout, which represents a collection of particles in a Lagrangian
// framework
class ParticleCloud
{
public:

    // Constructor for ParticleCloud class, initializes the particle cloud with given properties
    ParticleCloud(
        const NeoN::Executor& exec,
        const std::vector<NeoN::Vec3>& x0,
        const std::vector<NeoN::Vec3>& u0,
        const std::vector<NeoN::scalar>& d0,
        const std::vector<NeoN::scalar>& rho0,
        const std::vector<NeoN::label>& cell0
    )
        : x(exec, x0)
        , U(exec, u0)
        , d(exec, d0)
        , rho(exec, rho0)
        , cell(exec, cell0)
        , active(exec, static_cast<NeoN::localIdx>(x0.size()), NeoN::label(1))
    {}

    // Returns the number of particles in the cloud by returning the size of the position vector x.
    NeoN::localIdx size() const { return x.size(); }

    // Returns the executor associated with the particle cloud, which is the same as the executor of
    // the position vector x.
    const NeoN::Executor& exec() const { return x.exec(); }

    NeoN::Vector<NeoN::Vec3> x;       ///< position [m]
    NeoN::Vector<NeoN::Vec3> U;       ///< velocity [m/s]
    NeoN::Vector<NeoN::scalar> d;     ///< diameter [m]
    NeoN::Vector<NeoN::scalar> rho;   ///< density [kg/m^3]
    NeoN::Vector<NeoN::label> cell;   ///< cell of the carrier mesh the particle is in
    NeoN::Vector<NeoN::label> active; ///< 1 = moving, 0 = left the domain (escaped)
};

} // namespace NeoFOAM::lagrangian
