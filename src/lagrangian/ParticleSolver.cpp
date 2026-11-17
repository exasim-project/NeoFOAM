// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include "NeoFOAM/lagrangian/ParticleSolver.hpp"

#include "NeoFOAM/lagrangian/integrator/EulerIntegrator.hpp"
#include "NeoFOAM/lagrangian/integrator/RK2Integrator.hpp"

namespace NeoFOAM::lagrangian
{

/// Constructor
ParticleSolver::ParticleSolver(
    const DragModel& drag,
    const ForceModels& forces,
    IntegrationScheme scheme,
    const CarrierProperties& carrier,
    LocateMethod locate,
    Interpolation interpolation,
    const NeoN::Executor& exec,
    const MeshAdapter& mesh,
    const Foam::volVectorField& U
)
    : drag_(drag)
    , forces_(forces)
    , scheme_(scheme)
    , carrier_(carrier)
    , locate_(locate)
    , mesh_(mesh)
    , U_(U)
{
    if (scheme_ == IntegrationScheme::analytical)
    {
        NF_ERROR_EXIT("ParticleSolver: integration scheme analytical is not implemented yet (use "
                      "Euler or RK2)");
    }

    // locate walk: mesh and carrier velocity on the executor, built once
    if (locate_ == LocateMethod::walk)
    {
        trackingMesh_.emplace(buildTrackingMesh(exec, mesh_));
        carrierField_.emplace(buildCarrierField(exec, U_, interpolation));
    }
}


// Wrapper Function choose whoich locate method to use, either the OpenFOAM findCell or the new
// locateWalk method.
NeoN::label ParticleSolver::locateAndInterpolate(
    ParticleCloud& cloud,
    NeoN::Vector<NeoN::Vec3>& uf,
    const NeoN::Vector<NeoN::Vec3>& uCells
) const
{
    switch (locate_)
    {
    case LocateMethod::walk:
        return locateWalk(cloud, uf);
    default:
        return locateFindCell(cloud, uf, uCells);
    }
}


// Openfoam FindCell METHOD, nothing new
NeoN::label ParticleSolver::locateFindCell(
    ParticleCloud& cloud,
    NeoN::Vector<NeoN::Vec3>& uf,
    const NeoN::Vector<NeoN::Vec3>& uCells
) const
{
    // findCell is OpenFOAM (host only): copy the particle data to the host, update
    // it there and copy it back to the executor (no-op copy on Serial/CPU)
    const auto xH = cloud.x.copyToHost();
    auto cellH = cloud.cell.copyToHost();
    auto activeH = cloud.active.copyToHost();
    auto ufH = uf.copyToHost();
    const auto uCellsH = uCells.copyToHost();

    const auto xV = xH.view();
    auto cellV = cellH.view();
    auto activeV = activeH.view();
    auto ufV = ufH.view();
    const auto uCellsV = uCellsH.view();

    NeoN::label left = 0;
    for (NeoN::localIdx i = 0; i < cloud.size(); i++)
    {
        if (activeV[i] == 0)
        {
            continue;
        }

        // Q1: the cell containing x
        const Foam::label celli = mesh_.findCell(Foam::point(xV[i][0], xV[i][1], xV[i][2]));

        // Q10: outside the mesh
        if (celli < 0)
        {
            activeV[i] = 0;
            left++;
            continue;
        }

        // Q2: cell value of the carrier velocity (as handed in this time step)
        cellV[i] = static_cast<NeoN::label>(celli);
        ufV[i] = uCellsV[celli];
    }

    // back to the executor of the cloud
    cloud.cell = cellH.copyToExecutor(cloud.exec());
    cloud.active = activeH.copyToExecutor(cloud.exec());
    uf = ufH.copyToExecutor(cloud.exec());
    return left;
}


// New METHOD for locate walk , will use the New Data structure trackingmesh and carrierfield to
// locate the particles and interpolate the carrier velocity at the particle position.
NeoN::label ParticleSolver::locateWalk(ParticleCloud& cloud, NeoN::Vector<NeoN::Vec3>& uf) const
{
    auto xV = cloud.x.view();
    auto cellV = cloud.cell.view();
    auto activeV = cloud.active.view();
    auto ufV = uf.view();
    const NeoN::TrackingMeshView m = trackingMesh_->view();
    const CarrierFieldView f = view(*carrierField_);

    // ─── One thread per particle ───
    NeoN::label left = 0;
    NeoN::parallelReduce(
        cloud.exec(),
        cloud.x.range(),
        NEON_LAMBDA(const NeoN::localIdx i, NeoN::label& nLeft) {
            if (activeV[i] == 0)
            {
                return;
            }

            // Q1: walk from the old cell to the cell containing x (x shifted on cyclics)
            NeoN::Vec3 x = xV[i];
            const NeoN::label c = NeoN::locateCell(x, cellV[i], m);

            // Q10: left the domain through a non-periodic boundary
            if (c < 0)
            {
                activeV[i] = 0;
                nLeft += 1;
                return;
            }

            // Q2: carrier velocity at the particle position
            xV[i] = x;
            cellV[i] = c;
            ufV[i] = interpolate(f, m, c, x);
        },
        left
    );
    return left;
}


// Advance Particle METHOD Main GPU Kernel ( where the parallel take place for each particle)
// This Method will return the new particle position and velocity after the time step dt.
void ParticleSolver::evolve(
    ParticleCloud& cloud,
    const NeoN::Vector<NeoN::Vec3>& uf,
    NeoN::scalar dt
) const
{
    auto xV = cloud.x.view();
    auto uV = cloud.U.view();
    const auto activeV = cloud.active.view();
    const auto dV = cloud.d.view();
    const auto rhoV = cloud.rho.view();
    const auto ufV = uf.view();

    //  Local variable setup for the lambda function to avoid capturing member variables
    const DragModel drag = drag_;
    const ForceModels forces = forces_;
    const IntegrationScheme scheme = scheme_;
    const NeoN::scalar rhoF = carrier_.rho;
    const NeoN::scalar muF = carrier_.mu;

    // ─── One thread per particle: Q3 - Q8 ───
    NeoN::parallelFor(
        cloud.exec(),
        cloud.x.range(),
        NEON_LAMBDA(const NeoN::localIdx i) {
            if (activeV[i] == 0)
            {
                return;
            }

            const NeoN::scalar d = dV[i];
            const NeoN::scalar rhoP = rhoV[i];

            // ─── Acceleration callback (used by the integrator) ───
            auto accel = [&](const NeoN::Vec3& up, const NeoN::Vec3&) -> Acceleration
            {
                // Q3: slip velocity
                const NeoN::Vec3 slip = ufV[i] - up;

                // Q4: particle Reynolds number
                const NeoN::scalar re = rhoF * d * NeoN::mag(slip) / muF;

                // Q5: drag coefficient (as Cd * Re, finite at Re = 0)
                const NeoN::scalar cdRe = drag.cdRe(re);

                // Q6: sum of forces, and implicit coefficient for the implicit Euler integrator
                ForceContext ctx;
                ctx.uf = ufV[i];
                ctx.cdRe = cdRe;
                ctx.rhoF = rhoF;
                ctx.muF = muF;

                // Compute the foce balance
                const NeoN::Vec3 force = forces.compute(up, d, rhoP, ctx);
                // Compute the implicit coefficient for the implicit Euler integrator
                const NeoN::scalar sp = forces.sp(d, rhoP, ctx);

                // Q7: acceleration
                const NeoN::scalar m = rhoP * M_PI / 6.0 * d * d * d;
                return {force / m, sp / m};
            };

            // Q8: integrate (the integrator calls accel as many times as needed)
            NeoN::Vec3 u = uV[i];
            NeoN::Vec3 x = xV[i];
            if (scheme == IntegrationScheme::RK2)
            {
                RK2Integrator {}.step(u, x, accel, dt);
            }
            else
            {
                EulerIntegrator {}.step(u, x, accel, dt);
            }
            uV[i] = u;
            xV[i] = x;
        },
        "lagrangian::ParticleSolver::evolve"
    );
}

} // namespace NeoFOAM::lagrangian
