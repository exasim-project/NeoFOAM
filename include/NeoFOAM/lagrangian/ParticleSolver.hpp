// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once
#include "CarrierField.hpp"
#include "CarrierProperties.hpp"
#include "IntegrationScheme.hpp"
#include "LocateMethod.hpp"
#include "ParticleCloud.hpp"
#include "TrackingMesh.hpp"
#include "drag/DragModel.hpp"
#include "force/ForceModel.hpp"
#include "NeoFOAM/datastructures/meshAdapter.hpp"
#include "volFields.H"
#include <optional>

namespace NeoFOAM::lagrangian
{
/* @brief Orchestrates the Lagrangian particle tracking chain.
 *
 * The solver knows the ORDER of operations (the 10 questions),
 * but delegates the PHYSICS to the injected models:
 *
 *     Q1  locate cell              — locateAndInterpolate(): findCell or walk (LocateMethod)
 *     Q2  interpolate u_f          — locateAndInterpolate(): U[cell] or CarrierField interpolate()
 *     Q3  slip = u_f - u_p         — evolve()
 *     Q4  Re = f(slip)             — evolve()
 *     Q5  cdRe = drag.cdRe(re)     — DragModel
 *     Q6  F = sum(forces)          — ForceModels
 *     Q7  a = F / m                — evolve()
 *     Q8  integrate                — EulerIntegrator / RK2Integrator
 *     Q9  track through mesh       — (deferred: the next locate finds the new cell)
 *     Q10 exit check               — locateAndInterpolate()
 *
 * Every model is chosen in the case: drag law, forces, integration scheme and the
 * locate method. evolve() and the walk locate run one kernel each (one particle per
 * thread) on the cloud's executor: Serial, CPU or GPU. The models are small value
 * types (no virtual functions), copied into the kernel.
 */
class ParticleSolver
{
public:

    /* @brief Construct from models and carrier properties.
     *
     * @param drag           drag coefficient model
     * @param forces         forces acting on the particles (drag, gravity, ...)
     * @param scheme         time integration scheme (Euler or RK2)
     * @param carrier        fluid properties (rho, mu)
     * @param locate         locate method (findCell or walk)
     * @param interpolation  carrier interpolation for walk (cell or cellPoint);
     *                       findCell always uses the cell value
     * @param exec           executor of the cloud
     * @param mesh           mesh adapter: OpenFOAM and NeoN mesh (kept by reference)
     * @param U              frozen carrier velocity (kept by reference)
     */

    // Constructor for ParticleSolver class
    ParticleSolver(
        const DragModel& drag,
        const ForceModels& forces,
        IntegrationScheme scheme,
        const CarrierProperties& carrier,
        LocateMethod locate,
        Interpolation interpolation,
        const NeoN::Executor& exec,
        const MeshAdapter& mesh,
        const Foam::volVectorField& U
    );


    /* @brief Q1, Q2, Q10: locate every particle and sample the carrier velocity there.
     *
     * The carrier velocity is an argument, not a member: the same solver can be used
     * with a frozen field (uncoupled program) or with the air of the current time step
     * (coupled program), without being rebuilt.
     *
     * @param cloud    particle cloud (x, cell and active are updated)
     * @param uf       [out] carrier velocity at each particle position [m/s]
     * @param uCells   carrier velocity in every mesh cell [m/s] (used by findCell)
     * @return number of particles deactivated in this call
     */
    NeoN::label locateAndInterpolate(
        ParticleCloud& cloud,
        NeoN::Vector<NeoN::Vec3>& uf,
        const NeoN::Vector<NeoN::Vec3>& uCells
    ) const;

    void evolve(ParticleCloud& cloud, const NeoN::Vector<NeoN::Vec3>& uf, NeoN::scalar dt) const;

    /* @brief findCell path: OpenFOAM mesh.findCell on the host, cell value of uCells. */
    NeoN::label locateFindCell(
        ParticleCloud& cloud,
        NeoN::Vector<NeoN::Vec3>& uf,
        const NeoN::Vector<NeoN::Vec3>& uCells
    ) const;

    NeoN::label locateWalk(ParticleCloud& cloud, NeoN::Vector<NeoN::Vec3>& uf) const;

private:

    DragModel drag_;
    ForceModels forces_;
    IntegrationScheme scheme_;
    CarrierProperties carrier_;
    LocateMethod locate_;
    const MeshAdapter& mesh_;
    const Foam::volVectorField& U_;
    std::optional<NeoN::TrackingMesh> trackingMesh_; ///< built for locate walk only
    std::optional<CarrierField> carrierField_; ///< built for locate walk only
};

} // namespace NeoFOAM::lagrangian
