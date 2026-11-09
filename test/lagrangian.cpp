// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

// Unit tests for the Lagrangian particle tracking library (include/NeoFOAM/lagrangian).

#define CATCH_CONFIG_RUNNER

#include "common.hpp"

#include "NeoFOAM/lagrangian/drag/DragModel.hpp"
#include "NeoFOAM/lagrangian/force/GravityForce.hpp"
#include "NeoFOAM/lagrangian/integrator/EulerIntegrator.hpp"
#include "NeoFOAM/lagrangian/integrator/RK2Integrator.hpp"
#include "NeoFOAM/lagrangian/ParticleCloud.hpp"
#include "NeoFOAM/lagrangian/ParticleSolver.hpp"

namespace lag = NeoFOAM::lagrangian;

extern Foam::Time* timePtr;

TEST_CASE("lagrangian drag models", "[lagrangian]")
{
    SECTION("Stokes: cdRe = 24 for any Re")
    {
        REQUIRE(lag::StokesDrag {}.cdRe(0.0) == Catch::Approx(24.0));
        REQUIRE(lag::StokesDrag {}.cdRe(500.0) == Catch::Approx(24.0));
    }

    SECTION("SchillerNaumann: 24 (1 + 0.15 Re^0.687), 0.44 Re above Re = 1000")
    {
        REQUIRE(lag::SchillerNaumann {}.cdRe(1.0) == Catch::Approx(27.6));
        REQUIRE(lag::SchillerNaumann {}.cdRe(2000.0) == Catch::Approx(0.44 * 2000.0));
    }

    SECTION("SphereDrag: 24 (1 + Re^(2/3) / 6), 0.424 Re above Re = 1000")
    {
        REQUIRE(lag::SphereDrag {}.cdRe(1.0) == Catch::Approx(28.0));
        REQUIRE(lag::SphereDrag {}.cdRe(2000.0) == Catch::Approx(0.424 * 2000.0));
    }

    SECTION("DragModel dispatches to the selected model")
    {
        lag::DragModel model;
        REQUIRE(model.cdRe(1.0) == Catch::Approx(28.0)); // default: sphere

        model.type = lag::DragType::SchillerNaumann;
        REQUIRE(model.cdRe(1.0) == Catch::Approx(27.6));

        model.type = lag::DragType::Stokes;
        REQUIRE(model.cdRe(1.0) == Catch::Approx(24.0));
    }
}

TEST_CASE("lagrangian gravity and buoyancy", "[lagrangian]")
{
    // 1 mm solid particle (rho_p = 2500) in air (rho_f = 1.2), g = (0, 0, -9.81)
    lag::GravityForce gravity;
    gravity.g = {0.0, 0.0, -9.81};
    const NeoN::scalar d = 1e-3;
    const NeoN::Vec3 up = {0.3, 0.0, 0.0}; // particle velocity, must not matter
    lag::ForceContext ctx {.uf = {0.0, 0.0, 0.0}, .cdRe = 24.0, .rhoF = 1.2, .muF = 1.8e-5};

    SECTION("F = (rho_p - rho_f) pi d^3 / 6 g")
    {
        // by hand: (2500 - 1.2) * pi * 1e-9 / 6 * (-9.81)
        const auto force = gravity.compute(up, d, 2500.0, ctx);
        REQUIRE(force[0] == 0.0);
        REQUIRE(force[1] == 0.0);
        REQUIRE(force[2] == Catch::Approx(-1.2835096166761938e-05));
    }

    SECTION("neutrally buoyant particle feels no force")
    {
        const auto force = gravity.compute(up, d, 1.2, ctx);
        REQUIRE(force[2] == 0.0);
    }

    SECTION("gravity has no implicit part: sp = 0") { REQUIRE(gravity.sp(d, 2500.0, ctx) == 0.0); }
}

TEST_CASE("lagrangian integrators", "[lagrangian]")
{
    // One step for a particle with u = 1 in still air, drag only: a = -u / tau, bp = 1 / tau.
    const NeoN::scalar tau = 0.01;
    const NeoN::scalar dt = 0.001;
    auto accel = [tau](const NeoN::Vec3& u, const NeoN::Vec3&) {
        return lag::Acceleration {u * (-1.0 / tau), 1.0 / tau};
    };

    NeoN::Vec3 u = {1.0, 0.0, 0.0};
    NeoN::Vec3 x = {0.0, 0.0, 0.0};

    SECTION("Euler (implicit drag): u = tau / (tau + dt), x = dt")
    {
        lag::EulerIntegrator {}.step(u, x, accel, dt);
        REQUIRE(u[0] == Catch::Approx(0.01 / 0.011).epsilon(1e-12));
        REQUIRE(x[0] == Catch::Approx(0.001).epsilon(1e-12));
    }

    SECTION("RK2: u = 1 - dt/tau + dt^2/(2 tau^2), x = dt - dt^2/(2 tau)")
    {
        lag::RK2Integrator {}.step(u, x, accel, dt);
        REQUIRE(u[0] == Catch::Approx(0.905).epsilon(1e-12));
        REQUIRE(x[0] == Catch::Approx(0.00095).epsilon(1e-12));
    }
}

TEST_CASE("lagrangian cell finding", "[lagrangian]")
{
    Foam::Time& runTime = *timePtr;
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto meshPtr = NeoFOAM::createMesh(exec, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    // carrier velocity = (cell index, 0, 0), so the sampled velocity tells which cell was found
    const NeoN::localIdx nCells = mesh.nCells();
    std::vector<NeoN::Vec3> uCellsHost(nCells);
    for (NeoN::localIdx c = 0; c < nCells; c++)
    {
        uCellsHost[c] = {NeoN::scalar(c), 0.0, 0.0};
    }
    NeoN::Vector<NeoN::Vec3> uCells(exec, uCellsHost);

    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh),
        mesh,
        Foam::dimensionedVector(Foam::dimVelocity, Foam::Zero)
    );
    lag::ParticleSolver solver(
        lag::DragModel {},
        lag::ForceModels {},
        lag::IntegrationScheme::Euler,
        lag::CarrierProperties {1.2, 1.8e-5},
        lag::LocateMethod::findCell,
        lag::Interpolation::cell,
        exec,
        mesh,
        U
    );

    // particle 0 at the centre of a known cell, particle 1 outside the domain
    const NeoN::label target = nCells / 2;
    const Foam::point centre = mesh.C()[target];
    lag::ParticleCloud cloud(
        exec,
        {{centre.x(), centre.y(), centre.z()}, {5.0, 5.0, 0.05}},
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {1e-4, 1e-4},
        {2500.0, 2500.0},
        {0, 0}
    );
    NeoN::Vector<NeoN::Vec3> uf(exec, 2, NeoN::Vec3 {0.0, 0.0, 0.0});

    const NeoN::label left = solver.locateAndInterpolate(cloud, uf, uCells);

    const auto cellH = cloud.cell.copyToHost();
    const auto activeH = cloud.active.copyToHost();
    const auto ufH = uf.copyToHost();

    SECTION("particle at a cell centre is found in that cell (" + execName + ")")
    {
        REQUIRE(cellH.view()[0] == target);
        REQUIRE(activeH.view()[0] == 1);
        REQUIRE(ufH.view()[0][0] == Catch::Approx(NeoN::scalar(target)));
    }

    SECTION("particle outside the domain is deactivated (" + execName + ")")
    {
        REQUIRE(left == 1);
        REQUIRE(activeH.view()[1] == 0);
    }
}
