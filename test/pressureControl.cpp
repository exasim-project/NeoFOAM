// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#include <cstddef>
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

#include "NeoFOAM/auxiliary/pressureControl.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

extern Foam::Time* timePtr;    // A single time object
extern Foam::argList* argsPtr; // Some forks want argList access at createMesh.H
extern Foam::fvMesh* meshPtr;  // A single mesh object

// PressureControl ports OpenFOAM's Foam::pressureControl (pRefRequired = false, as in
// rhoPimpleFoam): the limits come from absolute pMin/pMax or from factors/rho applied to the
// value-fixing p patches, and limit() clips p only where it leaves [pMin, pMax].
TEST_CASE("pressureControl")
{
    Foam::Time& runTime = *timePtr;

    auto exec = NeoN::SerialExecutor {};
    auto meshPtrLocal = NeoFOAM::createMesh(exec, runTime);
    NeoN::UnstructuredMesh& mesh = meshPtrLocal->nfMesh();

    const auto nCells = mesh.nCells();
    REQUIRE(nCells > 2);

    // p with the given internal values; fixesValue selects fixedValue patches (boundary value
    // pBoundary) over calculated ones.
    auto makeP = [&](const std::vector<NeoN::scalar>& values, bool fixesValue, NeoN::scalar pBoundary)
    {
        std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs;
        if (fixesValue)
        {
            for (NeoN::localIdx patchi = 0; patchi < static_cast<NeoN::localIdx>(mesh.nBoundaries());
                 ++patchi)
            {
                NeoN::Dictionary dict;
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", NeoN::scalar(pBoundary));
                bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
            }
        }
        else
        {
            bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh);
        }
        fvcc::VolumeField<NeoN::scalar> p(exec, "p", mesh, bcs);
        p.internalVector() = NeoN::Vector<NeoN::scalar>(exec, values);
        NeoN::fill(p.boundaryData().value(), pBoundary);
        return p;
    };
    auto makeRho = [&](NeoN::scalar rhoBoundary)
    {
        fvcc::VolumeField<NeoN::scalar> rho(
            exec,
            "rho",
            mesh,
            fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh)
        );
        NeoN::fill(rho.internalVector(), rhoBoundary);
        NeoN::fill(rho.boundaryData().value(), rhoBoundary);
        return rho;
    };
    auto internal = [](fvcc::VolumeField<NeoN::scalar>& f) { return f.internalVector().copyToHost(); };

    const auto n = static_cast<std::size_t>(nCells);
    auto rho = makeRho(1.0);

    SECTION("without limits nothing is limited and nothing changes")
    {
        auto p = makeP(std::vector<NeoN::scalar>(n, 5.0), false, 5.0);
        NeoFOAM::PressureControl pc(p, rho, {});
        REQUIRE(pc.active() == false);
        REQUIRE(pc.limit(p) == false);
        REQUIRE(internal(p).view()[0] == 5.0);
    }

    SECTION("absolute pMin/pMax clip the cells that leave the range")
    {
        std::vector<NeoN::scalar> values(n, 100.0);
        values[0] = 10.0;
        values[1] = 500.0;
        auto p = makeP(values, false, 100.0);
        NeoFOAM::PressureLimitSettings s;
        s.pMin = 50.0;
        s.pMax = 200.0;
        NeoFOAM::PressureControl pc(p, rho, s);
        REQUIRE(pc.limitsMin());
        REQUIRE(pc.limitsMax());
        REQUIRE(pc.limit(p) == true);
        auto h = internal(p);
        REQUIRE(h.view()[0] == Catch::Approx(50.0));
        REQUIRE(h.view()[1] == Catch::Approx(200.0));
        REQUIRE(h.view()[2] == Catch::Approx(100.0));
    }

    SECTION("limit() reports the limits are configured even when nothing is clipped")
    {
        auto p = makeP(std::vector<NeoN::scalar>(n, 100.0), false, 100.0);
        NeoFOAM::PressureLimitSettings s;
        s.pMin = 50.0;
        s.pMax = 200.0;
        NeoFOAM::PressureControl pc(p, rho, s);
        REQUIRE(pc.limit(p) == true);
        REQUIRE(internal(p).view()[0] == Catch::Approx(100.0));
    }

    SECTION("only one absolute limit set still limits that side")
    {
        std::vector<NeoN::scalar> values(n, 100.0);
        values[0] = 10.0;
        values[1] = 500.0;
        auto p = makeP(values, false, 100.0);
        NeoFOAM::PressureLimitSettings s;
        s.pMin = 50.0;
        NeoFOAM::PressureControl pc(p, rho, s);
        REQUIRE(pc.limitsMin());
        REQUIRE_FALSE(pc.limitsMax());
        pc.limit(p);
        auto h = internal(p);
        REQUIRE(h.view()[0] == Catch::Approx(50.0));
        REQUIRE(h.view()[1] == Catch::Approx(500.0)); // no upper limit
    }

    SECTION("the factors scale the value-fixing patch values")
    {
        std::vector<NeoN::scalar> values(n, 100.0);
        values[0] = 1.0;
        values[1] = 1000.0;
        auto p = makeP(values, true, 100.0);
        NeoFOAM::PressureLimitSettings s;
        s.pMinFactor = 0.5;
        s.pMaxFactor = 2.0;
        NeoFOAM::PressureControl pc(p, rho, s);
        REQUIRE(pc.pMin() == Catch::Approx(50.0));
        REQUIRE(pc.pMax() == Catch::Approx(200.0));
        pc.limit(p);
        auto h = internal(p);
        REQUIRE(h.view()[0] == Catch::Approx(50.0));
        REQUIRE(h.view()[1] == Catch::Approx(200.0));
    }

    SECTION("rhoMax/rhoMin are converted with the reference density")
    {
        auto p = makeP(std::vector<NeoN::scalar>(n, 100.0), true, 100.0);
        auto rhoRef = makeRho(2.0);
        NeoFOAM::PressureLimitSettings s;
        s.rhoMax = 6.0; // max(6/2, 1) * 100
        s.rhoMin = 0.5; // min(0.5/2, 1) * 100
        NeoFOAM::PressureControl pc(p, rhoRef, s);
        REQUIRE(pc.pMax() == Catch::Approx(300.0));
        REQUIRE(pc.pMin() == Catch::Approx(25.0));
    }

    SECTION("an absolute pMax wins over its factor")
    {
        auto p = makeP(std::vector<NeoN::scalar>(n, 100.0), true, 100.0);
        NeoFOAM::PressureLimitSettings s;
        s.pMax = 150.0;
        s.pMaxFactor = 2.0;
        NeoFOAM::PressureControl pc(p, rho, s);
        REQUIRE(pc.pMax() == Catch::Approx(150.0));
    }

    SECTION("a factor without a value-fixing patch is an error, as in OpenFOAM")
    {
        auto p = makeP(std::vector<NeoN::scalar>(n, 100.0), false, 100.0);
        NeoFOAM::PressureLimitSettings s;
        s.pMaxFactor = 2.0;
        REQUIRE_THROWS(NeoFOAM::PressureControl(p, rho, s));
        NeoFOAM::PressureLimitSettings t;
        t.rhoMin = 0.5;
        REQUIRE_THROWS(NeoFOAM::PressureControl(p, rho, t));
    }

    SECTION("a boundary value beyond the limit clips nothing inside and keeps the boundary")
    {
        auto p = makeP(std::vector<NeoN::scalar>(n, 100.0), false, 500.0); // boundary above pMax
        NeoFOAM::PressureLimitSettings s;
        s.pMin = 50.0;
        s.pMax = 200.0;
        NeoFOAM::PressureControl pc(p, rho, s);
        REQUIRE(pc.limit(p) == true);
        REQUIRE(internal(p).view()[0] == Catch::Approx(100.0));
        auto b = p.boundaryData().value().copyToHost();
        REQUIRE(b.view()[0] == Catch::Approx(500.0)); // the caller re-evaluates the BCs
    }
}
