// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"

class Laplacian
{

public:

    std::string display() const { return "Laplacian"; }

    void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) { exp += 1 * scale; }
};

class Divergence
{

public:

    std::string display() const { return "Divergence"; }

    void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) { exp += 1 * scale; }
};

namespace dsl = NeoFOAM::DSL;

TEST_CASE("DSL")
{
    dsl::EqnTerm lapTerm = Laplacian();

    REQUIRE("Laplacian" == lapTerm.display());

    dsl::EqnTerm divTerm = Divergence();

    REQUIRE("Divergence" == divTerm.display());
    NeoFOAM::scalar source = 0;

    lapTerm.explicitOperation(source);
    divTerm.explicitOperation(source);

    REQUIRE(source == 2.0);

    {
        dsl::EqnSystem eqnSys = lapTerm + divTerm;
        REQUIRE(eqnSys.eqnTerms_.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 2);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + lapTerm + divTerm + divTerm);
        REQUIRE(eqnSys.eqnTerms_.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == 4.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + Laplacian());
        REQUIRE(eqnSys.eqnTerms_.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 2.0);
    }

    {
        dsl::EqnSystem eqnSys = lapTerm - divTerm;
        REQUIRE(eqnSys.eqnTerms_.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 0.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm - Laplacian());
        REQUIRE(eqnSys.eqnTerms_.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 0.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm - lapTerm - divTerm - divTerm);
        REQUIRE(eqnSys.eqnTerms_.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == -2.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + Laplacian() + divTerm + Divergence());
        dsl::EqnSystem eqnSys2(lapTerm + Laplacian() + divTerm + Divergence());
        REQUIRE(eqnSys.eqnTerms_.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == 4.0);
        dsl::EqnSystem combinedEqnSys = eqnSys + eqnSys2;
        REQUIRE(combinedEqnSys.eqnTerms_.size() == 8);
        REQUIRE(combinedEqnSys.explicitOperation() == 8.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + Laplacian() - divTerm - Divergence());
        dsl::EqnSystem eqnSys2(lapTerm - Laplacian() - divTerm - Divergence());
        REQUIRE(eqnSys.eqnTerms_.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == 0.0);
        REQUIRE(eqnSys2.eqnTerms_.size() == 4);
        REQUIRE(eqnSys2.explicitOperation() == -2.0);
        REQUIRE(-eqnSys2.explicitOperation() == 2.0);
        dsl::EqnSystem combinedEqnSys = eqnSys2 - eqnSys;
        REQUIRE(combinedEqnSys.eqnTerms_.size() == 8);
        REQUIRE(combinedEqnSys.explicitOperation() == -2.0);
    }
}
