// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"


namespace dsl = NeoFOAM::DSL;

class Laplacian
{

public:

    std::string display() const { return "Laplacian"; }

    void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) { exp += 1 * scale; }

    dsl::EqnTerm::Type getType() const { return termType_; }

    dsl::EqnTerm::Type termType_;
};

class Divergence
{

public:

    std::string display() const { return "Divergence"; }

    void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) { exp += 1 * scale; }

    dsl::EqnTerm::Type getType() const { return termType_; }

    dsl::EqnTerm::Type termType_;
};


TEST_CASE("DSL")
{
    dsl::EqnTerm lapTerm = Laplacian(dsl::EqnTerm::Type::Explicit);

    REQUIRE("Laplacian" == lapTerm.display());

    dsl::EqnTerm divTerm = Divergence(dsl::EqnTerm::Type::Explicit);

    REQUIRE("Divergence" == divTerm.display());
    NeoFOAM::scalar source = 0;

    lapTerm.explicitOperation(source);
    divTerm.explicitOperation(source);

    REQUIRE(source == 2.0);

    {
        dsl::EqnSystem eqnSys = lapTerm + divTerm;
        REQUIRE(eqnSys.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 2);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + lapTerm + divTerm + divTerm);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == 4.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit));
        REQUIRE(eqnSys.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 2.0);
    }

    {
        dsl::EqnSystem eqnSys = lapTerm - divTerm;
        REQUIRE(eqnSys.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 0.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm - Laplacian(dsl::EqnTerm::Type::Explicit));
        REQUIRE(eqnSys.size() == 2);
        REQUIRE(eqnSys.explicitOperation() == 0.0);
    }

    {
        dsl::EqnSystem eqnSys(lapTerm - lapTerm - divTerm - divTerm);
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == -2.0);
    }

    {
        dsl::EqnSystem eqnSys(
            lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit) + divTerm
            + Divergence(dsl::EqnTerm::Type::Explicit)
        );
        dsl::EqnSystem eqnSys2(
            lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit) + divTerm
            + Divergence(dsl::EqnTerm::Type::Explicit)
        );
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == 4.0);
        dsl::EqnSystem combinedEqnSys = eqnSys + eqnSys2;
        REQUIRE(combinedEqnSys.size() == 8);
        REQUIRE(combinedEqnSys.explicitOperation() == 8.0);
    }

    {
        dsl::EqnSystem eqnSys(
            lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit) - divTerm
            - Divergence(dsl::EqnTerm::Type::Explicit)
        );
        dsl::EqnSystem eqnSys2(
            lapTerm - Laplacian(dsl::EqnTerm::Type::Explicit) - divTerm
            - Divergence(dsl::EqnTerm::Type::Explicit)
        );
        REQUIRE(eqnSys.size() == 4);
        REQUIRE(eqnSys.explicitOperation() == 0.0);
        REQUIRE(eqnSys2.size() == 4);
        REQUIRE(eqnSys2.explicitOperation() == -2.0);
        REQUIRE(-eqnSys2.explicitOperation() == 2.0);

        SECTION("multiplying eqnSys by 2")
        {
            dsl::EqnSystem multiplyEqnSys = 2.0 * eqnSys2;
            REQUIRE(multiplyEqnSys.size() == 4);
            REQUIRE(multiplyEqnSys.explicitOperation() == -4.0);
        }

        SECTION("adding eqnSys to eqnSys2")
        {
            dsl::EqnSystem addEqnSys = eqnSys2 + eqnSys;
            REQUIRE(addEqnSys.size() == 8);
            REQUIRE(addEqnSys.explicitOperation() == -2.0);
        }
        SECTION("subtracting eqnSys from eqnSys2")
        {
            std::cout << "subtracting eqnSys from eqnSys2" << std::endl;
            dsl::EqnSystem subtractEqnSys = eqnSys - eqnSys2;
            REQUIRE(subtractEqnSys.size() == 8);
            REQUIRE(subtractEqnSys.explicitOperation() == 2.0);
        }
    }
}
