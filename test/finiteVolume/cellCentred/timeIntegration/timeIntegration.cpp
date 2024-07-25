// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/core/dictionary.hpp"


#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"


namespace dsl = NeoFOAM::DSL;

class Divergence
{

public:

    std::string display() const { return "Divergence"; }

    void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) { exp += 1 * scale; }

    dsl::EqnTerm::Type getType() const { return termType_; }

    dsl::EqnTerm::Type termType_;
};

class TimeTerm
{

public:

    std::string display() const { return "TimeTerm"; }

    void explicitOperation(NeoFOAM::scalar& exp, NeoFOAM::scalar scale) { exp += 1 * scale; }

    dsl::EqnTerm::Type getType() const { return termType_; }

    dsl::EqnTerm::Type termType_;
};


TEST_CASE("TimeIntegration")
{
    namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

    NeoFOAM::Dictionary dict;
    dict.insert("type", std::string("forwardEuler"));

    dsl::EqnTerm divTerm = Divergence(dsl::EqnTerm::Type::Explicit);

    dsl::EqnTerm ddtTerm = TimeTerm(dsl::EqnTerm::Type::Temporal);

    dsl::EqnSystem eqnSys = ddtTerm + divTerm;

    fvcc::TimeIntegration timeIntergrator(eqnSys, dict);
}
