// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/explicitRungeKutta.hpp"


using namespace NeoFOAM;
namespace dsl = NeoFOAM::DSL;


class Divergence
{

public:

    std::string display() const { return "Divergence"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale)
    {
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1.0 * scale; }
        );
    }

    dsl::EqnTerm::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() const { return nullptr; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::EqnTerm::Type termType_;
    NeoFOAM::Executor exec_;
    std::size_t nCells_;
};

class Temporal
{

public:

    std::string display() const { return "Temporal"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale)
    {
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1 * scale; }
        );
    }

    dsl::EqnTerm::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() const { return nullptr; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::EqnTerm::Type termType_;
    const NeoFOAM::Executor exec_;
    std::size_t nCells_;
};


TEST_CASE("TimeIntegration")
{
    auto exec = NeoFOAM::SerialExecutor();
    namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

    NeoFOAM::Dictionary dict;
    dict.insert("type", std::string("ExplicitRungeKutta"));
    dict.insert("Relative Tolerance", scalar(1.e-5));
    dict.insert("Absolute Tolerance", scalar(1.e-10));
    dict.insert("Fixed Step Size", scalar(1.0e-3));
    dict.insert("End Time", scalar(0.005));

    dsl::EqnTerm divTerm = Divergence(dsl::EqnTerm::Type::Explicit, exec, 1);
    dsl::EqnTerm ddtTerm = Temporal(dsl::EqnTerm::Type::Temporal, exec, 1);
    dsl::EqnSystem eqnSys = ddtTerm + divTerm;

    dsl::EqnSystem sys2 = eqnSys;


    fvcc::ExplicitRungeKutta timeIntergrator(eqnSys, dict);
    timeIntergrator.solve();
}
