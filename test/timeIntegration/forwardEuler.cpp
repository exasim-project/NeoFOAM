// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/NeoFOAM.hpp"


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

class TimeTerm
{

public:

    std::string name() const { return "TimeTerm"; }

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
    dict.insert("type", std::string("forwardEuler"));

    dsl::EqnTerm divTerm = Divergence(dsl::EqnTerm::Type::Explicit, exec, 1);

    dsl::EqnTerm ddtTerm = TimeTerm(dsl::EqnTerm::Type::Temporal, exec, 1);

    dsl::EqnSystem eqnSys = ddtTerm + divTerm;

    fvcc::TimeIntegration timeIntergrator(eqnSys, dict);
}
