// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/finiteVolume/cellCentred/timeIntegration/timeIntegration.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/DSL/operator.hpp"
#include "NeoFOAM/DSL/eqnSystem.hpp"

namespace dsl = NeoFOAM::DSL;

class Divergence : public NeoFOAM::DSL::OperatorMixin
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

    dsl::Operator::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() const { return nullptr; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::Operator::Type termType_;

    NeoFOAM::Executor exec_;

    std::size_t nCells_;
};

class TimeOperator
{

public:

    std::string display() const { return "TimeOperator"; }

    void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale)
    {
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1 * scale; }
        );
    }

    dsl::Operator::Type getType() const { return termType_; }

    fvcc::VolumeField<NeoFOAM::scalar>* volumeField() const { return nullptr; }

    const NeoFOAM::Executor& exec() const { return exec_; }

    std::size_t nCells() const { return nCells_; }

    dsl::Operator::Type termType_;
    const NeoFOAM::Executor exec_;
    std::size_t nCells_;
};


TEST_CASE("TimeIntegration")
{
    auto exec = NeoFOAM::SerialExecutor();
    namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

    NeoFOAM::Dictionary dict;
    dict.insert("type", std::string("forwardEuler"));

    // dsl::Operator divOperator = Divergence(dsl::Operator::Type::Explicit, exec, 1);

    // dsl::Operator ddtOperator = TimeOperator(dsl::Operator::Type::Temporal, exec, 1);

    // dsl::EqnSystem eqnSys = ddtOperator + divOperator;

    // fvcc::TimeIntegration timeIntergrator(eqnSys, dict);
}
