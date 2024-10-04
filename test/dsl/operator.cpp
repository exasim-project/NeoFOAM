// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/DSL/coeff.hpp"
#include "NeoFOAM/DSL/operator.hpp"

using Field = NeoFOAM::Field<NeoFOAM::scalar>;
using Coeff = NeoFOAM::DSL::Coeff;
using Operator = NeoFOAM::DSL::Operator;
using OperatorMixin = NeoFOAM::DSL::OperatorMixin;
using Executor = NeoFOAM::Executor;
using VolumeField = fvcc::VolumeField<NeoFOAM::scalar>;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

/* A dummy implementation of a Operator
 * following the Operator interface */
class Dummy : public OperatorMixin
{

public:

    Dummy(const Executor& exec, VolumeField& field) : OperatorMixin(exec), field_(field) {}

    void explicitOperation(Field& source, NeoFOAM::scalar scale)
    {
        auto sourceField = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            {0, source.size()},
            KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1.0 * scale; }
        );
    }

    const VolumeField& volumeField() const { return field_; }

    VolumeField& volumeField() { return field_; }

    Operator::Type getType() const { return Operator::Type::Explicit; }

private:

    VolumeField& field_;
};

TEST_CASE("Operator")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("Operator creation on " + execName)
    {

        Field fA(exec, 3, 2.0);

        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        auto vf = VolumeField(exec, mesh, bcs);
        auto b = Dummy(exec, vf);
    }

    SECTION("Supports Coefficients" + execName)
    {

        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        auto vf = VolumeField(exec, mesh, bcs);

        Field fA(exec, 3, 2.0);
        Field fB(exec, 3, 3.0);

        auto b = Dummy(exec, &fA);

        auto c = 2 * Dummy(exec, &fA);
        auto d = fB * Dummy(exec, &fA);
        auto e = Coeff(-3, fB) * Dummy(exec, &fA);

        auto coeffc = c.getCoefficient();
        auto coeffd = d.getCoefficient();
        auto coeffE = e.getCoefficient();
    }
}
