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
using BoundaryFields = NeoFOAM::BoundaryFields<NeoFOAM::scalar>;
namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

/* A dummy implementation of a Operator
 * following the Operator interface */
class Dummy : public OperatorMixin
{

public:

    Dummy(const Executor& exec, VolumeField& field) : OperatorMixin(exec), field_(field) {}

    void explicitOperation(Field& source)
    {
        auto sourceSpan = source.span();
        auto fieldSpan = field_.internalField().span();
        auto coeff = getCoefficient();
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] += coeff[i] * fieldSpan[i]; }
        );
    }

    std::string getName() const { return "Dummy"; }

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
        Field fA(exec, 1, 2.0);
        BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        auto vf = VolumeField(exec, mesh, fA, bf, bcs);
        auto b = Dummy(exec, vf);

        REQUIRE(b.getName() == "Dummy");
        REQUIRE(b.getType() == Operator::Type::Explicit);
    }

    SECTION("Supports Coefficients" + execName)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};

        Field fA(exec, 1, 2.0);
        Field fB(exec, 1, 2.0);
        BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
        auto vf = VolumeField(exec, mesh, fA, bf, bcs);

        auto c = 2 * Dummy(exec, vf);
        auto d = fB * Dummy(exec, vf);
        auto e = Coeff(-3, fB) * Dummy(exec, vf);

        auto coeffc = c.getCoefficient();
        auto coeffd = d.getCoefficient();
        auto coeffE = e.getCoefficient();

        Field source(exec, 1, 2.0);
        c.explicitOperation(source);

        // 2 += 2 * 2
        auto hostSourceC = source.copyToHost();
        REQUIRE(hostSourceC.span()[0] == 6.0);

        // 6 += 2 * 2
        d.explicitOperation(source);
        auto hostSourceD = source.copyToHost();
        REQUIRE(hostSourceD.span()[0] == 10.0);

        // 10 += - 6 * 2
        e.explicitOperation(source);
        auto hostSourceE = source.copyToHost();
        REQUIRE(hostSourceE.span()[0] == -2.0);
    }
}
