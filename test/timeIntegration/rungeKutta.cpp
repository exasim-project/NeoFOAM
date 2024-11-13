// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <string>

#include "../dsl/common.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/dsl/ddt.hpp"
#include "NeoFOAM/dsl/expression.hpp"
#include "NeoFOAM/dsl/operator.hpp"
#include "NeoFOAM/dsl/solver.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/timeIntegration/rungeKutta.hpp"


namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using Field = NeoFOAM::Field<NeoFOAM::scalar>;
using Coeff = NeoFOAM::dsl::Coeff;
using Operator = NeoFOAM::dsl::Operator;
using Executor = NeoFOAM::Executor;
using VolumeField = fvcc::VolumeField<NeoFOAM::scalar>;
using OperatorMixin = NeoFOAM::dsl::OperatorMixin<VolumeField>;
using BoundaryFields = NeoFOAM::BoundaryFields<NeoFOAM::scalar>;
using Ddt = NeoFOAM::dsl::temporal::Ddt<VolumeField>;

class DivLikeOpper : public OperatorMixin
{

public:

    DivLikeOpper(VolumeField& field) : OperatorMixin(field.exec(), field, Operator::Type::Explicit)
    {}

    void explicitOperation(Field& source) const
    {
        auto sourceSpan = source.span();
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] += source[i] * source[i]; }
        );
    }

    std::string getName() const { return "DivLikeOpper"; }
};

TEST_CASE("TimeIntegration - Runge Kutta")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    NeoFOAM::Dictionary fvSchemes;
    NeoFOAM::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("Runge-Kutta"));
    ddtSchemes.insert("Runge-Kutta Method", std::string("Heun"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoFOAM::Dictionary fvSolution;

    auto mesh = NeoFOAM::createSingleCellMesh(exec);
    Field fA(exec, 1.0, 1.0);
    BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    auto vf = VolumeField(exec, mesh, fA, bf, bcs);
    double time = 0.0;
    double deltaTime = 0.01;

    SECTION("Solve on " + execName)
    {
        std::cout << "\n";
        Operator ddtOp = Ddt(vf);
        auto divOp = DivLikeOpper(vf);
        auto eqn = ddtOp + divOp; // here the time integrator will deal with this.
        for (auto i = 0; i < 10; i++)
        {
            std::cout << "\nb: " << vf.internalField().copyToHost()[0];
            solve(eqn, vf, time, deltaTime, fvSchemes, fvSolution); // perform 1 step.
            time += deltaTime;
            std::cout << "\ta: " << vf.internalField().copyToHost()[0];
        }
        std::cout << "Numerical: " << std::setprecision(10) << vf.internalField().copyToHost()[0]
                  << " Analytical: " << 1.0 / (1.0 - 0.01);
    }
}
