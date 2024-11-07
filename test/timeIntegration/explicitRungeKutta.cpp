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
#include "NeoFOAM/timeIntegration/explicitRungeKutta.hpp"


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

    DivLikeOpper(VolumeField& field, NeoFOAM::scalar timeOffset)
        : OperatorMixin(field.exec(), field, Operator::Type::Explicit),
          internalTime_(field.exec(), field.size(), timeOffset)
    {}

    void explicitOperation(Field& source) const
    {
        auto sourceSpan = source.span();
        auto fieldSpan = field_.internalField().span();
        auto coeff = getCoefficient();
        auto internalTimeSpan = internalTime_.span();

        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) {
                sourceSpan[i] += coeff[i] * fieldSpan[i] + internalTimeSpan[i];
            }
        );
        std::cout << "\n" << __LINE__ << std::flush;
    }

    void update(NeoFOAM::scalar dt)
    {
        internalTime_ += Field(internalTime_.exec(), internalTime_.size(), dt);
    } // I am a bad person

    std::string getName() const { return "DivLikeOpper"; }


private:

    Field internalTime_;
};

TEST_CASE("TimeIntegration")
{
    auto exec = NeoFOAM::SerialExecutor();

    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    Field fA(exec, 1, 2.0);
    BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());
    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    auto vf = VolumeField(exec, mesh, fA, bf, bcs);

    double dt {1.0e-3};
    NeoFOAM::Dictionary fvSchemes;
    NeoFOAM::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("explicitRungeKutta"));
    ddtSchemes.insert("Relative Tolerance", NeoFOAM::scalar(1.e-5));
    ddtSchemes.insert("Absolute Tolerance", NeoFOAM::scalar(1.e-10));
    ddtSchemes.insert("Fixed Step Size", NeoFOAM::scalar(dt));
    ddtSchemes.insert("End Time", NeoFOAM::scalar(0.005));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoFOAM::Dictionary fvSolution;

    Operator ddtOp = Ddt(vf);
    auto divOp = DivLikeOpper(vf, 1.0);
    auto eqn = ddtOp + divOp;                  // here the time integrator will deal with this.
    solve(eqn, vf, dt, fvSchemes, fvSolution); // perform 1 step.
    divOp.update(dt);
    solve(eqn, vf, dt, fvSchemes, fvSolution); // perform 1 step.
}
