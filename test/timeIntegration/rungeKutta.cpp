// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <string>

#include "../dsl/common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"


namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using Field = NeoFOAM::Field<NeoFOAM::scalar>;
using Coeff = NeoFOAM::dsl::Coeff;
using SpatialOperator = NeoFOAM::dsl::SpatialOperator<NeoFOAM::scalar>;
using TemporalOperator = NeoFOAM::dsl::TemporalOperator<NeoFOAM::scalar>;
using Executor = NeoFOAM::Executor;
using VolumeField = fvcc::VolumeField<NeoFOAM::scalar>;
using OperatorMixin = NeoFOAM::dsl::OperatorMixin<VolumeField>;
using BoundaryFields = NeoFOAM::BoundaryFields<NeoFOAM::scalar>;
using Ddt = NeoFOAM::dsl::temporal::Ddt<VolumeField>;

// only for msvc
template class NeoFOAM::timeIntegration::RungeKutta<VolumeField>;

class YSquared : public OperatorMixin
{

public:

    using FieldValueType = NeoFOAM::scalar;

    YSquared(VolumeField& field)
        : OperatorMixin(field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit)
    {}

    void explicitOperation(Field& source) const
    {
        auto sourceSpan = source.span();
        auto fieldData = field_.internalField().data();
        NeoFOAM::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const size_t i) { sourceSpan[i] -= fieldData[i] * fieldData[i]; }
        );
    }

    std::string getName() const { return "YSquared"; }
};

struct CreateField
{
    std::string name;
    const NeoFOAM::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoFOAM::Document operator()(NeoFOAM::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        NeoFOAM::DomainField<NeoFOAM::scalar> domainField(
            mesh.exec(),
            NeoFOAM::Field<NeoFOAM::scalar>(mesh.exec(), mesh.nCells(), 1.0),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<NeoFOAM::scalar> vf(
            mesh.exec(), name, mesh, domainField, bcs, db, "", ""
        );
        return NeoFOAM::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateFieldDoc
        );
    }
};

TEST_CASE("TimeIntegration - Runge Kutta")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    NeoFOAM::scalar convergenceTolerance = 1.0e-4; // how much lower we accept that expected order.

    // Set up dictionary.
    NeoFOAM::Database db;
    NeoFOAM::Dictionary fvSchemes;
    NeoFOAM::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("Runge-Kutta"));
    ddtSchemes.insert("Runge-Kutta-Method", std::string("Forward-Euler"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoFOAM::Dictionary fvSolution;

    // Set up fields.
    auto mesh = NeoFOAM::createSingleCellMesh(exec);
    fvcc::FieldCollection& fieldCollection = fvcc::FieldCollection::instance(db, "fieldCollection");
    fvcc::VolumeField<NeoFOAM::scalar>& vf =
        fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(
            CreateField {.name = "vf", .mesh = mesh, .timeIndex = 1}
        );

    // Setup solve parameters.
    const NeoFOAM::scalar maxTime = 0.1;
    const NeoFOAM::scalar initialValue = 1.0;
    std::array<NeoFOAM::scalar, 2> deltaTime = {0.01, 0.001};

    SECTION("Solve on " + execName)
    {
        int iTest = 0;
        std::array<NeoFOAM::scalar, 2> error;
        for (auto dt : deltaTime)
        {
            // reset
            auto& vfOld = fvcc::oldTime(vf);
            NeoFOAM::scalar time = 0.0;
            vf.internalField() = initialValue;
            vfOld.internalField() = initialValue;

            // Set expression
            TemporalOperator ddtOp = NeoFOAM::dsl::imp::ddt(vfOld);

            auto divOp = YSquared(vfOld);
            auto eqn = ddtOp + divOp;

            // solve.
            while (time < maxTime)
            {
                NeoFOAM::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
                time += dt;
            }

            // check error.
            NeoFOAM::scalar analytical = 1.0 / (initialValue - maxTime);
            error[iTest] = std::abs(vf.internalField().copyToHost()[0] - analytical);
            iTest++;
        }

        // check order of convergence.
        NeoFOAM::scalar order = (std::log(error[0]) - std::log(error[1]))
                              / (std::log(deltaTime[0]) - std::log(deltaTime[1]));
        REQUIRE(order > (1.0 - convergenceTolerance));
    }
}
