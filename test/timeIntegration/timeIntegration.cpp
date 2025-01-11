// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "../dsl/common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

// only needed for msvc
template class NeoFOAM::timeIntegration::ForwardEuler<VolumeField>;

struct CreateField
{
    std::string name;
    const NeoFOAM::UnstructuredMesh& mesh;
    NeoFOAM::scalar value = 0;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoFOAM::Document operator()(NeoFOAM::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
        NeoFOAM::Field<NeoFOAM::scalar> internalField(mesh.exec(), mesh.nCells(), value);
        fvcc::VolumeField<NeoFOAM::scalar> vf(
            mesh.exec(), name, mesh, internalField, bcs, db, "", ""
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

TEST_CASE("TimeIntegration")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    NeoFOAM::Database db;
    auto mesh = NeoFOAM::createSingleCellMesh(exec);
    fvcc::FieldCollection& fieldCollection = fvcc::FieldCollection::instance(db, "fieldCollection");

    NeoFOAM::Dictionary fvSchemes;
    NeoFOAM::Dictionary ddtSchemes;
    ddtSchemes.insert("type", std::string("forwardEuler"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);
    NeoFOAM::Dictionary fvSolution;

    fvcc::VolumeField<NeoFOAM::scalar>& vf =
        fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(
            CreateField {.name = "vf", .mesh = mesh, .value = 2.0, .timeIndex = 1}
        );

    SECTION("Create expression and perform explicitOperation on " + execName)
    {
        auto dummy = Dummy(vf);
        Operator ddtOperator = NeoFOAM::dsl::temporal::ddt(vf);

        // ddt(U) = f
        auto eqn = ddtOperator + dummy;
        double dt {2.0};
        double time {1.0};


        // int(ddt(U)) + f = 0
        // (U^1-U^0)/dt = -f
        // U^1 = - f * dt + U^0, where dt = 2, f=1, U^0=2.0 -> U^1=-2.0
        NeoFOAM::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
        REQUIRE(getField(vf.internalField()) == -2.0);
    }
}
