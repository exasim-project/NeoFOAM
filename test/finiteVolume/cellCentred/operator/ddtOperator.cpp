// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoFOAM/NeoFOAM.hpp"
#include "NeoFOAM/core/database/database.hpp"
#include "NeoFOAM/core/database/collection.hpp"
#include "NeoFOAM/core/database/document.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

using Operator = NeoFOAM::dsl::Operator;

namespace NeoFOAM
{

template<typename ValueType>
struct CreateField
{
    std::string name;
    const NeoFOAM::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoFOAM::Document operator()(NeoFOAM::Database& db)
    {

        std::vector<fvcc::VolumeBoundary<ValueType>> bcs {};
        for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
        {
            NeoFOAM::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
        }
        NeoFOAM::Field<ValueType> internalField(mesh.exec(), mesh.nCells(), 1.0);
        fvcc::VolumeField<ValueType> vf(
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


TEMPLATE_TEST_CASE("DdtOperator", "[template]", NeoFOAM::scalar) //, NeoFOAM::Vector)
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {})//,
        //NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        //NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );


    NeoFOAM::Database db;
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    auto mesh = createSingleCellMesh(exec);
    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);

    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<TestType> coeff(exec, "coeff", mesh, coeffBCs);
    fill(coeff.internalField(), 2.0);
    fill(coeff.boundaryField().value(), 0.0);
    coeff.correctBoundaryConditions();

    fvcc::FieldCollection& fieldCollection =
            fvcc::FieldCollection::instance(db, "testFieldCollection");

    fvcc::VolumeField<TestType>& phi =
    fieldCollection.registerField<fvcc::VolumeField<TestType>>(
        CreateField<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );
    fill(phi.internalField(), 10 * one<TestType>());
    fill(phi.boundaryField().value(), zero<TestType>());
    fill(oldTime(phi).internalField(), zero<TestType>());
    phi.correctBoundaryConditions();

    SECTION("explicit DdtOperator " + execName)
    {
        fvcc::DdtOperator<TestType> ddtTerm(Operator::Type::Explicit, phi);
        auto source = Field<TestType>(exec, phi.size(), zero<TestType>());
        ddtTerm.explicitOperation(source, 1.0, 1.0);

        // cell has one cell
        auto hostSource = source.copyToHost();
        REQUIRE(mag(hostSource[0] - 10 * one<TestType>()) == Catch::Approx(0.0).margin(1e-8));
    }

    // SECTION("implicit DdtOperator " + execName)
    // {
    //     fvcc::SourceTerm<TestType> sTerm(Operator::Type::Explicit, coeff, phi);
    //     auto ls = sTerm.createEmptyLinearSystem();
    //     sTerm.implicitOperation(ls);
    //     auto lsHost = ls.copyToHost();
    //     auto vol = mesh.cellVolumes().copyToHost();
    //     // results = coeff*vol
    //     REQUIRE(
    //         mag(lsHost.matrix().values()[0] - 2.0 * vol[0] * one<TestType>())
    //         == Catch::Approx(0.0).margin(1e-8)
    //     );
    // }
}

}
