// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoFOAM/NeoFOAM.hpp"

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
            dict.insert("fixedValue", ValueType(2.0));
            bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
        }
        NeoFOAM::Field<ValueType> internalField(mesh.exec(), mesh.nCells(), ValueType(1.0));
        fvcc::VolumeField<ValueType> vf(mesh.exec(), name, mesh, internalField, bcs, db, "", "");

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

TEMPLATE_TEST_CASE("DdtOperator", "[template]", NeoFOAM::scalar, NeoFOAM::Vector)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoFOAM::Database db;
    auto mesh = createSingleCellMesh(exec);
    auto sp = NeoFOAM::finiteVolume::cellCentred::SparsityPattern {mesh};

    fvcc::FieldCollection& fieldCollection =
        fvcc::FieldCollection::instance(db, "testFieldCollection");

    fvcc::VolumeField<TestType>& phi = fieldCollection.registerField<fvcc::VolumeField<TestType>>(
        CreateField<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );
    fill(phi.internalField(), 10 * one<TestType>());
    fill(phi.boundaryField().value(), zero<TestType>());
    fill(oldTime(phi).internalField(), -1.0 * one<TestType>());
    phi.correctBoundaryConditions();

    SECTION("explicit DdtOperator " + execName)
    {
        fvcc::DdtOperator<TestType> ddtTerm(Operator::Type::Explicit, phi);
        auto source = Field<TestType>(exec, phi.size(), zero<TestType>());
        ddtTerm.explicitOperation(source, 1.0, 0.5);

        // cell has one cell
        const auto vol = mesh.cellVolumes().copyToHost();
        auto hostSource = source.copyToHost();
        auto values = hostSource.span();
        for (auto ii = 0; ii < values.size(); ++ii)
        {
            REQUIRE(
                values[ii] == vol[0] * TestType(22.0)
            ); // => (phi^{n + 1} - phi^{n})/dt*V => (10 -- 1)/.5*V = 22V
        }
    }

    SECTION("implicit DdtOperator " + execName)
    {
        auto ls = NeoFOAM::la::createEmptyLinearSystem<
            TestType,
            NeoFOAM::localIdx,
            NeoFOAM::finiteVolume::cellCentred::SparsityPattern>(sp);
        fvcc::DdtOperator<TestType> ddtTerm(Operator::Type::Implicit, phi);
        ddtTerm.implicitOperation(ls, 1.0, 0.5);

        auto lsHost = ls.copyToHost();
        const auto vol = mesh.cellVolumes().copyToHost();
        const auto matrixValues = lsHost.matrix().values();
        const auto rhs = lsHost.rhs().span();

        for (auto ii = 0; ii < matrixValues.size(); ++ii)
        {
            REQUIRE(matrixValues[ii] == 2.0 * vol[0] * one<TestType>()); // => 1/dt*V => 1/.5*V = 2V
            REQUIRE(rhs[ii] == -2.0 * vol[0] * one<TestType>()); // => phi^{n}/dt*V => -1/.5*V = -2V
        }
    }
}

}
