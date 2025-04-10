// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

template<typename T>
using I = std::initializer_list<T>;

TEST_CASE("volumeField")
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::createSingleCellMesh(exec);
    std::vector<fvcc::VolumeBoundary<NeoN::scalar>> bcs {};
    for (auto patchi : I<size_t> {0, 1, 2, 3})
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 2.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoN::scalar>(mesh, dict, patchi));
    }

    SECTION("can instantiate volumeField with fixedValues on: " + execName)
    {
        fvcc::VolumeField<NeoN::scalar> vf(exec, "vf", mesh, bcs);
        NeoN::fill(vf.internalField(), 1.0);
        vf.correctBoundaryConditions();

        NeoN::Field<NeoN::scalar> internalField(mesh.exec(), mesh.nCells(), 1.0);

        REQUIRE(vf.exec() == exec);
        REQUIRE(vf.internalField().size() == 1);

        auto internalValues = vf.internalField().copyToHost();
        for (size_t i = 0; i < internalValues.size(); ++i)
        {
            REQUIRE(internalValues.span()[i] == 1.0);
        }

        auto values = vf.boundaryField().value().copyToHost();

        for (size_t i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.span()[i] == 2.0);
        }

        auto refValue = vf.boundaryField().refValue().copyToHost();
        for (size_t i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue.span()[i] == 2.0);
        }
    }

    SECTION("can instantiate volumeField with fixedValues from internal Field on: " + execName)
    {
        NeoN::Field<NeoN::scalar> internalField(mesh.exec(), mesh.nCells(), 1.0);

        fvcc::VolumeField<NeoN::scalar> vf(exec, "vf", mesh, internalField, bcs);
        vf.correctBoundaryConditions();

        auto internalValues = vf.internalField().copyToHost();
        for (size_t i = 0; i < internalValues.size(); ++i)
        {
            REQUIRE(internalValues.span()[i] == 1.0);
        }

        auto values = vf.boundaryField().value().copyToHost();

        for (size_t i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.span()[i] == 2.0);
        }

        auto refValue = vf.boundaryField().refValue().copyToHost();
        for (size_t i = 0; i < refValue.size(); ++i)
        {
            REQUIRE(refValue.span()[i] == 2.0);
        }
    }
}
