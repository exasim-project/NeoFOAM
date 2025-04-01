// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/NeoFOAM.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

fvcc::VolumeField<NeoFOAM::scalar>
createVolumeField(const NeoFOAM::UnstructuredMesh& mesh, std::string fieldName)
{
    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
    {
        NeoFOAM::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", 2.0);
        bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
    }
    fvcc::VolumeField<NeoFOAM::scalar> vf(mesh.exec(), fieldName, mesh, bcs);
    NeoFOAM::fill(vf.internalField(), 1.0);
    return vf;
}

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
        for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
        {
            NeoFOAM::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 2.0);
            bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
        }
        NeoFOAM::Field internalField =
            NeoFOAM::Field<NeoFOAM::scalar>(mesh.exec(), mesh.nCells(), 1.0);
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


TEST_CASE("Field Document")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );


    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("create petscSolverContextDoc: " + execName)
    {

        NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar> petsctx(exec);
        fvcc::petscSolverContextDocument petscSolverContextDoc(petsctx, "testEqn");


        SECTION("validate Document")
        {
            REQUIRE(petscSolverContextDoc.doc().validate());
            REQUIRE_NOTHROW(petscSolverContextDoc.doc().validate());

            REQUIRE(petscSolverContextDoc.doc().keys().size() == 3);
            REQUIRE(petscSolverContextDoc.id().substr(0, 4) == "doc_");
            REQUIRE(petscSolverContextDoc.eqnName() == "testEqn");


            NeoFOAM::Document& doc = petscSolverContextDoc.doc();
            REQUIRE(doc.validate());
            REQUIRE_NOTHROW(doc.validate());
            REQUIRE(doc.keys().size() == 3);
            REQUIRE(doc.id().substr(0, 4) == "doc_");


            const auto& constContext =
                petscSolverContextDoc
                    .context<NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar>>(
                    );
            auto& context =
                petscSolverContextDoc
                    .context<NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar>>(
                    );

            REQUIRE(context.initialized() == false);
            REQUIRE(&constContext == &context);
        }

        SECTION("modify Document")
        {
            auto& context =
                petscSolverContextDoc
                    .context<NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar>>(
                    );

            NeoFOAM::Field<NeoFOAM::scalar> values(
                exec, {10.0, 4.0, 7.0, 2.0, 10.0, 8.0, 3.0, 6.0, 10.0}
            );

            NeoFOAM::Field<int> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
            NeoFOAM::Field<int> rowPtrs(exec, {0, 3, 6, 9});
            NeoFOAM::la::CSRMatrix<NeoFOAM::scalar, int> csrMatrix(values, colIdx, rowPtrs);

            NeoFOAM::Field<NeoFOAM::scalar> rhs(exec, {1.0, 2.0, 3.0});
            NeoFOAM::la::LinearSystem<NeoFOAM::scalar, int> linearSystem(csrMatrix, rhs, "custom");
            NeoFOAM::Field<NeoFOAM::scalar> x(exec, {0.0, 0.0, 0.0});

            context.initialize(linearSystem);

            REQUIRE(context.initialized() == true);
        }
    }
}

TEST_CASE("FieldCollection")
{
    NeoFOAM::Database db;

    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.name(); }, exec);
    NeoFOAM::UnstructuredMesh mesh = NeoFOAM::createSingleCellMesh(exec);

    SECTION("create petscSolverContextCollection: " + execName)
    {
        fvcc::petscSolverContextCollection petscSolverContextCollection(db, "testCollection");
        REQUIRE(petscSolverContextCollection.size() == 0);
    }

    SECTION("add FieldDocument to FieldCollection" + execName)
    {
        fvcc::petscSolverContextCollection petscSolverContextCollection =
            fvcc::petscSolverContextCollection::instance(db, "testFieldCollection");
        REQUIRE(db.size() == 1);

        NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar> petsctx1(exec);
        fvcc::petscSolverContextDocument petscSolverContextDoc1(petsctx1, "testEqn1");

        NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar> petsctx2(exec);
        fvcc::petscSolverContextDocument petscSolverContextDoc2(petsctx2, "testEqn2");


        NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar> petsctx3(exec);
        fvcc::petscSolverContextDocument petscSolverContextDoc3(petsctx3, "testEqn3");
        std::cout << petscSolverContextDoc1.id() << "\n";
        std::cout << petscSolverContextDoc2.id() << "\n";
        std::cout << petscSolverContextDoc3.id() << "\n";

        REQUIRE(petscSolverContextCollection.insert(petscSolverContextDoc1) != std::string(""));
        REQUIRE(petscSolverContextCollection.insert(petscSolverContextDoc2) != std::string(""));
        REQUIRE(petscSolverContextCollection.insert(petscSolverContextDoc3) != std::string(""));

        REQUIRE(petscSolverContextCollection.size() == 3);


        SECTION("get petscSolverContextDoc from petscSolverContextCollection")
        {
            fvcc::petscSolverContextDocument petscSolverContextDoc =
                petscSolverContextCollection.petscSolverContextDoc(petscSolverContextDoc3.id());
            REQUIRE(petscSolverContextDoc.doc().validate());
            REQUIRE_NOTHROW(petscSolverContextDoc.doc().validate());
            REQUIRE(petscSolverContextDoc.doc().keys().size() == 3);
            REQUIRE(petscSolverContextDoc.doc().id().substr(0, 4) == "doc_");


            const auto& constContext =
                petscSolverContextDoc
                    .context<NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar>>(
                    );
            auto& context =
                petscSolverContextDoc
                    .context<NeoFOAM::la::petscSolverContext::petscSolverContext<NeoFOAM::scalar>>(
                    );

            REQUIRE(context.initialized() == false);
            REQUIRE(&constContext == &context);
        }
        /*
                SECTION("query")
                {
                    auto resTimeIndex =
                        fieldCollection.find([](const NeoFOAM::Document& doc)
                                             { return doc.get<std::int64_t>("timeIndex") == 1; });

                    REQUIRE(resTimeIndex.size() == 3);

                    auto resSubCycleIndex =
                        fieldCollection.find([](const NeoFOAM::Document& doc)
                                             { return doc.get<std::int64_t>("subCycleIndex") == 5;
           });

                    REQUIRE(resSubCycleIndex.size() == 0);

                    auto resName = fieldCollection.find([](const NeoFOAM::Document& doc)
                                                        { return doc.get<std::string>("name") ==
           "T3"; });

                    REQUIRE(resName.size() == 1);

                    const auto& fieldDoc2 = fieldCollection.fieldDoc(resName[0]);
                    REQUIRE(fieldDoc2.timeIndex() == 1);
                }
        */
    }

    SECTION("register " + execName)
    {

        fvcc::FieldCollection& fieldCollection1 =
            fvcc::FieldCollection::instance(db, "newTestFieldCollection");
        REQUIRE(db.size() == 1);

        fvcc::VolumeField<NeoFOAM::scalar>& t =
            fieldCollection1.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(CreateField {
                .name = "T", .mesh = mesh, .timeIndex = 1, .iterationIndex = 1, .subCycleIndex = 1
            });

        REQUIRE(t.name == "T");
        REQUIRE(t.hasDatabase());
        REQUIRE(t.internalField().copyToHost()[0] == 1.0);
        REQUIRE(t.registered());

        SECTION("Construct from Field")
        {
            fvcc::FieldCollection& fieldCollection2 = fvcc::FieldCollection::instance(t);
            REQUIRE(fieldCollection2.size() == 1);
            const fvcc::VolumeField<NeoFOAM::scalar>& constT = t;
            const fvcc::FieldCollection& fieldCollection3 = fvcc::FieldCollection::instance(constT);
            REQUIRE(fieldCollection3.size() == 1);
        }


        SECTION("register from existing field")
        {
            fvcc::FieldCollection& fieldCollection2 = fvcc::FieldCollection::instance(t);
            fvcc::VolumeField<NeoFOAM::scalar>& t3 =
                fieldCollection2.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(
                    fvcc::CreateFromExistingField<fvcc::VolumeField<NeoFOAM::scalar>> {
                        .name = "T3", .field = t
                    }
                );

            const fvcc::FieldDocument& docT = fieldCollection2.fieldDoc(t3.key);
            const fvcc::FieldDocument& docT3 = fieldCollection2.fieldDoc(t.key);

            REQUIRE(docT.timeIndex() == docT3.timeIndex());
            REQUIRE(docT.iterationIndex() == docT3.iterationIndex());
            REQUIRE(docT.subCycleIndex() == docT3.subCycleIndex());
        }
    }
}
