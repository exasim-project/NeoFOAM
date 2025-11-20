// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"

extern Foam::Time* timePtr; // A single time object

TEST_CASE("Convert OpenFOAM::dictionary dict to NeoN::Dict")
{
    Foam::dictionary testDict;
    testDict.add("label", 1);
    testDict.add("scalar", 2.1);
    testDict.add("scalar2", 2.0);
    testDict.add("vector", Foam::vector(1.0, 2.0, 3.0));
    testDict.add("word", "word");

    Foam::dictionary subDict;
    subDict.add("subScalar", 4.1);
    subDict.add("subVector", Foam::vector(5.0, 6.0, 7.0));
    subDict.add("subWord", "subWord");
    testDict.add("subDict", subDict);

    auto nfDict = NeoFOAM::convert(testDict);

    REQUIRE(nfDict.getVal<NeoN::label>("label") == 1);
    REQUIRE(nfDict.getVal<NeoN::scalar>("scalar") == 2.1);
    REQUIRE(nfDict.getVal<NeoN::scalar>("scalar2") == 2.0);
    REQUIRE(nfDict.getVal<NeoN::Vec3>("vector") == NeoN::Vec3(1.0, 2.0, 3.0));
    REQUIRE(nfDict.getVal<std::string>("word") == "word");

    auto& nfSubDict = nfDict.subDict("subDict");
    REQUIRE(nfSubDict.getVal<NeoN::scalar>("subScalar") == 4.1);
    REQUIRE(nfSubDict.getVal<NeoN::Vec3>("subVector") == NeoN::Vec3(5.0, 6.0, 7.0));
}


TEST_CASE("read fvSchemes")
{
    Foam::Time& runTime = *timePtr;
    auto meshPtr = NeoFOAM::createMesh(NeoN::SerialExecutor {}, runTime);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;

    Foam::dictionary fvSchemes = mesh.schemesDict();

    NeoN::Dictionary fvSchemesDict = NeoFOAM::convert(mesh.schemesDict());
    auto keys = fvSchemesDict.keys();

    REQUIRE(fvSchemesDict.subDict("ddtSchemes").getVal<std::string>("ddt(T)") == "Euler");
    auto gradSchemeKeys = fvSchemesDict.subDict("gradSchemes").keys();
    NeoN::TokenList limitedToken =
        fvSchemesDict.subDict("gradSchemes").getVal<NeoN::TokenList>("limited");
    REQUIRE(limitedToken.size() == 4);
    REQUIRE(limitedToken.get<std::string>(0) == "cellLimited");
    REQUIRE(limitedToken.get<std::string>(1) == "Gauss");
    REQUIRE(limitedToken.get<std::string>(2) == "linear");
    REQUIRE(limitedToken.get<NeoN::label>(3) == 1);

    NeoN::TokenList gradU = fvSchemesDict.subDict("gradSchemes").getVal<NeoN::TokenList>("grad(U)");
    REQUIRE(gradU.size() == 2);
    REQUIRE(gradU.get<std::string>(0) == "Gauss");
    REQUIRE(gradU.get<std::string>(1) == "linear");
}

TEST_CASE("read testDictionary from disk")
{
    Foam::Time& runTime = *timePtr;
    Foam::IOdictionary ofTestDict(Foam::IOobject(
        "testDictionary",
        runTime.system(),
        runTime,
        Foam::IOobject::MUST_READ,
        Foam::IOobject::NO_WRITE
    ));

    NeoN::Dictionary nfTestDict = NeoFOAM::convert(ofTestDict);

    REQUIRE(nfTestDict.getVal<NeoN::label>("label") == 1);
    REQUIRE(nfTestDict.getVal<NeoN::scalar>("scalar") == 2.1);
    REQUIRE(nfTestDict.getVal<NeoN::scalar>("scalar2") == 2.0);
    REQUIRE(nfTestDict.getVal<int>("scalarWriteAnsInt") == 2);
    REQUIRE(nfTestDict.getVal<NeoN::Vec3>("vector") == NeoN::Vec3(1.0, 2.0, 3.0));
    REQUIRE(nfTestDict.getVal<std::string>("word") == "word");

    NeoN::Dictionary& nfSubDict = nfTestDict.subDict("subDict");
    REQUIRE(nfSubDict.getVal<NeoN::scalar>("subScalar") == 4.1);
    REQUIRE(nfSubDict.getVal<NeoN::Vec3>("subVector") == NeoN::Vec3(5.0, 6.0, 7.0));
    REQUIRE(nfSubDict.getVal<std::string>("subWord") == "subWord");
}
