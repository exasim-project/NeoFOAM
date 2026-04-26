// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
// common test functions
#pragma once

#include <random>
#include <span>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_approx.hpp>
#include "catch2/common.hpp"

#include "NeoN/NeoN.hpp"
#include "catch2/executorGenerator.hpp"
#include "NeoFOAM/NeoFOAM.hpp"

#include "fvm.H"
#include "fvc.H"
#include "processorFvPatch.H"
// #include "fvCFD.H"

namespace NeoFOAM
{

/**
 * @brief Conditionally enters a Catch2 SECTION if the given condition is true.
 * @param COND The boolean condition to evaluate.
 * @param ... The section name and optional description forwarded to SECTION.
 */
#define SECTION_IF(COND, ...)                                                                      \
    if (COND) SECTION(__VA_ARGS__)

void randomizeField(auto& field)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);

    forAll(field, celli)
    {
        field[celli] = rand();
    }

    field.correctBoundaryConditions();
}

template<typename FieldType, typename RandomFunc>
FieldType createRandomField(
    const Foam::Time& runTime,
    const Foam::fvMesh& mesh,
    Foam::word name,
    RandomFunc rand
)
{
    FieldType t(
        Foam::IOobject(
            name,
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh
    );

    if constexpr (std::is_same_v<FieldType, Foam::surfaceScalarField> || std::is_same_v<FieldType, Foam::surfaceVectorField>)
    {
        // Surface field — fill internal + boundary
        auto& intF = t.primitiveFieldRef();
        forAll(intF, facei)
            intF[facei] = rand();

        forAll(t.boundaryField(), patchi)
        {
            auto& p = t.boundaryFieldRef()[patchi];
            forAll(p, i)
                p[i] = rand();
        }
    }
    else
    {
        for (auto celli = 0; celli < t.size(); celli++)
        {
            t[celli] = rand();
        }

        t.correctBoundaryConditions();
    }
    return t;
}


/* function to create a volScalarField filled with random values for test purposes */
auto randomScalarField(const Foam::Time& runTime, const Foam::fvMesh& mesh, Foam::word name)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);
    return createRandomField<Foam::volScalarField>(runTime, mesh, name, [&]() { return dis(gen); });
}

auto randomVectorField(const Foam::Time& runTime, const Foam::fvMesh& mesh, Foam::word name)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);
    return createRandomField<Foam::volVectorField>(
        runTime,
        mesh,
        name,
        [&]() {
            return Foam::vector {dis(gen), dis(gen), dis(gen)};
        }
    );
}

auto randomSurfaceScalarField(const Foam::Time& runTime, const Foam::fvMesh& mesh, Foam::word name)
{
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);
    return createRandomField<Foam::surfaceScalarField>(
        runTime,
        mesh,
        name,
        [&]() { return dis(gen); }
    );
}

/* comparison function for volumeFields */
template<typename NFFIELD, typename OFFIELD, typename Compare>
void compare(const NFFIELD& a, const OFFIELD& b, Compare comp, const bool withBoundaries = true)
{
    if constexpr (std::is_same_v<NFFIELD, NeoN::Vector<NeoN::scalar>> || std::is_same_v<NFFIELD, NeoN::Vector<NeoN::Vec3>>)
    {
        auto aHost = a.copyToHost();
        auto bSpan = std::span(b.cdata(), b.size());
        REQUIRE(aHost.size() >= bSpan.size());
        REQUIRE_THAT(aHost.view({0, bSpan.size()}), Catch::Matchers::RangeEquals(bSpan, comp));
    }
    else
    {
        auto aHost = a.internalVector().copyToHost();
        const auto bSpan = std::span(b.primitiveField().cdata(), b.size());
        // nf a span might be shorter than bSpan for surface fields
        REQUIRE(aHost.size() >= bSpan.size());
        REQUIRE_THAT(aHost.view({0, bSpan.size()}), Catch::Matchers::RangeEquals(bSpan, comp));

        // Assumes boundaryData() is stored as a contiguous concatenation
        // of boundary patches in the same order as b.boundaryField()
        if (withBoundaries)
        {
            auto ofBoundaryData = std::vector<typename NFFIELD::VectorValueType> {};
            auto nfBoundaryHost = a.boundaryData().value().copyToHost();

            //
            for (const auto& patch : b.boundaryField())
            {
                // FIXME For skip processor boundaries
                if (patch.type() == "processor")
                {
                    continue;
                }
                auto bBoundarySpan = std::span(patch.cdata(), patch.size());
                for (auto b : bBoundarySpan)
                {
                    ofBoundaryData.push_back(convert(b));
                }
            }

            REQUIRE_THAT(
                nfBoundaryHost.view({0, ofBoundaryData.size()}),
                Catch::Matchers::RangeEquals(ofBoundaryData, comp)
            );
        }
    }
}

/* function to create a volScalarField filled with random values for test purposes */
auto randomScalarField(const Foam::fvMesh& mesh, Foam::word name)
{
    Foam::volScalarField t(
        Foam::IOobject(
            name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh
    );

    randomizeField(t);
    return t;
}

/**@brief create a dimensionedScalarField with random entries*/
template<typename FieldType>
FieldType randDimField(const Foam::fvMesh& mesh, Foam::dimensionSet dimensionSet, Foam::word name)
{
    auto field = FieldType(
        Foam::IOobject(
            name,
            mesh.time().timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::AUTO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar(name, dimensionSet, 0.0)
    );

    randomizeField(field);
    return field;
}


}
