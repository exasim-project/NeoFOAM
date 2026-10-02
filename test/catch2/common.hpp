// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <algorithm>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoFOAM/auxiliary/convert.hpp"
#include "NeoN/NeoN.hpp"
#include "executorGenerator.hpp"
#include "NeoFOAM/NeoFOAM.hpp"

#include "fvm.H"
#include "fvc.H"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

#define SECTION_IF(condition, ...) SECTION(__VA_ARGS__) if (condition)


/** @brief Approximate scalar comparison predicate for use with EqualsInternal/EqualsBoundary. */
struct ApproxScalar
{
    Foam::scalar margin;

    ApproxScalar(Foam::scalar margin)
        : margin(margin)
    {}

    bool operator()(double rhs, double lhs) const
    {
        return Catch::Approx(rhs).margin(margin) == lhs;
    }
};

/** @brief Approximate vector comparison predicate for use with EqualsInternal/EqualsBoundary. */
struct ApproxVector
{
    NeoN::Vec3 margin;

    ApproxVector(NeoN::Vec3 v)
        : margin(v)
    {}
    ApproxVector(NeoN::scalar v)
        : margin({v, v, v})
    {}

    bool operator()(NeoN::Vec3 rhs, Foam::vector lhs) const
    {
        NeoN::Vec3 diff(rhs[0] - lhs[0], rhs[1] - lhs[1], rhs[2] - lhs[2]);
        return Catch::Approx(0).margin(margin[0]) == diff[0]
            && Catch::Approx(0).margin(margin[1]) == diff[1]
            && Catch::Approx(0).margin(margin[2]) == diff[2];
    }

    bool operator()(NeoN::Vec3 rhs, NeoN::Vec3 lhs) const
    {
        NeoN::Vec3 diff(rhs[0] - lhs[0], rhs[1] - lhs[1], rhs[2] - lhs[2]);
        return Catch::Approx(0).margin(margin[0]) == diff[0]
            && Catch::Approx(0).margin(margin[1]) == diff[1]
            && Catch::Approx(0).margin(margin[2]) == diff[2];
    }

    bool operator()(NeoN::Vec3 rhs, Foam::scalar lhs) const
    {
        NeoN::Vec3 diff(rhs[0] - lhs, rhs[1] - lhs, rhs[2] - lhs);
        return Catch::Approx(0).margin(margin[0]) == diff[0]
            && Catch::Approx(0).margin(margin[1]) == diff[1]
            && Catch::Approx(0).margin(margin[2]) == diff[2];
    }
};

namespace NeoFOAM
{

/** @brief Fill every cell of an OpenFOAM field with random values and sync boundary conditions. */
void randomizeField(auto& field)
{
    using FieldType = std::decay_t<decltype(field)>;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(1.0, 2.0);

    if constexpr (std::is_same_v<FieldType, Foam::surfaceScalarField>)
    {
        // Surface fields: boundary patches hold the face values themselves;
        // correctBoundaryConditions() does not randomize them. Without this,
        // a Dirichlet-anchored equation built from a random surface gamma
        // would see zero coefficients on the Dirichlet patch and behave as
        // a singular pure-Neumann system.
        auto& intF = field.primitiveFieldRef();
        forAll(intF, facei)
        {
            intF[facei] = dis(gen);
        }
        forAll(field.boundaryField(), patchi)
        {
            auto& p = field.boundaryFieldRef()[patchi];
            forAll(p, i)
            {
                p[i] = dis(gen);
            }
        }
    }
    else
    {
        forAll(field, celli)
        {
            field[celli] = dis(gen);
        }
        field.correctBoundaryConditions();
    }
}

/** @brief Create and return a named OpenFOAM field of type @p FieldType with values supplied by @p
 * rand. */
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

/** @brief Create a volScalarField with uniform random values in [1, 2]. */
auto randomScalarField(const Foam::Time& runTime, const Foam::fvMesh& mesh, Foam::word name)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(1.0, 2.0);
    return createRandomField<Foam::volScalarField>(runTime, mesh, name, [&]() { return dis(gen); });
}

/** @brief Create a volVectorField with random component values in [1, 2]. */
auto randomVectorField(const Foam::Time& runTime, const Foam::fvMesh& mesh, Foam::word name)
{
    std::mt19937 gen(42);
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

/** @brief Create a surfaceScalarField with random values in [1, 2]. */
auto randomSurfaceScalarField(const Foam::Time& runTime, const Foam::fvMesh& mesh, Foam::word name)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(1.0, 2.0);
    return createRandomField<Foam::surfaceScalarField>(
        runTime,
        mesh,
        name,
        [&]() { return dis(gen); }
    );
}

/** @brief Create a volScalarField by reading @p name from disk and randomizing its values. */
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

/** @brief Create a dimensioned field of type @p FieldType initialized to zero then randomized. */
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

/**
 * @brief Catch2 matcher that checks the internal field of a NeoN field against an OpenFOAM field.
 *
 * Use as: REQUIRE_THAT(nfField, EqualsInternal(ofField, predicate)).
 */
template<typename OFField, typename Predicate>
struct EqualsInternalMatcher : Catch::Matchers::MatcherGenericBase
{
    EqualsInternalMatcher(OFField expected, Predicate pred)
        : expected_(std::move(expected))
        , pred_(pred)
    {}

    template<typename NFField>
        requires requires(const NFField& f) { f.internalVector(); }
    bool match(const NFField& actual) const
    {
        auto aHost = actual.internalVector().copyToHost();
        const auto bSpan = std::span(expected_.primitiveField().cdata(), expected_.size());
        if (static_cast<std::size_t>(aHost.size()) < bSpan.size()) return false;
        auto aSub = aHost.view({0, bSpan.size()});
        return std::equal(begin(aSub), end(aSub), bSpan.begin(), bSpan.end(), pred_);
    }

    template<typename NeoNType>
    bool match(const NeoN::Vector<NeoNType>& actual) const
    {
        auto aHost = actual.copyToHost();
        auto bSpan = std::span(expected_.cdata(), expected_.size());
        if (static_cast<std::size_t>(aHost.size()) < bSpan.size()) return false;
        auto aSub = aHost.view({0, bSpan.size()});
        return std::equal(begin(aSub), end(aSub), bSpan.begin(), bSpan.end(), pred_);
    }

    std::string describe() const override
    {
        if constexpr (requires { expected_.primitiveField(); })
        {
            auto bSpan = std::span(expected_.primitiveField().cdata(), expected_.size());
            return "equals OpenFOAM field: " + Catch::rangeToString(bSpan);
        }
        else
        {
            auto bSpan = std::span(expected_.cdata(), expected_.size());
            return "equals OpenFOAM field: " + Catch::rangeToString(bSpan);
        }
    }

private:

    OFField expected_;
    Predicate pred_;
};

/**
 * @brief Catch2 matcher that checks boundary data of a NeoN field against an OpenFOAM field.
 *
 * Use as: REQUIRE_THAT(field.boundaryData(), EqualsBoundary(ofField, predicate)).
 * Passing @c field.boundaryData() ensures failure messages show boundary values.
 * Non-processor patches are compared first, then processor patches (MPI ordering).
 * The @c NeoN::Vector overload is a no-op — raw vectors carry no boundary data.
 */
template<typename OFField, typename Predicate>
struct EqualsBoundaryMatcher : Catch::Matchers::MatcherGenericBase
{
    EqualsBoundaryMatcher(OFField expected, Predicate pred)
        : expected_(std::move(expected))
        , pred_(pred)
    {}

    template<typename T>
    bool match(const NeoN::Vector<T>&) const
    {
        return true;
    }

    template<typename T>
    bool match(const NeoN::BoundaryData<T>& actual) const
    {
        return matchBoundary(actual.value());
    }

    std::string describe() const override
    {
        std::string result = "equals OpenFOAM boundary field: ";
        for (const auto& patch : expected_.boundaryField())
        {
            auto bSpan = std::span(patch.cdata(), patch.size());
            result += Catch::rangeToString(bSpan) + " ";
        }
        return result;
    }

private:

    template<typename ValueType>
    bool matchBoundary(const NeoN::Vector<ValueType>& boundaryVec) const
    {
        std::vector<ValueType> ofBoundaryData;

        for (const auto& patch : expected_.boundaryField())
        {
            if (patch.type() == "processor") continue;
            auto bBoundarySpan = std::span(patch.cdata(), patch.size());
            for (auto val : bBoundarySpan)
                ofBoundaryData.push_back(convert(val));
        }
        for (const auto& patch : expected_.boundaryField())
        {
            if (patch.type() != "processor") continue;
            auto bBoundarySpan = std::span(patch.cdata(), patch.size());
            for (auto val : bBoundarySpan)
                ofBoundaryData.push_back(convert(val));
        }
        auto nfBoundaryHost = boundaryVec.copyToHost();
        auto bndSub = nfBoundaryHost.view({0, ofBoundaryData.size()});
        return std::equal(
            begin(bndSub),
            end(bndSub),
            ofBoundaryData.begin(),
            ofBoundaryData.end(),
            pred_
        );
    }

    OFField expected_;
    Predicate pred_;
};

/** @brief Factory for EqualsInternalMatcher. */
template<typename OFField, typename Predicate>
auto EqualsInternal(OFField expected, Predicate pred)
{
    return EqualsInternalMatcher<OFField, Predicate> {std::move(expected), pred};
}

/** @brief Factory for EqualsBoundaryMatcher. */
template<typename OFField, typename Predicate>
auto EqualsBoundary(OFField expected, Predicate pred)
{
    return EqualsBoundaryMatcher<OFField, Predicate> {std::move(expected), pred};
}

} // namespace NeoFOAM

using NeoFOAM::EqualsInternal;
using NeoFOAM::EqualsBoundary;

namespace Catch
{

/** @brief Disable Catch2's range StringMaker for NeoN::Vector to avoid ambiguous specializations.
 */
template<typename T>
struct is_range<NeoN::Vector<T>> : std::false_type
{
};

/** @brief Stringify a NeoN::Vector by copying to host first. */
template<typename T>
struct StringMaker<NeoN::Vector<T>>
{
    static std::string convert(const NeoN::Vector<T>& v)
    {
        auto host = v.copyToHost();
        return rangeToString(host.view());
    }
};

/** @brief Stringify any NeoN field that exposes internalVector() (VolumeField, SurfaceField). */
template<typename Field>
    requires requires(const Field& f) { f.internalVector(); }
struct StringMaker<Field>
{
    static std::string convert(const Field& v)
    {
        auto host = v.internalVector().copyToHost();
        return rangeToString(host.view());
    }
};

/** @brief Stringify NeoN::BoundaryData by printing the computed boundary values. */
template<typename NeoNValueType>
struct StringMaker<NeoN::BoundaryData<NeoNValueType>>
{
    static std::string convert(const NeoN::BoundaryData<NeoNValueType>& v)
    {
        auto host = v.value().copyToHost();
        return rangeToString(host.view());
    }
};

} // namespace Catch
