// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <vector>


#include "common.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

extern Foam::Time* timePtr;   // A single time object
extern Foam::fvMesh* meshPtr; // A single mesh object

namespace NeoFOAM
{

template<typename OFMesh, typename NFMesh, typename Accessor>
struct EqualsRangeMatcher : Catch::Matchers::MatcherGenericBase
{
    EqualsRangeMatcher(const OFMesh& range, const NFMesh& nfMesh, Accessor accessor)
        : ofMesh {range}
        , nfMesh {nfMesh}
        , accessor {accessor}
    {}

    template<typename OtherRange>
    bool match(OtherRange const& other) const
    {
        for (auto i = 0; i < ofMesh.size(); i++)
        {
            auto a = other.copyToHost();
            const auto& b = accessor(ofMesh[i]);
            auto start = nfMesh.offset()[i];
            auto end = nfMesh.offset()[i + 1];
            auto aV = a.view({start, end});

            for (auto j = 0; j < aV.size(); j++)
            {
                if (aV[j] != convert(b[j])) return false;
            }
        }
        return true;
    }

    std::string describe() const override { return "Equals: "; }

private:

    OFMesh const& ofMesh;
    NFMesh const& nfMesh;
    Accessor accessor;
};

template<typename OFMesh, typename NFMesh, typename Accessor>
auto allPatchesMatch(const OFMesh& ofMesh, const NFMesh& nfMesh, Accessor accessor)
    -> EqualsRangeMatcher<OFMesh, NFMesh, Accessor>
{
    return EqualsRangeMatcher<OFMesh, NFMesh, Accessor> {ofMesh, nfMesh, accessor};
}

TEST_CASE("UnstructuredMesh")
{
    Foam::Time& runTime = *timePtr;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto meshPtr = createMesh(exec, runTime);
    const Foam::fvMesh& ofMesh = *meshPtr;
    const NeoN::UnstructuredMesh& nfMesh = meshPtr->nfMesh();

    SECTION("Internal mesh data members on " + execName)
    {

        REQUIRE(nfMesh.nCells() == ofMesh.nCells());

        REQUIRE(nfMesh.nInternalFaces() == ofMesh.nInternalFaces());

        // NOTE for some reason using our operator== inside the REQUIRE macro fails
        auto samePoints = nfMesh.points() == ofMesh.points();
        REQUIRE(samePoints);

        auto sameCellVolumes = nfMesh.cellVolumes() == ofMesh.cellVolumes();
        REQUIRE(sameCellVolumes);

        auto sameFaceCentres = nfMesh.faceCentres() == ofMesh.faceCentres();
        REQUIRE(sameFaceCentres);

        auto sameFaceAreas = nfMesh.faceAreas() == ofMesh.faceAreas();
        REQUIRE(sameFaceAreas);

        auto magSf = Foam::mag(ofMesh.faceAreas());
        auto sameMagSf = nfMesh.magFaceAreas() == magSf();
        REQUIRE(sameMagSf);

        // TODO NeoN::Vector to OF List comparison currently not supported
        // auto sameOwner = nfMesh.faceOwner() == ofMesh.faceOwner();
        // REQUIRE(sameOwner);
        // SECTION("faceNeighbour") { REQUIRE(nfMesh.faceNeighbour() == ofMesh.faceNeighbour()); }
    }

    SECTION("BoundaryMesh on " + execName)
    {
        const Foam::fvBoundaryMesh& ofBMesh = ofMesh.boundary();
        const NeoN::BoundaryMesh& nfBMesh = nfMesh.boundaryMesh();
        const auto& offset = nfBMesh.offset();

        SECTION("offset")
        {
            forAll(ofBMesh, patchi)
            {
                REQUIRE(ofBMesh[patchi].size() == nfBMesh.faceCells(patchi).size());
            }
        }

        REQUIRE_THAT(
            nfBMesh.faceCells(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.faceCells(); })
        );

        REQUIRE_THAT(
            nfBMesh.cf(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.Cf(); })
        );

        REQUIRE_THAT(
            nfBMesh.cn(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.Cn()(); })
        );

        REQUIRE_THAT(
            nfBMesh.sf(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.Sf(); })
        );

        REQUIRE_THAT(
            nfBMesh.magSf(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.magSf(); })
        );

        REQUIRE_THAT(
            nfBMesh.nf(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.nf()(); })
        );

        REQUIRE_THAT(
            nfBMesh.delta(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.delta()(); })
        );

        REQUIRE_THAT(
            nfBMesh.weights(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.weights(); })
        );

        REQUIRE_THAT(
            nfBMesh.deltaCoeffs(),
            allPatchesMatch(ofBMesh, nfBMesh, [](const auto& p) { return p.deltaCoeffs(); })
        );
    }
}


TEST_CASE("fvccGeometryScheme")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    std::unique_ptr<NeoFOAM::MeshAdapter> meshPtr = NeoFOAM::createMesh(exec, *timePtr);
    NeoFOAM::MeshAdapter& mesh = *meshPtr;
    const NeoN::UnstructuredMesh& nfMesh = mesh.nfMesh();

    SECTION("BasicFvccGeometryScheme" + execName)
    {
        // update on construction
        auto scheme =
            fvcc::GeometryScheme(exec, nfMesh, std::make_unique<fvcc::BasicGeometryScheme>(nfMesh));
        scheme.update(); // make sure it uptodate
        auto foamWeights = mesh.weights();

        auto weightsHost = scheme.weights().internalVector().copyToHost();
        std::span<Foam::scalar> sFoamWeights(
            foamWeights.primitiveFieldRef().data(),
            foamWeights.size()
        );
        REQUIRE_THAT(
            weightsHost.view({0, foamWeights.size()}),
            Catch::Matchers::RangeEquals(sFoamWeights, ApproxScalar(1e-16))
        );
    }

    SECTION("DefaultBasicFvccGeometryScheme" + execName)
    {
        // update on construction
        fvcc::GeometryScheme scheme(nfMesh);
        scheme.update(); // make sure it uptodate
        auto foamWeights = mesh.weights();

        auto weightsHost = scheme.weights().internalVector().copyToHost();
        std::span<Foam::scalar> sFoamWeights(
            foamWeights.primitiveFieldRef().data(),
            foamWeights.size()
        );
        REQUIRE_THAT(
            weightsHost.view({0, foamWeights.size()}),
            Catch::Matchers::RangeEquals(sFoamWeights, ApproxScalar(1e-16))
        );
    }
}

}
