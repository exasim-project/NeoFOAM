// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace detail
{

// Defaults match OpenFOAM's wallFunctionCoefficients
//  (Cmu = 0.09, kappa = 0.41, E = 9.8).
inline constexpr scalar NUTK_WF_DEFAULT_CMU = 0.09;
inline constexpr scalar NUTK_WF_DEFAULT_KAPPA = 0.41;
inline constexpr scalar NUTK_WF_DEFAULT_E = 9.8;

// Applies the nutkWallFunction kernel face-by-face.
//
// Mirrors Foam::nutkWallFunctionFvPatchScalarField::calcNut() for the
// BINOMIAL n=2 blender (the upstream default):
//
//   y+      = C_µ^0.25 · y · √k / ν              (per face)
//   ν_t,log = ν · y+ · κ / log(max(E·y+, 1+1e-4))
//   ν_t,vis = ν                                   (viscous-sublayer estimate)
//   ν_t,w   = √(ν_t,vis² + ν_t,log²)              (BINOMIAL n=2 blend)
//
// Reads k from the BoundaryContext (used by kEpsilon to drive ν_t at walls
// without depending on |∂U/∂n| like the Spalding form). The boundary face
// value is written; no internal stomp.
inline void setNutkWallFunction(
    Field<scalar>& domainVector,
    const fvcc::VolumeField<scalar>& k,
    const fvcc::VolumeField<scalar>& nu,
    const fvcc::VolumeField<scalar>& nearWallDist,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    scalar Cmu,
    scalar kappa,
    scalar E
)
{
    const scalar Cmu25 = Kokkos::pow(Cmu, scalar(0.25));

    const auto kInternal = k.internalVector().view();
    const auto nuBoundary = nu.boundaryData().value().view();
    const auto nearWallBoundary = nearWallDist.boundaryData().value().view();

    auto [refGradient, value, valueFraction, refValue, faceCells] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = faceCells[i];
            const scalar y = nearWallBoundary[i];
            const scalar nuw = nuBoundary[i];
            const scalar kw = Kokkos::max(kInternal[owner], scalar(0));

            const scalar yPlus = Cmu25 * y * Kokkos::sqrt(kw) / nuw;
            const scalar nutLog =
                nuw * yPlus * kappa / Kokkos::log(Kokkos::max(E * yPlus, scalar(1) + scalar(1e-4)));
            const scalar nutw = Kokkos::sqrt(nuw * nuw + nutLog * nutLog);

            value[i] = nutw;
            refValue[i] = nutw;
            valueFraction[i] = 1.0;
            refGradient[i] = 0.0;
        },
        "setNutkWallFunction"
    );
}

} // namespace detail


// Mirrors Foam::nutkWallFunctionFvPatchScalarField
//   (src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
//    nutWallFunctions/nutkWallFunction/nutkWallFunctionFvPatchScalarField.{H,C}).
//
// The k-based ν_t wall function used by kEpsilon (kqRWallFunction's companion
// on the nut side). Single-patch-corner only.
class NutkWallFunction : public VolumeBoundaryFactory<scalar>::template Register<NutkWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<NutkWallFunction>;

public:

    NutkWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true})
        , mesh_(mesh)
        , Cmu_(dict.contains("Cmu") ? dict.get<scalar>("Cmu") : detail::NUTK_WF_DEFAULT_CMU)
        , kappa_(dict.contains("kappa") ? dict.get<scalar>("kappa") : detail::NUTK_WF_DEFAULT_KAPPA)
        , E_(dict.contains("E") ? dict.get<scalar>("E") : detail::NUTK_WF_DEFAULT_E)
    {}

    void correctBoundaryCondition(Field<scalar>& /*domainVector*/) final {}

    void
    correctBoundaryCondition(Field<scalar>& domainVector, const fvcc::BoundaryContext& ctx) final
    {
        detail::setNutkWallFunction(
            domainVector,
            ctx.scalarFieldPtr("k"),
            ctx.scalarFieldPtr("nu"),
            ctx.scalarFieldPtr("nearWallDist"),
            mesh_,
            this->range(),
            Cmu_,
            kappa_,
            E_
        );
    }

    static std::string name() { return "nutkWallFunction"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "k-based turbulent viscosity wall function. y+ derived from "
               "k (not from |∂U/∂n| like Spalding). BINOMIAL n=2 blend of "
               "viscous (nu) and log-layer (nu·y+·κ/log(E·y+)) contributions.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<NutkWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    scalar Cmu_;
    scalar kappa_;
    scalar E_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
