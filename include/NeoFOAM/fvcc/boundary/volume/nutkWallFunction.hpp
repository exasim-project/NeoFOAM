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

/// Default coefficients matching OpenFOAM's wallFunctionCoefficients (Cmu=0.09, κ=0.41, E=9.8).
inline constexpr scalar NUTK_WF_DEFAULT_CMU = 0.09;
inline constexpr scalar NUTK_WF_DEFAULT_KAPPA = 0.41;
inline constexpr scalar NUTK_WF_DEFAULT_E = 9.8;

/// Intersection of the viscous and log-law layers, y⁺_lam = log(E·y⁺_lam)/κ
/// (Foam::nutWallFunctionFvPatchScalarField::yPlusLam fixed-point iteration).
inline scalar nutWallFunctionYPlusLam(scalar kappa, scalar E)
{
    scalar ypl = 11.0;
    for (int i = 0; i < 10; ++i)
    {
        ypl = Kokkos::log(Kokkos::max(E * ypl, scalar(1))) / kappa;
    }
    return ypl;
}

/**
 * @brief Apply the nutkWallFunction kernel face-by-face.
 *
 * Mirrors Foam::nutkWallFunctionFvPatchScalarField::calcNut() with the
 * upstream v2406 defaults (blender STEPWISE):
 *   - y⁺    = C_µ^0.25·y·√k/ν
 *   - ν_t,w = y⁺ > y⁺_lam ? ν·(y⁺·κ/log(E·y⁺) − 1) : 0
 *
 * k is read from the BoundaryContext; the boundary face value is written,
 * no internal-cell stomp.
 */
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
    const scalar yPlusLam = nutWallFunctionYPlusLam(kappa, E);

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
            const scalar nutw = yPlus > yPlusLam
                                  ? nuw * (yPlus * kappa / Kokkos::log(E * yPlus) - scalar(1))
                                  : scalar(0);

            value[i] = nutw;
            refValue[i] = nutw;
            valueFraction[i] = 1.0;
            refGradient[i] = 0.0;
        },
        "setNutkWallFunction"
    );
}

} // namespace detail


/**
 * @brief k-based turbulent viscosity wall function BC.
 *
 * Mirrors Foam::nutkWallFunctionFvPatchScalarField: derives y⁺ from k rather
 * than |∂U/∂n|, making it the companion to kqRWallFunction for kEpsilon flows.
 *
 * Required BoundaryContext fields:
 *   - @c "k"            VolumeField<scalar> — turbulent kinetic energy
 *   - @c "nu"           VolumeField<scalar> — laminar viscosity, boundary values
 *   - @c "nearWallDist" VolumeField<scalar> — cell-to-wall distance y, boundary values
 *
 * Single-patch-corner only.
 */
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
               "k (not from |∂U/∂n| like Spalding). Upstream v2406 default "
               "(STEPWISE): nut = nu*(y+*kappa/log(E*y+) - 1) above y+_lam, "
               "0 below.";
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
