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

// Default coefficients matching OpenFOAM's wallFunctionCoefficients defaults
// (src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
//  wallFunction/wallFunctionCoefficients/wallFunctionCoefficients.C:61-63).
inline constexpr scalar OMEGA_WF_DEFAULT_BETA1 = 0.075;
inline constexpr scalar OMEGA_WF_DEFAULT_CMU = 0.09;
inline constexpr scalar OMEGA_WF_DEFAULT_KAPPA = 0.41;

inline constexpr scalar OMEGA_WF_OMEGA_MAX = scalar(1e12);

/**
 * Applies the omegaWallFunction kernel face-by-face.
 *
 *   ω_vis = 6ν / (β₁·y²)           (viscous-sublayer estimate)
 *   ω_log = √k / (C_µ^0.25·κ·y)   (log-layer estimate)
 *   ω_w   = √(ω_vis² + ω_log²)     (BINOMIAL n=2 blend)
 *
 * Known limitations:
 *  - Single-patch-corner only: each wall cell is assumed to see at most one
 *    wall-function face.
 *  - G (production) accumulation is not applied here; kOmegaSST computes G
 *    internally from nut·GbyNu0.
 */
inline void setOmegaWallFunction(
    Field<scalar>& omega,
    const fvcc::VolumeField<scalar>& k,
    const fvcc::VolumeField<scalar>& nu,
    const fvcc::VolumeField<scalar>& nearWallDist,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    scalar beta1,
    scalar cmu,
    scalar kappa
)
{
    const scalar cmu25 = Kokkos::pow(cmu, scalar(0.25));
    // Function-local copy: a namespace-scope constexpr cannot be referenced inside a
    // __host__ __device__ lambda (it is "undefined in device code"); bind it here first.
    const scalar omegaMax = OMEGA_WF_OMEGA_MAX;

    auto kInternal = k.internalVector().view();
    const auto nuBoundary = nu.boundaryData().value().view();
    const auto nearWallBoundary = nearWallDist.boundaryData().value().view();

    auto [refGrad, value, valueFraction, refValue, faceOwners] = views(
        omega.boundaryData().refGrad(),
        omega.boundaryData().value(),
        omega.boundaryData().valueFraction(),
        omega.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        omega.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = faceOwners[i];
            const scalar y = nearWallBoundary[i];
            const scalar nuw = nuBoundary[i];
            const scalar kw = Kokkos::max(kInternal[owner], scalar(0));

            // Viscous sublayer: 6ν / (β₁·y²)   (omega.C:218-224)
            const scalar wVis = 6.0 * nuw / (beta1 * y * y);

            const scalar wLog = Kokkos::sqrt(kw) / (cmu25 * kappa * y);
            const scalar wOmega = Kokkos::min(Kokkos::sqrt(wVis * wVis + wLog * wLog), omegaMax);

            // Set the wall face value only. Upstream also writes
            // omega_internal[wall_cell] via manipulateMatrix(setValues) — a
            // matrix-row pin we have no equivalent for. Without that pin a
            // direct stomp on omega.internalVector() only persists until the
            // next omega solve overwrites it; meanwhile correctNutInternal
            // reads the stomped value and collapses nut.internal at the wall
            // cell. Leaving the internal value alone keeps the fixedValue
            // semantics correct via the boundary face.
            value[i] = wOmega;
            refValue[i] = wOmega;
            valueFraction[i] = 1.0;
            refGrad[i] = 0.0;
        },
        "setOmegaWallFunction"
    );
}

} // namespace detail


// Mirrors Foam::omegaWallFunctionFvPatchScalarField
// (src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/
//  omegaWallFunctions/omegaWallFunction/omegaWallFunctionFvPatchScalarField.{H,C}).
//
// Upstream is a fixedValueFvPatchField<scalar> whose updateCoeffs() blends a
// viscous-sublayer and a log-layer estimate of omega and also overwrites the
// adjacent wall cell's omega in the internal field. We replicate that here
// inside correctBoundaryCondition(ctx).
//
// Context fields required:
//   - "k"            VolumeField<scalar>  — to evaluate √k at the wall cell
//   - "nu"           VolumeField<scalar>  — laminar viscosity, boundary face values
//   - "nearWallDist" VolumeField<scalar>  — y for the wall cell, boundary face values
//
// Mirroring NutUSpaldingWallFunction (nutWallFunction.hpp:142-153), the no-arg
// correctBoundaryCondition() overload is a no-op: this BC must be driven from
// kOmegaSST::correct() (or equivalent) with a BoundaryContext that supplies
// the three fields above.
class OmegaWallFunction : public VolumeBoundaryFactory<scalar>::template Register<OmegaWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<OmegaWallFunction>;

public:

    OmegaWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true})
        , mesh_(mesh)
        , beta1_(
              dict.contains("beta1") ? dict.get<scalar>("beta1") : detail::OMEGA_WF_DEFAULT_BETA1
          )
        , cmu_(dict.contains("Cmu") ? dict.get<scalar>("Cmu") : detail::OMEGA_WF_DEFAULT_CMU)
        , kappa_(
              dict.contains("kappa") ? dict.get<scalar>("kappa") : detail::OMEGA_WF_DEFAULT_KAPPA
          )
    {}

    void correctBoundaryCondition(Field<scalar>& /*domainVector*/) final {}

    void
    correctBoundaryCondition(Field<scalar>& domainVector, const fvcc::BoundaryContext& ctx) final
    {
        detail::setOmegaWallFunction(
            domainVector,
            ctx.scalarFieldPtr("k"),
            ctx.scalarFieldPtr("nu"),
            ctx.scalarFieldPtr("nearWallDist"),
            mesh_,
            this->range(),
            beta1_,
            cmu_,
            kappa_
        );
    }

    static std::string name() { return "omegaWallFunction"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Specific dissipation rate wall function. Blends viscous "
               "(6ν/(β₁y²)) and log-layer (√k/(C_µ^0.25·κ·y)) omega "
               "contributions with the BINOMIAL n=2 blender (upstream default) "
               "and sets the wall face value. Does NOT overwrite the wall cell "
               "internal value (upstream's setValues equivalent is absent). "
               "Single-patch-corner only; G is fed back through kOmegaSST.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<OmegaWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    scalar beta1_;
    scalar cmu_;
    scalar kappa_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
