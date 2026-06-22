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

inline constexpr scalar OMEGA_WF_DEFAULT_BETA1 = 0.075;
inline constexpr scalar OMEGA_WF_DEFAULT_CMU = 0.09;
inline constexpr scalar OMEGA_WF_DEFAULT_KAPPA = 0.41;

/**
 * Applies the omegaWallFunction kernel face-by-face.
 *
 *   ω_vis = 6ν / (β₁·y²)           (viscous-sublayer estimate)
 *   ω_log = √k / (C_µ^0.25·κ·y)   (log-layer estimate)
 *   ω_w   = √(ω_vis² + ω_log²)     (BINOMIAL n=2 blend)
 *
 * Known limitations:
 *  - Single-patch-corner only: each wall cell is assumed to see at most one
 *    wall-function face. Meshes with re-entrant corners shared by two wall
 *    patches will diverge from multi-patch corner averaging.
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
    scalar Cmu,
    scalar kappa
)
{
    const scalar Cmu25 = Kokkos::pow(Cmu, scalar(0.25));

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

            const scalar wVis = 6.0 * nuw / (beta1 * y * y);
            const scalar wLog = Kokkos::sqrt(kw) / (Cmu25 * kappa * y);
            const scalar wOmega = Kokkos::sqrt(wVis * wVis + wLog * wLog);

            value[i] = wOmega;
            refValue[i] = wOmega;
            valueFraction[i] = 1.0;
            refGrad[i] = 0.0;
        },
        "setOmegaWallFunction"
    );
}

} // namespace detail


/**
 * Omega wall function boundary condition.
 *
 * Requires BoundaryContext fields: "k", "nu", "nearWallDist".
 * The no-arg correctBoundaryCondition() overload is a no-op.
 *
 * Known limitations:
 *  - Single-patch-corner only: each wall cell is assumed to see at most one
 *    wall-function face.
 *  - G production-term feedback is not applied here; it is handled by
 *    kOmegaSST internally.
 */
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
        , Cmu_(dict.contains("Cmu") ? dict.get<scalar>("Cmu") : detail::OMEGA_WF_DEFAULT_CMU)
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
            Cmu_,
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
    scalar Cmu_;
    scalar kappa_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
