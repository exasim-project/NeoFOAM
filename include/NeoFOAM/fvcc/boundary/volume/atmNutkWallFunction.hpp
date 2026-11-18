// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

#include "NeoFOAM/fvcc/boundary/volume/nutkWallFunction.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace detail
{

/**
 * @brief Apply the atmNutkWallFunction kernel face-by-face.
 *
 * Mirrors Foam::atmNutkWallFunctionFvPatchScalarField::calcNut() (HW:Eq. 5), the rough-wall
 * log law with aerodynamic roughness length z0:
 *   - u*   = C_µ^0.25·√k
 *   - y⁺   = u*·y/ν
 *   - E'   = (y + z0)/z0
 *   - ν_t,w = ν·(y⁺·κ/log(max(E', 1+1e-4)) − 1), clipped at zero when boundNut is set
 *
 * Unlike nutkWallFunction there is no viscous-sublayer switch. k is clamped at zero before the
 * square root (OpenFOAM would take √ of a negative k and produce NaN).
 */
inline void setAtmNutkWallFunction(
    Field<scalar>& domainVector,
    const fvcc::VolumeField<scalar>& k,
    const fvcc::VolumeField<scalar>& nu,
    const fvcc::VolumeField<scalar>& nearWallDist,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    const Vector<scalar>& z0,
    scalar Cmu,
    scalar kappa,
    bool boundNut
)
{
    const scalar Cmu25 = Kokkos::pow(Cmu, scalar(0.25));

    const auto kInternal = k.internalVector().view();
    const auto nuBoundary = nu.boundaryData().value().view();
    const auto nearWallBoundary = nearWallDist.boundaryData().value().view();
    const auto z0V = z0.view();
    const localIdx start = range.first;

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
            const scalar z0w = z0V[i - start];

            const scalar uStar = Cmu25 * Kokkos::sqrt(kw);
            const scalar yPlus = uStar * y / nuw;
            const scalar Edash = (y + z0w) / z0w;

            scalar nutw = nuw
                        * (yPlus * kappa / Kokkos::log(Kokkos::max(Edash, scalar(1) + scalar(1e-4)))
                           - scalar(1));
            if (boundNut)
            {
                nutw = Kokkos::max(nutw, scalar(0));
            }

            value[i] = nutw;
            refValue[i] = nutw;
            valueFraction[i] = 1.0;
            refGradient[i] = 0.0;
        },
        "setAtmNutkWallFunction"
    );
}

} // namespace detail


/**
 * @brief Rough-wall k-based turbulent viscosity wall function for atmospheric flows.
 *
 * Mirrors Foam::atmNutkWallFunctionFvPatchScalarField. The per-face roughness length is read
 * from @c z0Values, which the reader evaluates from OpenFOAM's z0 PatchFunction1.
 *
 * Required BoundaryContext fields:
 *   - @c "k"            VolumeField<scalar> — turbulent kinetic energy
 *   - @c "nu"           VolumeField<scalar> — laminar viscosity, boundary values
 *   - @c "nearWallDist" VolumeField<scalar> — cell-to-wall distance y, boundary values
 */
class AtmNutkWallFunction :
    public VolumeBoundaryFactory<scalar>::template Register<AtmNutkWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<AtmNutkWallFunction>;

public:

    AtmNutkWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true})
        , mesh_(mesh)
        , z0_(mesh.exec(), dict.get<std::vector<scalar>>("z0Values"))
        , Cmu_(dict.contains("Cmu") ? dict.get<scalar>("Cmu") : detail::NUTK_WF_DEFAULT_CMU)
        , kappa_(dict.contains("kappa") ? dict.get<scalar>("kappa") : detail::NUTK_WF_DEFAULT_KAPPA)
        , boundNut_(dict.contains("boundNut") ? dict.get<bool>("boundNut") : false)
    {
        if (toLocalIdx(z0_.size()) != this->patchSize())
        {
            NF_THROW(
                "'z0Values' holds " + std::to_string(z0_.size()) + " values but patch "
                + std::to_string(patchID) + " has " + std::to_string(this->patchSize()) + " faces"
            );
        }
    }

    void correctBoundaryCondition(Field<scalar>& /*domainVector*/) final {}

    void
    correctBoundaryCondition(Field<scalar>& domainVector, const fvcc::BoundaryContext& ctx) final
    {
        detail::setAtmNutkWallFunction(
            domainVector,
            ctx.scalarFieldPtr("k"),
            ctx.scalarFieldPtr("nu"),
            ctx.scalarFieldPtr("nearWallDist"),
            mesh_,
            this->range(),
            z0_,
            Cmu_,
            kappa_,
            boundNut_
        );
    }

    static std::string name() { return "atmNutkWallFunction"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Rough-wall k-based turbulent viscosity wall function for atmospheric "
               "boundary layers: nu*(y+*kappa/log((y+z0)/z0) - 1).";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<AtmNutkWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    Vector<scalar> z0_;
    scalar Cmu_;
    scalar kappa_;
    bool boundNut_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
