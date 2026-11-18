// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/traits.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

/**
 * @brief Apply the atmBoundaryLayerInlet kernel face-by-face.
 *
 * NeoN's inletOutlet with a per-face inlet value. Per boundary face it switches by the sign of
 * the face flux phi:
 *   inflow  (phi <  0): valueFraction = 1, refValue = inletValues[face]
 *   outflow (phi >= 0): valueFraction = 0 (zero gradient)
 *
 * Without a flux (hasFlux == false) the inflow/outflow state of the previous flux-aware update
 * is kept, as OpenFOAM only re-evaluates valueFraction in updateCoeffs(); the very first update
 * (initial == true) marks every face as inflow, which is the valueFraction = 1 the OpenFOAM
 * constructors start from.
 */
template<typename ValueType>
void setAtmBoundaryLayerInletValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    const Vector<ValueType>& inletValues,
    View<const scalar> boundaryFlux,
    bool hasFlux,
    bool initial
)
{
    const auto internalV = domainVector.internalVector().view();
    const auto inletV = inletValues.view();
    auto [refGradV, valueV, valueFractionV, refValueV, faceOwnersV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );
    const localIdx start = range.first;

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            scalar vf = valueFractionV[i];
            if (hasFlux)
            {
                vf = boundaryFlux[i] < 0.0 ? 1.0 : 0.0;
            }
            else if (initial)
            {
                vf = 1.0;
            }
            const ValueType inlet = inletV[i - start];
            valueFractionV[i] = vf;
            refValueV[i] = inlet;
            refGradV[i] = NeoN::zero<ValueType>();
            valueV[i] = inlet * vf + internalV[faceOwnersV[i]] * (1.0 - vf);
        },
        "setAtmBoundaryLayerInletValue"
    );
}

} // namespace detail


/**
 * @brief Atmospheric boundary layer inlet for U, k and epsilon.
 *
 * Mirrors Foam::atmBoundaryLayerInlet{Velocity,K,Epsilon}FvPatchField: an inletOutlet whose
 * inlet value is the Richards-Hoxey / Yang et al. ABL profile of the patch face centres. The
 * profile is not evaluated here: the reader takes OpenFOAM's refValue(), which the OpenFOAM
 * patch field computes from atmBoundaryLayer::U/k/epsilon, and hands it over per face as
 * @c inletValues. The profile is therefore fixed at the start time; time-varying Uref, Zref,
 * flowDir or zDir are not followed.
 *
 * The flux is looked up by name (key "phi", default "phi") from the BoundaryContext.
 */
template<typename ValueType>
class AtmBoundaryLayerInlet :
    public VolumeBoundaryFactory<ValueType>::template Register<AtmBoundaryLayerInlet<ValueType>>
{
    using Base = typename VolumeBoundaryFactory<ValueType>::template Register<
        AtmBoundaryLayerInlet<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    // fixesValue = true, unlike NeoN's inletOutlet: the Gauss gradient reconstructs the boundary
    // normal gradient as (value - owner)·δ only on fixesValue patches and takes refGrad (here 0)
    // on all others. With value = vf·inlet + (1 - vf)·owner the former is OpenFOAM's mixed
    // snGrad on inflow (vf = 1) and outflow (vf = 0) faces alike; the latter drops the inlet
    // gradient from grad(U) and with it from the explicit viscous stress. The flag's other use,
    // the transient ddt flux correction, is then zero on the patch where OpenFOAM keeps it.
    AtmBoundaryLayerInlet(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = true})
        , mesh_(mesh)
        , inletValues_(mesh.exec(), dict.get<std::vector<ValueType>>("inletValues"))
        , phiName_(dict.contains("phi") ? dict.get<std::string>("phi") : std::string("phi"))
    {
        if (toLocalIdx(inletValues_.size()) != this->patchSize())
        {
            NF_THROW(
                "'inletValues' holds " + std::to_string(inletValues_.size()) + " values but patch "
                + std::to_string(patchID) + " has " + std::to_string(this->patchSize()) + " faces"
            );
        }
    }

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setAtmBoundaryLayerInletValue(
            domainVector,
            mesh_,
            this->range(),
            inletValues_,
            View<const scalar> {},
            false,
            !initialised_
        );
        initialised_ = true;
    }

    virtual void
    correctBoundaryCondition(Field<ValueType>& domainVector, const BoundaryContext& ctx) final
    {
        if (!ctx.hasSurfaceScalar(phiName_))
        {
            correctBoundaryCondition(domainVector);
            return;
        }
        const auto& phi = ctx.surfaceScalarField(phiName_);
        detail::setAtmBoundaryLayerInletValue(
            domainVector,
            mesh_,
            this->range(),
            inletValues_,
            phi.boundaryData().value().view(),
            true,
            false
        );
        initialised_ = true;
    }

    static std::string name() { return "atmBoundaryLayerInlet"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Atmospheric boundary layer inlet: inletOutlet whose per-face inlet value is the "
               "ABL profile (atmBoundaryLayerInletVelocity/K/Epsilon) evaluated by OpenFOAM.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<AtmBoundaryLayerInlet>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    Vector<ValueType> inletValues_;
    std::string phiName_;
    bool initialised_ = false;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
