// SPDX-FileCopyrightText: 2026 NeoFOAM authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

/**
 * @brief Mixed outflow/inflow BC: zeroGradient on faces with outward flux, fixedValue
 * (inletValue) on faces with inward flux. Mirrors OpenFOAM's inletOutletFvPatchField.
 *
 * Flow direction is determined from the boundary face velocity:
 *  - For Vec3 fields: reads lagged boundary velocity from domainVector.boundaryData().value().
 *  - For scalar fields: requires "U" in BoundaryContext; defaults to zeroGradient without it.
 */
template<typename ValueType>
class InletOutlet :
    public VolumeBoundaryFactory<ValueType>::template Register<InletOutlet<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<InletOutlet<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    InletOutlet(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = false})
        , mesh_(mesh)
        , inletValue_(dict.get<ValueType>("inletValue"))
    {}

    // One-time init: start with outflow assumption.
    void set(Field<ValueType>& domainVector) final { setZeroGradient(domainVector); }

    // Without context: Vec3 fields use lagged boundary velocity; scalar falls back to zeroGradient.
    void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        if constexpr (std::is_same_v<ValueType, Vec3>)
        {
            applyInletOutlet(domainVector, domainVector.boundaryData().value().view());
        }
        else
        {
            setZeroGradient(domainVector);
        }
    }

    void
    correctBoundaryCondition(Field<ValueType>& domainVector, const fvcc::BoundaryContext& ctx) final
    {
        if (ctx.hasVector("U"))
        {
            applyInletOutlet(domainVector, ctx.vectorFieldPtr("U").boundaryData().value().view());
        }
        else
        {
            correctBoundaryCondition(domainVector);
        }
    }

    static std::string name() { return "inletOutlet"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "zeroGradient on outflow faces, fixedValue (inletValue) on inflow faces.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<InletOutlet>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    ValueType inletValue_;

    void setZeroGradient(Field<ValueType>& domainVector)
    {
        auto [refGrad, value, valueFraction, refValue, faceOwners, deltaCoeffs] = views(
            domainVector.boundaryData().refGrad(),
            domainVector.boundaryData().value(),
            domainVector.boundaryData().valueFraction(),
            domainVector.boundaryData().refValue(),
            mesh_.boundaryMesh().faceOwners(),
            mesh_.boundaryMesh().deltaCoeffs()
        );
        const auto iVector = domainVector.internalVector().view();

        NeoN::parallelFor(
            domainVector.exec(),
            this->range(),
            NEON_LAMBDA(const localIdx i) {
                refGrad[i] = zero<ValueType>();
                value[i] = iVector[faceOwners[i]];
                valueFraction[i] = scalar(0);
                refValue[i] = zero<ValueType>();
            },
            "InletOutletZeroGradient"
        );
    }

    void applyInletOutlet(Field<ValueType>& domainVector, auto uBoundaryView)
    {
        auto [refGrad, value, valueFraction, refValue, faceOwners, deltaCoeffs] = views(
            domainVector.boundaryData().refGrad(),
            domainVector.boundaryData().value(),
            domainVector.boundaryData().valueFraction(),
            domainVector.boundaryData().refValue(),
            mesh_.boundaryMesh().faceOwners(),
            mesh_.boundaryMesh().deltaCoeffs()
        );
        const auto iVector = domainVector.internalVector().view();
        const auto faceNormals = mesh_.boundaryMesh().faceNormals().view();
        const ValueType inletVal = inletValue_;

        NeoN::parallelFor(
            domainVector.exec(),
            this->range(),
            NEON_LAMBDA(const localIdx i) {
                const Vec3 n = faceNormals[i];
                const Vec3 u = uBoundaryView[i];
                const scalar outflux = n[0] * u[0] + n[1] * u[1] + n[2] * u[2];
                if (outflux >= scalar(0))
                {
                    refGrad[i] = zero<ValueType>();
                    value[i] = iVector[faceOwners[i]];
                    valueFraction[i] = scalar(0);
                    refValue[i] = zero<ValueType>();
                }
                else
                {
                    refValue[i] = inletVal;
                    value[i] = inletVal;
                    valueFraction[i] = scalar(1);
                    refGrad[i] = zero<ValueType>();
                }
            },
            "InletOutletBC"
        );
    }
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
