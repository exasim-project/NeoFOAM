// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/auxiliary/comparison.hpp"
#include "NeoFOAM/auxiliary/typeConversion.hpp"
#include "NeoFOAM/auxiliary/convert.hpp"

namespace NeoFOAM
{

template<typename NT, typename OT>
bool operator==(const NeoN::Vector<NT>& nf, const Foam::Field<OT>& of)
{
    auto nfHost = nf.copyToHost();
    auto nfView = nfHost.view();
    for (int i = 0; i < nfView.size(); i++)
    {
        if (nfView[i] != convert(of[i]))
        {
            return false;
        }
    }
    return true;
};

#define FIELD_EQUALITY_OPERATOR(NF_TYPE, OF_TYPE)                                                  \
    template bool                                                                                  \
    operator== <NF_TYPE, OF_TYPE>(const NeoN::Vector<NF_TYPE>&, const Foam::Field<OF_TYPE>&)

FIELD_EQUALITY_OPERATOR(NeoN::label, Foam::label);
FIELD_EQUALITY_OPERATOR(NeoN::scalar, Foam::scalar);
FIELD_EQUALITY_OPERATOR(NeoN::Vec3, Foam::vector);

template<typename NT, typename OT>
bool operator==(
    fvcc::VolumeField<NT>& nf,
    const Foam::GeometricField<OT, Foam::fvPatchField, Foam::volMesh>& of
)
{
    if (nf.internalVector() != of.internalField())
    {
        return false;
    }

    /* compare boundaryVector */
    /* NeoFOAM boundaries are stored in contiguous memory
     * whereas OpenFOAM boundaries are
     * stored in a vector of patches */
    auto nfBoundaryHost = nf.boundaryData().value().copyToHost();
    auto nfBoundaryView = nfBoundaryHost.view();
    NeoN::label pFacei = 0;
    for (const auto& patch : of.boundaryField())
    {
        int patchSize = patch.size();
        for (const auto& patchValue : patch)
        {
            if (nfBoundaryView[pFacei] != convert(patchValue))
            {
                return false;
            }
            pFacei++;
        }
    }

    return true;
}

#define VOLGEOFIELD_EQUALITY_OPERATOR(NF_TYPE, OF_TYPE)                                            \
    template bool operator== <NF_TYPE, OF_TYPE>(                                                   \
        fvcc::VolumeField<NF_TYPE> & nf,                                                           \
        const Foam::GeometricField<OF_TYPE, Foam::fvPatchField, Foam::volMesh>& of                 \
    )

VOLGEOFIELD_EQUALITY_OPERATOR(NeoN::scalar, Foam::scalar);
VOLGEOFIELD_EQUALITY_OPERATOR(NeoN::Vec3, Foam::vector);

} // namespace Foam
