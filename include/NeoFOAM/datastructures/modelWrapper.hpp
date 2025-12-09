// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

template<
    typename ExecuteFunctionType,
    // typename NNInField,
    // typename OFInField,
    typename NNOutField>
class ModelAdapter
{

private:

    std::vector<std::pair<fvcc::VolumeField<NeoN::Vec3>&, Foam::volVectorField&>> syncFields_;
    ExecuteFunctionType execute_;
    NNOutField& outField_;
    Foam::volScalarField& tmpRet_;

public:

    ModelAdapter(
        std::vector<std::pair<fvcc::VolumeField<NeoN::Vec3>&, Foam::volVectorField&>> syncFields,
        ExecuteFunctionType executeFunction,
        NNOutField& outField,
        Foam::volScalarField& buffer
    )
        : syncFields_(syncFields)
        , execute_(executeFunction)
        , outField_(outField),
        tmpRet_(buffer)
    {}

    void execute()
    {
        for (auto& [src, dst] : syncFields_)
        {
            syncToFoam(src, dst);
        }
        tmpRet_ = execute_();
    }

    NNOutField& getValue() {
      syncFromFoam(tmpRet_(), outField_);
      return outField_;
    };
};

}
