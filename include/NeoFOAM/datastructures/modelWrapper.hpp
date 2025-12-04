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
    typename NNOutField
    >
  class ModelAdapter {

    private:
        std::vector<std::pair<fvcc::VolumeField<NeoN::Vec3>&,Foam::volVectorField&>> syncFields_;
        ExecuteFunctionType execute_;
        NNOutField& outField_;

    public:

        ModelAdapter(
          std::vector<std::pair<fvcc::VolumeField<NeoN::Vec3>&,Foam::volVectorField&>> syncFields,
            ExecuteFunctionType executeFunction,
            NNOutField& outField
                     ) :
    syncFields_(syncFields),
    execute_(executeFunction),
    outField_(outField)
    {}

        void execute(){
          auto execRet = execute_();
          // sync(outField_, execRet);
        }


        NNOutField& getValue() {
          return outField_;
        };


  };

}
