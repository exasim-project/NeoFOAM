// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <future>

#include "NeoN/NeoN.hpp"

namespace nnfvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{


/*@class A class to hold an OpenFOAM Model which executes on CPU and needs synchronization before execution
*
*/
template<
    typename ExecuteFunctionType,
    typename SyncTupleType,
    typename NNOutField>
class ModelAdapter
{

private:

    SyncTupleType syncFields_;
    ExecuteFunctionType execute_;
    NNOutField& outField_;
    std::future<Foam::volScalarField> tmpRet_;

public:

    ModelAdapter(
        SyncTupleType syncFields,
        ExecuteFunctionType executeFunction,
        NNOutField& outField
    )
        : syncFields_(syncFields)
        , execute_(executeFunction)
        , outField_(outField),
        tmpRet_()
    {
        execute();
    }

    void execute()
    {
        // NOTE currently hardcodes just two fields,
        // needs to iterate over tuple in pairs
        syncToFoam(std::get<0>(syncFields_), std::get<1>(syncFields_));
        tmpRet_ = std::async(std::launch::async, execute_);
    }

    NNOutField& getValue() {
        tmpRet_.wait();
        syncFromFoam(tmpRet_.get(), outField_);
        return outField_;
    };
};

}
