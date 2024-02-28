// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/blas/field.hpp"
#include "NeoFOAM/mesh/unstructuredMesh/unstructuredMesh.hpp"
#include "Kokkos_Core.hpp"
#include <functional>


namespace NeoFOAM
{

void grad_atmoic(NeoFOAM::vectorField& gradPhi, const NeoFOAM::unstructuredMesh& mesh_, const NeoFOAM::scalarField& phi);

void grad_not_atmoic(NeoFOAM::vectorField& gradPhi, const NeoFOAM::unstructuredMesh& mesh_, const NeoFOAM::scalarField& phi);

vectorField create_gradField(int nCells);

class gaussGreenGrad
{
public:

    static inline std::map<std::string, std::function<void(NeoFOAM::vectorField&, const NeoFOAM::unstructuredMesh&, const scalarField&)>> algorithmMap_;

    template<typename func>
    static void registerAlgorithm(std::string name, func algorithm)
    {
        algorithmMap_[name] = algorithm;
    }


    gaussGreenGrad(const unstructuredMesh& mesh);

    const vectorField& grad(const scalarField& phi, std::string algorithm);

    const vectorField& grad(const scalarField& phi);

    void grad_allocate(const scalarField& phi);

    void grad(vectorField& gradPhi, const scalarField& phi);

    void grad_atomic(vectorField& gradPhi, const scalarField& phi);

    // Register functions in the map using a static data member

    static struct RegisterFunctions
    {
        RegisterFunctions()
        {
            // Register functions here
            registerAlgorithm("atomic", grad_atmoic);
            registerAlgorithm("not_atomic", grad_not_atmoic);
        }
    } registerFunctions; // Static data member that registers functions at compile time


private:

    vectorField gradPhi_;
    const unstructuredMesh& mesh_;
};

} // namespace NeoFOAM