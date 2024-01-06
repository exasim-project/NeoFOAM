// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/cellCentredFiniteVolume/grad/gaussGreenGrad.hpp"
#include "NeoFOAM/blas/field.hpp"
#include <functional>

namespace NeoFOAM {

    void grad_atmoic(NeoFOAM::vectorField &gradPhi, const NeoFOAM::unstructuredMesh &mesh_, const NeoFOAM::scalarField &phi)
    {
        const NeoFOAM::labelField &owner = mesh_.owner_;
        const NeoFOAM::labelField &neighbour = mesh_.neighbour_;
        const NeoFOAM::vectorField &Sf = mesh_.Sf_;
        const NeoFOAM::scalarField &V = mesh_.V_;

        Kokkos::parallel_for(
            "gaussGreenGrad", mesh_.nInternalFaces_, KOKKOS_LAMBDA(const int i) {
                int32_t own = owner(i);
                int32_t nei = neighbour(i);
                NeoFOAM::scalar phif = 0.5 * (phi(nei) + phi(own));
                NeoFOAM::vector value_own = Sf(i) * (phif / V(own));
                NeoFOAM::vector value_nei = Sf(i) * (phif / V(nei));
                Kokkos::atomic_add(&gradPhi(own), value_own);
                Kokkos::atomic_sub(&gradPhi(nei), value_nei);
            });
    }

    void grad_not_atmoic(NeoFOAM::vectorField &gradPhi, const NeoFOAM::unstructuredMesh &mesh_, const NeoFOAM::scalarField &phi)
    {
        const NeoFOAM::labelField &owner = mesh_.owner_;
        const NeoFOAM::labelField &neighbour = mesh_.neighbour_;
        const NeoFOAM::vectorField &Sf = mesh_.Sf_;
        const NeoFOAM::scalarField &V = mesh_.V_;

        Kokkos::parallel_for(
            "gaussGreenGrad", mesh_.nInternalFaces_, KOKKOS_LAMBDA(const int i) {
                int32_t own = owner(i);
                int32_t nei = neighbour(i);
                NeoFOAM::scalar phif = 0.5 * (phi(nei) + phi(own));
                NeoFOAM::vector value_own = Sf(i) * (phif / V(own));
                NeoFOAM::vector value_nei = Sf(i) * (phif / V(nei));
                gradPhi(own) += value_own;
                gradPhi(nei) += value_nei;
                // Kokkos::atomic_add(&gradPhi(own), value_own);
                // Kokkos::atomic_sub(&gradPhi(nei), value_nei);
            });
    }

    vectorField create_gradField(int nCells)
    {
        NeoFOAM::vectorField gradPhi("gradPhi", nCells);
        return gradPhi;
    }

    gaussGreenGrad::gaussGreenGrad(const unstructuredMesh &mesh) : mesh_(mesh), gradPhi_("gradPhi", mesh.nCells_)
    {
    };

    const vectorField& gaussGreenGrad::grad(const scalarField &phi, std::string algorithm)
    {

        // check if algorithm is in map
        if (algorithmMap_.find(algorithm) == algorithmMap_.end())
        {
            std::cout << "algorithm not found" << std::endl;
            // display available algorithmMap_
            std::cout << "available options are: " << std::endl;
            for (const auto& pair : algorithmMap_) {
                const std::string& key = pair.first;
                std::cout << key << std::endl;
            }
            throw std::runtime_error("algorithm not found");
        }
        algorithmMap_[algorithm](gradPhi_,mesh_,phi);

        return gradPhi_;
    };

    const vectorField& gaussGreenGrad::grad(const scalarField &phi)
    {
        // vectorField gradPhi = create_gradField(mesh_.nCells_);

        // loop(gradPhi, mesh_, phi);
        const labelField &owner = mesh_.owner_;
        const labelField &neighbour = mesh_.neighbour_;
        const vectorField &Sf = mesh_.Sf_;
        const scalarField &V = mesh_.V_;

        Kokkos::parallel_for(
            "gaussGreenGrad", mesh_.nInternalFaces_, KOKKOS_CLASS_LAMBDA(const int i) {
                int32_t own = owner(i);
                int32_t nei = neighbour(i);
                scalar phif = 0.5 * (phi(nei) + phi(own));
                vector value_own = Sf(i) * (phif / V(own));
                vector value_nei = Sf(i) * (phif / V(nei));
                gradPhi_(own) = gradPhi_(own) + value_own;
                gradPhi_(nei) = gradPhi_(nei) + value_nei;
                // Kokkos::atomic_add(&gradPhi_(own), value_own);
                // Kokkos::atomic_sub(&gradPhi_(nei), value_nei);
            });

        return gradPhi_;
    };

    void gaussGreenGrad::grad_allocate(const scalarField &phi)
    {
        // vectorField gradPhi = create_gradField(mesh_.nCells_);

        // loop(gradPhi, mesh_, phi);
        const labelField &owner = mesh_.owner_;
        const labelField &neighbour = mesh_.neighbour_;
        const vectorField &Sf = mesh_.Sf_;
        const scalarField &V = mesh_.V_;

        Kokkos::parallel_for(
            "gaussGreenGrad", mesh_.nInternalFaces_, KOKKOS_CLASS_LAMBDA(const int i) {
                int32_t own = owner(i);
                int32_t nei = neighbour(i);
                scalar phif = 0.5 * (phi(nei) + phi(own));
                vector value_own = Sf(i) * (phif / V(own));
                vector value_nei = Sf(i) * (phif / V(nei));
                Kokkos::atomic_add(&gradPhi_(own), value_own);
                Kokkos::atomic_sub(&gradPhi_(nei), value_nei);
            });

    };

    void gaussGreenGrad::grad(vectorField& gradPhi, const scalarField &phi)
    {
        // vectorField gradPhi = create_gradField(mesh_.nCells_);

        // loop(gradPhi, mesh_, phi);
        const labelField &owner = mesh_.owner_;
        const labelField &neighbour = mesh_.neighbour_;
        const vectorField &Sf = mesh_.Sf_;
        const scalarField &V = mesh_.V_;

        Kokkos::parallel_for(
            "gaussGreenGrad", mesh_.nInternalFaces_, KOKKOS_LAMBDA(const int i) {
                int32_t own = owner(i);
                int32_t nei = neighbour(i);
                scalar phif = 0.5 * (phi(nei) + phi(own));
                vector value_own = Sf(i) * (phif / V(own));
                vector value_nei = Sf(i) * (phif / V(nei));
                gradPhi(own) = gradPhi(own) + value_own;
                gradPhi(nei) = gradPhi(nei) + value_nei;
                // Kokkos::atomic_add(&gradPhi(own), value_own);
                // Kokkos::atomic_sub(&gradPhi(nei), value_nei);
            });

    };

    void gaussGreenGrad::grad_atomic(vectorField& gradPhi, const scalarField &phi)
    {
        // vectorField gradPhi = create_gradField(mesh_.nCells_);

        // loop(gradPhi, mesh_, phi);
        const labelField &owner = mesh_.owner_;
        const labelField &neighbour = mesh_.neighbour_;
        const vectorField &Sf = mesh_.Sf_;
        const scalarField &V = mesh_.V_;

        Kokkos::parallel_for(
            "gaussGreenGrad", mesh_.nInternalFaces_, KOKKOS_LAMBDA(const int i) {
                int32_t own = owner(i);
                int32_t nei = neighbour(i);
                scalar phif = 0.5 * (phi(nei) + phi(own));
                vector value_own = Sf(i) * (phif / V(own));
                vector value_nei = Sf(i) * (phif / V(nei));
                // gradPhi(own) = gradPhi(own) + value_own;
                // gradPhi(nei) = gradPhi(nei) + value_nei;
                Kokkos::atomic_add(&gradPhi(own), value_own);
                Kokkos::atomic_sub(&gradPhi(nei), value_nei);
            });

    };

};