// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <iostream>
#include "primitives/scalar.hpp"

#include <span>

namespace NeoFOAM
{
    // Basically a CSR approach to connectivity - avoids 'vector of vectors' memory layout for connectivity.
    template <typename Tlabel>
    class deviceAdjacency
    {
    public:
        KOKKOS_FUNCTION
        deviceAdjacency(const deviceAdjacency<Tlabel> &rhs)
            : size_(rhs.size_), adjacency_(rhs.adjacency_), offset_(rhs.offset_)
        {

        }

        KOKKOS_FUNCTION
        deviceAdjacency(const Kokkos::View<Tlabel *> &adjacency, const Kokkos::View<Tlabel *> &offset)
            : size_(field.size()), adjacency_(adjacency_), offset_(rhs.offset)
        {

        }

        deviceAdjacency(const std::string &name, const int size)
            : size_(size), offset_(Kokkos::View<Tlabel *>(name, size)) // note adjacency not sized
        {

        }

        [[nodiscard]] inline auto data()
        {
            return {adjacency_.data(), offset_.data()};
        }

        [[nodiscard]] inline std::string name()
        {
            return offset_.name(); 
        }

        [[nodiscard]] inline Tlabel size()
        {
            return offset_.size();
        }

        [[nodiscard]] void insert(Kokkos::pair<Tlabel, Tlabel> connect);
        {
            std::cout<<"not-implemented"<<std::endl;
            exit(1);
        }

        [[nodiscard]] inline std::span<const Tlabel> at(const Tlabel& index) const {
            return (*this)[index];
        }

        [[nodiscard]] inline std::span<const Tlabel> operator[](const Tlabel& index) const {
            return {adjacency_.begin() + static_cast<Tlabel>(offset_[index]),
                    offset_[index + 1] - offset_[index]};
        }

    private:
        Kokkos::View<Tlabel *> adjacency_;
        Kokkos::View<Tlabel *> offset_;    // NOTE used .name here for class name

        Tlabel size_;
    };
} // namespace NeoFOAM
