// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include <span>

#include <Kokkos_Core.hpp>

#include "primitives/label.hpp"
#include "deviceField.hpp"

namespace NeoFOAM
{
    // Basically a CSR approach to connectivity - avoids 'vector of vectors' memory layout for connectivity.
    template <typename Tlabel>
    class deviceAdjacency
    {
        public:
        KOKKOS_FUNCTION
        deviceAdjacency(const deviceAdjacency<Tlabel> &other)
            : adjacency_(other.adjacency_), offset_(other.offset_) {}

        KOKKOS_FUNCTION
        deviceAdjacency(const Kokkos::View<Tlabel* > &adjacency, const Kokkos::View<Tlabel *> &offset)
            : adjacency_(adjacency), offset_(offset) {}

        KOKKOS_FUNCTION
        deviceAdjacency(const std::string& name, const int size)
            : name_(name), offset_(Kokkos::View<Tlabel* >(name + offset_.name(), size)) {} // note adjacency not sized
        
        KOKKOS_FUNCTION
        deviceAdjacency(const std::string& name, const deviceField<Tlabel>& adjacency,
                        const deviceField<Tlabel>& offset) 
                        : name_(name)
        {   
            Kokkos::resize(Kokkos::WithoutInitializing, adjacency_, adjacency.size());
            Kokkos::resize(Kokkos::WithoutInitializing, offset_, offset.size());

            Kokkos::parallel_for("init adjacency_", adjacency_.size(),
                         KOKKOS_CLASS_LAMBDA (const int& i) {
                        adjacency_(i) = adjacency(i);
                        });

            Kokkos::parallel_for("init offset", offset_.size(),
                         KOKKOS_CLASS_LAMBDA (const int& i) {
                        offset_(i) = offset(i);
                        });
        }

        // move assignment operator
        deviceAdjacency<Tlabel> &operator=(deviceAdjacency<Tlabel> &&other)
        {
            if (this != &other)
            {
                adjacency_ = std::move(other.field_);
                offset_ = std::move(other.field_);
            }
            return *this;
        }

        [[nodiscard]] inline auto data() const
        {
            return {adjacency_.data(), offset_.data()};
        }

        [[nodiscard]] inline std::string name() const noexcept
        {
            return name_; 
        }

        [[nodiscard]] inline Tlabel size() const noexcept 
        {
            return offset_.size() - 1;
        }

        void insert(Kokkos::pair<Tlabel, Tlabel> connect)
        {
            std::cout<<"not-implemented => deviceAdjacency::insert"<<std::endl;
            exit(1);
        }

//        [[nodiscard]] inline std::span<const Tlabel> at(const Tlabel& index) const {
//            return (*this)[index];
//        }

        [[nodiscard]] inline Kokkos::View<Tlabel *> operator()(const Tlabel& index) const {
            return Kokkos::View<Tlabel *>(adjacency_, Kokkos::make_pair(offset_(index), offset_(index + 1)));
        }

    private:

        Kokkos::View<Tlabel *> adjacency_;
        Kokkos::View<Tlabel *> offset_;  // is one greater than size
        std::string name_;



    };
} // namespace NeoFOAM
