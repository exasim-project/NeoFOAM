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
    template <typename Tlabel, bool directed>
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

        bool insert(Kokkos::pair<Tlabel, Tlabel> connect)
        {
            // Preliminaries check if the edge exists, then determine if we need to resize.
            //if(contains(edge)) return false;            
            const auto& [i_lower_vertex, i_upper_vertex] = order_edge_vertex(&edge);
            const bool is_insert_lower = !directed || i_lower_vertex == connect.first;
            const bool is_insert_upper = !directed || i_upper_vertex == connect.first; 
            
            if(i_upper_vertex >= size()) 
            {
                Kokkos::resize(offset_, i_upper_vertex + 1);
            }

            // Insert lower vertex.
            if(is_insert_lower) {
             //   insert_vertex_adjacent_list(i_lower_vertex, i_upper_vertex);
                Kokkos::parallel_for("lower_offset_update", offset_.size(), 
                                     KOKKOS_LAMBDA((const Tlabel i){++offset_(i);})
            );
            }

            // Insert upper vertex (tricky: last for loop is safe -> The highest vertex + 1 (for size) + 1 (for offset) <= end()).
            if(is_insert_upper) {
             //   insert_vertex_adjacent_list(i_upper_vertex, i_lower_vertex);
                // Kokkos::parallel_for("upper_offset_update",
                //                      std::next(offset_.begin(), static_cast<Tlabel>(i_upper_vertex + 1)), offset_.end(),
                //                      [](Tlabel& off){++off;});
            }

            return true;

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

    /**
     * @details This function inserts a new vertex into the adjacency list of the specified vertex in the graph. If the
     * inserted vertex is already present in the adjacency list, this function does nothing.
     */
    void insertAdjacency(std::size_t node, std::size_t connection) 
    {
        const auto& adjacency = vertex_adjacency_iter(vertex);
        const auto& insert_iter = std::lower_bound(adjacency.first, adjacency.second, insert_vertex);
        const auto& distance = std::distance(vertex_adjacent_list.begin(), insert_iter);
        vertex_adjacent_list.insert(vertex_adjacent_list.begin() + distance, insert_vertex);
    }


    };
} // namespace NeoFOAM
