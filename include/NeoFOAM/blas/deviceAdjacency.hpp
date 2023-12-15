// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include <span>

#include <Kokkos_Core.hpp>

#include "primitives/label.hpp"
#include "deviceField.hpp"

namespace NeoFOAM {

    template<typename Tlabel>
    constexpr Kokkos::pair<const Tlabel&, const Tlabel&> order_edge_vertex(const Kokkos::pair<Tlabel, Tlabel>* edge) {
        return edge->first < edge->second ? Kokkos::pair<const Tlabel&, const Tlabel&>({edge->first, edge->second})
                                          : Kokkos::pair<const Tlabel&, const Tlabel&>({edge->second, edge->first});
    }

    // Basically a CSR approach to connectivity - avoids 'vector of vectors' memory layout for connectivity.
    template <typename Tlabel, bool directed>
    class deviceAdjacency
    {
        public:
        KOKKOS_FUNCTION
        deviceAdjacency() = default;

        KOKKOS_FUNCTION
        deviceAdjacency(const std::string& name) : name_(name) {}

        KOKKOS_FUNCTION
        deviceAdjacency(const deviceAdjacency<Tlabel, directed> &other)
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
        deviceAdjacency<Tlabel, directed> &operator=(deviceAdjacency<Tlabel, directed> &&other)
        {
            if (this != &other)
            {
                adjacency_ = std::move(other.field_);
                offset_ = std::move(other.field_);
            }
            return *this;
        }

        [[nodiscard]] constexpr bool empty() const { return offset_.size() < 2; }

        [[nodiscard]] inline auto data() const
        {
            return {adjacency_.data(), offset_.data()};
        }

        [[nodiscard]] inline std::string name() const noexcept
        {
            return name_; 
        }

        void resize(Tlabel size) {

            // Branch based on expansion or contraction of the graph.
            if(offset_.size() < (size + 1)) {
                const auto oldsize = offset_.size();
                Kokkos::resize(offset_, size + 1);
                Kokkos::parallel_for(offset_.size(), KOKKOS_LAMBDA(const Tlabel i){
                    offset_(i) = i >= oldsize ? offset_(oldsize - 1) : offset_(i);
                    });
                
                
                //offset_.size() == 0 ? 0 : offset_(offset_.size() - 1)
            }
            else {

                // Resize the containers.
                // offset.resize(size + 1);
                // vertex_adjacent_list.resize(offset.back());

                // Loop over remaining vertices, looking for edges which no longer exist.
                // std::size_t offset_carry = 0;
                // FOR(i_vertex, size_vertex()) {
                // auto [adjacency_begin, adjacency_end] = vertex_adjacency_iter(i_vertex);
                // const auto erase_iter = std::lower_bound(adjacency_begin, adjacency_end, size);
                // if(erase_iter != adjacency_end) {
                //     const auto offset_current = std::distance(erase_iter, adjacency_end);
                //     vertex_adjacent_list.erase(erase_iter, adjacency_end);
                //     offset[i_vertex + 1] -= offset_current;
                //     offset_carry += offset_current;
                // }
                // offset[i_vertex + 2] -= offset_carry;
                // }
            }
        }


        [[nodiscard]] inline Tlabel size() const noexcept 
        {
            return offset_.size() > 0 ? offset_.size() - 1 : 0; 
        }

        [[nodiscard]] bool contains(Kokkos::pair<Tlabel, Tlabel> connect) const {
            if(!directed) {
                const auto ordered_edge = order_edge_vertex(&connect);
                if(ordered_edge.second >= size() || empty()) return false;
                auto adjacency = (*this)(ordered_edge.first);
                for(auto index = 0; index < adjacency.size(); ++index) {
                    if(ordered_edge.second == adjacency(index)) return true;
                }
                return false;
            }
        }

        bool insert(Kokkos::pair<Tlabel, Tlabel> connect) {
            
            std::cout<<"\n----------------";
 

            // Preliminaries check if the edge exists, then determine if we need to resize.
            if(contains(connect)) return false;            
            const auto& [i_lower_vertex, i_upper_vertex] = order_edge_vertex(&connect);
            const bool is_insert_lower = !directed || i_lower_vertex == connect.first;
            const bool is_insert_upper = !directed || i_upper_vertex == connect.first; 
            if(i_upper_vertex >= size()) resize(i_upper_vertex + 1);

            std::cout<<"\n"<<offset_.size()<<"\n";
            for(auto i = 0; i < offset_.size(); ++i){
                std::cout<<offset_(i)<<", ";
            }
            std::cout<<"\n"<<adjacency_.size()<<"\n";
            for(auto i = 0; i < adjacency_.size(); ++i){
                std::cout<<adjacency_(i)<<", ";
            }

            std::cout<<"\n...";
            insertAdjacency(connect);
            std::cout<<"\n...";

            // Insert lower vertex.
             //   insert_vertex_adjacent_list(i_lower_vertex, i_upper_vertex);
            Kokkos::parallel_for("lower_offset_update", offset_.size(), 
                                  KOKKOS_LAMBDA(const Tlabel i) {
                                  offset_(i) += static_cast<Tlabel>(i > i_lower_vertex) + static_cast<Tlabel>(i > i_upper_vertex);
                                        });
            std::cout<<"\n"<<offset_.size()<<"\n";
            for(auto i = 0; i < offset_.size(); ++i){
                std::cout<<offset_(i)<<", ";
            }
            std::cout<<"\n"<<adjacency_.size()<<"\n";
            for(auto i = 0; i < adjacency_.size(); ++i){
                std::cout<<adjacency_(i)<<", ";
            }
            return true;

        }

//        [[nodiscard]] inline std::span<const Tlabel> at(const Tlabel& index) const {
//            return (*this)[index];
//        }

        [[nodiscard]] inline Kokkos::View<const Tlabel *> operator()(const Tlabel& index) const {
            return Kokkos::View<const Tlabel *>(adjacency_, Kokkos::make_pair(offset_(index), offset_(index + 1)));
        }

    private:

    Kokkos::View<Tlabel *> adjacency_;
    Kokkos::View<Tlabel *> offset_;  // is one greater than size
    std::string name_;

    // offsets must be correct, will be incorrect after return - assumes the connection does not exist
    void insertAdjacency(const Kokkos::pair<Tlabel, Tlabel>& connect) 
    {
        // assert !contains(connection)
        const bool is_adjacency_empty = adjacency_.size() == 0;
        Kokkos::pair<Tlabel, Tlabel> index_insert = {0, 0};
        
        if(!is_adjacency_empty)
            for(int i_connection = 0; i_connection < 2; ++i_connection) {
                    const Tlabel& connect_0 = i_connection == 0 ? connect.first : connect.second; 
                    const Tlabel& connect_1 = i_connection == 0 ? connect.second : connect.first; 
                    Tlabel& insert = i_connection == 0 ? index_insert.first : index_insert.second; 
                    std::cout<<"\nis_row_empty: "<<(offset_(connect_0) == offset_(connect_0 + 1));
                    
                    // Determine offsets, different if the row is empty.
                    if(offset_(connect_0) == offset_(connect_0 + 1)) insert = offset_(connect_0);
                    else
                        insert = offset_(connect_0 + 1);
                        for(auto i_offset = offset_(connect_0); i_offset < offset_(connect_0 + 1); ++i_offset)
                        {
                             std::cout<<"\nchecking "<<connect_1<<" against connection: "<<connect_0<<" - "<<adjacency_(i_offset);
                            if(adjacency_(i_offset) > connect_1) {
                                insert = i_offset;
                                break;
                            }
                        }   
                if(directed) break;
            }
        

        std::cout<<"\nadjacency_empty: "<<is_adjacency_empty;
        std::cout<<"\n0: "<<connect.first;
        std::cout<<"\n1: "<<connect.second;
        std::cout<<"\ni_0: "<<index_insert.first;
        std::cout<<"\ni_1: "<<index_insert.second;
        Kokkos::View<Tlabel *> temp(adjacency_.label(), adjacency_.size());
        Kokkos::deep_copy(temp, adjacency_);
        Kokkos::resize(adjacency_, adjacency_.size() + 1 + offset_shift());
        Kokkos::parallel_for("adjacency_insert", adjacency_.size(), 
                             KOKKOS_LAMBDA(const Tlabel i) {
                             if(i < index_insert.first) adjacency_(i) = is_adjacency_empty ? 0 : temp(i);
                             else if(i == index_insert.first) adjacency_(i) = connect.second;
                             else if(index_insert.first < i && i < index_insert.second + 1) 
                                adjacency_(i) = is_adjacency_empty ? 0 : temp(i - 1);
                             else if(!directed && i == index_insert.second + 1) adjacency_(i) = connect.first;
                             else adjacency_(i) = is_adjacency_empty ? 0 : temp(i - 1 - offset_shift());
                            });
       
    }

 

    constexpr size_t offset_shift() const noexcept {return static_cast<size_t>(!directed);}

    };
} // namespace NeoFOAM
