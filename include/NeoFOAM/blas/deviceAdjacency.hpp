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
    using connection = Kokkos::pair<Tlabel, Tlabel>;

    template<typename Tlabel>
    constexpr connection<const Tlabel&> order_connection(const connection<Tlabel>* connect) {
        return connect->first < connect->second ? connection<const Tlabel&>({connect->first, connect->second})
                                                : connection<const Tlabel&>({connect->second, connect->first});
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
            }
            else {

                Kokkos::resize(offset_, size + 1);
                Kokkos::resize(adjacency_, offset_(offset_.size() - 1));
                if(directed) return;

                // Remove any connections to the removed vertices.

                Kokkos::View<Tlabel*> temp_offset(offset_.label(), offset_.size());
                Kokkos::deep_copy(temp_offset, offset_);

                Tlabel total_offset = 0;
                Kokkos::parallel_scan("Loop1", offset_.size() - 1,
                    KOKKOS_LAMBDA(Tlabel i_node, Tlabel& partial_sum, bool is_final) {
                        for(auto i_offset = temp_offset(i_node); i_offset < temp_offset(i_node + 1); ++i_offset) {
                            if(adjacency_(i_offset) < size) partial_sum += 1;
                        }               
                        if(is_final) offset_(i_node) = partial_sum;
                    }, total_offset);

                Kokkos::View<Tlabel*> temp_adjacency("",adjacency_.size());
                Kokkos::deep_copy(temp_adjacency, adjacency_);         
                Kokkos::resize(adjacency_, total_offset);

                Kokkos::parallel_for("Loop2", offset_.size() - 1,
                    KOKKOS_LAMBDA(Tlabel i_node) {
                        Tlabel i_offset = 0;
                        for(auto i_adjacency = temp_offset(i_node); i_adjacency < temp_offset(i_node + 1); ++i_adjacency) {
                            if(temp_adjacency(i_adjacency) < size) {
                                adjacency_(i_offset) = temp_adjacency(i_adjacency);
                                i_offset += 1;
                            }
                        }              
                    });
                           
            }
        }


        [[nodiscard]] inline Tlabel size() const noexcept 
        {
            return offset_.size() > 0 ? offset_.size() - 1 : 0; 
        }

        [[nodiscard]] bool contains(const connection<Tlabel>& connect) const {
            if(!directed) {
                const auto ordered_edge = order_connection(&connect);
                if(ordered_edge.second >= size() || empty()) return false;
                auto adjacency = (*this)(ordered_edge.first);
                for(auto index = 0; index < adjacency.size(); ++index) {
                    if(ordered_edge.second == adjacency(index)) return true;
                }
            }
            return false;
        }

        bool insert(const connection<Tlabel>& connect) {
           
            // Do we need to resize the graph (offset_ container)?
            if(contains(connect)) return false;            
            if(directed && connect.first >= size()) resize(connect.first + 1);
            if(!directed && std::max(connect.first, connect.second) >= size()) 
                resize(std::max(connect.first, connect.second) + 1);

            // Updated adjacency and offset.
            insertAdjacency(connect);
            Kokkos::parallel_for("lower_offset_update", offset_.size(), 
                                  KOKKOS_LAMBDA(const Tlabel i) {
                                  offset_(i) += static_cast<Tlabel>(i > connect.first) 
                                                + static_cast<Tlabel>((i > connect.second)*(!directed));
                                        });
            return true;
        }

        [[nodiscard]] inline Kokkos::View<const Tlabel *> operator()(const Tlabel& index) const {
            return Kokkos::View<const Tlabel *>(adjacency_, Kokkos::make_pair(offset_(index), offset_(index + 1)));
        }

    private:

    Kokkos::View<Tlabel *> adjacency_;
    Kokkos::View<Tlabel *> offset_;  // is one greater than size
    std::string name_;

    // Condition of the offset_ veiw, must be sized correctly for the new vertices but containd the old offset data,
    // this can be achieved using the resize function before calling this function
    // 0, will be incorrect after return - assumes the connection does not exist
    void insertAdjacency(const connection<Tlabel>& connect) 
    {
        const auto ordered_edge = order_connection(&connect);
        const bool is_adjacency_empty = adjacency_.size() == 0;
        Kokkos::pair<Tlabel, Tlabel> index_insert = {0, 0};
       
        if(!is_adjacency_empty)
            for(int i_connection = 0; i_connection < 2; ++i_connection) {
                    const Tlabel& connect_0 = i_connection == 0 ? connect.first : connect.second; 
                    const Tlabel& connect_1 = i_connection == 0 ? connect.second : connect.first; 
                    Tlabel& insert = i_connection == 0 ? index_insert.first : index_insert.second; 
                    
                    // Determine offsets, different if the row is empty.
                    if(offset_(connect_0) == offset_(connect_0 + 1)) insert = offset_(connect_0);
                    else
                        insert = offset_(connect_0 + 1);
                        for(auto i_offset = offset_(connect_0); i_offset < offset_(connect_0 + 1); ++i_offset)
                        {
                            if(adjacency_(i_offset) > connect_1) {
                                insert = i_offset;
                                break;
                            }
                        }   
                if(directed) break;
            }


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
