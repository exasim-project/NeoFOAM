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
using edge = Kokkos::pair<Tlabel, Tlabel>;  //!< A pair of labels representing an the 0 and 1 nodes of an edge. In the case of directed edge, the 0 node is the source and the 1 node is the target.

template<typename Tlabel>
/**
 * @brief Orders an unordered edge in accending order.
 * @param[in] unorderedEdge Pointer to the unordered edge.
 * @return The ordered edge, where the 0 node points to the 1 node.
 */
constexpr edge<const Tlabel&> order_edge(const edge<Tlabel>* unorderedEdge) {
    return unorderedEdge->first < unorderedEdge->second 
            ? edge<const Tlabel&>({unorderedEdge->first, unorderedEdge->second})
            : edge<const Tlabel&>({unorderedEdge->second, unorderedEdge->first});
}

/**
 * @brief Represents an adjacency graph, for both directed and undirected variants.
 * 
 * @tparam Tlabel The type of the labels in the adjacency matrix.
 * @tparam directed A boolean value indicating whether the adjacency matrix is directed or not.
 * 
 * @details The `deviceAdjacency` class represents an adjacency graph, which records the connectivity relationship 
 * between nodes through edges. The graph stores specifically the connectivity between nodes in the graph. The template
 * parameter `Tlabel` specifies the type of the labels int, uint32, etc. The `directed` parameter indicates whether the 
 * graph is directed or undirected. In the undirected variant the graph edges are considered to connect both nodes (of
 * the edge), and therefor no distiction is made interms of direction between the 0 and 1 node of the edge. Thus the
 * graph is symmetric, for example, if node 10 is connected to node 20 then node 20 is connected to 10. In the directed 
 * variant, an edge connects the 0 node to the 1 node, and therefor the graph is not symmetric. For example, if node 10
 * is connected to node 20, the reverse is not true. 
 * 
 * A word no memory layout: To optimise memory access the data in the class is stored 'flat' with an adjacency View and
 * an offset View. The former contains the connections of the graph while the later stores the start and end position 
 * (offset to) of each nodes' connectivity in the adjacency container. Within each node's connectivity the connections 
 * are sorted in accending order. Finally, the offset container is one greater than the size of the adjacency container,
 * to faciliate slightly more lazy programing - since the first element will always be zero.
 * 
 * @note An directed graph can therefor be used to represent the connection between two different 'types/classes' of 
 * nodes rather than the same. For example, a graph could be used to represent the connection between cells and faces. 
 *
 * @todo Nice to have functions/operators operator+ (graph unition), operator-(graph intersection), remove to an edge.
 * @todo Batch insert for improved performance.
 * @todo Optimise undirected memory layout - currently data is duplicated.
 * @todo Add further 'stl like functions'
 */
template <typename Tlabel, bool directed>
class deviceAdjacency
{
    public:
 
    // ----------------------------------------------------------------------------------------------------------------
    // Constructors and Destructors
    // ----------------------------------------------------------------------------------------------------------------
    
    KOKKOS_FUNCTION
    deviceAdjacency() = default;

    KOKKOS_FUNCTION
    deviceAdjacency(const std::string& name) : name_(name) {}

    KOKKOS_FUNCTION
    deviceAdjacency(const deviceAdjacency<Tlabel, directed> &other)
        : adjacency_(other.adjacency_), offset_(other.offset_), name_(other.name_) {}

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

    // ----------------------------------------------------------------------------------------------------------------
    // Assignement Operator
    // ----------------------------------------------------------------------------------------------------------------

    /**
     * @brief Move Assignment operator for deviceAdjacency. 
     * @tparam Tlabel The type of the labels in the adjacency matrix.
     * @tparam directed A boolean value indicating whether the adjacency matrix is directed or not.
     * @param[in] other The deviceAdjacency object to be assigned.
     * @return A reference to this deviceAdjacency object.
     * 
     * @details This operator assigns the contents of another deviceAdjacency object to this object.
     */     
    deviceAdjacency<Tlabel, directed> &operator=(deviceAdjacency<Tlabel, directed> &&other)
    {
        if(this != &other)
        {
            adjacency_ = std::move(other.adjacency_);
            offset_ = std::move(other.offset_);
            name_ = std::move(other.name_);
        }
        return *this;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Capacity
    // ----------------------------------------------------------------------------------------------------------------

    /**
     * @brief Checks if the deviceAdjacency is empty.
     * @return True if the deviceAdjacency is empty, false otherwise.
     */
    [[nodiscard]] constexpr bool empty() const { return offset_.size() < 2; }

    /**
     * @brief Returns the size of the adjacency list.
     * @return The size of the adjacency list.
     */
    [[nodiscard]] inline Tlabel size() const noexcept 
    {
        return offset_.size() > 0 ? offset_.size() - 1 : 0; 
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Element Access
    // ----------------------------------------------------------------------------------------------------------------

    /**
     * @brief Returns a Kokkos::View object containing the connection for the parsed node index. 
     * @param[in] i_node The index for which the adjacency information is requested.
     * @return A Kokkos::View object containing the adjacency information.
     * 
     * @details This function returns a Kokkos::View object that provides a const pointer to Tlabel.
     * The view is created using the adjacency_ array and the offset_ array, with the range
     * specified by the index and index + 1. The resulting view represents the adjacency
     * information for the given index.
     * 
     * @warning The node index must be in the rage [0: size()].
     */
    [[nodiscard]] inline Kokkos::View<const Tlabel *> operator()(const Tlabel& i_node) const 
    {
        //todo: Error handling - range check. 
        return Kokkos::View<const Tlabel *>(adjacency_, Kokkos::make_pair(offset_(i_node), offset_(i_node + 1)));
    }

    /**
     * @brief Returns a pair of pointers to the data stored in the private adjacency and offset arrays.
     * @return A pair of pointers to the adjacency and offset data.
     */
    [[nodiscard]] inline auto data() const
    {
        return {adjacency_.data(), offset_.data()};
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Modifiers
    // ----------------------------------------------------------------------------------------------------------------

    /**
     * @brief Inserts an edge connection into the graph.
     * @param[in] edge The edge connection to add.
     * @return True if the edge was successfully inserted, false otherwise (because it already exists in the graph).
 
     * @details This function inserts the specified edge connection into the graph. In the case of an undirected graph
     * both the connectivity of the first and second node are updated. In the case of a directed graph only the edge is
     * interpreted as first -> second and therefore only the connectivity of the first node is updated.
     * 
     * @note The graph will be resized to accomodate the new edge if necessary. 
     */
    bool insert(const edge<Tlabel>& edge) {
        
        // Do we need to resize the graph (offset_ container)?
        if(contains(edge)) return false;            
        if(directed && edge.first >= size()) resize(edge.first + 1);
        if(!directed && std::max(edge.first, edge.second) >= size()) resize(std::max(edge.first, edge.second) + 1);

        // Updated adjacency and offset.
        insertAdjacency(edge);
        Kokkos::parallel_for("lower_offset_update", offset_.size(), 
                                KOKKOS_LAMBDA(const Tlabel i) {
                                offset_(i) += static_cast<Tlabel>(i > edge.first) 
                                            + static_cast<Tlabel>((i > edge.second)*(!directed));
                                    });
        return true;
    }

    /**
     * @brief Resizes the adjacency container to accomodate more nodes.
     * @param[in] size The new size of the container.
     *
     * @details This function resizes the adjacency container to accommodate the specified size. It may allocate 
     * additional memory or deallocate existing memory as necessary. New nodes are have no connections.
     * 
     * @note When sizing down the interprentation is that 'nodes' are being lost. Therefor remaining nodes which are
     * connected to the lost nodes will have their connectivity updated to reflect the loss of the node.
     */
    void resize(const Tlabel& size) {

        // Branch based on expansion or contraction of the graph.
        if(offset_.size() < (size + 1)) {
            const auto oldsize = offset_.size();
            Kokkos::resize(offset_, size + 1);
            Kokkos::parallel_for(offset_.size(), KOKKOS_LAMBDA(const Tlabel i){
                offset_(i) = i >= oldsize ? offset_(oldsize - 1) : offset_(i);
                });
        }
        else 
        {
            Kokkos::resize(offset_, size + 1);
            Kokkos::resize(adjacency_, offset_(offset_.size() - 1));
            if(directed) return;

            // Remove any connections to the removed vertices.
            Kokkos::View<Tlabel*> temp_offset(offset_.label(), offset_.size());
            Kokkos::deep_copy(temp_offset, offset_);

            Tlabel total_offset = 0;
            Kokkos::parallel_scan("determine_new_offsets", offset_.size() - 1,
                KOKKOS_LAMBDA(Tlabel i_node, Tlabel& partial_sum, bool is_final) {
                    for(auto i_offset = temp_offset(i_node); i_offset < temp_offset(i_node + 1); ++i_offset) {
                        if(adjacency_(i_offset) < size) partial_sum += 1;
                    }               
                    if(is_final) offset_(i_node) = partial_sum;
                }, total_offset);

            Kokkos::View<Tlabel*> temp_adjacency("",adjacency_.size());
            Kokkos::deep_copy(temp_adjacency, adjacency_);         
            Kokkos::resize(adjacency_, total_offset);

            Kokkos::parallel_for("update_adjacency", offset_.size() - 1,
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

    // ----------------------------------------------------------------------------------------------------------------
    // Lookup
    // ----------------------------------------------------------------------------------------------------------------

    /**
     * @brief Checks if the adjacency list contains a specific connection.
     * @param[in] check_edge The edge to check for in the graph.
     * @return True if the edge is found, false otherwise.
     *
     * @details This function checks if the adjacency list contains a specific connection.
     * It returns true if the connection is found, and false otherwise.
     *
     * @warning The connection must have a valid label.
     * 
     * @note This function assumes that the adjacency list is properly initialized.
     */
    [[nodiscard]] bool contains(const edge<Tlabel>& check_edge) const 
    {   
        if(empty()) return false;
        const auto& ordered_edge = !directed ? order_edge(&check_edge) 
                                             : edge<const Tlabel&>({check_edge.first, check_edge.second});
        if((!directed ? ordered_edge.second : ordered_edge.first) >= size()) return false;
        const auto& adjacency = (*this)(ordered_edge.first);
        for(auto index = 0; index < adjacency.size(); ++index) 
        {
            if(adjacency(index) == ordered_edge.second) return true;
            if(adjacency(index) > ordered_edge.second) return false;
        }
        return false;
    }

    /**
     * @brief Returns the name of the object.
     * @return The name of the object.
     */
    [[nodiscard]] constexpr std::string name() const noexcept
    {
        return name_; 
    }

    private:

    Kokkos::View<Tlabel *> adjacency_;  //!< Stores adjacency information for the nodes of the graph.
    Kokkos::View<Tlabel *> offset_;     //!< One greater than size, stores offset positions of the adjaceny for each node.
    std::string name_;                  //!< Name of the object.

    /**
     * @brief Inserts an adjacency edge into the adjacency list.
     * @param[in] edge The edge to be used to inserted further adjacency.
     * 
     * @details This function inserts an adjacency edge into the adjacency list. It determines the correct position to
     * insert the edge based on the existing connections of the parse edge. Where correct position refers to the class
     * storage of connections being sorted in an ascending fashion for each node.
     * 
     * @note This function assumes that the adjacency list is already sorted in ascending order.
     * 
     * @warning The offset_ view must already be resized to accomodate the new edge, but must contain the old offset 
     * data.
     * @warning The offset_ veiw is not updated, it is of the caller to update the offset_ view post this call.
     * @warning This function does not perform any checks for duplicate edges. It is the responsibility of the caller 
     * to ensure that duplicate edges are not inserted. The graph will be in an undefined state if duplicate are added.
     */
    void insertAdjacency(const edge<Tlabel>& edge) 
    {
        const auto ordered_edge = order_edge(&edge);
        const bool is_adjacency_empty = adjacency_.size() == 0;
        Kokkos::pair<Tlabel, Tlabel> index_insert = {0, 0};
       
        if(!is_adjacency_empty)
            for(int i_connection = 0; i_connection < 2; ++i_connection) 
            {
                const Tlabel& connect_0 = i_connection == 0 ? edge.first : edge.second; 
                const Tlabel& connect_1 = i_connection == 0 ? edge.second : edge.first; 
                Tlabel& insert = i_connection == 0 ? index_insert.first : index_insert.second; 
                
                // Determine offsets, different if the row is empty.
                if(offset_(connect_0) == offset_(connect_0 + 1)) insert = offset_(connect_0);
                else insert = offset_(connect_0 + 1);
                for(auto i_offset = offset_(connect_0); i_offset < offset_(connect_0 + 1); ++i_offset)
                {
                    if(adjacency_(i_offset) > connect_1) 
                    {
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
                             KOKKOS_LAMBDA(const Tlabel i) 
                             {
                                if(i < index_insert.first) adjacency_(i) = is_adjacency_empty ? 0 : temp(i);
                                else if(i == index_insert.first) adjacency_(i) = edge.second;
                                else if(index_insert.first < i && i < index_insert.second + 1) 
                                    adjacency_(i) = is_adjacency_empty ? 0 : temp(i - 1);
                                else if(!directed && i == index_insert.second + 1) adjacency_(i) = edge.first;
                                else adjacency_(i) = is_adjacency_empty ? 0 : temp(i - 1 - offset_shift());
                            });
       
    }

    /**
     * @brief Returns an offset shift value, determined by being a directed or undirected graph.
     * @return The offset shift value.
     */
    constexpr size_t offset_shift() const noexcept {return static_cast<size_t>(!directed);}

    };
} // namespace NeoFOAM
