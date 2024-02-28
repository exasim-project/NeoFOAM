// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <cstdint>

namespace NeoFOAM
{

class treeAMRMeshModifier
{
public:

    /**
     * @brief abstract class that modfies for treeAMRMesh.
     * @param initialLevel_ The initial level of refinement for the mesh.
     * @param maxLevel_ The maximum level of refinement for the mesh.
     */
    treeAMRMeshModifier(int32_t initialLevel_, int32_t maxLevel_)
        : initialLevel_(initialLevel_), maxLevel_(maxLevel_), nElements_(-1), V_(), Cx_(), Cy_(), Cz_(), Cfx_(), Cfy_(), Cfz_(), Ax_(), Ay_(), Az_(), owner_(), neighbour_() {};

    /**
     * @brief Destructor for treeAMRMesh.
     */
    virtual ~treeAMRMeshModifier() = default;

    /**
     * @brief Refines the mesh.
     * @return True if the mesh was successfully refined, false otherwise.
     */
    virtual bool refine() = 0;

    /**
     * @brief Coarsens the mesh.
     * @return True if the mesh was successfully coarsened, false otherwise.
     */
    virtual bool coarsen() = 0;

    /**
     * @brief Balances the mesh.
     * @return True if the mesh was successfully balanced, false otherwise.
     */
    virtual bool balance() = 0;

    /**
     * @brief Writes the mesh to a file.
     */
    virtual void write() = 0;

    virtual int32_t nElements() const
    {
        return nElements_;
    }

    virtual int32_t initialLevel() const
    {
        return initialLevel_;
    }

    virtual int32_t maxLevel() const
    {
        return maxLevel_;
    }

    // const getters to protected members
    const std::vector<double>& V() const
    {
        return V_;
    }

    const std::vector<double>& level() const
    {
        return level_;
    }

    const std::vector<double>& Cx() const
    {
        return Cx_;
    }

    const std::vector<double>& Cy() const
    {
        return Cy_;
    }

    const std::vector<double>& Cz() const
    {
        return Cz_;
    }

    const std::vector<double>& Cfx() const
    {
        return Cfx_;
    }

    const std::vector<double>& Cfy() const
    {
        return Cfy_;
    }

    const std::vector<double>& Cfz() const
    {
        return Cfz_;
    }

    const std::vector<double>& Ax() const
    {
        return Ax_;
    }

    const std::vector<double>& Ay() const
    {
        return Ay_;
    }

    const std::vector<double>& Az() const
    {
        return Az_;
    }

    const std::vector<int32_t>& owner() const
    {
        return owner_;
    }

    const std::vector<int32_t>& neighbour() const
    {
        return neighbour_;
    }

protected:

    int32_t initialLevel_; ///< The initial level of refinement for the mesh.
    int32_t maxLevel_;     ///< The maximum level of refinement for the mesh.
    int32_t nElements_;    ///< The number of elements in the mesh.

    std::vector<double> V_;
    std::vector<double> level_;

    // centroid
    std::vector<double> Cx_;
    std::vector<double> Cy_;
    std::vector<double> Cz_;

    // face centroid
    std::vector<double> Cfx_;
    std::vector<double> Cfy_;
    std::vector<double> Cfz_;

    // face area
    std::vector<double> Ax_;
    std::vector<double> Ay_;
    std::vector<double> Az_;

    std::vector<int32_t> owner_;     // face: cell index owning the face outwards normal
    std::vector<int32_t> neighbour_; // face: cell index not-owning the face inwards normal
};

} // namespace NeoFOAM
