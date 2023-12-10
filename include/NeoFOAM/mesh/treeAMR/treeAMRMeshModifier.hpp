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
    : initialLevel_(initialLevel_), maxLevel_(maxLevel_), nElements_(-1)
    {
    };

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
    
    protected:
      int32_t initialLevel_; ///< The initial level of refinement for the mesh.
      int32_t maxLevel_;     ///< The maximum level of refinement for the mesh.
      int32_t nElements_;    ///< The number of elements in the mesh.
  };

} // namespace NeoFOAM

