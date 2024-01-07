// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <cstdint>
#include "NeoFOAM/blas/fields.hpp"
#include "treeAMRMeshModifier.hpp"
#include <memory>

namespace NeoFOAM
{
  /**
   * @class treeAMRMesh
   * @brief Represents a mesh for tree-based adaptive mesh refinement (AMR).
   *
   * The treeAMRMesh class provides functionality for refining, coarsening, balancing, and writing the mesh.
   * It also provides information about the number of elements in the mesh.
   */
  class treeAMRMesh
  {
  public:
    /**
     * @brief Constructor for treeAMRMesh.
     * @param initialLevel_ The initial level of refinement for the mesh.
     * @param maxLevel_ The maximum level of refinement for the mesh.
     */
    treeAMRMesh(std::unique_ptr<treeAMRMeshModifier> meshModifier)
        : meshModifier_(std::move(meshModifier))
        , initialLevel_(meshModifier_->initialLevel())
        , maxLevel_(meshModifier_->maxLevel())
        , nElements_(meshModifier_->nElements())
        , V_("V", nElements_)
        , Cx_("Cx", nElements_)
        , Cy_("Cy", nElements_)
        , Cz_("Cz", nElements_)
        , level_("level", nElements_)
    {
    }

    /**
     * @brief Destructor for treeAMRMesh.
     */
    ~treeAMRMesh() = default;

    /**
     * @brief Refines the mesh.
     * @return True if the mesh was successfully refined
     * , false otherwise.
     */
    bool refine()
    {
      return meshModifier_->refine();
    }

    /**
     * @brief Coarsens the mesh.
     * @return True if the mesh was successfully coarsened, false otherwise.
     */
    bool coarsen()
    {
      return meshModifier_->coarsen();
    }

    /**
     * @brief Balances the mesh.
     * @return True if the mesh was successfully balanced, false otherwise.
     */
    bool balance()
    {
      return meshModifier_->balance();
    }

    /**
     * @brief Writes the mesh to a file.
     */
    void write()
    {
      meshModifier_->write();
    }

    /**
     * @brief Gets the number of elements in the mesh.
     * @return The number of elements in the mesh.
     */
    int32_t nElements() const
    {
      return nElements_;
    }

    const treeAMRMeshModifier& meshModifier() const
    {
      return *meshModifier_;
    }
    

  private:
    std::unique_ptr<treeAMRMeshModifier> meshModifier_;
    int32_t initialLevel_; ///< The initial level of refinement for the mesh.
    int32_t maxLevel_;     ///< The maximum level of refinement for the mesh.
    int32_t nElements_;    ///< The number of elements in the mesh.

    scalarField V_;  // The volume of each cell in the mesh.
    scalarField Cx_; // The x-coordinate of the centroid of each cell in the mesh.
    scalarField Cy_; // The y-coordinate of the centroid of each cell in the mesh.
    scalarField Cz_; // The z-coordinate of the centroid of each cell in the mesh.

    scalarField level_; // The level of refinement of each cell in the mesh.
  };

} // namespace NeoFOAM
