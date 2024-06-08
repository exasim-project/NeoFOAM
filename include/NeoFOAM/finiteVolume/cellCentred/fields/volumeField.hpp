// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once
#include <vector>

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/fields/domainField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


/**
 * @class VolumeField
 * @brief Represents a cell-centered finite volume field at cell centers.
 *
 * The VolumeField class is used to store field information at the cell center.
 * It provides methods to correct boundary conditions, access the internal field, boundary field,
 * and executor, and retrieve information about the mesh and boundary conditions.
 *
 * @tparam ValueType The type of the field values.
 */
template<typename ValueType>
class VolumeField
{
public:

    /**
     * @brief Constructor for VolumeField.
     *
     * @param exec The executor for parallel execution.
     * @param mesh The unstructured mesh.
     * @param boundaryConditions The boundary conditions for the field.
     */
    VolumeField(const Executor& exec)
        : exec_(exec), mesh_(nullptr), field_(exec), boundaryConditions_()
    {}

    /**
     * @brief Constructor for VolumeField.
     *
     * @param exec The executor for parallel execution.
     * @param mesh The unstructured mesh.
     * @param boundaryConditions The boundary conditions for the field.
     */
    VolumeField(
        const Executor& exec,
        std::shared_ptr<const UnstructuredMesh> mesh,
        std::vector<std::unique_ptr<BoundaryField<ValueType>>>&& boundaryConditions
    )
        : exec_(exec), mesh_(mesh),
          field_(exec, mesh->nCells(), mesh->nBoundaryFaces(), mesh->nBoundaries()),
          boundaryConditions_(std::move(boundaryConditions))
    {}

    /**
     * @brief Corrects the boundary conditions of the field.
     *
     * This method applies the correctBoundaryConditions() method to each boundary condition
     * in the field.
     */
    void correctBoundaryConditions()
    {
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition->correctBoundaryConditions(
                field_.boundaryField(), field_.internalField()
            );
        }
    }

    /**
     * @brief Returns a const reference to the internal field.
     *
     * @return A const reference to the internal field.
     */
    const Field<ValueType>& internalField() const { return field_.internalField(); }

    /**
     * @brief Returns a reference to the internal field.
     *
     * @return A reference to the internal field.
     */
    Field<ValueType>& internalField() { return field_.internalField(); }

    /**
     * @brief Returns a const reference to the boundary field.
     *
     * @return A const reference to the boundary field.
     */
    const BoundaryFields<ValueType>& boundaryField() const { return field_.boundaryField(); }

    /**
     * @brief Returns a reference to the boundary field.
     *
     * @return A reference to the boundary field.
     */
    BoundaryFields<ValueType>& boundaryField() { return field_.boundaryField(); }

    /**
     * @brief Returns a const reference to the executor.
     *
     * @return A const reference to the executor.
     */
    const Executor& exec() const { return exec_; }

    /**
     * @brief Returns a const reference to the unstructured mesh object.
     *
     * @return The const reference to the unstructured mesh object.
     */
    std::shared_ptr<const UnstructuredMesh> mesh() const { return mesh_; }

private:

    Executor exec_;                                /**< The executor for parallel execution. */
    std::shared_ptr<const UnstructuredMesh> mesh_; /**< The unstructured mesh. */
    DomainField<ValueType> field_;                 /**< The domain field. */
    std::vector<std::unique_ptr<BoundaryField<ValueType>>>
        boundaryConditions_; /**< The boundary conditions for the field. */
};

} // namespace NeoFOAM
