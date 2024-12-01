// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <vector>

#include "NeoFOAM/finiteVolume/cellCentred/fields/geometricField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoFOAM/core/database/database.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{

/**
 * @class VolumeField
 * @brief Represents a surface field in a finite volume method.
 *
 * The VolumeField class is a template class that represents a cell-centered field in a finite
 * volume method. It inherits from the GeometricFieldMixin class and provides methods for correcting
 * boundary conditions.
 *
 * @tparam ValueType The value type of the field.
 */
template<typename ValueType>
class VolumeField : public GeometricFieldMixin<ValueType>
{

public:

    /* @brief Constructor for a uninitialized VolumeField
     *
     * @param exec The executor
     * @param name The name of the field
     * @param mesh The underlying mesh
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(
              exec,
              name,
              mesh,
              DomainField<ValueType>(exec, mesh.nCells(), mesh.nBoundaryFaces(), mesh.nBoundaries())
          ),
          boundaryConditions_(boundaryConditions), db_(std::nullopt), key(""),
          fieldCollectionName("")
    {}

    /* @brief Constructor for a VolumeField with a given internal field
     *
     * @param exec The executor
     * @param mesh The underlying mesh
     * @param internalField the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     */
    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& internalField,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
    )
        : GeometricFieldMixin<ValueType>(
              exec,
              name,
              mesh,
              DomainField<ValueType>(exec, internalField, mesh.nBoundaryFaces(), mesh.nBoundaries())
          ),
          boundaryConditions_(boundaryConditions), db_(std::nullopt), key(""),
          fieldCollectionName("")
    {}

    /* @brief Constructor for a VolumeField with a given internal field and database
     *
     * @param exec The executor
     * @param name The name of the field
     * @param mesh The underlying mesh
     * @param internalField the underlying internal field
     * @param boundaryConditions a vector of boundary conditions
     * @param db The database
     * @param key The key of the field in the database
     * @param fieldCollectionName The name of the field collection in the database
     */
    VolumeField(
        const Executor& exec,
        std::string name,
        const UnstructuredMesh& mesh,
        const Field<ValueType>& internalField,
        const std::vector<VolumeBoundary<ValueType>>& boundaryConditions,
        Database& db,
        std::string key,
        std::string fieldCollectionName
    )
        : GeometricFieldMixin<ValueType>(
              exec,
              name,
              mesh,
              DomainField<ValueType>(exec, internalField, mesh.nBoundaryFaces(), mesh.nBoundaries())
          ),
          boundaryConditions_(boundaryConditions), db_(&db), key(key),
          fieldCollectionName(fieldCollectionName)
    {}

    VolumeField(const VolumeField& other)
        : GeometricFieldMixin<ValueType>(other), boundaryConditions_(other.boundaryConditions_),
          db_(other.db_), key(other.key), fieldCollectionName(other.fieldCollectionName)
    {}

    /**
     * @brief Corrects the boundary conditions of the surface field.
     *
     * This function applies the correctBoundaryConditions() method to each boundary condition in
     * the field.
     */
    void correctBoundaryConditions()
    {
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition.correctBoundaryCondition(this->field_);
        }
    }

    /**
     * @brief Returns true if the field has a database, false otherwise.
     *
     * @return true if the field has a database, false otherwise.
     */
    bool hasDatabase() const { return db_.has_value(); }

    /**
     * @brief Retrieves the database.
     *
     * @return Database& A reference to the database.
     */
    Database& db()
    {
        if (!db_.has_value())
        {
            throw std::runtime_error("Database not set");
        }
        return *db_.value();
    }

    /**
     * @brief Retrieves the database.
     *
     * @return const Database& A const reference to the database.
     */
    const Database& db() const
    {
        if (!db_.has_value())
        {
            throw std::runtime_error("Database not set");
        }
        return *db_.value();
    }

    /**
     * @brief Returns true if the field is registered in the database, false otherwise.
     *
     * @return true if the field is registered in the database, false otherwise.
     */
    bool registered() const { return key != "" && fieldCollectionName != "" && db_.has_value(); }

    std::vector<VolumeBoundary<ValueType>> boundaryConditions() const
    {
        return boundaryConditions_;
    }

    std::string key;                 // The key of the field in the database
    std::string fieldCollectionName; // The name of the field collection in the database

private:

    std::vector<VolumeBoundary<ValueType>> boundaryConditions_; // The vector of boundary conditions
    std::optional<Database*> db_; // The optional pointer to the database
};

} // namespace NeoFOAM
