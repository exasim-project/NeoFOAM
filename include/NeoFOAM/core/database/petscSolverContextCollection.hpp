// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <limits>
#include <string>
#include <functional>

#include "NeoFOAM/core/demangle.hpp"
#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/database/database.hpp"
#include "NeoFOAM/core/database/collection.hpp"
#include "NeoFOAM/core/database/document.hpp"
#include "NeoFOAM/linearAlgebra/petscSolverContext.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{
/**
 * @brief Validates a petscSolverContextDocument.
 *
 * This function validates a petscSolverContextDocument by checking if it contains the required
 * fields.
 *
 * @param doc The Document to validate.
 * @return true if the Document is valid, false otherwise.
 */
bool validatContextDoc(const Document& doc);

/**
 * @class petscSolverContextDocument
 * @brief A class representing a petscSolverContext document in a database.
 *
 * The petscSolverContextDocument class represents a field document in a database. It is a subclass
 * of the Document class and provides additional functionality for accessing field-specific data.
 */
class petscSolverContextDocument
{
public:

    /**
     * @brief Constructs a petscSolverContextDocument with the given field and metadata.
     *
     * @tparam FieldType The type of the field.
     * @param field The field to store in the document.
     * @param timeIndex The time index of the field.
     * @param iterationIndex The iteration index of the field.
     * @param subCycleIndex The sub-cycle index of the field.
     */
    template<class ContextType>
    petscSolverContextDocument(const ContextType& context, std::string eqnName)
        : doc_(Document({{"eqnName", eqnName}, {"context", context}}), validatContextDoc)
    {}

    /**
     * @brief Constructs a petscSolverContextDocument with the given Document.
     *
     * @param doc The Document to construct the petscSolverContextDocument from.
     */
    petscSolverContextDocument(const Document& doc);

    /**
     * @brief Retrieves the underlying Document.
     *
     * @return Document& A reference to the underlying Document.
     */
    Document& doc();

    /**
     * @brief Retrieves the underlying Document (const version).
     *
     * @return const Document& A const reference to the underlying Document.
     */
    const Document& doc() const;


    /**
     * @brief Retrieves the unique identifier of the field collection.
     *
     * @return A string representing the unique identifier.
     */
    std::string id() const;

    /**
     * @brief Retrieves the type name of the field.
     *
     * @return A string representing the type name.
     */
    static std::string typeName();

    /**
     * @brief Retrieves the field from the document.
     *
     * @tparam FieldType The type of the field.
     * @return A reference to the field.
     */
    template<class ContextType>
    ContextType& context()
    {
        return doc_.get<ContextType&>("context");
    }

    /**
     * @brief Retrieves the field from the document (const version).
     *
     * @tparam FieldType The type of the field.
     * @return A const reference to the field.
     */
    template<class ContextType>
    const ContextType& context() const
    {
        return doc_.get<const ContextType&>("context");
    }

    /**
     * @brief Retrieves the name of the equation for which the petsc solver is used.
     *
     * @return A string representing the name of the field.
     */
    std::string eqnName() const;

    /**
     * @brief Retrieves the time index of the field.
     *
     * @return An integer representing the time index.
     */
    std::string& eqnName();


private:

    Document doc_; /**< The underlying Document. */
};

/**
 * @class petscSolverContextCollection
 * @brief A class representing a collection of field documents in a database.
 *
 * The petscSolverContextCollection class represents a collection of field documents in a database
 * and provides additional functionality for accessing field-specific data.
 */
class petscSolverContextCollection : public CollectionMixin<petscSolverContextDocument>
{
public:

    /**
     * @brief Constructs a petscSolverContextCollection with the given database and name.
     *
     * @param db The database to create the collection in.
     * @param name The name of the collection.
     */
    petscSolverContextCollection(NeoFOAM::Database& db, std::string name);

    /**
     * @brief Checks if the collection contains a field with the given ID.
     *
     * @param id The ID of the field to check for.
     * @return true if the collection contains the field, false otherwise.
     */
    bool contains(const std::string& id) const;

    /**
     * @brief Inserts a field document into the collection.
     *
     * @param fd The field document to insert.
     * @return A string representing the unique identifier of the inserted field.
     */
    std::string insert(const petscSolverContextDocument& fd);

    /**
     * @brief Retrieves a field document by its ID.
     *
     * @param id The ID of the field document to retrieve.
     * @return petscSolverContextDocument& A reference to the field document.
     */
    petscSolverContextDocument& petscSolverContextDoc(const std::string& id);

    /**
     * @brief Retrieves a field document by its ID (const version).
     *
     * @param id The ID of the field document to retrieve.
     * @return const petscSolverContextDocument& A const reference to the field document.
     */
    const petscSolverContextDocument& petscSolverContextDoc(const std::string& id) const;

    /**
     * @brief Retrieves the instance of the petscSolverContextCollection with the given name.
     *
     * creates the petscSolverContextCollection if it does not exist.
     *
     * @param db The database to retrieve the petscSolverContextCollection from.
     * @param name The name of the petscSolverContextCollection.
     * @return petscSolverContextCollection& A reference to the petscSolverContextCollection.
     */
    static petscSolverContextCollection& instance(NeoFOAM::Database& db, std::string name);


    /**
     * @brief Retrieves the instance of the petscSolverContextCollection with the given name (const
     * version).
     *
     * creates the petscSolverContextCollection if it does not exist.
     *
     * @param db The database to retrieve the petscSolverContextCollection from.
     * @param name The name of the petscSolverContextCollection.
     * @return const petscSolverContextCollection& A const reference to the
     * petscSolverContextCollection.
     */
    static const petscSolverContextCollection&
    instance(const NeoFOAM::Database& db, std::string name);
};


} // namespace NeoFOAM
