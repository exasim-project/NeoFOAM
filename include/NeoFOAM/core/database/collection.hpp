// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#pragma once

#include <unordered_map>
#include <string>
#include <memory>

#include "NeoFOAM/core/database/document.hpp"

namespace NeoFOAM
{


// forward declaration
class Database;

/**
 * @class Collection
 * @brief A type-erased interface collection types.
 *
 * The Collection class provides a type-erased interface for different collection types.
 * It allows operations such as retrieving documents, finding documents based on predicates,
 * and accessing collection metadata.
 *
 * @tparam CollectionType The type of the underlying collection.
 */
class Collection
{
public:

    /**
     * @brief Constructs a Collection from a specific collection type.
     *
     * @tparam CollectionType The type of the collection to be wrapped.
     * @param collection The collection instance to be wrapped.
     */
    template<typename CollectionType>
    Collection(CollectionType collection);

    /**
     * @brief Copy constructor.
     *
     * @param other The Collection instance to copy from.
     */
    Collection(const Collection& other);

    /**
     * @brief Retrieves a document by its ID.
     *
     * @param id The ID of the document to retrieve.
     * @return Document& A reference to the document.
     */
    Document& doc(const std::string& id);

    /**
     * @brief Retrieves a document by its ID (const version).
     *
     * @param id The ID of the document to retrieve.
     * @return const Document& A const reference to the document.
     */
    const Document& doc(const std::string& id) const;

    /**
     * @brief Finds documents that match a given predicate.
     *
     * @param predicate A function that takes a Document and returns true if it matches the
     * criteria.
     * @return std::vector<std::string> A vector of document IDs that match the predicate.
     */
    std::vector<std::string> find(const std::function<bool(const Document&)>& predicate) const;

    /**
     * @brief Returns the number of documents in the collection.
     *
     * @return size_t The number of documents.
     */
    size_t size() const;

    /**
     * @brief Returns the type of the collection.
     *
     * @return std::string The type of the collection.
     */
    std::string type() const;

    /**
     * @brief Returns the name of the collection.
     *
     * @return std::string The name of the collection.
     */
    std::string name() const;

    /**
     * @brief Returns a reference to the database containing the collection.
     *
     * @return Database& A reference to the database.
     */
    Database& db();

    /**
     * @brief Returns a const reference to the database containing the collection.
     *
     * @return const Database& A const reference to the database.
     */
    const Database& db() const;

    /**
     * @brief Casts the collection to a specific collection type.
     *
     * @tparam CollectionType The type to cast to.
     * @return CollectionType& A reference to the casted collection.
     * @throws std::bad_cast if the cast fails.
     */
    template<typename CollectionType>
    CollectionType& as();

    /**
     * @brief Casts the collection to a specific collection type (const version).
     *
     * @tparam CollectionType The type to cast to.
     * @return const CollectionType& A const reference to the casted collection.
     * @throws std::bad_cast if the cast fails.
     */
    template<typename CollectionType>
    const CollectionType& as() const;

private:

    /**
     * @brief The base class for the type-erased collection.
     *
     * This class defines the common interface that all collection types must implement.
     */
    struct Concept;

    /**
     * @brief The derived class template for the type-erased collection.
     *
     * This class template implements the Concept interface for a specific collection type.
     *
     * @tparam CollectionType The type of the collection.
     */
    template<typename CollectionType>
    struct Model;

    /**
     * @brief A unique pointer to the type-erased collection implementation.
     */
    std::unique_ptr<Concept> impl_;
};

/**
 * @brief A mixin class for collection of documents in a database to simplify the implementation of
 * common operations.
 *
 * @tparam DocumentType The type of documents stored in the collection.
 */
template<typename DocumentType>
class CollectionMixin
{

public:

    /**
     * @brief Constructs a CollectionMixin with the given database and collection name.
     *
     * @param db The database reference.
     * @param name The name of the collection.
     */
    CollectionMixin(NeoFOAM::Database& db, std::string name);

    /**
     * @brief Retrieves a document by its ID.
     *
     * @param id The ID of the document.
     * @return Document& A reference to the document.
     */
    Document& doc(const std::string& id);

    /**
     * @brief Retrieves a document by its ID (const version).
     *
     * @param id The ID of the document.
     * @return const Document& A const reference to the document.
     */
    const Document& doc(const std::string& id) const;

    /**
     * @brief Finds documents that match a given predicate.
     *
     * @param predicate A function that takes a const reference to a Document and returns a bool.
     * @return std::vector<std::string> A vector of document IDs that match the predicate.
     */
    std::vector<std::string> find(const std::function<bool(const Document&)>& predicate) const;

    /**
     * @brief Gets the number of documents in the collection.
     *
     * @return std::size_t The number of documents.
     */
    std::size_t size() const;

    /**
     * @brief Gets a const reference to the database.
     *
     * @return const NeoFOAM::Database& A const reference to the database.
     */
    const NeoFOAM::Database& db() const;

    /**
     * @brief Gets a reference to the database.
     *
     * @return NeoFOAM::Database& A reference to the database.
     */
    NeoFOAM::Database& db();

    /**
     * @brief Gets the name of the collection.
     *
     * @return const std::string& A const reference to the collection name.
     */
    const std::string& name() const;

    /**
     * @brief Gets the type name of the documents in the collection.
     *
     * @return std::string The type name of the documents.
     */
    std::string type() const;

    /**
     * @brief Gets the sorted keys of the documents in the collection.
     *
     * @return std::vector<std::string> A vector of sorted document keys.
     */
    std::vector<std::string> sortedKeys() const;

protected:

    std::unordered_map<std::string, DocumentType> docs_; ///< The map of document IDs to documents.
    std::string name_;                                   ///< The name of the collection.
    NeoFOAM::Database& db_;                              ///< The reference to the database.
};

} // namespace NeoFOAM
