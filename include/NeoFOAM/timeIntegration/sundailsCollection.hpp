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

namespace NeoFOAM::timeIntegration
{

/**
 * TODO
 */
bool validateFieldDoc(const Document& doc);

/**
 * TODO
 */
class SundailsDocument
{
public:

    /**
     * TODO
     */
    template<class FieldType>
    SundailsDocument()
        : doc_(
            Document(
                {{
                     "solutionNVector",
                 },
                 {
                     "initialNVector",
                 },
                 {
                     "GlobalContext",
                 },
                 {
                     "RKSolver",
                 },
                 {
                     "Expression",
                 }}
            ),
            validateFieldDoc
        )
    {}

    NeoFOAM::sundials::SKVector<ValueType>
        solution_; /**< Solution vector, contains the sundails N_Vector. */
    NeoFOAM::sundials::SKVector<ValueType>
        initialConditions_; /**< Initial conditions vector, contains the sundails N_Vector. */
    std::shared_ptr<SUNContext> context_ {
        nullptr, sundials::SUN_CONTEXT_DELETER
    }; /**< The SUNContext for the solve. */
    std::unique_ptr<char, decltype(sundials::SUN_ARK_DELETER)> ODEMemory_ {
        nullptr, sundials::SUN_ARK_DELETER
    }; /**< The 'memory' sundails for the RK solver. (note void* is not stl compliant). */
    std::unique_ptr<NeoFOAM::dsl::Expression> pdeExpr_ {nullptr
    }; /**< Pointer to the pde system we are integrating in time. */


    /**
     * TODO
     */
    SundailsDocument(const Document& doc);

    /**
     * TODO
     */
    Document& doc();

    /**
     * TODO
     */
    const Document& doc() const;


    /**
     * TODO
     */
    std::string id() const;

    /**
     * TODO
     */
    static std::string typeName();

private:

    Document doc_; /**< The underlying Document. */
};

/**
 * TODO
 */
using CreateFunction = std::function<SundailsDocument(NeoFOAM::Database& db)>;

/**
 * TODO
 */
class SundailsCollection : public CollectionMixin<SundailsDocument>
{
public:

    /**
     * TODO
     */
    SundailsCollection(NeoFOAM::Database& db, std::string name);

    /**
     * TODO
     */
    bool contains(const std::string& id) const;

    /**
     * TODO
     */
    std::string insert(const SundailsCollection& fd);

    /**
     * TODO
     */
    SundailsCollection& fieldDoc(const std::string& id);

    /**
     * TODO
     */
    const SundailsCollection& fieldDoc(const std::string& id) const;

    /**
     * TODO
     */
    static SundailsCollection& instance(NeoFOAM::Database& db, std::string name);


    /**
     * TODO
     */
    static const SundailsCollection& instance(const NeoFOAM::Database& db, std::string name);

    /**
     * TODO
     */
    template<class FieldType>
    static SundailsCollection& instance(FieldType& field)
    {
        validateRegistration(
            field, "attempting to retrieve FieldCollection from unregistered field"
        );
        return instance(field.db(), field.fieldCollectionName);
    }

    /**
     * TODO
     */
    template<class FieldType>
    static const SundailsCollection& instance(const FieldType& field)
    {
        validateRegistration(
            field, "attempting to retrieve FieldCollection from unregistered field"
        );
        const Database& db = field.db();
        const Collection& collection = db.at(field.fieldCollectionName);
        return collection.as<FieldCollection>();
        // return instance(field.db(), field.fieldCollectionName);
    }

    /**
     * TODO
     */
    template<class FieldType>
    FieldType& registerField(CreateFunction createFunc)
    {
        FieldDocument doc = createFunc(db());
        if (!validateFieldDoc(doc.doc()))
        {
            throw std::runtime_error("Document is not valid");
        }

        std::string key = insert(doc);
        FieldDocument& fd = fieldDoc(key);
        FieldType& field = fd.field<FieldType>();
        field.key = key;
        field.fieldCollectionName = name();
        return field;
    }
};


/**
 * TODO
 */
template<typename FieldType>
class CreateFromExistingField
{
public:

    std::string name;
    const FieldType& field;
    std::int64_t timeIndex = std::numeric_limits<std::int64_t>::max();
    std::int64_t iterationIndex = std::numeric_limits<std::int64_t>::max();
    std::int64_t subCycleIndex = std::numeric_limits<std::int64_t>::max();

    FieldDocument operator()(Database& db)
    {
        FieldType vf(
            field.exec(),
            name,
            field.mesh(),
            field.internalField(),
            field.boundaryConditions(),
            db,
            "",
            ""
        );

        if (field.registered())
        {
            const FieldCollection& fieldCollection = FieldCollection::instance(field);
            const FieldDocument& fieldDoc = fieldCollection.fieldDoc(field.key);
            if (timeIndex == std::numeric_limits<std::int64_t>::max())
            {
                timeIndex = fieldDoc.timeIndex();
            }
            if (iterationIndex == std::numeric_limits<std::int64_t>::max())
            {
                iterationIndex = fieldDoc.iterationIndex();
            }
            if (subCycleIndex == std::numeric_limits<std::int64_t>::max())
            {
                subCycleIndex = fieldDoc.subCycleIndex();
            }
        }
        return NeoFOAM::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            validateFieldDoc
        );
    }
};


} // namespace NeoFOAM
