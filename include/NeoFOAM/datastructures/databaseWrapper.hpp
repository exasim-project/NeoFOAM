// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#include "NeoN/NeoN.hpp"

#include "regIOobject.H"

namespace NeoFOAM
{

/**
 * @class DatabaseWrapper
 * @brief Registers a @c NeoN::Database in the @c Foam::Time objectRegistry.
 *
 * Allows NeoFOAM functionObjects (which only have @c const Foam::Time&) to
 * access the NeoN field database via the standard OpenFOAM lookup:
 * @code
 *   time_.lookupObject<DatabaseWrapper>(DatabaseWrapper::registryName).db()
 * @endcode
 *
 * @par Lifetime
 * The wrapper holds a non-owning pointer to its associated @c NeoN::Database,
 * which must outlive the wrapper.  In practice the wrapper is stored as a
 * @c std::unique_ptr inside @c RunTime so that its destructor (which calls
 * @c regIOobject::checkOut) fires before @c Foam::Time is destroyed.
 */
class DatabaseWrapper : public Foam::regIOobject
{
    NeoN::Database* db_;

public:

    /// Key under which the wrapper is registered in Foam::Time.
    static const Foam::word registryName;

    /**
     * @brief Construct and register in @p obr.
     *
     * The @c regIOobject base class calls @c checkIn() during construction,
     * which inserts this object into @p obr under @c registryName.
     *
     * @param obr  The objectRegistry to register in (typically Foam::Time).
     * @param db   The NeoN database to wrap (must outlive this object).
     */
    DatabaseWrapper(const Foam::objectRegistry& obr, NeoN::Database& db);

    NeoN::Database& db() { return *db_; }

    const NeoN::Database& db() const { return *db_; }

    bool writeData(Foam::Ostream&) const override { return true; }
};

} // namespace NeoFOAM
