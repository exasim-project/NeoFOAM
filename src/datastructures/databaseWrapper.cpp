// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#include "NeoFOAM/datastructures/databaseWrapper.hpp"

namespace NeoFOAM
{

const Foam::word DatabaseWrapper::registryName("NeoFOAMDatabase");

DatabaseWrapper::DatabaseWrapper(const Foam::objectRegistry& obr, NeoN::Database& db)
    : Foam::regIOobject(
          Foam::IOobject(
              registryName,
              obr.time().timeName(),
              obr,
              Foam::IOobject::NO_READ,
              Foam::IOobject::NO_WRITE
          )
      )
    , db_(&db)
{}

} // namespace NeoFOAM
